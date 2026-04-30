"""
PyTorch-MPS Video VAE — Encoder + Decoder with conv2d decomposition.

Designed to outperform torch.nn.Conv3d on MPS, which is notoriously slow for
video-VAE workloads. Three optimisations driven by microbenchmarks on M3 Pro:

  * Canonical layout = (T, C, H, W). T is folded into batch, so 2D ops
    (GroupNorm, attention, projection) run as plain NCHW conv2d/linear with
    no permute/contiguous copies. Only when doing a kT>1 temporal conv do we
    explicitly materialise temporal taps along the batch axis.

  * **Per-tap conv2d** (NOT stacked). Microbench at 480x480x33, in_c=out_c=128
    showed:
        F.conv3d (kT=3):                4941 ms
        F.conv2d stacked (kT*in_c):     4207 ms
        F.conv2d 3 per-tap conv2ds:      992 ms   <- 4-5x faster
        F.conv2d single (no temporal):   281 ms
    PyTorch MPS conv2d is well-tuned for plain in_c but suffers when the
    "in" dim is artificially inflated by kT.

  * Causal temporal padding via first-frame replication, no padded tensor
    materialised — taps slice the input directly via index arithmetic.

Provides:
  * TorchVideoVAE: high-level wrapper with tiled_encode / tiled_decode that
    accept numpy arrays in (T, H, W, C) format (same I/O as MLXVideoVAE).
"""

import math
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

VAE_SCALING_FACTOR = 0.9152


# ── Helpers ──────────────────────────────────────────────────────────────────

def _causal_pad_temporal(x_4d, pad_n):
    """Replicate the first frame `pad_n` times on the left."""
    if pad_n <= 0:
        return x_4d
    pad_left = x_4d[:1].expand(pad_n, -1, -1, -1)
    return torch.cat([pad_left, x_4d], dim=0)


# ── Modules ──────────────────────────────────────────────────────────────────

class CausalConv(nn.Module):
    """InflatedCausalConv3d as kT separate F.conv2d calls.

    Weight stored as (out_c, in_c, kT, kH, kW) — same as the source safetensors.
    Each tap k uses weight[:, :, k, :, :] as a plain (out_c, in_c, kH, kW)
    conv2d weight. Outputs are summed before bias.
    """

    def __init__(self, out_c, in_c, kT, kH, kW, pad_h, pad_w):
        super().__init__()
        self.kT = kT
        self.in_c = in_c
        self.out_c = out_c
        self.temporal_padding = (kT // 2) if kT > 1 else 0
        self._pad = (pad_h, pad_w)
        self.weight = nn.Parameter(torch.zeros(out_c, in_c, kT, kH, kW))
        self.bias = nn.Parameter(torch.zeros(out_c))

    def forward(self, x):
        # x: (T, C, H, W)
        if self.kT <= 1:
            w2d = self.weight[:, :, 0, :, :]
            return F.conv2d(x, w2d, bias=self.bias, padding=self._pad)
        T = x.shape[0]
        pad_n = self.temporal_padding * 2
        # ONE cat for the whole conv layer: replicate first frame `pad_n` times
        # on the left, then taps are cheap views (slices) of x_pad.
        x_pad = _causal_pad_temporal(x, pad_n)
        acc = None
        for k in range(self.kT):
            tap = x_pad[k:k + T]
            wk = self.weight[:, :, k, :, :]
            o = F.conv2d(tap, wk, padding=self._pad)
            acc = o if acc is None else acc + o
        return acc + self.bias.view(1, -1, 1, 1)


class StridedCausalConv(nn.Module):
    def __init__(self, out_c, in_c, kT, kH, kW, stride_t, stride_h, stride_w, pad_h, pad_w):
        super().__init__()
        self.kT = kT
        self.in_c = in_c
        self.out_c = out_c
        self.stride_t = stride_t
        self.stride_hw = (stride_h, stride_w)
        self.temporal_padding = (kT // 2) if kT > 1 else 0
        self._pad = (pad_h, pad_w)
        self.weight = nn.Parameter(torch.zeros(out_c, in_c, kT, kH, kW))
        self.bias = nn.Parameter(torch.zeros(out_c))

    def forward(self, x):
        if self.kT <= 1:
            w2d = self.weight[:, :, 0, :, :]
            return F.conv2d(x, w2d, bias=self.bias,
                            stride=self.stride_hw, padding=self._pad)
        T = x.shape[0]
        pad_n = self.temporal_padding * 2
        T_padded = T + pad_n
        T_out = (T_padded - self.kT) // self.stride_t + 1
        x_pad = _causal_pad_temporal(x, pad_n)
        acc = None
        for k in range(self.kT):
            # Strided start indices: tap k contributes to output frame t at
            # input position k + t*stride_t. We stride the whole tap range.
            tap = x_pad[k:k + T_out * self.stride_t:self.stride_t]
            wk = self.weight[:, :, k, :, :]
            o = F.conv2d(tap, wk, stride=self.stride_hw, padding=self._pad)
            acc = o if acc is None else acc + o
        return acc + self.bias.view(1, -1, 1, 1)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, groups=32, kT=3):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_c, affine=True)
        self.conv1 = CausalConv(out_c, in_c, kT, 3, 3, 1, 1)
        self.norm2 = nn.GroupNorm(groups, out_c, affine=True)
        self.conv2 = CausalConv(out_c, out_c, kT, 3, 3, 1, 1)
        self.has_shortcut = in_c != out_c
        if self.has_shortcut:
            self.conv_shortcut = CausalConv(out_c, in_c, 1, 1, 1, 0, 0)

    def forward(self, x):
        # x: (T, C, H, W)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        if self.has_shortcut:
            x = self.conv_shortcut(x)
        return x + h


class SelfAttention(nn.Module):
    def __init__(self, dim, groups=32):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim, affine=True)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])
        self.scale = dim ** -0.5

    def forward(self, x):
        # x: (T, C, H, W) — treat each frame as an independent attention map
        residual = x
        x = self.group_norm(x)
        T, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(T, H * W, C)
        q = self.to_q(x).view(T, 1, H * W, C)
        k = self.to_k(x).view(T, 1, H * W, C)
        v = self.to_v(x).view(T, 1, H * W, C)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.view(T, H * W, C)
        out = self.to_out[0](out)
        out = out.view(T, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out + residual


class MidBlock(nn.Module):
    def __init__(self, ch, groups=32, kT=3):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock(ch, ch, groups, kT),
            ResnetBlock(ch, ch, groups, kT),
        ])
        self.attentions = nn.ModuleList([SelfAttention(ch, groups)])

    def forward(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


# ── Decoder ──────────────────────────────────────────────────────────────────

class Upsample3d(nn.Module):
    def __init__(self, ch, out_ch, temporal_up, kT=3):
        super().__init__()
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2
        self.temporal_up = temporal_up
        ratio = self.spatial_ratio ** 2 * self.temporal_ratio
        self.upscale_conv = CausalConv(ch * ratio, ch, 1, 1, 1, 0, 0)
        self.conv = CausalConv(out_ch, ch, kT, 3, 3, 1, 1)

    def forward(self, x):
        # x: (T, C, H, W)
        T, C, H, W = x.shape
        sr = self.spatial_ratio
        tr = self.temporal_ratio
        up = self.upscale_conv(x)  # (T, sr*sr*tr*Cout, H, W) — Cout==C
        Cout = C
        # Out channel layout (matching MLX/original): (sr, sr, tr, Cout)
        up = up.view(T, sr, sr, tr, Cout, H, W)
        # Want: (T, tr, Cout, H, sr, W, sr) -> then reshape to (T*tr, Cout, H*sr, W*sr)
        up = up.permute(0, 3, 4, 5, 1, 6, 2).contiguous()
        up = up.view(T * tr, Cout, H * sr, W * sr)
        if self.temporal_up:
            # Drop the second time-step from the very first frame block to keep
            # the (T-1)*tr+1 causal layout that the original VAE uses.
            up = torch.cat([up[:1], up[2:]], dim=0)
        return self.conv(up)


class UpDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, num_layers, add_upsample, temporal_up,
                 groups=32, kT=3):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            rc = in_c if i == 0 else out_c
            resnets.append(ResnetBlock(rc, out_c, groups, kT))
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList(
            [Upsample3d(out_c, out_c, temporal_up, kT)] if add_upsample else []
        )

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        for u in self.upsamplers:
            x = u(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch=16, out_ch=3, block_channels=(128, 256, 512, 512),
                 layers_per_block=2, groups=32, temporal_up_num=2, kT=3):
        super().__init__()
        rev = list(reversed(block_channels))
        self.conv_in = CausalConv(rev[0], in_ch, kT, 3, 3, 1, 1)
        self.mid_block = MidBlock(rev[0], groups, kT)
        up_blocks = []
        n = len(block_channels)
        out_c = rev[0]
        for i in range(n):
            prev_c = out_c
            out_c = rev[i]
            up_blocks.append(
                UpDecoderBlock(prev_c, out_c, num_layers=layers_per_block + 1,
                               add_upsample=(i < n - 1), temporal_up=(i < temporal_up_num),
                               groups=groups, kT=kT))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv_norm_out = nn.GroupNorm(groups, block_channels[0], affine=True)
        self.conv_out = CausalConv(out_ch, block_channels[0], kT, 3, 3, 1, 1)

    def forward(self, z):
        x = self.conv_in(z)
        x = self.mid_block(x)
        for blk in self.up_blocks:
            x = blk(x)
        x = self.conv_norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


# ── Encoder ──────────────────────────────────────────────────────────────────

class Downsample3d(nn.Module):
    def __init__(self, ch, out_ch, temporal_down):
        super().__init__()
        self.temporal_down = temporal_down
        stride_t = 2 if temporal_down else 1
        kT = 3 if temporal_down else 1
        self.conv = StridedCausalConv(out_ch, ch, kT, 3, 3,
                                      stride_t=stride_t, stride_h=2, stride_w=2,
                                      pad_h=0, pad_w=0)

    def forward(self, x):
        # 4D NCHW pad: (left, right, top, bottom) on the last two dims.
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class DownEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, num_layers, add_downsample, temporal_down,
                 groups=32, kT=3):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            rc = in_c if i == 0 else out_c
            resnets.append(ResnetBlock(rc, out_c, groups, kT))
        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList(
            [Downsample3d(out_c, out_c, temporal_down)] if add_downsample else []
        )

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        for d in self.downsamplers:
            x = d(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=32, block_channels=(128, 256, 512, 512),
                 layers_per_block=2, groups=32, temporal_down_num=2, kT=3):
        super().__init__()
        self.conv_in = CausalConv(block_channels[0], in_ch, kT, 3, 3, 1, 1)
        down_blocks = []
        n = len(block_channels)
        in_c = block_channels[0]
        for i in range(n):
            out_c = block_channels[i]
            is_final = (i == n - 1)
            is_temporal_down = (i >= n - temporal_down_num - 1)
            down_blocks.append(
                DownEncoderBlock(in_c, out_c, num_layers=layers_per_block,
                                 add_downsample=not is_final, temporal_down=is_temporal_down,
                                 groups=groups, kT=kT))
            in_c = out_c
        self.down_blocks = nn.ModuleList(down_blocks)
        self.mid_block = MidBlock(block_channels[-1], groups, kT)
        self.conv_norm_out = nn.GroupNorm(groups, block_channels[-1], affine=True)
        self.conv_out = CausalConv(out_ch, block_channels[-1], kT, 3, 3, 1, 1)

    def forward(self, x, profile=False):
        if profile:
            import time as _t
            torch.mps.synchronize()
            t0 = _t.perf_counter()
            x = self.conv_in(x)
            torch.mps.synchronize()
            logger.info(f"    conv_in: {(_t.perf_counter()-t0)*1000:.0f}ms x.shape={tuple(x.shape)}")
            for i, blk in enumerate(self.down_blocks):
                t0 = _t.perf_counter()
                x = blk(x)
                torch.mps.synchronize()
                logger.info(f"    down_blocks[{i}]: {(_t.perf_counter()-t0)*1000:.0f}ms x.shape={tuple(x.shape)}")
            t0 = _t.perf_counter()
            x = self.mid_block(x)
            torch.mps.synchronize()
            logger.info(f"    mid_block: {(_t.perf_counter()-t0)*1000:.0f}ms")
            t0 = _t.perf_counter()
            x = self.conv_norm_out(x)
            x = F.silu(x)
            x = self.conv_out(x)
            torch.mps.synchronize()
            logger.info(f"    norm/silu/conv_out: {(_t.perf_counter()-t0)*1000:.0f}ms")
            return x
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


# ── Weight Loading ───────────────────────────────────────────────────────────

def _load_weights(model, vae_path, prefix, dtype=torch.float16, device="mps"):
    """Load safetensors into the model.

    Our CausalConv keeps the (out_c, in_c, kT, kH, kW) layout verbatim, so we
    can drop in 5D tensors as-is from the source safetensors.
    """
    from safetensors import safe_open
    sd = model.state_dict()
    loaded = 0
    skipped = 0
    with safe_open(vae_path, framework="pt") as f:
        for k in f.keys():
            if not k.startswith(prefix):
                continue
            short = k[len(prefix):]
            if short not in sd:
                logger.warning(f"  skipping unknown key: {short}")
                skipped += 1
                continue
            tensor = f.get_tensor(k)
            if tensor.shape != sd[short].shape:
                logger.warning(
                    f"  shape mismatch on {short}: got {tuple(tensor.shape)}, "
                    f"want {tuple(sd[short].shape)}"
                )
                skipped += 1
                continue
            sd[short] = tensor.to(dtype=dtype)
            loaded += 1

    model.load_state_dict(sd, strict=False)
    model = model.to(device=device, dtype=dtype)
    logger.info(f"  loaded {loaded} weights ({skipped} skipped) into {prefix}*")
    return model


# ── Tiled Encode / Decode ────────────────────────────────────────────────────

def _make_blend_weight(tile_h, tile_w, overlap, n_h, n_w, ti, tj):
    bw = np.ones((tile_h, tile_w), dtype=np.float32)
    if ti > 0:
        bw[:overlap, :] *= np.linspace(0, 1, overlap)[:, None]
    if ti < n_h - 1:
        bw[-overlap:, :] *= np.linspace(1, 0, overlap)[:, None]
    if tj > 0:
        bw[:, :overlap] *= np.linspace(0, 1, overlap)[None, :]
    if tj < n_w - 1:
        bw[:, -overlap:] *= np.linspace(1, 0, overlap)[None, :]
    return bw


# ── High-Level Wrapper ───────────────────────────────────────────────────────

class TorchVideoVAE:
    """PyTorch-MPS Video VAE with tiled encode/decode.

    Internal layout: (T, C, H, W). T plays the role of batch for 2D ops.
    """

    def __init__(self, vae_path, dtype=torch.float16, device="mps"):
        self.dtype = dtype
        self.device = torch.device(device)
        self.vae_path = vae_path

        logger.info("Loading PyTorch Video VAE encoder...")
        self.encoder = Encoder(in_ch=3, out_ch=32, block_channels=(128, 256, 512, 512),
                               layers_per_block=2, groups=32, temporal_down_num=2, kT=3)
        _load_weights(self.encoder, vae_path, "encoder.", dtype, device)
        self.encoder.eval()

        logger.info("Loading PyTorch Video VAE decoder...")
        self.decoder = Decoder(in_ch=16, out_ch=3, block_channels=(128, 256, 512, 512),
                               layers_per_block=2, groups=32, temporal_up_num=2, kT=3)
        _load_weights(self.decoder, vae_path, "decoder.", dtype, device)
        self.decoder.eval()
        logger.info(f"PyTorch Video VAE ready ({dtype})")

    @torch.no_grad()
    def tiled_encode(self, frames_f16_np, tile_size=480, tile_overlap=48):
        """Encode video frames to latent using spatial tiling.

        Args:
            frames_f16_np: numpy array [T, H, W, 3] in [-1, 1] float16
            tile_size: spatial tile size in pixels
            tile_overlap: overlap between tiles in pixels

        Returns:
            latent: numpy array [1, T_lat, H_lat, W_lat, 16] float32 (mean of Gaussian)
        """
        T, H, W, _ = frames_f16_np.shape
        lat_tile = tile_size // 8
        lat_ov = tile_overlap // 8
        lat_H, lat_W = H // 8, W // 8
        T_lat = 1 if T == 1 else (T - 1) // 4 + 1

        n_h = max(1, int(math.ceil((H - tile_overlap) / (tile_size - tile_overlap))))
        n_w = max(1, int(math.ceil((W - tile_overlap) / (tile_size - tile_overlap))))

        # Accumulators on GPU in fp32 for precision
        latent = torch.zeros(
            (T_lat, 32, lat_H, lat_W), dtype=torch.float32, device=self.device
        )
        lat_wt = torch.zeros_like(latent)

        t0 = time.perf_counter()
        tc = 0
        for ti in range(n_h):
            for tj in range(n_w):
                hs = min(ti * (tile_size - tile_overlap), H - tile_size)
                ws = min(tj * (tile_size - tile_overlap), W - tile_size)
                tile = frames_f16_np[:, hs:hs + tile_size, ws:ws + tile_size, :]
                # numpy (T,H,W,C) -> torch (T, C, H, W)
                tile_t = torch.from_numpy(tile).to(
                    device=self.device, dtype=self.dtype
                ).permute(0, 3, 1, 2).contiguous()
                # Profile only the first tile to keep logs short
                profile = (tc == 0)
                enc_out = self.encoder(tile_t, profile=profile).float()  # (T_lat, 32, lat_tile, lat_tile)

                lh, lw = hs // 8, ws // 8
                bw = _make_blend_weight(lat_tile, lat_tile, lat_ov, n_h, n_w, ti, tj)
                bw_t = torch.from_numpy(bw).to(self.device).view(1, 1, lat_tile, lat_tile)

                latent[:, :, lh:lh + lat_tile, lw:lw + lat_tile] += enc_out * bw_t
                lat_wt[:, :, lh:lh + lat_tile, lw:lw + lat_tile] += bw_t

                del tile_t, enc_out, bw_t
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                tc += 1
                logger.info(f"  Enc tile {tc}/{n_h*n_w}: {time.perf_counter()-t0:.1f}s")

        latent = latent / lat_wt.clamp_min(1e-8)
        # (T_lat, 32, H, W) -> (1, T_lat, H, W, 16) (mean only, drop logvar)
        mean = latent[:, :16].permute(0, 2, 3, 1).unsqueeze(0).contiguous().cpu().numpy()
        elapsed = time.perf_counter() - t0
        logger.info(f"Encoded in {elapsed:.1f}s -> latent {mean.shape}")
        return mean

    @torch.no_grad()
    def tiled_decode(self, latent_f32_np, tile_size=480, tile_overlap=48, num_frames=None):
        """Decode latent (numpy [1, T_lat, H_lat, W_lat, 16] float32) to frames."""
        _, T_lat, lat_H, lat_W, _ = latent_f32_np.shape
        lat_tile = tile_size // 8
        lat_ov = tile_overlap // 8
        pixel_H = lat_H * 8
        pixel_W = lat_W * 8
        T_out = 1 if T_lat == 1 else (T_lat - 1) * 4 + 1

        n_h = max(1, int(math.ceil((lat_H - lat_ov) / (lat_tile - lat_ov))))
        n_w = max(1, int(math.ceil((lat_W - lat_ov) / (lat_tile - lat_ov))))

        output = torch.zeros(
            (T_out, 3, pixel_H, pixel_W), dtype=torch.float32, device=self.device
        )
        out_wt = torch.zeros_like(output)

        # numpy (1, T, H, W, 16) -> torch (T, 16, H, W)
        latent_full = torch.from_numpy(latent_f32_np[0]).to(
            device=self.device, dtype=self.dtype
        ).permute(0, 3, 1, 2).contiguous()

        t0 = time.perf_counter()
        tc = 0
        pix_tile = tile_size
        pix_ov = tile_overlap
        for ti in range(n_h):
            for tj in range(n_w):
                lh = min(ti * (lat_tile - lat_ov), lat_H - lat_tile)
                lw = min(tj * (lat_tile - lat_ov), lat_W - lat_tile)
                tile_z = latent_full[:, :, lh:lh + lat_tile, lw:lw + lat_tile].contiguous()
                tile_out = self.decoder(tile_z).float()  # (T_out, 3, pix_tile, pix_tile)

                ph, pw = lh * 8, lw * 8
                bw = _make_blend_weight(pix_tile, pix_tile, pix_ov, n_h, n_w, ti, tj)
                bw_t = torch.from_numpy(bw).to(self.device).view(1, 1, pix_tile, pix_tile)

                output[:, :, ph:ph + pix_tile, pw:pw + pix_tile] += tile_out * bw_t
                out_wt[:, :, ph:ph + pix_tile, pw:pw + pix_tile] += bw_t
                del tile_z, tile_out, bw_t
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                tc += 1
                logger.info(f"  Dec tile {tc}/{n_h*n_w}: {time.perf_counter()-t0:.1f}s")

        output = output / out_wt.clamp_min(1e-8)
        # (T, 3, H, W) -> [T, H, W, 3]
        frames = output.permute(0, 2, 3, 1).cpu().numpy()
        if num_frames is not None and frames.shape[0] > num_frames:
            frames = frames[:num_frames]
        elapsed = time.perf_counter() - t0
        logger.info(f"Decoded in {elapsed:.1f}s -> {frames.shape}")
        return frames
