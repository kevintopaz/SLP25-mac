"""Clean PyTorch port of the reference Video VAE (decoder + encoder).

This port is purpose-built for CoreML conversion: every op is a standard
torch.nn module, no streaming state, no MemoryState, no chunked GroupNorm.
Math is identical to the reference VAE for a single fixed-size input.

Architecture (matches src/models/video_vae_v3/modules/video_vae.py):
  - Causal 3D conv: replicate first frame ``kT//2 * 2`` times before conv3d
  - GroupNorm: 32 groups, eps=1e-6, applied PER FRAME via reshape (B,C,T,H,W)
    -> (B*T,C,H,W) -> nn.GroupNorm -> back. Same as causal_norm_wrapper.
  - SiLU after every GN (except attention), ReLU/SDPA in MidBlock
  - ResnetBlock3D: GN -> SiLU -> Conv3d -> GN -> SiLU -> Conv3d -> +shortcut
  - SelfAttention (mid only): GN -> q/k/v projections -> SDPA -> proj -> +residual
  - Upsample3d: 1x1x1 conv (channel expand) -> pixel-shuffle -> [drop frame 1
    if temporal_up] -> causal 3x3x3 conv
  - Downsample3d: zero-pad bottom/right -> strided causal 3x3x3 conv
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# GroupNorm eps must match reference VAE (1e-6, NOT default 1e-5).
GN_EPS = 1e-6


# ── Causal Convs ─────────────────────────────────────────────────────────────

class CausalConv3d(nn.Module):
    """Causal 3D conv: replicate first frame ``temporal_padding * 2`` times
    on the left, then plain conv3d with no temporal padding."""

    def __init__(self, in_c: int, out_c: int,
                 kT: int, kH: int, kW: int,
                 pad_h: int, pad_w: int):
        super().__init__()
        self.kT = kT
        self.temporal_padding = (kT // 2) if kT > 1 else 0
        self.conv = nn.Conv3d(in_c, out_c, (kT, kH, kW),
                              padding=(0, pad_h, pad_w))

    def _causal_pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_padding == 0:
            return x
        pad_n = self.temporal_padding * 2  # for kT=3, pad 2 on left -> output T frames
        first = x[:, :, :1].expand(-1, -1, pad_n, -1, -1)
        return torch.cat([first, x], dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self._causal_pad(x))


class StridedCausalConv3d(nn.Module):
    """CausalConv3d with stride for downsampling."""

    def __init__(self, in_c: int, out_c: int,
                 kT: int, kH: int, kW: int,
                 stride_t: int, stride_h: int, stride_w: int,
                 pad_h: int, pad_w: int):
        super().__init__()
        self.kT = kT
        self.temporal_padding = (kT // 2) if kT > 1 else 0
        self.conv = nn.Conv3d(in_c, out_c, (kT, kH, kW),
                              stride=(stride_t, stride_h, stride_w),
                              padding=(0, pad_h, pad_w))

    def _causal_pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_padding == 0:
            return x
        pad_n = self.temporal_padding * 2
        first = x[:, :, :1].expand(-1, -1, pad_n, -1, -1)
        return torch.cat([first, x], dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self._causal_pad(x))


# ── Per-frame GroupNorm wrapper ──────────────────────────────────────────────

def _per_frame_gn(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    """Apply 2D GroupNorm to a 5D tensor by collapsing T into batch.
    Matches causal_norm_wrapper: (B,C,T,H,W) -> (B*T,C,H,W) -> GN -> (B,C,T,H,W)."""
    B, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    x = norm(x)
    return x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()


# ── Modules ──────────────────────────────────────────────────────────────────

class ResnetBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int, groups: int = 32, kT: int = 3):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_c, eps=GN_EPS)
        self.conv1 = CausalConv3d(in_c, out_c, kT, 3, 3, 1, 1)
        self.norm2 = nn.GroupNorm(groups, out_c, eps=GN_EPS)
        self.conv2 = CausalConv3d(out_c, out_c, kT, 3, 3, 1, 1)
        self.has_shortcut = in_c != out_c
        if self.has_shortcut:
            self.conv_shortcut = CausalConv3d(in_c, out_c, 1, 1, 1, 0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(_per_frame_gn(self.norm1, x))
        h = self.conv1(h)
        h = F.silu(_per_frame_gn(self.norm2, h))
        h = self.conv2(h)
        if self.has_shortcut:
            x = self.conv_shortcut(x)
        return x + h


class SelfAttention(nn.Module):
    """Single-head self-attention (matches reference attn_video_vae.py)."""

    def __init__(self, dim: int, groups: int = 32):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim, eps=GN_EPS)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BT, C, H, W) — applied per-frame (BT in batch).
        residual = x
        x = self.group_norm(x)
        BT, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (BT, H*W, C)
        q = self.to_q(x).unsqueeze(1)      # (BT, 1, HW, C)
        k = self.to_k(x).unsqueeze(1)
        v = self.to_v(x).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.squeeze(1)               # (BT, HW, C)
        out = self.to_out[0](out)
        out = out.transpose(1, 2).reshape(BT, C, H, W)
        return out + residual


class MidBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 32, kT: int = 3):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(ch, ch, groups, kT),
            ResnetBlock3D(ch, ch, groups, kT),
        ])
        self.attentions = nn.ModuleList([SelfAttention(ch, groups)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        B, C, T, H, W = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        flat = self.attentions[0](flat)
        x = flat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        x = self.resnets[1](x)
        return x


class Upsample3d(nn.Module):
    def __init__(self, ch: int, out_ch: int, temporal_up: bool, kT: int = 3):
        super().__init__()
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2
        self.temporal_up = temporal_up
        self.ratio = self.spatial_ratio ** 2 * self.temporal_ratio
        self.upscale_conv = nn.Conv3d(ch, ch * self.ratio, kernel_size=1)
        self.conv = CausalConv3d(ch, out_ch, kT, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        sr = self.spatial_ratio
        tr = self.temporal_ratio
        # 1x1x1 channel expand
        up = self.upscale_conv(x)  # (B, C*ratio, T, H, W)
        # Reshape for pixel-shuffle. Matches the reference rearrange:
        #   "b (x y z c) f h w -> b c (f z) (h x) (w y)"
        # i.e. C*ratio is structured as (sr_x, sr_y, tr_z, c).
        up = up.reshape(B, sr, sr, tr, C, T, H, W)
        # Permute to (B, C, T, tr_z, H, sr_x, W, sr_y)
        up = up.permute(0, 4, 5, 3, 6, 1, 7, 2).contiguous()
        up = up.reshape(B, C, T * tr, H * sr, W * sr)
        if self.temporal_up:
            # remove_head: drop the second frame after upsampling
            up = torch.cat([up[:, :, :1], up[:, :, 2:]], dim=2)
        return self.conv(up)


class Downsample3d(nn.Module):
    def __init__(self, ch: int, out_ch: int, temporal_down: bool):
        super().__init__()
        self.temporal_down = temporal_down
        stride_t = 2 if temporal_down else 1
        kT = 3 if temporal_down else 1
        self.conv = StridedCausalConv3d(ch, out_ch, kT, 3, 3,
                                         stride_t=stride_t,
                                         stride_h=2, stride_w=2,
                                         pad_h=0, pad_w=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad bottom/right by 1 (matches MLX path)
        x = F.pad(x, (0, 1, 0, 1), value=0.0)
        return self.conv(x)


class UpDecoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, num_layers: int,
                 add_upsample: bool, temporal_up: bool,
                 groups: int = 32, kT: int = 3):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            rc = in_c if i == 0 else out_c
            resnets.append(ResnetBlock3D(rc, out_c, groups, kT))
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList(
            [Upsample3d(out_c, out_c, temporal_up, kT)] if add_upsample else []
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.resnets:
            x = r(x)
        for u in self.upsamplers:
            x = u(x)
        return x


class DownEncoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, num_layers: int,
                 add_downsample: bool, temporal_down: bool,
                 groups: int = 32, kT: int = 3):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            rc = in_c if i == 0 else out_c
            resnets.append(ResnetBlock3D(rc, out_c, groups, kT))
        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList(
            [Downsample3d(out_c, out_c, temporal_down)] if add_downsample else []
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.resnets:
            x = r(x)
        for d in self.downsamplers:
            x = d(x)
        return x


# ── Encoder / Decoder ────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, in_ch: int = 16, out_ch: int = 3,
                 block_channels=(128, 256, 512, 512),
                 layers_per_block: int = 2, groups: int = 32,
                 temporal_up_num: int = 2, kT: int = 3):
        super().__init__()
        rev = list(reversed(block_channels))
        self.conv_in = CausalConv3d(in_ch, rev[0], kT, 3, 3, 1, 1)
        self.mid_block = MidBlock(rev[0], groups, kT)
        up_blocks = []
        n = len(block_channels)
        out_c = rev[0]
        for i in range(n):
            prev_c = out_c
            out_c = rev[i]
            up_blocks.append(UpDecoderBlock(
                prev_c, out_c,
                num_layers=layers_per_block + 1,
                add_upsample=(i < n - 1),
                temporal_up=(i < temporal_up_num),
                groups=groups, kT=kT,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv_norm_out = nn.GroupNorm(groups, block_channels[0], eps=GN_EPS)
        self.conv_out = CausalConv3d(block_channels[0], out_ch, kT, 3, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        x = self.mid_block(x)
        for blk in self.up_blocks:
            x = blk(x)
        x = F.silu(_per_frame_gn(self.conv_norm_out, x))
        x = self.conv_out(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 32,
                 block_channels=(128, 256, 512, 512),
                 layers_per_block: int = 2, groups: int = 32,
                 temporal_down_num: int = 2, kT: int = 3):
        super().__init__()
        self.conv_in = CausalConv3d(in_ch, block_channels[0], kT, 3, 3, 1, 1)
        down_blocks = []
        n = len(block_channels)
        in_c = block_channels[0]
        for i in range(n):
            out_c = block_channels[i]
            is_final = (i == n - 1)
            is_temporal_down = (i >= n - temporal_down_num - 1)
            down_blocks.append(DownEncoderBlock(
                in_c, out_c,
                num_layers=layers_per_block,
                add_downsample=not is_final,
                temporal_down=is_temporal_down,
                groups=groups, kT=kT,
            ))
            in_c = out_c
        self.down_blocks = nn.ModuleList(down_blocks)
        self.mid_block = MidBlock(block_channels[-1], groups, kT)
        self.conv_norm_out = nn.GroupNorm(groups, block_channels[-1], eps=GN_EPS)
        self.conv_out = CausalConv3d(block_channels[-1], out_ch, kT, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.mid_block(x)
        x = F.silu(_per_frame_gn(self.conv_norm_out, x))
        x = self.conv_out(x)
        return x


# ── Weight Loading ───────────────────────────────────────────────────────────

def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _remap_keys_for_torch(weights: dict) -> dict:
    """Map safetensors keys to our PyTorch module structure.

    Reference safetensors uses bare conv keys (e.g. ``conv1.weight``) while we
    wrap them in ``CausalConv3d.conv``. So ``conv1.weight`` becomes
    ``conv1.conv.weight``.

    The CausalConv3d wrappers in our port are:
      - conv_in, conv_out (Encoder/Decoder top-level)
      - conv1, conv2, conv_shortcut (ResnetBlock3D)
      - upsamplers.X.conv (Upsample3d's trailing 3x3x3 causal conv)
      - downsamplers.X.conv (Downsample3d's strided causal conv)

    Bare nn.Conv3d (no remap):
      - upsamplers.X.upscale_conv (1x1x1 channel expand)
    """
    import re
    # Patterns to remap: trailing component that should be wrapped in '.conv'.
    # We use precise patterns to avoid hitting upscale_conv.{weight,bias}.
    causal_conv_names = (
        "conv1", "conv2", "conv_in", "conv_out", "conv_shortcut",
    )
    # For upsamplers/downsamplers: only remap when the trailing name is exactly
    # 'conv' (NOT upscale_conv).
    samp_re = re.compile(r"(\.(?:up|down)samplers\.\d+)\.conv(\.(?:weight|bias))$")

    out = {}
    for k, v in weights.items():
        new_k = k
        # Pattern 1: explicit causal-conv-only names. Match either at top
        # level (e.g. "conv_in.weight") or as a sub-path (".conv1.weight").
        matched = False
        for name in causal_conv_names:
            for suffix in (".weight", ".bias"):
                # Top-level (whole-key match)
                whole = f"{name}{suffix}"
                if new_k == whole:
                    new_k = f"{name}.conv{suffix}"
                    matched = True
                    break
                # Sub-path match
                old = f".{name}{suffix}"
                if new_k.endswith(old):
                    new_k = new_k[:-len(old)] + f".{name}.conv{suffix}"
                    matched = True
                    break
            if matched:
                break
        # Pattern 2: upsamplers.X.conv / downsamplers.X.conv
        m = samp_re.search(new_k)
        if m:
            new_k = samp_re.sub(r"\1.conv.conv\2", new_k)
        out[new_k] = v
    return out


def load_decoder_weights(decoder: Decoder, vae_path: str,
                         dtype: torch.dtype = torch.float16) -> None:
    from safetensors import safe_open
    raw = {}
    with safe_open(vae_path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("decoder."):
                raw[k[len("decoder."):]] = f.get_tensor(k).to(dtype)
    remapped = _remap_keys_for_torch(raw)
    missing, unexpected = decoder.load_state_dict(remapped, strict=False)
    if missing:
        print(f"[load_decoder] missing keys: {len(missing)}; first few: {missing[:5]}")
    if unexpected:
        print(f"[load_decoder] unexpected keys: {len(unexpected)}; first few: {unexpected[:5]}")


def load_encoder_weights(encoder: Encoder, vae_path: str,
                         dtype: torch.dtype = torch.float16) -> None:
    from safetensors import safe_open
    raw = {}
    with safe_open(vae_path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("encoder."):
                raw[k[len("encoder."):]] = f.get_tensor(k).to(dtype)
    remapped = _remap_keys_for_torch(raw)
    missing, unexpected = encoder.load_state_dict(remapped, strict=False)
    if missing:
        print(f"[load_encoder] missing keys: {len(missing)}; first few: {missing[:5]}")
    if unexpected:
        print(f"[load_encoder] unexpected keys: {len(unexpected)}; first few: {unexpected[:5]}")
