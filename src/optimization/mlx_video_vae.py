"""
MLX Video VAE — Encoder + Decoder with conv2d decomposition and tiled encode/decode.

Decomposes InflatedCausalConv3d (kernel 3x3x3) into a temporal sliding-window
+ single batched mx.conv2d call per layer, leveraging MLX's lazy graph fusion
on Apple Silicon. This is the bloom_precision-style fast implementation: NO
per-layer mx.eval calls, NO temporal/channel chunking — let MLX fuse adjacent
ops (norm + silu + conv) into single Metal kernels.

Provides:
  * MLXVideoVAE: high-level wrapper that accepts/returns numpy/PyTorch tensors
  * tiled_encode / tiled_decode with spatial overlap blending
"""

import math
import os
import time
import logging

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

logger = logging.getLogger(__name__)

VAE_SCALING_FACTOR = 0.9152

# ── Compile flag ─────────────────────────────────────────────────────────────
# Set MLX_VAE_USE_COMPILE=0 to disable mx.compile for debugging.
_USE_COMPILE = os.environ.get("MLX_VAE_USE_COMPILE", "1") == "1"

# Adaptive batched-vs-per-frame threshold for CausalConv in low_mem mode.
# When the kT-tap stacked intermediate is below this size, the (much faster)
# batched path is used; otherwise per-frame.
#
# IMPORTANT: This is OFF by default (threshold=0 means "always per-frame")
# because, while mathematically identical, batched and per-frame paths use
# different Metal GEMM reduction orders for different batch sizes, which
# produces fp16 ULP-level differences (~0.039 max abs vs full per-frame
# accumulated across layers). Set MLX_VAE_BATCHED_MAX_GB to enable, e.g.:
#   1.5  -> batched at stages 2/3 (T=17, T=9), per-frame at stages 0/1
#          (one-tile bench: ~6.7% faster, +0.01 GB peak)
#   8.0  -> batched everywhere (fastest, ~30 GB peak — needs lots of VRAM)
_BATCHED_MAX_BYTES = int(float(os.environ.get("MLX_VAE_BATCHED_MAX_GB", "0")) * 1024**3)

# Like TRT path empty_cache between tiles: drop MLX buffer pool between spatial
# tiles (slower). Set MLX_VAE_CLEAR_CACHE_PER_TILE=1 when chasing peak footprint.
_CLEAR_CACHE_PER_TILE = os.environ.get("MLX_VAE_CLEAR_CACHE_PER_TILE", "0") == "1"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _gn5d(norm_layer, x):
    """Apply a GroupNorm that expects 4-D (BT, H, W, C) to a 5-D tensor."""
    B, T, H, W, C = x.shape
    return norm_layer(x.reshape(B * T, H, W, C)).reshape(B, T, H, W, C)


# ── Compiled fused kernels ───────────────────────────────────────────────────
# These are bit-equivalent (modulo MLX implementation changes) to the eager
# versions but let MLX fuse trailing element-wise ops into single Metal kernels
# and avoid Python-side dispatch overhead on hot loops.

# GroupNorm eps used by the reference VAE. The PyTorch reference (see
# src/models/video_vae_v3/modules/video_vae.py and attn_video_vae.py)
# constructs every nn.GroupNorm with eps=1e-6. MLX's default is 1e-5;
# using the wrong eps shifts activations and accumulates over 30+ layers,
# so we set 1e-6 explicitly here AND in every nn.GroupNorm constructor.
_GN_EPS = 1e-6


def _silu_gn_pt32_eager(x, gamma, beta):
    """Pytorch-compatible GroupNorm(num_groups=32, eps=1e-6) followed by silu.

    Replicates ``nn.GroupNorm._pytorch_compatible_group_norm`` line-for-line
    (same ``mx.fast.layer_norm`` call) and then applies silu. Math is
    identical to ``nn.silu(nn.GroupNorm(32, C, eps=1e-6, pytorch_compatible=True)(x))``.

    Reduction axes: per (BT, G) over (H, W, C/G). Per-frame; temporal axis
    is NOT included in the reduction (matches the reference VAE which
    reshapes (B,C,T,H,W) -> (B*T,C,H,W) before applying nn.GroupNorm).

    Shapes: x (BT, H, W, C), gamma (C,), beta (C,) -> (BT, H, W, C).
    """
    BT, H, W, C = x.shape
    G = 32
    cg = C // G
    x = x.reshape(BT, H * W, G, cg)
    x = x.transpose(0, 2, 1, 3).reshape(BT, G, H * W * cg)
    x = mx.fast.layer_norm(x, weight=None, bias=None, eps=_GN_EPS)
    x = x.reshape(BT, G, H * W, cg)
    x = x.transpose(0, 2, 1, 3).reshape(BT, H, W, C)
    x = gamma * x + beta
    return x * mx.sigmoid(x)


_silu_gn_pt32 = mx.compile(_silu_gn_pt32_eager) if _USE_COMPILE else _silu_gn_pt32_eager


def _silu_gn5d(norm, x_5d):
    """Fused silu(GroupNorm(x)) for a 5-D tensor; assumes pytorch_compatible
    GroupNorm with num_groups=32, eps=1e-6 (the only config we use)."""
    B, T, H, W, C = x_5d.shape
    out = _silu_gn_pt32(x_5d.reshape(B * T, H, W, C), norm.weight, norm.bias)
    return out.reshape(B, T, H, W, C)


# Cache of compiled (conv2d + bias) wrappers keyed by (padding, stride).
# Each key has its compiled trace; subsequent calls with same input shape
# reuse the cached Metal kernel plan.
_compiled_conv2d_cache: dict = {}


def _get_compiled_conv2d_step(pad_hw, stride_hw=(1, 1)):
    """Return a compiled function ``f(x, weight, bias) -> conv2d(x,w,pad,stride)+b``.

    Falls back to None when compile is disabled.
    """
    if not _USE_COMPILE:
        return None
    key = (pad_hw, stride_hw)
    if key not in _compiled_conv2d_cache:
        if stride_hw == (1, 1):
            def _step(x, weight, bias):
                return mx.conv2d(x, weight, padding=pad_hw) + bias
        else:
            def _step(x, weight, bias):
                return mx.conv2d(x, weight, stride=stride_hw,
                                 padding=pad_hw) + bias
        _compiled_conv2d_cache[key] = mx.compile(_step)
    return _compiled_conv2d_cache[key]


# ── Modules ──────────────────────────────────────────────────────────────────

class CausalConv(nn.Module):
    """InflatedCausalConv3d → temporal sliding-window + mx.conv2d.

    Weight shape: (out_c, kH, kW, kT * in_c)  [MLX OHWI].
    For kT == 1 the temporal stacking is skipped.

    Two execution modes:
      * `_batched` (default): all T output frames in one mx.conv2d call.
      * `_per_frame` (low_mem=True): one output frame at a time + mx.eval per frame.
        Same arithmetic, ~5x lower peak activation but slightly slower due to
        per-frame sync. This was the bloom_precision-style 156 s/1080p config.
    """

    def __init__(self, out_c, in_c, kT, kH, kW, pad_h, pad_w):
        super().__init__()
        self.kT = kT
        self.in_c = in_c
        self.temporal_padding = (kT // 2) if kT > 1 else 0
        self._pad = (pad_h, pad_w)
        in_c_eff = kT * in_c if kT > 1 else in_c
        self.weight = mx.zeros((out_c, kH, kW, in_c_eff))
        self.bias = mx.zeros((out_c,))
        # Compiled (conv2d + bias) step. Same Metal output as eager; saves
        # one kernel launch per call by fusing the bias add.
        self._compiled_step = _get_compiled_conv2d_step(self._pad)

    def _do_conv(self, flat):
        if self._compiled_step is not None:
            return self._compiled_step(flat, self.weight, self.bias)
        return mx.conv2d(flat, self.weight, padding=self._pad) + self.bias

    def _batched(self, x):
        B, T, H, W, C = x.shape
        if self.kT <= 1:
            flat = x.reshape(B * T, H, W, C)
            out = self._do_conv(flat)
            _, Ho, Wo, Co = out.shape
            return out.reshape(B, T, Ho, Wo, Co)
        pad_n = self.temporal_padding * 2
        first = mx.broadcast_to(x[:, :1], [B, pad_n, H, W, C])
        xp = mx.concatenate([first, x], axis=1)
        T_out = xp.shape[1] - self.kT + 1
        windows = [xp[:, i : i + T_out] for i in range(self.kT)]
        stacked = mx.concatenate(windows, axis=-1)
        flat = stacked.reshape(B * T_out, H, W, self.kT * C)
        out = self._do_conv(flat)
        _, Ho, Wo, Co = out.shape
        return out.reshape(B, T_out, Ho, Wo, Co)

    def _per_frame(self, x):
        # Avoids the (T+pad_n)-frame pre-padded buffer (xp) by clipping the
        # temporal source index per-tap. Bit-equivalent: tap k at output frame
        # t reads x[:, max(0, t+k-pad_n)], which equals the corresponding entry
        # in the padded xp where the first frame is replicated pad_n times.
        B, T, H, W, C = x.shape
        if self.kT <= 1:
            return self._batched(x)
        pad_n = self.temporal_padding * 2
        # Preallocate [B, T, Ho, Wo, Co] and fill per frame — avoids mx.stack
        # retaining T separate arrays plus a contiguous concat (TRT-style bound).
        o0 = self._do_conv(
            mx.concatenate(
                [x[:, max(0, 0 + k - pad_n)] for k in range(self.kT)],
                axis=-1,
            )
        )
        mx.eval(o0)
        _, Ho, Wo, Co = o0.shape
        out = mx.zeros((B, T, Ho, Wo, Co), dtype=o0.dtype)
        out[:, 0] = o0
        for t in range(1, T):
            stacked = mx.concatenate(
                [x[:, max(0, t + k - pad_n)] for k in range(self.kT)],
                axis=-1,
            )
            o = self._do_conv(stacked)
            mx.eval(o)
            out[:, t] = o
        mx.eval(out)
        return out

    def __call__(self, x, low_mem=False):
        # Adaptive: even in low_mem mode, fall back to the (much faster)
        # batched path at stages where the kT-tap stacked intermediate is
        # small enough not to grow encoder peak memory. Encoder peak is set
        # by stage 0 (~10 GB); a ~1 GB transient at later stages is safe.
        if not low_mem or self.kT <= 1:
            return self._batched(x)
        B, T, H, W, C = x.shape
        # bytes for the kT-tap stacked tensor (fp16)
        stacked_bytes = B * T * H * W * self.kT * C * 2
        if stacked_bytes <= _BATCHED_MAX_BYTES:
            return self._batched(x)
        return self._per_frame(x)


class StridedCausalConv(nn.Module):
    """CausalConv3d with stride, decomposed into mx.conv2d."""

    def __init__(self, out_c, in_c, kT, kH, kW, stride_t, stride_h, stride_w,
                 pad_h, pad_w):
        super().__init__()
        self.kT = kT
        self.in_c = in_c
        self.stride_t = stride_t
        self.stride_hw = (stride_h, stride_w)
        self.temporal_padding = (kT // 2) if kT > 1 else 0
        self._pad = (pad_h, pad_w)
        in_c_eff = kT * in_c if kT > 1 else in_c
        self.weight = mx.zeros((out_c, kH, kW, in_c_eff))
        self.bias = mx.zeros((out_c,))
        self._compiled_step = _get_compiled_conv2d_step(self._pad, self.stride_hw)

    def _do_conv(self, flat):
        if self._compiled_step is not None:
            return self._compiled_step(flat, self.weight, self.bias)
        return (mx.conv2d(flat, self.weight, stride=self.stride_hw,
                          padding=self._pad) + self.bias)

    def _batched(self, x):
        B, T, H, W, C = x.shape
        if self.kT <= 1:
            flat = x.reshape(B * T, H, W, C)
            out = self._do_conv(flat)
            _, Ho, Wo, Co = out.shape
            return out.reshape(B, T, Ho, Wo, Co)
        pad_n = self.temporal_padding * 2
        first = mx.broadcast_to(x[:, :1], [B, pad_n, H, W, C])
        xp = mx.concatenate([first, x], axis=1)
        T_padded = xp.shape[1]
        T_out = (T_padded - self.kT) // self.stride_t + 1
        slices = []
        for t in range(T_out):
            t_start = t * self.stride_t
            windows = [xp[:, t_start + k] for k in range(self.kT)]
            slices.append(mx.concatenate(windows, axis=-1))
        batched = mx.stack(slices, axis=1)
        flat = batched.reshape(B * T_out, H, W, self.kT * C)
        out = self._do_conv(flat)
        _, Ho, Wo, Co = out.shape
        return out.reshape(B, T_out, Ho, Wo, Co)

    def _per_frame(self, x):
        # Skip the xp pre-padding buffer; clip source index per-tap.
        # Bit-equivalent to padding x with pad_n replicas of x[:, :1] on left.
        B, T, H, W, C = x.shape
        if self.kT <= 1:
            return self._batched(x)
        pad_n = self.temporal_padding * 2
        T_padded = T + pad_n
        T_out = (T_padded - self.kT) // self.stride_t + 1
        t_start0 = 0
        o0 = self._do_conv(
            mx.concatenate(
                [x[:, max(0, t_start0 + k - pad_n)] for k in range(self.kT)],
                axis=-1,
            )
        )
        mx.eval(o0)
        _, Ho, Wo, Co = o0.shape
        out = mx.zeros((B, T_out, Ho, Wo, Co), dtype=o0.dtype)
        out[:, 0] = o0
        for t in range(1, T_out):
            t_start = t * self.stride_t
            stacked = mx.concatenate(
                [x[:, max(0, t_start + k - pad_n)] for k in range(self.kT)],
                axis=-1,
            )
            o = self._do_conv(stacked)
            mx.eval(o)
            out[:, t] = o
        mx.eval(out)
        return out

    def __call__(self, x, low_mem=False):
        # Same adaptive heuristic as CausalConv.
        if not low_mem or self.kT <= 1:
            return self._batched(x)
        B, T, H, W, C = x.shape
        stacked_bytes = B * T * H * W * self.kT * C * 2
        if stacked_bytes <= _BATCHED_MAX_BYTES:
            return self._batched(x)
        return self._per_frame(x)


class ResnetBlock(nn.Module):
    """ResnetBlock3D: norm → silu → conv → norm → silu → conv (+shortcut)."""

    def __init__(self, in_c, out_c, groups=32, kT=3):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_c, eps=_GN_EPS, pytorch_compatible=True)
        self.conv1 = CausalConv(out_c, in_c, kT, 3, 3, 1, 1)
        self.norm2 = nn.GroupNorm(groups, out_c, eps=_GN_EPS, pytorch_compatible=True)
        self.conv2 = CausalConv(out_c, out_c, kT, 3, 3, 1, 1)
        self.has_shortcut = in_c != out_c
        if self.has_shortcut:
            self.conv_shortcut = CausalConv(out_c, in_c, 1, 1, 1, 0, 0)

    def __call__(self, x, low_mem=False):
        h = _silu_gn5d(self.norm1, x)
        if low_mem: mx.eval(h)
        h = self.conv1(h, low_mem=low_mem)
        h = _silu_gn5d(self.norm2, h)
        if low_mem: mx.eval(h)
        h = self.conv2(h, low_mem=low_mem)
        if self.has_shortcut:
            x = self.conv_shortcut(x, low_mem=low_mem)
        out = x + h
        if low_mem:
            mx.eval(out)
        return out


class SelfAttention(nn.Module):
    """Single-head self-attention via Metal SDPA (no explicit attn matrix)."""

    def __init__(self, dim, groups=32):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim, eps=_GN_EPS, pytorch_compatible=True)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = [nn.Linear(dim, dim)]
        self.scale = dim ** -0.5

    def __call__(self, x):
        residual = x
        x = self.group_norm(x)
        BT, H, W, C = x.shape
        x = x.reshape(BT, H * W, C)
        q = self.to_q(x).reshape(BT, 1, H * W, C)
        k = self.to_k(x).reshape(BT, 1, H * W, C)
        v = self.to_v(x).reshape(BT, 1, H * W, C)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.reshape(BT, H * W, C)
        out = self.to_out[0](out)
        return out.reshape(BT, H, W, C) + residual


class MidBlock(nn.Module):
    """UNetMidBlock3D: resnet → attention → resnet."""

    def __init__(self, ch, groups=32, kT=3):
        super().__init__()
        self.resnets = [
            ResnetBlock(ch, ch, groups, kT),
            ResnetBlock(ch, ch, groups, kT),
        ]
        self.attentions = [SelfAttention(ch, groups)]

    def __call__(self, x, low_mem=False):
        B, T, H, W, C = x.shape
        x = self.resnets[0](x, low_mem=low_mem)
        if low_mem: mx.eval(x)
        flat = x.reshape(B * T, H, W, C)
        flat = self.attentions[0](flat)
        x = flat.reshape(B, T, H, W, C)
        if low_mem: mx.eval(x)
        x = self.resnets[1](x, low_mem=low_mem)
        if low_mem: mx.eval(x)
        return x


# ── Decoder ──────────────────────────────────────────────────────────────────

class Upsample3d(nn.Module):
    """Upsample3D: MAGViT pixel-shuffle, optional temporal up, causal conv.

    The upscale_conv is a 1x1x1 conv that EXPANDS channels by ratio = sr*sr*tr
    (4 or 8). The reshape+transpose+reshape that follows is pure metadata
    (lazy in MLX), so we avoid forcing an mx.eval between upscale_conv and
    the shuffle — that lets MLX fuse the 1x1 conv with the layout shuffle
    into a single Metal kernel (saves one full materialization pass through
    the channel-expanded intermediate).
    """

    def __init__(self, ch, out_ch, temporal_up, kT=3):
        super().__init__()
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2
        self.temporal_up = temporal_up
        ratio = self.spatial_ratio ** 2 * self.temporal_ratio
        self.upscale_conv = CausalConv(ch * ratio, ch, 1, 1, 1, 0, 0)
        self.conv = CausalConv(out_ch, ch, kT, 3, 3, 1, 1)

    def __call__(self, x, low_mem=False):
        B, T, H, W, C = x.shape
        sr = self.spatial_ratio
        tr = self.temporal_ratio
        up = self.upscale_conv(x, low_mem=low_mem)
        up = up.reshape(B, T, H, W, sr, sr, tr, C)
        up = up.transpose(0, 1, 6, 2, 4, 3, 5, 7)
        up = up.reshape(B, T * tr, H * sr, W * sr, C)
        if self.temporal_up:
            up = mx.concatenate([up[:, :1], up[:, 2:]], axis=1)
        return self.conv(up, low_mem=low_mem)


class UpDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, num_layers, add_upsample, temporal_up,
                 groups=32, kT=3):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            rc = in_c if i == 0 else out_c
            resnets.append(ResnetBlock(rc, out_c, groups, kT))
        self.resnets = resnets
        self.upsamplers = (
            [Upsample3d(out_c, out_c, temporal_up, kT)]
            if add_upsample else []
        )

    def __call__(self, x, low_mem=False):
        for r in self.resnets:
            x = r(x, low_mem=low_mem)
            if low_mem: mx.eval(x)
        for u in self.upsamplers:
            x = u(x, low_mem=low_mem)
            if low_mem: mx.eval(x)
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
                               add_upsample=(i < n - 1),
                               temporal_up=(i < temporal_up_num),
                               groups=groups, kT=kT))
        self.up_blocks = up_blocks
        self.conv_norm_out = nn.GroupNorm(groups, block_channels[0],
                                          eps=_GN_EPS, pytorch_compatible=True)
        self.conv_out = CausalConv(out_ch, block_channels[0], kT, 3, 3, 1, 1)

    def __call__(self, z, low_mem=False):
        x = self.conv_in(z, low_mem=low_mem)
        if low_mem: mx.eval(x)
        x = self.mid_block(x, low_mem=low_mem)
        for blk in self.up_blocks:
            x = blk(x, low_mem=low_mem)
        x = _silu_gn5d(self.conv_norm_out, x)
        if low_mem: mx.eval(x)
        x = self.conv_out(x, low_mem=low_mem)
        return x


# ── Encoder ──────────────────────────────────────────────────────────────────

class Downsample3d(nn.Module):
    def __init__(self, ch, out_ch, temporal_down):
        super().__init__()
        self.temporal_down = temporal_down
        stride_t = 2 if temporal_down else 1
        kT = 3 if temporal_down else 1
        self.conv = StridedCausalConv(out_ch, ch, kT, 3, 3,
                                      stride_t=stride_t,
                                      stride_h=2, stride_w=2,
                                      pad_h=0, pad_w=0)

    def __call__(self, x, low_mem=False):
        x = mx.pad(x, [(0, 0), (0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x, low_mem=low_mem)


class DownEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, num_layers, add_downsample, temporal_down,
                 groups=32, kT=3):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            rc = in_c if i == 0 else out_c
            resnets.append(ResnetBlock(rc, out_c, groups, kT))
        self.resnets = resnets
        self.downsamplers = (
            [Downsample3d(out_c, out_c, temporal_down)]
            if add_downsample else []
        )

    def __call__(self, x, low_mem=False):
        for r in self.resnets:
            x = r(x, low_mem=low_mem)
            if low_mem: mx.eval(x)
        for d in self.downsamplers:
            x = d(x, low_mem=low_mem)
            if low_mem: mx.eval(x)
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
                                 add_downsample=not is_final,
                                 temporal_down=is_temporal_down,
                                 groups=groups, kT=kT))
            in_c = out_c
        self.down_blocks = down_blocks
        self.mid_block = MidBlock(block_channels[-1], groups, kT)
        self.conv_norm_out = nn.GroupNorm(groups, block_channels[-1],
                                          eps=_GN_EPS, pytorch_compatible=True)
        self.conv_out = CausalConv(out_ch, block_channels[-1], kT, 3, 3, 1, 1)

    def __call__(self, x, low_mem=False):
        x = self.conv_in(x, low_mem=low_mem)
        if low_mem: mx.eval(x)
        for blk in self.down_blocks:
            x = blk(x, low_mem=low_mem)
        x = self.mid_block(x, low_mem=low_mem)
        x = _silu_gn5d(self.conv_norm_out, x)
        if low_mem: mx.eval(x)
        return self.conv_out(x, low_mem=low_mem)


# ── Weight Loading ───────────────────────────────────────────────────────────

def _conv3d_to_mlx(arr):
    """[O, I, kT, kH, kW] -> MLX conv2d weight [O, kH, kW, kT*I]."""
    out_c, in_c, kT, kH, kW = arr.shape
    if kT > 1:
        arr = arr.transpose(0, 2, 1, 3, 4).reshape(out_c, kT * in_c, kH, kW)
    else:
        arr = arr.reshape(out_c, in_c, kH, kW)
    return arr.transpose(0, 2, 3, 1)


def _load_weights(model, vae_path, prefix, dtype=mx.float16):
    from safetensors import safe_open
    raw = {}
    with safe_open(vae_path, framework="numpy") as f:
        for k in f.keys():
            if k.startswith(prefix):
                raw[k[len(prefix):]] = f.get_tensor(k)
    pairs = []
    for key, arr in raw.items():
        if arr.ndim == 5:
            arr = _conv3d_to_mlx(arr)
        pairs.append((key, mx.array(arr).astype(dtype)))
    model.load_weights(pairs)
    return model


# ── Tiled Encode / Decode ────────────────────────────────────────────────────

def _as_hw(v):
    """Normalize a tile-size / overlap argument to (H, W) tuple."""
    if isinstance(v, (tuple, list)):
        if len(v) != 2:
            raise ValueError(f"tile arg must be int or (H, W) tuple, got {v!r}")
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _make_blend_weight(tile_h, tile_w, overlap, n_h, n_w, ti, tj):
    """Bilinear cosine ramp at tile borders. ``overlap`` may be an int
    (square) or an (overlap_h, overlap_w) pair."""
    if isinstance(overlap, (tuple, list)):
        ov_h, ov_w = int(overlap[0]), int(overlap[1])
    else:
        ov_h = ov_w = int(overlap)
    bw = np.ones((tile_h, tile_w), dtype=np.float32)
    if ti > 0:
        bw[:ov_h, :] *= np.linspace(0, 1, ov_h)[:, None]
    if ti < n_h - 1:
        bw[-ov_h:, :] *= np.linspace(1, 0, ov_h)[:, None]
    if tj > 0:
        bw[:, :ov_w] *= np.linspace(0, 1, ov_w)[None, :]
    if tj < n_w - 1:
        bw[:, -ov_w:] *= np.linspace(1, 0, ov_w)[None, :]
    return bw


def _gpu_blend_paste(canvas, weight_canvas, tile, bw_5d, t_slice, h_slice, w_slice):
    """Add `tile * bw_5d` into `canvas[..., h_slice, w_slice, :]` and accumulate weight.
    All ops stay on GPU. Returns updated (canvas, weight_canvas)."""
    blended = (tile.astype(canvas.dtype) * bw_5d)
    canvas[:, t_slice, h_slice, w_slice, :] = canvas[:, t_slice, h_slice, w_slice, :] + blended
    weight_canvas[:, t_slice, h_slice, w_slice, :] = (
        weight_canvas[:, t_slice, h_slice, w_slice, :] + bw_5d
    )
    return canvas, weight_canvas


# ── High-Level Wrapper ───────────────────────────────────────────────────────

class MLXVideoVAE:
    """High-level wrapper: holds encoder/decoder, provides tiled encode/decode,
    and bridges PyTorch <-> MLX tensors."""

    def __init__(self, vae_path, cache_limit_gb=1.0, dtype=mx.float16, low_mem=True):
        # Defaults match the bloom_precision config that hit 156 s for 33 frames
        # @ 1080p (encode): low_mem=True (per-frame conv) + 1 GB MLX buffer cache.
        # Set cache_limit_gb=None to disable cache cap (faster but ~22 GB system RAM
        # for cached buffers); set low_mem=False to use the all-at-once batched
        # path (fastest but ~32 GB peak active memory).
        if cache_limit_gb is not None:
            mx.set_cache_limit(int(cache_limit_gb * 1024**3))
        self.low_mem = low_mem
        self.dtype = dtype
        self.vae_path = vae_path

        logger.info("Loading MLX Video VAE encoder...")
        self.encoder = Encoder(in_ch=3, out_ch=32, block_channels=(128, 256, 512, 512),
                               layers_per_block=2, groups=32, temporal_down_num=2, kT=3)
        _load_weights(self.encoder, vae_path, "encoder.", dtype)
        mx.eval(self.encoder.parameters())

        logger.info("Loading MLX Video VAE decoder...")
        self.decoder = Decoder(in_ch=16, out_ch=3, block_channels=(128, 256, 512, 512),
                               layers_per_block=2, groups=32, temporal_up_num=2, kT=3)
        _load_weights(self.decoder, vae_path, "decoder.", dtype)
        mx.eval(self.decoder.parameters())
        logger.info(f"MLX Video VAE ready ({dtype})")

    def tiled_encode(self, frames_f16_np, tile_size=480, tile_overlap=48):
        """Encode video frames to latent using spatial tiling.

        Args:
            frames_f16_np: numpy array [T, H, W, 3] in [-1, 1] float16
            tile_size: int (square) or (tile_H, tile_W) tuple in pixels
            tile_overlap: int (square) or (overlap_H, overlap_W) tuple in pixels

        Returns:
            latent: numpy array [1, T_lat, H_lat, W_lat, 16] float32 (mean only)
        """
        tile_H, tile_W = _as_hw(tile_size)
        ov_H, ov_W = _as_hw(tile_overlap)
        T, H, W, _ = frames_f16_np.shape
        lat_tile_H = tile_H // 8
        lat_tile_W = tile_W // 8
        lat_ov_H = ov_H // 8
        lat_ov_W = ov_W // 8
        lat_H, lat_W = H // 8, W // 8
        T_lat = 1 if T == 1 else (T - 1) // 4 + 1

        n_h = max(1, int(math.ceil((H - ov_H) / (tile_H - ov_H))))
        n_w = max(1, int(math.ceil((W - ov_W) / (tile_W - ov_W))))

        latent = mx.zeros((1, T_lat, lat_H, lat_W, 32), dtype=mx.float32)
        lat_wt = mx.zeros((1, T_lat, lat_H, lat_W, 32), dtype=mx.float32)

        t0 = time.perf_counter()
        tc = 0
        for ti in range(n_h):
            for tj in range(n_w):
                hs = min(ti * (tile_H - ov_H), H - tile_H)
                ws = min(tj * (tile_W - ov_W), W - tile_W)
                tile = frames_f16_np[:, hs:hs+tile_H, ws:ws+tile_W, :]
                tile_mx = mx.array(tile[np.newaxis])
                enc_out = self.encoder(tile_mx, low_mem=self.low_mem).astype(mx.float32)
                # Materialize the tile latent before blending. Otherwise the blend
                # stays fused into the full encoder lazy graph until mx.eval below,
                # which retains every encoder intermediate and can push peak VRAM
                # far above what low_mem + per-frame convs were designed for.
                mx.eval(enc_out)

                lh, lw = hs // 8, ws // 8
                bw = _make_blend_weight(lat_tile_H, lat_tile_W,
                                        (lat_ov_H, lat_ov_W), n_h, n_w, ti, tj)
                bw5d = mx.array(bw[None, None, :, :, None])
                latent, lat_wt = _gpu_blend_paste(
                    latent, lat_wt, enc_out, bw5d,
                    slice(0, T_lat),
                    slice(lh, lh + lat_tile_H),
                    slice(lw, lw + lat_tile_W),
                )
                # Eval canvas after each tile (encoder graph already released via
                # mx.eval(enc_out) above).
                mx.eval(latent, lat_wt)
                if _CLEAR_CACHE_PER_TILE:
                    mx.clear_cache()
                tc += 1
                logger.info(f"  Enc tile {tc}/{n_h*n_w}: {time.perf_counter()-t0:.1f}s")

        latent = latent / mx.maximum(lat_wt, 1e-8)
        mean = latent[:, :, :, :, :16]
        mx.eval(mean)
        mean_np = np.array(mean).astype(np.float32)
        elapsed = time.perf_counter() - t0
        logger.info(f"Encoded in {elapsed:.1f}s -> latent {mean_np.shape}")
        return mean_np

    def tiled_decode(self, latent_f32_np, tile_size=480, tile_overlap=48, num_frames=None):
        """Decode latent (numpy [1, T_lat, H_lat, W_lat, 16] float32) to frames.

        Args:
            tile_size: int (square) or (tile_H, tile_W) tuple in pixels
            tile_overlap: int (square) or (overlap_H, overlap_W) tuple in pixels
        """
        tile_H, tile_W = _as_hw(tile_size)
        ov_H, ov_W = _as_hw(tile_overlap)
        _, T_lat, lat_H, lat_W, _ = latent_f32_np.shape
        lat_tile_H = tile_H // 8
        lat_tile_W = tile_W // 8
        lat_ov_H = ov_H // 8
        lat_ov_W = ov_W // 8
        pixel_H = lat_H * 8
        pixel_W = lat_W * 8
        T_out = 1 if T_lat == 1 else (T_lat - 1) * 4 + 1

        n_h = max(1, int(math.ceil((lat_H - lat_ov_H) / (lat_tile_H - lat_ov_H))))
        n_w = max(1, int(math.ceil((lat_W - lat_ov_W) / (lat_tile_W - lat_ov_W))))

        output = mx.zeros((1, T_out, pixel_H, pixel_W, 3), dtype=mx.float32)
        out_wt = mx.zeros((1, T_out, pixel_H, pixel_W, 3), dtype=mx.float32)

        t0 = time.perf_counter()
        tc = 0
        for ti in range(n_h):
            for tj in range(n_w):
                lh = min(ti * (lat_tile_H - lat_ov_H), lat_H - lat_tile_H)
                lw = min(tj * (lat_tile_W - lat_ov_W), lat_W - lat_tile_W)
                tile_z = mx.array(
                    latent_f32_np[:, :, lh:lh+lat_tile_H, lw:lw+lat_tile_W, :]
                    .astype(np.float16)
                )
                tile_out = self.decoder(tile_z, low_mem=self.low_mem).astype(mx.float32)
                mx.eval(tile_out)

                ph, pw = lh * 8, lw * 8
                bw = _make_blend_weight(tile_H, tile_W,
                                        (ov_H, ov_W), n_h, n_w, ti, tj)
                bw5d = mx.array(bw[None, None, :, :, None])
                output, out_wt = _gpu_blend_paste(
                    output, out_wt, tile_out, bw5d,
                    slice(0, T_out),
                    slice(ph, ph + tile_H),
                    slice(pw, pw + tile_W),
                )
                mx.eval(output, out_wt)
                if _CLEAR_CACHE_PER_TILE:
                    mx.clear_cache()
                tc += 1
                logger.info(f"  Dec tile {tc}/{n_h*n_w}: {time.perf_counter()-t0:.1f}s")

        output = output / mx.maximum(out_wt, 1e-8)
        mx.eval(output)
        output_np = np.array(output).astype(np.float32)
        elapsed = time.perf_counter() - t0
        frames = output_np[0]
        if num_frames is not None and frames.shape[0] > num_frames:
            frames = frames[:num_frames]
        logger.info(f"Decoded in {elapsed:.1f}s -> {frames.shape}")
        return frames

    def encode_torch(self, frames_torch, tile_size=480, tile_overlap=48):
        """Bridge: PyTorch [T,H,W,C] float16 [0,1] -> encode -> PyTorch latent [1,T_lat,H,W,16]."""
        import torch
        frames_np = (frames_torch.cpu().float().numpy() * 2.0 - 1.0).astype(np.float16)
        latent_np = self.tiled_encode(frames_np, tile_size, tile_overlap)
        latent_scaled = (latent_np * VAE_SCALING_FACTOR).astype(np.float16)
        return torch.from_numpy(latent_scaled)

    def decode_torch(self, latent_torch, tile_size=480, tile_overlap=48, num_frames=None):
        """Bridge: PyTorch latent [1,T,H,W,16] -> decode -> PyTorch [T,H,W,C] float16 [0,1]."""
        import torch
        latent_np = (latent_torch.cpu().float().numpy() / VAE_SCALING_FACTOR).astype(np.float32)
        frames_np = self.tiled_decode(latent_np, tile_size, tile_overlap, num_frames)
        frames_01 = np.clip((frames_np + 1.0) / 2.0, 0.0, 1.0).astype(np.float16)
        return torch.from_numpy(frames_01)
