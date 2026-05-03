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

Forward policy: same operations as the reference path. Per-frame causal conv
assembles time with a preallocated ``[B, T, Ho, Wo, Co]`` (lower peak VRAM)
instead of ``mx.stack``; values match in real arithmetic; fp16 may differ by
ULPs vs ``stack`` (same caveat as batched vs per-frame in MLX_VAE_BATCHED_MAX_GB).
We only insert mx.eval where it materializes the same tensor values earlier
(e.g. enc_out before blend). No approximate spatial conv tiling.
"""

import math
import os
import time
import logging
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

logger = logging.getLogger(__name__)

VAE_SCALING_FACTOR = 0.9152


def _detach_copy(arr):
    """Create an independent MLX array, breaking any view/slice reference.

    MLX slices (e.g. ``xp[:, -2:]``) share the source buffer.  This helper
    stacks individual frames along dim-1 to produce a fresh contiguous
    allocation so the source buffer can be freed.  Stays entirely on GPU
    (no numpy round-trip).
    """
    if arr.ndim >= 2 and arr.shape[1] > 0:
        frames = [arr[:, i] for i in range(arr.shape[1])]
        out = mx.stack(frames, axis=1)
    else:
        out = arr + 0  # fallback: arithmetic forces new buffer
    mx.eval(out)
    return out

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

# Streaming-mode adaptive threshold: use fast batched conv when the stacked
# intermediate fits within this budget.  Later encoder stages (160x160,
# 80x80) have small intermediates that are much faster batched than per-frame.
# Default 1 GB: batched at stages 2/3/mid, per-frame at stages 0/1.
_STREAMING_BATCHED_MAX = int(float(os.environ.get("MLX_VAE_STREAMING_BATCHED_MAX_GB", "1.0")) * 1024**3)

# Threshold for intermediate mx.eval calls in ResNet/DownEncoder/MidBlock.
# Skip eval when the activation is below this size — lets MLX fuse
# norm+silu+conv at later stages where tensors are small.
_EVAL_MIN_BYTES = int(float(os.environ.get("MLX_VAE_EVAL_MIN_MB", "100")) * 1024 * 1024)


def _should_eval(x):
    """Return True if x is large enough to warrant an mx.eval checkpoint."""
    return x.nbytes >= _EVAL_MIN_BYTES

# Like TRT path empty_cache between tiles: drop MLX buffer pool between spatial
# tiles (slower). Set MLX_VAE_CLEAR_CACHE_PER_TILE=1 when chasing peak footprint.
_CLEAR_CACHE_PER_TILE = os.environ.get("MLX_VAE_CLEAR_CACHE_PER_TILE", "0") == "1"

# Very detailed VRAM logging for the first spatial tile only (encoder + blend).
# Set MLX_VAE_MEM_PROFILE_FIRST_TILE=1. Each line resets peak before the segment,
# then logs peak during that segment (MiB), plus active/cache after mx.eval.
# Run: python profile_mlx_vae_encoder_tile.py (see script docstring).
_MEM_PROFILE_FIRST_TILE = os.environ.get("MLX_VAE_MEM_PROFILE_FIRST_TILE", "0") == "1"

# Profiling shows peak segments at full-res 480²: enc.down[0].resnet[0] / resnet[1]
# (Metal conv workspace). Optional cache flush between ResNets (slower).
_CLEAR_CACHE_BETWEEN_RESNETS = os.environ.get(
    "MLX_VAE_CLEAR_CACHE_BETWEEN_RESNETS", "0"
) == "1"
_enc_mem_profiling = False  # toggled around encoder + first-tile blend in tiled_encode


def _mem_prof_active() -> bool:
    return _enc_mem_profiling and _MEM_PROFILE_FIRST_TILE


def _mem_seg_reset():
    if _mem_prof_active():
        mx.reset_peak_memory()


def _mem_seg_dump(tag: str, *arrays):
    """Log segment peak (since last _mem_seg_reset), active, cache; optional shapes."""
    if not _mem_prof_active():
        return
    if arrays:
        mx.eval(*arrays)
    peak = mx.get_peak_memory() / (1024 * 1024)
    active = mx.get_active_memory() / (1024 * 1024)
    cache = mx.get_cache_memory() / (1024 * 1024)
    shp = tuple(arrays[0].shape) if arrays else ()
    logger.info(
        "[MLX_VAE_MEM] %-52s shape=%-28s seg_peak=%8.2f MiB  active=%8.2f MiB  cache=%8.2f MiB",
        tag, str(shp), peak, active, cache,
    )


def _mem_tile_dump(tag: str, *arrays, do_eval: bool = True):
    """First-tile-only summary line (caller gates with prof). peak = since last global reset_peak."""
    if do_eval and arrays:
        mx.eval(*arrays)
    peak = mx.get_peak_memory() / (1024 * 1024)
    active = mx.get_active_memory() / (1024 * 1024)
    cache = mx.get_cache_memory() / (1024 * 1024)
    shp = tuple(arrays[0].shape) if arrays else ()
    logger.info(
        "[MLX_VAE_MEM_TILE] %-50s shape=%-26s peak=%8.2f MiB  active=%8.2f MiB  cache=%8.2f MiB",
        tag, str(shp), peak, active, cache,
    )


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
        # retaining T separate arrays plus a contiguous concat (lower peak VRAM).
        o0 = self._do_conv(
            mx.concatenate(
                [x[:, max(0, 0 + k - pad_n)] for k in range(self.kT)],
                axis=-1,
            )
        )
        mx.eval(o0)
        _, Ho, Wo, Co = o0.shape
        # Per-scatter eval breaks the lazy chain so each frame's o can be freed
        # immediately. Only worthwhile when per-frame output is large enough
        # that the T-frame chain accumulation (T × frame_bytes) matters.
        frame_bytes = Ho * Wo * Co * 2
        eager_scatter = (frame_bytes * T) > 200 * 1024 * 1024  # >200 MB chain
        out = mx.zeros((B, T, Ho, Wo, Co), dtype=o0.dtype)
        out[:, 0] = o0
        if eager_scatter:
            mx.eval(out)
        for t in range(1, T):
            stacked = mx.concatenate(
                [x[:, max(0, t + k - pad_n)] for k in range(self.kT)],
                axis=-1,
            )
            o = self._do_conv(stacked)
            mx.eval(o)
            out[:, t] = o
            if eager_scatter:
                mx.eval(out)
        if not eager_scatter:
            mx.eval(out)
        return out

    # ── Streaming cache for temporal continuity across slices ──────────

    def init_cache(self):
        """Prepare for streaming: clear cached tail frames."""
        self._cache = None

    def reset_cache(self):
        """Reset streaming state (equivalent to init_cache)."""
        self._cache = None

    def _streaming(self, x, low_mem=False):
        """Process a temporal slice using cached tail from the previous call.

        First call pads with replicated first frame (identical to _batched).
        Subsequent calls prepend the cached (kT-1) frames for continuity.
        Output T equals input T for every call.

        When ``low_mem=True`` (the default for MLXVideoVAE), frames are
        processed one at a time with ``mx.eval`` after each, matching the
        ``_per_frame`` memory profile.  Otherwise all T frames are stacked
        and convolved in one shot (fast but large intermediate).
        """
        B, T, H, W, C = x.shape
        if self.kT <= 1:
            return self._batched(x)

        pad_n = self.temporal_padding * 2  # kT - 1
        if self._cache is None:
            first = mx.broadcast_to(x[:, :1], [B, pad_n, H, W, C])
            xp = mx.concatenate([first, x], axis=1)
        else:
            xp = mx.concatenate([self._cache, x], axis=1)

        self._cache = _detach_copy(xp[:, -(self.kT - 1):])

        T_out = xp.shape[1] - self.kT + 1  # == T (stride=1)

        # Adaptive: use fast batched path when the stacked intermediate is
        # small enough (later encoder stages).  Same heuristic as __call__.
        stacked_bytes = B * T_out * H * W * self.kT * C * 2
        if not low_mem or stacked_bytes <= _STREAMING_BATCHED_MAX:
            windows = [xp[:, i : i + T_out] for i in range(self.kT)]
            stacked = mx.concatenate(windows, axis=-1)
            flat = stacked.reshape(B * T_out, H, W, self.kT * C)
            out = self._do_conv(flat)
            _, Ho, Wo, Co = out.shape
            return out.reshape(B, T_out, Ho, Wo, Co)

        # Per-frame path — mirrors _per_frame's eager_scatter heuristic.
        o0 = self._do_conv(
            mx.concatenate([xp[:, 0 + k] for k in range(self.kT)], axis=-1)
        )
        mx.eval(o0)
        _, Ho, Wo, Co = o0.shape
        frame_bytes = Ho * Wo * Co * 2
        eager_scatter = (frame_bytes * T_out) > 200 * 1024 * 1024
        out = mx.zeros((B, T_out, Ho, Wo, Co), dtype=o0.dtype)
        out[:, 0] = o0
        if eager_scatter:
            mx.eval(out)
        del o0
        for t in range(1, T_out):
            ot = self._do_conv(
                mx.concatenate([xp[:, t + k] for k in range(self.kT)], axis=-1)
            )
            mx.eval(ot)
            out[:, t] = ot
            if eager_scatter:
                mx.eval(out)
            del ot
        if not eager_scatter:
            mx.eval(out)
        return out

    def __call__(self, x, low_mem=False, streaming=False):
        if streaming:
            return self._streaming(x, low_mem=low_mem)
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
        frame_bytes = Ho * Wo * Co * 2
        eager_scatter = (frame_bytes * T_out) > 200 * 1024 * 1024
        out = mx.zeros((B, T_out, Ho, Wo, Co), dtype=o0.dtype)
        out[:, 0] = o0
        if eager_scatter:
            mx.eval(out)
        for t in range(1, T_out):
            t_start = t * self.stride_t
            stacked = mx.concatenate(
                [x[:, max(0, t_start + k - pad_n)] for k in range(self.kT)],
                axis=-1,
            )
            o = self._do_conv(stacked)
            mx.eval(o)
            out[:, t] = o
            if eager_scatter:
                mx.eval(out)
        if not eager_scatter:
            mx.eval(out)
        return out

    # ── Streaming cache for temporal continuity across slices ──────────

    def init_cache(self):
        self._cache = None

    def reset_cache(self):
        self._cache = None

    def _streaming(self, x, low_mem=False):
        """Process a temporal slice with strided conv cache.

        For stride_t > 1 the cache stores the tail of the padded sequence
        from ``T_out * stride_t`` onward — this naturally carries both the
        (kT - stride_t) overlap context AND any unconsumed remainder frames,
        so the next call's first window lines up perfectly with the full-pass
        output sequence.

        When ``low_mem=True``, processes one output frame at a time with
        ``mx.eval`` after each to limit peak memory.
        """
        B, T, H, W, C = x.shape
        if self.kT <= 1:
            return self._batched(x)

        if self._cache is None:
            pad_n = self.temporal_padding * 2
            first = mx.broadcast_to(x[:, :1], [B, pad_n, H, W, C])
            xp = mx.concatenate([first, x], axis=1)
        else:
            xp = mx.concatenate([self._cache, x], axis=1)

        T_padded = xp.shape[1]
        T_out = (T_padded - self.kT) // self.stride_t + 1

        if T_out <= 0:
            self._cache = _detach_copy(xp)
            Ho = (H + 2 * self._pad[0] - (self.weight.shape[1] - 1) - 1) // self.stride_hw[0] + 1
            Wo = (W + 2 * self._pad[1] - (self.weight.shape[2] - 1) - 1) // self.stride_hw[1] + 1
            Co = self.weight.shape[0]
            return mx.zeros((B, 0, Ho, Wo, Co), dtype=x.dtype)

        cache_start = T_out * self.stride_t
        _raw = xp[:, cache_start:] if cache_start < T_padded else xp[:, -(self.kT - self.stride_t):]
        self._cache = _detach_copy(_raw)
        del _raw

        # Adaptive: use fast batched path when stacked intermediate is small.
        stacked_bytes = B * T_out * H * W * self.kT * C * 2
        if not low_mem or stacked_bytes <= _STREAMING_BATCHED_MAX:
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

        # Per-frame path — mirrors _per_frame's eager_scatter heuristic.
        t_start0 = 0
        o0 = self._do_conv(
            mx.concatenate([xp[:, t_start0 + k] for k in range(self.kT)], axis=-1)
        )
        mx.eval(o0)
        _, Ho, Wo, Co = o0.shape
        frame_bytes = Ho * Wo * Co * 2
        eager_scatter = (frame_bytes * T_out) > 200 * 1024 * 1024
        out = mx.zeros((B, T_out, Ho, Wo, Co), dtype=o0.dtype)
        out[:, 0] = o0
        if eager_scatter:
            mx.eval(out)
        del o0
        for t in range(1, T_out):
            t_start = t * self.stride_t
            ot = self._do_conv(
                mx.concatenate([xp[:, t_start + k] for k in range(self.kT)], axis=-1)
            )
            mx.eval(ot)
            out[:, t] = ot
            if eager_scatter:
                mx.eval(out)
            del ot
        if not eager_scatter:
            mx.eval(out)
        return out

    def __call__(self, x, low_mem=False, streaming=False):
        if streaming:
            return self._streaming(x, low_mem=low_mem)
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

    def __call__(self, x, low_mem=False, streaming=False):
        h = _silu_gn5d(self.norm1, x)
        if low_mem and _should_eval(h): mx.eval(h)
        h = self.conv1(h, low_mem=low_mem, streaming=streaming)
        h = _silu_gn5d(self.norm2, h)
        if low_mem and _should_eval(h): mx.eval(h)
        h = self.conv2(h, low_mem=low_mem, streaming=streaming)
        if self.has_shortcut:
            x = self.conv_shortcut(x, low_mem=low_mem, streaming=streaming)
        out = x + h
        if low_mem and _should_eval(out):
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
        # The PyTorch reference VAE sets diffusers Attention(upcast_softmax=
        # True) at attn_video_vae.py:629. MLX's mx.fast.scaled_dot_product_
        # attention has no softmax-only upcast knob — but empirical A/B test
        # against the reference (v2 vs v3 benchmark) showed bit-equivalent
        # output, confirming Metal's fused SDPA already upcasts the softmax
        # reduction internally. Running in bf16 here is ~13% faster than
        # explicitly upcasting Q/K/V to fp32 with no observable quality loss.
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

    def __call__(self, x, low_mem=False, streaming=False):
        B, T, H, W, C = x.shape
        if _mem_prof_active():
            _mem_seg_reset()
        x = self.resnets[0](x, low_mem=low_mem, streaming=streaming)
        if low_mem and _should_eval(x): mx.eval(x)
        if _mem_prof_active():
            _mem_seg_dump("enc.mid.resnet[0]", x)

        flat = x.reshape(B * T, H, W, C)
        if _mem_prof_active():
            _mem_seg_reset()
        flat = self.attentions[0](flat)
        if _mem_prof_active():
            _mem_seg_dump("enc.mid.attention", flat)

        x = flat.reshape(B, T, H, W, C)
        if low_mem and _should_eval(x): mx.eval(x)
        if _mem_prof_active():
            _mem_seg_dump("enc.mid.after_reshape", x)

        if _mem_prof_active():
            _mem_seg_reset()
        x = self.resnets[1](x, low_mem=low_mem, streaming=streaming)
        if low_mem and _should_eval(x): mx.eval(x)
        if _mem_prof_active():
            _mem_seg_dump("enc.mid.resnet[1]", x)
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

    def init_cache(self):
        self._first_streaming_call = True
        self.upscale_conv.init_cache()
        self.conv.init_cache()

    def reset_cache(self):
        self._first_streaming_call = True
        self.upscale_conv.reset_cache()
        self.conv.reset_cache()

    def __call__(self, x, low_mem=False, streaming=False):
        B, T, H, W, C = x.shape
        sr = self.spatial_ratio
        tr = self.temporal_ratio
        up = self.upscale_conv(x, low_mem=low_mem, streaming=streaming)
        up = up.reshape(B, T, H, W, sr, sr, tr, C)
        up = up.transpose(0, 1, 6, 2, 4, 3, 5, 7)
        up = up.reshape(B, T * tr, H * sr, W * sr, C)
        if self.temporal_up:
            if streaming and not self._first_streaming_call:
                pass  # all frames valid after the first slice
            else:
                up = mx.concatenate([up[:, :1], up[:, 2:]], axis=1)
            if streaming:
                self._first_streaming_call = False
        return self.conv(up, low_mem=low_mem, streaming=streaming)


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

    def __call__(self, x, low_mem=False, streaming=False):
        for j, r in enumerate(self.resnets):
            x = r(x, low_mem=low_mem, streaming=streaming)
            if low_mem and _should_eval(x): mx.eval(x)
            if low_mem and _CLEAR_CACHE_BETWEEN_RESNETS and j + 1 < len(self.resnets):
                mx.clear_cache()
        for u in self.upsamplers:
            x = u(x, low_mem=low_mem, streaming=streaming)
            if low_mem and _should_eval(x): mx.eval(x)
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

    def _iter_causal_convs(self):
        """Yield all CausalConv / StridedCausalConv modules in the decoder."""
        yield self.conv_in
        for r in self.mid_block.resnets:
            yield r.conv1; yield r.conv2
            if r.has_shortcut:
                yield r.conv_shortcut
        for blk in self.up_blocks:
            for r in blk.resnets:
                yield r.conv1; yield r.conv2
                if r.has_shortcut:
                    yield r.conv_shortcut
            for u in blk.upsamplers:
                yield u.upscale_conv; yield u.conv
        yield self.conv_out

    def init_all_caches(self):
        for c in self._iter_causal_convs():
            c.init_cache()
        for blk in self.up_blocks:
            for u in blk.upsamplers:
                u.init_cache()

    def reset_all_caches(self):
        for c in self._iter_causal_convs():
            c.reset_cache()
        for blk in self.up_blocks:
            for u in blk.upsamplers:
                u.reset_cache()

    def __call__(self, z, low_mem=False, streaming=False):
        x = self.conv_in(z, low_mem=low_mem, streaming=streaming)
        if low_mem and _should_eval(x): mx.eval(x)
        x = self.mid_block(x, low_mem=low_mem, streaming=streaming)
        for blk in self.up_blocks:
            x = blk(x, low_mem=low_mem, streaming=streaming)
        x = _silu_gn5d(self.conv_norm_out, x)
        if low_mem and _should_eval(x): mx.eval(x)
        x = self.conv_out(x, low_mem=low_mem, streaming=streaming)
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

    def __call__(self, x, low_mem=False, streaming=False):
        x = mx.pad(x, [(0, 0), (0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x, low_mem=low_mem, streaming=streaming)


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

    def __call__(self, x, low_mem=False, mem_tag=None, streaming=False):
        for j, r in enumerate(self.resnets):
            if _mem_prof_active() and mem_tag:
                _mem_seg_reset()
            x = r(x, low_mem=low_mem, streaming=streaming)
            if low_mem and _should_eval(x): mx.eval(x)
            if _mem_prof_active() and mem_tag:
                _mem_seg_dump(f"{mem_tag}.resnet[{j}]", x)
            if low_mem and _CLEAR_CACHE_BETWEEN_RESNETS and j + 1 < len(self.resnets):
                mx.clear_cache()
        for k, d in enumerate(self.downsamplers):
            if _mem_prof_active() and mem_tag:
                _mem_seg_reset()
            x = d(x, low_mem=low_mem, streaming=streaming)
            if low_mem and _should_eval(x): mx.eval(x)
            if _mem_prof_active() and mem_tag:
                _mem_seg_dump(f"{mem_tag}.downsample[{k}]", x)
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

    def _iter_causal_convs(self):
        """Yield all CausalConv / StridedCausalConv modules in the encoder."""
        yield self.conv_in
        for blk in self.down_blocks:
            for r in blk.resnets:
                yield r.conv1; yield r.conv2
                if r.has_shortcut:
                    yield r.conv_shortcut
            for d in blk.downsamplers:
                yield d.conv
        for r in self.mid_block.resnets:
            yield r.conv1; yield r.conv2
            if r.has_shortcut:
                yield r.conv_shortcut
        yield self.conv_out

    def init_all_caches(self):
        for c in self._iter_causal_convs():
            c.init_cache()

    def reset_all_caches(self):
        for c in self._iter_causal_convs():
            c.reset_cache()

    def __call__(self, x, low_mem=False, streaming=False):
        if _mem_prof_active():
            mx.eval(x)
            _mem_seg_reset()

        if _mem_prof_active():
            _mem_seg_reset()
        x = self.conv_in(x, low_mem=low_mem, streaming=streaming)
        if low_mem and _should_eval(x): mx.eval(x)
        if _mem_prof_active():
            _mem_seg_dump("enc.01_conv_in", x)

        for i, blk in enumerate(self.down_blocks):
            if _mem_prof_active():
                _mem_seg_reset()
            x = blk(x, low_mem=low_mem, mem_tag=f"enc.down[{i}]", streaming=streaming)
            if _mem_prof_active():
                _mem_seg_dump(f"enc.02_down_block[{i}]_all", x)

        if _mem_prof_active():
            _mem_seg_reset()
        x = self.mid_block(x, low_mem=low_mem, streaming=streaming)
        if _mem_prof_active():
            _mem_seg_dump("enc.03_mid_block", x)

        if _mem_prof_active():
            _mem_seg_reset()
        x = _silu_gn5d(self.conv_norm_out, x)
        if low_mem and _should_eval(x): mx.eval(x)
        if _mem_prof_active():
            _mem_seg_dump("enc.04_conv_norm_silu", x)

        if _mem_prof_active():
            _mem_seg_reset()
        x = self.conv_out(x, low_mem=low_mem, streaming=streaming)
        if _mem_prof_active():
            _mem_seg_dump("enc.05_conv_out", x)
        return x


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
    """Load VAE weights from safetensors via native MLX.

    ``mx.load`` reads bf16/fp16/fp32 safetensors directly into ``mx.array``
    with the on-disk dtype preserved, so bf16 weights stay bf16 (numpy's
    safetensors framework can't read bf16). The final ``.astype(dtype)`` is
    a cheap cast — usually a no-op when on-disk dtype matches the target.
    ``_conv3d_to_mlx`` is dtype-agnostic and works on mx.array directly.
    """
    weights = mx.load(vae_path)
    pairs = []
    for k, arr in weights.items():
        if not k.startswith(prefix):
            continue
        if arr.ndim == 5:
            arr = _conv3d_to_mlx(arr)
        pairs.append((k[len(prefix):], arr.astype(dtype)))
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


def _read_tile_as_fp16(info, frame_start, frame_count):
    """Read a temporal sub-range from a tile memmap, returning fp16 [-1,1]."""
    _tile_dtype = np.dtype(info.get("dtype", "uint8"))
    mm = np.memmap(info["path"], dtype=_tile_dtype, mode='r',
                   shape=tuple(info["shape"]))
    tile_raw = np.array(mm[frame_start:frame_start + frame_count])
    del mm
    if _tile_dtype == np.uint8:
        return tile_raw.astype(np.float16) / np.float16(127.5) - np.float16(1.0)
    if _tile_dtype == np.uint16:
        return tile_raw.astype(np.float16) / np.float16(32767.5) - np.float16(1.0)
    return tile_raw.astype(np.float16)


def assemble_tiles_from_disk(tile_infos, pixel_H, pixel_W, num_frames=None):
    """Blend decoded pixel tiles from memmap files into a single output.

    Args:
        tile_infos: list of dicts from ``tiled_decode_streaming``.
        pixel_H, pixel_W: full output resolution.
        num_frames: optional frame cap.

    Returns:
        output: numpy ``[T, H, W, 3]`` float16.
    """
    if not tile_infos:
        raise ValueError("No tile_infos provided")

    info0 = tile_infos[0]
    T_pix = info0["shape"][0]
    if num_frames is not None:
        T_pix = min(T_pix, num_frames)
    tile_H = info0["tile_H"]
    tile_W = info0["tile_W"]
    ov_H = info0["ov_H"]
    ov_W = info0["ov_W"]
    n_h = info0["n_h"]
    n_w = info0["n_w"]

    # Precompute total blend weight map (spatial only)
    total_wt = np.zeros((pixel_H, pixel_W), dtype=np.float32)
    for info in tile_infos:
        bw = _make_blend_weight(tile_H, tile_W, (ov_H, ov_W),
                                n_h, n_w, info["ti"], info["tj"])
        ph, pw = info["pixel_h"], info["pixel_w"]
        clip_h = min(tile_H, pixel_H - ph)
        clip_w = min(tile_W, pixel_W - pw)
        total_wt[ph:ph+clip_h, pw:pw+clip_w] += bw[:clip_h, :clip_w]
    total_wt = np.maximum(total_wt, 1e-4)

    output = np.zeros((T_pix, pixel_H, pixel_W, 3), dtype=np.float32)

    t0 = time.perf_counter()
    for info in tile_infos:
        tile_data = _read_tile_as_fp16(info, 0, T_pix).astype(np.float32)

        ph, pw = info["pixel_h"], info["pixel_w"]
        clip_h = min(tile_H, pixel_H - ph)
        clip_w = min(tile_W, pixel_W - pw)
        bw = _make_blend_weight(tile_H, tile_W, (ov_H, ov_W),
                                n_h, n_w, info["ti"], info["tj"])
        bw = bw[:clip_h, :clip_w]
        norm_bw = bw / total_wt[ph:ph+clip_h, pw:pw+clip_w]
        bw4d = norm_bw[None, :, :, None]
        output[:, ph:ph+clip_h, pw:pw+clip_w, :] += (
            tile_data[:, :clip_h, :clip_w, :] * bw4d
        )
        del tile_data

    elapsed = time.perf_counter() - t0
    logger.info(f"Assembled tiles from disk in {elapsed:.1f}s -> {output.shape}")
    return output.astype(np.float16)


def assemble_tiles_from_disk_range(tile_infos, pixel_H, pixel_W,
                                   frame_start, frame_count):
    """Blend a temporal sub-range of decoded pixel tiles from memmap files.

    Same spatial blending as ``assemble_tiles_from_disk`` but only reads
    frames ``[frame_start : frame_start + frame_count]`` from each tile,
    allowing the caller to process the output in small temporal batches
    without materialising the full ``[T, H, W, 3]`` tensor.

    Args:
        tile_infos: list of dicts from ``tiled_decode_streaming``.
        pixel_H, pixel_W: full output spatial resolution.
        frame_start: first pixel frame index to read.
        frame_count: number of pixel frames to read.

    Returns:
        output: numpy ``[frame_count, H, W, 3]`` float16.
    """
    if not tile_infos:
        raise ValueError("No tile_infos provided")

    info0 = tile_infos[0]
    tile_H = info0["tile_H"]
    tile_W = info0["tile_W"]
    ov_H = info0["ov_H"]
    ov_W = info0["ov_W"]
    n_h = info0["n_h"]
    n_w = info0["n_w"]

    total_wt = np.zeros((pixel_H, pixel_W), dtype=np.float32)
    for info in tile_infos:
        bw = _make_blend_weight(tile_H, tile_W, (ov_H, ov_W),
                                n_h, n_w, info["ti"], info["tj"])
        ph, pw = info["pixel_h"], info["pixel_w"]
        clip_h = min(tile_H, pixel_H - ph)
        clip_w = min(tile_W, pixel_W - pw)
        total_wt[ph:ph+clip_h, pw:pw+clip_w] += bw[:clip_h, :clip_w]
    total_wt = np.maximum(total_wt, 1e-4)

    output = np.zeros((frame_count, pixel_H, pixel_W, 3), dtype=np.float32)

    fs, fe = frame_start, frame_start + frame_count
    for info in tile_infos:
        tile_data = _read_tile_as_fp16(info, fs, fe - fs).astype(np.float32)

        ph, pw = info["pixel_h"], info["pixel_w"]
        clip_h = min(tile_H, pixel_H - ph)
        clip_w = min(tile_W, pixel_W - pw)
        bw = _make_blend_weight(tile_H, tile_W, (ov_H, ov_W),
                                n_h, n_w, info["ti"], info["tj"])
        bw = bw[:clip_h, :clip_w]
        norm_bw = bw / total_wt[ph:ph+clip_h, pw:pw+clip_w]
        bw4d = norm_bw[None, :, :, None]
        output[:, ph:ph+clip_h, pw:pw+clip_w, :] += (
            tile_data[:, :clip_h, :clip_w, :] * bw4d
        )
        del tile_data

    return output.astype(np.float16)


def cleanup_tile_files(tile_infos):
    """Remove temporary memmap files and their parent directory."""
    dirs = set()
    for info in tile_infos:
        p = info["path"]
        if os.path.exists(p):
            os.remove(p)
        dirs.add(os.path.dirname(p))
    for d in dirs:
        try:
            os.rmdir(d)
        except OSError:
            pass


# ── High-Level Wrapper ───────────────────────────────────────────────────────

class MLXVideoVAE:
    """High-level wrapper: holds encoder/decoder, provides tiled encode/decode,
    and bridges PyTorch <-> MLX tensors."""

    def __init__(self, vae_path, cache_limit_gb=1.0, dtype=mx.bfloat16, low_mem=True):
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

    def tiled_encode(
        self,
        frames: Union[np.ndarray, "torch.Tensor"],
        tile_size=480,
        tile_overlap=48,
    ):
        """Encode video frames to latent using spatial tiling.

        Args:
            frames: numpy ``[T, H, W, 3]`` in [-1, 1] (float16/float32), **or**
                a PyTorch **CPU** tensor with the same layout. When a tensor is
                passed, each spatial tile is converted to NumPy only for that
                tile (avoids a second full-frame NumPy copy of the whole video).
            tile_size: int (square) or (tile_H, tile_W) tuple in pixels
            tile_overlap: int (square) or (overlap_H, overlap_W) tuple in pixels

        Returns:
            latent: numpy array [1, T_lat, H_lat, W_lat, 16] float32 (mean only)
        """
        # Free decoder weights during encoding.
        if getattr(self, 'decoder', None) is not None:
            self.decoder = None
            mx.clear_cache()

        # Lazily reload encoder if it was freed during decode.
        if getattr(self, 'encoder', None) is None:
            logger.info("Reloading MLX VAE encoder weights...")
            self.encoder = Encoder(in_ch=3, out_ch=32,
                                   block_channels=(128, 256, 512, 512),
                                   layers_per_block=2, groups=32,
                                   temporal_down_num=2, kT=3)
            _load_weights(self.encoder, self.vae_path, "encoder.", self.dtype)
            mx.eval(self.encoder.parameters())

        tile_H, tile_W = _as_hw(tile_size)
        ov_H, ov_W = _as_hw(tile_overlap)

        torch_frames = None
        frames_np = None
        try:
            import torch

            _torch_ok = isinstance(frames, torch.Tensor)
        except ImportError:
            _torch_ok = False

        if _torch_ok:
            torch_frames = frames.contiguous()
            if torch_frames.dtype == torch.bfloat16:
                torch_frames = torch_frames.to(torch.float16)
            elif torch_frames.dtype not in (torch.float16, torch.float32):
                torch_frames = torch_frames.float()
            T, H, W, _ = torch_frames.shape
        else:
            frames_np = np.asarray(frames)
            T, H, W, _ = frames_np.shape
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
        global _enc_mem_profiling
        for ti in range(n_h):
            for tj in range(n_w):
                hs = min(ti * (tile_H - ov_H), H - tile_H)
                ws = min(tj * (tile_W - ov_W), W - tile_W)
                if torch_frames is not None:
                    # One tile worth of host memory at a time (no full-frame np).
                    tile_np = (
                        torch_frames[:, hs : hs + tile_H, ws : ws + tile_W, :]
                        .contiguous()
                        .numpy()
                    )
                else:
                    tile_np = frames_np[:, hs : hs + tile_H, ws : ws + tile_W, :]
                tile_mx = mx.array(tile_np[np.newaxis]).astype(self.dtype)
                del tile_np
                prof = _MEM_PROFILE_FIRST_TILE and ti == 0 and tj == 0
                if prof:
                    _enc_mem_profiling = True
                enc_out = self.encoder(tile_mx, low_mem=self.low_mem).astype(mx.float32)
                if prof:
                    _enc_mem_profiling = False
                # Materialize the tile latent before blending. Otherwise the blend
                # stays fused into the full encoder lazy graph until mx.eval below,
                # which retains every encoder intermediate and can push peak VRAM
                # far above what low_mem + per-frame convs were designed for.
                mx.eval(enc_out)
                del tile_mx
                if prof:
                    _mem_tile_dump("tile.after_encoder_mx_eval", enc_out, do_eval=False)

                lh, lw = hs // 8, ws // 8
                bw = _make_blend_weight(lat_tile_H, lat_tile_W,
                                        (lat_ov_H, lat_ov_W), n_h, n_w, ti, tj)
                bw5d = mx.array(bw[None, None, :, :, None])
                if prof:
                    mx.reset_peak_memory()
                latent, lat_wt = _gpu_blend_paste(
                    latent, lat_wt, enc_out, bw5d,
                    slice(0, T_lat),
                    slice(lh, lh + lat_tile_H),
                    slice(lw, lw + lat_tile_W),
                )
                del enc_out, bw5d
                mx.eval(latent, lat_wt)
                if prof:
                    _mem_tile_dump("tile.after_blend_paste_mx_eval", latent, lat_wt, do_eval=False)
                tc += 1
                logger.info(f"  Enc tile {tc}/{n_h*n_w}: {time.perf_counter()-t0:.1f}s")

        if torch_frames is not None:
            del torch_frames

        latent = latent / mx.maximum(lat_wt, 1e-8)
        mean = latent[:, :, :, :, :16]
        mx.eval(mean)
        mean_np = np.array(mean).astype(np.float32)
        del latent, lat_wt
        mx.clear_cache()
        elapsed = time.perf_counter() - t0
        logger.info(f"Encoded in {elapsed:.1f}s -> latent {mean_np.shape}")
        return mean_np

    def tiled_decode(self, latent_f32_np, tile_size=480, tile_overlap=48, num_frames=None):
        """Decode latent (numpy [1, T_lat, H_lat, W_lat, 16] float32) to frames.

        Args:
            tile_size: int (square) or (tile_H, tile_W) tuple in pixels
            tile_overlap: int (square) or (overlap_H, overlap_W) tuple in pixels
        """
        # Free encoder weights during decoding.
        if getattr(self, 'encoder', None) is not None:
            self.encoder = None
            mx.clear_cache()

        # Lazily reload decoder if it was freed during encode.
        if getattr(self, 'decoder', None) is None:
            logger.info("Reloading MLX VAE decoder weights...")
            self.decoder = Decoder(in_ch=16, out_ch=3,
                                   block_channels=(128, 256, 512, 512),
                                   layers_per_block=2, groups=32,
                                   temporal_up_num=2, kT=3)
            _load_weights(self.decoder, self.vae_path, "decoder.", self.dtype)
            mx.eval(self.decoder.parameters())

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

        # Precompute 2D total blend-weight map (~16 MB at 4K vs ~1.6 GB for a
        # full 5D out_wt canvas).  Blend weights are purely spatial — identical
        # across T and channels — so a [pixel_H, pixel_W] accumulator suffices.
        total_wt = np.zeros((pixel_H, pixel_W), dtype=np.float32)
        for ti in range(n_h):
            for tj in range(n_w):
                lh = min(ti * (lat_tile_H - lat_ov_H), lat_H - lat_tile_H)
                lw = min(tj * (lat_tile_W - lat_ov_W), lat_W - lat_tile_W)
                ph, pw = lh * 8, lw * 8
                bw = _make_blend_weight(tile_H, tile_W,
                                        (ov_H, ov_W), n_h, n_w, ti, tj)
                clip_h = min(tile_H, pixel_H - ph)
                clip_w = min(tile_W, pixel_W - pw)
                total_wt[ph:ph+clip_h, pw:pw+clip_w] += bw[:clip_h, :clip_w]
        total_wt = np.maximum(total_wt, 1e-4)

        output = np.zeros((1, T_out, pixel_H, pixel_W, 3), dtype=np.float32)

        t0 = time.perf_counter()
        tc = 0
        for ti in range(n_h):
            for tj in range(n_w):
                lh = min(ti * (lat_tile_H - lat_ov_H), lat_H - lat_tile_H)
                lw = min(tj * (lat_tile_W - lat_ov_W), lat_W - lat_tile_W)
                tile_z = mx.array(
                    latent_f32_np[:, :, lh:lh+lat_tile_H, lw:lw+lat_tile_W, :]
                ).astype(self.dtype)
                tile_out = self.decoder(tile_z, low_mem=self.low_mem).astype(mx.float32)
                mx.eval(tile_out)

                ph, pw = lh * 8, lw * 8
                clip_h = min(tile_H, pixel_H - ph)
                clip_w = min(tile_W, pixel_W - pw)
                bw = _make_blend_weight(tile_H, tile_W,
                                        (ov_H, ov_W), n_h, n_w, ti, tj)
                tile_np = np.asarray(tile_out)
                del tile_out
                norm_bw = bw[:clip_h, :clip_w] / total_wt[ph:ph+clip_h, pw:pw+clip_w]
                bw5d = norm_bw[None, None, :, :, None]
                output[:, :, ph:ph+clip_h, pw:pw+clip_w, :] += (
                    tile_np[:, :, :clip_h, :clip_w, :] * bw5d
                )
                del tile_np
                tc += 1
                logger.info(f"  Dec tile {tc}/{n_h*n_w}: {time.perf_counter()-t0:.1f}s")

        output_np = output.astype(np.float16)
        del output, total_wt
        mx.clear_cache()
        elapsed = time.perf_counter() - t0
        frames = output_np[0]
        if num_frames is not None and frames.shape[0] > num_frames:
            frames = frames[:num_frames]
        logger.info(f"Decoded in {elapsed:.1f}s -> {frames.shape}")
        return frames

    # ── Streaming Encode / Decode ──────────────────────────────────────

    def tiled_encode_streaming(
        self,
        frames: Union[np.ndarray, "torch.Tensor"],
        tile_size=480,
        tile_overlap=48,
        temporal_slice=11,
        input_layout='THWC',
        pixel_scale: float = 0.0,
    ):
        """Encode frames using streaming temporal slices per spatial tile.

        Each spatial tile is processed through the encoder in small temporal
        chunks, maintaining causal conv caches for continuity.  The per-tile
        latent slices are concatenated and then blended spatially, producing
        the same result as ``tiled_encode`` but with bounded peak memory
        proportional to ``temporal_slice`` instead of the full video length.

        Args:
            frames: numpy/memmap array or PyTorch CPU tensor.
                    When ``pixel_scale > 0`` the data is uint8/uint16 in
                    [0, scale] and will be normalised to [-1, 1] on-the-fly.
                    Otherwise expected in [-1, 1] float.
                    Shape ``[T, H, W, C]`` when ``input_layout='THWC'``,
                    or ``[C, T, H, W]`` when ``input_layout='CTHW'``.
            tile_size: int or (H, W) pixel tile size.
            tile_overlap: int or (H, W) pixel overlap between tiles.
            temporal_slice: number of pixel frames per streaming chunk.
            input_layout: ``'THWC'`` or ``'CTHW'``.
            pixel_scale: if > 0, pixel values are divided by this and
                         mapped from [0,1] to [-1,1].

        Returns:
            latent: numpy ``[1, T_lat, H_lat, W_lat, 16]`` float32 (mean).
        """
        import tempfile as _tmpmod

        if getattr(self, 'decoder', None) is not None:
            self.decoder = None
            mx.clear_cache()
        if getattr(self, 'encoder', None) is None:
            logger.info("Reloading MLX VAE encoder weights...")
            self.encoder = Encoder(in_ch=3, out_ch=32,
                                   block_channels=(128, 256, 512, 512),
                                   layers_per_block=2, groups=32,
                                   temporal_down_num=2, kT=3)
            _load_weights(self.encoder, self.vae_path, "encoder.", self.dtype)
            mx.eval(self.encoder.parameters())

        tile_H, tile_W = _as_hw(tile_size)
        ov_H, ov_W = _as_hw(tile_overlap)

        _cthw_input = (input_layout == 'CTHW')
        _thwc_input = (input_layout == 'THWC')
        _need_normalize = (pixel_scale > 0)
        _owned_mm_path = None  # track internal memmap for cleanup

        if _cthw_input and not _need_normalize:
            frames_arr = np.asarray(frames)
            C_in, T, H, W = frames_arr.shape
            frames_mm = frames_arr
            del frames
        elif _thwc_input:
            frames_arr = np.asarray(frames)
            T, H, W, C_in = frames_arr.shape
            frames_mm = frames_arr
            del frames
        else:
            try:
                import torch as _torch
                _is_torch = isinstance(frames, _torch.Tensor)
            except ImportError:
                _is_torch = False

            if _is_torch:
                frames_cpu = frames.contiguous().cpu()
                if frames_cpu.dtype == _torch.bfloat16:
                    frames_cpu = frames_cpu.to(_torch.float16)
                elif frames_cpu.dtype not in (_torch.float16, _torch.float32):
                    frames_cpu = frames_cpu.float()
                frames_arr = frames_cpu.numpy()
                del frames_cpu
            else:
                frames_arr = np.asarray(frames)

            T, H, W, C_in = frames_arr.shape
            mm_dtype = frames_arr.dtype
            _owned_mm_path = _tmpmod.mktemp(prefix="vae_enc_input_", suffix=".dat")
            frames_mm = np.memmap(_owned_mm_path, dtype=mm_dtype, mode='w+',
                                  shape=(T, H, W, C_in))
            frames_mm[:] = frames_arr
            frames_mm.flush()
            del frames_arr, frames
            logger.info("Wrote encoder input to memmap (%s, %.1f MB)",
                        _owned_mm_path, os.path.getsize(_owned_mm_path) / 1e6)
            _thwc_input = True

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

        # Build temporal slice boundaries (merge last slice if too short)
        min_subsequent = 4  # minimum frames for subsequent slices (through 2 stride-2 downsamples)
        t_slices = []
        pos = 0
        while pos < T:
            end = min(pos + temporal_slice, T)
            remaining = T - end
            if 0 < remaining < min_subsequent:
                end = T  # absorb remainder into this slice
            t_slices.append((pos, end))
            pos = end
        logger.info(
            "Streaming encode: %d spatial tiles (%dx%d), %d temporal slices "
            "(slice=%d), tile=%dx%d, video=%dx%d, T=%d",
            n_h * n_w, n_h, n_w, len(t_slices), temporal_slice,
            tile_H, tile_W, W, H, T,
        )
        try:
            _active = mx.get_active_memory() / 1024**3
            _cache = mx.get_cache_memory() / 1024**3
            _peak = mx.get_peak_memory() / 1024**3
            logger.info(
                "  MLX mem before encode tiles: active=%.2f GB, cache=%.2f GB, peak=%.2f GB",
                _active, _cache, _peak,
            )
        except Exception:
            pass

        t0 = time.perf_counter()
        tc = 0
        for ti in range(n_h):
            for tj in range(n_w):
                hs = min(ti * (tile_H - ov_H), H - tile_H)
                ws = min(tj * (tile_W - ov_W), W - tile_W)

                self.encoder.init_all_caches()
                tile_latents = []

                for si, (ts, te) in enumerate(t_slices):
                    if _cthw_input and not _need_normalize:
                        cthw_crop = np.array(frames_mm[:, ts:te, hs:hs+tile_H, ws:ws+tile_W])
                        slice_np = np.transpose(cthw_crop, (1, 2, 3, 0))  # [T_s, H, W, C]
                        del cthw_crop
                    else:
                        slice_np = np.array(
                            frames_mm[ts:te, hs:hs+tile_H, ws:ws+tile_W, :]
                        )  # [T_s, tile_H, tile_W, C]

                    if _need_normalize:
                        slice_f = slice_np.astype(np.float32) / pixel_scale
                        slice_np = (slice_f * 2.0 - 1.0).astype(np.float16)
                        del slice_f

                    slice_mx = mx.array(slice_np[np.newaxis]).astype(self.dtype)
                    del slice_np
                    enc_out = self.encoder(slice_mx, low_mem=self.low_mem, streaming=True).astype(mx.float32)
                    mx.eval(enc_out)
                    del slice_mx
                    mx.clear_cache()
                    tile_latents.append(enc_out)

                enc_full = mx.concatenate(tile_latents, axis=1)
                mx.eval(enc_full)
                del tile_latents
                self.encoder.reset_all_caches()

                lh, lw = hs // 8, ws // 8
                bw = _make_blend_weight(lat_tile_H, lat_tile_W,
                                        (lat_ov_H, lat_ov_W), n_h, n_w, ti, tj)
                bw5d = mx.array(bw[None, None, :, :, None])
                latent, lat_wt = _gpu_blend_paste(
                    latent, lat_wt, enc_full, bw5d,
                    slice(0, T_lat),
                    slice(lh, lh + lat_tile_H),
                    slice(lw, lw + lat_tile_W),
                )
                del enc_full, bw5d
                mx.eval(latent, lat_wt)
                mx.clear_cache()
                tc += 1
                logger.info(f"  Enc tile {tc}/{n_h*n_w}: {time.perf_counter()-t0:.1f}s")

        # Cleanup internal memmap (only if we created one)
        del frames_mm
        if _owned_mm_path is not None:
            try:
                os.remove(_owned_mm_path)
            except OSError:
                pass

        latent = latent / mx.maximum(lat_wt, 1e-8)
        mean = latent[:, :, :, :, :16]
        mx.eval(mean)
        mean_np = np.array(mean).astype(np.float32)
        del latent, lat_wt
        mx.clear_cache()
        elapsed = time.perf_counter() - t0
        logger.info(f"Streaming encoded in {elapsed:.1f}s -> latent {mean_np.shape}")
        return mean_np

    def tiled_decode_streaming(
        self,
        latent_f32_np,
        tile_size=480,
        tile_overlap=48,
        temporal_slice=5,
        num_frames=None,
        tmp_dir=None,
        tile_dtype="uint8",
    ):
        """Decode latent using streaming temporal slices, saving tiles to disk.

        Args:
            tile_dtype: Storage format for decoded pixel tiles on disk.
                ``"uint8"`` (default) — smallest, fine for 8-bit sources.
                ``"uint16"`` — preserves 10/12/16-bit source precision.
                ``"float16"`` — raw VAE output.

        Each spatial tile is decoded in small temporal chunks (default 5 latent
        frames) and the full pixel output for that tile is written to a memmap
        file.  After all tiles are decoded, ``assemble_tiles_from_disk`` can
        blend them spatially.

        Args:
            latent_f32_np: ``[1, T_lat, H_lat, W_lat, 16]`` float32 numpy.
            tile_size: pixel tile size (int or (H, W)).
            tile_overlap: pixel overlap (int or (H, W)).
            temporal_slice: latent frames per streaming chunk.
            num_frames: optional cap on output pixel frames.
            tmp_dir: directory for memmap files (default: system temp).

        Returns:
            list of dicts with keys ``path``, ``shape``, ``dtype``,
            ``pixel_h``, ``pixel_w``, ``tile_H``, ``tile_W``, ``ti``, ``tj``,
            ``n_h``, ``n_w``, ``ov_H``, ``ov_W``.
        """
        import tempfile

        if getattr(self, 'encoder', None) is not None:
            self.encoder = None
            mx.clear_cache()
        if getattr(self, 'decoder', None) is None:
            logger.info("Reloading MLX VAE decoder weights...")
            self.decoder = Decoder(in_ch=16, out_ch=3,
                                   block_channels=(128, 256, 512, 512),
                                   layers_per_block=2, groups=32,
                                   temporal_up_num=2, kT=3)
            _load_weights(self.decoder, self.vae_path, "decoder.", self.dtype)
            mx.eval(self.decoder.parameters())

        tile_H, tile_W = _as_hw(tile_size)
        ov_H, ov_W = _as_hw(tile_overlap)
        _, T_lat, lat_H, lat_W, _ = latent_f32_np.shape
        lat_tile_H = tile_H // 8
        lat_tile_W = tile_W // 8
        lat_ov_H = ov_H // 8
        lat_ov_W = ov_W // 8
        T_pix = 1 if T_lat == 1 else (T_lat - 1) * 4 + 1
        if num_frames is not None:
            T_pix = min(T_pix, num_frames)

        n_h = max(1, int(math.ceil((lat_H - lat_ov_H) / (lat_tile_H - lat_ov_H))))
        n_w = max(1, int(math.ceil((lat_W - lat_ov_W) / (lat_tile_W - lat_ov_W))))

        # Temporal slice boundaries for latent frames
        t_slices = []
        pos = 0
        while pos < T_lat:
            end = min(pos + temporal_slice, T_lat)
            remaining = T_lat - end
            if 0 < remaining < 1:
                end = T_lat
            t_slices.append((pos, end))
            pos = end

        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="vae_tiles_")
        logger.info(
            "Streaming decode: %d spatial tiles, %d temporal slices (slice=%d lat frames), tmp=%s",
            n_h * n_w, len(t_slices), temporal_slice, tmp_dir,
        )

        t0 = time.perf_counter()
        tile_infos = []
        tc = 0
        for ti in range(n_h):
            for tj in range(n_w):
                lh = min(ti * (lat_tile_H - lat_ov_H), lat_H - lat_tile_H)
                lw = min(tj * (lat_tile_W - lat_ov_W), lat_W - lat_tile_W)
                ph, pw = lh * 8, lw * 8

                self.decoder.init_all_caches()
                pixel_slices = []

                for si, (ls, le) in enumerate(t_slices):
                    tile_z = mx.array(
                        latent_f32_np[:, ls:le, lh:lh+lat_tile_H, lw:lw+lat_tile_W, :]
                    ).astype(self.dtype)
                    tile_out = self.decoder(tile_z, low_mem=self.low_mem, streaming=True).astype(mx.float16)
                    mx.eval(tile_out)
                    del tile_z
                    tile_np = np.asarray(tile_out)
                    del tile_out
                    pixel_slices.append(tile_np[0])  # remove batch dim -> [T_s, H, W, 3]

                self.decoder.reset_all_caches()

                full_tile = np.concatenate(pixel_slices, axis=0)  # [T_pix_tile, tile_H, tile_W, 3]
                del pixel_slices
                if num_frames is not None and full_tile.shape[0] > T_pix:
                    full_tile = full_tile[:T_pix]

                mm_path = os.path.join(tmp_dir, f"tile_{ti}_{tj}.dat")
                actual_shape = full_tile.shape
                np_dtype = np.dtype(tile_dtype)
                mm = np.memmap(mm_path, dtype=np_dtype, mode='w+', shape=actual_shape)
                if tile_dtype == "uint8":
                    mm[:] = ((full_tile.astype(np.float32) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                elif tile_dtype == "uint16":
                    mm[:] = ((full_tile.astype(np.float32) + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)
                else:
                    mm[:] = full_tile.astype(np_dtype)
                mm.flush()
                del mm, full_tile

                tile_infos.append({
                    "path": mm_path,
                    "shape": actual_shape,
                    "dtype": tile_dtype,
                    "pixel_h": ph,
                    "pixel_w": pw,
                    "tile_H": tile_H,
                    "tile_W": tile_W,
                    "ti": ti, "tj": tj,
                    "n_h": n_h, "n_w": n_w,
                    "ov_H": ov_H, "ov_W": ov_W,
                })
                tc += 1
                logger.info(f"  Dec tile {tc}/{n_h*n_w}: {time.perf_counter()-t0:.1f}s")

        mx.clear_cache()
        elapsed = time.perf_counter() - t0
        logger.info(f"Streaming decoded in {elapsed:.1f}s -> {len(tile_infos)} tile files")
        return tile_infos

    def encode_torch(self, frames_torch, tile_size=480, tile_overlap=48):
        """Bridge: PyTorch [T,H,W,C] [0,1] -> encode -> PyTorch latent [1,T_lat,H,W,16]."""
        import torch
        frames_np = (frames_torch.cpu().float().numpy() * 2.0 - 1.0).astype(np.float16)
        latent_np = self.tiled_encode(frames_np, tile_size, tile_overlap)
        latent_scaled = (latent_np * VAE_SCALING_FACTOR).astype(np.float16)
        return torch.from_numpy(latent_scaled)

    def decode_torch(self, latent_torch, tile_size=480, tile_overlap=48, num_frames=None):
        """Bridge: PyTorch latent [1,T,H,W,16] -> decode -> PyTorch [T,H,W,C] [0,1]."""
        import torch
        latent_np = (latent_torch.cpu().float().numpy() / VAE_SCALING_FACTOR).astype(np.float32)
        frames_np = self.tiled_decode(latent_np, tile_size, tile_overlap, num_frames)
        frames_01 = np.clip((frames_np + 1.0) / 2.0, 0.0, 1.0).astype(np.float16)
        return torch.from_numpy(frames_01)
