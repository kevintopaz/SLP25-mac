"""
MLX port of the NaDiT 3B model for Apple Silicon.

Math-identical (within bf16 tolerance) replica of
``src/models/dit_3b/`` PyTorch modules using MLX primitives.

Architecture mirrors the original:
    NaDiT
    |-- vid_in / vid_out (NaPatch)
    |-- emb_in (TimeEmbedding)
    |-- txt_in (Linear)
    |-- 32 x NaMMSRTransformerBlock
    |   |-- attn_norm  (RMSNorm, no affine)
    |   |-- ada (AdaSingle, in/out modulation)
    |   |-- attn (NaSwinAttention with 3D MM-RoPE)
    |   |-- mlp_norm (RMSNorm, no affine)
    |   |-- mlp (SwiGLU)
    |-- vid_out_norm (RMSNorm with affine)
    |-- vid_out_ada (FixedVidOutAda)

For blocks 0-9 (mm_layers=10) the attn/mlp weights are stored separately
under ``.vid``/``.txt``; for blocks 10-31 they are shared under ``.all``.

Bridge:
    Input  PyTorch tensor -> numpy -> mx.array  (one-time per forward)
    Output mx.array -> numpy -> PyTorch tensor  (one-time per forward)
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ============================================================================
# Numerical primitives
# ============================================================================

def _rms_norm(x: mx.array, weight: Optional[mx.array], eps: float) -> mx.array:
    """RMSNorm matching CustomRMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    var = mx.mean(x.astype(mx.float32) * x.astype(mx.float32), axis=-1, keepdims=True)
    x = x.astype(mx.float32) * mx.rsqrt(var + eps)
    if weight is not None:
        x = x * weight.astype(mx.float32)
    return x.astype(weight.dtype if weight is not None else mx.bfloat16)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


# ============================================================================
# 3D MM-RoPE - matches NaMMRotaryEmbedding3d
# ============================================================================
#
# The reference uses rotary_embedding_torch with:
#   - freqs_for="lang", theta=10000
#   - dim = head_dim // rope_dim = 128 // 3 = 42 (integer division)
#   - per-axis freqs = 1 / (theta ** (arange(0, 42, 2) / 42))  -> 21 values
#
# get_axial_freqs(*dims):
#   for each axis, build pos[0..N-1] x freqs (outer product), shape (N, 21)
#   meshgrid the axes, combine: freqs(T,H,W) shape -> (T, H, W, 3*21*2=126)
#   apply_rotary_emb rotates the first 126 features, leaves last 2 unchanged
# ============================================================================


def _build_axis_freqs(theta: float, dim_per_axis: int, max_pos: int, dtype) -> mx.array:
    """
    Per-axis frequency table.  Returns shape (max_pos, dim_per_axis) with
    interleaved pattern matching rotary_embedding_torch.

    rotary_embedding_torch internally does:
        freqs_inv = 1 / theta ** (arange(0, dim_per_axis, 2) / dim_per_axis)  # (dim/2,)
        pos_freqs = einsum('..., f -> ... f', positions, freqs_inv)            # (N, dim/2)
        pos_freqs = repeat(pos_freqs, '... f -> ... (f r)', r=2)               # (N, dim) interleaved
    """
    half = dim_per_axis // 2  # 21 for dim_per_axis=42
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim_per_axis, 2, dtype=mx.float32) / dim_per_axis))
    positions = mx.arange(max_pos, dtype=mx.float32)
    pos_freqs = positions[:, None] * inv_freq[None, :]  # (max_pos, half)
    # Interleave: each freq value repeated twice -> shape (max_pos, 2*half)
    pos_freqs = mx.repeat(pos_freqs, 2, axis=-1)  # (max_pos, dim_per_axis)
    return pos_freqs.astype(dtype)


def _apply_rotary_emb(freqs: mx.array, t: mx.array) -> mx.array:
    """
    Match rotary_embedding_torch.apply_rotary_emb on the LAST dim.

    rotate_half in rotary_embedding_torch is PAIR-WISE (not split-in-half):
        rearrange(x, '... (d r) -> ... d r', r=2)   # split adjacent pairs
        stack((-x2, x1), -1).flatten()               # swap and negate even
    For [a,b,c,d,e,f] -> [-b, a, -d, c, -f, e].

    The math:  out = t * cos(freqs) + rotate_half(t) * sin(freqs)
    """
    rot_dim = freqs.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]

    cos = mx.cos(freqs.astype(mx.float32))
    sin = mx.sin(freqs.astype(mx.float32))

    # rotate_half pair-wise: pairs (a,b) -> (-b, a)
    even = t_rot[..., 0::2]
    odd = t_rot[..., 1::2]
    rotated = mx.stack([-odd, even], axis=-1).reshape(*t_rot.shape)

    out = t_rot.astype(mx.float32) * cos + rotated.astype(mx.float32) * sin
    out = out.astype(t.dtype)
    if t_pass.shape[-1] > 0:
        out = mx.concatenate([out, t_pass], axis=-1)
    return out


# ============================================================================
# Window slicing - matches src/models/dit_3b/window.py
# ============================================================================

def make_720p_windows(t: int, h: int, w: int, num_windows: Tuple[int, int, int],
                      shifted: bool = False,
                      fixed_wt: Optional[int] = None) -> List[Tuple[slice, slice, slice]]:
    """Returns the window slices for 720pwin / 720pswin (matches window.py).

    Args:
        fixed_wt: if set, overrides the computed temporal window size ``wt``
            so that longer latent sequences still use small attention windows
            (e.g. ``fixed_wt=3`` keeps the same window as a 9-latent-frame input).
    """
    nt_w, nh_w, nw_w = num_windows
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh = math.ceil(resized_h / nh_w)
    ww = math.ceil(resized_w / nw_w)
    wt = fixed_wt if fixed_wt is not None else math.ceil(min(t, 30) / nt_w)

    if not shifted:
        nt = math.ceil(t / wt)
        nh = math.ceil(h / wh)
        nw = math.ceil(w / ww)
        return [
            (slice(it * wt, min((it + 1) * wt, t)),
             slice(ih * wh, min((ih + 1) * wh, h)),
             slice(iw * ww, min((iw + 1) * ww, w)))
            for iw in range(nw)
            if min((iw + 1) * ww, w) > iw * ww
            for ih in range(nh)
            if min((ih + 1) * wh, h) > ih * wh
            for it in range(nt)
            if min((it + 1) * wt, t) > it * wt
        ]
    else:
        st = 0.5 if wt < t else 0
        sh = 0.5 if wh < h else 0
        sw = 0.5 if ww < w else 0
        nt = math.ceil((t - st) / wt) + (1 if st > 0 else 0)
        nh = math.ceil((h - sh) / wh) + (1 if sh > 0 else 0)
        nw = math.ceil((w - sw) / ww) + (1 if sw > 0 else 0)
        if st == 0:
            nt = 1
        if sh == 0:
            nh = 1
        if sw == 0:
            nw = 1
        return [
            (slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
             slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
             slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)))
            for iw in range(nw)
            if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
            for ih in range(nh)
            if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
            for it in range(nt)
            if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
        ]


# ============================================================================
# Modules
# ============================================================================

class MLXLinear:
    """nn.Linear: y = x @ W^T + b. Weight shape: (out, in)."""

    def __init__(self, weight: mx.array, bias: Optional[mx.array] = None):
        self.weight = weight  # (out, in)
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class MLXRMSNorm:
    """CustomRMSNorm with optional learned weight."""

    def __init__(self, weight: Optional[mx.array], eps: float = 1e-5):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        var = mx.mean(x.astype(mx.float32) * x.astype(mx.float32), axis=-1, keepdims=True)
        x_norm = x.astype(mx.float32) * mx.rsqrt(var + self.eps)
        if self.weight is not None:
            x_norm = x_norm * self.weight.astype(mx.float32)
            return x_norm.astype(self.weight.dtype)
        return x_norm.astype(x.dtype)


class MLXAdaSingle:
    """
    AdaSingle modulation.  Per-block parameters are 6 scalars per dim:
        attn_shift / attn_scale / attn_gate / mlp_shift / mlp_scale / mlp_gate

    The embedding (b, 6*dim) gets reshaped (b, dim, 2, 3) where l=2 (attn,mlp), g=3 (shift,scale,gate).
    For mode='in':  hid = hid * (scaleA + scaleB) + (shiftA + shiftB)
    For mode='out': hid = hid * (gateA + gateB)
    """

    def __init__(self, params: dict, dim: int, layers: List[str] = ("attn", "mlp")):
        # params dict: {"attn_shift", "attn_scale", "attn_gate", "mlp_shift", ...}
        self.params = params
        self.dim = dim
        self.layers = list(layers)

    def _emb_for_layer(self, emb: mx.array, layer: str) -> Tuple[mx.array, mx.array, mx.array]:
        """emb: (b, 6*dim) -> shift, scale, gate each (b, dim)."""
        idx = self.layers.index(layer)
        b = emb.shape[0]
        # rearrange "b (d l g) -> b d l g" with l=2, g=3
        emb_r = emb.reshape(b, self.dim, len(self.layers), 3)
        e = emb_r[:, :, idx, :]  # (b, dim, 3)
        shift = e[..., 0]
        scale = e[..., 1]
        gate = e[..., 2]
        return shift, scale, gate

    def in_modulate(self, hid: mx.array, emb: mx.array, layer: str, hid_lens: List[int]) -> mx.array:
        """hid: (L, dim).  emb: (b, 6*dim).  hid_lens: per-batch token counts."""
        shiftA, scaleA, _ = self._emb_for_layer(emb, layer)
        shiftB = self.params[f"{layer}_shift"]
        scaleB = self.params[f"{layer}_scale"]
        # repeat_interleave emb along batch -> tokens
        shift = self._repeat(shiftA, hid_lens)
        scale = self._repeat(scaleA, hid_lens)
        return hid * (scale + scaleB) + (shift + shiftB)

    def out_modulate(self, hid: mx.array, emb: mx.array, layer: str, hid_lens: List[int]) -> mx.array:
        _, _, gateA = self._emb_for_layer(emb, layer)
        gateB = self.params[f"{layer}_gate"]
        gate = self._repeat(gateA, hid_lens)
        return hid * (gate + gateB)

    @staticmethod
    def _repeat(x: mx.array, hid_lens: List[int]) -> mx.array:
        """Repeat (b, dim) -> (sum(hid_lens), dim) by repeating each batch element."""
        if len(hid_lens) == 1 and x.shape[0] == 1:
            # Common case: single batch, just broadcast via expand
            return mx.broadcast_to(x, (hid_lens[0], x.shape[1]))
        parts = [mx.broadcast_to(x[i:i + 1], (hid_lens[i], x.shape[1])) for i in range(len(hid_lens))]
        return mx.concatenate(parts, axis=0)


class MLXSwiGLUMLP:
    """SwiGLU: out = down(silu(gate(x)) * up(x)). All linears bias=False.

    For long sequences the gate(x) and up(x) intermediates are each
    (L, hidden) ~ 1 GB at 1080p in bf16 and live simultaneously.  We chunk
    along the token axis so peak intermediate = chunk_size/L of full size,
    and call mx.eval per chunk to drop the prior chunk from the lazy graph.
    """

    def __init__(self, gate_w: mx.array, up_w: mx.array, down_w: mx.array,
                 chunk_size: int = 16384):
        self.gate = MLXLinear(gate_w)
        self.up = MLXLinear(up_w)
        self.down = MLXLinear(down_w)
        self.chunk_size = chunk_size

    def __call__(self, x: mx.array) -> mx.array:
        L = x.shape[0]
        if L <= self.chunk_size:
            return self.down(_silu(self.gate(x)) * self.up(x))
        # Token-axis chunking: process self.chunk_size rows at a time.
        out_chunks = []
        for s in range(0, L, self.chunk_size):
            e = min(s + self.chunk_size, L)
            xc = x[s:e]
            yc = self.down(_silu(self.gate(xc)) * self.up(xc))
            mx.eval(yc)   # materialize and drop intermediate (gate, up, silu) refs
            out_chunks.append(yc)
        return mx.concatenate(out_chunks, axis=0)


# ============================================================================
# Branch (vid or txt) container - holds per-branch weights for one block
# ============================================================================

@dataclass
class BranchWeights:
    """
    One branch's worth of per-block parameters.
    For mm_layers (blocks 0-9): two BranchWeights per block (vid and txt)
    For shared blocks (10-31): one BranchWeights, used for both vid and txt
    """
    proj_qkv_w: mx.array     # (3*head_dim*heads, dim)
    proj_out_w: mx.array     # (dim, head_dim*heads)
    proj_out_b: mx.array     # (dim,)
    norm_q_w: mx.array       # (head_dim,)
    norm_k_w: mx.array       # (head_dim,)
    mlp_gate_w: mx.array     # (hidden, dim)
    mlp_up_w: mx.array       # (hidden, dim)
    mlp_down_w: mx.array     # (dim, hidden)
    ada_attn_shift: mx.array # (dim,)
    ada_attn_scale: mx.array
    ada_attn_gate: mx.array
    ada_mlp_shift: mx.array
    ada_mlp_scale: mx.array
    ada_mlp_gate: mx.array


@dataclass
class BlockWeights:
    """Per-block weights. Either separate (vid, txt) or shared (all)."""
    vid: BranchWeights
    txt: Optional[BranchWeights]   # None if shared
    rope_freqs: mx.array            # (21,) - learned freqs for legacy compat (unused; we recompute)
    shared: bool                    # True if vid weights are also used for txt


# ============================================================================
# IO Component weights (top-level, persistent across all blocks)
# ============================================================================

@dataclass
class IOWeights:
    vid_in_w: mx.array   # (vid_dim, in_channels * patch_t * patch_h * patch_w)  -> (2560, 132)
    vid_in_b: mx.array
    vid_out_w: mx.array  # (out_channels * patch, vid_dim) -> (64, 2560)
    vid_out_b: mx.array
    vid_out_norm_w: mx.array
    vid_out_ada_shift: mx.array
    vid_out_ada_scale: mx.array
    txt_in_w: mx.array
    txt_in_b: mx.array
    emb_proj_in_w: mx.array
    emb_proj_in_b: mx.array
    emb_proj_hid_w: mx.array
    emb_proj_hid_b: mx.array
    emb_proj_out_w: mx.array
    emb_proj_out_b: mx.array


# ============================================================================
# Loading helpers
# ============================================================================

def _t_to_mx(t, dtype=mx.bfloat16) -> mx.array:
    """torch.Tensor -> mx.array, preserving values.

    For bf16 torch tensors we go through a uint16 view to avoid an extra fp32
    staging copy (numpy doesn't support bf16 natively).
    """
    import torch
    if isinstance(t, mx.array):
        return t.astype(dtype)
    if hasattr(t, "detach"):
        if t.dtype == torch.bfloat16:
            # Bit-preserving path: bf16 -> uint16 numpy -> bf16 mx.array.
            # Avoids the bf16->fp32 expansion that doubles peak memory.
            t_cpu = t.detach().contiguous().cpu()
            arr_u16 = t_cpu.view(torch.uint16).numpy()
            mx_u16 = mx.array(arr_u16)
            return mx_u16.view(mx.bfloat16).astype(dtype)
        else:
            arr = t.detach().cpu().numpy()
    else:
        arr = np.asarray(t)
    return mx.array(arr).astype(dtype)


def load_io_weights(io_paths: dict, dtype=mx.bfloat16) -> IOWeights:
    """Load IO component weights via native mx.load (no fp32 staging)."""
    def _load(path):
        d = mx.load(path)
        if dtype is not None:
            return {k: v.astype(dtype) for k, v in d.items()}
        return d

    vid_in = _load(io_paths["vid_in"])
    vid_out = _load(io_paths["vid_out"])
    vid_out_norm = _load(io_paths["vid_out_norm"])
    vid_out_ada = _load(io_paths["vid_out_ada"])
    txt_in = _load(io_paths["txt_in"])
    emb_in = _load(io_paths["emb_in"])

    return IOWeights(
        vid_in_w=vid_in["proj.weight"], vid_in_b=vid_in["proj.bias"],
        vid_out_w=vid_out["proj.weight"], vid_out_b=vid_out["proj.bias"],
        vid_out_norm_w=vid_out_norm["weight"],
        vid_out_ada_shift=vid_out_ada["out_shift"],
        vid_out_ada_scale=vid_out_ada["out_scale"],
        txt_in_w=txt_in["weight"], txt_in_b=txt_in["bias"],
        emb_proj_in_w=emb_in["proj_in.weight"], emb_proj_in_b=emb_in["proj_in.bias"],
        emb_proj_hid_w=emb_in["proj_hid.weight"], emb_proj_hid_b=emb_in["proj_hid.bias"],
        emb_proj_out_w=emb_in["proj_out.weight"], emb_proj_out_b=emb_in["proj_out.bias"],
    )


def load_block_weights(path: str, shared: bool, dtype=mx.bfloat16) -> BlockWeights:
    """Load one block's weights (.vid/.txt for mm_layers, .all for shared).

    Uses native mx.load - bf16 weights stay as bf16, no fp32 staging copy.
    """
    raw = mx.load(path)
    if dtype is not None:
        raw = {k: v.astype(dtype) for k, v in raw.items()}

    def _branch(prefix: str) -> BranchWeights:
        return BranchWeights(
            proj_qkv_w=raw[f"attn.proj_qkv.{prefix}.weight"],
            proj_out_w=raw[f"attn.proj_out.{prefix}.weight"],
            proj_out_b=raw[f"attn.proj_out.{prefix}.bias"],
            norm_q_w=raw[f"attn.norm_q.{prefix}.weight"],
            norm_k_w=raw[f"attn.norm_k.{prefix}.weight"],
            mlp_gate_w=raw[f"mlp.{prefix}.proj_in_gate.weight"],
            mlp_up_w=raw[f"mlp.{prefix}.proj_in.weight"],
            mlp_down_w=raw[f"mlp.{prefix}.proj_out.weight"],
            ada_attn_shift=raw[f"ada.{prefix}.attn_shift"],
            ada_attn_scale=raw[f"ada.{prefix}.attn_scale"],
            ada_attn_gate=raw[f"ada.{prefix}.attn_gate"],
            ada_mlp_shift=raw[f"ada.{prefix}.mlp_shift"],
            ada_mlp_scale=raw[f"ada.{prefix}.mlp_scale"],
            ada_mlp_gate=raw[f"ada.{prefix}.mlp_gate"],
        )

    if shared:
        vid = _branch("all")
        txt = None
    else:
        vid = _branch("vid")
        txt = _branch("txt")

    return BlockWeights(
        vid=vid, txt=txt, shared=shared,
        rope_freqs=raw["attn.rope.rope.freqs"],
    )


# ============================================================================
# Time embedding (sinusoidal -> 3-layer MLP)
# ============================================================================

def time_embedding(t: mx.array, dim: int = 256) -> mx.array:
    """
    Sinusoidal time embedding matching diffusers' get_timestep_embedding with
    flip_sin_to_cos=False, downscale_freq_shift=0:
        exponent = -log(10000) * arange(0, half_dim) / half_dim
        emb = exp(exponent)
        emb = t[:,None] * emb[None,:]
        emb = cat([sin, cos], -1)
    t: (b,) -> (b, dim)
    """
    half = dim // 2
    exponent = -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
    freqs = mx.exp(exponent)
    args = t[:, None].astype(mx.float32) * freqs[None, :]
    emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1))], axis=-1)
    return emb


def emb_in_forward(t: mx.array, io: IOWeights) -> mx.array:
    """TimeEmbedding: sinusoidal -> proj_in -> silu -> proj_hid -> silu -> proj_out."""
    sin_dim = io.emb_proj_in_w.shape[1]  # 256
    e = time_embedding(t, dim=sin_dim).astype(io.emb_proj_in_w.dtype)
    e = MLXLinear(io.emb_proj_in_w, io.emb_proj_in_b)(e)
    e = _silu(e)
    e = MLXLinear(io.emb_proj_hid_w, io.emb_proj_hid_b)(e)
    e = _silu(e)
    e = MLXLinear(io.emb_proj_out_w, io.emb_proj_out_b)(e)
    return e  # (b, 6*dim) e.g. (1, 15360)


# ============================================================================
# IO components (single-batch; we run B=1)
# ============================================================================

def vid_in_forward(vid: mx.array, vid_shape: Tuple[int, int, int],
                   io: IOWeights, patch: Tuple[int, int, int] = (1, 2, 2)
                   ) -> Tuple[mx.array, Tuple[int, int, int]]:
    """
    Input:  vid (L_in, in_channels=33), vid_shape=(T, H, W) latent shape
    Output: vid (L_out, vid_dim=2560), vid_shape=(T, H/2, W/2)

    NaPatchIn for patch=(1,2,2):
        unflatten (L, C) -> (T, H, W, C)
        rearrange "(T t) (H h) (W w) c -> T H W (t h w c)" with (t,h,w)=(1,2,2)
        flatten (T, H/2, W/2, 4*C)
        proj: Linear(4*C, vid_dim)
    """
    pt, ph, pw = patch
    T, H, W = vid_shape
    C = vid.shape[-1]
    assert pt == 1, "patch_t > 1 requires temporal padding handling"
    assert H % ph == 0 and W % pw == 0, f"H={H} W={W} not divisible by patch ({ph},{pw})"

    x = vid.reshape(T, H, W, C)
    # rearrange (T t) (H h) (W w) c -> T H W (t h w c) with t=1, h=ph, w=pw
    # Equivalently: reshape (T, H/h, h, W/w, w, c), permute to (T, H/h, W/w, h, w, c), reshape
    x = x.reshape(T, H // ph, ph, W // pw, pw, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (T, H', W', h, w, c)
    x = x.reshape(T * (H // ph) * (W // pw), ph * pw * C)

    x = MLXLinear(io.vid_in_w, io.vid_in_b)(x)
    return x, (T, H // ph, W // pw)


def vid_out_forward(vid: mx.array, vid_shape: Tuple[int, int, int],
                    io: IOWeights, patch: Tuple[int, int, int] = (1, 2, 2)
                    ) -> Tuple[mx.array, Tuple[int, int, int]]:
    """Inverse: project then un-patch."""
    pt, ph, pw = patch
    T, H_pat, W_pat = vid_shape
    out = MLXLinear(io.vid_out_w, io.vid_out_b)(vid)  # (L, out_channels*pt*ph*pw)
    out_C = out.shape[-1] // (pt * ph * pw)
    # reshape (L, h*w*c) and unfold
    out = out.reshape(T, H_pat, W_pat, ph, pw, out_C)
    out = out.transpose(0, 1, 3, 2, 4, 5)  # (T, H', h, W', w, c)
    out = out.reshape(T * pt, H_pat * ph, W_pat * pw, out_C)
    out = out.reshape(T * pt * H_pat * ph * W_pat * pw, out_C)
    return out, (T * pt, H_pat * ph, W_pat * pw)


# ============================================================================
# Build vid/txt RoPE freq tables (matches NaMMRotaryEmbedding3d.get_freqs)
# ============================================================================

def build_rope_freqs(vid_shape: Tuple[int, int, int], txt_len: int,
                     head_dim: int = 128, theta: float = 10000.0,
                     dtype=mx.float32) -> Tuple[mx.array, mx.array]:
    """
    Build vid_freqs of shape (T+txt_len_buffer, H_buffer, W_buffer, 126) and
    txt_freqs of shape (txt_len_buffer, 126) matching the per-axis interleaved
    pair pattern of rotary_embedding_torch.

    The vid_freqs include the full 3-axial concatenation (3 * dim_per_axis).
    For head_dim=128, rope_dim=3 -> dim_per_axis=42, so freqs last dim = 126.

    Returns the FULL frequency tables (not yet sliced); callers index per-window.
    """
    T, H, W = vid_shape
    rope_dim = 3
    dim_per_axis = head_dim // rope_dim  # 42 (integer division)

    # Build with same buffers as PyTorch (matches caching policy in get_freqs)
    max_temporal = min(txt_len + T + 16, 1024)
    max_height = min(H + 4, 128)
    max_width = min(W + 4, 128)
    max_txt_len = min(txt_len + 16, 1024)

    fT = _build_axis_freqs(theta, dim_per_axis, max_temporal, dtype)
    fH = _build_axis_freqs(theta, dim_per_axis, max_height, dtype)
    fW = _build_axis_freqs(theta, dim_per_axis, max_width, dtype)

    # broadcast and concat along last dim (matches rotary_embedding_torch.broadcat)
    vT = mx.broadcast_to(fT[:, None, None, :], (max_temporal, max_height, max_width, dim_per_axis))
    vH = mx.broadcast_to(fH[None, :, None, :], (max_temporal, max_height, max_width, dim_per_axis))
    vW = mx.broadcast_to(fW[None, None, :, :], (max_temporal, max_height, max_width, dim_per_axis))
    vid_freqs = mx.concatenate([vT, vH, vW], axis=-1)  # (T_buf, H_buf, W_buf, 126)
    txt_freqs = _build_axis_freqs(theta, dim_per_axis, max_txt_len, dtype)  # (L_buf, 42)
    return vid_freqs, txt_freqs


# ============================================================================
# Window indexing for batch=1 (MLX-friendly)
# ============================================================================

@dataclass
class WindowInfo:
    """Pre-computed window indexing for a single (T, H, W) input.

    All token lengths/indices are MLX arrays for graph-friendly slicing.
    """
    window_shapes: List[Tuple[int, int, int]]    # per-window (T_w, H_w, W_w)
    vid_lens: List[int]                          # per-window token count = T_w*H_w*W_w
    tgt_idx: mx.array                            # (L_vid,) - reorder vid tokens window-major
    src_idx: mx.array                            # (L_vid,) - inverse for window_reverse
    cum_vid_lens: List[int]                      # prefix sums
    n_windows: int


def make_window_info(vid_shape: Tuple[int, int, int], window: Tuple[int, int, int],
                     shifted: bool, fixed_wt: Optional[int] = None) -> WindowInfo:
    """Build per-window slicing for batch=1.

    Mirrors na.window_idx but specialized for B=1, returning MLX index arrays.
    """
    T, H, W = vid_shape
    slices = make_720p_windows(T, H, W, window, shifted=shifted, fixed_wt=fixed_wt)

    # Build idx flat = arange(T*H*W) reshaped to (T,H,W); slice per window, flatten
    flat = np.arange(T * H * W, dtype=np.int32).reshape(T, H, W)
    parts = []
    win_shapes = []
    vid_lens = []
    for (st, sh, sw) in slices:
        seg = flat[st, sh, sw]  # (T_w, H_w, W_w)
        win_shapes.append(seg.shape)
        vid_lens.append(int(seg.size))
        parts.append(seg.flatten())
    tgt = np.concatenate(parts)
    src = np.argsort(tgt)
    cum = [0]
    for v in vid_lens:
        cum.append(cum[-1] + v)
    return WindowInfo(
        window_shapes=win_shapes,
        vid_lens=vid_lens,
        tgt_idx=mx.array(tgt),
        src_idx=mx.array(src),
        cum_vid_lens=cum,
        n_windows=len(slices),
    )


# ============================================================================
# NaSwinAttention (windowed varlen with 3D MM-RoPE) - batch=1 specialization
# ============================================================================

def windowed_swin_attention(
    vid: mx.array,                   # (L_vid, dim)
    txt: mx.array,                   # (L_txt, dim)  L_txt = txt_len
    vid_branch: BranchWeights,
    txt_branch: Optional[BranchWeights],   # None if shared
    vid_shape: Tuple[int, int, int],
    txt_len: int,
    win_info: WindowInfo,
    vid_freqs: mx.array,             # (T_buf, H_buf, W_buf, 126)
    txt_freqs_full: mx.array,        # (L_txt_buf, 42)
    heads: int,
    head_dim: int,
) -> Tuple[mx.array, mx.array]:
    """
    Per-block windowed multi-modal attention.  Returns (vid_out, txt_out) with
    same shapes as inputs.

    Steps mirror NaSwinAttention.forward:
        1. proj_qkv -> split (Q, K, V)
        2. window_partition vid tokens (reorder)
        3. norm_q / norm_k on head_dim
        4. RoPE on per-window vid + repeated text
        5. SDPA per window (vid_window || txt) varlen
        6. unconcat (txt averaged across windows)
        7. window_reverse vid -> proj_out
    """
    txt_b = txt_branch if txt_branch is not None else vid_branch
    vid_dtype = vid.dtype
    txt_dtype = txt.dtype
    scale = 1.0 / math.sqrt(head_dim)
    rope_dim = 126   # 3 axes * (head_dim // 3) * pair = 3 * 42 = 126

    # 1. proj_qkv on full vid (1.13 GB at 1080p) - the only full-vid alloc.
    vid_qkv = MLXLinear(vid_branch.proj_qkv_w)(vid)
    # 2. window-permute (gather along L) and reshape so per-window slice is cheap
    vid_qkv_win = vid_qkv[win_info.tgt_idx].reshape(-1, 3, heads, head_dim)
    mx.eval(vid_qkv_win)   # materialize so original vid_qkv can be freed

    # txt branch — small; do RMSNorm+RoPE once outside loop
    txt_qkv = MLXLinear(txt_b.proj_qkv_w)(txt).reshape(-1, 3, heads, head_dim)
    txt_q = MLXRMSNorm(txt_b.norm_q_w, eps=1e-5)(txt_qkv[:, 0])
    txt_k = MLXRMSNorm(txt_b.norm_k_w, eps=1e-5)(txt_qkv[:, 1])
    txt_v = txt_qkv[:, 2]
    txt_freq_one = mx.tile(txt_freqs_full[:txt_len], (1, 3))   # (txt_len, 126)
    txt_q_h = txt_q.transpose(1, 0, 2).astype(mx.float32)
    txt_k_h = txt_k.transpose(1, 0, 2).astype(mx.float32)
    txt_q_h = _apply_rotary_emb(txt_freq_one, txt_q_h)
    txt_k_h = _apply_rotary_emb(txt_freq_one, txt_k_h)
    txt_q = txt_q_h.transpose(1, 0, 2).astype(txt_dtype)
    txt_k = txt_k_h.transpose(1, 0, 2).astype(txt_dtype)
    mx.eval(txt_q, txt_k, txt_v)

    norm_q_w = vid_branch.norm_q_w
    norm_k_w = vid_branch.norm_k_w

    # 3. Per-window streaming attention.  Each iteration touches only this
    # window's small (W, heads, head_dim) Q/K/V plus the constant txt branch.
    # Q/K norm and 3D RoPE are done on the per-window slice (fp32 RoPE, then
    # cast back) so we never materialize a full-vid Q/K/V or fp32 tensor.
    vid_out_parts = []
    txt_out_sum = None    # (txt_len, heads, head_dim) fp32 running sum

    for i in range(win_info.n_windows):
        s = win_info.cum_vid_lens[i]
        e = win_info.cum_vid_lens[i + 1]
        T_w, H_w, W_w = win_info.window_shapes[i]

        win_qkv = vid_qkv_win[s:e]                    # (W, 3, heads, head_dim)
        vid_q = MLXRMSNorm(norm_q_w, eps=1e-5)(win_qkv[:, 0])
        vid_k = MLXRMSNorm(norm_k_w, eps=1e-5)(win_qkv[:, 1])
        vid_v = win_qkv[:, 2]
        # 3D RoPE on this window only (freqs depend on (T_w,H_w,W_w))
        win_freqs = vid_freqs[txt_len:txt_len + T_w, :H_w, :W_w, :].reshape(-1, rope_dim)
        vid_q_h = vid_q.transpose(1, 0, 2).astype(mx.float32)
        vid_k_h = vid_k.transpose(1, 0, 2).astype(mx.float32)
        vid_q_h = _apply_rotary_emb(win_freqs, vid_q_h)
        vid_k_h = _apply_rotary_emb(win_freqs, vid_k_h)
        vid_q = vid_q_h.transpose(1, 0, 2).astype(vid_dtype)
        vid_k = vid_k_h.transpose(1, 0, 2).astype(vid_dtype)

        # Concat with txt for varlen, run SDPA on (1, heads, seq, head_dim)
        q = mx.concatenate([vid_q, txt_q], axis=0)
        k = mx.concatenate([vid_k, txt_k], axis=0)
        v = mx.concatenate([vid_v, txt_v], axis=0)
        q4 = q.transpose(1, 0, 2)[None, :]
        k4 = k.transpose(1, 0, 2)[None, :]
        v4 = v.transpose(1, 0, 2)[None, :]
        out4 = mx.fast.scaled_dot_product_attention(q4, k4, v4, scale=scale)
        out = out4[0].transpose(1, 0, 2)
        vid_part = out[:e - s]
        txt_part = out[e - s:].astype(mx.float32)
        # Materialize this window's outputs so per-window intermediates drop
        mx.eval(vid_part, txt_part)
        vid_out_parts.append(vid_part)
        if txt_out_sum is None:
            txt_out_sum = txt_part
        else:
            txt_out_sum = txt_out_sum + txt_part

    vid_out_win = mx.concatenate(vid_out_parts, axis=0)
    txt_out = (txt_out_sum / float(win_info.n_windows)).astype(txt_dtype)
    mx.eval(vid_out_win, txt_out)

    vid_out_win = vid_out_win.reshape(-1, heads * head_dim)
    txt_out = txt_out.reshape(-1, heads * head_dim)

    # 4. window_reverse vid (inverse permutation)
    vid_out = vid_out_win[win_info.src_idx]

    # 5. proj_out
    vid_out = MLXLinear(vid_branch.proj_out_w, vid_branch.proj_out_b)(vid_out)
    txt_out = MLXLinear(txt_b.proj_out_w, txt_b.proj_out_b)(txt_out)

    return vid_out, txt_out


# ============================================================================
# Single transformer block - batch=1
# ============================================================================

def transformer_block_forward(
    vid: mx.array, txt: mx.array, emb: mx.array,
    vid_shape: Tuple[int, int, int], txt_len: int,
    block: BlockWeights, win_info: WindowInfo,
    vid_freqs: mx.array, txt_freqs_full: mx.array,
    heads: int, head_dim: int, dim: int,
    norm_eps: float = 1e-5,
    is_last_layer: bool = False,
) -> Tuple[mx.array, mx.array]:
    """One MMSR block, batch=1.

    For shared blocks (block.shared=True), vid and txt branches both use block.vid weights.
    For mm_layers, txt uses block.txt weights.
    """
    vid_b = block.vid
    txt_b = block.vid if block.shared else block.txt
    vid_dim = dim
    L_vid = vid.shape[0]
    L_txt = txt.shape[0]

    # Ada (vid + txt for non-last; vid only for last layer)
    vid_ada = MLXAdaSingle(
        params={"attn_shift": vid_b.ada_attn_shift, "attn_scale": vid_b.ada_attn_scale,
                "attn_gate": vid_b.ada_attn_gate, "mlp_shift": vid_b.ada_mlp_shift,
                "mlp_scale": vid_b.ada_mlp_scale, "mlp_gate": vid_b.ada_mlp_gate},
        dim=dim,
    )
    txt_ada = None if is_last_layer else MLXAdaSingle(
        params={"attn_shift": txt_b.ada_attn_shift, "attn_scale": txt_b.ada_attn_scale,
                "attn_gate": txt_b.ada_attn_gate, "mlp_shift": txt_b.ada_mlp_shift,
                "mlp_scale": txt_b.ada_mlp_scale, "mlp_gate": txt_b.ada_mlp_gate},
        dim=dim,
    )

    # ---- attn block ----
    vid_a = MLXRMSNorm(None, eps=norm_eps)(vid)  # attn_norm (no affine)
    txt_a = MLXRMSNorm(None, eps=norm_eps)(txt)

    vid_a = vid_ada.in_modulate(vid_a, emb, "attn", [L_vid])
    if txt_ada is not None:
        txt_a = txt_ada.in_modulate(txt_a, emb, "attn", [L_txt])

    # Flush norm+modulate so fp32 RMSNorm intermediates are freed before
    # the large QKV projection allocates ~2 GB in the attention function.
    mx.eval(vid_a, txt_a)

    vid_a, txt_a = windowed_swin_attention(
        vid_a, txt_a, vid_b, None if block.shared else block.txt,
        vid_shape, txt_len, win_info, vid_freqs, txt_freqs_full,
        heads, head_dim,
    )

    vid_a = vid_ada.out_modulate(vid_a, emb, "attn", [L_vid])
    if txt_ada is not None:
        txt_a = txt_ada.out_modulate(txt_a, emb, "attn", [L_txt])

    vid = vid + vid_a
    txt = txt + txt_a
    mx.eval(vid, txt)   # free attention intermediates before MLP allocates

    # ---- mlp block ----
    # For is_last_layer: PyTorch MMModule(vid_only=True) returns txt UNCHANGED
    # through mlp_norm, ada(in/out), mlp.  Then the residual `txt_attn + txt_mlp`
    # becomes `txt_attn + txt_attn = 2*txt_attn`.  We replicate that quirk for
    # math identity, even though the txt output of the last block is discarded
    # downstream (only vid is used by vid_out_norm/ada/out).
    vid_m = MLXRMSNorm(None, eps=norm_eps)(vid)
    if is_last_layer:
        txt_m = txt   # mlp_norm with vid_only=True returns txt unchanged
    else:
        txt_m = MLXRMSNorm(None, eps=norm_eps)(txt)

    vid_m = vid_ada.in_modulate(vid_m, emb, "mlp", [L_vid])
    if not is_last_layer:
        txt_m = txt_ada.in_modulate(txt_m, emb, "mlp", [L_txt])

    mx.eval(vid_m, txt_m)  # flush norm+modulate before MLP allocates

    vid_mlp = MLXSwiGLUMLP(vid_b.mlp_gate_w, vid_b.mlp_up_w, vid_b.mlp_down_w)
    vid_m = vid_mlp(vid_m)
    if not is_last_layer:
        txt_mlp = MLXSwiGLUMLP(txt_b.mlp_gate_w, txt_b.mlp_up_w, txt_b.mlp_down_w)
        txt_m = txt_mlp(txt_m)

    vid_m = vid_ada.out_modulate(vid_m, emb, "mlp", [L_vid])
    if not is_last_layer:
        txt_m = txt_ada.out_modulate(txt_m, emb, "mlp", [L_txt])

    vid = vid + vid_m
    txt = txt + txt_m   # PyTorch quirk on last layer: 2*txt_attn

    return vid, txt


# ============================================================================
# Top-level orchestrator
# ============================================================================

@dataclass
class NaDiTConfig:
    vid_dim: int = 2560
    txt_dim: int = 2560
    heads: int = 20
    head_dim: int = 128
    expand_ratio: int = 4
    norm_eps: float = 1e-5
    in_channels: int = 33     # 16 latent + 16 noise + 1 mask
    out_channels: int = 16
    patch: Tuple[int, int, int] = (1, 2, 2)
    num_blocks: int = 32
    mm_layers: int = 10        # blocks [0..mm_layers) have separate vid/txt; rest shared
    window: Tuple[int, int, int] = (4, 3, 3)


def run_full_dit(
    vid: mx.array,                # (L_vid, in_channels=33)
    txt: mx.array,                # (L_txt, txt_in_dim=5120)
    timestep: float,
    vid_shape: Tuple[int, int, int],   # (T, H, W) latent
    txt_len: int,
    io: IOWeights,
    blocks: List[BlockWeights],
    cfg: NaDiTConfig,
    fixed_wt: Optional[int] = None,
) -> mx.array:
    """Forward through the full NaDiT.  Returns vid output with shape (L_orig, out_channels).

    blocks must be a list of length cfg.num_blocks, with blocks[:cfg.mm_layers] having
    .shared=False and blocks[cfg.mm_layers:] having .shared=True.
    """
    # txt projection
    txt_proj = MLXLinear(io.txt_in_w, io.txt_in_b)(txt)

    # vid patch in
    vid_proj, vid_shape_pat = vid_in_forward(vid, vid_shape, io, cfg.patch)

    # emb
    t_arr = mx.array([timestep], dtype=mx.float32)
    emb = emb_in_forward(t_arr, io)  # (1, 6*dim)

    # Pre-compute RoPE freqs (constant across blocks)
    vid_freqs, txt_freqs_full = build_rope_freqs(vid_shape_pat, txt_len, cfg.head_dim)

    # Pre-compute window info for both shifted/non-shifted
    win_even = make_window_info(vid_shape_pat, cfg.window, shifted=False, fixed_wt=fixed_wt)
    win_odd = make_window_info(vid_shape_pat, cfg.window, shifted=True, fixed_wt=fixed_wt)

    vid_h = vid_proj
    txt_h = txt_proj
    log_mem = os.environ.get("MLX_DIT_LOG_MEM", "0") == "1"
    if log_mem:
        am = mx.get_active_memory() / 1024**3
        print(f"[mlx-dit-fwd] before blocks: active={am:.2f}GB", flush=True)
    for i, block in enumerate(blocks):
        win_info = win_even if (i % 2 == 0) else win_odd
        is_last = (i == len(blocks) - 1)
        vid_h, txt_h = transformer_block_forward(
            vid_h, txt_h, emb, vid_shape_pat, txt_len,
            block, win_info, vid_freqs, txt_freqs_full,
            cfg.heads, cfg.head_dim, cfg.vid_dim,
            norm_eps=cfg.norm_eps, is_last_layer=is_last,
        )
        # MLX is lazy; eval here to bound memory
        mx.eval(vid_h, txt_h)
        if log_mem:
            am = mx.get_active_memory() / 1024**3
            pm = mx.get_peak_memory() / 1024**3
            print(f"[mlx-dit-fwd] after block {i:2d}: active={am:.2f}GB peak={pm:.2f}GB", flush=True)

    # vid_out_norm + vid_out_ada + vid_out
    vid_h = MLXRMSNorm(io.vid_out_norm_w, eps=cfg.norm_eps)(vid_h)
    # FixedVidOutAda: emb reshaped (b, dim, 2, 3); use [..., 0, :] (first group)
    # then hid * (scaleA + out_scale) + (shiftA + out_shift). NO gate.
    b = emb.shape[0]
    emb_r = emb.reshape(b, cfg.vid_dim, 2, 3)[:, :, 0, :]  # (b, dim, 3)
    shiftA = emb_r[..., 0]
    scaleA = emb_r[..., 1]
    L_vid = vid_h.shape[0]
    shift = mx.broadcast_to(shiftA, (L_vid, cfg.vid_dim))
    scale = mx.broadcast_to(scaleA, (L_vid, cfg.vid_dim))
    vid_h = vid_h * (scale + io.vid_out_ada_scale) + (shift + io.vid_out_ada_shift)

    vid_out, _ = vid_out_forward(vid_h, vid_shape_pat, io, cfg.patch)
    return vid_out


# ============================================================================
# Production-ready wrapper class
# ============================================================================

class _NoOpStreamManager:
    """No-op stub matching MPSNaDiTStreamManager interface for pipeline compatibility."""
    def prefetch_start(self, num_blocks=2):
        pass
    def enable_timing(self, flag):
        pass
    def cleanup(self):
        pass
    block_pool = None


# ============================================================================
# MLX-native block pool — same lifecycle as NeuroStream's MPSTransformerBlockPool
#   prefetch_block(i)    : start async load of block i
#   wait_for_load(i)     : block until i is in memory
#   get_block(i)         : returns BlockWeights for i
#   release_block(i)     : drops i from cache (returns memory)
#   reset_pool_state()   : reset for new forward
# ============================================================================

class MLXBlockPool:
    """Streaming MLX block pool.

    Loads transformer blocks from safetensors on demand using MLX's native
    ``mx.load``.  A single background ``ThreadPoolExecutor`` worker handles
    prefetches so that block N+1 is already in memory by the time we're done
    computing block N.  All tensors stay in MLX unified memory — no IPC, no
    shared-memory tricks needed (``mx.load`` releases the GIL during disk I/O).

    API mirrors NeuroStream's ``MPSTransformerBlockPool`` so the orchestration
    pattern matches the MPS/PyTorch path.
    """

    def __init__(
        self,
        block_paths: List[str],
        mm_layers: int,
        dtype=mx.bfloat16,
        pool_size: int = 2,
        full_pool: bool = False,
    ):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        self.block_paths = list(block_paths)
        self.num_blocks = len(self.block_paths)
        self.mm_layers = mm_layers
        self.dtype = dtype
        self.full_pool = full_pool
        # When full_pool=True, keep all blocks resident; else use a small ring.
        self.pool_size = self.num_blocks if full_pool else max(2, pool_size)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_loader")
        self.lock = threading.Lock()
        # In-memory cache: block_idx -> BlockWeights
        self.cache: Dict[int, BlockWeights] = {}
        # In-flight load futures: block_idx -> Future
        self.futures: Dict[int, "Future"] = {}
        # Order in which blocks were inserted (for FIFO eviction)
        self.cache_order: List[int] = []
        self.debug = False
        self.load_times: List[float] = []

    # ─── internal loader ───────────────────────────────────────────────
    def _load(self, block_idx: int) -> BlockWeights:
        t0 = time.time()
        path = self.block_paths[block_idx]
        shared = (block_idx >= self.mm_layers)
        blk = load_block_weights(path, shared=shared, dtype=self.dtype)
        # Force the loader buffers to be freed before returning.
        if not shared:
            mx.eval(*[getattr(blk.vid, f) for f in blk.vid.__dataclass_fields__])
            mx.eval(*[getattr(blk.txt, f) for f in blk.txt.__dataclass_fields__])
        else:
            mx.eval(*[getattr(blk.vid, f) for f in blk.vid.__dataclass_fields__])
        mx.eval(blk.rope_freqs)
        if self.debug:
            self.load_times.append(time.time() - t0)
        return blk

    # ─── public API (mirrors MPS pool) ────────────────────────────────
    def prefetch_block(self, block_idx: int) -> None:
        """Start async load of block ``block_idx`` if not already loading/cached."""
        block_idx = block_idx % self.num_blocks
        with self.lock:
            if block_idx in self.cache or block_idx in self.futures:
                return
            self.futures[block_idx] = self.executor.submit(self._load, block_idx)

    def wait_for_load(self, block_idx: int, timeout: Optional[float] = None) -> bool:
        """Block until ``block_idx`` is in memory."""
        block_idx = block_idx % self.num_blocks
        # Snapshot the future under lock, then await it without holding the lock.
        with self.lock:
            if block_idx in self.cache:
                return True
            fut = self.futures.get(block_idx)
        if fut is None:
            # Caller forgot to prefetch — load synchronously
            self.prefetch_block(block_idx)
            with self.lock:
                fut = self.futures.get(block_idx)
        if fut is None:
            return False
        try:
            blk = fut.result(timeout=timeout)
        except Exception:
            with self.lock:
                self.futures.pop(block_idx, None)
            raise
        with self.lock:
            if block_idx not in self.cache:
                self.cache[block_idx] = blk
                self.cache_order.append(block_idx)
            self.futures.pop(block_idx, None)
        return True

    def get_block(self, block_idx: int) -> BlockWeights:
        block_idx = block_idx % self.num_blocks
        if block_idx not in self.cache:
            self.wait_for_load(block_idx)
        return self.cache[block_idx]

    def release_block(self, block_idx: int) -> None:
        """Drop block from cache (full_pool: no-op).  Caller still holds the
        BlockWeights reference until they release it themselves."""
        if self.full_pool:
            return
        block_idx = block_idx % self.num_blocks
        with self.lock:
            self.cache.pop(block_idx, None)
            try:
                self.cache_order.remove(block_idx)
            except ValueError:
                pass
            self.futures.pop(block_idx, None)
        # Note: actual memory return happens when the caller drops their last
        # reference to the BlockWeights and MLX's cache is cleared.

    def reset_pool_state(self) -> None:
        with self.lock:
            self.cache.clear()
            self.cache_order.clear()
            self.futures.clear()

    def cleanup(self) -> None:
        self.reset_pool_state()
        self.executor.shutdown(wait=True)

    def get_stats(self) -> Dict[str, float]:
        if not self.load_times:
            return {}
        return {
            "num_loads": len(self.load_times),
            "avg_load_time": sum(self.load_times) / len(self.load_times),
            "min_load_time": min(self.load_times),
            "max_load_time": max(self.load_times),
            "total_load_time": sum(self.load_times),
        }


# ============================================================================
# MLXNaDiT — supports both full-resident and streaming modes
# ============================================================================

# Map dtype string flag → mlx dtype
def _resolve_dtype(dtype) -> "mx.Dtype":
    if isinstance(dtype, str):
        d = dtype.lower()
        return {"bf16": mx.bfloat16, "bfloat16": mx.bfloat16,
                "fp16": mx.float16, "float16": mx.float16, "half": mx.float16,
                "fp32": mx.float32, "float32": mx.float32}[d]
    return dtype


import torch.nn as torch_nn   # local alias to avoid clashing with mlx.nn


class MLXNaDiT(torch_nn.Module):
    """Drop-in MLX replacement for MPSStreamingNaDiT.

    Two modes:
    - ``full_pool=True`` (default): all 32 blocks resident in memory (~6.3 GB
      bf16, ~6.3 GB fp16, ~12.6 GB fp32).  Fastest forward.
    - ``full_pool=False``: stream blocks from disk, only ~2 blocks resident at
      any time (~0.4 GB bf16/fp16).  Adds ~0–5 % time overhead due to async
      disk I/O overlapping with compute.

    Inherits ``torch.nn.Module`` solely so the diffusion pipeline guards
    (``next(runner.dit.parameters()).device.type``, ``.to(device)``,
    ``.zero_grad()``, etc.) work transparently.  The single registered
    parameter is a 1-element CPU tensor — all real weights live in MLX
    unified memory.

    Args:
        split_dir: directory containing ``block_manifest.json`` + per-block
            safetensors (produced by the model split tool).
        dtype: ``"bf16"`` (default), ``"fp16"`` (matches training precision),
            or ``"fp32"`` (bit-accurate, 2× memory).
        full_pool: True keeps all blocks resident; False streams.
        pool_size: number of blocks resident when streaming (default 2).
        mm_layers: number of multi-modal blocks (separate vid/txt branches).
        cache_limit_gb: MLX metal-buffer cache cap (returned to OS beyond this).
    """

    def __init__(self, split_dir: str, dtype="bf16", num_blocks: Optional[int] = None,
                 mm_layers: int = 10, log: Optional[Callable[[str], None]] = None,
                 full_pool: bool = True, pool_size: int = 2,
                 cache_limit_gb: Optional[float] = None):
        super().__init__()
        import json
        import torch
        # CPU stub parameter so pipeline ``.parameters().device.type`` reads
        # ``cpu`` (not ``meta``) and ``materialize_model`` skips this model.
        self._torch_stub_param = torch.nn.Parameter(torch.zeros(1, device="cpu"),
                                                    requires_grad=False)
        log = log or (lambda s: None)
        # MLX's metal-buffer cache is process-global; the VAE typically wants
        # a generous limit (~4 GB) for its tiled forward.  We deliberately do
        # NOT lower it here — that would starve the VAE's working set on the
        # first encode/decode after DiT init.  Instead we rely on
        # ``mx.clear_cache()`` at the end of every DiT forward to release the
        # cached metal buffers explicitly.  Pass ``cache_limit_gb`` only if
        # you want to override the global cap.
        if cache_limit_gb is not None:
            try:
                mx.set_cache_limit(int(cache_limit_gb * 1024**3))
            except AttributeError:
                try:
                    mx.metal.set_cache_limit(int(cache_limit_gb * 1024**3))
                except Exception:
                    pass

        self.cfg = NaDiTConfig(mm_layers=mm_layers)
        self.dtype = _resolve_dtype(dtype)
        self.full_pool = full_pool
        self.fixed_wt: Optional[int] = None  # set externally to override temporal window size

        with open(os.path.join(split_dir, "block_manifest.json")) as f:
            self.manifest = json.load(f)
        self.num_blocks = num_blocks if num_blocks is not None else len(self.manifest["blocks"])

        # IO components are tiny; always resident.
        log(f"[mlx-dit] loading IO components ({self.dtype})")
        self.io = load_io_weights(self.manifest["io_components"], dtype=self.dtype)
        mx.eval(*[v for v in self.io.__dict__.values() if isinstance(v, mx.array)])
        active0 = mx.get_active_memory() / 1024**3
        log(f"[mlx-dit] IO loaded, active mem: {active0:.2f} GB")

        block_paths = [self.manifest["blocks"][i] for i in range(self.num_blocks)]
        # Block pool — works in both full_pool and streaming modes.
        self.block_pool = MLXBlockPool(
            block_paths=block_paths,
            mm_layers=mm_layers,
            dtype=self.dtype,
            pool_size=pool_size,
            full_pool=full_pool,
        )
        # Stub for pipeline-compat callers that touch .stream_manager.
        self.stream_manager = _MLXStreamManagerShim(self.block_pool)

        if full_pool:
            log(f"[mlx-dit] preloading all {self.num_blocks} blocks (full_pool=True)")
            for i in range(self.num_blocks):
                self.block_pool.prefetch_block(i)
            for i in range(self.num_blocks):
                self.block_pool.wait_for_load(i)
                if (i + 1) % 8 == 0:
                    am = mx.get_active_memory() / 1024**3
                    pm = mx.get_peak_memory() / 1024**3
                    log(f"[mlx-dit]   loaded {i + 1}/{self.num_blocks} blocks "
                        f"(active={am:.2f}GB peak={pm:.2f}GB)")
        else:
            log(f"[mlx-dit] streaming mode (pool_size={pool_size})")

        am = mx.get_active_memory() / 1024**3
        pm = mx.get_peak_memory() / 1024**3
        log(f"[mlx-dit] ready (active={am:.2f}GB peak={pm:.2f}GB)")

    @property
    def blocks(self) -> List[BlockWeights]:
        """Convenience accessor (full_pool only) — returns all 32 blocks."""
        if not self.full_pool:
            raise RuntimeError(".blocks accessor is only valid in full_pool=True mode")
        return [self.block_pool.cache[i] for i in range(self.num_blocks)]

    def forward(self, vid, txt, vid_shape, txt_shape, timestep, downscale=None,
                disable_cache=False, **kwargs):
        """Mirrors MPSStreamingNaDiT.forward interface but uses MLX backend.

        Inputs are PyTorch tensors (any dtype/device); we convert to MLX, run
        the forward (streaming or full), then convert back to a PyTorch tensor
        on the same device/dtype as the input ``vid``.
        """
        import torch
        from src.models.dit_3b.nadit import NaDiTOutput

        target_device = vid.device
        target_dtype = vid.dtype

        # Torch and MLX share the same Metal heap.  Free torch's cached MPS
        # buffers (sampler scratch, autograd graph leftovers, etc.) so MLX has
        # the full heap available for its ~8 GB peak activation working set.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        vid_mx = _t_to_mx(vid, dtype=self.dtype)
        txt_mx = _t_to_mx(txt, dtype=self.dtype)
        T, H, W = (int(vid_shape[0, 0].item()), int(vid_shape[0, 1].item()), int(vid_shape[0, 2].item()))
        txt_len = int(txt_shape[0, 0].item())
        if torch.is_tensor(timestep):
            ts = float(timestep.float().cpu().item()) if timestep.ndim == 0 else float(timestep[0].float().cpu().item())
        else:
            ts = float(timestep)

        out_mx = run_full_dit_streaming(
            vid_mx, txt_mx, ts, (T, H, W), txt_len,
            self.io, self.block_pool, self.cfg, self.num_blocks,
            fixed_wt=self.fixed_wt,
        )
        mx.eval(out_mx)

        del vid_mx, txt_mx
        out_np = np.asarray(out_mx.astype(mx.float32))
        del out_mx

        # NOTE: deliberately NOT calling mx.clear_cache() here.  The VAE
        # encode/decode that runs before/after DiT relies on the MLX metal-
        # buffer cache for fast tile re-allocation (~4 GB working set).
        # Clearing the cache forces the VAE to round-trip every allocation
        # through the OS on the very next call, which we measured doubles
        # tile times (28 s → 11 s/tile decode at 1080p).  MLX's own LRU
        # eviction (capped by the VAE's set_cache_limit) keeps memory
        # bounded automatically.

        out_t = torch.from_numpy(out_np).to(device=target_device, dtype=target_dtype)
        return NaDiTOutput(vid_sample=out_t)

    def forward_tiles(self, tile_vids, tile_vid_shapes, txt, txt_shape, timestep):
        """Block-first multi-tile forward: each block loaded once, all tiles processed.

        Args:
            tile_vids:       list of PyTorch tensors (L_i, C) per spatial tile
            tile_vid_shapes: list of PyTorch tensors [[T, H, W]] per tile
            txt:             PyTorch tensor (L_txt, txt_dim) shared text
            txt_shape:       PyTorch tensor [[txt_len]]
            timestep:        PyTorch tensor or float

        Returns:
            list of NaDiTOutput, one per tile
        """
        import torch
        from src.models.dit_3b.nadit import NaDiTOutput

        target_device = tile_vids[0].device
        target_dtype = tile_vids[0].dtype

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Convert inputs to MLX
        tile_vids_mx = [_t_to_mx(v, dtype=self.dtype) for v in tile_vids]
        tile_shapes = [
            (int(s[0, 0].item()), int(s[0, 1].item()), int(s[0, 2].item()))
            for s in tile_vid_shapes
        ]
        txt_mx = _t_to_mx(txt, dtype=self.dtype)
        txt_len = int(txt_shape[0, 0].item())
        if torch.is_tensor(timestep):
            ts = float(timestep.float().cpu().item()) if timestep.ndim == 0 else float(timestep[0].float().cpu().item())
        else:
            ts = float(timestep)

        out_list_mx = run_full_dit_streaming_tiled(
            tile_vids_mx, tile_shapes, txt_mx, ts, txt_len,
            self.io, self.block_pool, self.cfg, self.num_blocks,
            fixed_wt=self.fixed_wt,
        )

        # Convert outputs back to PyTorch
        results = []
        for out_mx in out_list_mx:
            out_np = np.asarray(out_mx.astype(mx.float32))
            out_t = torch.from_numpy(out_np).to(device=target_device, dtype=target_dtype)
            results.append(NaDiTOutput(vid_sample=out_t))
        del out_list_mx, tile_vids_mx, txt_mx
        return results

    # nn.Module's ``__call__`` would route to ``forward`` already, but our
    # ``forward`` takes a custom signature without **kwargs hooks etc., so we
    # override ``__call__`` to bypass nn.Module's hook machinery.
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # nn.Module's default ``.to(device)`` would walk ``self.parameters()`` and
    # move them; that would move our 1-element CPU stub to MPS, which is fine
    # but pointless.  Override to no-op to keep the stub on CPU and avoid any
    # unnecessary MPS allocation.
    def to(self, *args, **kwargs):
        return self

    @property
    def device(self):
        import torch
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class _MLXStreamManagerShim:
    """Thin shim so callers using the MPS interface (.stream_manager.prefetch_start
    / .stream_manager.cleanup / .stream_manager.block_pool) keep working."""
    def __init__(self, block_pool: MLXBlockPool):
        self.block_pool = block_pool
        self._timing = False
    def prefetch_start(self, num_blocks: int = 2) -> None:
        for i in range(min(num_blocks, self.block_pool.num_blocks)):
            self.block_pool.prefetch_block(i)
    def enable_timing(self, flag: bool) -> None:
        self._timing = flag
        self.block_pool.debug = flag
    def cleanup(self) -> None:
        self.block_pool.cleanup()


def run_full_dit_streaming(
    vid: mx.array, txt: mx.array, timestep: float,
    vid_shape: Tuple[int, int, int], txt_len: int,
    io: IOWeights, block_pool: MLXBlockPool, cfg: NaDiTConfig, num_blocks: int,
    fixed_wt: Optional[int] = None,
) -> mx.array:
    """Same as ``run_full_dit`` but pulls each block from ``block_pool`` and
    overlaps the next-block prefetch with current-block compute.  Works in
    both full_pool and streaming modes (the pool just no-ops the loads when
    full_pool=True)."""
    txt_proj = MLXLinear(io.txt_in_w, io.txt_in_b)(txt)
    vid_proj, vid_shape_pat = vid_in_forward(vid, vid_shape, io, cfg.patch)
    t_arr = mx.array([timestep], dtype=mx.float32)
    emb = emb_in_forward(t_arr, io)
    vid_freqs, txt_freqs_full = build_rope_freqs(vid_shape_pat, txt_len, cfg.head_dim)
    win_even = make_window_info(vid_shape_pat, cfg.window, shifted=False, fixed_wt=fixed_wt)
    win_odd = make_window_info(vid_shape_pat, cfg.window, shifted=True, fixed_wt=fixed_wt)

    vid_h = vid_proj
    txt_h = txt_proj
    log_mem = os.environ.get("MLX_DIT_LOG_MEM", "0") == "1"
    if log_mem:
        am = mx.get_active_memory() / 1024**3
        print(f"[mlx-dit-fwd] before blocks: active={am:.2f}GB", flush=True)

    # Make sure block 0 is being loaded; in full_pool mode it's a no-op.
    block_pool.prefetch_block(0)

    for i in range(num_blocks):
        # Async prefetch of next block while we compute the current one.
        if i < num_blocks - 1:
            block_pool.prefetch_block(i + 1)

        block_pool.wait_for_load(i)
        block = block_pool.get_block(i)

        win_info = win_even if (i % 2 == 0) else win_odd
        is_last = (i == num_blocks - 1)
        vid_h, txt_h = transformer_block_forward(
            vid_h, txt_h, emb, vid_shape_pat, txt_len,
            block, win_info, vid_freqs, txt_freqs_full,
            cfg.heads, cfg.head_dim, cfg.vid_dim,
            norm_eps=cfg.norm_eps, is_last_layer=is_last,
        )
        # Force eval before we drop the block reference so its weights stay
        # alive for the matmuls.
        mx.eval(vid_h, txt_h)

        # Release this block's slot if streaming (pool_size < num_blocks).
        if not block_pool.full_pool:
            block_pool.release_block(i)
            del block   # last reference to the BlockWeights → memory freed
            mx.clear_cache()  # force freed weights out of metal buffer cache

        if log_mem:
            am = mx.get_active_memory() / 1024**3
            pm = mx.get_peak_memory() / 1024**3
            print(f"[mlx-dit-fwd] after block {i:2d}: active={am:.2f}GB peak={pm:.2f}GB",
                  flush=True)

    # Output head (same as run_full_dit)
    vid_h = MLXRMSNorm(io.vid_out_norm_w, eps=cfg.norm_eps)(vid_h)
    b = emb.shape[0]
    emb_r = emb.reshape(b, cfg.vid_dim, 2, 3)[:, :, 0, :]
    shiftA = emb_r[..., 0]
    scaleA = emb_r[..., 1]
    L_vid = vid_h.shape[0]
    shift = mx.broadcast_to(shiftA, (L_vid, cfg.vid_dim))
    scale = mx.broadcast_to(scaleA, (L_vid, cfg.vid_dim))
    vid_h = vid_h * (scale + io.vid_out_ada_scale) + (shift + io.vid_out_ada_shift)
    vid_out, _ = vid_out_forward(vid_h, vid_shape_pat, io, cfg.patch)
    return vid_out


def _output_head(vid_h: mx.array, emb: mx.array, vid_shape_pat: Tuple[int, int, int],
                 io: IOWeights, cfg: NaDiTConfig) -> mx.array:
    """Shared output head: vid_out_norm → vid_out_ada → vid_out (un-patch)."""
    vid_h = MLXRMSNorm(io.vid_out_norm_w, eps=cfg.norm_eps)(vid_h)
    b = emb.shape[0]
    emb_r = emb.reshape(b, cfg.vid_dim, 2, 3)[:, :, 0, :]
    shiftA = emb_r[..., 0]
    scaleA = emb_r[..., 1]
    L_vid = vid_h.shape[0]
    shift = mx.broadcast_to(shiftA, (L_vid, cfg.vid_dim))
    scale = mx.broadcast_to(scaleA, (L_vid, cfg.vid_dim))
    vid_h = vid_h * (scale + io.vid_out_ada_scale) + (shift + io.vid_out_ada_shift)
    vid_out, _ = vid_out_forward(vid_h, vid_shape_pat, io, cfg.patch)
    return vid_out


def run_full_dit_streaming_tiled(
    tile_vids: List[mx.array],
    tile_vid_shapes: List[Tuple[int, int, int]],
    txt: mx.array,
    timestep: float,
    txt_len: int,
    io: IOWeights,
    block_pool: MLXBlockPool,
    cfg: NaDiTConfig,
    num_blocks: int,
    fixed_wt: Optional[int] = None,
) -> List[mx.array]:
    """Block-first multi-tile streaming: load each block once, process all tiles.

    For N tiles and M blocks this does M block loads instead of N*M.
    Each tile's peak activation stays at the single-tile level (~6.5 GB at 1080p fp16).

    Args:
        tile_vids: list of (L_vid_i, in_channels) per tile
        tile_vid_shapes: list of (T, H, W) per tile (latent space, before patch)
        txt: (L_txt, txt_in_dim) shared text embedding
        timestep: diffusion timestep
        txt_len: number of text tokens
        io: IO weights (always resident)
        block_pool: streaming block pool
        cfg: model config
        num_blocks: number of transformer blocks
        fixed_wt: if set, overrides the temporal window size for attention

    Returns:
        list of (L_vid_i, out_channels) per tile
    """
    num_tiles = len(tile_vids)
    log_mem = os.environ.get("MLX_DIT_LOG_MEM", "0") == "1"

    # Shared: text projection + time embedding (tiny, shared across tiles)
    txt_proj = MLXLinear(io.txt_in_w, io.txt_in_b)(txt)
    t_arr = mx.array([timestep], dtype=mx.float32)
    emb = emb_in_forward(t_arr, io)

    # Per-tile: vid_in + RoPE freqs + window info
    tile_states = []
    for i in range(num_tiles):
        vid_proj, vid_shape_pat = vid_in_forward(tile_vids[i], tile_vid_shapes[i], io, cfg.patch)
        vid_freqs, txt_freqs = build_rope_freqs(vid_shape_pat, txt_len, cfg.head_dim)
        win_even = make_window_info(vid_shape_pat, cfg.window, shifted=False, fixed_wt=fixed_wt)
        win_odd = make_window_info(vid_shape_pat, cfg.window, shifted=True, fixed_wt=fixed_wt)
        mx.eval(vid_proj)
        tile_states.append({
            "vid_h": vid_proj,
            "txt_h": txt_proj,  # shared reference, will diverge per-tile
            "vid_shape_pat": vid_shape_pat,
            "vid_freqs": vid_freqs,
            "txt_freqs": txt_freqs,
            "win_even": win_even,
            "win_odd": win_odd,
        })
    # Each tile gets its own txt_h copy (txt diverges per-tile through blocks)
    for ts in tile_states[1:]:
        ts["txt_h"] = txt_proj  # same initial value; will diverge in the loop

    if log_mem:
        am = mx.get_active_memory() / 1024**3
        print(f"[mlx-dit-tiled] {num_tiles} tiles, before blocks: active={am:.2f}GB", flush=True)

    block_pool.prefetch_block(0)

    for block_idx in range(num_blocks):
        if block_idx < num_blocks - 1:
            block_pool.prefetch_block(block_idx + 1)

        block_pool.wait_for_load(block_idx)
        block = block_pool.get_block(block_idx)

        win_info_key = "win_even" if (block_idx % 2 == 0) else "win_odd"
        is_last = (block_idx == num_blocks - 1)

        # Process ALL tiles through this block
        for tile_idx in range(num_tiles):
            ts = tile_states[tile_idx]
            ts["vid_h"], ts["txt_h"] = transformer_block_forward(
                ts["vid_h"], ts["txt_h"], emb, ts["vid_shape_pat"], txt_len,
                block, ts[win_info_key], ts["vid_freqs"], ts["txt_freqs"],
                cfg.heads, cfg.head_dim, cfg.vid_dim,
                norm_eps=cfg.norm_eps, is_last_layer=is_last,
            )
            mx.eval(ts["vid_h"], ts["txt_h"])

        # Release block after all tiles are done with it
        if not block_pool.full_pool:
            block_pool.release_block(block_idx)
            del block
            mx.clear_cache()

        if log_mem:
            am = mx.get_active_memory() / 1024**3
            pm = mx.get_peak_memory() / 1024**3
            print(f"[mlx-dit-tiled] after block {block_idx:2d}: active={am:.2f}GB peak={pm:.2f}GB",
                  flush=True)

    # Output head per tile
    results = []
    for ts in tile_states:
        out = _output_head(ts["vid_h"], emb, ts["vid_shape_pat"], io, cfg)
        mx.eval(out)
        results.append(out)

    return results
