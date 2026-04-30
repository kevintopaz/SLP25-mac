"""Custom Metal kernels for the MLX VAE.

Currently implements:
  - ``fused_gn_silu``: per-frame GroupNorm(num_groups=32, eps=1e-6) + SiLU,
    in a single kernel pass. Replaces the chain
    (mx.fast.layer_norm + gamma*x + beta + sigmoid + multiply).

Layout: input (BT, H, W, C) row-contiguous, gamma/beta length C.
Reduction is per (BT, group), exactly matching ``causal_norm_wrapper`` in
the reference VAE which collapses temporal into batch before applying GN.

Numerical notes:
  - Reductions accumulate in fp32 (single-pass: sum and sum-of-squares).
  - eps = 1e-6 (matches reference VAE).
  - SiLU computed as a numerically-stable branchless form.

Limitations:
  - Hard-coded to G=32 groups (the only config used in the VAE).
  - Threads per group = 256 (chosen for occupancy on M-series GPUs).
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_GN_EPS = 1e-6


_KERNEL_BODY = r"""
    // ─── Setup ────────────────────────────────────────────────────────────
    uint tx     = thread_position_in_threadgroup.x;
    uint g      = threadgroup_position_in_grid.y;   // group [0, G)
    uint bt     = threadgroup_position_in_grid.z;   // batch-time [0, BT)

    uint H  = x_shape[1];
    uint W  = x_shape[2];
    uint C  = x_shape[3];
    constexpr uint G  = 32;
    constexpr uint TPG = 256;          // threads per group
    uint cg = C / G;
    uint HW = H * W;
    uint per_group = HW * cg;

    // Coalesced layout: groups of `cg` adjacent threads cooperatively
    // process one pixel of `cg` channels (32 bytes for cg=16). The next
    // group of `cg` threads handles the next pixel, etc.
    uint thread_per_pixel = cg;
    uint pixels_per_iter  = TPG / cg;
    uint local_pixel = tx / cg;       // [0, pixels_per_iter)
    uint local_chan  = tx % cg;       // [0, cg)

    // base = address of x[bt, 0, 0, g*cg] in the flat buffer.
    uint base   = (bt * HW * C) + (g * cg);
    uint c_base = g * cg;

    // ─── Pass 1: per-thread partial sums in fp32 ─────────────────────────
    float sum    = 0.0f;
    float sum_sq = 0.0f;
    for (uint p = local_pixel; p < HW; p += pixels_per_iter) {
        uint idx = base + p * C + local_chan;
        float v = (float)x[idx];
        sum    += v;
        sum_sq += v * v;
    }

    // ─── Pass 2: cross-thread reduction ──────────────────────────────────
    // Step A: SIMD-wide reduction (32 lanes). After this, lane 0 of each
    // SIMD has the partial sum for that SIMD.
    sum    = simd_sum(sum);
    sum_sq = simd_sum(sum_sq);

    // Step B: write SIMD partials to threadgroup memory and reduce across
    // the 8 SIMDs that make up our 256-thread block.
    threadgroup float sm[16];   // 8 SIMDs * 2 (sum, sum_sq)
    uint simd_id = tx / 32;
    uint lane_id = tx % 32;
    if (lane_id == 0) {
        sm[simd_id * 2 + 0] = sum;
        sm[simd_id * 2 + 1] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SIMD finalizes the reduction and stashes (mean, rstd) in sm[0..1].
    if (simd_id == 0) {
        float s  = (lane_id < 8) ? sm[lane_id * 2 + 0] : 0.0f;
        float ss = (lane_id < 8) ? sm[lane_id * 2 + 1] : 0.0f;
        s  = simd_sum(s);
        ss = simd_sum(ss);
        if (lane_id == 0) {
            float n    = float(per_group);
            float mean = s / n;
            float var  = ss / n - mean * mean;
            sm[0] = mean;
            sm[1] = rsqrt(var + EPS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = sm[0];
    float rstd = sm[1];

    // ─── Pass 3: re-read, normalize, scale, shift, silu, write ───────────
    for (uint p = local_pixel; p < HW; p += pixels_per_iter) {
        uint idx   = base + p * C + local_chan;
        uint c_idx = c_base + local_chan;

        float v  = (float)x[idx];
        float gm = (float)gamma[c_idx];
        float bv = (float)beta[c_idx];

        float yv = (v - mean) * rstd;
        yv = gm * yv + bv;

        // Numerically stable SiLU: choose branch by sign to avoid overflow
        // of exp() in fp32 for large |yv|.
        float sig;
        if (yv >= 0.0f) {
            sig = 1.0f / (1.0f + exp(-yv));
        } else {
            float e = exp(yv);
            sig = e / (1.0f + e);
        }
        y[idx] = (half)(yv * sig);
    }
"""


@lru_cache(maxsize=1)
def _get_kernel():
    """Build the kernel once; MLX caches compiled binaries internally."""
    header = f"""
        #include <metal_stdlib>
        using namespace metal;
        constexpr constant float EPS = {_GN_EPS}f;
    """
    return mx.fast.metal_kernel(
        name="fused_gn_silu",
        input_names=["x", "gamma", "beta"],
        output_names=["y"],
        source=_KERNEL_BODY,
        header=header,
        ensure_row_contiguous=True,
    )


def fused_gn_silu_5d(x_5d: mx.array, gamma: mx.array, beta: mx.array) -> mx.array:
    """Fused GroupNorm(G=32, eps=1e-6, per-frame) + SiLU for a 5-D tensor.

    Args:
      x_5d: shape (B, T, H, W, C). C must be divisible by 32.
      gamma, beta: 1-D (C,). dtype must match x_5d.

    Returns:
      Same shape and dtype as ``x_5d``.
    """
    B, T, H, W, C = x_5d.shape
    assert C % 32 == 0, f"C={C} must be divisible by 32"
    BT = B * T

    x_flat = x_5d.reshape(BT, H, W, C)

    kernel = _get_kernel()
    (y,) = kernel(
        inputs=[x_flat, gamma, beta],
        output_shapes=[(BT, H, W, C)],
        output_dtypes=[x_5d.dtype],
        # grid = (TPG, G, BT). MLX dispatches grid/threadgroup threadgroups.
        grid=(256, 32, BT),
        threadgroup=(256, 1, 1),
    )
    return y.reshape(B, T, H, W, C)
