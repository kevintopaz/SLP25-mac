"""MLX-native wavelet color correction.

Drop-in replacement for ``src.utils.color_fix.wavelet_reconstruction`` that runs
the multi-scale Gaussian-pyramid decomposition on MLX/Metal instead of torch
CPU. Identical algorithm; same numerical output to ULP precision in fp32.

Why MLX: at 4K with 33 frames the torch CPU wavelet path takes 28-60 s. MLX's
``mx.conv2d`` runs the dilated 3x3 depthwise convs on the Apple GPU in ~5x
less time. The torch <-> MLX bridges are cheap on unified memory.

Public API:
    wavelet_reconstruction_mlx(content_torch, style_torch, debug=None)
        -> torch.Tensor

Both inputs are torch tensors of shape ``[T, C, H, W]`` (or ``[B, C, H, W]``)
in ``[-1, 1]``. Returns a torch tensor with the same shape, dtype, and device
as ``content_torch``.
"""

from typing import Optional

import mlx.core as mx
import numpy as np

# Grab the original torch implementation BEFORE inference_mac_streaming
# monkey-patches color_fix.wavelet_reconstruction with the MLX version.
from .color_fix import wavelet_reconstruction as _torch_wavelet_reconstruction


# 3x3 Gaussian-approx kernel used by the torch wavelet_blur.
# Values must match exactly (color_fix.py:148-152).
_GAUSSIAN_3X3 = np.array(
    [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ],
    dtype=np.float32,
)


def _torch_to_mx_nhwc(t):
    """torch [B, C, H, W] -> mlx [B, H, W, C].

    Uses the bf16-preserving uint16 view path when possible, and a fp32 detour
    for fp16 (numpy can't carry bf16 but can carry fp16 via float16 dtype).
    """
    import torch

    t_cpu = t.detach().contiguous().cpu()
    if t_cpu.dtype == torch.bfloat16:
        # bf16 -> uint16 numpy -> bf16 mx.array -> NHWC view
        arr_u16 = t_cpu.view(torch.uint16).numpy()
        mx_u16 = mx.array(arr_u16)
        nchw = mx_u16.view(mx.bfloat16)
    else:
        # fp16/fp32: numpy supports both
        nchw = mx.array(t_cpu.numpy())
    # NCHW -> NHWC
    return nchw.transpose(0, 2, 3, 1)


def _mx_nhwc_to_torch(mx_arr, ref_torch):
    """mlx [B, H, W, C] -> torch [B, C, H, W] with ref_torch's dtype/device."""
    import torch

    nchw = mx_arr.transpose(0, 3, 1, 2)
    # Bridge through fp32 numpy (lossless for bf16/fp16 ranges in [-1, 1]).
    np_arr = np.asarray(nchw.astype(mx.float32))
    return torch.from_numpy(np_arr).to(dtype=ref_torch.dtype, device=ref_torch.device)


def _wavelet_blur_mlx(image: mx.array, radius: int) -> mx.array:
    """Dilated Gaussian blur. image is NHWC mx.array (fp32)."""
    B, H, W, C = image.shape
    max_safe_radius = max(1, min(H, W) // 8)
    if radius > max_safe_radius:
        radius = max_safe_radius

    kernel_np = np.broadcast_to(_GAUSSIAN_3X3[None, :, :, None], (C, 3, 3, 1))
    kernel = mx.array(kernel_np.copy())  # fp32

    image_padded = mx.pad(
        image, [(0, 0), (radius, radius), (radius, radius), (0, 0)], mode="edge"
    )
    return mx.conv2d(
        image_padded, kernel, stride=1, padding=0, dilation=radius, groups=C
    )


def _wavelet_decomposition_mlx(image: mx.array, levels: int = 5):
    """Returns (high_freq, low_freq) — runs in fp32 for precision."""
    high_freq = mx.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = _wavelet_blur_mlx(image, radius)
        high_freq = high_freq + (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction_mlx(content_feat, style_feat, debug=None):
    """MLX-native wavelet color transfer (drop-in for torch version).

    Args:
        content_feat: torch tensor [B, C, H, W] in [-1, 1] — desired details.
        style_feat:   torch tensor [B, C, H, W] in [-1, 1] — desired colors.
        debug:        unused; kept for signature compatibility.

    Returns:
        torch tensor [B, C, H, W] with content's high-frequency details and
        style's low-frequency colors, clamped to [-1, 1]. Dtype and device
        match ``content_feat``.
    """
    if content_feat.shape != style_feat.shape:
        return _torch_wavelet_reconstruction(content_feat, style_feat, debug)

    # Cast to fp32 for the decomposition to avoid precision loss in the
    # high-frequency residual (image - low_freq) across 5 iterative levels.
    content_mx = _torch_to_mx_nhwc(content_feat).astype(mx.float32)
    style_mx = _torch_to_mx_nhwc(style_feat).astype(mx.float32)

    content_high, _ = _wavelet_decomposition_mlx(content_mx)
    _, style_low = _wavelet_decomposition_mlx(style_mx)

    out_mx = mx.clip(content_high + style_low, -1.0, 1.0)
    mx.eval(out_mx)
    del content_mx, style_mx, content_high, style_low

    return _mx_nhwc_to_torch(out_mx, content_feat)
