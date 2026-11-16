"""Unsharp-mask sharpening for video frames.

Consolidates the USM sharpening logic that was previously duplicated
in generation_phases_latent.py and generation_phases_pix.py.
"""

import torch
import torch.nn.functional as F


class USMSharpener:
    """Threshold-aware unsharp-mask sharpening using separable Gaussian blur.

    Parameters
    ----------
    weight : float
        Sharpening strength (0 = disabled, typical 0.3–1.0).
    radius : int
        Gaussian kernel size (forced odd).
    threshold : int
        Pixel-difference threshold (in 0–255 scale) below which
        sharpening is suppressed to avoid amplifying noise.
    """

    def __init__(self, weight: float = 0.0, radius: int = 51, threshold: int = 10):
        self.weight = weight
        self.radius = radius if radius % 2 == 1 else radius + 1
        self.threshold = threshold

    # ------------------------------------------------------------------ #

    def apply(self, video: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Sharpen *video* in-place (writes back to CPU float16).

        Parameters
        ----------
        video : Tensor
            ``[T, H, W, C]`` in ``[0, 1]``, **float16 on CPU**.
        device : str
            GPU device string for the convolution kernels
            (e.g. ``"cuda:0"``).

        Returns
        -------
        Tensor
            Same shape/dtype/device as input.
        """
        if self.weight <= 0:
            return video

        gpu = torch.device(device)
        T, H, W, C = video.shape
        radius = self.radius

        sigma = 0.3 * ((radius - 1) * 0.5 - 1) + 0.8
        ax = torch.arange(radius, device=gpu, dtype=torch.float32) - (radius - 1) / 2.0
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        pad = radius // 2

        kernel_h = gauss_1d.view(1, 1, 1, -1).expand(C, -1, -1, -1).contiguous()
        kernel_v = gauss_1d.view(1, 1, -1, 1).expand(C, -1, -1, -1).contiguous()

        def _blur(x: torch.Tensor) -> torch.Tensor:
            x = F.pad(x, [pad, pad, 0, 0], mode='reflect')
            x = F.conv2d(x, kernel_h, groups=C)
            x = F.pad(x, [0, 0, pad, pad], mode='reflect')
            x = F.conv2d(x, kernel_v, groups=C)
            return x

        chunk_size = max(1, min(8, T))
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            # [chunk, H, W, C] → [chunk, C, H, W] float32 on GPU
            chunk = video[s:e].to(device=gpu, dtype=torch.float32).permute(0, 3, 1, 2)

            blur = _blur(chunk)
            residual = chunk - blur
            del blur

            mask = (residual.abs() * 255.0 > self.threshold).float()
            soft_mask = _blur(mask)
            del mask

            sharp = (chunk + self.weight * residual).clamp_(0, 1)
            out = soft_mask * sharp + (1.0 - soft_mask) * chunk
            del residual, soft_mask, sharp, chunk

            video[s:e] = out.permute(0, 2, 3, 1).float().to(dtype=torch.float16).cpu()
            del out
            torch.cuda.empty_cache()

        del kernel_h, kernel_v, gauss_1d
        return video
