"""Temporal and spatial overlap blending for chunked video processing.

Consolidates the blending logic previously duplicated across
generation_utils.py, generation_phases_pix.py, and generation_phases_latent.py.

Supports multiple blending modes:
    - ``"linear"``:  simple linear crossfade
    - ``"hann"``:    Hann-window crossfade (smooth cosine ramp in the middle third)
    - ``"cosine"``:  cosine ramp from 0→1 (used for spatial tile blending)
"""

import math
from typing import Literal, Optional, Tuple

import torch


class Blend:
    """Configurable overlap blending for temporal and spatial tile seams.

    Parameters
    ----------
    mode : str
        Default blending curve.
        ``"linear"`` — uniform linear crossfade.
        ``"hann"``   — Hann-window crossfade; ramps only the middle third
                        of the overlap (keeps first/last third at 1/0).
                        Falls back to linear when overlap < 3.
        ``"cosine"`` — half-cosine ramp (sin/cos quarter-period).
    """

    MODES = frozenset({"linear", "hann", "cosine"})

    def __init__(self, mode: Literal["linear", "hann", "cosine"] = "hann"):
        if mode not in self.MODES:
            raise ValueError(f"Unknown blend mode {mode!r}; choose from {sorted(self.MODES)}")
        self.mode = mode

    # ------------------------------------------------------------------ #
    #  Weight generation                                                   #
    # ------------------------------------------------------------------ #

    def _weights(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Return a 1-D weight tensor of length *n* (1→0 for prev, complement for cur).

        Parameters
        ----------
        n : int
            Number of overlapping frames / pixels.
        device, dtype :
            Tensor placement.
        mode : str or None
            Override the instance default.
        """
        mode = mode or self.mode
        device = torch.device("cpu")    # a handful of values: computed on the CPU whatever the data's device,
                                        # so GPU blending uses exactly the same weights as CPU blending

        if mode == "hann" and n >= 3:
            t = torch.linspace(0.0, 1.0, steps=n, device=device, dtype=dtype)
            blend_start = 1.0 / 3.0
            blend_end   = 2.0 / 3.0
            u = ((t - blend_start) / (blend_end - blend_start)).clamp(0.0, 1.0)
            return 0.5 + 0.5 * torch.cos(math.pi * u)

        if mode == "cosine":
            t = torch.linspace(0.0, math.pi / 2, steps=n, device=device, dtype=dtype)
            return torch.cos(t)  # 1 → 0

        # linear (or hann fallback for n < 3)
        return torch.linspace(1.0, 0.0, steps=n, device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    #  Temporal blending                                                   #
    # ------------------------------------------------------------------ #

    def temporal_thwc(
        self,
        prev_tail: torch.Tensor,
        cur_head: torch.Tensor,
        overlap: int,
        mode: Optional[str] = None,
        frames: Optional[slice] = None,
    ) -> torch.Tensor:
        """Blend overlapping frames in ``[T, H, W, C]`` layout.

        With *frames*, the inputs hold only those frames of the *overlap*-frame
        blend (weights sliced to match), so a long overlap can be blended in batches.

        Parameters
        ----------
        prev_tail : Tensor
            Last *overlap* frames from the previous chunk.
        cur_head : Tensor
            First *overlap* frames from the current chunk.
        overlap : int
            Number of overlapping frames.
        mode : str or None
            Override the instance default blending mode.
        """
        w = self._weights(overlap, prev_tail.device, prev_tail.dtype, mode)
        if frames is not None:
            w = w[frames]
        w = w.to(prev_tail.device).view(-1, 1, 1, 1)
        return prev_tail * w + cur_head * (1.0 - w)

    def temporal_cthw(
        self,
        prev_tail: torch.Tensor,
        cur_head: torch.Tensor,
        overlap: int,
        mode: Optional[str] = None,
        frames: Optional[slice] = None,
    ) -> torch.Tensor:
        """Blend overlapping frames in ``[C, T, H, W]`` layout (*frames*: see ``temporal_thwc``)."""
        w = self._weights(overlap, prev_tail.device, prev_tail.dtype, mode)
        if frames is not None:
            w = w[frames]
        w = w.to(prev_tail.device).view(1, -1, 1, 1)
        return prev_tail * w + cur_head * (1.0 - w)

    # ------------------------------------------------------------------ #
    #  Spatial tile blending                                               #
    # ------------------------------------------------------------------ #

    def tile_weight_mask(
        self,
        tile_h: int,
        tile_w: int,
        y_start: int,
        x_start: int,
        full_h: int,
        full_w: int,
        overlap_h: int,
        overlap_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a smooth 2-D weight mask for a spatial tile.

        Uses cosine ramps at boundaries that overlap with neighbouring tiles.
        Returns shape ``(1, tile_h, tile_w, 1)`` for broadcasting with
        ``[T, H, W, C]`` tensors.
        """
        weight_h = torch.ones(tile_h, device=device, dtype=dtype)
        weight_w = torch.ones(tile_w, device=device, dtype=dtype)

        # vertical ramps
        if overlap_h > 0:
            if y_start > 0:
                ramp_len = min(overlap_h, tile_h)
                t = torch.linspace(0, math.pi / 2, ramp_len, device=device, dtype=dtype)
                weight_h[:ramp_len] = torch.sin(t)
            if y_start + tile_h < full_h:
                ramp_len = min(overlap_h, tile_h)
                t = torch.linspace(0, math.pi / 2, ramp_len, device=device, dtype=dtype)
                weight_h[-ramp_len:] = weight_h[-ramp_len:] * torch.cos(t)

        # horizontal ramps
        if overlap_w > 0:
            if x_start > 0:
                ramp_len = min(overlap_w, tile_w)
                t = torch.linspace(0, math.pi / 2, ramp_len, device=device, dtype=dtype)
                weight_w[:ramp_len] = torch.sin(t)
            if x_start + tile_w < full_w:
                ramp_len = min(overlap_w, tile_w)
                t = torch.linspace(0, math.pi / 2, ramp_len, device=device, dtype=dtype)
                weight_w[-ramp_len:] = weight_w[-ramp_len:] * torch.cos(t)

        weight_2d = weight_h.unsqueeze(1) * weight_w.unsqueeze(0)
        return weight_2d.unsqueeze(0).unsqueeze(-1)  # (1, tile_h, tile_w, 1)

    def blend_tiles(
        self,
        tiles_info: list,
        tile_outputs: list,
        T: int,
        H: int,
        W: int,
        C: int,
        overlap_h: int,
        overlap_w: int,
        device: torch.device,
        dtype: torch.dtype,
        accum_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Weighted-blend a list of tile outputs onto a full-resolution canvas.

        Parameters
        ----------
        tiles_info : list of (y, x, h, w)
            Top-left corner and size of each tile.
        tile_outputs : list of Tensor
            ``[T, H_tile, W_tile, C]`` decoded tiles.
        T, H, W, C : int
            Full canvas dimensions.
        overlap_h, overlap_w : int
            Spatial overlap in pixels.
        device, dtype :
            Tensor placement / precision for the weight masks.
        accum_device : device or None
            Device for the accumulation buffer (defaults to *device*).
        """
        accum_device = accum_device or device
        result = torch.zeros((T, H, W, C), device=accum_device, dtype=dtype)
        weight_sum = torch.zeros((1, H, W, 1), device=accum_device, dtype=dtype)

        for (y, x, th, tw), tile_out in zip(tiles_info, tile_outputs):
            mask = self.tile_weight_mask(
                th, tw, y, x, H, W, overlap_h, overlap_w, tile_out.device, dtype,
            )
            weighted = tile_out * mask
            if accum_device != weighted.device:
                weighted = weighted.to(accum_device)
                mask = mask.to(accum_device)
            result[:, y:y + th, x:x + tw, :] += weighted
            weight_sum[:, y:y + th, x:x + tw, :] += mask
            del weighted, mask

        weight_sum = weight_sum.clamp(min=1e-6)
        result /= weight_sum
        return result
