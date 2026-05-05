"""Single-dispatch color correction for video frames.

Consolidates the color correction if/elif dispatch that was previously
duplicated across generation_phases.py, generation_phases_latent.py,
and generation_phases_pix.py into one canonical implementation.

Supports per-method overrides so that platform-specific backends (e.g.
MLX-accelerated wavelet) can be injected at construction time without
monkey-patching the underlying module.
"""

import logging
import torch
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

ColorFn = Callable[[torch.Tensor, torch.Tensor, Optional[object]], torch.Tensor]


class ColorMatcher:
    """Dispatch to one of the supported color correction methods.

    Parameters
    ----------
    method : str
        Color correction method name.  Supported values:
        ``"lab"``, ``"lab_reinhard"``, ``"wavelet"``, ``"wavelet_adaptive"``,
        ``"hsv"``, ``"adain"``, ``"none"``.
    overrides : dict, optional
        Mapping of ``{method_name: callable}`` that replaces the built-in
        implementation for that method.  Each callable must accept
        ``(content, reference, debug)`` and return a tensor with the same
        layout.  This is the preferred way to inject platform-specific
        backends (e.g. an MLX wavelet implementation) instead of
        monkey-patching ``color_fix`` at the module level.

    Example
    -------
    ::

        from src.utils.color_fix_mlx import wavelet_reconstruction_mlx

        matcher = ColorMatcher(
            method="wavelet",
            overrides={"wavelet": wavelet_reconstruction_mlx},
        )
    """

    SUPPORTED = frozenset({
        "lab", "lab_reinhard", "wavelet", "wavelet_adaptive",
        "hsv", "adain", "none",
    })

    def __init__(
        self,
        method: str = "lab",
        overrides: Optional[Dict[str, ColorFn]] = None,
    ):
        self.method = method
        self._overrides: Dict[str, ColorFn] = overrides or {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def apply(
        self,
        content: torch.Tensor,
        reference: torch.Tensor,
        debug: Optional[object] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Apply color correction to *content* using *reference* as style guide.

        Both tensors must be ``[T, C, H, W]`` in ``[-1, 1]`` range.
        Returns the corrected tensor in the same layout and range.
        """
        if self.method == "none":
            return content

        # Check for a user-supplied override first
        override = self._overrides.get(self.method)
        if override is not None:
            return override(content, reference, debug)

        # Lazy import — avoids pulling heavy deps when method is "none"
        from ..utils.color_fix import (
            lab_color_transfer,
            lab_reinhard_transfer,
            wavelet_reconstruction,
            hsv_saturation_histogram_match,
            wavelet_adaptive_color_correction,
            adaptive_instance_normalization,
        )

        if self.method == "lab":
            return lab_color_transfer(content, reference, debug, luminance_weight=0.8)
        if self.method == "lab_reinhard":
            return lab_reinhard_transfer(content, reference, debug)
        if self.method == "wavelet":
            return wavelet_reconstruction(content, reference, debug)
        if self.method == "wavelet_adaptive":
            return wavelet_adaptive_color_correction(content, reference, debug)
        if self.method == "hsv":
            orig_device = content.device
            result = hsv_saturation_histogram_match(
                content.cpu(), reference.cpu(), debug
            )
            return result.to(orig_device)
        if self.method == "adain":
            return adaptive_instance_normalization(content, reference)

        logger.warning("Unknown color correction method: %s", self.method)
        return content
