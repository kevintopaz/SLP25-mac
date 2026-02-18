"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401

try:
    from .src.interfaces import comfy_entrypoint, SeedVR2Extension
except ImportError:
    comfy_entrypoint = None
    SeedVR2Extension = None

__all__ = ["comfy_entrypoint", "SeedVR2Extension"]