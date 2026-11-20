"""This model's video pre/post-processing: video_tools' ``processing_utils``, from the submodule
``third_party/video_tools`` (pinned like neuroplatform), served as this package.

The modules load as ``src.video_utils.<name>``, so the existing imports (``from ..video_utils.transform_memmap
import ...``) work unchanged, and the colour methods' ``..utils.color_fix`` resolves to this repo's
``src/utils/color_fix``. Model-specific modules (``tiled_vae``) stay in this directory. Nothing here changes video_tools: this package's own directory comes first on its
path, then the submodule's ``processing_utils``.
"""
import os

_VT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "third_party", "video_tools", "processing_utils")
if not os.path.isfile(os.path.join(_VT, "video_prepostprocess.py")):
    raise ImportError(f"{_VT} is missing: run `git submodule update --init third_party/video_tools`")
__path__.append(_VT)

from .blend import Blend  # noqa: E402
from .color_matcher import ColorMatcher  # noqa: E402
from .tile_assembly import (  # noqa: E402
    assemble_tiles_from_disk,
    assemble_tiles_from_disk_range,
    cleanup_tile_files,
    make_blend_weight,
    read_tile_as_fp16,
)
from .transform_memmap import TransformMemmap  # noqa: E402
from .usm_sharpener import USMSharpener  # noqa: E402
from .video_prepostprocess import ResizeDivisible, VideoPrePostProcess  # noqa: E402

__all__ = [
    "Blend",
    "ColorMatcher",
    "ResizeDivisible",
    "TransformMemmap",
    "USMSharpener",
    "VideoPrePostProcess",
    "assemble_tiles_from_disk",
    "assemble_tiles_from_disk_range",
    "cleanup_tile_files",
    "make_blend_weight",
    "read_tile_as_fp16",
]
