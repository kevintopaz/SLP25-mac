"""This model's video reading and writing: video_tools (the submodule ``third_party/video_tools``, pinned like
neuroplatform), served as this package -- ``from .video_io.video_io import VideoWriter``.

It is always the pinned version, also under a server that installs its own ``video_tools``: the two never share a
module name. ``available()`` says whether it can run here (the submodule is present and PyAV imports).
"""
import importlib
import os

_VT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "third_party", "video_tools")
__path__ = [_VT] if os.path.isfile(os.path.join(_VT, "video_writer.py")) else []

_LAZY_EXPORTS = {
    "CudaP010Frame": "video_writer",
    "VideoReader": "video_reader",
    "VideoWriter": "video_writer",
    "get_audio_data": "video_reader",
}


def available() -> bool:
    """The submodule is checked out and its dependency (PyAV) imports."""
    if not __path__:
        return False
    try:
        importlib.import_module("av")
    except ImportError:
        return False
    return True


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module_name}", __name__), name)


def writer_factory(output_path, fps, use_10bit: bool, video_profile: str = "420", ffmpeg_encoding: str = ""):
    """``(factory, description)`` for VideoPrePostProcess.open_writer: video_tools' writer -- NVENC where this
    machine has it (direct, else FFmpeg's), else the software encoder of the same codec -- for the default
    request: H.264 high (8-bit) or H.265 main10 (10-bit), 4:2:0. ``(None, why)`` where it does not serve the
    request (custom ffmpeg arguments, 4:2:2 / 4:4:4, no PyAV): the caller keeps its ffmpeg writer."""
    if ffmpeg_encoding:
        return None, "custom ffmpeg_encoding"
    if str(video_profile) != "420":
        return None, f"{video_profile} chroma"
    if not available():
        return None, "third_party/video_tools or PyAV missing"
    options = ({"video_codec": "h265", "video_profile": "main10", "video_bit_depth": 10} if use_10bit else
               {"video_codec": "h264", "video_profile": "high", "video_bit_depth": 8})

    def factory(H, W):
        from fractions import Fraction

        import numpy as np

        VideoWriter = getattr(importlib.import_module(".video_writer", __name__), "VideoWriter")
        writer = VideoWriter(output_path, W, H, Fraction(fps).limit_denominator(1000000), options=dict(options))
        opaque = np.full((H, W, 1), 255, np.uint8)

        def write(frame):       # VPP's finalised RGB: uint8 -> RGBA (NVENC's 8-bit input), uint16 RGB as is
            writer.write(frame if use_10bit else np.concatenate([frame, opaque], -1))
        return write, writer.close
    return factory, "video_tools ({video_codec} {video_profile} {video_bit_depth}-bit)".format(**options)
