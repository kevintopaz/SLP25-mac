#!/usr/bin/env python3
"""
Video Upscaler — one pipeline for every platform.

Architecture:
    - DiT and VAE: the PyTorch models (model_factory). Runs as is with PyTorch,
      or on another backend through a platform port (see neuroplatform_port.py).
    - VAE: plain-PyTorch tiling with temporal streaming and disk-backed decode
      (video_utils.tiled_vae).
    - Pipeline: temporal chunking (generation_phases_pix/latent).
"""

import os as _os
# PyTorch's allocator without fragmentation between the pipeline's phases (the DiT reserved ~3 GB
# more than it used); must be set before CUDA initializes
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import math
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import time
import gc
import logging
import subprocess
import shlex
import shutil
import tempfile
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, '..'))
_package_name = os.path.basename(script_dir)

# workspace_root on sys.path so relative imports resolve correctly
# when running as a script or when imported as a package.
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

if __name__ == '__main__' and __package__ is None:
    __package__ = _package_name

# Also ensure this script's directory is on sys.path for direct `src` imports
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import torch
import numpy as np
import av
import cv2
import json

from .core.generation_utils import (
    setup_generation_context,
    prepare_runner,
)
from .core.generation_phases_pix import process_video_pix_chunked
from .utils.debug import Debug
from . import model_factory
from . import neuroplatform_port

# Run the models on this machine's backend (MLX on Apple silicon, streamed
# PyTorch on NVIDIA/AMD); a no-op when the platform layer is not installed.
neuroplatform_port.install()

# Large numpy temporaries (4K frame batches) must not request transparent huge pages: on a
# machine whose page cache has fragmented memory each request stalls in kernel compaction.
from .video_utils.transform_memmap import disable_numpy_hugepages
disable_numpy_hugepages()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

DIT_TILE_OVERLAP = 0.1

# Mode-dependent defaults for chunk_size / chunk_overlap
_CHUNK_DEFAULTS = {
    "pixel":  {"chunk_size": 33,  "chunk_overlap": 8},
    "latent": {"chunk_size": 121, "chunk_overlap": 21},
}
HYBRID_FIXED_WT = 9           # temporal attention window size for full-latent DiT
DIT_LATENT_CHUNK = 0          # latent frames per DiT temporal chunk (0 = all at once)
DIT_LATENT_OVERLAP = 0        # overlap in latent frames between DiT chunks
HYBRID_TEMPORAL_ENCODE_SLICE = 17  # pixel frames per streaming encode slice (larger slices: fewer, fuller GPU calls)
HYBRID_TEMPORAL_DECODE_SLICE = 5    # latent frames per streaming decode slice
COLOR_CORRECTION = "wavelet"
USM_WEIGHT = 0.0
USM_RADIUS = 10
USM_THRESHOLD = 10
CFG_SCALE = 1.0
LATENT_NOISE_SCALE = 0.0
SHARED_NOISE = False
SEED = 32
ATTENTION_MODE = "sdpa"

# ====================================================================== #
#  Parameter schema: defines every accepted model_parameter, its type,
#  default value, and (optionally) the set of valid choices.
#  Used by _parse_parameters() to validate & extract in one place.
# ====================================================================== #
PARAMETER_SCHEMA = {
    "resolution": {
        "output_width":  {"type": int, "default": 1920},
        "output_height": {"type": int, "default": 1080},
        "start_frame_idx": {"type": int, "default": 0, "aliases": ["start_idx"]},
        "end_frame_idx":   {"type": int, "default": None, "aliases": ["end_idx"]},
    },
    "pipeline": {
        "chunking_mode":    {"type": str, "default": "latent",
                             "choices": ("pixel", "latent")},
        "chunk_size":       {"type": int, "default": None},
        "chunk_overlap":    {"type": int, "default": None},
        "color_correction": {"type": str, "default": COLOR_CORRECTION},
        "resize_strategy":  {"type": str, "default": "pad",
                             "choices": ("resize", "pad"),
                             "sub_options": {
                                 "pad": {
                                     "pad_mode": {"type": str, "default": "reflect",
                                                  "choices": ("replicate", "reflect",
                                                              "constant", "circular")},
                                 },
                             }},
        "vae_tile_blend":   {"type": str, "default": "cosine",
                             "choices": ("cosine", "linear")},
        "temporal_blend":   {"type": str, "default": "hann",
                             "choices": ("hann", "cosine", "linear")},
        "latent_temporal_blend": {"type": str, "default": "cosine",
                                  "choices": ("cosine", "linear")},
    },
    "encoding": {
        "ffmpeg_encoding":        {"type": str, "default": ""},
        "ffmpeg_preproc_filters": {"type": str, "default": ""},
        "output_video_profile":   {"type": str, "default": "420",
                                   "choices": ("420", "422", "444")},
        "output_bit_depth":       {"type": int, "default": None},
    },
}


def _parse_one(name: str, spec: dict, model_parameters: dict) -> any:
    """Extract, cast, and validate a single parameter."""
    aliases = spec.get("aliases", [])
    raw = model_parameters.get(name)
    if raw is None:
        for alias in aliases:
            raw = model_parameters.get(alias)
            if raw is not None:
                break

    if raw is None:
        value = spec["default"]
    else:
        value = spec["type"](raw) if spec["type"] is not None else raw

    choices = spec.get("choices")
    if choices and value is not None and value not in choices:
        default = spec["default"]
        logger.warning(f"Unknown {name}={value!r} (valid: {choices}), "
                       f"defaulting to {default!r}")
        value = default

    if spec["type"] is str and not value:
        value = spec["default"]

    return value


def _parse_parameters(model_parameters: dict) -> dict:
    """Validate and extract parameters from *model_parameters* using
    :data:`PARAMETER_SCHEMA`.  Returns a flat dict keyed by parameter name."""
    result = {}
    for _group, fields in PARAMETER_SCHEMA.items():
        for name, spec in fields.items():
            value = _parse_one(name, spec, model_parameters)
            result[name] = value

            sub_options = spec.get("sub_options", {}).get(value, {})
            for sub_name, sub_spec in sub_options.items():
                result[sub_name] = _parse_one(sub_name, sub_spec, model_parameters)
    return result


# Total system RAM. Low-RAM machines run the DiT tile by tile.
def _system_ram_gb() -> float:
    sim = os.environ.get("SIMULATE_RAM_GB")
    if sim:
        try:
            return float(sim)
        except ValueError:
            pass
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024**3
    except (ValueError, OSError):
        return 0.0  # conservative: triggers small-tile / small-cache tier


_TOTAL_RAM_GB = _system_ram_gb()

if "DIT_TILE_FIRST" not in os.environ:
    os.environ["DIT_TILE_FIRST"] = "1" if _TOTAL_RAM_GB < 24 else "0"

VAE_ENCODE_TILE_SIZE = (640, 640) 
VAE_ENCODE_TILE_OVERLAP = (80, 80) 
VAE_DECODE_TILE_SIZE = (480, 480) 
VAE_DECODE_TILE_OVERLAP = (60, 60) 


def _vae_max_tile(kind: str):
    """Largest VAE tile edge (px) for per-axis tiling (TiledVAE plan_tile). Default: the
    fixed tile sizes above, so no tile is larger than today's -- tiles are only resized
    to fit the frame (1080p decode: 1.65x -> 1.24x the latent area). The device memory
    of a tile grows with its area and most of it is the backend's own (ggml: ~10 GB at
    640 px encode tiles), so larger tiles are opt-in: VAE_DECODE_MAX_TILE /
    VAE_ENCODE_MAX_TILE (px); 0 keeps the fixed tiles."""
    env = os.environ.get(f"VAE_{kind.upper()}_MAX_TILE")
    if env is not None:
        return int(env) or None
    return VAE_DECODE_TILE_SIZE if kind == "decode" else VAE_ENCODE_TILE_SIZE

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ═══════════════════════════════════════════════════════════════════════════
# Video I/O (adapted from inference_cli_pix_chunked_for_serving.py)
# ═══════════════════════════════════════════════════════════════════════════

def _find_ffmpeg():
    found = shutil.which('ffmpeg')
    if found:
        return found
    for p in ('/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg'):
        if os.path.isfile(p):
            return p
    return 'ffmpeg'


def _find_ffprobe():
    ffmpeg_bin = _find_ffmpeg()
    probe = os.path.join(os.path.dirname(ffmpeg_bin), 'ffprobe')
    if os.path.exists(probe):
        return probe
    found = shutil.which('ffprobe')
    return found or 'ffprobe'


_FFMPEG_ENCODERS_CACHE = None


def _list_ffmpeg_encoders():
    global _FFMPEG_ENCODERS_CACHE
    if _FFMPEG_ENCODERS_CACHE is not None:
        return _FFMPEG_ENCODERS_CACHE
    encoders = set()
    try:
        result = subprocess.run(
            [_find_ffmpeg(), '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith('V'):
                encoders.add(parts[1])
    except Exception:
        pass
    _FFMPEG_ENCODERS_CACHE = encoders
    return encoders


def _select_h264_codec_args():
    """Pick an H.264 encoder supported by the active ffmpeg build.

    h264_videotoolbox is deliberately excluded: it's reported as available
    by ffmpeg but frequently fails with "cannot create compression session"
    on many macOS configurations.
    """
    encoders = _list_ffmpeg_encoders()
    candidates = [
        ('libx264', ['-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p']),
        ('h264_nvenc', ['-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '19', '-pix_fmt', 'yuv420p']),
        ('h264_amf', ['-c:v', 'h264_amf', '-quality', 'quality', '-rc', 'cqp', '-qp_i', '19', '-qp_p', '19', '-pix_fmt', 'yuv420p']),
        ('mpeg4', ['-c:v', 'mpeg4', '-qscale:v', '3', '-pix_fmt', 'yuv420p']),
    ]
    for name, args in candidates:
        if name in encoders:
            logger.info(f"FFMPEGVideoWriter: selected encoder '{name}'")
            return args
    logger.warning("FFMPEGVideoWriter: no preferred encoder found, letting ffmpeg choose")
    return ['-pix_fmt', 'yuv420p']


_CHROMA_PROFILES = {
    '420': 'yuv420p', '422': 'yuv422p', '444': 'yuv444p',
}
_CHROMA_PROFILES_10BIT = {
    '420': 'yuv420p10le', '422': 'yuv422p10le', '444': 'yuv444p10le',
}


class FFMPEGVideoWriter:
    def __init__(self, path, width, height, fps, use_10bit=False,
                 encoding_options="", video_profile="420"):
        ffmpeg_bin = _find_ffmpeg()
        self.use_10bit = use_10bit
        input_pix_fmt = 'rgb48le' if use_10bit else 'rgb24'
        if encoding_options:
            codec_args = shlex.split(encoding_options)
        else:
            profiles = _CHROMA_PROFILES_10BIT if use_10bit else _CHROMA_PROFILES
            out_pix_fmt = profiles.get(video_profile, profiles['420'])
            codec = 'libx265' if use_10bit else 'libx264'
            codec_args = ['-c:v', codec, '-pix_fmt', out_pix_fmt,
                          '-preset', 'medium', '-crf', '15']
        self.proc = subprocess.Popen(
            [ffmpeg_bin, '-y', '-f', 'rawvideo', '-pix_fmt', input_pix_fmt,
             '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
             *codec_args, '-movflags', '+faststart', path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def write(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.use_10bit:
            self.proc.stdin.write(frame_rgb.astype(np.uint16).tobytes())
        else:
            self.proc.stdin.write(frame_rgb.astype(np.uint8).tobytes())
        self.proc.stdin.flush()

    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None

    def release(self):
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.wait()
            self.proc = None


def get_video_sar(video_path):
    try:
        result = subprocess.run(
            [_find_ffprobe(), '-v', 'quiet', '-select_streams', 'v:0',
             '-show_entries', 'stream=sample_aspect_ratio',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=10)
        sar_str = result.stdout.strip()
        if not sar_str or sar_str in ('N/A', '0:1', '0:0'):
            return 1.0
        if ':' in sar_str:
            num, den = sar_str.split(':')
            return float(num) / float(den)
        return max(float(sar_str), 0.0) or 1.0
    except Exception:
        return 1.0


def compute_resolution(input_w, input_h, target_w, target_h):
    scale_w = target_w / input_w
    scale_h = target_h / input_h
    if scale_w > scale_h:
        return round(input_w * scale_h), target_h
    return target_w, round(input_h * scale_w)


def _has_audio_stream(media_path):
    """Return True if *media_path* contains at least one audio stream."""
    try:
        result = subprocess.run(
            [_find_ffprobe(), '-v', 'error', '-select_streams', 'a',
             '-show_entries', 'stream=index', '-of', 'csv=p=0', media_path],
            capture_output=True, text=True, timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def copy_audio_to_video(audio_source, video_target,
                        audio_start=0.0, audio_duration=None):
    """Mux the audio track from *audio_source* into *video_target* (in-place).

    When ``audio_start`` / ``audio_duration`` are provided the audio input is
    trimmed to that range before muxing — required when the video was rendered
    from a restricted frame range so the container duration matches the video.
    """
    if not _has_audio_stream(audio_source):
        logger.info("No audio stream in %s; skipping audio mux", audio_source)
        return

    video_dir = os.path.dirname(video_target) or "."
    file_ext = os.path.splitext(video_target)[1]
    trimming = audio_start > 0.0 or (audio_duration is not None and audio_duration > 0.0)

    fd, temp_output = tempfile.mkstemp(dir=video_dir, suffix=file_ext)
    os.close(fd)
    try:
        command = [_find_ffmpeg(), "-hide_banner", "-loglevel", "warning",
                   "-i", video_target]
        if audio_start > 0.0:
            command += ["-ss", f"{audio_start:.6f}"]
        if audio_duration is not None and audio_duration > 0.0:
            command += ["-t", f"{audio_duration:.6f}"]
        command += ["-i", audio_source, "-c:v", "copy"]
        if trimming:
            command += ["-c:a", "aac", "-b:a", "192k"]
        else:
            command += ["-c:a", "copy"]
        command += ["-map", "0:v:0", "-map", "1:a:0",
                    "-shortest", "-movflags", "+faststart",
                    temp_output, "-y"]
        try:
            subprocess.run(command, check=True, timeout=300)
            shutil.copy(src=temp_output, dst=video_target)
            if trimming:
                logger.info(
                    "Audio copied into %s (start=%.3fs, dur=%s)",
                    video_target, audio_start,
                    f"{audio_duration:.3f}s" if audio_duration else "n/a",
                )
            else:
                logger.info("Audio copied from source into %s", video_target)
        except Exception as e:
            logger.warning("Could not copy audio: %s", e)
    finally:
        try:
            os.unlink(temp_output)
        except OSError:
            pass


def _parse_ffmpeg_filters(filter_string):
    """Parse a comma-separated FFmpeg filter string into ``[(name, args_or_None), ...]``."""
    chain = []
    for f in filter_string.split(','):
        f = f.strip()
        if not f:
            continue
        if '=' in f:
            name, args = f.split('=', 1)
            chain.append((name, args))
        else:
            chain.append((f, None))
    return chain


def _build_filter_graph(video_stream, filter_chain):
    """Build a PyAV filter graph from a parsed filter chain."""
    graph = av.filter.Graph()
    src = graph.add_buffer(template=video_stream)
    sink = graph.add("buffersink")
    prev = src
    for name, option in filter_chain:
        node = graph.add(name, option) if option else graph.add(name)
        prev.link_to(node)
        prev = node
    prev.link_to(sink)
    graph.configure()
    return graph


class LazyVideoReader:
    """Lazy per-chunk video decoder using PyAV with background thread.

    Supports optional ``ffmpeg_preproc_filters`` — a comma-separated PyAV
    filter chain (e.g. ``"scale=1920:1080,hflip"``) applied to every decoded
    frame before it enters the pipeline.
    """

    def __init__(self, video_path, skip_frames=0, load_cap=0,
                 ffmpeg_preproc_filters=""):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.video_path = video_path
        self.skip_frames = skip_frames
        self._container = av.open(video_path, 'r')
        stream = self._container.streams.video[0]
        self.fps = float(stream.average_rate or stream.guessed_rate or 30)
        raw_w = stream.width
        raw_h = stream.height
        total = stream.frames
        if total <= 0 and stream.duration and stream.time_base:
            total = round(float(stream.duration * stream.time_base) * self.fps)
        available = max(0, total - skip_frames)
        if load_cap > 0:
            available = min(available, load_cap)
        self._total_frames = available

        # Detect source bit depth from the pixel format
        pix_fmt = stream.codec_context.pix_fmt or ""
        self.bit_depth = 10 if any(t in pix_fmt for t in ("10le", "10be", "p010", "p016", "16le", "16be")) else 8
        if self.bit_depth > 8:
            self._decode_fmt = 'rgb48le'
            self._decode_scale = np.float32(65535.0)
        else:
            self._decode_fmt = 'rgb24'
            self._decode_scale = np.float32(255.0)
        logger.debug(f"LazyVideoReader: source pix_fmt={pix_fmt!r}, bit_depth={self.bit_depth}, decode_fmt={self._decode_fmt}")

        self._filter_chain = _parse_ffmpeg_filters(ffmpeg_preproc_filters) if ffmpeg_preproc_filters else []
        self._filter_graph = None

        if self._filter_chain:
            self._sar = 1.0
            probe_w, probe_h = self._probe_filtered_dimensions(video_path, stream, self._filter_chain)
            self._display_w = probe_w
            self._display_h = probe_h
            logger.info(
                f"LazyVideoReader: preproc filters active, "
                f"output dims {probe_w}x{probe_h} (raw {raw_w}x{raw_h})"
            )
            self._filter_graph = _build_filter_graph(stream, self._filter_chain)
        else:
            self._sar = get_video_sar(video_path)
            self._display_w = round(raw_w * self._sar) if self._sar != 1.0 and self._sar > 0 else raw_w
            self._display_h = raw_h

        import queue as _q, threading as _t
        self._queue = _q.Queue(maxsize=16)
        self._buffer = {}
        self._decode_pos = 0
        self._exhausted = False
        self._trim_floor = 0
        self._thread = _t.Thread(target=self._worker, daemon=True)
        self._thread.start()
        for _ in range(skip_frames):
            if not self._advance_one():
                break

    @staticmethod
    def _probe_filtered_dimensions(video_path, video_stream, filter_chain):
        """Decode the first frame through a throwaway filter graph to learn output dims."""
        probe_container = av.open(video_path, 'r')
        try:
            probe_stream = probe_container.streams.video[0]
            graph = _build_filter_graph(probe_stream, filter_chain)
            for packet in probe_container.demux(probe_stream):
                for frame in packet.decode():
                    graph.push(frame)
                    while True:
                        try:
                            out = graph.pull()
                            return out.width, out.height
                        except av.error.BlockingIOError:
                            break
        finally:
            probe_container.close()
        return video_stream.width, video_stream.height

    def _worker(self):
        stream = self._container.streams.video[0]
        stream.thread_type = "AUTO"
        try:
            if self._filter_graph is not None:
                for packet in self._container.demux(stream):
                    for frame in packet.decode():
                        self._filter_graph.push(frame)
                        while True:
                            try:
                                self._queue.put(self._filter_graph.pull())
                            except av.error.BlockingIOError:
                                break
            else:
                for packet in self._container.demux(stream):
                    for frame in packet.decode():
                        self._queue.put(frame)
        except Exception:
            pass
        self._queue.put(None)

    def _advance_one(self):
        if self._exhausted:
            return False
        item = self._queue.get()
        if item is None:
            self._exhausted = True
            return False
        self._buffer[self._decode_pos] = item.to_ndarray(format=self._decode_fmt)
        self._decode_pos += 1
        return True

    def _decode_up_to(self, frame_idx):
        while frame_idx not in self._buffer and not self._exhausted:
            self._advance_one()

    @property
    def shape(self):
        return (self._total_frames, self._display_h, self._display_w, 3)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = max(0, min(key.start or 0, self._total_frames))
            stop = max(start, min(key.stop if key.stop is not None else self._total_frames, self._total_frames))
            return self._read_range(start, stop)
        raise TypeError(f"LazyVideoReader only supports slice indexing, got {type(key)}")

    def _read_range(self, start, end):
        """Read decoded frames [start, end), applying SAR correction.

        Returns np.ndarray [N, H, W, C] in raw uint8 or uint16.

        SAR (sample aspect ratio) correction is a container-level property
        that converts coded pixels to display pixels.  This happens *before*
        frames reach VideoPrePostProcess, which handles divisible-crop resize.
        """
        abs_end = self.skip_frames + end - 1
        self._decode_up_to(abs_end)
        abs_start = self.skip_frames + start
        frames = [self._buffer[i] for i in range(abs_start, abs_start + (end - start)) if i in self._buffer]
        floor = self._trim_floor
        for k in list(self._buffer):
            if k < abs_start and (floor <= 0 or k < floor):
                del self._buffer[k]
        if not frames:
            return np.empty((0, self._display_h, self._display_w, 3), dtype=np.uint8)
        if self._sar != 1.0:
            h, w = frames[0].shape[:2]
            new_w = round(w * self._sar)
            if new_w != w:
                frames = [cv2.resize(f, (new_w, h), interpolation=cv2.INTER_CUBIC) for f in frames]
        return np.stack(frames)

    def close(self):
        if self._container is not None:
            if self._thread.is_alive():
                self._thread.join(timeout=2)
            self._container.close()
            self._container = None
            self._buffer.clear()

    def __del__(self):
        self.close()



# ═══════════════════════════════════════════════════════════════════════════
# Upscaler
# ═══════════════════════════════════════════════════════════════════════════

class VideoUpscaler:
    """
    Video upscaler: streaming DiT + tiled, streamed VAE, on any backend.
    """

    def __init__(
        self,
        split_dir,
        model_dir=None,
        vae_path=None,
        device=None,
        debug_enabled=False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = Debug(enabled=debug_enabled)

        if model_dir is None:
            model_dir = os.path.join(script_dir, "models")
        if vae_path is None:
            for _cand in ("ema_vae_bf16.safetensors", "ema_vae_fp16.safetensors"):
                _p = os.path.join(model_dir, _cand)
                if os.path.exists(_p):
                    vae_path = _p
                    break
            else:
                raise FileNotFoundError(f"no VAE safetensors in {model_dir}")

        logger.info("Initializing Video Upscaler...")
        logger.info(f"  Model directory: {model_dir}")
        logger.info(f"  Device: {self.device}")

        # The VAE streams tiles (frames from the pipeline's memmaps, decoded tiles
        # to disk), so full-resolution video stays on the CPU while the models
        # themselves live on the compute device.
        self.ctx = setup_generation_context(
            dit_device=self.device, vae_device=self.device,
            dit_offload_device=None, vae_offload_device=None,
            tensor_offload_device=None, debug=self.debug,
        )
        self.ctx['compute_dtype'] = torch.bfloat16

        # Runner: diffusion config + schedule. Model weights come from the
        # split DiT blocks and the VAE safetensors below.
        dit_name = "nadit_3b"
        vae_name = os.path.basename(vae_path)
        logger.info("Preparing runner...")
        self.runner, cache_ctx = prepare_runner(
            dit_model=dit_name, vae_model=vae_name,
            model_dir=model_dir, debug=self.debug, ctx=self.ctx,
            dit_cache=False, vae_cache=False,
            skip_vae_weights=True,
            encode_tiled=True,
            encode_tile_size=VAE_ENCODE_TILE_SIZE,
            encode_tile_overlap=VAE_ENCODE_TILE_OVERLAP,
            decode_tiled=True,
            decode_tile_size=VAE_DECODE_TILE_SIZE,
            decode_tile_overlap=VAE_DECODE_TILE_OVERLAP,
            attention_mode=ATTENTION_MODE,
        )
        self.ctx['cache_context'] = cache_ctx

        logger.info("Initializing codec...")
        self.tiled_vae = model_factory.create_vae(vae_path, device=self.device)
        self.runner.vae = self.tiled_vae.vae
        self._patch_vae()

        logger.info("Loading enhancement model...")
        self.streaming_dit = model_factory.create_dit(split_dir, device=self.device)
        if hasattr(self.runner, 'dit') and self.runner.dit is not None:
            del self.runner.dit
            gc.collect()
        self.runner.dit = self.streaming_dit
        torch.cuda.empty_cache()
        logger.info("Video Upscaler initialized successfully!")

    def _release_codec(self):
        """Drop the codec's carried causal-conv memory between phases."""
        for m in self.tiled_vae.vae.modules():
            if torch.is_tensor(getattr(m, "memory", None)):
                m.memory = None
        gc.collect()
        torch.cuda.empty_cache()

    def _patch_vae(self):
        """runner.vae_encode / vae_decode through the tiled, streamed VAE."""
        runner = self.runner
        tiled = self.tiled_vae
        debug = self.debug
        scale = runner.config.vae.scaling_factor
        shift = runner.config.vae.get("shifting_factor", 0.0)

        @torch.no_grad()
        def vae_encode(samples):
            """List[CTHW tensor] -> List[THWC latent]."""
            torch.cuda.empty_cache()
            latents = []
            for i, sample in enumerate(samples):
                frames_torch = sample.permute(1, 2, 3, 0).contiguous().cpu()
                if frames_torch.dtype != torch.float16:
                    frames_torch = frames_torch.to(torch.float16)
                samples[i] = None
                del sample
                latent_np = tiled.tiled_encode(
                    frames_torch,
                    tile_size=VAE_ENCODE_TILE_SIZE,
                    tile_overlap=VAE_ENCODE_TILE_OVERLAP,
                    max_tile=_vae_max_tile("encode"),
                    temporal_slice=frames_torch.shape[0],
                    progress_fn=getattr(runner, '_vae_progress', None),
                    blend_mode=self.ctx.get('_vae_tile_blend', 'cosine'),
                )
                del frames_torch
                latent_t = torch.from_numpy(latent_np.astype(np.float32)).squeeze(0)
                latent_t = ((latent_t - shift) * scale).to(dtype=torch.bfloat16)
                latents.append(latent_t)
                del latent_np
                debug.log(f"Encode: intermediate {list(latent_t.shape)}", category="vae")
            self._release_codec()
            return latents

        @torch.no_grad()
        def vae_decode(latents):
            """List[THWC latent] -> List[CTHW tensor]."""
            torch.cuda.empty_cache()
            from .video_utils.tile_assembly import assemble_tiles_from_disk, cleanup_tile_files
            samples = []
            for latent in latents:
                if latent.dim() == 5:
                    latent = latent.squeeze(0)
                lat_np = (latent.float() / scale + shift).cpu().numpy()[np.newaxis]
                del latent
                _, T_lat, lat_H, lat_W, _ = lat_np.shape
                pixel_H, pixel_W = lat_H * 8, lat_W * 8
                _vtb = self.ctx.get('_vae_tile_blend', 'cosine')
                tile_infos = tiled.tiled_decode(
                    lat_np,
                    tile_size=VAE_DECODE_TILE_SIZE,
                    tile_overlap=VAE_DECODE_TILE_OVERLAP,
                    max_tile=_vae_max_tile("decode"),
                    temporal_slice=T_lat,
                    progress_fn=getattr(runner, '_vae_progress', None),
                    tile_dtype=self.ctx.get('_tile_dtype', 'uint8'),
                    blend_mode=_vtb,
                )
                del lat_np
                frames_np = assemble_tiles_from_disk(tile_infos, pixel_H, pixel_W, blend_mode=_vtb)
                cleanup_tile_files(tile_infos)
                sample_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2)   # [C, T, H, W]
                samples.append(sample_t)
                del frames_np, tile_infos
                debug.log(f"Decode: {list(sample_t.shape)}", category="vae")
            self._release_codec()
            return samples

        runner.vae_encode = vae_encode
        runner.vae_decode = vae_decode
        logger.info("Codec encode/decode ready")

    def __call__(self, input_path, output_path, model_parameters=None,
                 progress_func=None):
        if model_parameters is None:
            model_parameters = {}

        params = _parse_parameters(model_parameters)

        output_width      = params['output_width']
        output_height     = params['output_height']
        start_frame_idx   = params['start_frame_idx']
        end_frame_idx     = params['end_frame_idx']

        skip_frames = max(0, start_frame_idx)
        if end_frame_idx is not None and end_frame_idx > skip_frames:
            load_cap = end_frame_idx - skip_frames + 1  # inclusive
        else:
            load_cap = 0

        chunking_mode     = params['chunking_mode']
        chunk_size        = params['chunk_size']
        chunk_overlap     = params['chunk_overlap']
        color_correction  = params['color_correction']
        resize_strategy   = params['resize_strategy']
        pad_mode          = params.get('pad_mode', 'constant')
        vae_tile_blend    = params['vae_tile_blend']
        temporal_blend    = params['temporal_blend']
        latent_temporal_blend = params['latent_temporal_blend']

        ffmpeg_encoding        = params['ffmpeg_encoding']
        ffmpeg_preproc_filters = params['ffmpeg_preproc_filters']
        video_profile          = params['output_video_profile']
        output_bit_depth       = params['output_bit_depth']

        logger.info(f"Input video: {input_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Resolution: {output_width}x{output_height}")
        if skip_frames or load_cap:
            logger.info(f"Frame range: start={skip_frames}, load_cap={load_cap}")
        if ffmpeg_encoding:
            logger.info(f"ffmpeg_encoding: {ffmpeg_encoding}")
        if ffmpeg_preproc_filters:
            logger.info(f"ffmpeg_preproc_filters: {ffmpeg_preproc_filters}")

        if progress_func:
            progress_func(0.0, message="Processing started...")

        reader = LazyVideoReader(
            input_path, skip_frames=skip_frames, load_cap=load_cap,
            ffmpeg_preproc_filters=ffmpeg_preproc_filters,
        )
        self.ctx['source_bit_depth'] = reader.bit_depth
        fps = reader.fps

        if reader.shape[0] < 5:
            logger.error(f"Video too short: {reader.shape[0]} frames (need >= 5)")
            return output_path

        input_h, input_w = reader.shape[1], reader.shape[2]
        resolution_w, resolution_h = compute_resolution(input_w, input_h, output_width, output_height)
        resolution = min(resolution_w, resolution_h)
        logger.debug(f"Input: {input_w}x{input_h}, target: {output_width}x{output_height} -> {resolution_w}x{resolution_h}")


        total_frames = reader.shape[0]

        # Ensure VPP exists before opening the writer.
        # encode_whole_video will reuse this instance via ctx['vpp'].
        if 'vpp' not in self.ctx:
            from .video_utils import VideoPrePostProcess
            self.ctx['vpp'] = VideoPrePostProcess(
                resolution=resolution,
                strategy=resize_strategy,
                pad_mode=pad_mode,
                source_bit_depth=reader.bit_depth,
                output_bit_depth=output_bit_depth,
                color_overrides=self.ctx.get('color_overrides'),
                # video_tools' options for this model's VPP: antialiased on MPS too, frames encoded on a
                # background thread (VIDEO_WRITER_ASYNC=0: on the caller's thread)
                antialias=True,
                async_write=os.environ.get("VIDEO_WRITER_ASYNC", "1") == "1",
            )

        vpp = self.ctx['vpp']
        use_10bit = vpp.output_bit_depth > 8
        self.ctx['_tile_dtype'] = "uint16" if use_10bit else "uint8"

        def _ffmpeg_writer_factory(H, W):
            writer = FFMPEGVideoWriter(
                output_path, W, H, fps,
                use_10bit=use_10bit,
                encoding_options=ffmpeg_encoding,
                video_profile=video_profile,
            )
            def _write_frame(frame):
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return _write_frame, writer.release

        # video_tools' writer (NVENC where the machine has it, else the same codec in software) for the default
        # request; the ffmpeg writer for custom ffmpeg arguments and 4:2:2 / 4:4:4
        from .video_io import writer_factory as _vt_writer_factory
        _vt_factory, _writer_desc = _vt_writer_factory(output_path, fps, use_10bit, video_profile, ffmpeg_encoding)
        logger.info("Video writer: %s", _writer_desc if _vt_factory else f"ffmpeg ({_writer_desc})")
        vpp.open_writer(
            writer_factory=_vt_factory or _ffmpeg_writer_factory,
            total_frames=total_frames,
            # video_tools' writer encodes on its own thread already
            **({"async_write": False} if _vt_factory else {}),
        )

        t_start = time.time()

        # --- Select chunk parameters ---
        _defaults = _CHUNK_DEFAULTS[chunking_mode]
        _chunk_size = chunk_size if chunk_size is not None else _defaults["chunk_size"]
        _overlap = chunk_overlap if chunk_overlap is not None else _defaults["chunk_overlap"]
        _shift_back = True

        # --- Hybrid latent-chunk mode setup ---
        _orig_vae_encode = None
        _orig_vae_decode = None
        if chunking_mode == 'latent':
            logger.debug("Processing mode: chunked (chunk_size=%d)", _chunk_size)
            # Set fixed temporal window on DiT
            dit_model = self.runner.dit.dit_model if hasattr(self.runner.dit, 'dit_model') else self.runner.dit
            dit_model.fixed_wt = HYBRID_FIXED_WT

            # Stream the codec: raw frames from the pipeline's memmap, decoded tiles to disk.
            _orig_vae_encode = self.runner.vae_encode
            _orig_vae_decode = self.runner.vae_decode
            self.runner._hybrid_encode_active = True
            tiled = self.tiled_vae
            scale = self.runner.config.vae.scaling_factor
            shift = self.runner.config.vae.get("shifting_factor", 0.0)

            @torch.no_grad()
            def _hybrid_vae_encode(samples):
                runner = self.runner
                torch.cuda.empty_cache()
                # Read from VPP's memmap (THWC uint8/uint16 raw pixels);
                # tiled_encode normalises on the fly via pixel_scale.
                vpp = self.ctx.get('vpp')
                _input_mm = vpp.get_memmap("input") if vpp is not None else None
                if _input_mm is not None:
                    latent_np = tiled.tiled_encode(
                        _input_mm._mm,
                        tile_size=VAE_ENCODE_TILE_SIZE,
                        tile_overlap=VAE_ENCODE_TILE_OVERLAP,
                        max_tile=_vae_max_tile("encode"),
                        temporal_slice=HYBRID_TEMPORAL_ENCODE_SLICE,
                        input_layout=_input_mm.layout,
                        pixel_scale=_input_mm.scale,
                        progress_fn=getattr(runner, '_vae_progress', None),
                        blend_mode=vae_tile_blend,
                    )
                    del _input_mm
                else:
                    sample = samples[0]
                    samples[0] = None
                    frames_torch = sample.permute(1, 2, 3, 0).contiguous().cpu()
                    del sample
                    if frames_torch.dtype != torch.float16:
                        frames_torch = frames_torch.to(torch.float16)
                    latent_np = tiled.tiled_encode(
                        frames_torch,
                        tile_size=VAE_ENCODE_TILE_SIZE,
                        tile_overlap=VAE_ENCODE_TILE_OVERLAP,
                        max_tile=_vae_max_tile("encode"),
                        temporal_slice=HYBRID_TEMPORAL_ENCODE_SLICE,
                        progress_fn=getattr(runner, '_vae_progress', None),
                        blend_mode=vae_tile_blend,
                    )
                    del frames_torch
                latent_t = torch.from_numpy(latent_np.astype(np.float32)).squeeze(0)
                latent_t = ((latent_t - shift) * scale).to(dtype=torch.bfloat16)
                del latent_np
                self._release_codec()
                return [latent_t]

            @torch.no_grad()
            def _hybrid_vae_decode(latents):
                runner = self.runner
                torch.cuda.empty_cache()
                samples = []
                for latent in latents:
                    if latent.dim() == 5:
                        latent = latent.squeeze(0)  # [T, H, W, C]
                    T_lat, lat_H, lat_W = latent.shape[0], latent.shape[1], latent.shape[2]
                    pixel_H, pixel_W = lat_H * 8, lat_W * 8
                    lat_np = (latent.float() / scale + shift).cpu().numpy()[np.newaxis].astype(np.float32)
                    del latent
                    tile_infos = tiled.tiled_decode(
                        lat_np,
                        tile_size=VAE_DECODE_TILE_SIZE,
                        tile_overlap=VAE_DECODE_TILE_OVERLAP,
                        max_tile=_vae_max_tile("decode"),
                        temporal_slice=HYBRID_TEMPORAL_DECODE_SLICE,
                        vpp=self.ctx.get('vpp'),
                        progress_fn=getattr(runner, '_vae_progress', None),
                        tile_dtype=self.ctx.get('_tile_dtype', 'uint8'),
                        blend_mode=vae_tile_blend,
                    )
                    del lat_np
                    T_pix_full = 1 if T_lat == 1 else (T_lat - 1) * 4 + 1
                    T_pix = min(T_pix_full, self.ctx.get('original_temporal_frames', T_pix_full))
                    self.ctx['_decode_meta'] = {
                        'tile_infos': tile_infos,
                        'pixel_H': pixel_H,
                        'pixel_W': pixel_W,
                        'T_pix': T_pix,
                    }
                    samples.append(torch.empty(3, T_pix, pixel_H, pixel_W, dtype=torch.float16))
                self._release_codec()
                return samples

            self.runner.vae_encode = _hybrid_vae_encode
            self.runner.vae_decode = _hybrid_vae_decode

        self.ctx['_dit_latent_chunk'] = DIT_LATENT_CHUNK
        self.ctx['_dit_latent_overlap'] = DIT_LATENT_OVERLAP
        self.ctx['_vae_tile_blend'] = vae_tile_blend
        self.ctx['_temporal_blend'] = temporal_blend
        self.ctx['_latent_temporal_blend'] = latent_temporal_blend

        def _progress_adapter(frac: float, message: str = ""):
            if progress_func:
                progress_func(frac, message=message)

        try:
            _, timings = process_video_pix_chunked(
                runner=self.runner, ctx=self.ctx, frames=reader,
                debug=self.debug, resolution=resolution,
                max_resolution=0,
                chunk_size=_chunk_size, overlap=_overlap,
                dit_tile_overlap=DIT_TILE_OVERLAP,
                seed=SEED, latent_noise_scale=LATENT_NOISE_SCALE,
                shared_noise=SHARED_NOISE,
                prompt=None, negative_prompt=None,
                cfg_scale=CFG_SCALE,
                color_correction=color_correction,
                usm_weight=USM_WEIGHT, usm_radius=USM_RADIUS,
                usm_threshold=USM_THRESHOLD,
                device=self.device, output_callback=vpp.write_frames,
                shift_back=_shift_back,
                progress_fn=_progress_adapter,
            )
        finally:
            reader.close()
            vpp.close_writer()
            # Restore original VAE patches if hybrid mode was used
            if _orig_vae_encode is not None:
                self.runner.vae_encode = _orig_vae_encode
                self.runner.vae_decode = _orig_vae_decode
                dit_model = self.runner.dit.dit_model if hasattr(self.runner.dit, 'dit_model') else self.runner.dit
                dit_model.fixed_wt = None

        elapsed = time.time() - t_start

        if progress_func:
            progress_func(0.95, message="Copying audio track...")

        # Align the muxed audio with the rendered frame range
        audio_start = (skip_frames / fps) if fps and skip_frames > 0 else 0.0
        frames_written = vpp.frames_written if vpp.frames_written > 0 else (load_cap if load_cap > 0 else 0)
        audio_duration = (frames_written / fps) if (fps and frames_written > 0) else None
        copy_audio_to_video(
            input_path, output_path,
            audio_start=audio_start,
            audio_duration=audio_duration,
        )

        logger.info(f"Saved {vpp.frames_written} frames to {output_path}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Encode: {timings.get('encode', 0):.1f}s, "
                    f"Enhance: {timings.get('dit', 0):.1f}s, "
                    f"Decode: {timings.get('decode', 0):.1f}s, "
                    f"Post: {timings.get('post', 0):.1f}s")

        if progress_func:
            progress_func(0.99, message="Done")
        return output_path

    def cleanup(self):
        """Release all model state so memory is fully reclaimed."""
        vpp = self.ctx.get('vpp') if hasattr(self, 'ctx') else None
        if vpp is not None:
            vpp.cleanup()
            self.ctx.pop('vpp', None)
        if getattr(self, 'tiled_vae', None) is not None:
            self._release_codec()
            self.tiled_vae = None
        if hasattr(self, 'runner') and self.runner is not None:
            if hasattr(self.runner, 'dit') and hasattr(self.runner.dit, 'cleanup'):
                self.runner.dit.cleanup()
            self.runner.dit = None
            if hasattr(self.runner, 'vae'):
                del self.runner.vae
            self.runner = None
        gc.collect()
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_arguments():
    parser = argparse.ArgumentParser(description="SeedVR2 Video Upscaler (streaming, any backend)")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output_width", type=int, default=1920)
    parser.add_argument("--output_height", type=int, default=1080)
    parser.add_argument("--split_dir", type=str, default="./nadit_blocks_3b",
                        help="Directory containing split DiT blocks")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory containing VAE + embeddings (default: ./models)")
    parser.add_argument("--vae_path", type=str, default=None,
                        help="VAE safetensors path (overrides model_dir)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--chunking_mode", default="latent", choices=["pixel", "latent"],
                        help="Chunking strategy: 'latent' (default, hybrid streaming) or 'pixel'")
    return parser.parse_args()


def main():
    args = parse_arguments()
    output = args.output or str(Path(args.input).with_stem(Path(args.input).stem + "_upscaled"))
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    try:
        upscaler = VideoUpscaler(
            split_dir=args.split_dir,
            model_dir=args.model_dir,
            vae_path=args.vae_path,
            debug_enabled=args.debug,
        )

        upscaler(
            input_path=args.input,
            output_path=output,
            model_parameters={
                'output_width': args.output_width,
                'output_height': args.output_height,
                'chunking_mode': args.chunking_mode,
            },
        )
        logger.info("Processing complete!")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'upscaler' in dir():
            upscaler.cleanup()


if __name__ == "__main__":
    main()
