#!/usr/bin/env python3
"""
Hyp2 HDR Video Upscaler — Mac Apple Silicon with NeuroStream DiT + MLX VAE

HDR variant of the Mac streaming inference pipeline:
    - Reader applies srgb_to_pq so the model operates on PQ-encoded BT.2020 values
    - Color correction is disabled ("none")
    - Pre-upscales to a 1.33x supersampled box, then INTER_AREA-downscales to
      the user-requested resolution before writing
    - Output is HDR10 ProRes (rgb48le -> yuv422p10le, smpte2084) or an EXR
      sequence via HDRVideoWriter

Architecture:
    - DiT: NeuroStream MPS disk-streaming (2-block pool, loads from split safetensors)
    - VAE: MLX conv2d decomposition (tiled encode/decode on Metal)
    - Pipeline: Pixel-space temporal chunking (from generation_phases_pix/latent)
"""

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"

import argparse
import time
import gc
import logging
import subprocess
import shlex
import shutil
import tempfile
from copy import deepcopy
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

# Add NeuroStream to path (and PYTHONPATH so subprocess workers can
# find `streampool` for the DiT block loader).
neurostream_path = os.path.join(workspace_root, 'NeuroStream')
if neurostream_path not in sys.path:
    sys.path.insert(0, neurostream_path)
if neurostream_path not in os.environ.get('PYTHONPATH', ''):
    os.environ['PYTHONPATH'] = neurostream_path + os.pathsep + os.environ.get('PYTHONPATH', '')

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
from .utils.hdr import srgb_to_pq, HDRVideoWriter

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─── MLX wavelet color correction override ────────────────────────────────
# Build a color-override dict that is passed through ctx to every
# ColorMatcher instance via VideoPrePostProcess. This replaces the old
# monkey-patch approach and removes the import-order dependency.
# Disable with MLX_WAVELET=0 to fall back to the torch implementation.
_COLOR_OVERRIDES: dict = {}
if os.environ.get("MLX_WAVELET", "1") == "1":
    from .utils.color_fix_mlx import wavelet_reconstruction_mlx
    _COLOR_OVERRIDES["wavelet"] = wavelet_reconstruction_mlx
    logger.info("Accelerated wavelet color correction enabled")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# ─── Encrypted model blob mapping ──────────────────────────────────────
# Maps original filenames -> UUID blob names inside the encrypted hyp-2
# model folder (produced by encrypt_models.py). Used when loading from the
# encrypted "hyp-2" distribution instead of plaintext nadit_blocks_3b.
_ENCRYPTED_NAME_MAP = {
    "block_0000.safetensors": "blob.799237b5d65e496da7c60d61058f5846",
    "block_0001.safetensors": "blob.cc11ed654666426f95b813a0546593fd",
    "block_0002.safetensors": "blob.75fd43714186433d9a2bf2865595b14c",
    "block_0003.safetensors": "blob.462b6b0e688e49b2adaf2ff93fa2f048",
    "block_0004.safetensors": "blob.4c911459928a4b50a6fdbb849f4fbe2d",
    "block_0005.safetensors": "blob.05dba6fd63c64324b47ada5de423f1f7",
    "block_0006.safetensors": "blob.735b52ac1dfa4b83a4844d64c8a393fa",
    "block_0007.safetensors": "blob.109ea444433a4abfac5c53b912da6e4d",
    "block_0008.safetensors": "blob.d7556319d6104799bfe3853feb1064e0",
    "block_0009.safetensors": "blob.190397a34210498b877ea6047c0b8e2d",
    "block_0010.safetensors": "blob.87eedfc133ee4b6e8732e63739c25a9b",
    "block_0011.safetensors": "blob.e25aa0f98f66434c80a88d7b338d6443",
    "block_0012.safetensors": "blob.363034a147d04efe8decb28c0ad4cc66",
    "block_0013.safetensors": "blob.dbccf1c4838d4cf8acb166f0a44b4dc0",
    "block_0014.safetensors": "blob.ad4673f926674a298308f1bd5565dad4",
    "block_0015.safetensors": "blob.21153ac77afb4feabac89f5cb3d2a6e4",
    "block_0016.safetensors": "blob.ba20a24a3d8e42fc9c1ada6afcc3fe79",
    "block_0017.safetensors": "blob.5030789d3dbf4b739cf254ad0d183223",
    "block_0018.safetensors": "blob.c2e5d771e8cd4a5d87270acfee10d390",
    "block_0019.safetensors": "blob.5f2f3bec54f04d749b93b6b135d96764",
    "block_0020.safetensors": "blob.bcc1387b29b34b3f9a8622db703e46a9",
    "block_0021.safetensors": "blob.245c13d7bdf84bb8a1b7f47f337e80d2",
    "block_0022.safetensors": "blob.e8dbdcb934da4954b5442640041be00f",
    "block_0023.safetensors": "blob.14ba9d2c00ea4c7483bffb9937ae38fa",
    "block_0024.safetensors": "blob.5abce9f90420458f98fd139b6b1a50e0",
    "block_0025.safetensors": "blob.8066a5abb65f4226be8a349da504808a",
    "block_0026.safetensors": "blob.8f111d9cc45a4cfc8a4d6318865d2ed4",
    "block_0027.safetensors": "blob.3291ce04cbb0436480ab522f4642215a",
    "block_0028.safetensors": "blob.b57173b565e54f15be3f0adf309375c5",
    "block_0029.safetensors": "blob.92c6a3ffb4f241e1b60a81701c89f26b",
    "block_0030.safetensors": "blob.bad85269bbca428cb70168b61daeaff0",
    "block_0031.safetensors": "blob.89af214bcb424b0fb16535c09bdeb6e1",
    "block_manifest.json": "blob.651c209ca7d842a29b302618903790da",
    "emb_in.safetensors": "blob.996abaa4c56543b7a7de4cd7109918b2",
    "txt_in.safetensors": "blob.2cceeeec93404c94804b324d61f66118",
    "vid_in.safetensors": "blob.ab0c4052fd2640ed92580fe31107e808",
    "vid_out.safetensors": "blob.d6ee0757798d4bbf8d21afaeb611fe64",
    "vid_out_ada.safetensors": "blob.fc0f0414b4204d7aa73ef90b650019ef",
    "vid_out_norm.safetensors": "blob.98bdc684193841b9bca82742dcf3b7b4",
}

DIT_TILE_OVERLAP = 0.1

# Mode-dependent defaults for chunk_size / chunk_overlap
_CHUNK_DEFAULTS = {
    "pixel":  {"chunk_size": 33,  "chunk_overlap": 8},
    "latent": {"chunk_size": 121, "chunk_overlap": 21},
}
HYBRID_FIXED_WT = 9           # temporal attention window size for full-latent DiT
DIT_LATENT_CHUNK = 0          # latent frames per DiT temporal chunk (0 = all at once)
DIT_LATENT_OVERLAP = 0        # overlap in latent frames between DiT chunks
HYBRID_TEMPORAL_ENCODE_SLICE = 5   # pixel frames per streaming encode slice
HYBRID_TEMPORAL_DECODE_SLICE = 5    # latent frames per streaming decode slice
COLOR_CORRECTION = "none"
USM_WEIGHT = 0.0
USM_RADIUS = 10
USM_THRESHOLD = 10
CFG_SCALE = 1.0
LATENT_NOISE_SCALE = 0.0
SHARED_NOISE = False
SEED = 32
ATTENTION_MODE = "sdpa"

# Supersample each axis before the model,
# then INTER_AREA-downscale back to the user-requested resolution.
SUPERSAMPLE_FACTOR = 4.0 / 3.0
SUPERSAMPLE_MAX_PIXELS = 3840 * 2160


def compute_supersample_resolution(final_w: int, final_h: int):
    """scale each dim, cap so total pixels <= 3840*3840, snap to multiples of 16."""
    if SUPERSAMPLE_FACTOR <= 1.0:
        return final_w, final_h
    pre_w = final_w * SUPERSAMPLE_FACTOR
    pre_h = final_h * SUPERSAMPLE_FACTOR
    if pre_w * pre_h > SUPERSAMPLE_MAX_PIXELS:
        s = (SUPERSAMPLE_MAX_PIXELS / (pre_w * pre_h)) ** 0.5
        pre_w *= s
        pre_h *= s
    pre_w = max(16, int(round(pre_w / 16.0)) * 16)
    pre_h = max(16, int(round(pre_h / 16.0)) * 16)
    return pre_w, pre_h

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
        "output_codec":           {"type": str, "default": "prores",
                                   "choices": ("prores", "exr")},
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


# Total unified RAM (Apple Silicon shares CPU/GPU memory). Used to auto-tier
# the MLX cache cap and VAE tile sizes — both are RAM-sensitive.
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

def _auto_mlx_cache_gb() -> float:
    if _TOTAL_RAM_GB <= 16:
        return 2.0
    if _TOTAL_RAM_GB <= 24:
        return 4.0
    return 6.0


MLX_VAE_CACHE_LIMIT_GB = _auto_mlx_cache_gb()
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ═══════════════════════════════════════════════════════════════════════════
# Video I/O (adapted from inference_cli_pix_chunked_for_serving.py)
# ═══════════════════════════════════════════════════════════════════════════

def _find_ffmpeg():
    found = shutil.which('ffmpeg')
    return found or 'ffmpeg'


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
        command += ["-c:a", "aac", "-b:a", "192k"]
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

        # HDR pipeline operates on PQ-encoded values regardless of source pix_fmt:
        # always decode to rgb24 [0,1] and run srgb_to_pq before storing.
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
        rgb = item.to_ndarray(format=self._decode_fmt)
        pq = srgb_to_pq(rgb.astype(np.float32) / self._decode_scale)
        pq = (pq * self._decode_scale).clip(0, self._decode_scale).astype(rgb.dtype)
        self._buffer[self._decode_pos] = pq
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
    Video upscaler combining NeuroStream DiT streaming + MLX VAE.
    """

    def __init__(
        self,
        split_dir,
        model_dir=None,
        vae_path=None,
        device="mps",
        debug_enabled=False,
        mlx_dtype="bf16",
        mlx_stream=True,
        vae_cache_limit_gb=None,
        encryption_key=None,
    ):
        self.device = device

        # Configure encryption — crypto.mx_load / json_load will decrypt
        # transparently when a key is set.
        from .utils.crypto import configure as _configure_crypto
        _configure_crypto(encryption_key=encryption_key)

        self.debug = Debug(enabled=debug_enabled)
        self.mlx_dtype = mlx_dtype
        self.mlx_stream = mlx_stream
        if vae_cache_limit_gb is not None and float(vae_cache_limit_gb) > 0:
            self.vae_cache_limit_gb = float(vae_cache_limit_gb)
        elif vae_cache_limit_gb is not None and float(vae_cache_limit_gb) < 0:
            self.vae_cache_limit_gb = None  # MLXVideoVAE: skip set_cache_limit
        else:
            self.vae_cache_limit_gb = float(MLX_VAE_CACHE_LIMIT_GB)

        if model_dir is None:
            model_dir = os.path.join(script_dir, "models")

        # Resolve VAE path. Prefer bf16 safetensors (native MLX/DiT dtype, no
        # fp16↔bf16 boundary cast). Falls back to fp16 if bf16 isn't present.
        # The legacy .pth file isn't usable directly — convert it once via
        # convert_vae_pth_to_safetensors.py.
        if vae_path is None:
            for _cand in ("ema_vae_bf16.safetensors", "ema_vae_fp16.safetensors"):
                _p = os.path.join(model_dir, _cand)
                if os.path.exists(_p):
                    vae_path = _p
                    break
            else:
                vae_path = os.path.join(model_dir, "ema_vae_bf16.pth")

        logger.info("Initializing Video Upscaler...")
        logger.info(f"  Model directory: {model_dir}")
        logger.info(
            f"  System RAM: {_TOTAL_RAM_GB:.1f} GB  "
            f"-> enc tile {VAE_ENCODE_TILE_SIZE}, dec tile {VAE_DECODE_TILE_SIZE}"
        )

        # Setup generation context
        # MLX VAE runs on Metal via MLX (not torch MPS), so keep vae_device="cpu"
        # so the pipeline does not move full-resolution video to MPS.
        # dit_device follows constructor `device` (mps on Mac): embeddings,
        # diffusion config, and sampler tensors live on MPS so torch.mps.empty_cache()
        # can reclaim the MPS allocator pool between phases (dit_device=cpu made
        # those calls no-ops). Tradeoff: ~1–3 GB more MPS pool vs cpu dit.
        # compute_dtype=bf16 to match DiT's native bf16.
        dit_dev = device
        self.ctx = setup_generation_context(
            dit_device=dit_dev, vae_device="cpu",
            dit_offload_device=None, vae_offload_device=None,
            tensor_offload_device=None, debug=self.debug,
        )
        self.ctx['compute_dtype'] = torch.bfloat16
        if _COLOR_OVERRIDES:
            self.ctx['color_overrides'] = _COLOR_OVERRIDES

        # Prepare runner — sets up diffusion config + schedule.
        # dit_name is only used to select the config YAML (3B vs 7B) and as a
        # label; the actual DiT weights come from the split blocks via MLXNaDiT.
        dit_name = "nadit_3b"
        vae_name = os.path.basename(vae_path)
        logger.info("Preparing runner...")
        self.runner, cache_ctx = prepare_runner(
            dit_model=dit_name, vae_model=vae_name,
            model_dir=model_dir, debug=self.debug, ctx=self.ctx,
            dit_cache=False, vae_cache=False,
            skip_vae_weights=True,
            encode_tiled=True,
            encode_tile_size=(
                VAE_ENCODE_TILE_SIZE if isinstance(VAE_ENCODE_TILE_SIZE, tuple)
                else (VAE_ENCODE_TILE_SIZE, VAE_ENCODE_TILE_SIZE)
            ),
            encode_tile_overlap=(
                (VAE_ENCODE_TILE_OVERLAP, VAE_ENCODE_TILE_OVERLAP)
                if isinstance(VAE_ENCODE_TILE_OVERLAP, int) else VAE_ENCODE_TILE_OVERLAP
            ),
            decode_tiled=True,
            decode_tile_size=(
                VAE_DECODE_TILE_SIZE if isinstance(VAE_DECODE_TILE_SIZE, tuple)
                else (VAE_DECODE_TILE_SIZE, VAE_DECODE_TILE_SIZE)
            ),
            decode_tile_overlap=(
                (VAE_DECODE_TILE_OVERLAP, VAE_DECODE_TILE_OVERLAP)
                if isinstance(VAE_DECODE_TILE_OVERLAP, int) else VAE_DECODE_TILE_OVERLAP
            ),
            attention_mode=ATTENTION_MODE,
        )
        self.ctx['cache_context'] = cache_ctx

        # Text embeddings will be loaded by the generation_phases_latent module
        # from script_directory/models/pos_emb.pt and neg_emb.pt
        # We leave ctx['text_embeds'] unset so it gets loaded on first DiT run

        # Initialize MLX VAE (replaces PyTorch VAE for much faster encode/decode)
        logger.info("Initializing codec...")
        from .optimization.mlx_video_vae import MLXVideoVAE
        self.mlx_vae = MLXVideoVAE(vae_path, cache_limit_gb=self.vae_cache_limit_gb)

        # ─── PyTorch VAE: skip materialization entirely ─────────────────
        # MLX VAE handles all encode/decode.  We monkey-patch
        # runner.vae_encode / runner.vae_decode below; the only thing the
        # downstream pipeline reads from runner.vae is parameters().device
        # (just a meta-device guard), which a tiny CPU stub satisfies.
        # Skipping materialize_model("vae") avoids a ~478 MB MPS allocation.
        self._patch_vae_with_mlx()
        class _VAEStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._p = torch.nn.Parameter(torch.zeros(1, device="cpu"))
        self.runner.vae = _VAEStub()
        logger.debug("Skipped redundant codec materialization")

        # ─── Resolve model paths & create DiT ─────────────────────────
        from .optimization.mlx_nadit import MLXNaDiT
        from .utils.crypto import json_load as _crypto_json_load
        logger.info("Loading enhancement model...")

        nmap = _ENCRYPTED_NAME_MAP if encryption_key else None

        def _resolve(filename):
            blob = nmap.get(filename, filename) if nmap else filename
            return os.path.join(split_dir, blob)

        manifest = _crypto_json_load(_resolve("block_manifest.json"))

        if nmap:
            io_paths = {
                comp: _resolve(os.path.basename(p))
                for comp, p in manifest["io_components"].items()
            }
            block_paths = [
                _resolve(os.path.basename(manifest["blocks"][i]))
                for i in range(len(manifest["blocks"]))
            ]
        else:
            io_paths = manifest["io_components"]
            block_paths = list(manifest["blocks"])

        self.streaming_dit = MLXNaDiT(
            io_paths=io_paths,
            block_paths=block_paths,
            dtype=self.mlx_dtype,
            full_pool=not self.mlx_stream,
            log=lambda s: logger.info(s),
        )

        if hasattr(self.runner, 'dit') and self.runner.dit is not None:
            del self.runner.dit
            gc.collect()
        self.runner.dit = self.streaming_dit

        self.runner._mlx_dit_dtype = self.mlx_dtype
        self.runner._mlx_vae_dtype = str(self.mlx_vae.dtype).replace("mlx.core.", "")

        if device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        logger.info("Video Upscaler initialized successfully!")

    def _patch_vae_with_mlx(self):
        """Replace runner.vae_encode/vae_decode with MLX VAE for 5-10x speedup."""
        runner = self.runner
        mlx_vae = self.mlx_vae
        debug = self.debug

        scale = runner.config.vae.scaling_factor
        shift = runner.config.vae.get("shifting_factor", 0.0)

        original_encode = runner.vae_encode
        original_decode = runner.vae_decode

        @torch.no_grad()
        def mlx_vae_encode(samples):
            """MLX VAE encode: List[CTHW tensor] -> List[THWC latent]."""
            import mlx.core as mx
            # Temporarily free DiT IO weights during encoding to reduce peak
            # unified memory (~0.58 GB). They'll be reloaded on first DiT call.
            _dit = getattr(runner, 'dit', None)
            _dit_io_freed = False
            if _dit is not None and hasattr(_dit, 'io'):
                _dit._saved_io = _dit.io
                _dit.io = None
                gc.collect()
                mx.clear_cache()
                _dit_io_freed = True
                logger.debug("Freed model weights for encode phase")

            torch.mps.empty_cache()

            latents = []
            for i, sample in enumerate(samples):
                # sample: fp16 CTHW CPU (no full-frame bf16 cast any more).
                # Permute to THWC; tiled_encode converts each spatial tile to
                # NumPy/MLX individually — no full-frame NumPy copy.
                frames_torch = sample.permute(1, 2, 3, 0).contiguous().cpu()
                if frames_torch.dtype != torch.float16:
                    frames_torch = frames_torch.to(torch.float16)
                samples[i] = None
                del sample

                _vtb = self.ctx.get('_vae_tile_blend', 'cosine')
                latent_np = mlx_vae.tiled_encode(
                    frames_torch,
                    tile_size=VAE_ENCODE_TILE_SIZE,
                    tile_overlap=VAE_ENCODE_TILE_OVERLAP,
                    temporal_slice=frames_torch.shape[0],
                    progress_fn=getattr(runner, '_vae_progress', None),
                    blend_mode=_vtb,
                )
                del frames_torch
                # latent_np: [1, T_lat, H_lat, W_lat, 32] (mean+logvar)
                # Take mean (first 16 channels), apply scaling
                mean = latent_np[:, :, :, :, :16]  # [1, T_lat, H_lat, W_lat, 16]
                latent_t = torch.from_numpy(mean.astype(np.float32)).squeeze(0)  # [T_lat, H_lat, W_lat, 16]
                latent_t = (latent_t - shift) * scale
                latent_t = latent_t.to(dtype=torch.bfloat16)
                latents.append(latent_t)
                del latent_np, mean
                debug.log(f"Encode: intermediate {list(latent_t.shape)}", category="vae")

            # Full cleanup: free all encode-phase memory before DiT starts
            mlx_vae.encoder = None
            gc.collect()
            mx.clear_cache()
            torch.mps.empty_cache()
            logger.debug("Freed encoder weights after encode")

            # Restore DiT IO weights
            if _dit_io_freed and _dit is not None:
                _dit.io = _dit._saved_io
                del _dit._saved_io
                mx.eval(*[v for v in _dit.io.__dict__.values()
                          if isinstance(v, mx.array)])
                logger.debug("Restored model weights after encode")
            return latents

        @torch.no_grad()
        def mlx_vae_decode(latents):
            """MLX VAE decode: List[THWC latent] -> List[CTHW tensor]."""
            import mlx.core as mx
            # Free DiT IO weights and flush stale DiT activation buffers from
            # the MLX Metal cache before decode.  With streaming DiT the block
            # weights are tiny (~0.4 GB), but the Metal cache can hold several
            # GB of stale activation buffers from the DiT forward pass.
            _dit = getattr(runner, 'dit', None)
            _dit_io_freed = False
            if _dit is not None and hasattr(_dit, 'io'):
                _dit._saved_io = _dit.io
                _dit.io = None
                _dit_io_freed = True
            gc.collect()
            mx.clear_cache()
            torch.mps.empty_cache()

            samples = []
            for latent in latents:
                if latent.dim() == 5:
                    latent = latent.squeeze(0)  # [T, H, W, C]

                lat_unscaled = latent.float() / scale + shift  # [T, H, W, 16]
                lat_np = lat_unscaled.cpu().numpy()[np.newaxis]  # [1, T, H, W, 16]
                del lat_unscaled, latent

                _, T_lat, lat_H, lat_W, _ = lat_np.shape
                pixel_H, pixel_W = lat_H * 8, lat_W * 8
                _vtb = self.ctx.get('_vae_tile_blend', 'cosine')
                _tdtype = self.ctx.get('_tile_dtype', 'uint8')
                tile_infos = mlx_vae.tiled_decode(
                    lat_np,
                    tile_size=VAE_DECODE_TILE_SIZE,
                    tile_overlap=VAE_DECODE_TILE_OVERLAP,
                    temporal_slice=999,
                    progress_fn=getattr(runner, '_vae_progress', None),
                    tile_dtype=_tdtype,
                    blend_mode=_vtb,
                )
                del lat_np
                from .video_utils.tile_assembly import (
                    assemble_tiles_from_disk, cleanup_tile_files,
                )
                frames_np = assemble_tiles_from_disk(
                    tile_infos, pixel_H, pixel_W,
                    blend_mode=_vtb,
                )
                cleanup_tile_files(tile_infos)
                sample_t = torch.from_numpy(frames_np)
                sample_t = sample_t.permute(3, 0, 1, 2)  # [C, T, H, W]
                samples.append(sample_t)
                del frames_np, tile_infos
                debug.log(f"Decode: {list(sample_t.shape)}", category="vae")

            mlx_vae.decoder = None
            gc.collect()
            mx.clear_cache()
            torch.mps.empty_cache()

            if _dit_io_freed and _dit is not None:
                _dit.io = _dit._saved_io
                del _dit._saved_io
                if hasattr(_dit.io, '__dict__'):
                    mx.eval(*[v for v in _dit.io.__dict__.values()
                              if isinstance(v, mx.array)])
            return samples

        runner.vae_encode = mlx_vae_encode
        runner.vae_decode = mlx_vae_decode
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
        output_codec           = params.get('output_codec', 'prores')

        logger.info(f"Input video: {input_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Resolution: {output_width}x{output_height}")
        logger.info(f"color_correction={color_correction} output_codec={output_codec}")
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
        pre_w, pre_h = compute_supersample_resolution(resolution_w, resolution_h)
        resolution = min(pre_w, pre_h)
        logger.debug(
            f"Input: {input_w}x{input_h}, target box: {output_width}x{output_height} "
            f"({pre_w}x{pre_h}) -> final resolution: {resolution_w}x{resolution_h}"
        )

        self.streaming_dit.stream_manager.prefetch_start(num_blocks=2)

        total_frames = reader.shape[0]

        # HDR pipeline: force 16-bit precision out of VPP so finalised
        # frames have headroom before HDRVideoWriter applies its highlight
        # curve and dither.
        if 'vpp' not in self.ctx:
            from .video_utils import VideoPrePostProcess
            self.ctx['vpp'] = VideoPrePostProcess(
                resolution=resolution,
                strategy=resize_strategy,
                pad_mode=pad_mode,
                source_bit_depth=reader.bit_depth,
                output_bit_depth=16,
                color_overrides=self.ctx.get('color_overrides'),
            )

        vpp = self.ctx['vpp']
        use_10bit = vpp.output_bit_depth > 8
        self.ctx['_tile_dtype'] = "uint16" if use_10bit else "uint8"

        # Streaming HDR output: build HDRVideoWriter lazily on the first
        # batch (so its dimensions match the post-supersample-downscale
        # resolution).
        writer_holder = [None]
        saved_frames = [0]

        def _hdr_output_callback(frames_np):
            """Receive post-processed frames and stream to HDRVideoWriter.

            ``frames_np`` is ``[T, H, W, C]`` uint8 or uint16 from VPP's
            ``color_correct_and_finalize``. Convert back to float16 [0,1],
            downscale from the supersampled resolution to the user-requested
            ``(resolution_h, resolution_w)`` (INTER_AREA-equivalent via
            antialiased bilinear), then hand off to ``HDRVideoWriter``.
            """
            if frames_np.dtype == np.uint16:
                frames_tensor = torch.from_numpy(
                    frames_np.astype(np.float32) / 65535.0
                )
            else:
                frames_tensor = torch.from_numpy(
                    frames_np.astype(np.float32) / 255.0
                )
            frames_tensor = frames_tensor.contiguous()

            if frames_tensor.shape[1] != resolution_h or frames_tensor.shape[2] != resolution_w:
                x = frames_tensor.permute(0, 3, 1, 2).float()
                x = torch.nn.functional.interpolate(
                    x, size=(resolution_h, resolution_w),
                    mode='bilinear', antialias=True,
                )
                frames_tensor = x.permute(0, 2, 3, 1).contiguous()

            if writer_holder[0] is None:
                _, H, W, _ = frames_tensor.shape
                writer_holder[0] = HDRVideoWriter(
                    output_path, W, H, fps,
                    options=deepcopy(model_parameters),
                )

            writer_holder[0].write(frames_tensor)
            saved_frames[0] += frames_tensor.shape[0]

            if progress_func and total_frames > 0:
                frac = saved_frames[0] / total_frames
                progress_func(
                    frac,
                    message=f"Processed {saved_frames[0]}/{total_frames} frames...",
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

            # Re-patch VAE encode to use streaming
            _orig_vae_encode = self.runner.vae_encode
            _orig_vae_decode = self.runner.vae_decode
            self.runner._hybrid_encode_active = True
            mlx_vae = self.mlx_vae
            scale = self.runner.config.vae.scaling_factor
            shift = self.runner.config.vae.get("shifting_factor", 0.0)

            @torch.no_grad()
            def _hybrid_vae_encode(samples):
                import mlx.core as _mx
                runner = self.runner

                _dit = getattr(runner, 'dit', None)
                _dit_io_freed = False
                if _dit is not None and hasattr(_dit, 'io') and _dit.io is not None:
                    _dit._saved_io = _dit.io
                    _dit.io = None
                    _bp = getattr(_dit, 'block_pool', None)
                    if _bp is not None:
                        _bp.reset_pool_state()
                    gc.collect()
                    _mx.clear_cache()
                    _dit_io_freed = True

                torch.mps.empty_cache()

                # Read from VPP's memmap (THWC uint8/uint16 raw pixels);
                # tiled_encode normalises on-the-fly via pixel_scale.
                vpp = self.ctx.get('vpp')
                _input_mm = vpp.get_memmap("input") if vpp is not None else None
                if _input_mm is not None:
                    latent_np = mlx_vae.tiled_encode(
                        _input_mm._mm,
                        tile_size=VAE_ENCODE_TILE_SIZE,
                        tile_overlap=VAE_ENCODE_TILE_OVERLAP,
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
                    latent_np = mlx_vae.tiled_encode(
                        frames_torch,
                        tile_size=VAE_ENCODE_TILE_SIZE,
                        tile_overlap=VAE_ENCODE_TILE_OVERLAP,
                        temporal_slice=HYBRID_TEMPORAL_ENCODE_SLICE,
                        progress_fn=getattr(runner, '_vae_progress', None),
                        blend_mode=vae_tile_blend,
                    )
                    del frames_torch

                latent_t = torch.from_numpy(latent_np.astype(np.float32)).squeeze(0)
                latent_t = (latent_t - shift) * scale
                latent_t = latent_t.to(dtype=torch.bfloat16)
                del latent_np

                mlx_vae.encoder = None
                mlx_vae.decoder = None
                gc.collect()
                _mx.clear_cache()
                torch.mps.empty_cache()

                if _dit_io_freed and _dit is not None:
                    _dit.io = _dit._saved_io
                    del _dit._saved_io
                    _mx.eval(*[v for v in _dit.io.__dict__.values()
                               if isinstance(v, _mx.array)])
                    _bp = getattr(_dit, 'block_pool', None)
                    if _bp is not None:
                        _bp.prefetch_block(0)
                return [latent_t]

            @torch.no_grad()
            def _hybrid_vae_decode(latents):
                import mlx.core as _mx
                runner = self.runner
                # Flush stale DiT activation buffers from the MLX Metal cache.
                _dit = getattr(runner, 'dit', None)
                _dit_io_freed = False
                if _dit is not None and hasattr(_dit, 'io') and _dit.io is not None:
                    _dit._saved_io = _dit.io
                    _dit.io = None
                    _bp = getattr(_dit, 'block_pool', None)
                    if _bp is not None:
                        _bp.reset_pool_state()
                    _dit_io_freed = True
                gc.collect()
                _mx.clear_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                samples = []
                for latent in latents:
                    if latent.dim() == 5:
                        latent = latent.squeeze(0)  # [T, H, W, C]
                    T_lat, lat_H, lat_W = latent.shape[0], latent.shape[1], latent.shape[2]
                    pixel_H, pixel_W = lat_H * 8, lat_W * 8

                    lat_unscaled = latent.float() / scale + shift
                    lat_np = lat_unscaled.cpu().numpy()[np.newaxis].astype(np.float32)
                    del lat_unscaled, latent

                    tile_infos = mlx_vae.tiled_decode(
                        lat_np,
                        tile_size=VAE_DECODE_TILE_SIZE,
                        tile_overlap=VAE_DECODE_TILE_OVERLAP,
                        temporal_slice=HYBRID_TEMPORAL_DECODE_SLICE,
                        vpp=self.ctx.get('vpp'),
                        progress_fn=getattr(runner, '_vae_progress', None),
                        tile_dtype=self.ctx.get('_tile_dtype', 'uint8'),
                        blend_mode=vae_tile_blend,
                    )
                    del lat_np

                    T_pix_full = 1 if T_lat == 1 else (T_lat - 1) * 4 + 1
                    T_pix = min(T_pix_full,
                                self.ctx.get('original_temporal_frames', T_pix_full))
                    self.ctx['_decode_meta'] = {
                        'tile_infos': tile_infos,
                        'pixel_H': pixel_H,
                        'pixel_W': pixel_W,
                        'T_pix': T_pix,
                    }

                    sample_t = torch.empty(3, T_pix, pixel_H, pixel_W,
                                          dtype=torch.float16)
                    samples.append(sample_t)

                mlx_vae.decoder = None
                gc.collect()
                _mx.clear_cache()

                if _dit_io_freed and _dit is not None:
                    _dit.io = _dit._saved_io
                    del _dit._saved_io
                    _mx.eval(*[v for v in _dit.io.__dict__.values()
                               if isinstance(v, _mx.array)])
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
                device=self.device, output_callback=_hdr_output_callback,
                shift_back=_shift_back,
                progress_fn=_progress_adapter,
            )
        finally:
            reader.close()
            if writer_holder[0] is not None:
                try:
                    writer_holder[0].close()
                except Exception:
                    logger.exception("Error closing HDRVideoWriter")
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
        frames_written = saved_frames[0] if saved_frames[0] > 0 else (load_cap if load_cap > 0 else 0)
        audio_duration = (frames_written / fps) if (fps and frames_written > 0) else None
        if output_codec.lower() != 'exr':
            copy_audio_to_video(
                input_path, output_path,
                audio_start=audio_start,
                audio_duration=audio_duration,
            )

        logger.info(f"Saved {saved_frames[0]} frames to {output_path}")
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
        if hasattr(self, 'mlx_vae') and self.mlx_vae is not None:
            self.mlx_vae.encoder = None
            self.mlx_vae.decoder = None
            self.mlx_vae = None
        if hasattr(self, 'runner') and self.runner is not None:
            if hasattr(self.runner, 'dit') and hasattr(self.runner.dit, 'cleanup'):
                self.runner.dit.cleanup()
            self.runner.dit = None
            if hasattr(self.runner, 'vae'):
                del self.runner.vae
            self.runner = None
        try:
            import mlx.core as _mx
            _mx.clear_cache()
        except ImportError:
            pass
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyp2 Video Upscaler (Mac Streaming)")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output_width", type=int, default=1920)
    parser.add_argument("--output_height", type=int, default=1080)
    parser.add_argument("--output_codec", type=str, default="prores")
    parser.add_argument("--split_dir", type=str, default="./nadit_blocks_3b",
                        help="Directory containing split DiT blocks")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory containing VAE + embeddings (default: ./models)")
    parser.add_argument("--vae_path", type=str, default=None,
                        help="VAE safetensors path (overrides model_dir)")
    parser.add_argument(
        "--vae-cache-gb",
        type=float,
        default=None,
        help="MLX Metal buffer cache limit for MLXVideoVAE (default: module "
             f"MLX_VAE_CACHE_LIMIT_GB={MLX_VAE_CACHE_LIMIT_GB}). "
             "Use a negative value to leave MLX default (skip set_cache_limit).",
    )
    parser.add_argument("--mlx_dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="MLX DiT compute precision. bf16 is the default; fp16 matches "
                             "training; fp32 is bit-accurate but uses 2x memory.")
    parser.add_argument("--mlx_stream", action="store_true", default=True,
                        help="Stream DiT blocks from disk instead of holding all 32 in memory. "
                             "Drops weight residency from ~6 GB to ~0.4 GB at <1%% time cost. "
                             "Default: True.")
    parser.add_argument("--no_mlx_stream", action="store_false", dest="mlx_stream",
                        help="Load all 32 DiT blocks into memory instead of streaming.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--chunking_mode", default="latent", choices=["pixel", "latent"],
                        help="Chunking strategy: 'latent' (default, hybrid streaming) or 'pixel'")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not torch.backends.mps.is_available():
        logger.error("GPU not available. Requires Apple Silicon Mac.")
        sys.exit(1)

    logger.info("GPU detected")

    output = args.output or str(Path(args.input).with_stem(Path(args.input).stem + "_upscaled"))
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    # MLX DiT/VAE compute on MLX Metal; Torch still uses MPS for embeddings,
    # diffusion setup, and small tensors so torch.mps.empty_cache() can release
    # the MPS allocator pool between phases.
    upscaler_device = "mps"

    try:
        upscaler = VideoUpscaler(
            split_dir=args.split_dir,
            model_dir=args.model_dir,
            vae_path=args.vae_path,
            device=upscaler_device,
            debug_enabled=args.debug,
            mlx_dtype=args.mlx_dtype,
            mlx_stream=args.mlx_stream,
            vae_cache_limit_gb=args.vae_cache_gb,
        )

        upscaler(
            input_path=args.input,
            output_path=output,
            model_parameters={
                'output_width': args.output_width,
                'output_height': args.output_height,
                'chunking_mode': args.chunking_mode,
                'output_codec': args.output_codec,
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
