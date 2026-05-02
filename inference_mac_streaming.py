#!/usr/bin/env python3
"""
SeedVR2 Video Upscaler — Mac Apple Silicon with NeuroStream DiT + MLX VAE

Architecture:
    - DiT: NeuroStream MPS disk-streaming (2-block pool, loads from split safetensors)
    - VAE: MLX conv2d decomposition (tiled encode/decode on Metal)
    - Pipeline: Pixel-space temporal chunking (from generation_phases_pix/latent)

Usage:
    python inference_mac_streaming.py \\
        "assets/test/video.mp4" \\
        --output assets/test_output_mac/output.mp4 \\
        --output_width 1920 --output_height 1080 \\
        --split_dir ./nadit_blocks_3b
"""

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"

import argparse
import time
import gc
import math
import logging
import subprocess
import shlex
import shutil
import tempfile
from typing import Tuple, Dict, Any, Optional, List
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

from .src.core.generation_utils import (
    setup_generation_context,
    prepare_runner,
)
from .src.core.generation_phases_latent import (
    encode_whole_video,
    upscale_latent_chunked,
    decode_whole_latent,
    postprocess_latent_output,
)
from .src.core.generation_phases_pix import process_video_pix_chunked
from .src.core.model_loader import materialize_model
from .src.utils.debug import Debug
from .src.optimization.mlx_video_vae import assemble_tiles_from_disk, cleanup_tile_files

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─── MLX wavelet color correction monkey-patch ─────────────────────────────
# generation_phases_pix.py imports ``wavelet_reconstruction`` from
# ``src.utils.color_fix`` at function-call time. We replace the module-level
# symbol with the MLX-native version so the post-decode color correction runs
# on Apple GPU (~5x faster on 4K than torch CPU). Disable with
# MLX_WAVELET=0 to fall back to the torch implementation.
if os.environ.get("MLX_WAVELET", "1") == "1":
    try:
        from .src.utils import color_fix as _color_fix_mod
        from .src.utils.color_fix_mlx import wavelet_reconstruction_mlx
        _color_fix_mod.wavelet_reconstruction = wavelet_reconstruction_mlx
        logger.info("Patched wavelet_reconstruction with MLX implementation")
    except ImportError:
        # Fallback to relative import path if the package layout differs
        try:
            import src.utils.color_fix as _color_fix_mod
            from src.utils.color_fix_mlx import wavelet_reconstruction_mlx
            _color_fix_mod.wavelet_reconstruction = wavelet_reconstruction_mlx
            logger.info("Patched wavelet_reconstruction with MLX implementation")
        except Exception as _e:
            logger.warning(f"Failed to patch wavelet_reconstruction: {_e}")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

PIX_CHUNK_SIZE = 33
PIX_OVERLAP = 8
DIT_TILE_OVERLAP = 0.1

# Hybrid latent-chunk mode: pixel-space outer loop with streaming VAE
HYBRID_CHUNK_SIZE = 121       # 4*30+1 pixel frames per outer chunk
HYBRID_OVERLAP = 21           # pixel-space overlap between outer chunks
HYBRID_FIXED_WT = 3           # temporal attention window size for full-latent DiT
HYBRID_TEMPORAL_ENCODE_SLICE = 17   # pixel frames per streaming encode slice
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
# Total unified RAM (Apple Silicon shares CPU/GPU memory). Used to auto-tier
# the MLX cache cap and VAE tile sizes — both are RAM-sensitive.
def _system_ram_gb() -> float:
    # SIMULATE_RAM_GB env var: pretend the system has this much RAM, so the
    # auto-tier picks the smaller-tile / smaller-cache / tile-first-DiT path.
    # Useful for benchmarking how the pipeline would behave on a 16 / 24 GB
    # Mac without actually owning one. Compute speeds won't be slowed by
    # simulated memory pressure, but tile counts, cache cap, and DiT execution
    # mode all switch to the simulated tier.
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

# Hybrid mode uses smaller VAE tiles than the non-streaming path because
# conv caches from streaming accumulate across all temporal slices and stay
# alive alongside the current slice's stage-0 activations.
HYBRID_VAE_TILE_SIZE = (640, 640) if _TOTAL_RAM_GB >= 24 else (480, 320)
HYBRID_VAE_TILE_OVERLAP = (64, 64) if _TOTAL_RAM_GB >= 24 else (48, 32)


# DiT spatial-tile execution mode (only applies when dit_tiles > 1, i.e. >1080p
# output). Block-first holds num_tiles × per-block-activation simultaneously
# between blocks — fast but ~4 GB extra peak at 4K. Tile-first runs one tile
# fully through the DiT before the next tile, peaking at one tile's working
# set. Block-first re-loads each block from disk only once total (32 loads at
# 4K (2,2)); tile-first re-loads per tile (32 × num_tiles), so it's slower on
# systems where SSD bandwidth is the bottleneck. Auto-tier:
#
#   < 24 GB RAM -> tile-first (memory-constrained, accept slower disk loads)
#   >= 24 GB    -> block-first (memory available, optimize for speed)
#
# Override with MLX_DIT_TILE_FIRST=1 (force tile-first) or =0 (force block-first).
if "MLX_DIT_TILE_FIRST" not in os.environ:
    os.environ["MLX_DIT_TILE_FIRST"] = "1" if _TOTAL_RAM_GB < 24 else "0"


# VAE tile sizes: (H, W) tuples in pixels. MLX tiled_encode / tiled_decode
# operate on (height, width). Encoder tile size dominates peak VRAM —
# stage-0 activation ≈ 3 × (1 × 33 × H × W × 128 × 2 bytes):
#   480×320 → ~3.5 GB activation peak (needs ~8 GB total)
#   480×480 → ~5.3 GB activation peak (needs ~11 GB total)
#   640×640 → ~10  GB activation peak (needs ~14 GB total)
#
# Auto-tier:
#   <  24 GB RAM -> encoder & decoder both (480, 320)
#   >= 24 GB RAM -> encoder (640, 640), decoder (480, 480)
def _auto_tile_sizes():
    if _TOTAL_RAM_GB >= 24:
        return (640, 640), (64, 64), (480, 480), (48, 48)
    return (480, 320), (48, 32), (480, 320), (48, 32)


VAE_TILE_SIZE, VAE_TILE_OVERLAP, VAE_DECODE_TILE_SIZE, VAE_DECODE_TILE_OVERLAP = _auto_tile_sizes()


# MLX allocator cache cap (Metal buffer pool between tiles). Tiered by total
# unified RAM: too small a cap starves large conv activations and forces fresh
# system allocations every tile (allocator thrashing — low GPU%, high
# latency); too large pushes cache + active over physical RAM and the OS swaps.
#
#   < 16 GB total -> 2 GB cache
#   < 24 GB total -> 4 GB cache
#   >= 24 GB      -> 6 GB cache
#
# Override with --vae-cache-gb on the CLI.
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


class FFMPEGVideoWriter:
    def __init__(self, path, width, height, fps, use_10bit=False, encoding_options=""):
        ffmpeg_bin = _find_ffmpeg()
        if encoding_options:
            codec_args = shlex.split(encoding_options)
        else:
            pix_fmt = 'yuv420p10le' if use_10bit else 'yuv420p'
            codec = 'libx265' if use_10bit else 'libx264'
            codec_args = ['-c:v', codec, '-pix_fmt', pix_fmt,
                          '-preset', 'medium', '-crf', '15']
        self.proc = subprocess.Popen(
            [ffmpeg_bin, '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
             *codec_args, '-movflags', '+faststart', path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def write(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
        self._buffer[self._decode_pos] = (item.to_ndarray(format='rgb24').astype(np.float16) / np.float16(255.0))
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
        abs_end = self.skip_frames + end - 1
        self._decode_up_to(abs_end)
        abs_start = self.skip_frames + start
        frames = [self._buffer[i] for i in range(abs_start, abs_start + (end - start)) if i in self._buffer]
        for k in list(self._buffer):
            if k < abs_start:
                del self._buffer[k]
        if not frames:
            return torch.empty(0, self._display_h, self._display_w, 3, dtype=torch.float16)
        if self._sar != 1.0:
            h, w = frames[0].shape[:2]
            new_w = round(w * self._sar)
            if new_w != w:
                frames = [cv2.resize(f.astype(np.float32), (new_w, h), interpolation=cv2.INTER_CUBIC).astype(np.float16) for f in frames]
        return torch.from_numpy(np.stack(frames))

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
# Streaming NaDiT (reuses MPSStreamingNaDiT from bloom_precision pattern)
# ═══════════════════════════════════════════════════════════════════════════

class MPSStreamingNaDiT(torch.nn.Module):
    """
    Streaming NaDiT for 3B video model on MPS.
    Loads blocks from disk on-demand via MPSNaDiTStreamManager.
    """

    DEFAULT_CONFIG_3B = {
        'vid_dim': 2560,
        'txt_dim': 2560,
        'txt_in_dim': 5120,
        'emb_dim': 15360,
        'heads': 20,
        'head_dim': 128,
        'expand_ratio': 4,
        'norm': 'fusedrms',
        'norm_eps': 1e-5,
        'ada': 'single',
        'qk_bias': False,
        'qk_norm': 'fusedrms',
        'vid_in_channels': 33,
        'vid_out_channels': 16,
        'patch_size': (1, 2, 2),
        'block_type': 'mmdit_sr',
        'window': (4, 3, 3),
        'attention_mode': ATTENTION_MODE,
        'mlp_type': 'swiglu',
        'rope_type': 'mmrope3d',
        'rope_dim': 128,
        'mm_layers': 10,
        'txt_in_norm': 'fusedln',
        'vid_out_norm': 'fusedrms',
    }

    def __init__(self, split_dir, device, dtype=torch.float16, full_pool=False):
        super().__init__()
        from streampool.mps_naditstream import MPSNaDiTStreamManager

        self.device = device
        self.dtype = dtype
        self.split_dir = Path(split_dir)
        self.config = self.DEFAULT_CONFIG_3B.copy()

        manifest_path = self.split_dir / "block_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.num_blocks = len(manifest['blocks'])
        self.model_size = '3B'
        self.window_methods = ['720pwin_by_size_bysize', '720pswin_by_size_bysize'] * (self.num_blocks // 2)

        block_factory = self._create_block_factory()
        block_callback = self._create_block_callback()
        mm_layers = self.config.get('mm_layers')

        self.stream_manager = MPSNaDiTStreamManager.from_split_model(
            split_dir=split_dir, device=device,
            block_factory=block_factory, block_callback=block_callback,
            full_pool=full_pool, dtype=dtype, debug=True, mm_layers=mm_layers,
        )
        self.stream_manager.enable_timing(True)

        self._load_io_components(manifest, self.config, device, dtype)
        self.need_txt_repeat = False  # 3B doesn't repeat text along temporal dim
        self.block_pool = self.stream_manager.block_pool

        logger.info(f"Initialized streaming NaDiT 3B with {self.num_blocks} blocks")

    def _create_block_factory(self):
        config = self.config
        window_methods = self.window_methods
        num_blocks = self.num_blocks

        from src.models.dit_3b.nablocks import get_nablock
        from src.models.dit_3b.modulation import get_ada_layer
        from src.models.dit_3b.normalization import get_norm_layer

        def block_factory(block_idx):
            wm = window_methods[block_idx % len(window_methods)]
            ada = get_ada_layer(config.get('ada', 'single'))
            norm = get_norm_layer(config.get('norm', 'fusedrms'))
            qk_norm = get_norm_layer(config.get('qk_norm', 'fusedrms'))
            mm_layers = config.get('mm_layers', 10)
            shared_weights = not (block_idx < mm_layers)
            is_last_layer = (block_idx == num_blocks - 1)
            return get_nablock(config.get('block_type', 'mmdit_sr'))(
                vid_dim=config['vid_dim'], txt_dim=config['txt_dim'],
                emb_dim=config['emb_dim'], heads=config['heads'],
                head_dim=config['head_dim'], expand_ratio=config['expand_ratio'],
                norm=norm, norm_eps=config['norm_eps'], ada=ada,
                qk_bias=config['qk_bias'], qk_norm=qk_norm,
                mlp_type=config['mlp_type'], shared_weights=shared_weights,
                rope_type=config['rope_type'], rope_dim=config['rope_dim'],
                is_last_layer=is_last_layer, window=config['window'],
                window_method=wm, attention_mode=config['attention_mode'],
            )
        return block_factory

    def _create_block_callback(self):
        from src.models.dit_3b.window import get_window_op
        window_methods = self.window_methods
        attention_mode = self.config['attention_mode']
        dtype = self.dtype

        def configure_block(block, block_idx):
            correct_wm = window_methods[block_idx % len(window_methods)]
            if hasattr(block, 'attn'):
                if hasattr(block.attn, 'window_method'):
                    block.attn.window_method = correct_wm
                if hasattr(block.attn, 'window_op'):
                    block.attn.window_op = get_window_op(correct_wm)
            for module in block.modules():
                if type(module).__name__ == 'FlashAttentionVarlen':
                    module.attention_mode = attention_mode
                    module.compute_dtype = dtype
        return configure_block

    def _load_io_components(self, manifest, config, device, dtype):
        from safetensors.torch import load_file
        import torch.nn as tnn
        from src.models.dit_3b.patch import get_na_patch_layers
        from src.models.dit_3b.embedding import TimeEmbedding
        from src.models.dit_3b.modulation import get_ada_layer
        from src.models.dit_3b.normalization import get_norm_layer
        from einops import rearrange as einops_rearrange
        from src.common.distributed.ops import slice_inputs

        NaPatchIn, NaPatchOut = get_na_patch_layers('v1')
        io_paths = manifest.get('io_components', {})
        vid_dim = config['vid_dim']
        txt_dim = config['txt_dim']
        txt_in_dim = config['txt_in_dim']
        emb_dim = config['emb_dim']
        patch_size = config['patch_size']
        norm_eps = config['norm_eps']

        def _load(path, module):
            state = load_file(path, device=str(device))
            state = {k: v.to(dtype=dtype) for k, v in state.items()}
            module.load_state_dict(state)
            return module.to(device=device, dtype=dtype).requires_grad_(False)

        # vid_in
        self.vid_in = _load(io_paths['vid_in'],
                            NaPatchIn(in_channels=config['vid_in_channels'],
                                      patch_size=patch_size, dim=vid_dim))
        logger.info("Loaded vid_in")

        # vid_out
        self.vid_out = _load(io_paths['vid_out'],
                             NaPatchOut(out_channels=config['vid_out_channels'],
                                        patch_size=patch_size, dim=vid_dim))
        logger.info("Loaded vid_out")

        # vid_out_norm
        norm_layer = get_norm_layer(config.get('vid_out_norm', 'fusedrms'))
        self.vid_out_norm = _load(io_paths['vid_out_norm'],
                                  norm_layer(dim=vid_dim, eps=norm_eps, elementwise_affine=True))
        logger.info("Loaded vid_out_norm")

        # vid_out_ada (with corrected embedding handling)
        class FixedVidOutAda(tnn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.out_shift = tnn.Parameter(torch.zeros(dim))
                self.out_scale = tnn.Parameter(torch.zeros(dim))

            def forward(self, hid, emb, layer, mode, hid_len=None, cache=None, branch_tag=""):
                emb_r = einops_rearrange(emb, "b (d l g) -> b d l g", l=2, g=3)
                emb_layer = emb_r[..., 0, :]
                emb_layer = emb_layer.unsqueeze(1) if hid.ndim > emb_layer.ndim else emb_layer
                if hid_len is not None:
                    emb_layer = slice_inputs(torch.repeat_interleave(emb_layer, hid_len, dim=0), dim=0)
                shiftA, scaleA, gateA = emb_layer.unbind(-1)
                return hid * (scaleA + self.out_scale) + (shiftA + self.out_shift)

        self.vid_out_ada = _load(io_paths['vid_out_ada'], FixedVidOutAda(vid_dim))
        logger.info("Loaded vid_out_ada")

        # txt_in
        if txt_in_dim != txt_dim:
            self.txt_in = _load(io_paths['txt_in'], tnn.Linear(txt_in_dim, txt_dim))
        else:
            self.txt_in = tnn.Identity().to(device=device, dtype=dtype)
        logger.info("Loaded txt_in")

        # emb_in
        self.emb_in = _load(io_paths['emb_in'],
                            TimeEmbedding(sinusoidal_dim=256,
                                          hidden_dim=max(vid_dim, txt_dim),
                                          output_dim=emb_dim))
        logger.info("Loaded emb_in")

        self.emb_scale = None

    def forward(self, vid, txt, vid_shape, txt_shape, timestep, downscale=None, disable_cache=False):
        from src.common.cache import Cache
        from src.common.distributed.ops import slice_inputs
        from src.models.dit_3b import na
        from src.models.dit_3b.nadit import NaDiTOutput

        if txt_shape.size(-1) == 1 and self.need_txt_repeat:
            txt, txt_shape = na.repeat(txt, txt_shape, "l c -> t l c", t=vid_shape[:, 0])
        txt = slice_inputs(txt, dim=0)
        txt = self.txt_in(txt)

        cache = Cache(disable=disable_cache)
        vid, vid_shape = self.vid_in(vid, vid_shape, cache)
        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)
        if self.emb_scale is not None and downscale is not None:
            emb = emb + self.emb_scale(downscale, device=vid.device, dtype=vid.dtype)

        vid, txt, vid_shape, txt_shape = self.stream_manager.forward_all_blocks(
            vid, txt, vid_shape, txt_shape, emb, cache
        )

        if self.device.type == 'mps':
            torch.mps.synchronize()

        if self.vid_out_norm is not None:
            vid = self.vid_out_norm(vid)
            if self.vid_out_ada is not None:
                vid = self.vid_out_ada(
                    vid, emb=emb, layer="out", mode="in",
                    hid_len=cache("vid_len", lambda: vid_shape.prod(-1)),
                    cache=cache, branch_tag="vid",
                )

        vid, vid_shape = self.vid_out(vid, vid_shape, cache)
        return NaDiTOutput(vid_sample=vid)


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
        full_pool=False,
        debug_enabled=False,
        use_mlx_dit=False,
        mlx_dtype="bf16",
        mlx_stream=False,
        vae_cache_limit_gb=None,
    ):
        self.device = device
        self.debug = Debug(enabled=debug_enabled)
        # MLX-DiT options (read by _create_dit when use_mlx_dit=True)
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

        logger.info("Initializing Video Upscaler (Mac Streaming)...")
        logger.info(f"  Split dir: {split_dir}")
        logger.info(f"  Model dir: {model_dir}")
        logger.info(f"  VAE path:  {vae_path}")
        logger.info(
            f"  System RAM: {_TOTAL_RAM_GB:.1f} GB  "
            f"-> enc tile {VAE_TILE_SIZE}, dec tile {VAE_DECODE_TILE_SIZE}"
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
            encode_tiled=True,
            encode_tile_size=(
                VAE_TILE_SIZE if isinstance(VAE_TILE_SIZE, tuple) else (VAE_TILE_SIZE, VAE_TILE_SIZE)
            ),
            encode_tile_overlap=(
                (VAE_TILE_OVERLAP, VAE_TILE_OVERLAP)
                if isinstance(VAE_TILE_OVERLAP, int) else VAE_TILE_OVERLAP
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
        logger.info("Initializing MLX Video VAE...")
        from .src.optimization.mlx_video_vae import MLXVideoVAE
        logger.info(
            "  MLX VAE cache_limit_gb=%s (Metal buffer pool cap)",
            f"{self.vae_cache_limit_gb:.2f}" if self.vae_cache_limit_gb is not None else "None",
        )
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
                # CPU param so pipeline guards see a non-meta device, but the
                # tensor lives outside torch's MPS heap.
                self._p = torch.nn.Parameter(torch.zeros(1, device="cpu"))
        if hasattr(self.runner, "vae") and self.runner.vae is not None:
            del self.runner.vae
            gc.collect()
        self.runner.vae = _VAEStub()
        logger.info("Skipped PyTorch VAE materialization (~478 MB saved on MPS heap)")

        # ─── Create DiT ────────────────────────────────────────────────
        self.use_mlx_dit = use_mlx_dit
        if use_mlx_dit:
            from .src.optimization.mlx_nadit import MLXNaDiT
            mlx_dtype = getattr(self, "mlx_dtype", "bf16")
            mlx_stream = bool(getattr(self, "mlx_stream", False))
            logger.info(f"Creating MLX DiT (dtype={mlx_dtype}, "
                        f"{'streaming' if mlx_stream else 'full-pool'})...")
            self.streaming_dit = MLXNaDiT(
                split_dir=split_dir,
                dtype=mlx_dtype,
                full_pool=not mlx_stream,
                log=lambda s: logger.info(s),
            )
        else:
            logger.info("Creating streaming DiT from split blocks...")
            self.streaming_dit = MPSStreamingNaDiT(
                split_dir=split_dir,
                device=torch.device(device),
                dtype=self.ctx['compute_dtype'],
                full_pool=full_pool,
            )

        if self.use_mlx_dit:
            # MLX DiT doesn't need CompatibleDiT (no torch.nn.Module).
            # The MLXNaDiT.forward already returns NaDiTOutput with PyTorch tensor.
            wrapped_dit = self.streaming_dit
        else:
            # Wrap in CompatibleDiT for MPS dtype/RoPE handling
            from src.optimization.compatibility import CompatibleDiT
            wrapped_dit = CompatibleDiT(
                self.streaming_dit, self.debug,
                compute_dtype=self.ctx['compute_dtype'],
                skip_conversion=False,
            )

        if hasattr(self.runner, 'dit') and self.runner.dit is not None:
            del self.runner.dit
            gc.collect()
        self.runner.dit = wrapped_dit

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
                logger.debug("Freed DiT IO weights for VAE encode")

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

                latent_np = mlx_vae.tiled_encode(
                    frames_torch,
                    tile_size=VAE_TILE_SIZE,
                    tile_overlap=VAE_TILE_OVERLAP,
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
                debug.log(f"MLX Encode: latent {list(latent_t.shape)}", category="vae")

            # Full cleanup: free all encode-phase memory before DiT starts
            mlx_vae.encoder = None
            gc.collect()
            mx.clear_cache()
            torch.mps.empty_cache()
            logger.debug("Freed VAE encoder weights + caches after encode")

            # Restore DiT IO weights
            if _dit_io_freed and _dit is not None:
                _dit.io = _dit._saved_io
                del _dit._saved_io
                mx.eval(*[v for v in _dit.io.__dict__.values()
                          if isinstance(v, mx.array)])
                logger.debug("Restored DiT IO weights after VAE encode")
            return latents

        @torch.no_grad()
        def mlx_vae_decode(latents):
            """MLX VAE decode: List[THWC latent] -> List[CTHW tensor]."""
            import mlx.core as mx
            # Mirror the encode-path setup: drop DiT IO weights and flush both
            # allocator pools BEFORE the decoder peak ResNets allocate. Without
            # this, torch.mps + MLX caches retain DiT activations across the
            # handoff (~several GB on top of the decoder's transient working
            # set), which is what made 480x480 tiles peak at ~13 GB.
            _dit = getattr(runner, 'dit', None)
            _dit_io_freed = False
            if _dit is not None and hasattr(_dit, 'io'):
                _dit._saved_io = _dit.io
                _dit.io = None
                _dit_io_freed = True
                logger.debug("Freed DiT IO weights for VAE decode")
            gc.collect()
            mx.clear_cache()
            torch.mps.empty_cache()

            samples = []
            for latent in latents:
                if latent.dim() == 5:
                    latent = latent.squeeze(0)  # [T, H, W, C]

                # Undo scaling
                lat_unscaled = latent.float() / scale + shift  # [T, H, W, 16]
                lat_np = lat_unscaled.cpu().numpy()[np.newaxis]  # [1, T, H, W, 16]
                # Latent now lives in numpy/host; release the MPS-side copy.
                del lat_unscaled, latent

                frames_np = mlx_vae.tiled_decode(
                    lat_np,
                    tile_size=VAE_DECODE_TILE_SIZE,
                    tile_overlap=VAE_DECODE_TILE_OVERLAP,
                )
                # frames_np: [T_out, H, W, 3] in [-1, 1]
                sample_t = torch.from_numpy(frames_np)
                # Convert to CTHW
                sample_t = sample_t.permute(3, 0, 1, 2)  # [C, T, H, W]
                samples.append(sample_t)
                del lat_np, frames_np
                debug.log(f"MLX Decode: {list(sample_t.shape)}", category="vae")

            # Full cleanup after decode
            mlx_vae.decoder = None
            gc.collect()
            mx.clear_cache()
            torch.mps.empty_cache()
            logger.debug("Freed VAE decoder weights + caches after decode")

            # Restore DiT IO weights for the next chunk's DiT pass.
            if _dit_io_freed and _dit is not None:
                _dit.io = _dit._saved_io
                del _dit._saved_io
                if hasattr(_dit.io, '__dict__'):
                    mx.eval(*[v for v in _dit.io.__dict__.values()
                              if isinstance(v, mx.array)])
                logger.debug("Restored DiT IO weights after VAE decode")
            return samples

        runner.vae_encode = mlx_vae_encode
        runner.vae_decode = mlx_vae_decode
        logger.info("Patched runner with MLX VAE encode/decode")

    def __call__(self, input_path, output_path, model_parameters=None,
                 progress_func=None):
        if model_parameters is None:
            model_parameters = {}

        # --- Resolution / frame range ---
        output_width = int(model_parameters.get('output_width', 1920))
        output_height = int(model_parameters.get('output_height', 1080))
        start_frame_idx = int(model_parameters.get('start_idx',
                              model_parameters.get('start_frame_idx', 0)))
        end_frame_idx = model_parameters.get('end_idx',
                        model_parameters.get('end_frame_idx', None))
        if end_frame_idx is not None:
            end_frame_idx = int(end_frame_idx)

        skip_frames = max(0, start_frame_idx)
        if end_frame_idx is not None and end_frame_idx > skip_frames:
            load_cap = end_frame_idx - skip_frames + 1  # inclusive
        else:
            load_cap = 0

        # --- Chunking mode ---
        chunking_mode = str(model_parameters.get('chunking_mode', 'pixel'))
        if chunking_mode not in ('pixel', 'latent'):
            logger.warning(f"Unknown chunking_mode={chunking_mode!r}, defaulting to 'pixel'")
            chunking_mode = 'pixel'

        # --- Pipeline parameters ---
        color_correction = str(model_parameters.get('color_correction', COLOR_CORRECTION))

        # --- Output encoding ---
        ffmpeg_encoding = model_parameters.get('ffmpeg_encoding', "") or ""
        ffmpeg_preproc_filters = model_parameters.get('ffmpeg_preproc_filters', "") or ""
        use_10bit = model_parameters.get('output_bit_depth') is not None and \
                    int(model_parameters['output_bit_depth']) >= 10

        logger.info(f"Input video: {input_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Resolution: {output_width}x{output_height}")
        logger.info(f"color_correction={color_correction}")
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
        fps = reader.fps

        if reader.shape[0] < 5:
            logger.error(f"Video too short: {reader.shape[0]} frames (need >= 5)")
            return output_path

        input_h, input_w = reader.shape[1], reader.shape[2]
        resolution_w, resolution_h = compute_resolution(input_w, input_h, output_width, output_height)
        resolution = min(resolution_w, resolution_h)
        logger.info(f"Input: {input_w}x{input_h}, target: {output_width}x{output_height} -> {resolution_w}x{resolution_h}")

        if progress_func:
            progress_func(0.05, message="Processing video...")

        self.streaming_dit.stream_manager.prefetch_start(num_blocks=2)

        writer_holder = [None]
        saved_frames = [0]
        total_frames = reader.shape[0]

        def _output_callback(frames_tensor):
            if writer_holder[0] is None:
                _, H, W, _ = frames_tensor.shape
                writer_holder[0] = FFMPEGVideoWriter(
                    output_path, W, H, fps,
                    use_10bit=use_10bit,
                    encoding_options=ffmpeg_encoding,
                )
            for i in range(frames_tensor.shape[0]):
                frame_np = (frames_tensor[i].cpu().float().numpy() * 255.0).astype(np.uint8)
                writer_holder[0].write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
            saved_frames[0] += frames_tensor.shape[0]

            frac = saved_frames[0] / total_frames
            progress = 0.05 + frac * 0.85
            if progress_func:
                progress_func(progress, message=f"Processed {saved_frames[0]}/{total_frames} frames...")
            logger.info(f"Saved {saved_frames[0]}/{total_frames} frames")

        t_start = time.time()

        # --- Hybrid latent-chunk mode setup ---
        _orig_vae_encode = None
        _orig_vae_decode = None
        if chunking_mode == 'latent':
            logger.info("Hybrid latent-chunk mode: chunk=%d, overlap=%d, fixed_wt=%d",
                        HYBRID_CHUNK_SIZE, HYBRID_OVERLAP, HYBRID_FIXED_WT)
            # Set fixed temporal window on DiT
            dit_model = self.runner.dit.dit_model if hasattr(self.runner.dit, 'dit_model') else self.runner.dit
            dit_model.fixed_wt = HYBRID_FIXED_WT

            # Re-patch VAE encode to use streaming
            _orig_vae_encode = self.runner.vae_encode
            _orig_vae_decode = self.runner.vae_decode
            mlx_vae = self.mlx_vae
            scale = self.runner.config.vae.scaling_factor
            shift = self.runner.config.vae.get("shifting_factor", 0.0)

            @torch.no_grad()
            def _hybrid_vae_encode(samples):
                import mlx.core as _mx
                runner = self.runner

                # Free DiT IO weights during encode (~0.58 GB MLX Metal) —
                # mirrors the default mlx_vae_encode path.
                _dit = getattr(runner, 'dit', None)
                _dit_io_freed = False
                if _dit is not None and hasattr(_dit, 'io') and _dit.io is not None:
                    _dit._saved_io = _dit.io
                    _dit.io = None
                    gc.collect()
                    _mx.clear_cache()
                    _dit_io_freed = True

                torch.mps.empty_cache()

                latents = []
                for idx in range(len(samples)):
                    sample = samples[idx]
                    samples[idx] = None  # release original CTHW tensor early
                    frames_torch = sample.permute(1, 2, 3, 0).contiguous().cpu()
                    del sample
                    if frames_torch.dtype != torch.float16:
                        frames_torch = frames_torch.to(torch.float16)
                    latent_np = mlx_vae.tiled_encode_streaming(
                        frames_torch,
                        tile_size=HYBRID_VAE_TILE_SIZE,
                        tile_overlap=HYBRID_VAE_TILE_OVERLAP,
                        temporal_slice=HYBRID_TEMPORAL_ENCODE_SLICE,
                    )
                    del frames_torch
                    latent_t = torch.from_numpy(latent_np.astype(np.float32)).squeeze(0)
                    latent_t = (latent_t - shift) * scale
                    latent_t = latent_t.to(dtype=torch.bfloat16)
                    latents.append(latent_t)
                    del latent_np

                mlx_vae.encoder = None
                gc.collect()
                _mx.clear_cache()
                torch.mps.empty_cache()

                # Restore DiT IO weights
                if _dit_io_freed and _dit is not None:
                    _dit.io = _dit._saved_io
                    del _dit._saved_io
                    _mx.eval(*[v for v in _dit.io.__dict__.values()
                               if isinstance(v, _mx.array)])
                return latents

            @torch.no_grad()
            def _hybrid_vae_decode(latents):
                import mlx.core as _mx
                samples = []
                for latent in latents:
                    if latent.dim() == 5:
                        latent = latent.squeeze(0)  # [T, H, W, C]
                    # latent shape: [T_lat, H_lat, W_lat, C]
                    T_lat, lat_H, lat_W = latent.shape[0], latent.shape[1], latent.shape[2]
                    pixel_H, pixel_W = lat_H * 8, lat_W * 8

                    lat_unscaled = latent.float() / scale + shift
                    lat_np = lat_unscaled.cpu().numpy()[np.newaxis].astype(np.float32)
                    del lat_unscaled, latent

                    tile_infos = mlx_vae.tiled_decode_streaming(
                        lat_np,
                        tile_size=VAE_DECODE_TILE_SIZE,
                        tile_overlap=VAE_DECODE_TILE_OVERLAP,
                        temporal_slice=HYBRID_TEMPORAL_DECODE_SLICE,
                    )
                    del lat_np

                    frames_np = assemble_tiles_from_disk(tile_infos, pixel_H, pixel_W)
                    cleanup_tile_files(tile_infos)
                    sample_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2)  # [C, T, H, W]
                    samples.append(sample_t)
                    del frames_np
                mlx_vae.decoder = None
                gc.collect()
                _mx.clear_cache()
                return samples

            self.runner.vae_encode = _hybrid_vae_encode
            self.runner.vae_decode = _hybrid_vae_decode

        # --- Select chunk parameters ---
        if chunking_mode == 'latent':
            _chunk_size = HYBRID_CHUNK_SIZE
            _overlap = HYBRID_OVERLAP
            _shift_back = False
        else:
            _chunk_size = PIX_CHUNK_SIZE
            _overlap = PIX_OVERLAP
            _shift_back = True

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
                device=self.device, output_callback=_output_callback,
                shift_back=_shift_back,
            )
        finally:
            reader.close()
            if writer_holder[0] is not None:
                writer_holder[0].release()
            # Restore original VAE patches if hybrid mode was used
            if _orig_vae_encode is not None:
                self.runner.vae_encode = _orig_vae_encode
                self.runner.vae_decode = _orig_vae_decode
                dit_model = self.runner.dit.dit_model if hasattr(self.runner.dit, 'dit_model') else self.runner.dit
                dit_model.fixed_wt = None

        elapsed = time.time() - t_start

        if progress_func:
            progress_func(0.92, message="Copying audio track...")

        # Align the muxed audio with the rendered frame range
        audio_start = (skip_frames / fps) if fps and skip_frames > 0 else 0.0
        frames_written = saved_frames[0] if saved_frames[0] > 0 else (load_cap if load_cap > 0 else 0)
        audio_duration = (frames_written / fps) if (fps and frames_written > 0) else None
        copy_audio_to_video(
            input_path, output_path,
            audio_start=audio_start,
            audio_duration=audio_duration,
        )

        logger.info(f"Saved {saved_frames[0]} frames to {output_path}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Encode: {timings.get('encode', 0):.1f}s, "
                    f"DiT: {timings.get('dit', 0):.1f}s, "
                    f"Decode: {timings.get('decode', 0):.1f}s, "
                    f"Post: {timings.get('post', 0):.1f}s")

        if progress_func:
            progress_func(0.95, message="Done")
        return output_path

    def cleanup(self):
        if hasattr(self, 'runner'):
            if hasattr(self.runner, 'dit') and hasattr(self.runner.dit, 'cleanup'):
                self.runner.dit.cleanup()
            if hasattr(self.runner, 'vae'):
                del self.runner.vae
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_arguments():
    parser = argparse.ArgumentParser(description="SeedVR2 Video Upscaler (Mac Streaming)")
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
    parser.add_argument(
        "--vae-cache-gb",
        type=float,
        default=None,
        help="MLX Metal buffer cache limit for MLXVideoVAE (default: module "
             f"MLX_VAE_CACHE_LIMIT_GB={MLX_VAE_CACHE_LIMIT_GB}). "
             "Use a negative value to leave MLX default (skip set_cache_limit).",
    )
    parser.add_argument("--full_pool", action="store_true",
                        help="Keep all DiT blocks in memory (no streaming)")
    parser.add_argument("--mlx_dit", action="store_true",
                        help="Use MLX-native DiT backend (math-identical to PyTorch within "
                             "bf16/fp16 ULP, runs on Apple Silicon GPU via Metal).")
    parser.add_argument("--mlx_dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="MLX DiT compute precision. fp16 matches training; bf16 is the "
                             "default; fp32 is bit-accurate but uses 2x memory. (Only with --mlx_dit)")
    parser.add_argument("--mlx_stream", action="store_true",
                        help="Stream DiT blocks from disk instead of holding all 32 in memory. "
                             "Drops weight residency from ~6 GB to ~0.4 GB at <1%% time cost. "
                             "(Only with --mlx_dit)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not torch.backends.mps.is_available():
        logger.error("MPS not available. Requires Apple Silicon Mac.")
        sys.exit(1)

    logger.info("Apple Silicon MPS detected")

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
            full_pool=args.full_pool,
            debug_enabled=args.debug,
            use_mlx_dit=args.mlx_dit,
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
