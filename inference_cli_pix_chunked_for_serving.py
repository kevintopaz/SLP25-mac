#!/usr/bin/env python3
"""
SeedVR2 Video Upscaler — Pixel-Space Temporal Chunking (Simplified Serving)

Architecture:
    inference_cli_pix_chunked_for_serving.py   (this file — thin entry point)
        └── src/core/generation_phases_pix.py   (pixel-chunk pipeline)
            └── src/core/generation_phases_latent.py  (encode / dit / decode / postprocess)

Pipeline:
    Video → [Pixel chunks + context prepend] → (VAE Encode → DiT → VAE Decode) per chunk
          → Strip context → Concatenate → Post-process (whole)

Usage:
    python inference_cli_pix_chunked_for_serving.py video.mp4 --output_width 3840 --output_height 2160
"""

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import time
import platform
import logging
import multiprocessing as mp
from typing import Tuple, Dict, Any
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))

if os.environ.get('CLOUD_RUN', 'false') == 'true':
    # Cloud: script lives at <root>/models/astra_precision_3/
    matrix_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    _package_name = 'models.astra_precision_3'
else:
    # Local: script lives at <root>/astra_precision_3/
    matrix_root = os.path.abspath(os.path.join(script_dir, '..'))
    _package_name = 'astra_precision_3'

if matrix_root not in sys.path:
    sys.path.insert(0, matrix_root)

if __name__ == '__main__' and __package__ is None:
    __package__ = _package_name

# =============================================================================
# Hardcoded Configuration
# =============================================================================

VIDEO_BACKEND = "ffmpeg"          # "opencv" or "ffmpeg"
USE_10BIT = False
MAX_RESOLUTION = 0
SEED = 32
SKIP_FIRST_FRAMES = 0
LOAD_CAP = 0

# Pixel-space temporal chunking
# PIX_CHUNK_SIZE = batch_size: total frames per encode->dit->decode call.
#   Ideally follows 4n+1 (5, 9, 13, 17, 21, 25, ..., 65) for optimal temporal consistency.
#   Higher = better quality & speed but more VRAM.
# PIX_OVERLAP = temporal_overlap: context frames prepended from previous chunk's tail
#   for smooth blending between chunks.  Must be < PIX_CHUNK_SIZE.
#   New frames per chunk (stride) = PIX_CHUNK_SIZE - PIX_OVERLAP.
PIX_CHUNK_SIZE = 33               # batch_size (4*12+1 = 49) — reduced for 32GB VRAM
PIX_OVERLAP = 8                  # temporal_overlap
DIT_TILE_OVERLAP = 0.1 

# Post-processing
COLOR_CORRECTION = "wavelet" # "lab", "lab_reinhard", "wavelet", "hsv", "wavelet_adaptive", "adain", "none"
USM_WEIGHT = 0.5                  # 0 = disabled. Typical: 0.3-1.0
USM_RADIUS = 10                   # Gaussian blur kernel size (must be odd)
USM_THRESHOLD = 10                # Pixel-difference threshold

# Generation
CFG_SCALE = 1.0                   # 1.0 = no CFG (recommended for SeedVR2 one-step)
LATENT_NOISE_SCALE = 0.0
SHARED_NOISE = False
PROMPT = None
NEGATIVE_PROMPT = None

# Device / offload
CUDA_DEVICE = None
DIT_OFFLOAD_DEVICE = "none"       # "none" → mapped to "cpu"
VAE_OFFLOAD_DEVICE = "none"       # "none" → mapped to "cpu"
TENSOR_OFFLOAD_DEVICE = "cpu"     # "none" → mapped to None

# VAE tiling
VAE_ENCODE_TILED = True
VAE_ENCODE_TILE_SIZE = 1024
VAE_ENCODE_TILE_OVERLAP = 128
VAE_DECODE_TILED = True
VAE_DECODE_TILE_SIZE = 768
VAE_DECODE_TILE_OVERLAP = 128

# Attention
ATTENTION_MODE = "sdpa"           # "sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# =============================================================================
# Early Setup
# =============================================================================

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
os.environ['PYTHONPATH'] = script_dir + ':' + os.environ.get('PYTHONPATH', '')

if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if CUDA_DEVICE is not None:
        device_list_env = [x.strip() for x in CUDA_DEVICE.split(',') if x.strip() != '']
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None and len(device_list_env) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]

import torch
# torch.cuda.set_per_process_memory_fraction(0.9375)  # Limit VRAM to ~30GB on 32GB GPU
import cv2
import numpy as np
import shutil
import subprocess
import tempfile

from .src.core.generation_utils import setup_generation_context, prepare_runner
from .src.core.generation_phases_pix import process_video_pix_chunked
from .src.utils.debug import Debug

debug = Debug(enabled=True)

if os.environ.get('CLOUD_RUN', 'false') == 'true':
    from utils.register.registry_class import MODEL

    def register_if_cloud_run(cls):
        return MODEL.register_class()(cls)
else:
    def register_if_cloud_run(cls):
        return cls

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Video I/O
# =============================================================================

class FFMPEGVideoWriter:
    def __init__(self, path: str, width: int, height: int, fps: float, use_10bit: bool = False):
        pix_fmt = 'yuv420p10le' if use_10bit else 'yuv420p'
        codec = 'libx265' if use_10bit else 'libx264'
        self.proc = subprocess.Popen(
            ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
             '-c:v', codec, '-pix_fmt', pix_fmt, '-preset', 'medium', '-crf', '15', path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def write(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.proc.stdin.write(frame_rgb.astype(np.uint8).tobytes())
        self.proc.stdin.flush()

    def isOpened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def release(self):
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.wait()
            self.proc = None


def get_video_sar(video_path: str) -> float:
    """
    Read the Sample Aspect Ratio (SAR / pixel aspect ratio) from a video file
    using ffprobe.  Returns the SAR as a float (>0).  1.0 means square pixels.

    Reference: old_code.py – VideoReader reads ``stream.sample_aspect_ratio``
    via PyAV and FrameProcessor multiplies the width by SAR to convert to
    square-pixel space before computing the output resolution.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
             '-show_entries', 'stream=sample_aspect_ratio',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=10,
        )
        sar_str = result.stdout.strip()
        if not sar_str or sar_str in ('N/A', '0:1', '0:0'):
            return 1.0
        if ':' in sar_str:
            num, den = sar_str.split(':')
            sar = float(num) / float(den)
        else:
            sar = float(sar_str)
        return sar if sar > 0 else 1.0
    except Exception:
        return 1.0


def compute_resolution(input_w: int, input_h: int, target_w: int, target_h: int) -> Tuple[int, int]:
    """
    Compute final (width, height) that fits within *target_w × target_h*
    while preserving the input aspect ratio.
    """
    scale_w = target_w / input_w
    scale_h = target_h / input_h
    if scale_w > scale_h:
        final_h = target_h
        final_w = round(input_w * scale_h)
    else:
        final_w = target_w
        final_h = round(input_h * scale_w)
    return final_w, final_h


def copy_audio_to_video(audio_source: str, video_target: str):
    """Mux the audio track from *audio_source* into *video_target* (in-place)."""
    video_dir = os.path.dirname(video_target) or "."
    file_ext = os.path.splitext(video_target)[1]

    with tempfile.NamedTemporaryFile(dir=video_dir, suffix=file_ext, delete=True) as tmp:
        temp_output = tmp.name
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-i", video_target,
            "-i", audio_source,
            "-c:v", "copy", "-c:a", "copy",
            "-map", "0:v:0", "-map", "1:a:0",
            "-movflags", "+faststart",
            temp_output, "-y",
        ]
        try:
            subprocess.run(command, check=True, timeout=300)
            shutil.copy(src=temp_output, dst=video_target)
            logger.info("Audio copied from source into %s", video_target)
        except Exception as e:
            logger.warning("Could not copy audio: %s", e)


class LazyVideoReader:
    """
    Lazily reads video frames per-chunk to avoid loading everything into RAM.

    Exposes ``.shape`` and ``__getitem__`` (slice) so it can be used as a
    drop-in replacement for the eager ``torch.Tensor`` returned by
    ``extract_video_frames``.
    """

    def __init__(self, video_path: str, skip_frames: int = 0, load_cap: int = 0):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.skip_frames = skip_frames

        # Probe metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        available = max(0, total - skip_frames)
        if load_cap > 0:
            available = min(available, load_cap)
        self._total_frames = available

        # SAR correction
        self._sar = get_video_sar(video_path)
        if self._sar != 1.0 and self._sar > 0:
            self._display_w = round(raw_w * self._sar)
        else:
            self._display_w = raw_w
        self._display_h = raw_h
        self._raw_w = raw_w

        if self._sar != 1.0:
            debug.log(
                f"SAR={self._sar:.4f}, will pre-stretch {raw_w}x{raw_h} → {self._display_w}x{raw_h}",
                category="file", force=True,
            )
        debug.log(
            f"LazyVideoReader: {self._total_frames} frames, {self._display_w}x{self._display_h}, fps={self.fps}",
            category="file", force=True,
        )

    @property
    def shape(self):
        return (self._total_frames, self._display_h, self._display_w, 3)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self._total_frames
            start = max(0, min(start, self._total_frames))
            stop = max(start, min(stop, self._total_frames))
            return self._read_range(start, stop)
        raise TypeError(f"LazyVideoReader only supports slice indexing, got {type(key)}")

    def _read_range(self, start: int, end: int) -> torch.Tensor:
        """Read frames ``[start, end)`` and return ``[T, H, W, C]`` float16 ``[0, 1]``."""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.skip_frames + start)

        frames = []
        for _ in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        cap.release()

        if not frames:
            raise ValueError(f"No frames read for range [{start}, {end})")

        # SAR pre-stretching
        if self._sar != 1.0:
            h, w = frames[0].shape[:2]
            new_w = round(w * self._sar)
            if new_w != w:
                frames = [cv2.resize(f, (new_w, h), interpolation=cv2.INTER_CUBIC) for f in frames]

        return torch.from_numpy(np.stack(frames)).to(torch.float16)


def save_video(tensor: torch.Tensor, path: str, fps: float, backend: str = "opencv", use_10bit: bool = False):
    T, H, W, _ = tensor.shape
    os.makedirs(Path(path).parent, exist_ok=True)

    if backend == "ffmpeg":
        writer = FFMPEGVideoWriter(path, W, H, fps, use_10bit)
    else:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
        if not writer.isOpened():
            writer.release()
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    for i in range(T):
        frame = (tensor[i].cpu().float().numpy() * 255.0).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    debug.log(f"Saved {T} frames: {path}", category="file", force=True)


def _create_video_writer(path: str, width: int, height: int, fps: float,
                         backend: str = "ffmpeg", use_10bit: bool = False):
    """Create a video writer (used for streaming output)."""
    os.makedirs(Path(path).parent, exist_ok=True)
    if backend == "ffmpeg":
        return FFMPEGVideoWriter(path, width, height, fps, use_10bit)
    else:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        return writer


# =============================================================================
# Upscaler Class
# =============================================================================

@register_if_cloud_run
class AstraSLP3:

    def init_args(self, args):
        if os.environ.get('CLOUD_RUN') == 'true':
            args.model_dir = "./data"
            args.dit_name = "model.part4.bin.enc"
            args.vae_name = "model.part2.bin.enc"
            return args
        else:
            args.model_dir = "models"
            args.dit_name = "ema_3b_bf16.safetensors"
            args.vae_name = "ema_vae_bf16.pth"
            return args

    def __init__(self, **kwargs):
        args = argparse.Namespace()
        args = self.init_args(args)

        device = kwargs.get('device', 'cuda:0')
        self.device = device
        self.debug = Debug(enabled=False)

        dit_off = "cpu" if DIT_OFFLOAD_DEVICE == "none" else DIT_OFFLOAD_DEVICE
        vae_off = "cpu" if VAE_OFFLOAD_DEVICE == "none" else VAE_OFFLOAD_DEVICE
        tensor_off = None if TENSOR_OFFLOAD_DEVICE == "none" else TENSOR_OFFLOAD_DEVICE

        self.ctx = setup_generation_context(
            dit_device=device, vae_device=device,
            dit_offload_device=dit_off, vae_offload_device=vae_off,
            tensor_offload_device=tensor_off, debug=self.debug
        )

        self.runner, cache_ctx = prepare_runner(
            dit_model=args.dit_name, vae_model=args.vae_name,
            model_dir=args.model_dir,
            debug=self.debug, ctx=self.ctx,
            encode_tiled=VAE_ENCODE_TILED,
            encode_tile_size=(VAE_ENCODE_TILE_SIZE, VAE_ENCODE_TILE_SIZE),
            encode_tile_overlap=(VAE_ENCODE_TILE_OVERLAP, VAE_ENCODE_TILE_OVERLAP),
            decode_tiled=VAE_DECODE_TILED,
            decode_tile_size=(VAE_DECODE_TILE_SIZE, VAE_DECODE_TILE_SIZE),
            decode_tile_overlap=(VAE_DECODE_TILE_OVERLAP, VAE_DECODE_TILE_OVERLAP),
            attention_mode=ATTENTION_MODE
        )
        self.ctx['cache_context'] = cache_ctx

        if 'progress_func' in kwargs and kwargs['progress_func'] is not None:
            self.progress_func = kwargs['progress_func']

        logger.info("AstraPrecision3Upscaler initialized successfully")

    def progress_func(self, progress: float, message: str = "Processing..."):
        logger.info(f"Overall progress={progress * 100:.2f}% - {message}")

    def _process_video(self, frames, resolution: int,
                       output_callback=None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Delegate to the pixel-space chunking pipeline in generation_phases_pix.

        Args:
            frames: Input frames – either a ``torch.Tensor`` or a
                ``LazyVideoReader`` (both support ``.shape`` and slice
                indexing).
            resolution: Target shortest-edge resolution.
            output_callback: If set, finalized post-processed frames are
                streamed to this callback instead of being accumulated
                in RAM.  ``callback(frames_tensor)`` where
                ``frames_tensor`` is ``[N, H, W, C]`` float16 ``[0, 1]``.
        """
        return process_video_pix_chunked(
            runner=self.runner,
            ctx=self.ctx,
            frames=frames,
            debug=self.debug,
            resolution=resolution,
            max_resolution=MAX_RESOLUTION,
            chunk_size=PIX_CHUNK_SIZE,
            overlap=PIX_OVERLAP,
            dit_tile_overlap=DIT_TILE_OVERLAP,
            seed=SEED,
            latent_noise_scale=LATENT_NOISE_SCALE,
            shared_noise=SHARED_NOISE,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            cfg_scale=CFG_SCALE,
            color_correction=COLOR_CORRECTION,
            usm_weight=USM_WEIGHT,
            usm_radius=USM_RADIUS,
            usm_threshold=USM_THRESHOLD,
            device=self.device,
            output_callback=output_callback,
        )

    def __call__(
        self,
        input_path: str,
        output_path: str,
        model_parameters: Dict[str, Any] = None,
        progress_func=None
    ) -> str:
        if progress_func is not None:
            self.progress_func = progress_func

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        output_width = int(model_parameters.get('output_width', 3840))
        output_height = int(model_parameters.get('output_height', 2160))

        logger.info(f"Input video: {input_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Target box: widthxheight: {output_width}x{output_height}")

        if callable(getattr(self, 'progress_func', None)):
            self.progress_func(0.0, "Probing video metadata...")

        # ── Lazy video reader: reads frames per-chunk, not all at once ──
        reader = LazyVideoReader(input_path, SKIP_FIRST_FRAMES, LOAD_CAP)
        fps = reader.fps

        if reader.shape[0] < 5:
            debug.log(
                f"Cannot process: clip too short ({reader.shape[0]} frames, minimum 5)",
                level="ERROR", force=True,
            )
            sys.exit(1)

        input_h, input_w = reader.shape[1], reader.shape[2]
        resolution_w, resolution_h = compute_resolution(input_w, input_h, output_width, output_height)
        debug.log(
            f"Input: {input_w}x{input_h}, target box: {output_width}x{output_height} "
            f"→ final resolution: {resolution_w}x{resolution_h}",
            category="info", force=True,
        )

        if callable(getattr(self, 'progress_func', None)):
            self.progress_func(0.05, "Processing video...")

        # ── Streaming output: write finalized frames to disk as they
        #    become available so we never hold the full output in RAM. ──
        writer_holder = [None]   # lazily initialised on first callback
        saved_frames = [0]

        def _output_callback(frames_tensor: torch.Tensor):
            """Receive post-processed frames [N, H, W, C] float16 [0,1] and save."""
            if writer_holder[0] is None:
                _, H, W, _ = frames_tensor.shape
                writer_holder[0] = _create_video_writer(
                    output_path, W, H, fps, VIDEO_BACKEND, USE_10BIT,
                )
            for i in range(frames_tensor.shape[0]):
                frame_np = (frames_tensor[i].float().numpy() * 255.0).astype(np.uint8)
                writer_holder[0].write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
            saved_frames[0] += frames_tensor.shape[0]

        _, timings = self._process_video(
            reader, min(resolution_w, resolution_h),
            output_callback=_output_callback,
        )

        if writer_holder[0] is not None:
            writer_holder[0].release()

        copy_audio_to_video(input_path, output_path)

        if callable(getattr(self, 'progress_func', None)):
            self.progress_func(0.95, "Finalising...")

        logger.info(f"Saved {saved_frames[0]} frames to {output_path}")
        logger.info(
            f"[Profile] Total -- Encode: {timings['encode']:.2f}s, "
            f"DiT: {timings['dit']:.2f}s, "
            f"Decode: {timings['decode']:.2f}s, "
            f"Post: {timings['post']:.2f}s"
        )

        if callable(getattr(self, 'progress_func', None)):
            self.progress_func(1.0, "Video upscaling complete!")

        return output_path


# =============================================================================
# Argument Parsing (minimal)
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SeedVR2 — Pixel-Space Temporal Chunking (Simplified Serving)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output_width", type=int, default=3840, help="Target output width")
    parser.add_argument("--output_height", type=int, default=2160, help="Target output height")
    parser.add_argument("--debug", action="store_true")

    if len(sys.argv) == 1:
        sys.argv.append('--help')
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_arguments()
    debug.enabled = args.debug
    debug.print_header(cli=True)

    debug.log("Pixel-Space Temporal Chunking Pipeline", category="info", force=True)
    debug.log(f"Input: {args.input}", category="file", force=True)
    debug.log(f"Pixel chunks: size={PIX_CHUNK_SIZE}, overlap={PIX_OVERLAP}", category="info", force=True)

    if VIDEO_BACKEND == "ffmpeg" and shutil.which("ffmpeg") is None:
        debug.log("ffmpeg not found", level="ERROR", category="setup", force=True)
        sys.exit(1)

    upscaler = AstraSLP3(device='cuda:0')

    output = args.output or str(Path(args.input).with_stem(Path(args.input).stem + "_output"))

    start = time.time()

    # Validate input
    input_ext = os.path.splitext(args.input)[1].lower()
    if not os.path.isfile(args.input) or input_ext not in VIDEO_EXTENSIONS:
        debug.log(f"Input file is not a video file: {args.input}", level="ERROR", category="setup", force=True)
        sys.exit(1)

    try:
        upscaler(
            input_path=args.input,
            output_path=output,
            model_parameters={'output_width': args.output_width, 'output_height': args.output_height},
        )

        elapsed = time.time() - start
        debug.log("", category="none", force=True)
        debug.log(f"Complete in {elapsed:.2f}s", category="success", force=True)

    except Exception as e:
        debug.log(f"Error: {e}", level="ERROR", force=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    debug.print_footer()


if __name__ == "__main__":
    main()
