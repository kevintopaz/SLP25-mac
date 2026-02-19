#!/usr/bin/env python3
"""
SeedVR2 Video Upscaler - Latent Domain Temporal Chunking (Simplified)

Architecture:
    inference_latent_chunked.py  (this file - simplified entry point)
        └── src/core/generation_phases_latent.py  (phase functions)

Pipeline:
    Video → VAE Encode (whole) → [Latent chunks] → DiT → Merge latents → VAE Decode (whole)

Usage:
    python inference_latent_chunked.py video.mp4 --resolution 1080
"""

import sys
import os
import argparse
import time
import platform
import logging
import multiprocessing as mp
from typing import Tuple, Dict, Any
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
matrix_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

if matrix_root not in sys.path:
    sys.path.insert(0, matrix_root)

if __name__ == '__main__' and __package__ is None:
    __package__ = 'models.astra_precision_3'

# =============================================================================
# Hardcoded Configuration
# =============================================================================

VIDEO_BACKEND = "ffmpeg"          # "opencv" or "ffmpeg"
USE_10BIT = False
MAX_RESOLUTION = 0
SEED = 32
SKIP_FIRST_FRAMES = 0
LOAD_CAP = 0

# Latent chunking
LATENT_CHUNK_SIZE = 15             # Temporal frames per DiT chunk (4n+1)
LATENT_OVERLAP = 7                # Overlap between latent chunks

# Post-processing
COLOR_CORRECTION = "lab_reinhard"          # "lab", "lab_reinhard", "wavelet", "hsv", "wavelet_adaptive", "adain", "none"
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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    if CUDA_DEVICE is not None:
        device_list_env = [x.strip() for x in CUDA_DEVICE.split(',') if x.strip() != '']
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None and len(device_list_env) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]

import torch
import cv2
import numpy as np
import shutil
import subprocess

from .src.core.generation_utils import setup_generation_context, prepare_runner
from .src.core.generation_phases_latent import (
    encode_whole_video,
    upscale_latent_chunked,
    decode_whole_latent,
    postprocess_latent_output
)
from .src.utils.debug import Debug
from .src.optimization.memory_manager import get_gpu_backend

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
            except:
                pass
            self.proc.wait()
            self.proc = None



def compute_resolution(input_w: int, input_h: int, target_w: int, target_h: int) -> int:
    """
    Compute resolution (shorter edge) that fits within target_w x target_h
    while preserving the input aspect ratio.
    
    Example:
        1920x1080 input, 3840x2160 target → 3840x2160 → resolution = 2160
        1440x1080 input, 3840x2160 target → 2880x2160 → resolution = 2160
        1920x800  input, 3840x2160 target → 3840x1600 → resolution = 1600
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


def extract_video_frames(video_path: str, skip_frames: int = 0, load_cap: int = 0) -> Tuple[torch.Tensor, float]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if skip_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    
    to_read = total - skip_frames
    if load_cap > 0:
        to_read = min(to_read, load_cap)
    
    frames = []
    for _ in range(to_read):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames read")
    
    tensor = torch.from_numpy(np.stack(frames)).to(torch.float16)
    debug.log(f"Loaded {len(frames)} frames: {tensor.shape}", category="file", force=True)
    return tensor, fps


def save_video(tensor: torch.Tensor, path: str, fps: float, backend: str = "opencv", use_10bit: bool = False):
    frames = (tensor.cpu().numpy() * 255.0).astype(np.uint8)
    T, H, W, _ = frames.shape
    os.makedirs(Path(path).parent, exist_ok=True)
    
    if backend == "ffmpeg":
        writer = FFMPEGVideoWriter(path, W, H, fps, use_10bit)
    else:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
        if not writer.isOpened():
            writer.release()
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    debug.log(f"Saved {T} frames: {path}", category="file", force=True)


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
            args.dit_name = "ema_3b_fp16.safetensors"
            args.vae_name = "ema_vae_fp16.safetensors"
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

    def _process_video(self, frames: torch.Tensor, resolution: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        dit_tiles = (2, 2) if resolution >= 1800 else None
        dit_tile_overlap = 0.25

        t0 = time.time()
        ctx = encode_whole_video(self.runner, self.ctx, frames, self.debug, resolution, MAX_RESOLUTION)
        t1 = time.time()
        logger.info(f"[Profile] VAE Encode: {t1 - t0:.2f}s")

        ctx = upscale_latent_chunked(
            self.runner, ctx, self.debug,
            chunk_size=LATENT_CHUNK_SIZE, overlap=LATENT_OVERLAP,
            seed=SEED, latent_noise_scale=LATENT_NOISE_SCALE,
            shared_noise=SHARED_NOISE, dit_tiles=dit_tiles, dit_tile_overlap=dit_tile_overlap,
            prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
            cfg_scale=CFG_SCALE
        )
        t2 = time.time()
        logger.info(f"[Profile] DiT Upscale: {t2 - t1:.2f}s")

        ctx = decode_whole_latent(self.runner, ctx, self.debug)
        t3 = time.time()
        logger.info(f"[Profile] VAE Decode: {t3 - t2:.2f}s")

        ctx = postprocess_latent_output(
            ctx, self.debug, COLOR_CORRECTION,
            usm_weight=USM_WEIGHT, usm_radius=USM_RADIUS,
            usm_threshold=USM_THRESHOLD
        )
        t4 = time.time()
        logger.info(f"[Profile] Post-process: {t4 - t3:.2f}s")

        timings = {
            "encode": t1 - t0,
            "dit": t2 - t1,
            "decode": t3 - t2,
            "post": t4 - t3,
        }

        result = ctx['final_video']
        if result.is_cuda or (hasattr(result, 'is_mps') and result.is_mps):
            result = result.cpu()
        return result.to(torch.float32), timings

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
        logger.info(f"Resolution: {output_width}x{output_height}")

        if callable(getattr(self, 'progress_func', None)):
            self.progress_func(0.0, "Loading video frames...")

        frames, fps = extract_video_frames(input_path, SKIP_FIRST_FRAMES, LOAD_CAP)
        if frames.shape[0] < 5:
            debug.log(f"Cannot process: clip too short ({frames.shape[0]} frames, minimum 5)", level="ERROR", force=True)
            sys.exit(1)
        input_h, input_w = frames.shape[1], frames.shape[2]
        resolution_w, resolution_h = compute_resolution(input_w, input_h, output_width, output_height)
        debug.log(f"Input: {input_w}x{input_h}, target box: {output_width}x{output_height} → final resolution: {resolution_w}x{resolution_h}", category="info", force=True)

        if callable(getattr(self, 'progress_func', None)):
            self.progress_func(0.05, "Processing video...")

        result, timings = self._process_video(frames, min(resolution_w, resolution_h))

        if callable(getattr(self, 'progress_func', None)):
            self.progress_func(0.95, "Saving video...")

        t_save_start = time.time()
        save_video(result, output_path, fps, VIDEO_BACKEND, USE_10BIT)
        t_save_end = time.time()

        logger.info(f"Saved output to {output_path}")
        logger.info(
            f"[Profile] Total breakdown — Encode: {timings['encode']:.2f}s, "
            f"DiT: {timings['dit']:.2f}s, Decode: {timings['decode']:.2f}s, "
            f"Post: {timings['post']:.2f}s, Save: {t_save_end - t_save_start:.2f}s"
        )

        if callable(getattr(self, 'progress_func', None)):
            self.progress_func(1.0, "Video upscaling complete!")

        return output_path


# =============================================================================
# Argument Parsing (minimal)
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SeedVR2 - Latent Domain Temporal Chunking (Simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter
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

    debug.log("Latent Domain Temporal Chunking Pipeline", category="info", force=True)
    debug.log(f"Input: {args.input}", category="file", force=True)
    debug.log(f"Latent chunks: size={LATENT_CHUNK_SIZE}, overlap={LATENT_OVERLAP}", category="info", force=True)

    if VIDEO_BACKEND == "ffmpeg" and shutil.which("ffmpeg") is None:
        debug.log("ffmpeg not found", level="ERROR", category="setup", force=True)
        sys.exit(1)

    upscaler = AstraSLP3(device='cuda:0')

    output = args.output or str(Path(args.input).with_stem(Path(args.input).stem + "_output"))

    start = time.time()

    # parameter check 
    input_ext = os.path.splitext(args.input)[1].lower()
    if not os.path.isfile(args.input) or input_ext not in VIDEO_EXTENSIONS:
        debug.log(f"Input file is not a video file: {args.input}", level="ERROR", category="setup", force=True)
        sys.exit(1)

    try:
        upscaler(
            input_path=args.input,
            output_path=output,
            model_parameters={'resolution': args.resolution}
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
