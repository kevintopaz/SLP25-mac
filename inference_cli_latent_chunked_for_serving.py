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
import multiprocessing as mp
from typing import Tuple
from pathlib import Path

# =============================================================================
# Hardcoded Configuration
# =============================================================================

VIDEO_BACKEND = "opencv"          # "opencv" or "ffmpeg"
USE_10BIT = False
MODEL_DIR = "models"
DIT_MODEL = "ema_3b_fp16.safetensors"
VAE_MODEL = "ema_vae_fp16.safetensors"
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

# =============================================================================
# Early Setup
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
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

from src.core.generation_utils import setup_generation_context, prepare_runner
from src.core.generation_phases_latent import (
    encode_whole_video,
    upscale_latent_chunked,
    decode_whole_latent,
    postprocess_latent_output
)
from src.utils.debug import Debug
from src.optimization.memory_manager import get_gpu_backend

debug = Debug(enabled=False)


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
# Core Processing
# =============================================================================

def _process_video(frames: torch.Tensor, resolution: int, device_id: str) -> torch.Tensor:
    platform_type = get_gpu_backend()
    device = f"{platform_type}:{device_id}" if platform_type != "mps" else "mps"
    
    dit_off = "cpu" if DIT_OFFLOAD_DEVICE == "none" else DIT_OFFLOAD_DEVICE
    vae_off = "cpu" if VAE_OFFLOAD_DEVICE == "none" else VAE_OFFLOAD_DEVICE
    tensor_off = None if TENSOR_OFFLOAD_DEVICE == "none" else TENSOR_OFFLOAD_DEVICE
    
    ctx = setup_generation_context(
        dit_device=device, vae_device=device,
        dit_offload_device=dit_off, vae_offload_device=vae_off,
        tensor_offload_device=tensor_off, debug=debug
    )
    
    runner, cache_ctx = prepare_runner(
        dit_model=DIT_MODEL, vae_model=VAE_MODEL, model_dir=MODEL_DIR,
        debug=debug, ctx=ctx,
        encode_tiled=VAE_ENCODE_TILED,
        encode_tile_size=(VAE_ENCODE_TILE_SIZE, VAE_ENCODE_TILE_SIZE),
        encode_tile_overlap=(VAE_ENCODE_TILE_OVERLAP, VAE_ENCODE_TILE_OVERLAP),
        decode_tiled=VAE_DECODE_TILED,
        decode_tile_size=(VAE_DECODE_TILE_SIZE, VAE_DECODE_TILE_SIZE),
        decode_tile_overlap=(VAE_DECODE_TILE_OVERLAP, VAE_DECODE_TILE_OVERLAP),
        attention_mode=ATTENTION_MODE
    )
    ctx['cache_context'] = cache_ctx
    
    # Auto-determine DiT tiling based on resolution
    dit_tiles = (2, 2) if resolution >= 2160 else None
    dit_tile_overlap = 0.25
    
    # Phase 1: Encode whole video
    ctx = encode_whole_video(runner, ctx, frames, debug, resolution, MAX_RESOLUTION)
    
    # Phase 2: Upscale latent in chunks
    ctx = upscale_latent_chunked(
        runner, ctx, debug,
        chunk_size=LATENT_CHUNK_SIZE, overlap=LATENT_OVERLAP,
        seed=SEED, latent_noise_scale=LATENT_NOISE_SCALE,
        shared_noise=SHARED_NOISE, dit_tiles=dit_tiles, dit_tile_overlap=dit_tile_overlap,
        prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
        cfg_scale=CFG_SCALE
    )
    
    # Phase 3: Decode whole latent
    ctx = decode_whole_latent(runner, ctx, debug)
    
    # Phase 4: Post-process (color correction + optional USM sharpening)
    ctx = postprocess_latent_output(
        ctx, debug, COLOR_CORRECTION,
        usm_weight=USM_WEIGHT,
        usm_radius=USM_RADIUS,
        usm_threshold=USM_THRESHOLD
    )
    
    result = ctx['final_video']
    if result.is_cuda or (hasattr(result, 'is_mps') and result.is_mps):
        result = result.cpu()
    return result.to(torch.float32)


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
    parser.add_argument("--resolution", type=int, default=1080)
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
    
    device_id = "0"
    if platform.system() != "Darwin" and CUDA_DEVICE:
        device_id = CUDA_DEVICE.split(',')[0].strip()
    
    start = time.time()
    
    try:
        frames, fps = extract_video_frames(args.input, SKIP_FIRST_FRAMES, LOAD_CAP)
        result = _process_video(frames, args.resolution, device_id)
        
        output = args.output or str(Path(args.input).with_stem(Path(args.input).stem + "_latent_chunked"))
        save_video(result, output, fps, VIDEO_BACKEND, USE_10BIT)
        
        elapsed = time.time() - start
        debug.log("", category="none", force=True)
        debug.log(f"Complete in {elapsed:.2f}s ({result.shape[0]/elapsed:.2f} fps)", category="success", force=True)
        
    except Exception as e:
        debug.log(f"Error: {e}", level="ERROR", force=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    debug.print_footer()


if __name__ == "__main__":
    main()
