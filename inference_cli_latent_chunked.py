#!/usr/bin/env python3
"""
SeedVR2 Video Upscaler - Latent Domain Temporal Chunking

Architecture (mirrors inference_cli.py):
    inference_latent_chunked.py  (this file - CLI entry point)
        └── src/core/generation_phases_latent.py  (phase functions)

Pipeline:
    Video → VAE Encode (whole) → [Latent chunks] → DiT → Merge latents → VAE Decode (whole)

Usage:
    python inference_latent_chunked.py video.mp4 --resolution 1080 --latent_chunk_size 9 --latent_overlap 3
"""

import sys
import os
import argparse
import time
import platform
import multiprocessing as mp
from typing import Optional, Tuple
from pathlib import Path

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
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--cuda_device", type=str, default=None)
    _pre_args, _ = _pre_parser.parse_known_args()
    if _pre_args.cuda_device is not None:
        device_list_env = [x.strip() for x in _pre_args.cuda_device.split(',') if x.strip()!='']
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
    T, H, W, C = frames.shape
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

def _process_video(frames: torch.Tensor, args: argparse.Namespace, device_id: str) -> torch.Tensor:
    platform_type = get_gpu_backend()
    device = f"{platform_type}:{device_id}" if platform_type != "mps" else "mps"
    
    dit_off = "cpu" if args.dit_offload_device == "none" else args.dit_offload_device
    vae_off = "cpu" if args.vae_offload_device == "none" else args.vae_offload_device
    tensor_off = None if args.tensor_offload_device == "none" else args.tensor_offload_device
    
    ctx = setup_generation_context(
        dit_device=device, vae_device=device,
        dit_offload_device=dit_off, vae_offload_device=vae_off,
        tensor_offload_device=tensor_off, debug=debug
    )
    
    model_dir = args.model_dir
    runner, cache_ctx = prepare_runner(
        dit_model=args.dit_model, vae_model=args.vae_model, model_dir=model_dir,
        debug=debug, ctx=ctx,
        encode_tiled=args.vae_encode_tiled,
        encode_tile_size=(args.vae_encode_tile_size, args.vae_encode_tile_size),
        encode_tile_overlap=(args.vae_encode_tile_overlap, args.vae_encode_tile_overlap),
        decode_tiled=args.vae_decode_tiled,
        decode_tile_size=(args.vae_decode_tile_size, args.vae_decode_tile_size),
        decode_tile_overlap=(args.vae_decode_tile_overlap, args.vae_decode_tile_overlap),
        attention_mode=args.attention_mode
    )
    ctx['cache_context'] = cache_ctx
    
    # Auto-determine DiT tiling based on resolution
    dit_tiles = (2, 2) if args.resolution >= 2160 else None
    dit_tile_overlap = 0.25
    
    # Phase 1: Encode whole video
    ctx = encode_whole_video(runner, ctx, frames, debug, args.resolution, args.max_resolution)
    
    # Phase 2: Upscale latent in chunks
    ctx = upscale_latent_chunked(
        runner, ctx, debug,
        chunk_size=args.latent_chunk_size, overlap=args.latent_overlap,
        seed=args.seed, latent_noise_scale=args.latent_noise_scale,
        shared_noise=args.shared_noise, dit_tiles=dit_tiles, dit_tile_overlap=dit_tile_overlap,
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale
    )
    
    # Phase 3: Decode whole latent
    ctx = decode_whole_latent(runner, ctx, debug)
    
    # Phase 4: Post-process (color correction + optional USM sharpening)
    ctx = postprocess_latent_output(
        ctx, debug, args.color_correction,
        usm_weight=args.usm_weight,
        usm_radius=args.usm_radius,
        usm_threshold=args.usm_threshold
    )
    
    result = ctx['final_video']
    if result.is_cuda or (hasattr(result, 'is_mps') and result.is_mps):
        result = result.cpu()
    return result.to(torch.float32)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SeedVR2 - Latent Domain Temporal Chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--output", default=None)
    parser.add_argument("--video_backend", default="opencv", choices=["opencv", "ffmpeg"])
    parser.add_argument("--10bit", dest="use_10bit", action="store_true")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--dit_model", default=None)
    parser.add_argument("--vae_model", default=None)
    parser.add_argument("--resolution", type=int, default=1080)
    parser.add_argument("--max_resolution", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_first_frames", type=int, default=0)
    parser.add_argument("--load_cap", type=int, default=0)
    
    # Key latent chunking parameters
    parser.add_argument("--latent_chunk_size", type=int, default=9, help="Temporal frames per DiT chunk (4n+1)")
    parser.add_argument("--latent_overlap", type=int, default=3, help="Overlap between latent chunks")
    
    parser.add_argument("--color_correction", default="lab", choices=["lab", "lab_reinhard", "wavelet", "hsv", "wavelet_adaptive", "adain", "none"])
    parser.add_argument("--usm_weight", type=float, default=0.0,
                       help="USM sharpening weight (0 = disabled). Typical values: 0.3-1.0. "
                            "Applied after color correction.")
    parser.add_argument("--usm_radius", type=int, default=51,
                       help="USM Gaussian blur kernel size (must be odd). Default: 51.")
    parser.add_argument("--usm_threshold", type=int, default=10,
                       help="USM pixel-difference threshold to avoid sharpening noise. Default: 10.")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                       help="Classifier-free guidance scale. Default 1.0 (no CFG, recommended for SeedVR2 one-step). "
                            "Use higher values (e.g. 4.0-7.0) for stronger prompt influence (multi-step models).")
    parser.add_argument("--latent_noise_scale", type=float, default=0.0)
    parser.add_argument("--shared_noise", action="store_true")
    
    # Text prompt options
    parser.add_argument("--prompt", type=str, default=None,
                       help="Custom positive text prompt. If provided, encodes using T5-XXL. "
                            "If not provided, uses pre-computed embeddings (recommended for upscaling).")
    parser.add_argument("--negative_prompt", type=str, default=None,
                       help="Custom negative text prompt. Only used if --prompt is also provided.")
    
    if platform.system() != "Darwin":
        parser.add_argument("--cuda_device", default=None)
    parser.add_argument("--dit_offload_device", default="none")
    parser.add_argument("--vae_offload_device", default="none")
    parser.add_argument("--tensor_offload_device", default="cpu")
    
    parser.add_argument("--vae_encode_tiled", action="store_true")
    parser.add_argument("--vae_encode_tile_size", type=int, default=1024)
    parser.add_argument("--vae_encode_tile_overlap", type=int, default=128)
    parser.add_argument("--vae_decode_tiled", action="store_true")
    parser.add_argument("--vae_decode_tile_size", type=int, default=1024)
    parser.add_argument("--vae_decode_tile_overlap", type=int, default=128)
    parser.add_argument("--attention_mode", default="sdpa", 
                       choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"])
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
    debug.log(f"Latent chunks: size={args.latent_chunk_size}, overlap={args.latent_overlap}", category="info", force=True)
    
    if args.video_backend == "ffmpeg" and shutil.which("ffmpeg") is None:
        debug.log("ffmpeg not found", level="ERROR", category="setup", force=True)
        sys.exit(1)
    
    device_id = "0"
    if platform.system() != "Darwin" and args.cuda_device:
        device_id = args.cuda_device.split(',')[0].strip()
    
    start = time.time()
    
    try:        
        frames, fps = extract_video_frames(args.input, args.skip_first_frames, args.load_cap)
        result = _process_video(frames, args, device_id)
        
        output = args.output or str(Path(args.input).with_stem(Path(args.input).stem + "_latent_chunked"))
        save_video(result, output, fps, args.video_backend, args.use_10bit)
        
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
