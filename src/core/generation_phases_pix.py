"""
Pixel-Space Temporal Chunking -- Generation Phases

This module provides phase functions for pixel-space temporal chunking pipeline.
Used by inference_cli_pix_chunked_for_serving.py.

Architecture follows ComfyUI-SeedVR2_VideoUpscaler/generation_phases.py pattern:
    Input frames are split into overlapping windows of ``chunk_size`` (batch_size)
    with stride ``chunk_size - overlap``.  Each window goes through the full
    encode -> DiT -> decode pipeline independently.  Overlapping decoded regions
    are blended with a Hann-window crossfade, matching the ``decode_all_batches``
    blending in the reference.

Pipeline:
    Video -> [Overlapping pixel windows] -> (Encode -> DiT -> Decode) per window
          -> Blend overlapping decoded regions -> Post-process (whole)

Compared to generation_phases_latent.py (latent-domain chunking):
    Latent:  Encode(whole) -> Chunk latents -> DiT per chunk -> Merge latents -> Decode(whole)
    Pixel:   Chunk pixels  -> (Encode -> DiT -> Decode) per chunk -> Blend pixels -> Postprocess
"""

import time
import logging
import math
import torch
from typing import Dict, Any, Optional, Tuple, List

from .generation_phases_latent import (
    encode_whole_video,
    upscale_latent_chunked,
    decode_whole_latent,
    postprocess_latent_output,
)
from ..optimization.memory_manager import clear_memory

logger = logging.getLogger(__name__)

# Keys to preserve in ctx between chunks (device config, cached models/embeddings)
_CTX_PERSISTENT_KEYS = frozenset({
    'dit_device', 'vae_device', 'dit_offload_device',
    'vae_offload_device', 'tensor_offload_device',
    'compute_dtype', 'cache_context', 'text_embeds',
})


# =============================================================================
# Internal helpers
# =============================================================================

def _reset_ctx(ctx: Dict[str, Any]) -> None:
    """Clear per-run state from *ctx* while keeping device config and caches."""
    for key in list(ctx.keys()):
        if key not in _CTX_PERSISTENT_KEYS:
            del ctx[key]


def _blend_overlapping_frames(
    prev_tail: torch.Tensor,
    cur_head: torch.Tensor,
    overlap: int,
) -> torch.Tensor:
    """
    Blend two overlapping frame sequences using a Hann-window crossfade.

    Matches ``blend_overlapping_frames`` from the ComfyUI-SeedVR2 reference.

    Args:
        prev_tail: Last *overlap* frames from previous batch ``[overlap, H, W, C]``.
        cur_head:  First *overlap* frames from current batch ``[overlap, H, W, C]``.
        overlap:   Number of overlapping frames.

    Returns:
        Blended frames ``[overlap, H, W, C]``.
    """
    device = prev_tail.device
    dtype = prev_tail.dtype

    if overlap >= 3:
        t = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
        blend_start = 1.0 / 3.0
        blend_end = 2.0 / 3.0
        u = ((t - blend_start) / (blend_end - blend_start)).clamp(0.0, 1.0)
        w_prev_1d = 0.5 + 0.5 * torch.cos(math.pi * u)
    else:
        w_prev_1d = torch.linspace(1.0, 0.0, steps=overlap, device=device, dtype=dtype)

    w_prev = w_prev_1d.view(overlap, 1, 1, 1)
    w_cur = 1.0 - w_prev

    return prev_tail * w_prev + cur_head * w_cur


def _blend_overlapping_transformed(
    prev_tail: torch.Tensor,
    cur_head: torch.Tensor,
    overlap: int,
) -> torch.Tensor:
    """
    Blend overlapping transformed frames ``[C, T, H, W]`` layout.

    Same Hann crossfade as ``_blend_overlapping_frames`` but for CTHW tensors.
    """
    device = prev_tail.device
    dtype = prev_tail.dtype

    if overlap >= 3:
        t = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
        blend_start = 1.0 / 3.0
        blend_end = 2.0 / 3.0
        u = ((t - blend_start) / (blend_end - blend_start)).clamp(0.0, 1.0)
        w_prev_1d = 0.5 + 0.5 * torch.cos(math.pi * u)
    else:
        w_prev_1d = torch.linspace(1.0, 0.0, steps=overlap, device=device, dtype=dtype)

    w_prev = w_prev_1d.view(1, overlap, 1, 1)   # [1, overlap, 1, 1] for CTHW
    w_cur = 1.0 - w_prev

    return prev_tail * w_prev + cur_head * w_cur


# =============================================================================
# Single-chunk processing
# =============================================================================

def process_pix_chunk(
    runner,
    ctx: Dict[str, Any],
    chunk_frames: torch.Tensor,
    debug,
    resolution: int,
    max_resolution: int,
    seed: int,
    latent_noise_scale: float,
    shared_noise: bool,
    dit_tiles,
    dit_tile_overlap: float,
    prompt: Optional[str],
    negative_prompt: Optional[str],
    cfg_scale: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Run full encode -> DiT -> decode on a single pixel-space chunk.

    The *ctx* dict is reset at entry so that ``encode_whole_video`` recomputes
    the video transform for this chunk.

    Args:
        runner:             VideoDiffusionInfer instance (DiT + VAE loaded).
        ctx:                Shared generation context.
        chunk_frames:       Input frames ``[T, H, W, C]`` float16 ``[0, 1]``.
        debug:              Debug logger.
        resolution:         Target shortest-edge resolution.
        max_resolution:     Max resolution cap (0 = no limit).
        seed:               Random seed for reproducibility.
        latent_noise_scale: Noise injection scale in latent space.
        shared_noise:       If True, share noise across temporal frames.
        dit_tiles:          Optional ``(rows, cols)`` for spatial DiT tiling.
        dit_tile_overlap:   Overlap ratio for DiT tiles (0.0-0.5).
        prompt:             Optional positive text prompt.
        negative_prompt:    Optional negative text prompt.
        cfg_scale:          Classifier-free guidance scale (1.0 = disabled).

    Returns:
        ``(decoded, transformed)`` where
        - *decoded*      is ``[T, H, W, C]`` in ``[-1, 1]`` on **CPU**
        - *transformed*  is ``[C, T, H, W]`` in ``[-1, 1]`` on **CPU** (or ``None``)
    """
    _reset_ctx(ctx)

    # Phase 1 -- VAE Encode
    ctx = encode_whole_video(
        runner, ctx, chunk_frames, debug, resolution, max_resolution
    )

    # Phase 2 -- DiT Upscale  (chunk_size=0 -> process full latent in one shot)
    # cache_model=True keeps runner.dit alive for subsequent chunks
    ctx = upscale_latent_chunked(
        runner, ctx, debug,
        chunk_size=0, overlap=0,
        seed=seed,
        latent_noise_scale=latent_noise_scale,
        shared_noise=shared_noise,
        dit_tiles=dit_tiles,
        dit_tile_overlap=dit_tile_overlap,
        prompt=prompt,
        negative_prompt=negative_prompt,
        cfg_scale=cfg_scale,
        cache_model=True,
    )

    # Phase 3 -- VAE Decode
    # cache_model=True keeps runner.vae alive for subsequent chunks
    ctx = decode_whole_latent(runner, ctx, debug, cache_model=True)

    # Move results to CPU
    decoded = ctx['decoded_video']
    if decoded.is_cuda or (hasattr(decoded, 'is_mps') and decoded.is_mps):
        decoded = decoded.cpu()

    transformed = ctx.get('transformed_video')
    if transformed is not None:
        transformed = transformed.cpu()

    return decoded, transformed


# =============================================================================
# Full pixel-space temporal-chunking pipeline
# =============================================================================

def process_video_pix_chunked(
    runner,
    ctx: Dict[str, Any],
    frames: torch.Tensor,
    debug,
    resolution: int,
    max_resolution: int = 0,
    chunk_size: int = 65,
    overlap: int = 13,
    seed: int = 42,
    latent_noise_scale: float = 0.0,
    shared_noise: bool = False,
    dit_tiles=None,
    dit_tile_overlap: float = 0.25,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    cfg_scale: float = 1.0,
    color_correction: str = "lab_reinhard",
    usm_weight: float = 0.0,
    usm_radius: int = 51,
    usm_threshold: int = 10,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Complete pixel-space temporal chunking pipeline.

    ``chunk_size`` is the **batch size** -- the total number of frames fed to
    each encode -> DiT -> decode call (analogous to ``--batch_size`` in the
    ComfyUI-SeedVR2 reference).  ``overlap`` is the **temporal overlap** --
    number of frames shared between consecutive chunks (analogous to
    ``--temporal_overlap``).

    Frames are split into overlapping read windows:
        stride = chunk_size - overlap
        Chunk 0: frames[0 : chunk_size]
        Chunk 1: frames[stride : stride + chunk_size]
        Chunk 2: frames[2*stride : 2*stride + chunk_size]
        ...
    Each chunk reads ``chunk_size`` frames directly from the original input.
    Consecutive chunks share ``overlap`` frames in their input.

    After each chunk is decoded, overlapping regions in the decoded output
    are blended using a Hann-window crossfade (matching ``decode_all_batches``
    in the reference).  Finally the assembled video is post-processed as one
    (color correction + USM sharpening).

    Args:
        runner:             VideoDiffusionInfer (loaded DiT + VAE).
        ctx:                Shared generation context dict.
        frames:             All input frames ``[T, H, W, C]`` float16 ``[0, 1]``.
        debug:              Debug logger.
        resolution:         Target shortest-edge resolution.
        max_resolution:     Max resolution cap (0 = no limit).
        chunk_size:         Batch size -- total frames per encode->dit->decode call
                            (ideally 4n+1: 5, 9, 13, 17, 21, ..., 65).
        overlap:            Temporal overlap -- frames shared between consecutive
                            chunks for blending.  Must be < chunk_size.
        seed:               Random seed.
        latent_noise_scale: Latent noise injection scale.
        shared_noise:       Use shared noise across frames.
        dit_tiles:          Optional ``(rows, cols)`` for spatial DiT tiling.
        dit_tile_overlap:   Overlap ratio for spatial tiles.
        prompt:             Optional positive text prompt.
        negative_prompt:    Optional negative text prompt.
        cfg_scale:          CFG scale (1.0 = disabled).
        color_correction:   Color correction method name.
        usm_weight:         USM sharpening weight (0 = disabled).
        usm_radius:         USM Gaussian kernel size.
        usm_threshold:      USM pixel-difference threshold.
        device:             Target device string (e.g. ``"cuda:0"``).

    Returns:
        ``(result, timings)`` where
        - *result*  is ``[T, H, W, C]`` float32 ``[0, 1]`` on CPU
        - *timings* is a dict with ``encode_dit_decode``, ``blend``, ``post``
    """
    T_total = frames.shape[0]

    # stride = new (non-overlapping) frames advanced per chunk
    step = max(1, chunk_size - overlap) if overlap > 0 else chunk_size
    if step <= 0:
        step = chunk_size
        overlap = 0
        logger.warning("temporal_overlap >= chunk_size, resetting overlap to 0")

    # --- Compute overlapping read windows ---
    # Each window reads chunk_size frames; windows are spaced by step.
    # Matches: for batch_idx in range(0, total_frames, step): ... [batch_idx : batch_idx + batch_size]
    if chunk_size <= 0 or chunk_size >= T_total:
        read_windows: List[Tuple[int, int]] = [(0, T_total)]
    else:
        read_windows = []
        for start in range(0, T_total, step):
            end = min(start + chunk_size, T_total)
            # Skip tiny trailing window whose unique frames <= overlap
            if start > 0 and (end - start) <= overlap:
                break
            read_windows.append((start, end))

    num_chunks = len(read_windows)

    # Auto-determine spatial DiT tiling from resolution
    if dit_tiles is None:
        dit_tiles = (2, 2) if resolution >= 1800 else None

    logger.info(
        "Pixel-space chunking: %d chunk(s), batch_size=%d, temporal_overlap=%d, step=%d",
        num_chunks, chunk_size, overlap, step,
    )

    # --- Process each chunk through encode -> DiT -> decode ---
    t_pipeline_start = time.time()

    # We assemble into a pre-allocated final_video buffer and blend in-place,
    # matching the decode_all_batches pattern from the reference.
    # However, we don't know the output spatial dims until the first chunk is
    # decoded, so we handle that lazily.
    final_video: Optional[torch.Tensor] = None          # [T_total, H_out, W_out, C]
    final_transformed: Optional[torch.Tensor] = None     # [C, T_total, H_out, W_out]
    current_write_idx = 0   # tracks how many decoded frames have been written

    for ci, (ws, we) in enumerate(read_windows):
        chunk_input = frames[ws:we]
        chunk_len = chunk_input.shape[0]

        logger.info(
            "Chunk %d/%d: input frames [%d, %d) (%d frames)",
            ci + 1, num_chunks, ws, we, chunk_len,
        )

        t0 = time.time()
        decoded, transformed = process_pix_chunk(
            runner, ctx, chunk_input, debug,
            resolution, max_resolution, seed,
            latent_noise_scale, shared_noise,
            dit_tiles, dit_tile_overlap,
            prompt, negative_prompt, cfg_scale,
        )
        t1 = time.time()
        logger.info("[Profile] Chunk %d encode+dit+decode: %.2fs", ci + 1, t1 - t0)

        # decoded: [T_chunk, H_out, W_out, C]  on CPU, [-1,1]
        # transformed: [C, T_chunk, H_out, W_out] on CPU, [-1,1] or None
        decoded_frames = decoded.shape[0]

        # Lazy allocation of output buffers after first decode reveals spatial dims
        if final_video is None:
            _, H_out, W_out, C_out = decoded.shape
            final_video = torch.zeros(T_total, H_out, W_out, C_out,
                                      dtype=decoded.dtype, device='cpu')
            if transformed is not None:
                C_t = transformed.shape[0]
                final_transformed = torch.zeros(C_t, T_total, H_out, W_out,
                                                dtype=transformed.dtype, device='cpu')

        if ci == 0 or overlap == 0:
            # First chunk or no overlap: write all frames directly
            write_end = min(current_write_idx + decoded_frames, T_total)
            frames_to_write = write_end - current_write_idx
            final_video[current_write_idx:write_end] = decoded[:frames_to_write]
            if final_transformed is not None and transformed is not None:
                final_transformed[:, current_write_idx:write_end] = transformed[:, :frames_to_write]
            current_write_idx = write_end
        else:
            # Subsequent chunks with overlap: blend the overlapping region
            actual_overlap = min(overlap, decoded_frames, current_write_idx)

            if actual_overlap > 0:
                # Blend overlapping region in-place on final_video
                prev_tail = final_video[current_write_idx - actual_overlap:current_write_idx]
                cur_head = decoded[:actual_overlap]
                blended = _blend_overlapping_frames(prev_tail, cur_head, actual_overlap)
                final_video[current_write_idx - actual_overlap:current_write_idx] = blended

                # Blend transformed too
                if final_transformed is not None and transformed is not None:
                    prev_tail_t = final_transformed[:, current_write_idx - actual_overlap:current_write_idx]
                    cur_head_t = transformed[:, :actual_overlap]
                    blended_t = _blend_overlapping_transformed(prev_tail_t, cur_head_t, actual_overlap)
                    final_transformed[:, current_write_idx - actual_overlap:current_write_idx] = blended_t
                    del prev_tail_t, cur_head_t, blended_t

                debug.log(
                    f"Blended {actual_overlap} overlapping frames at positions "
                    f"{current_write_idx - actual_overlap}-{current_write_idx}",
                    category="video", indent_level=1,
                )
                del prev_tail, cur_head, blended

            # Write non-overlapping part
            non_overlap_decoded = decoded[actual_overlap:]
            non_overlap_len = non_overlap_decoded.shape[0]
            write_end = min(current_write_idx + non_overlap_len, T_total)
            frames_to_write = write_end - current_write_idx
            final_video[current_write_idx:write_end] = non_overlap_decoded[:frames_to_write]
            if final_transformed is not None and transformed is not None:
                non_overlap_t = transformed[:, actual_overlap:]
                final_transformed[:, current_write_idx:write_end] = non_overlap_t[:, :frames_to_write]
            current_write_idx = write_end

        del decoded, transformed, chunk_input
        if ci < num_chunks - 1:
            clear_memory(debug=debug, force=True)

    t_pipeline_end = time.time()
    logger.info("[Profile] All chunks encode+dit+decode+blend: %.2fs", t_pipeline_end - t_pipeline_start)

    # --- Post-process (color correction + USM) on the whole assembled video ---
    _reset_ctx(ctx)
    ctx['decoded_video'] = final_video.to(device)
    if final_transformed is not None:
        ctx['transformed_video'] = final_transformed       # stays on CPU

    t_post_start = time.time()
    ctx = postprocess_latent_output(
        ctx, debug, color_correction,
        usm_weight=usm_weight,
        usm_radius=usm_radius,
        usm_threshold=usm_threshold,
    )
    t_post_end = time.time()
    logger.info("[Profile] Post-process: %.2fs", t_post_end - t_post_start)

    timings = {
        "encode_dit_decode": t_pipeline_end - t_pipeline_start,
        "blend": 0.0,  # blending is done inline during chunk assembly
        "post": t_post_end - t_post_start,
    }

    result = ctx['final_video']
    if result.is_cuda or (hasattr(result, 'is_mps') and result.is_mps):
        result = result.cpu()

    return result.to(torch.float32), timings
