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
import torch.nn.functional as F
from typing import Callable, Dict, Any, Optional, Tuple, List

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


def _compute_dit_tiles(
    input_h: int, input_w: int, resolution: int
) -> Optional[Tuple[int, int]]:
    """
    Compute DiT spatial tiling ``(rows, cols)`` based on target pixel count.

    The target dimensions are reconstructed from the input aspect ratio and
    *resolution* (shortest edge of the target).

    Rules (by total pixel count):
        - Target ≤ 1920×1080 (2.07M pixels)  →  ``None``  (no tiling)
        - Target ≤ 3840×1080 (4.15M pixels)  →  ``(1, 2)`` or ``(2, 1)``
          (2 tiles along the longer edge only)
        - Target > 3840×1080 (4.15M pixels)  →  ``(2, 2)``
    """
    THRESHOLD_NO_TILE = (1920+100) * (1080+100)    # ~2.07M pixels
    THRESHOLD_ONE_AXIS = (3840+100) * (1080+100)   # ~4.15M pixels

    # Reconstruct target dimensions from input aspect ratio
    if input_w >= input_h:
        target_h = resolution
        target_w = round(resolution * input_w / input_h)
    else:
        target_w = resolution
        target_h = round(resolution * input_h / input_w)

    total_pixels = target_w * target_h

    if total_pixels <= THRESHOLD_NO_TILE:
        return None
    elif total_pixels <= THRESHOLD_ONE_AXIS:
        # Tile only the longer edge
        if target_w >= target_h:
            return (1, 2)   # 2 tiles along width (cols)
        else:
            return (2, 1)   # 2 tiles along height (rows)
    else:
        return (2, 2)


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
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
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
        ``(decoded, transformed, chunk_timings)`` where
        - *decoded*       is ``[T, H, W, C]`` in ``[-1, 1]`` on **CPU**
        - *transformed*   is ``[C, T, H, W]`` in ``[-1, 1]`` on **CPU** (or ``None``)
        - *chunk_timings* is a dict with ``encode``, ``dit``, ``decode`` times in seconds
    """
    _reset_ctx(ctx)

    def _peak_vram_gb():
        """Total GPU memory in use (PyTorch + TRT + any other allocator)."""
        free, total = torch.cuda.mem_get_info()
        return (total - free) / (1024 ** 3)

    # Phase 1 -- VAE Encode
    t_enc_start = time.time()
    ctx = encode_whole_video(
        runner, ctx, chunk_frames, debug, resolution, max_resolution
    )
    t_enc_end = time.time()
    vram_encode = getattr(runner, '_trt_vram_encode', None) or _peak_vram_gb()

    # Release fragmented CUDA cache between phases so DiT has contiguous blocks
    torch.cuda.empty_cache()

    # Phase 2 -- DiT Upscale  (chunk_size=0 -> process full latent in one shot)
    # cache_model=True keeps runner.dit alive for subsequent chunks
    t_dit_start = time.time()
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
    t_dit_end = time.time()
    vram_dit = _peak_vram_gb()

    # Release fragmented CUDA cache between phases so VAE decode has contiguous blocks
    torch.cuda.empty_cache()

    # Phase 3 -- VAE Decode
    # cache_model=True keeps runner.vae alive for subsequent chunks
    t_dec_start = time.time()
    ctx = decode_whole_latent(runner, ctx, debug, cache_model=True)
    t_dec_end = time.time()
    vram_decode = getattr(runner, '_trt_vram_decode', None) or _peak_vram_gb()

    # Move results to CPU
    decoded = ctx['decoded_video']
    if decoded.is_cuda or (hasattr(decoded, 'is_mps') and decoded.is_mps):
        decoded = decoded.cpu()

    transformed = ctx.get('transformed_video')
    if transformed is not None:
        transformed = transformed.cpu()

    chunk_timings = {
        "encode": t_enc_end - t_enc_start,
        "dit": t_dit_end - t_dit_start,
        "decode": t_dec_end - t_dec_start,
        "vram_encode": vram_encode,
        "vram_dit": vram_dit,
        "vram_decode": vram_decode,
    }

    return decoded, transformed, chunk_timings


# =============================================================================
# Inline post-processing (for streaming mode)
# =============================================================================

def _postprocess_frames_inline(
    decoded: torch.Tensor,
    transformed: Optional[torch.Tensor],
    color_correction: str,
    usm_weight: float,
    usm_radius: int,
    usm_threshold: int,
    device: str = "cuda:0",
) -> torch.Tensor:
    """
    Post-process a small batch of decoded frames **in-place** so they can be
    written to disk immediately.

    This is the streaming equivalent of ``postprocess_latent_output`` —
    it applies color correction + USM sharpening on a per-batch basis
    instead of on the whole video.

    Args:
        decoded:          ``[T, H, W, C]`` in ``[-1, 1]`` on **CPU**.
        transformed:      ``[C, T, H, W]`` in ``[-1, 1]`` on **CPU** (reference
                          for color correction, or ``None``).
        color_correction: Color-correction method name.
        usm_weight:       USM sharpening weight (0 = disabled).
        usm_radius:       USM Gaussian kernel size (must be odd).
        usm_threshold:    USM pixel-difference threshold.
        device:           GPU device string.

    Returns:
        ``[T, H, W, C]`` float16 ``[0, 1]`` on **CPU**, ready for saving.
    """
    T = decoded.shape[0]
    if T == 0:
        return decoded.to(dtype=torch.float16)

    # ── Color correction ──
    if color_correction != "none" and transformed is not None:
        from ..utils.color_fix import (
            lab_color_transfer, lab_reinhard_transfer, wavelet_reconstruction,
            hsv_saturation_histogram_match, wavelet_adaptive_color_correction,
            adaptive_instance_normalization,
        )

        # transformed: [C, T, H, W] → reference_cpu: [T, C, H, W]
        reference_cpu = transformed.permute(1, 0, 2, 3)

        # video: [T, H, W, C] → [T, C, H, W]
        video = decoded.permute(0, 3, 1, 2).to(device)

        chunk_sz = max(1, min(32, T))
        result_chunks = []
        for s in range(0, T, chunk_sz):
            e = min(s + chunk_sz, T)
            vc = video[s:e]
            rc = reference_cpu[s:e].to(device)

            if color_correction == "lab":
                vc = lab_color_transfer(vc, rc, None, luminance_weight=0.8)
            elif color_correction == "lab_reinhard":
                vc = lab_reinhard_transfer(vc, rc, None)
            elif color_correction == "wavelet":
                vc = wavelet_reconstruction(vc, rc, None)
            elif color_correction == "hsv":
                orig_dev = vc.device
                vc = hsv_saturation_histogram_match(vc.cpu(), rc.cpu(), None).to(orig_dev)
            elif color_correction == "wavelet_adaptive":
                vc = wavelet_adaptive_color_correction(vc, rc, None)
            elif color_correction == "adain":
                vc = adaptive_instance_normalization(vc, rc)

            vc.clamp_(-1, 1).mul_(0.5).add_(0.5)
            result_chunks.append(vc.permute(0, 2, 3, 1).to(dtype=torch.float16).cpu())
            del rc
        del video
        torch.cuda.empty_cache()
        video_out = torch.cat(result_chunks, dim=0)
        del result_chunks, reference_cpu
    else:
        video_out = decoded.clamp(-1, 1).mul(0.5).add(0.5).to(dtype=torch.float16)

    # ── USM sharpening ──
    if usm_weight > 0:
        radius = usm_radius if usm_radius % 2 == 1 else usm_radius + 1
        gpu_device = torch.device(device)
        _, H, W, C = video_out.shape

        sigma = 0.3 * ((radius - 1) * 0.5 - 1) + 0.8
        ax = torch.arange(radius, device=gpu_device, dtype=torch.float32) - (radius - 1) / 2.0
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        pad = radius // 2

        kernel_h = gauss_1d.view(1, 1, 1, -1).expand(C, -1, -1, -1).contiguous()
        kernel_v = gauss_1d.view(1, 1, -1, 1).expand(C, -1, -1, -1).contiguous()

        def _blur(x):
            x = F.pad(x, [pad, pad, 0, 0], mode='reflect')
            x = F.conv2d(x, kernel_h, groups=C)
            x = F.pad(x, [0, 0, pad, pad], mode='reflect')
            x = F.conv2d(x, kernel_v, groups=C)
            return x

        usm_chunk = max(1, min(8, T))
        for s in range(0, T, usm_chunk):
            e = min(s + usm_chunk, T)
            chunk = video_out[s:e].to(device=gpu_device, dtype=torch.float32).permute(0, 3, 1, 2)
            blur = _blur(chunk)
            residual = chunk - blur
            del blur
            mask = (residual.abs() * 255.0 > usm_threshold).float()
            soft_mask = _blur(mask)
            del mask
            sharp = (chunk + usm_weight * residual).clamp_(0, 1)
            out = soft_mask * sharp + (1.0 - soft_mask) * chunk
            del residual, soft_mask, sharp, chunk
            video_out[s:e] = out.permute(0, 2, 3, 1).to(dtype=torch.float16).cpu()
            del out
            torch.cuda.empty_cache()

        del kernel_h, kernel_v, gauss_1d

    return video_out


# =============================================================================
# Full pixel-space temporal-chunking pipeline
# =============================================================================

def process_video_pix_chunked(
    runner,
    ctx: Dict[str, Any],
    frames,
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
    output_callback: Optional[Callable[[torch.Tensor], None]] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
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
    in the reference).

    **Streaming mode** (``output_callback is not None``):
        Finalized frames are post-processed per-chunk and streamed via
        ``output_callback(tensor)`` (``[N, H, W, C]`` float16 ``[0, 1]``).
        Only the overlap buffer is kept in RAM.  Returns ``(None, timings)``.

    **Accumulate mode** (``output_callback is None``):
        All frames are accumulated in a CPU buffer, post-processed as one,
        and returned as ``(result, timings)``.

    Args:
        runner:             VideoDiffusionInfer (loaded DiT + VAE).
        ctx:                Shared generation context dict.
        frames:             Input frames -- ``torch.Tensor [T, H, W, C]`` or a
                            lazy reader with ``.shape`` and slice support.
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
        output_callback:    If set, stream post-processed frames to this
                            callable instead of accumulating in RAM.

    Returns:
        ``(result, timings)`` where
        - *result*  is ``[T, H, W, C]`` float16 ``[0, 1]`` on CPU (accumulate mode)
          or ``None`` (streaming mode).
        - *timings* is a dict with ``encode_dit_decode``, ``encode``, ``dit``,
          ``decode``, ``post``.
    """
    streaming = output_callback is not None
    T_total = frames.shape[0]

    # stride = new (non-overlapping) frames advanced per chunk
    step = max(1, chunk_size - overlap) if overlap > 0 else chunk_size
    if step <= 0:
        step = chunk_size
        overlap = 0
        logger.warning("temporal_overlap >= chunk_size, resetting overlap to 0")

    # --- Compute overlapping read windows ---
    if chunk_size <= 0 or chunk_size >= T_total:
        read_windows: List[Tuple[int, int]] = [(0, T_total)]
    else:
        read_windows = []
        for start in range(0, T_total, step):
            end = min(start + chunk_size, T_total)
            if start > 0 and (end - start) <= overlap:
                break
            read_windows.append((start, end))

    num_chunks = len(read_windows)

    # Auto-determine spatial DiT tiling from target output resolution
    if dit_tiles is None:
        _, in_h, in_w, _ = frames.shape
        dit_tiles = _compute_dit_tiles(in_h, in_w, resolution)

    logger.info(
        "Pixel-space chunking: %d chunk(s), batch_size=%d, temporal_overlap=%d, step=%d, dit_tiles=%s, streaming=%s",
        num_chunks, chunk_size, overlap, step, dit_tiles, streaming,
    )

    # --- Process each chunk through encode -> DiT -> decode ---
    t_pipeline_start = time.time()
    total_encode_time = 0.0
    total_dit_time = 0.0
    total_decode_time = 0.0
    total_post_time = 0.0
    peak_vram_encode = 0.0
    peak_vram_dit = 0.0
    peak_vram_decode = 0.0

    # ------------------------------------------------------------------
    # Helper: post-process and emit (streaming) or just return (accum.)
    # ------------------------------------------------------------------
    def _pp(dec: torch.Tensor, trans: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Post-process ``dec`` frames and stream / return them."""
        if dec.shape[0] == 0:
            return dec.to(dtype=torch.float16) if not streaming else None
        processed = _postprocess_frames_inline(
            dec, trans, color_correction,
            usm_weight, usm_radius, usm_threshold, device,
        )
        if streaming:
            output_callback(processed)
            del processed
            return None
        return processed

    if streaming:
        # ----- STREAMING MODE -----
        # Keep only the overlap tail between chunks; post-process and
        # stream finalized frames via callback.
        overlap_dec: Optional[torch.Tensor] = None     # [overlap, H, W, C]
        overlap_trans: Optional[torch.Tensor] = None    # [C, overlap, H, W]

        for ci, (ws, we) in enumerate(read_windows):
            chunk_input = frames[ws:we]
            chunk_len = chunk_input.shape[0]

            if chunk_len == 0:
                logger.warning(
                    "Chunk %d/%d: no readable frames at [%d, %d), "
                    "stopping (video shorter than reported)",
                    ci + 1, num_chunks, ws, we,
                )
                break

            video_ended_early = chunk_len < (we - ws)
            if video_ended_early:
                logger.warning(
                    "Chunk %d/%d: got %d frames instead of %d at [%d, %d), "
                    "video ended early — treating as last chunk",
                    ci + 1, num_chunks, chunk_len, we - ws, ws, we,
                )

            logger.info(
                "Chunk %d/%d: input frames [%d, %d) (%d frames)",
                ci + 1, num_chunks, ws, we, chunk_len,
            )

            t0 = time.time()
            decoded, transformed, chunk_timings = process_pix_chunk(
                runner, ctx, chunk_input, debug,
                resolution, max_resolution, seed,
                latent_noise_scale, shared_noise,
                dit_tiles, dit_tile_overlap,
                prompt, negative_prompt, cfg_scale,
            )
            t1 = time.time()
            total_encode_time += chunk_timings["encode"]
            total_dit_time += chunk_timings["dit"]
            total_decode_time += chunk_timings["decode"]
            peak_vram_encode = max(peak_vram_encode, chunk_timings["vram_encode"])
            peak_vram_dit = max(peak_vram_dit, chunk_timings["vram_dit"])
            peak_vram_decode = max(peak_vram_decode, chunk_timings["vram_decode"])
            logger.info(
                "[Profile] Chunk %d/%d: %.2fs (encode=%.2fs, dit=%.2fs, decode=%.2fs) "
                "| Peak VRAM: encode=%.2fGB, dit=%.2fGB, decode=%.2fGB",
                ci + 1, num_chunks, t1 - t0,
                chunk_timings["encode"], chunk_timings["dit"], chunk_timings["decode"],
                chunk_timings["vram_encode"], chunk_timings["vram_dit"], chunk_timings["vram_decode"],
            )

            is_last = (ci == num_chunks - 1) or video_ended_early
            t_pp0 = time.time()

            if ci == 0:
                # ── First chunk ──
                if num_chunks == 1:
                    # Only one chunk — everything is final
                    _pp(decoded, transformed)
                else:
                    # Save everything except the last *overlap* frames
                    if overlap > 0 and decoded.shape[0] > overlap:
                        safe_dec = decoded[:-overlap]
                        safe_trans = transformed[:, :-overlap] if transformed is not None else None
                        _pp(safe_dec, safe_trans)
                        overlap_dec = decoded[-overlap:].clone()
                        overlap_trans = transformed[:, -overlap:].clone() if transformed is not None else None
                    elif overlap > 0:
                        # Entire chunk is overlap-sized; hold it all
                        overlap_dec = decoded.clone()
                        overlap_trans = transformed.clone() if transformed is not None else None
                    else:
                        _pp(decoded, transformed)
            else:
                # ── Subsequent chunks ──
                actual_overlap = min(overlap, decoded.shape[0])
                if overlap_dec is not None:
                    actual_overlap = min(actual_overlap, overlap_dec.shape[0])

                if actual_overlap > 0 and overlap_dec is not None:
                    # Blend the overlap region
                    blended_dec = _blend_overlapping_frames(
                        overlap_dec, decoded[:actual_overlap], actual_overlap,
                    )
                    blended_trans = None
                    if overlap_trans is not None and transformed is not None:
                        blended_trans_raw = _blend_overlapping_transformed(
                            overlap_trans, transformed[:, :actual_overlap], actual_overlap,
                        )
                        blended_trans = blended_trans_raw

                    debug.log(
                        f"Blended {actual_overlap} overlapping frames",
                        category="video", indent_level=1,
                    )

                    # Blended overlap is now final — post-process & stream
                    _pp(blended_dec, blended_trans)
                    del blended_dec
                    if blended_trans is not None:
                        del blended_trans

                # Non-overlapping remainder of this chunk
                rest_dec = decoded[actual_overlap:]
                rest_trans = transformed[:, actual_overlap:] if transformed is not None else None

                if is_last or overlap == 0:
                    # All remaining frames are final
                    _pp(rest_dec, rest_trans)
                    overlap_dec = None
                    overlap_trans = None
                else:
                    # Hold back the tail for blending with the next chunk
                    if rest_dec.shape[0] > overlap:
                        safe_dec = rest_dec[:-overlap]
                        safe_trans = rest_trans[:, :-overlap] if rest_trans is not None else None
                        _pp(safe_dec, safe_trans)
                        overlap_dec = rest_dec[-overlap:].clone()
                        overlap_trans = rest_trans[:, -overlap:].clone() if rest_trans is not None else None
                    elif rest_dec.shape[0] > 0:
                        overlap_dec = rest_dec.clone()
                        overlap_trans = rest_trans.clone() if rest_trans is not None else None
                    else:
                        overlap_dec = None
                        overlap_trans = None

            t_pp1 = time.time()
            total_post_time += t_pp1 - t_pp0

            del decoded, transformed, chunk_input
            if not is_last:
                clear_memory(debug=debug, force=True)
            elif video_ended_early:
                break

        # Flush any remaining overlap buffer (shouldn't normally happen)
        if overlap_dec is not None and overlap_dec.shape[0] > 0:
            t_pp0 = time.time()
            _pp(overlap_dec, overlap_trans)
            t_pp1 = time.time()
            total_post_time += t_pp1 - t_pp0
            del overlap_dec, overlap_trans

        t_pipeline_end = time.time()
        logger.info(
            "[Profile] All %d chunks total: %.2fs (encode=%.2fs, dit=%.2fs, decode=%.2fs, post=%.2fs) "
            "| Peak VRAM: encode=%.2fGB, dit=%.2fGB, decode=%.2fGB",
            num_chunks, t_pipeline_end - t_pipeline_start,
            total_encode_time, total_dit_time, total_decode_time, total_post_time,
            peak_vram_encode, peak_vram_dit, peak_vram_decode,
        )

        timings = {
            "encode_dit_decode": t_pipeline_end - t_pipeline_start,
            "encode": total_encode_time,
            "dit": total_dit_time,
            "decode": total_decode_time,
            "post": total_post_time,
            "vram_encode": peak_vram_encode,
            "vram_dit": peak_vram_dit,
            "vram_decode": peak_vram_decode,
        }
        return None, timings

    else:
        # ----- ACCUMULATE MODE (original behaviour) -----
        final_video: Optional[torch.Tensor] = None
        final_transformed: Optional[torch.Tensor] = None
        current_write_idx = 0

        for ci, (ws, we) in enumerate(read_windows):
            chunk_input = frames[ws:we]
            chunk_len = chunk_input.shape[0]

            if chunk_len == 0:
                logger.warning(
                    "Chunk %d/%d: no readable frames at [%d, %d), "
                    "stopping (video shorter than reported)",
                    ci + 1, num_chunks, ws, we,
                )
                break

            video_ended_early = chunk_len < (we - ws)
            if video_ended_early:
                logger.warning(
                    "Chunk %d/%d: got %d frames instead of %d at [%d, %d), "
                    "video ended early — treating as last chunk",
                    ci + 1, num_chunks, chunk_len, we - ws, ws, we,
                )

            logger.info(
                "Chunk %d/%d: input frames [%d, %d) (%d frames)",
                ci + 1, num_chunks, ws, we, chunk_len,
            )

            t0 = time.time()
            decoded, transformed, chunk_timings = process_pix_chunk(
                runner, ctx, chunk_input, debug,
                resolution, max_resolution, seed,
                latent_noise_scale, shared_noise,
                dit_tiles, dit_tile_overlap,
                prompt, negative_prompt, cfg_scale,
            )
            t1 = time.time()
            total_encode_time += chunk_timings["encode"]
            total_dit_time += chunk_timings["dit"]
            total_decode_time += chunk_timings["decode"]
            peak_vram_encode = max(peak_vram_encode, chunk_timings["vram_encode"])
            peak_vram_dit = max(peak_vram_dit, chunk_timings["vram_dit"])
            peak_vram_decode = max(peak_vram_decode, chunk_timings["vram_decode"])
            logger.info(
                "[Profile] Chunk %d/%d: %.2fs (encode=%.2fs, dit=%.2fs, decode=%.2fs) "
                "| Peak VRAM: encode=%.2fGB, dit=%.2fGB, decode=%.2fGB",
                ci + 1, num_chunks, t1 - t0,
                chunk_timings["encode"], chunk_timings["dit"], chunk_timings["decode"],
                chunk_timings["vram_encode"], chunk_timings["vram_dit"], chunk_timings["vram_decode"],
            )

            decoded_frames = decoded.shape[0]

            # Lazy allocation
            if final_video is None:
                _, H_out, W_out, C_out = decoded.shape
                final_video = torch.zeros(T_total, H_out, W_out, C_out,
                                          dtype=decoded.dtype, device='cpu')
                if transformed is not None:
                    C_t = transformed.shape[0]
                    final_transformed = torch.zeros(C_t, T_total, H_out, W_out,
                                                    dtype=transformed.dtype, device='cpu')

            if ci == 0 or overlap == 0:
                write_end = min(current_write_idx + decoded_frames, T_total)
                frames_to_write = write_end - current_write_idx
                final_video[current_write_idx:write_end] = decoded[:frames_to_write]
                if final_transformed is not None and transformed is not None:
                    final_transformed[:, current_write_idx:write_end] = transformed[:, :frames_to_write]
                current_write_idx = write_end
            else:
                actual_overlap = min(overlap, decoded_frames, current_write_idx)

                if actual_overlap > 0:
                    prev_tail = final_video[current_write_idx - actual_overlap:current_write_idx]
                    cur_head = decoded[:actual_overlap]
                    blended = _blend_overlapping_frames(prev_tail, cur_head, actual_overlap)
                    final_video[current_write_idx - actual_overlap:current_write_idx] = blended

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
            if video_ended_early:
                break
            if ci < num_chunks - 1:
                clear_memory(debug=debug, force=True)

        t_pipeline_end = time.time()
        logger.info(
            "[Profile] All %d chunks total: %.2fs (encode=%.2fs, dit=%.2fs, decode=%.2fs) "
            "| Peak VRAM: encode=%.2fGB, dit=%.2fGB, decode=%.2fGB",
            num_chunks, t_pipeline_end - t_pipeline_start,
            total_encode_time, total_dit_time, total_decode_time,
            peak_vram_encode, peak_vram_dit, peak_vram_decode,
        )

        # Trim pre-allocated buffer to the actual number of written frames
        if current_write_idx < T_total and final_video is not None:
            logger.info("Trimming final video: %d → %d frames (video ended early)", T_total, current_write_idx)
            final_video = final_video[:current_write_idx]
            if final_transformed is not None:
                final_transformed = final_transformed[:, :current_write_idx]

        # --- Post-process (color correction + USM) on the whole assembled video ---
        _reset_ctx(ctx)
        ctx['decoded_video'] = final_video.to(device)
        del final_video
        if final_transformed is not None:
            ctx['transformed_video'] = final_transformed
            del final_transformed
        clear_memory(debug=debug, force=True)

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
            "encode": total_encode_time,
            "dit": total_dit_time,
            "decode": total_decode_time,
            "post": t_post_end - t_post_start,
            "vram_encode": peak_vram_encode,
            "vram_dit": peak_vram_dit,
            "vram_decode": peak_vram_decode,
        }

        result = ctx['final_video']
        if result.is_cuda or (hasattr(result, 'is_mps') and result.is_mps):
            result = result.cpu()

        return result, timings
