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

import os
import time
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Dict, Any, Optional, Tuple, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..video_utils import VideoPrePostProcess

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
    '_dit_latent_chunk', '_dit_latent_overlap',
    'vpp', 'video_transform',
})


# =============================================================================
# Internal helpers
# =============================================================================

def _reset_ctx(ctx: Dict[str, Any]) -> None:
    """Clear per-run state from *ctx* while keeping device config and caches."""
    for key in list(ctx.keys()):
        if key not in _CTX_PERSISTENT_KEYS:
            del ctx[key]


from ..video_utils import Blend

_blender_cache: Dict[str, Blend] = {}


def _get_blender(mode: str = "hann") -> Blend:
    if mode not in _blender_cache:
        _blender_cache[mode] = Blend(mode=mode, cpu_weights=True)   # CPU weights: GPU blending == CPU blending
    return _blender_cache[mode]


def _blend_overlapping_frames(
    prev_tail: torch.Tensor,
    cur_head: torch.Tensor,
    overlap: int,
    mode: str = "hann",
    frames: Optional[slice] = None,
) -> torch.Tensor:
    """Blend two overlapping frame sequences ``[overlap, H, W, C]`` (or the *frames* of them)."""
    return _get_blender(mode).temporal_thwc(prev_tail, cur_head, overlap, frames=frames)


def _blend_overlapping_transformed(
    prev_tail: torch.Tensor,
    cur_head: torch.Tensor,
    overlap: int,
    mode: str = "hann",
    frames: Optional[slice] = None,
) -> torch.Tensor:
    """Blend overlapping transformed frames ``[C, T, H, W]`` (or the *frames* of them)."""
    return _get_blender(mode).temporal_cthw(prev_tail, cur_head, overlap, frames=frames)


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
    reader_fn: Callable[[int, int], Any],
    frame_count: int,
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
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    """
    Run full encode -> DiT -> decode on a single pixel-space chunk.

    The *ctx* dict is reset at entry so that ``encode_whole_video`` recomputes
    the video transform for this chunk.

    Args:
        runner:             VideoDiffusionInfer instance (DiT + VAE loaded).
        ctx:                Shared generation context.
        reader_fn:          ``callable(start, count) -> np.ndarray`` returning
                            ``[count, H, W, C]`` raw uint8/uint16 pixels.
        frame_count:        Number of frames in this chunk.
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
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return (total - free) / (1024 ** 3)
        return 0.0

    # Phase 1 -- VAE Encode
    t_enc_start = time.time()
    ctx = encode_whole_video(
        runner, ctx, reader_fn, frame_count, debug, resolution, max_resolution,
        progress_callback=progress_callback,
    )
    t_enc_end = time.time()
    vram_encode = getattr(runner, '_trt_vram_encode', None) or _peak_vram_gb()

    torch.cuda.empty_cache()

    # Phase 2 -- DiT Upscale
    # cache_model=True keeps runner.dit alive for subsequent chunks
    _dit_chunk = ctx.get('_dit_latent_chunk', 0)
    _dit_ov = ctx.get('_dit_latent_overlap', 0)
    t_dit_start = time.time()
    ctx = upscale_latent_chunked(
        runner, ctx, debug,
        chunk_size=_dit_chunk, overlap=_dit_ov,
        seed=seed,
        latent_noise_scale=latent_noise_scale,
        shared_noise=shared_noise,
        dit_tiles=dit_tiles,
        dit_tile_overlap=dit_tile_overlap,
        prompt=prompt,
        negative_prompt=negative_prompt,
        cfg_scale=cfg_scale,
        progress_callback=progress_callback,
        cache_model=True,
    )
    t_dit_end = time.time()
    vram_dit = _peak_vram_gb()

    torch.cuda.empty_cache()

    # Phase 3 -- VAE Decode
    # cache_model=True keeps runner.vae alive for subsequent chunks
    t_dec_start = time.time()
    ctx = decode_whole_latent(runner, ctx, debug, progress_callback=progress_callback, cache_model=True)
    t_dec_end = time.time()
    vram_decode = getattr(runner, '_trt_vram_decode', None) or _peak_vram_gb()

    chunk_timings = {
        "encode": t_enc_end - t_enc_start,
        "dit": t_dit_end - t_dit_start,
        "decode": t_dec_end - t_dec_start,
        "vram_encode": vram_encode,
        "vram_dit": vram_dit,
        "vram_decode": vram_decode,
    }

    # ── Streaming decode path ──
    # When _decode_meta is in ctx, the decoded video is still on disk
    # as tile memmaps.  Return metadata instead of full tensors so the caller
    # can do streaming assembly + post-process in temporal batches.
    _dm = ctx.pop('_decode_meta', None)
    if _dm is not None:
        _dm['vpp'] = ctx.get('vpp')
        return _dm, None, chunk_timings

    # ── Legacy path: materialise full tensors ──
    decoded = ctx['decoded_video']
    if decoded.is_cuda:
        decoded = decoded.cpu()

    _vpp = ctx.get('vpp')
    if _vpp is not None and _vpp.memmap is not None and 'transformed_video' not in ctx:
        tv = _vpp.read_reference(as_cthw=True)
        if _vpp.needs_resize_back:
            tv = _vpp.resize_back(tv)
        ctx['transformed_video'] = tv
        del tv
        _vpp.cleanup_memmap()

    transformed = ctx.get('transformed_video')
    if transformed is not None:
        transformed = transformed.cpu()

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
    vpp: Optional['VideoPrePostProcess'] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Post-process decoded frames (color correct + denormalize + sharpen).

    Delegates to ``vpp.color_correct_and_finalize()`` when available.

    Args:
        decoded:          ``[T, H, W, C]`` in ``[-1, 1]`` on **CPU**.
        transformed:      ``[C, T, H, W]`` in ``[-1, 1]`` on **CPU** (reference).
        color_correction: Color-correction method name.
        usm_weight, usm_radius, usm_threshold: USM params.
        device:           GPU device string.
        vpp:              VideoPrePostProcess facade.

    Returns:
        When *vpp* is set: ``np.ndarray [T, H, W, C]`` uint8/uint16 pixel values.
        Fallback:          ``Tensor [T, H, W, C]`` float16 ``[0, 1]`` on CPU.
    """
    if vpp is not None:
        vpp.configure(color_method=color_correction,
                      usm_weight=usm_weight, usm_radius=usm_radius,
                      usm_threshold=usm_threshold)
        return vpp.color_correct_and_finalize(decoded, reference=transformed, device=device)

    video_out = decoded.clamp(-1, 1).mul(0.5).add(0.5).float().to(dtype=torch.float16)
    return video_out


_STREAM_PP_BATCH = 4  # frames per streaming post-process batch


def _detach_chunk_memmaps(decode_meta: dict) -> None:
    """Give a chunk's post-processing its own view of the VPP: the chunk's reference
    ("input") and tile memmaps move out of the shared registry into a shallow copy, so
    the next chunk can ingest and decode (same names) while this one is post-processed;
    the copy's cleanup deletes exactly this chunk's files."""
    import copy
    vpp = decode_meta.get('vpp')
    if vpp is None:
        return
    own = copy.copy(vpp)
    own._memmaps = {}
    own._writer = None      # the output writer stays the pipeline's: the copy's cleanup (also run by
                            # its __del__) must only delete this chunk's memmaps, never close the writer
    if vpp._memmaps.get("input") is not None:
        own._memmaps["input"] = vpp._memmaps.pop("input")
    for info in decode_meta.get('tile_infos', []):
        name = info.get("vpp_name")
        mm = vpp._memmaps.get(name) if name else None
        if mm is not None and getattr(mm, "path", None) == info.get("path"):
            own._memmaps[name] = vpp._memmaps.pop(name)
    decode_meta['vpp'] = own


def _streaming_postprocess_chunk(
    decode_meta: dict,
    ci: int,
    num_chunks: int,
    is_last: bool,
    overlap: int,
    extra_skip_val: int,
    overlap_dec: Optional[torch.Tensor],
    overlap_trans: Optional[torch.Tensor],
    pp_func,
    debug,
    blend_mode: str = "hann",
    vae_tile_blend: str = "cosine",
):
    """Streaming assembly + post-process for a single decode chunk.

    Instead of materialising the full decoded [T,H,W,C] + transformed
    [C,T,H,W] tensors (~12 GB for 4K 121 frames), this reads from tile
    memmaps and the transform memmap in temporal batches of
    ``_STREAM_PP_BATCH`` frames.

    Returns:
        (overlap_dec, overlap_trans) — the overlap tail to hold for the
        next chunk, or (None, None) if this is the last chunk.
    """
    import numpy as np
    from ..video_utils.tile_assembly import (
        assemble_tiles_from_disk_range, assemble_tiles_from_disk_range_gpu, cleanup_tile_files,
    )

    tile_infos = decode_meta['tile_infos']
    pixel_H = decode_meta['pixel_H']
    pixel_W = decode_meta['pixel_W']
    T_pix = decode_meta['T_pix']
    _vpp = decode_meta.get('vpp')
    _has_transform_ref = (_vpp is not None and _vpp.get_memmap("input") is not None)

    _gpu_assembly = torch.cuda.is_available() and os.environ.get("POSTPROCESS_ON_GPU", "1") == "1"

    def _read_decoded_range(start, count):
        """Assemble spatial tiles for frames [start : start+count]."""
        count = min(count, T_pix - start)
        if _gpu_assembly:                       # same blending on the GPU; stays there for post-processing
            dec = assemble_tiles_from_disk_range_gpu(
                tile_infos, pixel_H, pixel_W, start, count,
                blend_mode=vae_tile_blend,
            )
            if _vpp is not None and _vpp.needs_resize_back:
                dec = _vpp.resize_back(dec)
            return dec
        dec_np = assemble_tiles_from_disk_range(
            tile_infos, pixel_H, pixel_W, start, count,
            blend_mode=vae_tile_blend,
        )
        dec = torch.from_numpy(dec_np)  # [count, H, W, 3] fp16
        if _vpp is not None and _vpp.needs_resize_back:
            dec = _vpp.resize_back(dec)
        return dec

    def _read_transform_range(start, count):
        """Read colour-correction reference for frames [start : start+count]."""
        if not _has_transform_ref:
            return None
        # with GPU post-processing the integer frames go to the GPU and are converted there
        # (same float32 steps, identical values): half the transfer, no CPU float temporaries
        t = _vpp.read_reference(start, count, as_cthw=True, device="cuda" if _gpu_assembly else None)
        if _vpp.needs_resize_back:
            t = _vpp.resize_back(t)
        return t

    def _read_tail(start, count):
        """Overlap tail for the next chunk, in CPU memory (no GPU memory held across chunks),
        filled a batch at a time so no full-size temporaries are allocated."""
        dec = trans = None
        for fs in range(start, start + count, _STREAM_PP_BATCH):
            cnt = min(_STREAM_PP_BATCH, start + count - fs)
            d = _read_decoded_range(fs, cnt)
            t = _read_transform_range(fs, cnt)
            if dec is None:
                dec = torch.empty((count,) + tuple(d.shape[1:]), dtype=d.dtype)
                if t is not None:
                    trans = torch.empty((t.shape[0], count) + tuple(t.shape[2:]), dtype=t.dtype)
            dec[fs - start:fs - start + cnt] = d
            if t is not None:
                trans[:, fs - start:fs - start + cnt] = t
            del d, t
        return dec, trans

    def _cleanup_all():
        """Clean up tile files."""
        cleanup_tile_files(tile_infos, vpp=_vpp)

    # Materialise the full decoded + transformed only for the overlap
    # region (small: ≤21 frames for 4K).  Process the rest in streaming
    # batches via pp_func.

    skip = extra_skip_val
    T_effective = T_pix - skip

    if ci == 0:
        if num_chunks == 1:
            # Single chunk — stream everything through pp_func. The next batch's colour
            # reference is read on a background thread while this batch is processed.
            from concurrent.futures import ThreadPoolExecutor
            starts = list(range(0, T_pix, _STREAM_PP_BATCH))
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pp_reference") as ref_pool:
                nxt = ref_pool.submit(_read_transform_range, 0, min(_STREAM_PP_BATCH, T_pix)) if starts else None
                for k, fs in enumerate(starts):
                    cnt = min(_STREAM_PP_BATCH, T_pix - fs)
                    trans_b = nxt.result()
                    if k + 1 < len(starts):
                        f2 = starts[k + 1]
                        nxt = ref_pool.submit(_read_transform_range, f2, min(_STREAM_PP_BATCH, T_pix - f2))
                    dec_b = _read_decoded_range(fs, cnt)
                    pp_func(dec_b, trans_b)
                    del dec_b, trans_b
            _cleanup_all()
            return None, None
        else:
            # First of multiple chunks — stream non-overlap, hold overlap.
            safe_end = T_pix - overlap
            for fs in range(0, safe_end, _STREAM_PP_BATCH):
                cnt = min(_STREAM_PP_BATCH, safe_end - fs)
                dec_b = _read_decoded_range(fs, cnt)
                trans_b = _read_transform_range(fs, cnt)
                pp_func(dec_b, trans_b)
                del dec_b, trans_b
            new_ov_dec, new_ov_trans = _read_tail(safe_end, overlap)
            _cleanup_all()
            return new_ov_dec, new_ov_trans
    else:
        # Subsequent chunk — handle skip + overlap blending.
        actual_overlap = min(overlap, T_effective)
        if overlap_dec is not None:
            actual_overlap = min(actual_overlap, overlap_dec.shape[0])

        if actual_overlap > 0 and overlap_dec is not None:
            # blend + post-process the overlap a batch at a time (per-frame weights sliced to
            # match; colour correction is per frame), so no overlap-sized temporaries
            for fs in range(0, actual_overlap, _STREAM_PP_BATCH):
                fr = slice(fs, min(fs + _STREAM_PP_BATCH, actual_overlap))
                cnt = fr.stop - fr.start
                cur_head_dec = _read_decoded_range(skip + fs, cnt)
                cur_head_trans = _read_transform_range(skip + fs, cnt)
                blended_dec = _blend_overlapping_frames(
                    overlap_dec[fr].to(cur_head_dec.device), cur_head_dec, actual_overlap,
                    mode=blend_mode, frames=fr,
                )
                blended_trans = None
                if overlap_trans is not None and cur_head_trans is not None:
                    blended_trans = _blend_overlapping_transformed(
                        overlap_trans[:, fr].to(cur_head_trans.device), cur_head_trans, actual_overlap,
                        mode=blend_mode, frames=fr,
                    )
                pp_func(blended_dec, blended_trans)
                del blended_dec, blended_trans, cur_head_dec, cur_head_trans
            debug.log(
                f"Blended {actual_overlap} overlapping frames",
                category="video", indent_level=1,
            )

        # Stream the non-overlap remainder.
        rest_start = skip + actual_overlap
        rest_end = T_pix
        if not is_last and overlap > 0:
            rest_end = T_pix - overlap

        for fs in range(rest_start, rest_end, _STREAM_PP_BATCH):
            cnt = min(_STREAM_PP_BATCH, rest_end - fs)
            dec_b = _read_decoded_range(fs, cnt)
            trans_b = _read_transform_range(fs, cnt)
            pp_func(dec_b, trans_b)
            del dec_b, trans_b

        if is_last or overlap == 0:
            # Stream any remaining tail.
            if rest_end < T_pix:
                for fs in range(rest_end, T_pix, _STREAM_PP_BATCH):
                    cnt = min(_STREAM_PP_BATCH, T_pix - fs)
                    dec_b = _read_decoded_range(fs, cnt)
                    trans_b = _read_transform_range(fs, cnt)
                    pp_func(dec_b, trans_b)
                    del dec_b, trans_b
            _cleanup_all()
            return None, None
        else:
            new_ov_dec, new_ov_trans = _read_tail(T_pix - overlap, overlap)
            _cleanup_all()
            return new_ov_dec, new_ov_trans


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
    shift_back: bool = True,
    progress_fn: Optional[Callable[[float, str], None]] = None,
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
        ``output_callback(frames)`` (``[N, H, W, C]`` numpy uint8/uint16 pixels).
        Only the overlap buffer is kept in RAM.  Returns ``(None, timings)``.

    **Accumulate mode** (``output_callback is None``):
        All frames are accumulated in a CPU buffer, post-processed as one,
        and returned as ``(result, timings)``.

    Args:
        runner:             VideoDiffusionInfer (loaded DiT + VAE).
        ctx:                Shared generation context dict.
        frames:             Input frames -- ``np.ndarray [T, H, W, C]`` raw
                            uint8/uint16, a lazy reader with ``.shape`` and
                            slice support, or legacy ``torch.Tensor``.
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

        # Shift-back: if the last chunk's frame count doesn't satisfy 4n+1
        # (required by the temporal VAE), shift its start backwards by the
        # minimum amount needed.  The extra overlap with the previous chunk
        # is handled by discarding redundant leading output frames, then
        # blending the standard overlap.
        if len(read_windows) > 1:
            last_start, last_end = read_windows[-1]
            last_len = last_end - last_start
            remainder = (last_len - 1) % 4
            if remainder != 0:
                pad_needed = 4 - remainder
                new_start = max(0, last_start - pad_needed)
                logger.debug(
                    "Shift-back: last chunk [%d, %d) (%d frames) -> [%d, %d) (%d frames, 4n+1)",
                    last_start, last_end, last_len,
                    new_start, last_end, last_end - new_start,
                )
                read_windows[-1] = (new_start, last_end)

    num_chunks = len(read_windows)

    # For each consecutive chunk pair, compute how many decoded output frames
    # from the shifted-back chunk to skip (they were already streamed).
    # extra_skip[i] applies to chunk i and equals the overlap with the
    # previous chunk MINUS the standard blend overlap.
    extra_skip = [0] * num_chunks
    if num_chunks > 1:
        for i in range(1, num_chunks):
            actual_pair_overlap = read_windows[i - 1][1] - read_windows[i][0]
            extra_skip[i] = max(0, actual_pair_overlap - overlap)

    # Auto-determine spatial DiT tiling from target output resolution
    if dit_tiles is None:
        _, in_h, in_w, _ = frames.shape
        dit_tiles = _compute_dit_tiles(in_h, in_w, resolution)

    logger.debug(
        "Pixel-space chunking: %d chunk(s), batch_size=%d, temporal_overlap=%d, step=%d, dit_tiles=%s, streaming=%s",
        num_chunks, chunk_size, overlap, step, dit_tiles, streaming,
    )
    if any(s > 0 for s in extra_skip):
        logger.debug("Shift-back extra_skip per chunk: %s", extra_skip)

    # --- Progress helpers ---
    # Maps per-phase callbacks to a global 0.0 – 0.92 range.
    # Within each chunk: encode=10%, enhance=70%, decode=10%, post=10%.
    _PROGRESS_CEIL = 0.95
    _PHASE_WEIGHTS = (0.20, 0.20, 0.50, 0.10)  # encode, enhance, decode, post

    def _make_chunk_progress(chunk_idx: int):
        """Return a per-phase progress_callback for the given chunk."""
        if progress_fn is None:
            return None
        band = _PROGRESS_CEIL / max(num_chunks, 1)
        base = band * chunk_idx
        w_enc, w_dit, _, _ = _PHASE_WEIGHTS

        def _phase_cb(step, total, n_frames, message):
            if "Encoding" in message:
                frac = (step / max(total, 1)) * w_enc
            elif "Upscaling" in message:
                frac = w_enc + (step / max(total, 1)) * w_dit
            elif "Decoding" in message:
                frac = w_enc + w_dit + (step / max(total, 1)) * _PHASE_WEIGHTS[2]
            else:
                frac = w_enc + w_dit + _PHASE_WEIGHTS[2]
            progress_fn(base + frac * band, message)
        return _phase_cb

    def _report_post(chunk_idx: int):
        """Report post-processing progress for the given chunk."""
        if progress_fn is None:
            return
        band = _PROGRESS_CEIL / max(num_chunks, 1)
        base = band * chunk_idx
        frac = sum(_PHASE_WEIGHTS[:3])
        progress_fn(base + frac * band, "Post-processing")

    def _report_chunk_done(chunk_idx: int):
        """Report that a chunk is fully complete."""
        if progress_fn is None:
            return
        band = _PROGRESS_CEIL / max(num_chunks, 1)
        progress_fn(band * (chunk_idx + 1), "Processing")

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
            vpp=ctx.get('vpp'),
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
        from concurrent.futures import ThreadPoolExecutor
        _post_pool = (ThreadPoolExecutor(max_workers=1)
                      if os.environ.get("POSTPROCESS_PIPELINE", "1") == "1" else None)
        _post_pending = []
        _pp_state = {"overlap": (None, None), "busy": 0.0}

        for ci, (ws, we) in enumerate(read_windows):
            chunk_len = min(we, T_total) - ws

            if chunk_len == 0:
                logger.warning(
                    "Chunk %d/%d: no readable frames at [%d, %d), "
                    "stopping (video shorter than reported)",
                    ci + 1, num_chunks, ws, we,
                )
                break

            # Build a reader_fn that reads from the original frames
            # with offsets relative to this chunk's window.
            _ws = ws
            def _chunk_reader(start, count, _base=_ws):
                return frames[_base + start : _base + start + count]

            # Tell the reader to preserve overlap frames needed by the
            # next chunk.  Without this, batch reads inside VPP.ingest
            # would trim buffer entries that fall between the current
            # read head and the next chunk's start, making them
            # unrecoverable.  On the last chunk, allow full trimming.
            if hasattr(frames, '_trim_floor'):
                if ci + 1 < num_chunks:
                    _skip = getattr(frames, 'skip_frames', 0)
                    frames._trim_floor = _skip + read_windows[ci + 1][0]
                else:
                    frames._trim_floor = 0

            # Free reader buffer entries before the current chunk's start.
            # With lazy reading (reader_fn batches), we can only free frames
            # that precede ws — the current chunk still needs [ws, we).
            # The reader's _trim_floor prevents batch reads from evicting
            # overlap frames needed by the next chunk.
            if hasattr(frames, '_buffer') and isinstance(frames._buffer, dict):
                _skip = getattr(frames, 'skip_frames', 0)
                _freed = 0
                for k in list(frames._buffer):
                    if k < _skip + ws:
                        del frames._buffer[k]
                        _freed += 1
                if _freed:
                    logger.info("Freed %d reader buffer entries (< frame %d)", _freed, ws)

            video_ended_early = chunk_len < (we - ws)
            if video_ended_early:
                logger.warning(
                    "Chunk %d/%d: got %d frames instead of %d at [%d, %d), "
                    "video ended early — treating as last chunk",
                    ci + 1, num_chunks, chunk_len, we - ws, ws, we,
                )

            logger.debug(
                "Chunk %d/%d: input frames [%d, %d) (%d frames)",
                ci + 1, num_chunks, ws, we, chunk_len,
            )

            is_last = (ci == num_chunks - 1) or video_ended_early

            t0 = time.time()
            decoded, transformed, chunk_timings = process_pix_chunk(
                runner, ctx, _chunk_reader, chunk_len, debug,
                resolution, max_resolution, seed,
                latent_noise_scale, shared_noise,
                dit_tiles, dit_tile_overlap,
                prompt, negative_prompt, cfg_scale,
                progress_callback=_make_chunk_progress(ci),
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
            _report_post(ci)
            t_pp0 = time.time()

            # ── Streaming decode path ──
            # When decoded is a dict, the pixel data is still on disk as
            # tile memmaps.  Assemble + post-process in temporal batches
            # of _STREAM_PP_BATCH frames to avoid the 12 GB peak.
            if isinstance(decoded, dict):
                # Post-processing of this chunk runs on a background worker while the next
                # chunk encodes / runs the DiT / decodes. The worker owns this chunk's memmaps
                # (detached from the shared registry: the next chunk's ingest and tiles reuse
                # the names) and passes the overlap tail from one chunk's job to the next.
                _detach_chunk_memmaps(decoded)
                if overlap_dec is not None and _pp_state["overlap"] == (None, None):
                    _pp_state["overlap"] = (overlap_dec, overlap_trans)
                overlap_dec = overlap_trans = None

                def _post_job(meta=decoded, ci=ci, is_last=is_last, skip=extra_skip[ci]):
                    tj0 = time.time()
                    ov_dec, ov_trans = _pp_state["overlap"]
                    _pp_state["overlap"] = _streaming_postprocess_chunk(
                        meta, ci, num_chunks, is_last, overlap,
                        skip, ov_dec, ov_trans,
                        _pp, debug,
                        blend_mode=ctx.get('_temporal_blend', 'hann'),
                        vae_tile_blend=ctx.get('_vae_tile_blend', 'cosine'),
                    )
                    _pp_state["busy"] += time.time() - tj0

                if _post_pool is None:
                    _post_job()
                else:
                    _post_pending.append(_post_pool.submit(_post_job))
                    for f in _post_pending:              # a failed job stops the pipeline now
                        if f.done():
                            f.result()
                decoded = transformed = None
            else:
                # ── Legacy tensor path (non-hybrid) ──
                if ci == 0:
                    if num_chunks == 1:
                        _pp(decoded, transformed)
                    else:
                        if overlap > 0 and decoded.shape[0] > overlap:
                            safe_dec = decoded[:-overlap]
                            safe_trans = transformed[:, :-overlap] if transformed is not None else None
                            _pp(safe_dec, safe_trans)
                            overlap_dec = decoded[-overlap:].clone()
                            overlap_trans = transformed[:, -overlap:].clone() if transformed is not None else None
                        elif overlap > 0:
                            overlap_dec = decoded.clone()
                            overlap_trans = transformed.clone() if transformed is not None else None
                        else:
                            _pp(decoded, transformed)
                else:
                    skip = extra_skip[ci]
                    if skip > 0:
                        logger.debug(
                            "Shift-back: discarding %d leading decoded frames from chunk %d",
                            skip, ci + 1,
                        )
                        decoded = decoded[skip:]
                        if transformed is not None:
                            transformed = transformed[:, skip:]

                    actual_overlap = min(overlap, decoded.shape[0])
                    if overlap_dec is not None:
                        actual_overlap = min(actual_overlap, overlap_dec.shape[0])

                    _tb = ctx.get('_temporal_blend', 'hann')
                    if actual_overlap > 0 and overlap_dec is not None:
                        blended_dec = _blend_overlapping_frames(
                            overlap_dec, decoded[:actual_overlap], actual_overlap, mode=_tb,
                        )
                        blended_trans = None
                        if overlap_trans is not None and transformed is not None:
                            blended_trans_raw = _blend_overlapping_transformed(
                                overlap_trans, transformed[:, :actual_overlap], actual_overlap, mode=_tb,
                            )
                            blended_trans = blended_trans_raw
                        debug.log(
                            f"Blended {actual_overlap} overlapping frames",
                            category="video", indent_level=1,
                        )
                        _pp(blended_dec, blended_trans)
                        del blended_dec
                        if blended_trans is not None:
                            del blended_trans

                    rest_dec = decoded[actual_overlap:]
                    rest_trans = transformed[:, actual_overlap:] if transformed is not None else None

                    if is_last or overlap == 0:
                        _pp(rest_dec, rest_trans)
                        overlap_dec = None
                        overlap_trans = None
                    else:
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
            _report_chunk_done(ci)

            del decoded, transformed
            if not is_last:
                clear_memory(debug=debug, force=True)
            elif video_ended_early:
                break

        # Wait for the background post-processing (the part not hidden behind GPU work)
        t_wait0 = time.time()
        for f in _post_pending:
            f.result()
        if _post_pool is not None:
            _post_pool.shutdown()
            logger.info("[Profile] Background post-processing: %.2fs of work, %.2fs not overlapped",
                        _pp_state["busy"], time.time() - t_wait0)
        if _post_pending or _pp_state["overlap"] != (None, None):
            total_post_time += time.time() - t_wait0
            if overlap_dec is None:
                overlap_dec, overlap_trans = _pp_state["overlap"]

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
            chunk_len = min(we, T_total) - ws

            if chunk_len == 0:
                logger.warning(
                    "Chunk %d/%d: no readable frames at [%d, %d), "
                    "stopping (video shorter than reported)",
                    ci + 1, num_chunks, ws, we,
                )
                break

            _ws = ws
            def _chunk_reader(start, count, _base=_ws):
                return frames[_base + start : _base + start + count]

            if hasattr(frames, '_trim_floor'):
                if ci + 1 < num_chunks:
                    _skip = getattr(frames, 'skip_frames', 0)
                    frames._trim_floor = _skip + read_windows[ci + 1][0]
                else:
                    frames._trim_floor = 0

            video_ended_early = chunk_len < (we - ws)
            if video_ended_early:
                logger.warning(
                    "Chunk %d/%d: got %d frames instead of %d at [%d, %d), "
                    "video ended early — treating as last chunk",
                    ci + 1, num_chunks, chunk_len, we - ws, ws, we,
                )

            logger.debug(
                "Chunk %d/%d: input frames [%d, %d) (%d frames)",
                ci + 1, num_chunks, ws, we, chunk_len,
            )

            t0 = time.time()
            decoded, transformed, chunk_timings = process_pix_chunk(
                runner, ctx, _chunk_reader, chunk_len, debug,
                resolution, max_resolution, seed,
                latent_noise_scale, shared_noise,
                dit_tiles, dit_tile_overlap,
                prompt, negative_prompt, cfg_scale,
                progress_callback=_make_chunk_progress(ci),
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

            # For shift-back chunks, discard leading decoded frames that
            # overlap with already-written content from the previous chunk.
            skip = extra_skip[ci]
            if skip > 0:
                logger.debug(
                    "Shift-back: discarding %d leading decoded frames from chunk %d",
                    skip, ci + 1,
                )
                decoded = decoded[skip:]
                if transformed is not None:
                    transformed = transformed[:, skip:]

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

                _tb = ctx.get('_temporal_blend', 'hann')
                if actual_overlap > 0:
                    prev_tail = final_video[current_write_idx - actual_overlap:current_write_idx]
                    cur_head = decoded[:actual_overlap]
                    blended = _blend_overlapping_frames(prev_tail, cur_head, actual_overlap, mode=_tb)
                    final_video[current_write_idx - actual_overlap:current_write_idx] = blended

                    if final_transformed is not None and transformed is not None:
                        prev_tail_t = final_transformed[:, current_write_idx - actual_overlap:current_write_idx]
                        cur_head_t = transformed[:, :actual_overlap]
                        blended_t = _blend_overlapping_transformed(prev_tail_t, cur_head_t, actual_overlap, mode=_tb)
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

            del decoded, transformed
            _report_chunk_done(ci)
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
        if isinstance(result, torch.Tensor):
            if result.is_cuda:
                result = result.cpu()

        return result, timings
