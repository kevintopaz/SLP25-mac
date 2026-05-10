"""
Latent Domain Temporal Chunking - Generation Phases

Phase functions for the streaming / latent-domain temporal chunking pipeline.

Pipeline:
    Video → VAE Encode (whole) → [Latent chunks] → DiT → Merge latents → VAE Decode (whole)
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Tuple, List

from .generation_utils import (
    load_text_embeddings,
    script_directory,
    ensure_precision_initialized,
)
from .model_loader import materialize_model
from ..common.seed import set_seed
from ..optimization.memory_manager import (
    manage_tensor,
    manage_model_device,
    cleanup_dit,
    cleanup_text_embeddings,
    clear_memory
)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def _compute_dit_tile_params(
    latent_shape: Tuple[int, int, int, int],
    tiles: Tuple[int, int],
    min_overlap: float = 0.25,
    debug: Optional['Debug'] = None
) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int], Tuple[int, int]]:
    """Compute tile positions and sizes for tiled DiT processing.

    Latent shape is [T, H, W, C]. We tile spatially (H, W dimensions).
    Tile sizes are adjusted to be divisible by 4 for DiT compatibility.
    """
    T, H, W, C = latent_shape
    tile_rows, tile_cols = tiles

    min_overlap = max(0.0, min(0.5, min_overlap))

    base_tile_h = H // tile_rows
    base_tile_w = W // tile_cols

    overlap_h = max(4, int(base_tile_h * min_overlap))
    overlap_w = max(4, int(base_tile_w * min_overlap))

    tile_h = base_tile_h + overlap_h
    tile_w = base_tile_w + overlap_w

    tile_h = (tile_h + 3) // 4 * 4
    tile_w = (tile_w + 3) // 4 * 4

    tile_h = min(tile_h, H)
    tile_w = min(tile_w, W)

    # When only 1 tile along an axis the stride is unused, fall back to
    # tile size to avoid division-by-zero.
    stride_h = tile_h if tile_rows <= 1 else tile_h - ((tile_h * tile_rows - H) // (tile_rows - 1))
    stride_w = tile_w if tile_cols <= 1 else tile_w - ((tile_w * tile_cols - W) // (tile_cols - 1))

    tiles_info = []
    for row in range(tile_rows):
        for col in range(tile_cols):
            y_start = min(row * stride_h, max(0, H - tile_h))
            x_start = min(col * stride_w, max(0, W - tile_w))
            tiles_info.append((y_start, x_start, tile_h, tile_w))

    if debug:
        debug.log(f"Tiling: {len(tiles_info)} tiles, size=({tile_h}x{tile_w}), "
                 f"overlap=({overlap_h}x{overlap_w}), stride=({stride_h}x{stride_w}), full size=({H}x{W})",
                 category="dit", force=True, indent_level=1)

    return tiles_info


# Global text encoder cache to avoid reloading
_text_encoder_cache = {}


def encode_text_prompt(
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    debug: Optional['Debug'] = None,
    max_length: int = 256,
    target_dim: int = 5120
) -> torch.Tensor:
    """
    Encode a text prompt using T5-XXL encoder.
    
    Args:
        prompt: Text string to encode
        device: Target device for embeddings
        dtype: Target dtype for embeddings
        debug: Optional debug instance for logging
        max_length: Maximum token length (default: 256)
        target_dim: Target embedding dimension (default: 5120 for this model)
        
    Returns:
        Text embeddings tensor of shape [seq_len, target_dim]
    """
    global _text_encoder_cache
    
    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError:
        raise ImportError(
            "transformers library is required for text prompt encoding. "
            "Install with: pip install transformers sentencepiece"
        )
    
    # Load or get cached encoder
    cache_key = "t5_xxl"
    if cache_key not in _text_encoder_cache:
        if debug:
            debug.log("Loading text encoder...", category="text", force=True)
        
        model_name = "google/t5-v1_1-xxl"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        encoder = T5EncoderModel.from_pretrained(
            model_name, 
            torch_dtype=dtype
        )
        encoder.eval()
        
        # Create projection layer from T5 dim (4096) to target dim (5120)
        t5_dim = encoder.config.d_model  # 4096 for T5-XXL
        if t5_dim != target_dim:
            projection = nn.Linear(t5_dim, target_dim, bias=False)
            # Initialize with small weights
            nn.init.normal_(projection.weight, std=0.02)
            projection = projection.to(dtype)
        else:
            projection = None
        
        _text_encoder_cache[cache_key] = {
            'tokenizer': tokenizer,
            'encoder': encoder,
            'projection': projection
        }
        
        if debug:
            debug.log("Text encoder loaded", category="text")
    
    cache = _text_encoder_cache[cache_key]
    tokenizer = cache['tokenizer']
    encoder = cache['encoder']
    projection = cache['projection']
    
    # Move encoder to device
    encoder = encoder.to(device)
    if projection is not None:
        projection = projection.to(device)
    
    # Tokenize
    tokens = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    
    # Encode
    with torch.no_grad():
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[0]  # [seq_len, 4096]
        
        # Apply projection if needed
        if projection is not None:
            embeddings = projection(embeddings)
        
        # Mask padding tokens (optional - keep only actual tokens)
        # For now, keep full length for consistency with pre-computed embeddings
        
    embeddings = embeddings.to(dtype)
    
    if debug:
        debug.log(f"Encoded prompt: '{prompt[:50]}...' → shape {embeddings.shape}", category="text")
    
    return embeddings


def get_text_embeddings(
    ctx: Dict[str, Any],
    debug: 'Debug',
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None
) -> Dict[str, List[torch.Tensor]]:
    """
    Get text embeddings - either from user prompts or pre-computed files.
    
    Args:
        ctx: Generation context with device/dtype info
        debug: Debug instance
        prompt: Optional positive prompt (if None, uses pre-computed)
        negative_prompt: Optional negative prompt (if None, uses pre-computed or empty)
        
    Returns:
        Dict with 'texts_pos' and 'texts_neg' lists
    """
    device = ctx['dit_device']
    dtype = ctx['compute_dtype']
    
    if prompt is None:
        # Use pre-computed embeddings
        debug.log("Using pre-computed text embeddings", category="text")
        return load_text_embeddings(script_directory, device, dtype, debug)
    
    # Encode user prompts
    debug.log("Encoding custom text prompts...", category="text", force=True)
    
    pos_embeds = encode_text_prompt(prompt, device, dtype, debug)
    
    if negative_prompt:
        neg_embeds = encode_text_prompt(negative_prompt, device, dtype, debug)
    else:
        # Use empty/neutral negative prompt
        neg_embeds = encode_text_prompt("", device, dtype, debug)
    
    return {"texts_pos": [pos_embeds], "texts_neg": [neg_embeds]}


def encode_whole_video(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    reader_fn: Callable[[int, int], Any],
    frame_count: int,
    debug: 'Debug',
    resolution: int = 1080,
    max_resolution: int = 0,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Phase 1: Encode entire video to latent space at once.

    Parameters
    ----------
    reader_fn : callable(start, count) -> np.ndarray
        Returns ``[count, H, W, C]`` raw uint8/uint16 pixels for frames
        ``[start, start+count)``.
    frame_count : int
        Total number of input frames.
    """
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 1: Encoding video ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase1_encode_whole")

    T_input = frame_count

    ctx['total_frames'] = T_input

    debug.log(f"Input: frames streamed)", category="vae", force=True)

    # Materialize VAE if still on meta device
    if runner.vae and next(runner.vae.parameters()).device.type == 'meta':
        materialize_model(runner, "vae", ctx['vae_device'], runner.config, debug)

    manage_model_device(model=runner.vae, target_device=ctx['vae_device'],
                       model_name="VAE", debug=debug, runner=runner)

    # Temporal padding for 4n+1 constraint
    T_padded = T_input
    pad_count = 0
    if T_input % 4 != 1:
        T_padded = T_input + (4 - (T_input - 1) % 4)
        pad_count = T_padded - T_input

    ctx['original_temporal_frames'] = T_input

    vpp = ctx.get('vpp')

    # Ingest: resize + store to disk-backed memmap via VPP.
    # VPP probes dimensions internally; H_out/W_out come from vpp.padded_dims.
    vpp.ingest(
        reader_fn=reader_fn,
        total_frames=T_input,
        padded_frames=T_padded,
    )
    debug.log(f"Batch-transformed to disk cache: {vpp.memmap}",
              category="memory")

    import gc; gc.collect()

    _hybrid = getattr(runner, '_hybrid_encode_active', False)
    if _hybrid:
        video_transformed = torch.empty(1)
    else:
        video_transformed = vpp.read_reference(as_cthw=True)
        video_transformed = manage_tensor(
            tensor=video_transformed,
            target_device=ctx['vae_device'],
            tensor_name="video_for_encode",
            dtype=None,
            debug=debug,
            reason="VAE encoding"
        )

    if progress_callback:
        progress_callback(0, 1, ctx['total_frames'], "Phase 1: Encoding")
        def _enc_progress(done, total):
            progress_callback(done / max(total, 1), 1, ctx['total_frames'], "Phase 1: Encoding")
        runner._vae_progress = _enc_progress
    
    debug.start_timer("vae_encode")
    encode_input = [video_transformed]
    del video_transformed
    latents = runner.vae_encode(encode_input)
    del encode_input
    runner._vae_progress = None
    latent = latents[0]
    debug.end_timer("vae_encode", "VAE encode")
    
    debug.log(f"Encoded shape: {latent.shape}", category="vae")
    
    if ctx['tensor_offload_device'] is not None:
        latent = manage_tensor(
            tensor=latent,
            target_device=ctx['tensor_offload_device'],
            tensor_name="full_latent",
            dtype=ctx['compute_dtype'],
            debug=debug,
            reason="storing for DiT processing"
        )
    
    ctx['full_latent'] = latent
    
    ctx.pop('transformed_video', None)

    if ctx['vae_offload_device'] is not None:
        manage_model_device(model=runner.vae, target_device=ctx['vae_offload_device'],
                           model_name="VAE", debug=debug, reason="VAE offload", runner=runner)
    
    if progress_callback:
        progress_callback(1, 1, ctx['total_frames'], "Phase 1: Encoding")
    
    debug.end_timer("phase1_encode_whole", "Phase 1: VAE encoding complete")
    debug.log_memory_state("After Phase 1 (whole video encoding)", show_tensors=False, force=True)
    
    return ctx


def upscale_latent_chunked(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    debug: 'Debug',
    chunk_size: int = 9,
    overlap: int = 3,
    seed: int = 42,
    latent_noise_scale: float = 0.0,
    shared_noise: bool = False,
    dit_tiles: Optional[Tuple[int, int]] = None,
    dit_tile_overlap: float = 0.25,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    cache_model: bool = False,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    cfg_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Phase 2: Process latent through DiT in temporal chunks with overlap.
    
    Args:
        runner: VideoDiffusionInfer instance
        ctx: Generation context
        debug: Debug instance
        chunk_size: Temporal frames per chunk (default: 9)
        overlap: Overlap between chunks (default: 3)
        seed: Random seed
        latent_noise_scale: Noise scale for latent conditioning
        shared_noise: Use shared noise across frames
        dit_tiles: Optional (rows, cols) for spatial DiT tiling
        dit_tile_overlap: Overlap ratio for DiT tiles
        progress_callback: Optional progress callback
        cache_model: Keep model in memory after processing
        prompt: Optional custom positive text prompt (uses T5 encoder if provided)
        negative_prompt: Optional custom negative text prompt
    """
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 2: Enhancing video ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase2_upscale_chunked")
    
    if 'full_latent' not in ctx:
        raise ValueError("No full_latent in context. Run encode_whole_video first.")
    
    latent = ctx['full_latent']
    T, H, W, C = latent.shape
    
    debug.log(f"Latent shape: {latent.shape}, chunk_size={chunk_size}, overlap={overlap}",
             category="dit")
    
    # Get text embeddings - from user prompt or pre-computed
    if ctx.get('text_embeds') is None:
        ctx['text_embeds'] = get_text_embeddings(ctx, debug, prompt, negative_prompt)
    
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    debug.log(f"CFG scale: {cfg_scale}", category="dit")
    runner.configure_diffusion(device=ctx['dit_device'], dtype=ctx['compute_dtype'])
    
    if runner.dit and next(runner.dit.parameters()).device.type == 'meta':
        materialize_model(runner, "dit", ctx['dit_device'], runner.config, debug)
    
    ensure_precision_initialized(ctx, runner, debug)
    manage_model_device(model=runner.dit, target_device=ctx['dit_device'],
                       model_name="DiT", debug=debug, runner=runner)
    
    # Calculate temporal chunks with adaptive last-chunk shift-back.
    # If the naive last chunk would be shorter than chunk_size, shift its
    # start backwards so it is exactly chunk_size frames, enlarging the
    # overlap with the previous chunk instead of producing a runt.
    if chunk_size <= 0 or chunk_size >= T:
        chunks = [(0, T)]
    else:
        stride = chunk_size - overlap
        chunks = []
        start = 0
        while start < T:
            end = min(start + chunk_size, T)
            # Adaptive: if the remainder is too short, shift back
            if end - start < chunk_size and start > 0:
                start = max(0, T - chunk_size)
                end = T
            chunks.append((start, end))
            if end >= T:
                break
            start += stride
        # Deduplicate if shift-back produced the same window as the previous
        if len(chunks) >= 2 and chunks[-1] == chunks[-2]:
            chunks.pop()
    
    num_chunks = len(chunks)
    chunk_desc = ", ".join(f"[{s}:{e}]" for s, e in chunks)
    debug.log(f"Temporal chunking: {num_chunks} chunks  {chunk_desc}",
              category="dit")
    
    accum_device = ctx.get('tensor_offload_device') or latent.device
    result = torch.zeros((T, H, W, C), device=accum_device, dtype=latent.dtype)
    weight_sum = torch.zeros((T, 1, 1, 1), device=accum_device, dtype=latent.dtype)
    
    for chunk_idx, (start, end) in enumerate(chunks):
        chunk_len = end - start
        debug.log(f"Chunk {chunk_idx + 1}/{num_chunks}: frames {start}-{end} ({chunk_len} frames)",
                 category="dit")
        
        set_seed(seed)
        
        def _block_progress(blocks_done, blocks_total):
            if progress_callback:
                frac = chunk_idx + blocks_done / max(blocks_total, 1)
                progress_callback(frac, num_chunks, chunk_len, "Phase 2: Upscaling")

        if progress_callback:
            progress_callback(chunk_idx, num_chunks, chunk_len, "Phase 2: Upscaling")
        
        chunk_latent = latent[start:end].clone()
        chunk_latent = manage_tensor(
            tensor=chunk_latent,
            target_device=ctx['dit_device'],
            tensor_name=f"chunk_{chunk_idx}",
            dtype=ctx['compute_dtype'],
            debug=debug,
            reason="DiT processing",
            indent_level=1
        )
        
        if shared_noise and chunk_len > 1:
            single_noise = torch.randn(1, *chunk_latent.shape[1:],
                                       device=chunk_latent.device, dtype=ctx['compute_dtype'])
            base_noise = single_noise.expand(chunk_len, -1, -1, -1).contiguous()
            aug_noise = torch.randn(1, *chunk_latent.shape[1:],
                                   device=chunk_latent.device, dtype=ctx['compute_dtype'])
            aug_noise = aug_noise.expand(chunk_len, -1, -1, -1).contiguous()
            del single_noise
        else:
            base_noise = torch.randn_like(chunk_latent, dtype=ctx['compute_dtype'])
            aug_noise = torch.randn_like(base_noise)
        
        noises = [base_noise]
        aug_noises = [base_noise * 0.1 + aug_noise * 0.05]
        del aug_noise
        
        def _add_noise(x, aug):
            if latent_noise_scale == 0.0:
                return x
            t = torch.tensor([1000.0], device=ctx['dit_device'], dtype=ctx['compute_dtype']) * latent_noise_scale
            shape = torch.tensor(x.shape[1:], device=ctx['dit_device'])[None]
            t = runner.timestep_transform(t, shape)
            return runner.schedule.forward(x, aug, t)
        
        condition = runner.get_condition(
            noises[0], task="sr",
            latent_blur=_add_noise(chunk_latent, aug_noises[0])
        )
        conditions = [condition]
        
        # Get DiT dtype for autocast decision
        dit_model = runner.dit.dit_model if hasattr(runner.dit, 'dit_model') else runner.dit
        try:
            dit_dtype = next(dit_model.parameters()).dtype
        except StopIteration:
            dit_dtype = ctx['compute_dtype']
        
        # Use autocast if DiT dtype differs from compute dtype (e.g., fp16 model with bf16 compute)
        use_autocast = dit_dtype != ctx['compute_dtype'] and ctx['dit_device'].type != 'mps'
        
        debug.start_timer(f"dit_chunk_{chunk_idx}")
        _dit = getattr(runner, 'dit', None)
        if _dit is not None and progress_callback:
            _dit._progress_callback = _block_progress
        with torch.no_grad():
            if use_autocast:
                with torch.autocast(ctx['dit_device'].type, ctx['compute_dtype'], enabled=True):
                    if dit_tiles is not None:
                        upscaled = _tiled_dit_inference(
                            runner, chunk_latent, noises, conditions,
                            ctx['text_embeds'], ctx, dit_tiles, dit_tile_overlap, debug,
                            progress_callback=_block_progress,
                        )
                    else:
                        upscaled = runner.inference(
                            noises=noises, conditions=conditions,
                            **ctx['text_embeds']
                        )
            else:
                if dit_tiles is not None:
                    upscaled = _tiled_dit_inference(
                        runner, chunk_latent, noises, conditions,
                        ctx['text_embeds'], ctx, dit_tiles, dit_tile_overlap, debug,
                        progress_callback=_block_progress,
                    )
                else:
                    upscaled = runner.inference(
                        noises=noises, conditions=conditions,
                        **ctx['text_embeds']
                    )
        if _dit is not None:
            _dit._progress_callback = None
        debug.end_timer(f"dit_chunk_{chunk_idx}", f"DiT chunk {chunk_idx + 1}")
        
        upscaled_chunk = upscaled[0]
        if accum_device != upscaled_chunk.device:
            upscaled_chunk = upscaled_chunk.to(accum_device)
        
        # Compute actual overlaps with adjacent chunks (may differ from
        # nominal `overlap` when the last chunk was shifted back).
        chunk_len = end - start
        if chunk_idx > 0:
            prev_end = chunks[chunk_idx - 1][1]
            actual_ov_start = max(0, prev_end - start)
        else:
            actual_ov_start = 0
        if chunk_idx < num_chunks - 1:
            next_start = chunks[chunk_idx + 1][0]
            actual_ov_end = max(0, end - next_start)
        else:
            actual_ov_end = 0

        temporal_weight = torch.ones(chunk_len, device=accum_device, dtype=latent.dtype)
        _ltb = ctx.get('_latent_temporal_blend', 'cosine')

        if actual_ov_start > 0:
            ramp_len = min(actual_ov_start, chunk_len)
            t = torch.linspace(0, math.pi / 2, ramp_len, device=accum_device, dtype=latent.dtype)
            if _ltb == "cosine":
                temporal_weight[:ramp_len] = torch.sin(t)
            else:
                temporal_weight[:ramp_len] = torch.linspace(0, 1, ramp_len, device=accum_device, dtype=latent.dtype)

        if actual_ov_end > 0:
            ramp_len = min(actual_ov_end, chunk_len)
            if _ltb == "cosine":
                t = torch.linspace(0, math.pi / 2, ramp_len, device=accum_device, dtype=latent.dtype)
                temporal_weight[-ramp_len:] = temporal_weight[-ramp_len:] * torch.cos(t)
            else:
                temporal_weight[-ramp_len:] = temporal_weight[-ramp_len:] * torch.linspace(1, 0, ramp_len, device=accum_device, dtype=latent.dtype)
        
        temporal_weight = temporal_weight.view(-1, 1, 1, 1)
        
        result[start:end] += upscaled_chunk * temporal_weight
        weight_sum[start:end] += temporal_weight
        
        del chunk_latent, noises, aug_noises, conditions, condition, base_noise, upscaled, upscaled_chunk, temporal_weight
        
        if chunk_idx < num_chunks - 1:
            clear_memory(debug=debug, force=True)
    
    result = result / weight_sum.clamp(min=1e-6)
    ctx['upscaled_latent'] = result
    del ctx['full_latent']
    del weight_sum
    
    if progress_callback:
        progress_callback(num_chunks, num_chunks, T, "Phase 2: Upscaling")
    
    cleanup_dit(runner=runner, debug=debug, cache_model=cache_model)
    cleanup_text_embeddings(ctx, debug)
    
    debug.end_timer("phase2_upscale_chunked", "Phase 2: DiT upscaling complete")
    debug.log_memory_state("After Phase 2 (latent chunked upscaling)", show_tensors=False, force=True)
    
    return ctx


def _create_tile_weight_mask(
    tile_h: int, tile_w: int,
    y_start: int, x_start: int,
    full_h: int, full_w: int,
    overlap_h: int, overlap_w: int,
    device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Create a smooth weight mask for a tile with cosine ramps in overlap regions.
    
    Uses separate multiplicative weights for H and W dimensions.
    Each dimension ramps smoothly at tile boundaries that overlap with neighbors.
    """
    # Create 1D weight vectors for H and W dimensions
    weight_h = torch.ones(tile_h, device=device, dtype=dtype)
    weight_w = torch.ones(tile_w, device=device, dtype=dtype)
    
    # Vertical (H dimension) blending
    if overlap_h > 0:
        # Top edge: if tile doesn't start at y=0, it overlaps with tile above
        if y_start > 0:
            ramp_len = min(overlap_h, tile_h)
            # Cosine ramp: 0 at start -> 1 at end of overlap
            t = torch.linspace(0, math.pi / 2, ramp_len, device=device, dtype=dtype)
            weight_h[:ramp_len] = torch.sin(t)  # sin(0)=0 -> sin(pi/2)=1
        
        # Bottom edge: if tile doesn't end at full_h, it overlaps with tile below
        if y_start + tile_h < full_h:
            ramp_len = min(overlap_h, tile_h)
            # Cosine ramp: 1 at start of overlap -> 0 at end
            t = torch.linspace(0, math.pi / 2, ramp_len, device=device, dtype=dtype)
            ramp = torch.cos(t)  # cos(0)=1 -> cos(pi/2)=0
            # Multiply with existing weight (handles case where tile is smaller than 2*overlap)
            weight_h[-ramp_len:] = weight_h[-ramp_len:] * ramp
    
    # Horizontal (W dimension) blending
    if overlap_w > 0:
        # Left edge: if tile doesn't start at x=0, it overlaps with tile to the left
        if x_start > 0:
            ramp_len = min(overlap_w, tile_w)
            t = torch.linspace(0, math.pi / 2, ramp_len, device=device, dtype=dtype)
            weight_w[:ramp_len] = torch.sin(t)
        
        # Right edge: if tile doesn't end at full_w, it overlaps with tile to the right
        if x_start + tile_w < full_w:
            ramp_len = min(overlap_w, tile_w)
            t = torch.linspace(0, math.pi / 2, ramp_len, device=device, dtype=dtype)
            ramp = torch.cos(t)
            weight_w[-ramp_len:] = weight_w[-ramp_len:] * ramp
    
    # Create 2D weight mask: outer product of height and width weights
    # This naturally handles corners as the product of the two ramps
    # Shape: (tile_h, tile_w) -> (1, tile_h, tile_w, 1) for broadcasting with (T, H, W, C)
    weight_2d = weight_h.unsqueeze(1) * weight_w.unsqueeze(0)
    return weight_2d.unsqueeze(0).unsqueeze(-1)  # (1, tile_h, tile_w, 1)


def _get_tile_overlap(tiles_info, dit_tiles):
    """Compute spatial overlap in pixels from tile grid."""
    tile_rows, tile_cols = dit_tiles
    y0, x0, tile_h, tile_w = tiles_info[0]
    if tile_cols > 1:
        _, x1, _, _ = tiles_info[1]
        overlap_w = tile_w - (x1 - x0)
    else:
        overlap_w = 0
    if tile_rows > 1:
        y_next, _, _, _ = tiles_info[tile_cols]
        overlap_h = tile_h - (y_next - y0)
    else:
        overlap_h = 0
    return max(0, overlap_h), max(0, overlap_w)


def _blend_tiles(tiles_info, tile_outputs, T, H, W, C, overlap_h, overlap_w,
                 device, dtype, accum_device):
    """Weighted-blend tile outputs onto a full-resolution canvas."""
    result = torch.zeros((T, H, W, C), device=accum_device, dtype=dtype)
    weight_sum = torch.zeros((1, H, W, 1), device=accum_device, dtype=dtype)
    for (y_start, x_start, tile_h, tile_w), tile_out in zip(tiles_info, tile_outputs):
        weight_mask = _create_tile_weight_mask(
            tile_h, tile_w, y_start, x_start, H, W,
            overlap_h, overlap_w, tile_out.device, dtype,
        )
        tile_weighted = tile_out * weight_mask
        if accum_device != tile_weighted.device:
            tile_weighted = tile_weighted.to(accum_device)
            weight_mask = weight_mask.to(accum_device)
        result[:, y_start:y_start + tile_h, x_start:x_start + tile_w, :] += tile_weighted
        weight_sum[:, y_start:y_start + tile_h, x_start:x_start + tile_w, :] += weight_mask
        del tile_weighted, weight_mask
    result = result / weight_sum.clamp(min=1e-6)
    if result.device != device:
        result = result.to(device)
    return result


def _tiled_dit_block_first(
    runner, noises, conditions, text_embeds, ctx,
    tiles_info, T, H, W, C, overlap_h, overlap_w, debug,
    progress_callback=None,
):
    """Block-first tiled DiT: load each block once, process all tiles through it.

    Requires the underlying DiT to expose ``forward_tiles()``.
    Falls back to per-tile ``runner.inference()`` if not available.
    """
    from ..models.dit_3b import na

    device = noises[0].device
    dtype = noises[0].dtype
    accum_device = ctx.get('tensor_offload_device') or device
    num_tiles = len(tiles_info)

    dit_model = runner.dit.dit_model if hasattr(runner.dit, 'dit_model') else runner.dit

    # -- Prepare per-tile flattened inputs -----------------------------------
    tile_vids = []
    tile_vid_shapes = []
    tile_noises_flat = []

    for y_start, x_start, tile_h, tile_w in tiles_info:
        t_noise = noises[0][:, y_start:y_start + tile_h, x_start:x_start + tile_w, :]
        t_cond = conditions[0][:, y_start:y_start + tile_h, x_start:x_start + tile_w, :]
        flat_noise, shapes = na.flatten([t_noise])
        flat_cond, _ = na.flatten([t_cond])
        vid = torch.cat([flat_noise, flat_cond], dim=-1)
        tile_vids.append(vid)
        tile_vid_shapes.append(shapes)
        tile_noises_flat.append(flat_noise)

    # -- Shared text embeddings (flatten once) --------------------------------
    texts_pos = text_embeds['texts_pos']
    if isinstance(texts_pos[0], tuple):
        tp_list = []
        tp_shapes = []
        for pos in zip(*texts_pos):
            emb, shape = na.flatten(pos)
            tp_list.append(emb)
            tp_shapes.append(shape)
        txt_embed = tp_list
        txt_shape = tp_shapes
    else:
        txt_embed, txt_shape = na.flatten(texts_pos)

    # -- Timestep (use sampler's schedule) ------------------------------------
    timesteps = runner.sampler.timesteps.timesteps
    t_final = timesteps[-1]

    debug.log(f"Tiled enhancement: {num_tiles} tiles",
             category="dit", indent_level=1)

    # -- Block-first forward --------------------------------------------------
    dit_outputs = dit_model.forward_tiles(
        tile_vids, tile_vid_shapes, txt_embed, txt_shape, t_final,
        progress_callback=progress_callback,
    )

    # -- Sampler denoising per tile -------------------------------------------
    tile_results = []
    sampler = runner.sampler
    for i in range(num_tiles):
        pred = dit_outputs[i].vid_sample
        x_0, _ = sampler.schedule.convert_from_pred(
            pred, sampler.prediction_type, tile_noises_flat[i], t_final,
        )
        tile_out = na.unflatten(x_0, tile_vid_shapes[i])[0]
        tile_results.append(tile_out)
        del pred, x_0

    del dit_outputs, tile_vids, tile_vid_shapes, tile_noises_flat

    result = _blend_tiles(tiles_info, tile_results, T, H, W, C,
                          overlap_h, overlap_w, device, dtype, accum_device)
    debug.log(f"Merged {num_tiles} tiles", category="dit", indent_level=1)
    return [result]


def _tiled_dit_inference(
    runner, latent: torch.Tensor, noises: List[torch.Tensor],
    conditions: List[torch.Tensor], text_embeds: Dict[str, Any],
    ctx: Dict[str, Any], dit_tiles: Tuple[int, int],
    dit_tile_overlap: float, debug: 'Debug',
    progress_callback=None,
) -> List[torch.Tensor]:
    """Tiled spatial DiT inference for a single temporal chunk with smooth blending."""
    T, H, W, C = latent.shape
    tiles_info = _compute_dit_tile_params(latent.shape, dit_tiles, dit_tile_overlap, debug)
    
    if len(tiles_info) <= 1:
        return runner.inference(noises=noises, conditions=conditions, **text_embeds)
    
    device = latent.device
    dtype = latent.dtype
    accum_device = ctx.get('tensor_offload_device') or device
    
    overlap_h, overlap_w = _get_tile_overlap(tiles_info, dit_tiles)
    
    debug.log(f"Tile blending: overlap=({overlap_h}x{overlap_w}), tile=({tiles_info[0][2]}x{tiles_info[0][3]})",
             category="dit", indent_level=1)
    
    # Block-first path: load each block once, process all tiles through it.
    # Holds num_tiles × per-block-activation simultaneously between blocks
    # — fast but high RAM peak. Tile-first (below) holds only one tile at
    # a time, ~4 GB cheaper at 4K, but re-loads each block per tile.
    # Override with MLX_DIT_TILE_FIRST=1 to force the low-RAM path on
    # systems where memory is tighter than disk bandwidth (e.g. <24 GB Macs).
    import os as _os
    _force_tile_first = _os.environ.get("MLX_DIT_TILE_FIRST", "0") == "1"
    dit_model = runner.dit.dit_model if hasattr(runner.dit, 'dit_model') else runner.dit
    if hasattr(dit_model, 'forward_tiles') and not _force_tile_first:
        return _tiled_dit_block_first(
            runner, noises, conditions, text_embeds, ctx,
            tiles_info, T, H, W, C, overlap_h, overlap_w, debug,
            progress_callback=progress_callback,
        )
    
    # Tile-first fallback: process each tile independently through all blocks.
    num_tiles = len(tiles_info)
    tile_outputs = []
    for tile_idx, (y_start, x_start, tile_h, tile_w) in enumerate(tiles_info):
        debug.log(f"  Tile {tile_idx + 1}/{num_tiles}: pos=({y_start},{x_start}), size=({tile_h}x{tile_w})",
                 category="dit", indent_level=2)
        
        tile_noises = [n[:, y_start:y_start + tile_h, x_start:x_start + tile_w, :] for n in noises]
        tile_conditions = [c[:, y_start:y_start + tile_h, x_start:x_start + tile_w, :] for c in conditions]
        
        with torch.no_grad():
            tile_result = runner.inference(noises=tile_noises, conditions=tile_conditions, **text_embeds)
        
        tile_outputs.append(tile_result[0])
        del tile_noises, tile_conditions, tile_result
        if progress_callback is not None:
            progress_callback(tile_idx + 1, num_tiles)
    
    result = _blend_tiles(tiles_info, tile_outputs, T, H, W, C,
                          overlap_h, overlap_w, device, dtype, accum_device)
    debug.log(f"Merged {num_tiles} tiles", category="dit", indent_level=1)
    return [result]


def decode_whole_latent(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    debug: 'Debug',
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    cache_model: bool = False
) -> Dict[str, Any]:
    """
    Phase 3: Decode entire latent to video at once.
    """
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 3: Decoding video ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase3_decode_whole")
    
    if 'upscaled_latent' not in ctx:
        raise ValueError("No upscaled_latent in context. Run upscale_latent_chunked first.")
    
    latent = ctx['upscaled_latent']
    debug.log(f"Decoding shape: {latent.shape}", category="vae")
    
    manage_model_device(model=runner.vae, target_device=ctx['vae_device'],
                       model_name="VAE", debug=debug, runner=runner)
    
    latent = manage_tensor(
        tensor=latent,
        target_device=ctx['vae_device'],
        tensor_name="latent_for_decode",
        dtype=ctx['compute_dtype'],
        debug=debug,
        reason="VAE decoding"
    )
    
    if progress_callback:
        progress_callback(0, 1, latent.shape[0], "Phase 3: Decoding")
        def _dec_progress(done, total):
            progress_callback(done / max(total, 1), 1, latent.shape[0], "Phase 3: Decoding")
        runner._vae_progress = _dec_progress
    
    debug.start_timer("vae_decode")
    samples = runner.vae_decode([latent])
    runner._vae_progress = None
    sample = samples[0]
    debug.end_timer("vae_decode", "VAE decode")
    
    # VAE decode output is (C, T, H, W), convert to (T, H, W, C)
    debug.log(f"Decoded shape (CTHW): {sample.shape}", category="vae")
    sample = sample.permute(1, 2, 3, 0)  # (C, T, H, W) -> (T, H, W, C)
    
    # Trim to original frame count (remove padding added for 4n+1 constraint)
    T_original = ctx.get('original_temporal_frames', sample.shape[0])
    if sample.shape[0] > T_original:
        debug.log(f"Trimming temporal: {sample.shape[0]} → {T_original} frames", category="vae")
        sample = sample[:T_original]
    
    # Resize back to true target dims (undo divisibility rounding).
    vpp = ctx.get('vpp')
    if vpp is not None and vpp.needs_resize_back:
        true_h, true_w = vpp.true_dims
        current_h, current_w = sample.shape[1], sample.shape[2]
        if current_h != true_h or current_w != true_w:
            debug.log(f"Resize back: {current_w}x{current_h} → {true_w}x{true_h}",
                      category="vae")
            sample = vpp.resize_back(sample)
            if 'transformed_video' in ctx:
                ctx['transformed_video'] = vpp.resize_back(ctx['transformed_video'])
    
    ctx['decoded_video'] = sample
    del ctx['upscaled_latent']
    
    if not cache_model and ctx['vae_offload_device'] is not None:
        manage_model_device(model=runner.vae, target_device=ctx['vae_offload_device'],
                           model_name="VAE", debug=debug, reason="VAE offload", runner=runner)
    
    if progress_callback:
        progress_callback(1, 1, sample.shape[0], "Phase 3: Decoding")
    
    debug.end_timer("phase3_decode_whole", "Phase 3: VAE decoding complete")
    debug.log_memory_state("After Phase 3 (whole latent decoding)", show_tensors=False, force=True)
    
    return ctx


def postprocess_latent_output(
    ctx: Dict[str, Any],
    debug: 'Debug',
    color_correction: str = "lab",
    usm_weight: float = 0.0,
    usm_radius: int = 51,
    usm_threshold: int = 10,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Phase 4: Post-process decoded video with color correction and optional USM sharpening.
    
    Args:
        usm_weight: USM sharpening weight (0 = disabled, typical 0.3-1.0).
        usm_radius: Gaussian blur kernel size for USM (must be odd, default 51).
        usm_threshold: Pixel-difference threshold to avoid sharpening noise (default 10).
    """
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 4: Post-processing ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase4_postprocess")
    
    if 'decoded_video' not in ctx:
        raise ValueError("No decoded_video in context. Run decode_whole_latent first.")
    
    video = ctx['decoded_video']
    del ctx['decoded_video']
    T = video.shape[0]
    
    debug.log(f"Post-processing {T} frames, color_correction={color_correction}",
             category="color", force=True)
    
    if progress_callback:
        progress_callback(0, 1, T, "Phase 4: Post-processing")
    
    # Reload transformed_video from VPP's memmap for color correction.
    if color_correction != "none" and 'transformed_video' not in ctx:
        vpp = ctx.get('vpp')
        if vpp is not None and vpp.memmap is not None:
            tv = vpp.read_reference(as_cthw=True)
            if vpp.needs_resize_back:
                tv = vpp.resize_back(tv)
            ctx['transformed_video'] = tv
            del tv
            vpp.cleanup_memmap()
            debug.log("Reloaded transformed video from disk cache for color correction",
                      category="memory", force=True)

    vpp = ctx.get('vpp')
    reference = ctx.get('transformed_video')

    if vpp is not None:
        vpp.configure(color_method=color_correction,
                      usm_weight=usm_weight, usm_radius=usm_radius,
                      usm_threshold=usm_threshold)
        gpu_device = ('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')
        debug.log(f"Applying postprocess ({color_correction}, USM={usm_weight})",
                  category="color", force=True)
        video = vpp.color_correct_and_finalize(video, reference=reference, device=gpu_device)
    else:
        video.clamp_(-1, 1).mul_(0.5).add_(0.5)
        video = video.float().to(dtype=torch.float16).cpu()

    ctx['final_video'] = video
    
    if 'transformed_video' in ctx:
        del ctx['transformed_video']
    
    if progress_callback:
        progress_callback(1, 1, T, "Phase 4: Post-processing")
    
    debug.end_timer("phase4_postprocess", "Phase 4: Post-processing complete")
    
    return ctx
