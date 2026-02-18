"""
Latent Domain Temporal Chunking - Generation Phases

This module provides phase functions for latent-domain temporal chunking pipeline.
Used by inference_latent_chunked.py (mirrors how generation_phases.py is used by inference_cli.py).

Pipeline:
    Video → VAE Encode (whole) → Latent → [Chunk1, Chunk2, ...] → DiT → Merge latents → VAE Decode (whole)
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Tuple, List

from .generation_utils import (
    setup_video_transform,
    load_text_embeddings,
    script_directory,
    ensure_precision_initialized,
    pad_video_temporal
)
from .generation_phases import _compute_dit_tile_params
from .model_loader import materialize_model
from ..common.seed import set_seed
from ..optimization.memory_manager import (
    manage_tensor,
    manage_model_device,
    cleanup_dit,
    cleanup_text_embeddings,
    clear_memory
)


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
            debug.log("Loading T5-XXL text encoder...", category="text", force=True)
        
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
            debug.log(f"T5-XXL loaded (dim: {t5_dim} → {target_dim})", category="text")
    
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
        debug.log("Using pre-computed text embeddings", category="text", force=True)
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
    images: torch.Tensor,
    debug: 'Debug',
    resolution: int = 1080,
    max_resolution: int = 0,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Phase 1: Encode entire video to latent space at once.
    """
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 1: VAE Encode (whole video) ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase1_encode_whole")
    
    ctx['input_images'] = images
    ctx['total_frames'] = len(images)
    
    if 'true_target_dims' not in ctx:
        sample_frame = images[0].permute(2, 0, 1).unsqueeze(0)
        setup_video_transform(ctx, resolution, max_resolution, debug, sample_frame)
        del sample_frame
    else:
        setup_video_transform(ctx, resolution, max_resolution, debug)
    
    debug.log(f"Input frames: {images.shape}", category="vae", force=True)
    
    # Materialize VAE if still on meta device
    if runner.vae and next(runner.vae.parameters()).device.type == 'meta':
        materialize_model(runner, "vae", ctx['vae_device'], runner.config, debug)
    
    manage_model_device(model=runner.vae, target_device=ctx['vae_device'],
                       model_name="VAE", debug=debug, runner=runner)
    
    video = images.permute(0, 3, 1, 2)
    transform = ctx['video_transform']
    video_transformed = transform(video)
    
    debug.log(f"Transformed shape: {video_transformed.shape}", category="vae")
    ctx['transformed_video'] = video_transformed.cpu()
    
    # VAE requires temporal dimension to satisfy 4n+1 constraint
    # video_transformed is CTHW format, temporal_dim=1
    T_original = video_transformed.shape[1]
    ctx['original_temporal_frames'] = T_original
    
    if T_original % 4 != 1:
        video_transformed = pad_video_temporal(video_transformed, count=0, temporal_dim=1, prepend=False, debug=debug)
        T_padded = video_transformed.shape[1]
        debug.log(f"Padded temporal: {T_original} → {T_padded} frames (4n+1 constraint)", category="vae")
    
    video_transformed = manage_tensor(
        tensor=video_transformed,
        target_device=ctx['vae_device'],
        tensor_name="video_for_encode",
        dtype=ctx['compute_dtype'],
        debug=debug,
        reason="VAE encoding"
    )
    
    if progress_callback:
        progress_callback(0, 1, len(images), "Phase 1: Encoding")
    
    debug.start_timer("vae_encode")
    latents = runner.vae_encode([video_transformed])
    latent = latents[0]
    debug.end_timer("vae_encode", "VAE encode")
    
    debug.log(f"Encoded latent shape: {latent.shape}", category="vae", force=True)
    del video_transformed
    
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
    
    if ctx['vae_offload_device'] is not None:
        manage_model_device(model=runner.vae, target_device=ctx['vae_offload_device'],
                           model_name="VAE", debug=debug, reason="VAE offload", runner=runner)
    
    if progress_callback:
        progress_callback(1, 1, len(images), "Phase 1: Encoding")
    
    debug.end_timer("phase1_encode_whole", "Phase 1: VAE encoding complete")
    debug.log_memory_state("After Phase 1 (whole video encoding)", show_tensors=False)
    
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
    debug.log("━━━━━━━━ Phase 2: DiT Upscale (latent chunks) ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase2_upscale_chunked")
    
    if 'full_latent' not in ctx:
        raise ValueError("No full_latent in context. Run encode_whole_video first.")
    
    latent = ctx['full_latent']
    T, H, W, C = latent.shape
    
    debug.log(f"Latent shape: {latent.shape}, chunk_size={chunk_size}, overlap={overlap}",
             category="dit", force=True)
    
    # Get text embeddings - from user prompt or pre-computed
    if ctx.get('text_embeds') is None:
        ctx['text_embeds'] = get_text_embeddings(ctx, debug, prompt, negative_prompt)
    
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    debug.log(f"CFG scale: {cfg_scale}", category="dit", force=True)
    runner.configure_diffusion(device=ctx['dit_device'], dtype=ctx['compute_dtype'])
    
    if runner.dit and next(runner.dit.parameters()).device.type == 'meta':
        materialize_model(runner, "dit", ctx['dit_device'], runner.config, debug)
    
    ensure_precision_initialized(ctx, runner, debug)
    manage_model_device(model=runner.dit, target_device=ctx['dit_device'],
                       model_name="DiT", debug=debug, runner=runner)
    
    # Calculate temporal chunks
    if chunk_size <= 0 or chunk_size >= T:
        chunks = [(0, T)]
    else:
        stride = chunk_size - overlap
        chunks = []
        start = 0
        while start < T:
            end = min(start + chunk_size, T)
            chunks.append((start, end))
            if end >= T:
                break
            start += stride
    
    num_chunks = len(chunks)
    debug.log(f"Processing {num_chunks} temporal chunks", category="dit", force=True)
    
    accum_device = ctx.get('tensor_offload_device') or latent.device
    result = torch.zeros((T, H, W, C), device=accum_device, dtype=latent.dtype)
    weight_sum = torch.zeros((T, 1, 1, 1), device=accum_device, dtype=latent.dtype)
    
    for chunk_idx, (start, end) in enumerate(chunks):
        chunk_len = end - start
        debug.log(f"Chunk {chunk_idx + 1}/{num_chunks}: frames {start}-{end} ({chunk_len} frames)",
                 category="dit", force=True)
        
        set_seed(seed)
        
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
        with torch.no_grad():
            if use_autocast:
                with torch.autocast(ctx['dit_device'].type, ctx['compute_dtype'], enabled=True):
                    if dit_tiles is not None:
                        upscaled = _tiled_dit_inference(
                            runner, chunk_latent, noises, conditions,
                            ctx['text_embeds'], ctx, dit_tiles, dit_tile_overlap, debug
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
                        ctx['text_embeds'], ctx, dit_tiles, dit_tile_overlap, debug
                    )
                else:
                    upscaled = runner.inference(
                        noises=noises, conditions=conditions,
                        **ctx['text_embeds']
                    )
        debug.end_timer(f"dit_chunk_{chunk_idx}", f"DiT chunk {chunk_idx + 1}")
        
        upscaled_chunk = upscaled[0]
        if accum_device != upscaled_chunk.device:
            upscaled_chunk = upscaled_chunk.to(accum_device)
        
        # Create temporal weight with sin/cos ramps at chunk boundaries
        chunk_len = end - start
        temporal_weight = torch.ones(chunk_len, device=accum_device, dtype=latent.dtype)
        
        # Apply sin ramp at start (if not first chunk) - ramps from 0 to 1
        if start > 0 and overlap > 0:
            ramp_len = min(overlap, chunk_len)
            t = torch.linspace(0, math.pi / 2, ramp_len, device=accum_device, dtype=latent.dtype)
            temporal_weight[:ramp_len] = torch.sin(t)  # 0 -> 1
        
        # Apply cos ramp at end (if not last chunk) - ramps from 1 to 0
        if end < T and overlap > 0:
            ramp_len = min(overlap, chunk_len)
            t = torch.linspace(0, math.pi / 2, ramp_len, device=accum_device, dtype=latent.dtype)
            ramp = torch.cos(t)  # 1 -> 0
            # Multiply (don't replace) to handle small chunks correctly
            temporal_weight[-ramp_len:] = temporal_weight[-ramp_len:] * ramp
        
        # Reshape for broadcasting: (chunk_len,) -> (chunk_len, 1, 1, 1)
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
    debug.log_memory_state("After Phase 2 (latent chunked upscaling)", show_tensors=False)
    
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


def _tiled_dit_inference(
    runner, latent: torch.Tensor, noises: List[torch.Tensor],
    conditions: List[torch.Tensor], text_embeds: Dict[str, Any],
    ctx: Dict[str, Any], dit_tiles: Tuple[int, int],
    dit_tile_overlap: float, debug: 'Debug'
) -> List[torch.Tensor]:
    """Tiled spatial DiT inference for a single temporal chunk with smooth blending."""
    T, H, W, C = latent.shape
    tiles_info = _compute_dit_tile_params(latent.shape, dit_tiles, dit_tile_overlap, debug)
    
    if len(tiles_info) <= 1:
        return runner.inference(noises=noises, conditions=conditions, **text_embeds)
    
    device = latent.device
    dtype = latent.dtype
    accum_device = ctx.get('tensor_offload_device') or device
    
    # Calculate overlap from tile grid structure
    # tiles_info is arranged row-major: [(0,0), (0,1), ..., (1,0), (1,1), ...]
    tile_rows, tile_cols = dit_tiles
    y0, x0, tile_h, tile_w = tiles_info[0]
    
    # Get horizontal overlap (from tiles in same row, adjacent columns)
    if tile_cols > 1:
        y1, x1, _, _ = tiles_info[1]  # Second tile (same row, next column)
        stride_w = x1 - x0
        overlap_w = tile_w - stride_w
    else:
        overlap_w = 0
    
    # Get vertical overlap (from tiles in adjacent rows)
    if tile_rows > 1:
        y_next, x_next, _, _ = tiles_info[tile_cols]  # First tile of second row
        stride_h = y_next - y0
        overlap_h = tile_h - stride_h
    else:
        overlap_h = 0
    
    # Ensure non-negative
    overlap_h = max(0, overlap_h)
    overlap_w = max(0, overlap_w)
    
    debug.log(f"Tile blending: overlap=({overlap_h}x{overlap_w}), tile=({tile_h}x{tile_w})",
             category="dit", indent_level=1)
    
    result = torch.zeros((T, H, W, C), device=accum_device, dtype=dtype)
    weight_sum = torch.zeros((1, H, W, 1), device=accum_device, dtype=dtype)
    
    num_tiles = len(tiles_info)
    for tile_idx, (y_start, x_start, tile_h, tile_w) in enumerate(tiles_info):
        debug.log(f"  Tile {tile_idx + 1}/{num_tiles}: pos=({y_start},{x_start}), size=({tile_h}x{tile_w})",
                 category="dit", indent_level=2)
        
        tile_noises = [n[:, y_start:y_start + tile_h, x_start:x_start + tile_w, :] for n in noises]
        tile_conditions = [c[:, y_start:y_start + tile_h, x_start:x_start + tile_w, :] for c in conditions]
        
        with torch.no_grad():
            tile_result = runner.inference(noises=tile_noises, conditions=tile_conditions, **text_embeds)
        
        tile_upscaled = tile_result[0]
        
        # Create smooth weight mask for this tile
        weight_mask = _create_tile_weight_mask(
            tile_h, tile_w, y_start, x_start, H, W,
            overlap_h, overlap_w, tile_upscaled.device, dtype
        )
        
        # Apply weight to tile
        tile_weighted = tile_upscaled * weight_mask
        
        if accum_device != tile_weighted.device:
            tile_weighted = tile_weighted.to(accum_device)
            weight_mask = weight_mask.to(accum_device)
        
        # Accumulate weighted result and weights
        result[:, y_start:y_start + tile_h, x_start:x_start + tile_w, :] += tile_weighted
        weight_sum[:, y_start:y_start + tile_h, x_start:x_start + tile_w, :] += weight_mask
        
        del tile_noises, tile_conditions, tile_result, tile_upscaled, tile_weighted, weight_mask
    
    # Normalize by accumulated weights
    result = result / weight_sum.clamp(min=1e-6)
    
    if result.device != device:
        result = result.to(device)
    
    debug.log(f"Tiled DiT: merged {num_tiles} tiles with cosine blending", category="dit", indent_level=1)
    
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
    debug.log("━━━━━━━━ Phase 3: VAE Decode (whole latent) ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase3_decode_whole")
    
    if 'upscaled_latent' not in ctx:
        raise ValueError("No upscaled_latent in context. Run upscale_latent_chunked first.")
    
    latent = ctx['upscaled_latent']
    debug.log(f"Decoding latent shape: {latent.shape}", category="vae", force=True)
    
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
    
    debug.start_timer("vae_decode")
    samples = runner.vae_decode([latent])
    sample = samples[0]
    debug.end_timer("vae_decode", "VAE decode")
    
    # VAE decode output is (C, T, H, W), convert to (T, H, W, C)
    debug.log(f"Decoded shape (CTHW): {sample.shape}", category="vae", force=True)
    sample = sample.permute(1, 2, 3, 0)  # (C, T, H, W) -> (T, H, W, C)
    
    # Trim to original frame count (remove padding added for 4n+1 constraint)
    T_original = ctx.get('original_temporal_frames', sample.shape[0])
    if sample.shape[0] > T_original:
        debug.log(f"Trimming temporal: {sample.shape[0]} → {T_original} frames", category="vae")
        sample = sample[:T_original]
    
    ctx['decoded_video'] = sample
    del ctx['upscaled_latent']
    
    if not cache_model and ctx['vae_offload_device'] is not None:
        manage_model_device(model=runner.vae, target_device=ctx['vae_offload_device'],
                           model_name="VAE", debug=debug, reason="VAE offload", runner=runner)
    
    if progress_callback:
        progress_callback(1, 1, sample.shape[0], "Phase 3: Decoding")
    
    debug.end_timer("phase3_decode_whole", "Phase 3: VAE decoding complete")
    debug.log_memory_state("After Phase 3 (whole latent decoding)", show_tensors=False)
    
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
    video = video.clamp(-1, 1) * 0.5 + 0.5
    T = video.shape[0]
    
    debug.log(f"Post-processing {T} frames, color_correction={color_correction}",
             category="color", force=True)
    
    if progress_callback:
        progress_callback(0, 1, T, "Phase 4: Post-processing")
    
    if color_correction != "none" and 'transformed_video' in ctx:
        from ..utils.color_fix import (
            lab_color_transfer, lab_reinhard_transfer, wavelet_reconstruction,
            hsv_saturation_histogram_match, wavelet_adaptive_color_correction,
            adaptive_instance_normalization
        )
        
        # transformed_video is [C, T_ref, H, W] on CPU in [-1, 1] range
        # Permute to [T_ref, C, H, W] for per-frame chunking
        reference_cpu = ctx['transformed_video'].permute(1, 0, 2, 3)  # [T_ref, C, H, W]
        T_ref = reference_cpu.shape[0]
        
        # video is [T, H, W, C] in [0, 1] -> convert to [T, C, H, W] in [-1, 1]
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        video = video * 2.0 - 1.0           # [0, 1] -> [-1, 1]
        
        # Build frame indices for temporal alignment
        if T_ref != T:
            indices = torch.linspace(0, T_ref - 1, T).long()
        else:
            indices = None
        
        debug.log(f"Applying {color_correction} color correction ({T} frames, ref={T_ref})", category="color")
        
        # Process in chunks to avoid GPU OOM on high-res videos
        chunk_size = max(1, min(8, T))  # 8 frames at a time
        result_chunks = []
        
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            video_chunk = video[start:end]  # [chunk, C, H, W] already on GPU
            
            # Get corresponding reference frames on GPU
            if indices is not None:
                ref_chunk = reference_cpu[indices[start:end]].to(video.device)
            else:
                ref_chunk = reference_cpu[start:end].to(video.device)
            
            # Apply selected color correction method
            # content=video_chunk (upscaled), style=ref_chunk (original)
            if color_correction == "lab":
                video_chunk = lab_color_transfer(video_chunk, ref_chunk, debug, luminance_weight=0.8)
            elif color_correction == "lab_reinhard":
                video_chunk = lab_reinhard_transfer(video_chunk, ref_chunk, debug)
            elif color_correction == "wavelet":
                video_chunk = wavelet_reconstruction(video_chunk, ref_chunk, debug)
            elif color_correction == "hsv":
                # HSV runs on CPU to avoid GPU OOM (heavy per-hue-bin histogram matching)
                orig_device = video_chunk.device
                video_chunk = hsv_saturation_histogram_match(video_chunk.cpu(), ref_chunk.cpu(), debug).to(orig_device)
            elif color_correction == "wavelet_adaptive":
                video_chunk = wavelet_adaptive_color_correction(video_chunk, ref_chunk, debug)
            elif color_correction == "adain":
                video_chunk = adaptive_instance_normalization(video_chunk, ref_chunk)
            else:
                debug.log(f"Unknown color correction method: {color_correction}",
                          level="WARNING", category="color", force=True)
            
            result_chunks.append(video_chunk)
            del ref_chunk
        
        video = torch.cat(result_chunks, dim=0)  # [T, C, H, W] in [-1, 1]
        del result_chunks, reference_cpu
        
        # Convert back to [T, H, W, C] in [0, 1]
        video = video.clamp(-1, 1) * 0.5 + 0.5
        video = video.permute(0, 2, 3, 1)
    
    # ── USM Sharpening (ref: basicsr usm_sharp) ──
    if usm_weight > 0:
        import cv2
        import numpy as np
        
        radius = usm_radius if usm_radius % 2 == 1 else usm_radius + 1
        debug.log(f"Applying USM sharpening: weight={usm_weight}, radius={radius}, "
                  f"threshold={usm_threshold}", category="color", force=True)
        
        # video is [T, H, W, C] in [0, 1]
        video_np = video.cpu().float().numpy()
        
        for i in range(video_np.shape[0]):
            frame = video_np[i]  # [H, W, C] float32 in [0, 1]
            blur = cv2.GaussianBlur(frame, (radius, radius), 0)
            residual = frame - blur
            mask = (np.abs(residual) * 255 > usm_threshold).astype(np.float32)
            soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
            sharp = np.clip(frame + usm_weight * residual, 0, 1)
            video_np[i] = soft_mask * sharp + (1 - soft_mask) * frame
        
        video = torch.from_numpy(video_np).to(dtype=video.dtype, device=video.device)
        del video_np
        debug.log("USM sharpening completed", category="color", force=True)
    
    ctx['final_video'] = video
    
    if 'decoded_video' in ctx:
        del ctx['decoded_video']
    if 'transformed_video' in ctx:
        del ctx['transformed_video']
    
    if progress_callback:
        progress_callback(1, 1, T, "Phase 4: Post-processing")
    
    debug.end_timer("phase4_postprocess", "Phase 4: Post-processing complete")
    
    return ctx
