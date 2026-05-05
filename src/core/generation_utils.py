"""
Generation Utilities for SeedVR2

Setup, configuration, and utility functions shared by the streaming pipeline.

Setup Functions:
- setup_generation_context: Initialize context with device configuration
- prepare_runner: Configure VideoDiffusionInfer with all settings

Configuration Helpers:
- load_text_embeddings: Load positive/negative text embeddings for DiT
- ensure_precision_initialized: Log model dtype information
"""

import os
import torch
from typing import Dict, List, Optional, Tuple, Any, Union

from .model_configuration import configure_runner
from .infer import VideoDiffusionInfer
from ..optimization.compatibility import COMPUTE_DTYPE, BFLOAT16_SUPPORTED
from ..optimization.memory_manager import manage_tensor
from ..utils.constants import get_script_directory
if os.environ.get('CLOUD_RUN', 'false') == 'true':
    from utils.getData import get_path_or_bytesio


# Get script directory for embeddings
script_directory = get_script_directory()

EMBEDDING_NAME_MAP = {
    "pos_emb.pt.enc": "model.part3.bin.enc",
    "neg_emb.pt.enc": "model.part1.bin.enc",
}


def setup_generation_context(
    dit_device: Optional[Union[str, torch.device]] = None,
    vae_device: Optional[Union[str, torch.device]] = None,
    dit_offload_device: Optional[Union[str, torch.device]] = None,
    vae_offload_device: Optional[Union[str, torch.device]] = None,
    tensor_offload_device: Optional[Union[str, torch.device]] = None,
    debug: Optional['Debug'] = None
) -> Dict[str, Any]:
    """
    Initialize generation context with device configuration.
    
    Processes device objects, configures environment variables, and creates the
    generation context dictionary with all necessary state.
    """
    def _normalize_device(device_spec: Optional[Union[str, torch.device]]) -> torch.device:
        if device_spec is None:
            return torch.device("cpu")
        if isinstance(device_spec, torch.device):
            return device_spec
        return torch.device(device_spec)
    
    dit_device = _normalize_device(dit_device)
    vae_device = _normalize_device(vae_device)
    dit_offload_device = _normalize_device(dit_offload_device) if dit_offload_device is not None else None
    vae_offload_device = _normalize_device(vae_offload_device) if vae_offload_device is not None else None
    # MPS unified memory: CPU offload causes sync overhead with no memory benefit
    is_mps = dit_device.type == 'mps' or vae_device.type == 'mps'
    if is_mps and tensor_offload_device is not None and str(tensor_offload_device) == 'cpu':
        tensor_offload_device = None
    else:
        tensor_offload_device = _normalize_device(tensor_offload_device) if tensor_offload_device is not None else None
    
    os.environ.setdefault("LOCAL_RANK", "0")
    
    try:
        import comfy.model_management
        interrupt_fn = comfy.model_management.throw_exception_if_processing_interrupted
        comfyui_available = True
    except:
        interrupt_fn = None
        comfyui_available = False
    
    ctx = {
        'dit_device': dit_device,
        'vae_device': vae_device,
        'dit_offload_device': dit_offload_device,
        'vae_offload_device': vae_offload_device,
        'tensor_offload_device': tensor_offload_device,
        'compute_dtype': COMPUTE_DTYPE,
        'interrupt_fn': interrupt_fn,
        'video_transform': None,
        'text_embeds': None,
        'all_latents': [],
        'all_upscaled_latents': [],
        'batch_samples': [],
        'final_video': None,
        'comfyui_available': comfyui_available,
    }
    
    if debug:
        offload_info = []
        if dit_offload_device:
            offload_info.append(f"DiT offload={str(dit_offload_device)}")
        if vae_offload_device:
            offload_info.append(f"VAE offload={str(vae_offload_device)}")
        if tensor_offload_device:
            offload_info.append(f"Tensor offload={str(tensor_offload_device)}")
        
        offload_str = ", ".join(offload_info) if offload_info else "none"
        
        debug.log(
            f"Processing context initialized (device={str(dit_device)})",
            category="setup"
        )
        if ctx['compute_dtype'] == torch.float32:
            reason = "quality"
        elif not BFLOAT16_SUPPORTED:
            reason = "compatibility (GPU precision limitations detected)"
        else:
            reason = "performance"
        debug.log(f"Unified compute dtype: {ctx['compute_dtype']} across entire pipeline for maximum {reason}", category="precision")
    
    return ctx


def prepare_runner(
    dit_model: str,
    vae_model: str,
    model_dir: str,
    debug: 'Debug',
    ctx: Dict[str, Any],
    dit_cache: bool = False,
    vae_cache: bool = False,
    dit_id: Optional[int] = None,
    vae_id: Optional[int] = None,
    encode_tiled: bool = False,
    encode_tile_size: Optional[Tuple[int, int]] = None,
    encode_tile_overlap: Optional[Tuple[int, int]] = None,
    decode_tiled: bool = False,
    decode_tile_size: Optional[Tuple[int, int]] = None,
    decode_tile_overlap: Optional[Tuple[int, int]] = None,
    tile_debug: str = "false",
    attention_mode: str = 'sdpa',
    torch_compile_args_dit: Optional[Dict[str, Any]] = None,
    torch_compile_args_vae: Optional[Dict[str, Any]] = None,
    skip_vae_weights: bool = False
) -> Tuple['VideoDiffusionInfer', Dict[str, Any]]:
    """
    Prepare runner with model state management and global cache integration.
    Handles model changes and caching logic with independent DiT/VAE caching support.
    """
    debug.log("Configuring inference runner...", category="runner")
    runner, cache_context = configure_runner(
        dit_model=dit_model,
        vae_model=vae_model,
        base_cache_dir=model_dir,
        debug=debug,
        ctx=ctx,
        dit_cache=dit_cache,
        vae_cache=vae_cache,
        dit_id=dit_id,
        vae_id=vae_id,
        encode_tiled=encode_tiled,
        encode_tile_size=encode_tile_size,
        encode_tile_overlap=encode_tile_overlap,
        decode_tiled=decode_tiled,
        decode_tile_size=decode_tile_size,
        decode_tile_overlap=decode_tile_overlap,
        tile_debug=tile_debug,
        attention_mode=attention_mode,
        torch_compile_args_dit=torch_compile_args_dit,
        torch_compile_args_vae=torch_compile_args_vae,
        skip_vae_weights=skip_vae_weights
    )

    return runner, cache_context


def load_text_embeddings(script_directory: str, device: torch.device, 
                        dtype: torch.dtype, debug: Optional['Debug'] = None) -> Dict[str, List[torch.Tensor]]:
    """Load and prepare text embeddings for generation."""
    if os.environ.get('CLOUD_RUN', 'false') == 'true' and os.path.isdir('/app/data'):
        text_pos_embeds = torch.load(get_path_or_bytesio('/app/data/pos_emb.pt.enc', EMBEDDING_NAME_MAP), weights_only=True)
        text_neg_embeds = torch.load(get_path_or_bytesio('/app/data/neg_emb.pt.enc', EMBEDDING_NAME_MAP), weights_only=True)
    else:
        text_pos_embeds = torch.load(os.path.join(script_directory, 'models/pos_emb.pt'), weights_only=True)
        text_neg_embeds = torch.load(os.path.join(script_directory, 'models/neg_emb.pt'), weights_only=True)
    
    text_pos_embeds = manage_tensor(
        tensor=text_pos_embeds,
        target_device=device,
        tensor_name="text_pos_embeds",
        dtype=dtype,
        debug=debug,
        reason="DiT inference"
    )
    text_neg_embeds = manage_tensor(
        tensor=text_neg_embeds,
        target_device=device,
        tensor_name="text_neg_embeds",
        dtype=dtype,
        debug=debug,
        reason="DiT inference"
    )
    
    return {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}


def ensure_precision_initialized(
    ctx: Dict[str, Any],
    runner: 'VideoDiffusionInfer',
    debug: Optional['Debug'] = None
) -> None:
    """Log model dtypes for debugging. Compute dtype is hardcoded in context."""
    if not debug:
        return
    
    try:
        dit_dtype = None
        vae_dtype = None

        mlx_dit_dtype = getattr(runner, '_mlx_dit_dtype', None)
        mlx_vae_dtype = getattr(runner, '_mlx_vae_dtype', None)

        if mlx_dit_dtype is not None:
            dit_dtype = f"mlx.{mlx_dit_dtype}"
        elif runner.dit is not None:
            try:
                p = next(runner.dit.parameters())
                if p.device.type != 'meta':
                    dit_dtype = str(p.dtype)
            except StopIteration:
                pass

        if mlx_vae_dtype is not None:
            vae_dtype = f"mlx.{mlx_vae_dtype}"
        elif runner.vae is not None:
            try:
                p = next(runner.vae.parameters())
                if p.device.type != 'meta':
                    vae_dtype = str(p.dtype)
            except StopIteration:
                pass

        parts = []
        if dit_dtype is not None:
            parts.append(f"enhancement={dit_dtype}")
        if vae_dtype is not None:
            parts.append(f"codec={vae_dtype}")
        parts.append(f"compute={ctx['compute_dtype']}")

        if parts:
            debug.log(f"Compute precision: {', '.join(parts)}", category="precision")
            
    except Exception as e:
        debug.log(f"Could not log model dtypes: {e}", level="WARNING", category="precision", force=True)
