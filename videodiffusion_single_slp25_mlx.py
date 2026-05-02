"""
Neuroserver integration for the MLX-based Mac streaming video upscaler.

Registered class: SLP25MLX
Serve name:      "slp-25"  (macOS/Apple dispatch via local_models.py)

Supported model_parameters (from Slp25Filter / SLFilter / VideoFilter):
    output_width, output_height     – target resolution (default 1920x1080)
    start_idx, end_idx              – frame range (inclusive, 0-based)
    start_frame_idx, end_frame_idx  – alias for start_idx/end_idx
    input_frame_count               – max frames to process
    color_correction                – "wavelet" | "lab_reinhard" | "none"
    resolution                      – target short-side px (overrides w/h)
    cfg_scale                       – classifier-free guidance scale
    seed                            – RNG seed
    latent_noise_scale              – noise scale for latent space
    shared_noise                    – share noise across frames
    chunk_size, overlap             – temporal chunking
    dit_tile_overlap                – spatial overlap for DiT tiles
    output_bit_depth                – 10 for 10-bit HEVC output
    output_codec                    – (reserved for future use)
    ffmpeg_encoding                 – (reserved for future use)
    ffmpeg_preproc_filters          – (reserved for future use)
    creativity                      – "low" | "middle" | "high" (SLFilter)
    is_optimized_mode               – optimised path flag (SLFilter)
    enable_clc                      – colour-level correction (SLFilter)

Usage (smoke-test):
    python local_model_service.py \\
      --once \\
      --input-path input.mp4 \\
      --output-path output.mp4 \\
      --filters '[{"model": "slp-25", "output_width": 1920, "output_height": 1080}]' \\
      --device 0
"""

import os
import sys
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
matrix_root = os.path.abspath(os.path.join(script_dir, '..'))
if matrix_root not in sys.path:
    sys.path.insert(0, matrix_root)

from utils.register.registry_class import MODEL
from .inference_mac_streaming import VideoUpscaler

logger = logging.getLogger(__name__)


@MODEL.register_class()
class SLP25MLX:
    """MLX-accelerated video upscaler for macOS Apple Silicon."""

    def __init__(self, gpu_id=0, **kwargs):
        model_store = os.environ.get("TOPAZ_MODEL_STORE", "./data")
        split_dir = os.path.join(model_store, "nadit_blocks_3b")

        if not os.path.isdir(split_dir):
            split_dir = os.path.join(script_dir, "nadit_blocks_3b")

        self.upscaler = VideoUpscaler(
            split_dir=split_dir,
            model_dir=model_store if os.path.isdir(model_store) else None,
            device="mps",
            debug_enabled=True,
            use_mlx_dit=True,
            mlx_dtype="bf16",
            mlx_stream=True,
        )
        logger.info("SLP25MLX initialized (split_dir=%s, model_store=%s)", split_dir, model_store)

    def __call__(self, input_path, output_path, model_parameters, progress_func):
        if not os.path.exists(input_path):
            return 9  # MEDIA_READ_ERROR

        progress_func(0.0, message="Starting MLX upscale...")

        # If "resolution" shorthand is provided (short-side target), derive
        # output_width / output_height so VideoUpscaler picks them up.
        if 'resolution' in model_parameters:
            res = int(model_parameters['resolution'])
            iw = int(model_parameters.get('input_width', 0))
            ih = int(model_parameters.get('input_height', 0))
            if iw > 0 and ih > 0:
                if iw >= ih:
                    model_parameters.setdefault('output_height', res)
                    model_parameters.setdefault('output_width', int(res * iw / ih))
                else:
                    model_parameters.setdefault('output_width', res)
                    model_parameters.setdefault('output_height', int(res * ih / iw))

        # Honour input_frame_count as an upper bound on frames processed.
        if 'input_frame_count' in model_parameters and 'end_idx' not in model_parameters and 'end_frame_idx' not in model_parameters:
            ifc = int(model_parameters['input_frame_count'])
            start = int(model_parameters.get('start_idx', model_parameters.get('start_frame_idx', 0)))
            model_parameters['end_frame_idx'] = start + ifc

        try:
            self.upscaler(input_path, output_path, model_parameters,
                          progress_func=progress_func)
        except FileNotFoundError as e:
            logger.error("SLP25MLX: input not found: %s", e)
            return 9  # MEDIA_READ_ERROR
        except Exception:
            logger.exception("SLP25MLX processing failed")
            raise  # let neuroserver see the real error

        progress_func(1.0, message="Done", output_is_written=True)
        return 0
