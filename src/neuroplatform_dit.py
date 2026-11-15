"""NaDiT on neuroplatform: the unchanged PyTorch NaDiT, run on the local backend.

The same code runs on every platform; the backend is picked from the
hardware (NEURO_BACKEND overrides). The only model-specific facts: which
config builds the model, where its weights are, and that on MLX its RoPE
frequencies are computed with MLX (as MLXNaDiT does). Everything else (graph capture, attention/RMSNorm fusion,
per-block streaming, memory policy, drop-in module behaviour) is generic
platform code.
"""

import os

import torch

import neuroplatform
from neuroplatform import WeightFiles


def create_neuroplatform_dit(split_dir: str):
    from ..configs_3b.main import CONFIG_YAML
    from .common.config import create_object, load_config_from_string

    weights = WeightFiles.from_split_manifest(os.path.join(split_dir, "block_manifest.json"))
    model = neuroplatform.build_model(
        lambda: create_object(load_config_from_string(CONFIG_YAML).dit.model), weights, dtype=torch.bfloat16)
    backend = neuroplatform.detect_backend()       # mlx on Apple silicon, torch on NVIDIA/AMD
    if backend == "mlx":
        mlx_rope_frequencies(model)
    return neuroplatform.accelerate(model, weights=weights, backend=backend, match_input_dtype=True)


def mlx_rope_frequencies(model: torch.nn.Module, theta: float = 10000.0) -> torch.nn.Module:
    """Set the DiT's RoPE inverse frequencies, 1 / theta**(arange(0, d, 2) / d),
    to the float32 values MLX computes (MLXNaDiT computes them with MLX; MLX's
    pow and PyTorch's differ in the last bit for some values). The model's own
    values come from PyTorch when it is constructed."""
    import mlx.core as mx
    import numpy as np

    for mod in model.modules():
        if type(mod).__name__ == "RotaryEmbedding" and "freqs" in mod._buffers:
            d = 2 * mod.freqs.shape[0]
            inv = 1.0 / (theta ** (mx.arange(0, d, 2, dtype=mx.float32) / d))
            mod._buffers["freqs"] = torch.from_numpy(np.array(inv))
    return model
