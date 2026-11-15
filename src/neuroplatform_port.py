"""Model port: the unchanged PyTorch models of this repo, run on the machine's
backend by neuroplatform (MLX on Apple silicon, PyTorch CUDA on NVIDIA/AMD).

This is the only file that talks to the platform layer. ``install()`` (called
once by the entry script, a server adapter, or ``python -m neuroplatform.launch
--port <this module> ...``) swaps the platform versions in before models are built:

* ``model_factory.create_dit``         -> the NaDiT, weights streamed from split blocks
* ``model_factory.create_vae``         -> the video VAE inside the tiled/streamed codec
* ``color_fix.wavelet_reconstruction`` -> the same function, run on the backend

The model-specific facts live here too: which config builds each model, where
its weights are, and that on MLX the DiT's RoPE frequencies are computed with
MLX (as the hand-written MLX DiT does).
"""

import os
import sys

import torch

# The platform layer ships next to the code (neuroplatform/ at the repo root).
_PLATFORM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neuroplatform")
if os.path.isdir(_PLATFORM_DIR) and _PLATFORM_DIR not in sys.path:
    sys.path.insert(0, _PLATFORM_DIR)

try:
    import neuroplatform  # noqa: E402
    from neuroplatform import WeightFiles  # noqa: E402
except ImportError:          # no platform layer here: the pipeline stays plain PyTorch
    neuroplatform = None

from .video_utils.tiled_vae import TiledVAE  # noqa: E402


_INSTALLED = False


def install() -> None:
    """Replace the pipeline's plain-PyTorch model factories and colour fix with
    the platform versions (same signatures, same results). Does nothing when the
    platform layer is not available; safe to call more than once."""
    global _INSTALLED
    if neuroplatform is None or _INSTALLED:
        return
    _INSTALLED = True
    from . import model_factory
    from .utils import color_fix

    model_factory.create_dit = create_dit
    model_factory.create_vae = create_vae
    original = color_fix.wavelet_reconstruction
    wavelet = neuroplatform.accelerate(lambda content, style: original(content, style))
    color_fix.wavelet_reconstruction = lambda content, style, debug=None: wavelet(content, style)


def backend_name() -> str:
    return neuroplatform.detect_backend()


# ── DiT ─────────────────────────────────────────────────────────────────────

def create_dit(split_dir: str, device=None):
    from ..configs_3b.main import CONFIG_YAML
    from .common.config import create_object, load_config_from_string

    weights = WeightFiles.from_split_manifest(os.path.join(split_dir, "block_manifest.json"))
    model = neuroplatform.build_model(
        lambda: create_object(load_config_from_string(CONFIG_YAML).dit.model), weights, dtype=torch.bfloat16)
    backend = neuroplatform.detect_backend()
    if backend == "mlx":
        mlx_rope_frequencies(model)
    # Weights stream from the split blocks on every backend (MLX, or PyTorch on
    # NVIDIA/AMD): about two blocks resident instead of the whole model.
    return neuroplatform.accelerate(model, weights=weights, backend=backend, match_input_dtype=True,
                                    stream_weights=True)


def mlx_rope_frequencies(model: torch.nn.Module, theta: float = 10000.0) -> torch.nn.Module:
    """Set the DiT's RoPE inverse frequencies, 1 / theta**(arange(0, d, 2) / d),
    to the float32 values MLX computes (MLX's pow and PyTorch's differ in the
    last bit for some values). The model's own values come from PyTorch when it
    is constructed."""
    import mlx.core as mx
    import numpy as np

    for mod in model.modules():
        if type(mod).__name__ == "RotaryEmbedding" and "freqs" in mod._buffers:
            d = 2 * mod.freqs.shape[0]
            inv = 1.0 / (theta ** (mx.arange(0, d, 2, dtype=mx.float32) / d))
            mod._buffers["freqs"] = torch.from_numpy(np.array(inv))
    return model


# ── VAE ─────────────────────────────────────────────────────────────────────

def create_vae(vae_path: str, device=None) -> TiledVAE:
    from omegaconf import OmegaConf

    from ..configs_3b.main import CONFIG_YAML
    from .common.config import create_object, load_config_from_string
    from .models.video_vae_v3.vae_config import CONFIG_YAML as VAE_CONFIG_YAML

    cfg = load_config_from_string(CONFIG_YAML)
    model_cfg = OmegaConf.merge(cfg.vae.model, load_config_from_string(VAE_CONFIG_YAML))
    weights = WeightFiles({"": vae_path})
    vae = neuroplatform.build_model(lambda: create_object(model_cfg), weights, dtype=torch.bfloat16).eval()
    vae.encoder = neuroplatform.accelerate(vae.encoder, weights=weights.scoped("encoder."))
    vae.decoder = neuroplatform.accelerate(vae.decoder, weights=weights.scoped("decoder."))
    return TiledVAE(vae, dtype=torch.bfloat16)
