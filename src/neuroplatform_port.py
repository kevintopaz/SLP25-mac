"""Model port: the unchanged PyTorch models of this repo, run on the machine's
backends by neuroplatform (see PLACEMENT: which backend each model uses on each
platform).

This is the only file that talks to the platform layer. ``install()`` (called
once by the entry script, a server adapter, or ``python -m neuroplatform.launch
--port <this module> ...``) swaps the platform versions in before models are built:

* ``model_factory.create_dit``         -> the NaDiT, weights streamed from split blocks
* ``model_factory.create_vae``         -> the video VAE inside the tiled/streamed codec
* ``color_fix.wavelet_reconstruction`` -> the same function, run on the backend

The model-specific facts live here too: which config builds each model, where
its weights are, and that on MLX the DiT's RoPE frequencies are computed with
MLX (as the hand-written MLX DiT does).

Which backend runs each model, and with what options, is decided here, per
platform (PLACEMENT). For experiments it can be overridden without editing:
NEURO_BACKEND_DIT / NEURO_BACKEND_VAE / NEURO_BACKEND_COLORFIX (one model),
NEURO_BACKEND (all), NEURO_GGML_DIT_DTYPE / NEURO_GGML_VAE_DTYPE (f32, f16, bf16).
"""

import os
import sys

import torch

# The platform layer: the neuroplatform submodule at the repo root (git clone --recurse-submodules), else an installed
# neuroplatform wheel.
_PLATFORM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neuroplatform")
if os.path.isdir(_PLATFORM_DIR) and _PLATFORM_DIR not in sys.path:
    sys.path.insert(0, _PLATFORM_DIR)

try:
    import neuroplatform  # noqa: E402
    from neuroplatform import WeightFiles  # noqa: E402
except ImportError:          # no platform layer here: the pipeline stays plain PyTorch
    neuroplatform = None
if neuroplatform is not None:
    neuroplatform.check_api(1)    # an incompatible neuroplatform fails here, not by silently changing backends

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
    backend, opts = placement("colorfix")
    wavelet = neuroplatform.accelerate(lambda content, style: original(content, style), backend=backend, **opts)
    color_fix.wavelet_reconstruction = lambda content, style, debug=None: wavelet(content, style)


# ── placement: which backend runs each model, per platform ───────────────────

# On ggml both models run in bf16: the DiT carries activations beyond the f16
# range (73k from block 9), and the VAE overflows f16 on real video (its encoder
# output turns to NaN). The colour fix is a few elementwise/conv ops on MLX.
PLACEMENT = {
    "apple": {"dit": ("ggml", {"ggml_dtype": "bf16"}),
              "vae": ("ggml", {"ggml_dtype": "bf16"}),
              "colorfix": ("mlx", {})},
    "cuda": {"dit": ("torch", {}), "vae": ("torch", {}), "colorfix": ("torch", {})},   # NVIDIA, AMD (ROCm)
    "cpu": {"dit": ("torch", {}), "vae": ("torch", {}), "colorfix": ("torch", {})},
}


def platform_name() -> str:
    import platform
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "apple"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _available(backend: str) -> bool:
    return neuroplatform is not None and neuroplatform.available(backend)


def placement(kind: str):
    """(backend, accelerate options) for a model kind: "dit", "vae", "colorfix"."""
    backend, opts = PLACEMENT[platform_name()][kind]
    forced = (os.environ.get(f"NEURO_BACKEND_{kind.upper()}") or os.environ.get("NEURO_BACKEND") or "").strip().lower()
    if forced and forced != "auto" and forced != backend:
        backend, opts = forced, {}
    if not _available(backend):                  # e.g. the ggml library not built on this machine
        fallback = neuroplatform.detect_backend()
        print(f"neuroplatform_port: {backend} not available for {kind}; using {fallback}", file=sys.stderr)
        backend, opts = fallback, {}
    opts = dict(opts)
    if backend.startswith("ggml"):
        opts["ggml_dtype"] = os.environ.get(f"NEURO_GGML_{kind.upper()}_DTYPE", opts.get("ggml_dtype", "bf16"))
    return backend, opts


def backend_name(kind: str) -> str:
    return placement(kind)[0]


# ── DiT ─────────────────────────────────────────────────────────────────────

def create_dit(split_dir: str, device=None):
    from ..configs_3b.main import CONFIG_YAML
    from .common.config import create_object, load_config_from_string

    weights = WeightFiles.from_split_manifest(os.path.join(split_dir, "block_manifest.json"))
    model = neuroplatform.build_model(
        lambda: create_object(load_config_from_string(CONFIG_YAML).dit.model), weights, dtype=torch.bfloat16)
    backend, opts = placement("dit")
    if backend == "mlx":
        mlx_rope_frequencies(model)
    if backend == "torch" and os.environ.get("NEURO_TORCH_GRAPH", "0") == "1":
        opts = {**opts, "graph": True}      # recorded graph + fused kernels (backends.torch.graph)
    if backend != "torch" or opts.get("graph"):
        # The DiT's window-group attention loop and chunked MLP cap eager PyTorch's
        # activation memory by filling a preallocated buffer slice by slice. A
        # backend plans its own memory (and chunks itself), so it records the plain
        # forward instead.
        os.environ.setdefault("DIT_WINDOW_GROUP_TOKENS", "0")
    # Weights stream from the split blocks on every backend (MLX, ggml, or PyTorch
    # on NVIDIA/AMD): about two blocks resident instead of the whole model.
    return neuroplatform.accelerate(model, weights=weights, backend=backend, match_input_dtype=True,
                                    stream_weights=True, **opts)


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
    backend, opts = placement("vae")
    vae.encoder = neuroplatform.accelerate(vae.encoder, weights=weights.scoped("encoder."), backend=backend, **opts)
    vae.decoder = neuroplatform.accelerate(vae.decoder, weights=weights.scoped("decoder."), backend=backend, **opts)
    if vae.encoder.device.type == "cuda":    # ggml on CUDA exchanges CUDA tensors: the VAE's own
        vae.to("cuda")                       # torch layers (quant convs) live there too
    return TiledVAE(vae, dtype=torch.bfloat16)
