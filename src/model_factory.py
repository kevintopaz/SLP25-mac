"""Build the pipeline's models with plain PyTorch.

* ``create_dit(split_dir, device)``: the NaDiT from its config, weights loaded
  from a split checkpoint (``block_manifest.json``: one file per transformer
  block plus the input/output layers).
* ``create_vae(vae_path, device)``: the video VAE from its config, inside the
  tiled, temporally streamed codec (``video_utils.tiled_vae.TiledVAE``).

Platform ports may replace these functions (see neuroplatform_port.install).
"""

import json
import os

import torch
from safetensors import safe_open

from .video_utils.tiled_vae import TiledVAE

DTYPE = torch.bfloat16


def _load_into(model: torch.nn.Module, prefixed_files, device) -> None:
    """Load (prefix, safetensors path) pairs: key k of a file is parameter prefix + k."""
    state = {}
    for prefix, path in prefixed_files:
        with safe_open(path, "pt", device=str(device)) as f:
            for k in f.keys():
                state[prefix + k] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    params = {n for n, _ in model.named_parameters()}
    if [n for n in missing if n in params] or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")


def create_dit(split_dir: str, device="cuda") -> torch.nn.Module:
    from ..configs_3b.main import CONFIG_YAML
    from .common.config import create_object, load_config_from_string

    manifest = json.load(open(os.path.join(split_dir, "block_manifest.json")))
    with torch.device(device):
        model = create_object(load_config_from_string(CONFIG_YAML).dit.model)
    model = model.to(DTYPE).eval()
    files = [(f"blocks.{i}.", p) for i, p in enumerate(manifest["blocks"])]
    files += [(f"{name}.", p) for name, p in manifest.get("io_components", {}).items()]
    _load_into(model, files, device)
    return model


def create_vae(vae_path: str, device="cuda") -> TiledVAE:
    from omegaconf import OmegaConf

    from ..configs_3b.main import CONFIG_YAML
    from .common.config import create_object, load_config_from_string
    from .models.video_vae_v3.vae_config import CONFIG_YAML as VAE_CONFIG_YAML

    cfg = load_config_from_string(CONFIG_YAML)
    model_cfg = OmegaConf.merge(cfg.vae.model, load_config_from_string(VAE_CONFIG_YAML))
    with torch.device(device):
        vae = create_object(model_cfg)
    vae = vae.to(DTYPE).eval()
    _load_into(vae, [("", vae_path)], device)
    return TiledVAE(vae, dtype=DTYPE)
