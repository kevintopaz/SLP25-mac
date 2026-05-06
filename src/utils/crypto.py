"""
Encrypted model file loader.

Call :func:`configure` once with the encryption key.  After that,
:func:`mx_load` and :func:`json_load` are drop-in replacements for
``mx.load(path)`` and ``json.load(open(path))`` — if a key is set they
decrypt transparently, otherwise they delegate to the plain readers.

This module knows nothing about file-name mappings or directory layout;
callers are responsible for resolving paths before passing them here.
"""

import hashlib
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# ═══════════════════════════════════════════════════════════════════════
# Module-level key — set once via configure()
# ═══════════════════════════════════════════════════════════════════════

_encryption_key: Optional[str] = None


def configure(encryption_key: Optional[str] = None) -> None:
    """Set (or clear) the encryption key for all subsequent loads."""
    global _encryption_key
    _encryption_key = encryption_key


# ═══════════════════════════════════════════════════════════════════════
# Low-level crypto helpers
# ═══════════════════════════════════════════════════════════════════════

def _derive_key_fast(password: str, salt: bytes) -> bytes:
    return hashlib.sha256(password.encode() + salt).digest()


def _ctr_decrypt(key: bytes, blob: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.CTR(blob[:16]))
    dec = cipher.decryptor()
    return dec.update(blob[16:]) + dec.finalize()


def _decrypt_keys(encrypted_data: bytes, password: str) -> List[str]:
    salt = encrypted_data[:16]
    nonce = encrypted_data[16:28]
    ciphertext = encrypted_data[28:]
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=salt, iterations=100000)
    key = kdf.derive(password.encode())
    return json.loads(AESGCM(key).decrypt(nonce, ciphertext, None).decode())


_DTYPE_STR_TO_TORCH = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.int16": torch.int16,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.float64": torch.float64,
    "torch.bool": torch.bool,
}


def _bytes_to_tensor(data: bytes, dtype_str: str, shape: List[int]) -> torch.Tensor:
    td = _DTYPE_STR_TO_TORCH.get(dtype_str)
    if td is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    if td == torch.bfloat16:
        t = torch.frombuffer(bytearray(data), dtype=torch.uint16).view(torch.bfloat16)
    elif td == torch.bool:
        t = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(torch.bool)
    else:
        t = torch.frombuffer(bytearray(data), dtype=td)
    return t.reshape(shape)


# ═══════════════════════════════════════════════════════════════════════
# Internal encrypted loaders
# ═══════════════════════════════════════════════════════════════════════

def _encrypted_safetensors_load(path: str, password: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file, safe_open

    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()

    if not metadata or "__salt__" not in metadata or "__tensor_meta__" not in metadata:
        raise ValueError(f"Invalid or corrupted encrypted safetensors: {path}")

    salt = bytes.fromhex(metadata["__salt__"])
    tensor_meta = json.loads(metadata["__tensor_meta__"])
    derived_key = _derive_key_fast(password, salt)
    encrypted_tensors = load_file(path)

    use_projection = "__keys__" in metadata
    if use_projection:
        original_keys = _decrypt_keys(
            bytes.fromhex(metadata["__keys__"]), password=password)
    else:
        original_keys = list(encrypted_tensors.keys())

    projected: Dict[str, torch.Tensor] = {}
    for i, orig_key in enumerate(original_keys):
        proj_key = str(i) if use_projection else orig_key
        meta = tensor_meta[proj_key]
        encrypted_blob = encrypted_tensors[proj_key].numpy().tobytes()
        tensor_bytes = _ctr_decrypt(derived_key, encrypted_blob)
        projected[proj_key] = _bytes_to_tensor(
            tensor_bytes, meta["dtype"], meta["shape"])

    if not use_projection:
        return projected

    state_dict: Dict[str, torch.Tensor] = OrderedDict()
    for i, k in enumerate(original_keys):
        state_dict[k] = projected[str(i)]
    return state_dict


def _encrypted_bytes_load(path: str, password: str) -> bytes:
    with open(path, "rb") as f:
        encrypted_data = f.read()
    salt = encrypted_data[:16]
    nonce = encrypted_data[16:28]
    ciphertext = encrypted_data[28:]
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=salt, iterations=100000)
    key = kdf.derive(password.encode())
    return AESGCM(key).decrypt(nonce, ciphertext, None)


# ═══════════════════════════════════════════════════════════════════════
# Public drop-in loaders
# ═══════════════════════════════════════════════════════════════════════

def mx_load(path: str, dtype=None) -> dict:
    """Drop-in for ``mx.load(path)`` → ``{name: mx.array}``.

    Decrypts transparently when an encryption key has been configured.
    """
    import mlx.core as mx

    if _encryption_key:
        torch_dict = _encrypted_safetensors_load(path, password=_encryption_key)
        d = {}
        for k, v in torch_dict.items():
            if v.dtype == torch.bfloat16:
                d[k] = mx.array(
                    v.view(torch.uint16).numpy()
                ).view(mx.bfloat16)
            else:
                d[k] = mx.array(v.numpy())
            if dtype is not None:
                d[k] = d[k].astype(dtype)
        return d

    d = mx.load(path)
    if dtype is not None:
        return {k: v.astype(dtype) for k, v in d.items()}
    return d


def json_load(path: str) -> Any:
    """Drop-in for ``json.load(open(path))``.

    Decrypts transparently when an encryption key has been configured.
    """
    if _encryption_key:
        raw = _encrypted_bytes_load(path, password=_encryption_key)
        return json.loads(raw.decode())

    with open(path) as f:
        return json.load(f)
