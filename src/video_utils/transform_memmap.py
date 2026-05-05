"""Disk-backed uint8/uint16 memmap for the transform reference video.

Manages the lifecycle of a numpy memory-mapped file that stores the
pre-processed input video at padded resolution. Used during the encode
phase (write) and the post-process phase (read for color correction).

This class is intentionally separate from VideoPrePostProcess — it manages
storage (a different concern from pixel transforms).
"""

import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch


class TransformMemmap:
    """Disk-backed reference video storage using numpy memmap.

    Stores frames as uint8 (8-bit source) or uint16 (>8-bit source) in
    THWC layout, converting on-the-fly to/from float tensors.  A ``"raw"``
    mode is also available for float16 data that needs no conversion
    (e.g. offloading an already-normalised tensor).

    Parameters
    ----------
    shape : tuple
        Dimensions of the stored array.  ``(T, H, W, C)`` for quantised
        modes; arbitrary for ``"raw"``.
    mode : ``"uint8"`` | ``"uint16"`` | ``"raw"``
        Storage format.  ``"uint8"``/``"uint16"`` quantise ``[0, 1]`` float
        frames; ``"raw"`` stores float16 unchanged.
    """

    _MODE_MAP = {
        "uint8":  (np.uint8,   255.0,  'THWC'),
        "uint16": (np.uint16, 65535.0, 'THWC'),
        "raw":    (np.float16,  0.0,   'CTHW'),
    }

    def __init__(self, shape: Tuple[int, ...], mode: str = "uint8"):
        if mode not in self._MODE_MAP:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {set(self._MODE_MAP)}")
        self.mode = mode
        self.dtype, self.scale, self.layout = self._MODE_MAP[mode]
        self.shape = shape

        self._file = tempfile.NamedTemporaryFile(suffix='.dat', delete=False)
        self.path = self._file.name
        self._file.close()

        self._mm = np.memmap(self.path, dtype=self.dtype, mode='w+', shape=shape)

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def write_batch(self, frames: torch.Tensor, start: int) -> None:
        """Write a batch of float [0, 1] frames to the memmap.

        Parameters
        ----------
        frames : Tensor
            ``[B, H, W, C]`` float in ``[0, 1]``.
        start : int
            Temporal index to write at.
        """
        count = frames.shape[0]
        data = (frames.cpu().float() * self.scale).clamp(0, self.scale)
        self._mm[start:start + count] = data.numpy().astype(self.dtype)

    def write(self, tensor: torch.Tensor) -> None:
        """Write a full tensor directly (raw mode).

        For ``"raw"`` mode the data is stored as float16 without scaling.
        For quantised modes the tensor is assumed ``[0, 1]`` and scaled.
        """
        if self.mode == "raw":
            self._mm[:] = tensor.cpu().numpy()
        else:
            data = (tensor.cpu().float() * self.scale).clamp(0, self.scale)
            self._mm[:] = data.numpy().astype(self.dtype)

    def write_raw_batch(self, frames_np: np.ndarray, start: int) -> None:
        """Write raw integer pixels directly (no float conversion).

        Parameters
        ----------
        frames_np : np.ndarray
            ``[B, H, W, C]`` in the same dtype as the memmap (uint8 or uint16).
        start : int
            Temporal index to write at.
        """
        count = frames_np.shape[0]
        self._mm[start:start + count] = frames_np.astype(self.dtype)

    def mirror_pad(self, t_input: int) -> None:
        """Mirror-pad trailing frames up to self.shape[0]."""
        t_total = self.shape[0]
        for p in range(t_total - t_input):
            src = t_input - 1 - (p % t_input) if t_input > 1 else 0
            self._mm[t_input + p] = self._mm[src]

    def flush(self) -> None:
        self._mm.flush()

    def release(self) -> None:
        """Flush and unmap the backing array to free RSS, keeping the file on disk.

        Subsequent reads will lazily re-open the file in read-only mode.
        """
        if self._mm is not None:
            self._mm.flush()
            del self._mm
            self._mm = None

    def _ensure_mapped(self) -> None:
        """Re-open the memmap in read-only mode if it was released."""
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def read_range(
        self,
        start: int,
        count: int,
        as_cthw: bool = True,
    ) -> torch.Tensor:
        """Read a temporal slice and return as a float16 tensor in [-1, 1].

        Parameters
        ----------
        start : int
            First frame index.
        count : int
            Number of frames.
        as_cthw : bool
            If True, return ``[C, T, H, W]``; otherwise ``[T, H, W, C]``.

        Returns
        -------
        Tensor
            Float16 tensor on CPU.
        """
        self._ensure_mapped()
        count = min(count, self.shape[0] - start)
        raw = np.array(self._mm[start:start + count])
        if self.scale > 0:
            raw = raw.astype(np.float32) / self.scale
            raw = (raw * 2.0 - 1.0).astype(np.float16)  # [0,1] → [-1,1]
        t = torch.from_numpy(raw)
        if as_cthw and self.layout == 'THWC':
            t = t.permute(3, 0, 1, 2)  # THWC → CTHW
        return t

    def read_all(self, as_cthw: bool = True) -> torch.Tensor:
        """Read the entire memmap."""
        return self.read_range(0, self.shape[0], as_cthw=as_cthw)

    # ------------------------------------------------------------------ #
    #  Metadata (for serialization into decode_meta / ctx)                 #
    # ------------------------------------------------------------------ #

    def meta(self) -> dict:
        """Return a serializable dict for passing through ctx / decode_meta."""
        return {
            'path': self.path,
            'dtype': self.dtype,
            'shape': self.shape,
            'layout': self.layout,
            'scale': self.scale,
            'mode': self.mode,
        }

    @classmethod
    def from_meta(cls, meta: dict) -> 'TransformMemmap':
        """Reconstruct a read-only view from a meta dict."""
        obj = object.__new__(cls)
        obj.path = meta['path']
        obj.dtype = meta['dtype']
        obj.shape = meta['shape']
        obj.layout = meta.get('layout', 'THWC')
        obj.scale = meta.get('scale', 255.0)
        obj.mode = meta.get('mode', 'uint8')
        obj._file = None
        obj._mm = np.memmap(obj.path, dtype=obj.dtype, mode='r', shape=obj.shape)
        return obj

    # ------------------------------------------------------------------ #
    #  Cleanup                                                             #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        try:
            size_mb = os.path.getsize(self.path) / 1e6
        except OSError:
            size_mb = 0.0
        return (f"DiskCache(mode={self.mode!r}, shape={self.shape}, "
                f"dtype={self.dtype.__name__}, size={size_mb:.1f}MB)")

    def cleanup(self) -> None:
        """Flush and delete the backing file."""
        try:
            if self._mm is not None:
                del self._mm
                self._mm = None
        except Exception:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def __del__(self):
        # Best-effort cleanup if the caller forgets
        try:
            if hasattr(self, 'path') and os.path.exists(self.path):
                os.unlink(self.path)
        except Exception:
            pass
