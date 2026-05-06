"""Spatial tile assembly and cleanup for decoded VAE tiles.

Pure NumPy operations for blending overlapping spatial tiles from disk-backed
memmaps.  These functions are VAE-agnostic — they only need the ``tile_infos``
dict list produced by any tiled decoder (MLX, MPS, etc.).
"""

import logging
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Blend weights ────────────────────────────────────────────────────────────

def make_blend_weight(tile_h, tile_w, overlap, n_h, n_w, ti, tj):
    """Bilinear ramp at tile borders for spatial blending.

    Parameters
    ----------
    tile_h, tile_w : int
        Pixel dimensions of each tile.
    overlap : int or (int, int)
        Overlap in pixels — scalar (square) or ``(ov_h, ov_w)``.
    n_h, n_w : int
        Grid dimensions (number of tiles vertically / horizontally).
    ti, tj : int
        Row / column index of this tile in the grid.

    Returns
    -------
    np.ndarray
        ``(tile_h, tile_w)`` float32 weight mask.
    """
    if isinstance(overlap, (tuple, list)):
        ov_h, ov_w = int(overlap[0]), int(overlap[1])
    else:
        ov_h = ov_w = int(overlap)
    bw = np.ones((tile_h, tile_w), dtype=np.float32)
    if ti > 0:
        bw[:ov_h, :] *= np.linspace(0, 1, ov_h)[:, None]
    if ti < n_h - 1:
        bw[-ov_h:, :] *= np.linspace(1, 0, ov_h)[:, None]
    if tj > 0:
        bw[:, :ov_w] *= np.linspace(0, 1, ov_w)[None, :]
    if tj < n_w - 1:
        bw[:, -ov_w:] *= np.linspace(1, 0, ov_w)[None, :]
    return bw


# ── Tile I/O ─────────────────────────────────────────────────────────────────

def read_tile_as_fp16(info, frame_start, frame_count):
    """Read a temporal sub-range from a tile memmap, returning fp16 [-1, 1].

    Parameters
    ----------
    info : dict
        Tile metadata with ``"path"``, ``"shape"``, ``"dtype"`` keys.
    frame_start, frame_count : int
        Temporal window to read.

    Returns
    -------
    np.ndarray
        ``(frame_count, tile_H, tile_W, 3)`` float16 in ``[-1, 1]``.
    """
    _tile_dtype = np.dtype(info.get("dtype", "uint8"))
    mm = np.memmap(info["path"], dtype=_tile_dtype, mode='r',
                   shape=tuple(info["shape"]))
    tile_raw = np.array(mm[frame_start:frame_start + frame_count])
    del mm
    if _tile_dtype == np.uint8:
        return tile_raw.astype(np.float16) / np.float16(127.5) - np.float16(1.0)
    if _tile_dtype == np.uint16:
        return tile_raw.astype(np.float16) / np.float16(32767.5) - np.float16(1.0)
    return tile_raw.astype(np.float16)


# ── Assembly ─────────────────────────────────────────────────────────────────

def _compute_total_weight(tile_infos, pixel_H, pixel_W):
    """Pre-compute the normalisation weight map across all tiles."""
    info0 = tile_infos[0]
    tile_H, tile_W = info0["tile_H"], info0["tile_W"]
    ov_H, ov_W = info0["ov_H"], info0["ov_W"]
    n_h, n_w = info0["n_h"], info0["n_w"]

    total_wt = np.zeros((pixel_H, pixel_W), dtype=np.float32)
    for info in tile_infos:
        bw = make_blend_weight(tile_H, tile_W, (ov_H, ov_W),
                               n_h, n_w, info["ti"], info["tj"])
        ph, pw = info["pixel_h"], info["pixel_w"]
        clip_h = min(tile_H, pixel_H - ph)
        clip_w = min(tile_W, pixel_W - pw)
        total_wt[ph:ph + clip_h, pw:pw + clip_w] += bw[:clip_h, :clip_w]
    return np.maximum(total_wt, 1e-4), tile_H, tile_W, ov_H, ov_W, n_h, n_w


def assemble_tiles_from_disk(tile_infos, pixel_H, pixel_W, num_frames=None):
    """Blend decoded pixel tiles from memmap files into a single output.

    Parameters
    ----------
    tile_infos : list[dict]
        Tile metadata from ``tiled_decode``.
    pixel_H, pixel_W : int
        Full output resolution.
    num_frames : int, optional
        Cap on output pixel frames.

    Returns
    -------
    np.ndarray
        ``(T, H, W, 3)`` float16.
    """
    if not tile_infos:
        raise ValueError("No tile_infos provided")

    info0 = tile_infos[0]
    T_pix = info0["shape"][0]
    if num_frames is not None:
        T_pix = min(T_pix, num_frames)

    total_wt, tile_H, tile_W, ov_H, ov_W, n_h, n_w = \
        _compute_total_weight(tile_infos, pixel_H, pixel_W)

    output = np.zeros((T_pix, pixel_H, pixel_W, 3), dtype=np.float32)

    t0 = time.perf_counter()
    for info in tile_infos:
        tile_data = read_tile_as_fp16(info, 0, T_pix).astype(np.float32)
        ph, pw = info["pixel_h"], info["pixel_w"]
        clip_h = min(tile_H, pixel_H - ph)
        clip_w = min(tile_W, pixel_W - pw)
        bw = make_blend_weight(tile_H, tile_W, (ov_H, ov_W),
                               n_h, n_w, info["ti"], info["tj"])
        bw = bw[:clip_h, :clip_w]
        norm_bw = bw / total_wt[ph:ph + clip_h, pw:pw + clip_w]
        bw4d = norm_bw[None, :, :, None]
        output[:, ph:ph + clip_h, pw:pw + clip_w, :] += (
            tile_data[:, :clip_h, :clip_w, :] * bw4d
        )
        del tile_data

    elapsed = time.perf_counter() - t0
    logger.info(f"Assembled tiles from disk in {elapsed:.1f}s -> {output.shape}")
    return output.astype(np.float16)


def assemble_tiles_from_disk_range(tile_infos, pixel_H, pixel_W,
                                   frame_start, frame_count):
    """Blend a temporal sub-range of decoded pixel tiles from memmap files.

    Same spatial blending as ``assemble_tiles_from_disk`` but only reads
    frames ``[frame_start : frame_start + frame_count]``, allowing the
    caller to process in small temporal batches.

    Returns
    -------
    np.ndarray
        ``(frame_count, H, W, 3)`` float16.
    """
    if not tile_infos:
        raise ValueError("No tile_infos provided")

    total_wt, tile_H, tile_W, ov_H, ov_W, n_h, n_w = \
        _compute_total_weight(tile_infos, pixel_H, pixel_W)

    output = np.zeros((frame_count, pixel_H, pixel_W, 3), dtype=np.float32)

    fs, fe = frame_start, frame_start + frame_count
    for info in tile_infos:
        tile_data = read_tile_as_fp16(info, fs, fe - fs).astype(np.float32)
        ph, pw = info["pixel_h"], info["pixel_w"]
        clip_h = min(tile_H, pixel_H - ph)
        clip_w = min(tile_W, pixel_W - pw)
        bw = make_blend_weight(tile_H, tile_W, (ov_H, ov_W),
                               n_h, n_w, info["ti"], info["tj"])
        bw = bw[:clip_h, :clip_w]
        norm_bw = bw / total_wt[ph:ph + clip_h, pw:pw + clip_w]
        bw4d = norm_bw[None, :, :, None]
        output[:, ph:ph + clip_h, pw:pw + clip_w, :] += (
            tile_data[:, :clip_h, :clip_w, :] * bw4d
        )
        del tile_data

    return output.astype(np.float16)


# ── Cleanup ──────────────────────────────────────────────────────────────────

def cleanup_tile_files(tile_infos, vpp=None):
    """Remove temporary tile memmap files.

    If *vpp* is provided and tiles have ``"vpp_name"`` keys, cleanup goes
    through ``vpp.cleanup_memmap(name)``.  Otherwise falls back to raw
    ``os.remove`` on ``info["path"]``.
    """
    dirs = set()
    for info in tile_infos:
        vpp_name = info.get("vpp_name")
        if vpp is not None and vpp_name is not None:
            vpp.cleanup_memmap(vpp_name)
        else:
            p = info["path"]
            if os.path.exists(p):
                os.remove(p)
            dirs.add(os.path.dirname(p))
    for d in dirs:
        try:
            os.rmdir(d)
        except OSError:
            pass
