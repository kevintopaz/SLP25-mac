"""Spatially tiled, temporally streamed VAE encode / decode (plain PyTorch).

Wraps any VAE that streams time itself through ``slicing_encode`` /
``slicing_decode`` (causal-conv memory carried between temporal slices), and
adds the spatial side: tiles with overlap, blending, and decoded tiles written
to disk so a long clip never has to fit in memory at once.

Inputs and outputs match what the pipeline already consumes:

* ``tiled_encode`` reads ``[T, H, W, C]`` frames (raw integer pixels from the
  pipeline's memmap with ``pixel_scale``, or floats in ``[-1, 1]``) and
  returns the posterior mean ``[1, T_lat, H/8, W/8, 16]`` float32.
* ``tiled_decode`` takes that latent layout and returns ``tile_infos`` for
  ``video_utils.tile_assembly`` (one memmap file per spatial tile).
"""

import logging
import math
import os
import tempfile
import time
from typing import Optional

import numpy as np
import torch

from .tile_assembly import make_blend_weight
from .transform_memmap import drop_mapped_pages

logger = logging.getLogger(__name__)


def _hw(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _n_tiles(size, tile, overlap):
    return max(1, int(math.ceil((size - overlap) / (tile - overlap))))


def plan_tile(extent: int, max_tile: int, overlap: int) -> int:
    """Tile edge along one axis (all in latent rows): the fewest tiles whose edge
    fits ``max_tile``, each just long enough that n tiles overlapping by ``overlap``
    cover ``extent`` -- the only work beyond the frame is the overlap itself.
    Fixed-size tiles instead redo up to a whole tile of extra rows at the edge
    (today's 480 px decode tiles decode 1.65x a 1080p latent)."""
    if extent <= max_tile:
        return extent
    n = 2
    while True:
        tile = -(-(extent + (n - 1) * overlap) // n)
        if tile <= max_tile or tile <= overlap + 1:
            return max(tile, overlap + 1)
        n += 1


def _auto_tile(extent_lat: int, max_tile_px, overlap_lat: int) -> int:
    return plan_tile(extent_lat, max(overlap_lat + 1, int(max_tile_px) // 8), overlap_lat) * 8


class TiledVAE:
    """Tiled encode / decode around ``vae`` (an ``nn.Module`` with
    ``slicing_encode``, ``slicing_decode`` and ``set_causal_slicing``).

    dtype: the dtype the VAE runs in (inputs are cast to it).
    """

    def __init__(self, vae: torch.nn.Module, dtype: torch.dtype = torch.bfloat16, latent_channels: int = 16):
        self.vae = vae
        self.dtype = dtype
        self.latent_channels = latent_channels
        self.temporal_down = 4

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def _set_slices(self, encode_frames: Optional[int] = None, decode_latents: Optional[int] = None):
        """Temporal slice sizes. The causal VAE takes 1 + k*4 pixel frames first and
        k*4 after that, so the requested size is rounded down to that grid."""
        if encode_frames is not None:
            split = max(self.temporal_down, ((encode_frames - 1) // self.temporal_down) * self.temporal_down)
            self.vae.set_causal_slicing(split_size=split, memory_device="same")
        if decode_latents is not None:
            self.vae.slicing_latent_min_size = max(1, int(decode_latents))

    # ------------------------------------------------------------------ #
    #  Encode                                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def tiled_encode(self, frames, tile_size=640, tile_overlap=80, temporal_slice=9,
                     input_layout: str = "THWC", pixel_scale: float = 0.0, progress_fn=None,
                     blend_mode: str = "cosine", max_tile=None) -> np.ndarray:
        """frames: ``[T, H, W, C]`` (or ``[C, T, H, W]`` with input_layout="CTHW"),
        numpy / memmap / CPU tensor. With ``pixel_scale > 0`` values are raw
        pixels in ``[0, pixel_scale]``, normalised to ``[-1, 1]`` per tile."""
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu()
            if frames.dtype == torch.bfloat16:
                frames = frames.to(torch.float16)
            frames = frames.numpy()
        frames = np.asarray(frames) if not isinstance(frames, np.memmap) else frames
        cthw = input_layout == "CTHW"
        if cthw:
            _, T, H, W = frames.shape
        else:
            T, H, W, _ = frames.shape

        tile_H, tile_W = _hw(tile_size)
        ov_H, ov_W = _hw(tile_overlap)
        if max_tile:                        # per-axis tiles sized to the frame (see plan_tile)
            mh, mw = _hw(max_tile)
            tile_H, tile_W = _auto_tile(H // 8, mh, ov_H // 8), _auto_tile(W // 8, mw, ov_W // 8)
        tile_H, tile_W = min(tile_H, H), min(tile_W, W)
        n_h, n_w = _n_tiles(H, tile_H, ov_H), _n_tiles(W, tile_W, ov_W)
        lat_tile_H, lat_tile_W, lat_ov_H, lat_ov_W = tile_H // 8, tile_W // 8, ov_H // 8, ov_W // 8
        T_lat = 1 if T == 1 else (T - 1) // self.temporal_down + 1

        self._set_slices(encode_frames=temporal_slice)
        latent = torch.zeros(T_lat, H // 8, W // 8, 2 * self.latent_channels, dtype=torch.float32)
        weight = torch.zeros_like(latent)
        device = self.device
        t0 = time.perf_counter()
        for ti in range(n_h):
            for tj in range(n_w):
                hs = min(ti * (tile_H - ov_H), H - tile_H)
                ws = min(tj * (tile_W - ov_W), W - tile_W)
                if cthw:
                    crop = np.asarray(frames[:, :, hs:hs + tile_H, ws:ws + tile_W]).transpose(1, 2, 3, 0)
                else:
                    crop = np.asarray(frames[:, hs:hs + tile_H, ws:ws + tile_W, :])
                if pixel_scale > 0 and torch.cuda.is_available():
                    # normalise on the GPU (same float32 steps as below: identical values); the
                    # VAE takes the CUDA tensor directly
                    xt = torch.from_numpy(np.ascontiguousarray(crop)).to("cuda")
                    xt = ((xt.float() / pixel_scale) * 2.0 - 1.0).half()
                    x = xt.permute(3, 0, 1, 2)[None]
                    del crop, xt
                    xdev = x.device
                else:
                    if pixel_scale > 0:
                        crop = ((crop.astype(np.float32) / pixel_scale) * 2.0 - 1.0).astype(np.float16)
                    x = torch.from_numpy(np.ascontiguousarray(crop)).permute(3, 0, 1, 2)[None]   # 1 C T H W
                    del crop
                    xdev = device
                moments = self.vae.slicing_encode(x.to(device=xdev, dtype=self.dtype))
                del x
                drop_mapped_pages(frames)    # the crop is on the device now: unmap the input pages it touched
                tile_lat = moments[0].float().permute(1, 2, 3, 0).cpu()                      # T h w 2C
                del moments
                bw = torch.from_numpy(make_blend_weight(lat_tile_H, lat_tile_W, (lat_ov_H, lat_ov_W),
                                                        n_h, n_w, ti, tj, blend_mode=blend_mode))[None, :, :, None]
                lh, lw = hs // 8, ws // 8
                region = (slice(None), slice(lh, lh + lat_tile_H), slice(lw, lw + lat_tile_W))
                latent[region] += tile_lat * bw
                weight[region] += bw.expand_as(tile_lat)
                del tile_lat
                torch.cuda.empty_cache()
                if progress_fn is not None:
                    progress_fn(ti * n_w + tj + 1, n_h * n_w)
        latent = latent / torch.clamp(weight, min=1e-8)
        mean = latent[..., :self.latent_channels][None].numpy().astype(np.float32)
        logger.debug("Encoded %d tiles in %.1fs -> %s", n_h * n_w, time.perf_counter() - t0, mean.shape)
        return mean

    # ------------------------------------------------------------------ #
    #  Decode                                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def tiled_decode(self, latent_f32_np: np.ndarray, tile_size=480, tile_overlap=60, temporal_slice=5,
                     num_frames: Optional[int] = None, tmp_dir: Optional[str] = None, progress_fn=None,
                     tile_dtype: str = "uint8", vpp=None, blend_mode: str = "cosine", max_tile=None) -> list:
        """latent_f32_np: ``[1, T_lat, H_lat, W_lat, 16]``. Each spatial tile is
        decoded (temporal slices of ``temporal_slice`` latent frames), quantised
        and written to its own memmap; returns the tile descriptions."""
        _, T_lat, lat_H, lat_W, _ = latent_f32_np.shape
        tile_H, tile_W = _hw(tile_size)
        ov_H, ov_W = _hw(tile_overlap)
        if max_tile:                        # per-axis tiles sized to the frame (see plan_tile)
            mh, mw = _hw(max_tile)
            tile_H, tile_W = _auto_tile(lat_H, mh, ov_H // 8), _auto_tile(lat_W, mw, ov_W // 8)
        tile_H, tile_W = min(tile_H, lat_H * 8), min(tile_W, lat_W * 8)
        lat_tile_H, lat_tile_W, lat_ov_H, lat_ov_W = tile_H // 8, tile_W // 8, ov_H // 8, ov_W // 8
        n_h, n_w = _n_tiles(lat_H, lat_tile_H, lat_ov_H), _n_tiles(lat_W, lat_tile_W, lat_ov_W)
        logger.info("VAE decode tiles: %dx%d of %dx%d px (overlap %dx%d px), %.2fx the latent area",
                    n_h, n_w, tile_H, tile_W, lat_ov_H * 8, lat_ov_W * 8,
                    (n_h * lat_tile_H) * (n_w * lat_tile_W) / max(1, lat_H * lat_W))
        T_pix = 1 if T_lat == 1 else (T_lat - 1) * self.temporal_down + 1
        if num_frames is not None:
            T_pix = min(T_pix, num_frames)
        if vpp is None and tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="vae_tiles_")

        self._set_slices(decode_latents=temporal_slice)
        device = self.device
        tile_infos = []
        t0 = time.perf_counter()
        from concurrent.futures import ThreadPoolExecutor
        writer = ThreadPoolExecutor(max_workers=1)        # tile downloads + disk writes, off the GPU's path
        pending = []
        for ti in range(n_h):
            for tj in range(n_w):
                lh = min(ti * (lat_tile_H - lat_ov_H), lat_H - lat_tile_H)
                lw = min(tj * (lat_tile_W - lat_ov_W), lat_W - lat_tile_W)
                z = torch.from_numpy(np.ascontiguousarray(
                    latent_f32_np[0, :, lh:lh + lat_tile_H, lw:lw + lat_tile_W, :])).permute(3, 0, 1, 2)[None]
                out = self.vae.slicing_decode(z.to(device=device, dtype=self.dtype))
                del z
                if tile_dtype in ("uint8", "uint16") and torch.cuda.is_available():
                    # same float32 math as the numpy path below, on the GPU (identical values);
                    # the download and the disk write run in the background (writer) while the
                    # next tile decodes
                    # in chunks of frames into a result allocated first: whole-tile float copies
                    # (~0.5 GB at 1080p) would fragment PyTorch's cache, and the result, kept until
                    # the writer has saved it, would pin those blocks during the next tile's decode
                    o = out[0].to("cuda")
                    del out
                    n_t = min(T_pix, o.shape[1])
                    q = torch.empty((n_t, o.shape[2], o.shape[3], o.shape[0]), device="cuda",
                                    dtype=torch.uint8 if tile_dtype == "uint8" else torch.int32)
                    for f0 in range(0, n_t, 16):
                        t = o[:, f0:f0 + 16].to(torch.float16).permute(1, 2, 3, 0).float()
                        if tile_dtype == "uint8":
                            q[f0:f0 + t.shape[0]] = ((t + 1.0) * 127.5).clamp_(0, 255)
                        else:
                            q[f0:f0 + t.shape[0]] = ((t + 1.0) * 32767.5).clamp_(0, 65535)
                        del t
                    del o
                    shape = tuple(q.shape)
                    name = f"tile_{ti}_{tj}"
                    if vpp is not None:
                        tmm = vpp.create_memmap(name, shape[0], shape[1], shape[2], shape[3], mode=tile_dtype)
                        path = tmm.path

                        def _write(q=q, tmm=tmm):
                            a = q.cpu().numpy()
                            tmm.write_raw_batch(a if tile_dtype == "uint8" else a.astype(np.uint16), 0)
                            tmm.release()   # unmap: assembly opens its own read-only mapping per read
                    else:
                        path = os.path.join(tmp_dir, f"{name}.dat")
                        mm = np.memmap(path, dtype=np.dtype(tile_dtype), mode="w+", shape=shape)

                        def _write(q=q, mm=mm):
                            a = q.cpu().numpy()
                            mm[:] = a if tile_dtype == "uint8" else a.astype(np.uint16)
                            mm.flush()
                            drop_mapped_pages(mm)
                    pending.append(writer.submit(_write))
                    del q
                    info = {"shape": shape, "dtype": tile_dtype, "pixel_h": lh * 8, "pixel_w": lw * 8,
                            "tile_H": tile_H, "tile_W": tile_W, "ti": ti, "tj": tj, "n_h": n_h, "n_w": n_w,
                            "ov_H": ov_H, "ov_W": ov_W, "path": path}
                    if vpp is not None:
                        info["vpp_name"] = name
                    tile_infos.append(info)
                    torch.cuda.empty_cache()
                    if progress_fn is not None:
                        progress_fn(ti * n_w + tj + 1, n_h * n_w)
                    continue
                else:
                    tile = out[0].to(torch.float16).permute(1, 2, 3, 0).cpu().numpy()          # T H W 3
                    del out
                    tile = tile[:T_pix]
                    if tile_dtype == "uint8":
                        quantized = ((tile.astype(np.float32) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    elif tile_dtype == "uint16":
                        quantized = ((tile.astype(np.float32) + 1.0) * 32767.5).clip(0, 65535).astype(np.uint16)
                    else:
                        quantized = tile.astype(np.dtype(tile_dtype))
                    del tile
                shape = quantized.shape
                name = f"tile_{ti}_{tj}"
                if vpp is not None:
                    tmm = vpp.create_memmap(name, shape[0], shape[1], shape[2], shape[3], mode=tile_dtype)
                    tmm.write_raw_batch(quantized, 0)
                    tmm.release()
                    path = tmm.path
                else:
                    path = os.path.join(tmp_dir, f"{name}.dat")
                    mm = np.memmap(path, dtype=np.dtype(tile_dtype), mode="w+", shape=shape)
                    mm[:] = quantized
                    mm.flush()
                    del mm
                del quantized
                info = {"shape": shape, "dtype": tile_dtype, "pixel_h": lh * 8, "pixel_w": lw * 8,
                        "tile_H": tile_H, "tile_W": tile_W, "ti": ti, "tj": tj, "n_h": n_h, "n_w": n_w,
                        "ov_H": ov_H, "ov_W": ov_W, "path": path}
                if vpp is not None:
                    info["vpp_name"] = name
                tile_infos.append(info)
                torch.cuda.empty_cache()
                if progress_fn is not None:
                    progress_fn(ti * n_w + tj + 1, n_h * n_w)
        for f in pending:                                 # every tile on disk before anyone reads it
            f.result()
        writer.shutdown()
        logger.debug("Decoded %d tiles in %.1fs", len(tile_infos), time.perf_counter() - t0)
        return tile_infos
