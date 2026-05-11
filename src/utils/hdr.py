"""
HDR utilities — sRGB to PQ inverse tone mapping and HDR10 ProRes writer.

Self-contained helpers for the HDR pipeline. Originally lived in the
``starlight_hdr`` package; copied here so this model directory has no
external dependency on that package and can be shipped standalone.
"""

import colour
import numpy as np
import ffmpeg
import torch
import OpenImageIO
from pathlib import Path
from typing import Dict
import tempfile
import os
import shutil
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


def Rec1886_eotf(n):
    return n ** 2.4


def get_crosstalk_matrix(alpha=0.1):
    return np.array([[1 - 2 * alpha, alpha, alpha], [alpha, 1 - 2 * alpha, alpha], [alpha, alpha, 1 - 2 * alpha]])


def Rec2446c_itm_lin(col, k1, k2, k3, k4, hdr_ip):
    # lin SDR [0-1] -> lin [0-1] HDR
    
    sdr_peak = 100
    hdr_max = 10000

    alpha_desat = 0.1
    alpha_sat = alpha_desat

    XYZ_sdr = colour.RGB_to_XYZ(
        np.einsum('ij,...j->...i', get_crosstalk_matrix(alpha_desat), col).clip(0, 1),
        colourspace=colour.models.RGB_COLOURSPACE_BT709
    )
    
    def get_crosstalk_matrix_inv(alpha=0.1):
        return np.array([[1 - alpha, -alpha, -alpha], [-alpha, 1 - alpha, -alpha], [-alpha, -alpha, 1 - alpha]]) / (1 - 3 * alpha)

    xyY_hdr = colour.XYZ_to_xyY(XYZ_sdr) # sdr
    xyY_hdr[..., 2] = itm(xyY_hdr[..., 2] * sdr_peak, k1=k1, k2=k2, k3=k3, k4=k4, hdr_ip=hdr_ip) / hdr_max    # now hdr

    rgb_hdr = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(xyY_hdr),
        colourspace=colour.models.RGB_COLOURSPACE_BT2020,
    )

    return np.einsum('ij,...j->...i', get_crosstalk_matrix_inv(alpha_sat), rgb_hdr)


def itm(y, k1, k2, k3, k4, hdr_ip):
    return np.where(y < hdr_ip * k1, y / k1, hdr_ip * (np.exp((y - k4) / k2) + k3))


def pq_inv_eotf_norm(l):
    '''
    PQ inv EOTF, l: displayed intensity in nits / 10000 [0, 1]
    output: nonlinearly encoded signal [0, 1]
    '''
    m1 = 2610 / 4096 / 4
    m2 = 2523 / 4096 * 128
    c1 = 3424 / 4096
    c2 = 2413 / 4096 * 32
    c3 = 2392 / 4096 * 32

    lem1 = l.clip(1e-24, None) ** m1
    return ((c1 + c2 * lem1) / (1 + c3 * lem1)) ** m2


def Rec2446c_itm(col, k1, k2, k3, k4, hdr_ip):
    # SDR BT.709 [0-1] -> HDR PQ [0-1]
    return pq_inv_eotf_norm(Rec2446c_itm_lin(Rec1886_eotf(col), k1=k1, k2=k2, k3=k3, k4=k4, hdr_ip=hdr_ip))


def tpdf_dither_numpy(x, n_bit):
    # Triangular [-1, 1] / levels, 0 mean, 2/(levels^2)/3 variance.
    # Applied per-pixel, per-channel, per-frame (temporally fresh).
    levels = 2 ** n_bit - 1
    n = (np.random.rand(*x.shape) + np.random.rand(*x.shape) - 1.0) / float(levels)
    return (x + n).clip(0.0, 1.0)


def srgb_to_pq(x):
    # x is in the range [0, 1]
    k1 = 0.415
    k2 =15.932955017806862
    k3 =0.6774705461982417
    k4 =67.4291076019
    hdr_ip = 119.03614457831326
    x = Rec2446c_itm(x, k1=k1, k2=k2, k3=k3, k4=k4, hdr_ip=hdr_ip)
    x = tpdf_dither_numpy(x, n_bit=8)
    return x


def pq_highlight_curve(x):
    # x in [0,1] (PQ-encoded). Identity for x <= 0.2, smooth monotonic curve above.
    xs = x.clamp(0.0, 1.0).float()

    # KX = torch.tensor([0.20, 0.60, 0.80, 0.85, 1.00], device=x.device, dtype=torch.float32)
    # KY = torch.tensor([0.20, 0.70, 0.90, 0.975, 1.00], device=x.device, dtype=torch.float32)
    # KM = torch.tensor([1.0, 10.0/9.0, 1.2, 0.3, 1.0/6.0], device=x.device, dtype=torch.float32)
    
    KX = torch.tensor([0.20, 0.60, 0.75, 0.82, 1.00], device=x.device, dtype=torch.float32)
    KY = torch.tensor([0.20, 0.70, 0.92, 0.985, 1.00], device=x.device, dtype=torch.float32)
    KM = torch.tensor([1.0, 220.0/163.0, 176.0/137.0, 0.0, 0.0], device=x.device, dtype=torch.float32)

    idx = (torch.bucketize(xs.contiguous(), KX) - 1).clamp(0, KX.numel() - 2)
    x0, x1 = KX[idx], KX[idx + 1]
    y0, y1 = KY[idx], KY[idx + 1]
    m0, m1 = KM[idx], KM[idx + 1]
    h = x1 - x0
    t = (xs - x0) / h
    omt = 1.0 - t
    h00 = (1.0 + 2.0 * t) * omt * omt
    h10 = t * omt * omt
    h01 = t * t * (3.0 - 2.0 * t)
    h11 = t * t * (t - 1.0)
    interp = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

    return torch.where(xs <= 0.2, xs, interp).clamp(0.0, 1.0)


def apply_highlight_curve_max_rgb(rgb):
    rgb = rgb.clamp(0.0, 1.0).float()
    M = rgb.amax(dim=-1, keepdim=True)
    M_new = pq_highlight_curve(M)
    gain = M_new / M.clamp(min=1e-6)
    return (rgb * gain).clamp(0.0, 1.0)


def tpdf_dither(x, n_bit):
    # Triangular [-1, 1] / levels, 0 mean, 2/(levels^2)/3 variance.
    # Applied per-pixel, per-channel, per-frame (temporally fresh).
    levels = 2 ** n_bit - 1
    n = (torch.rand_like(x) + torch.rand_like(x) - 1.0) / float(levels)
    return (x + n).clamp(0.0, 1.0)


def postprocess_frames(frames, n_bit):
    frames = apply_highlight_curve_max_rgb(frames)
    frames = tpdf_dither(frames, n_bit=n_bit)
    return frames


# =============================================================================
# HDR10 Video Writer
# =============================================================================


class HDR10VideoWriter:
    def __init__(self, output_path, width, height, fps):
        # fps = str(float(fps))
        self.process = (
            ffmpeg
            .input("pipe:", format="rawvideo", pix_fmt="rgb48le", s=f"{width}x{height}", r=fps)
            .filter("setparams", range="pc", color_primaries="bt2020",
                    color_trc="smpte2084", colorspace="bt2020nc")
            .output(output_path, vcodec="prores_ks",
                pix_fmt="yuv422p10le", an=None,
                **{"profile:v": "3"})
            .overwrite_output()
            .run_async(pipe_stdin=True)
            )
        
    def write(self, frames):
        for frame in frames:
            self.process.stdin.write(frame.tobytes())
            
    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
            self.process = None
            
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# EXR Image-Sequence Writer
# =============================================================================


SDR_PEAK_NITS = 100.0
HDR_MAX_NITS = 10000.0


_VIDEO_SUFFIXES = {".mov", ".mp4", ".mkv", ".avi", ".webm"}


# (Rx, Ry, Gx, Gy, Bx, By, Wx, Wy) tuples written into the EXR header.
EXR_PRIMARIES = {
    "rec709": (
        (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290),
        "Linear Rec.709",
    ),
    "rec2020": (
        (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290),
        "Linear Rec.2020",
    ),
    "aces2065-1": (
        (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767),
        "ACES2065-1",
    ),
}


def _build_bt2020_to_target_matrix(target: str) -> np.ndarray:
    """Compute the 3x3 RGB->RGB matrix from BT.2020 (D65) to ``target``.

    Uses Bradford CAT for D65 -> D60 (ACES) when needed.
    """
    if target == "rec2020":
        return np.eye(3, dtype=np.float32)
    if target == "rec709":
        return colour.matrix_RGB_to_RGB(
            colour.models.RGB_COLOURSPACE_BT2020,
            colour.models.RGB_COLOURSPACE_BT709,
            chromatic_adaptation_transform="Bradford",
        ).astype(np.float32)
    if target == "aces2065-1":
        return colour.matrix_RGB_to_RGB(
            colour.models.RGB_COLOURSPACE_BT2020,
            colour.models.RGB_COLOURSPACE_ACES2065_1,
            chromatic_adaptation_transform="Bradford",
        ).astype(np.float32)
    raise ValueError(f"Unsupported target primaries: {target!r}")


# Lazy module-level cache so the matrices are computed once per process.
_BT2020_TO_TARGET_CACHE: Dict[str, np.ndarray] = {}


def _get_bt2020_to_target_matrix(target: str) -> np.ndarray:
    M = _BT2020_TO_TARGET_CACHE.get(target)
    if M is None:
        M = _build_bt2020_to_target_matrix(target)
        _BT2020_TO_TARGET_CACHE[target] = M
    return M


def _pq_eotf_norm(e: np.ndarray) -> np.ndarray:
    """PQ EOTF: nonlinear PQ in [0,1] -> normalized linear in [0,1] (1.0 == 10000 nits)."""
    m1 = 2610.0 / 4096.0 / 4.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0
    e = np.clip(e.astype(np.float32), 0.0, 1.0)
    p = np.power(e, 1.0 / m2)
    num = np.maximum(p - c1, 0.0)
    den = c2 - c3 * p
    return np.power(num / np.maximum(den, 1e-12), 1.0 / m1)


def _pq_bt2020_to_linear(
    pq_frame: np.ndarray, target: str, scene_white_nits: float
) -> np.ndarray:
    """PQ BT.2020 [0,1] HxWx3 -> scene-linear in ``target`` primaries.

    ``1.0`` in the output equals ``scene_white_nits`` nits of display-referred
    light. Negatives are preserved for primaries that allow them (AP0).
    """
    lin_nits = _pq_eotf_norm(pq_frame) * HDR_MAX_NITS
    lin_bt2020 = lin_nits / max(scene_white_nits, 1e-6)
    M = _get_bt2020_to_target_matrix(target)
    if target == "rec2020":
        out = lin_bt2020
    else:
        out = np.einsum("ij,...j->...i", M, lin_bt2020)
    if target == "aces2065-1":
        return out.astype(np.float32)
    return np.maximum(out, 0.0).astype(np.float32)


def _write_exr(
    frame_linear: np.ndarray,
    path: str,
    half: bool,
    chromas: tuple,
    cs_tag: str,
) -> None:
    arr = np.ascontiguousarray(frame_linear.astype(np.float32))
    h, w = arr.shape[:2]
    fmt = OpenImageIO.HALF if half else OpenImageIO.FLOAT
    spec = OpenImageIO.ImageSpec(w, h, 3, fmt)
    spec.channelnames = ("R", "G", "B")
    spec.attribute("compression", "zip")
    spec.attribute("chromaticities", "float[8]", chromas)
    spec.attribute("colorSpace", cs_tag)

    out = OpenImageIO.ImageOutput.create(path)
    if out is None:
        raise RuntimeError(
            f"Failed to create EXR writer for '{path}'. Ensure OpenImageIO is built with OpenEXR support."
        )
    try:
        if not out.open(path, spec):
            raise RuntimeError(f"Failed to open EXR file '{path}': {out.geterror()}")
        if not out.write_image(arr):
            raise RuntimeError(f"Failed to write EXR image '{path}': {out.geterror()}")
    finally:
        out.close()


class EXRSequenceWriter:
    """Drop-in replacement for ``HDR10VideoWriter`` that emits an EXR sequence.

    The constructor signature matches ``HDR10VideoWriter`` (positional
    ``output_path, width, height``); EXR-specific options are
    keyword-only.

    Input frames are PQ-encoded BT.2020 in [0, 1] (matching the curved
    pipeline's writer payload). They are PQ-decoded to display-referred nits,
    normalized so that ``scene_white_nits`` maps to 1.0, then converted to the
    requested ``target_primaries`` and written as float-EXR with proper
    chromaticities metadata.

    ``output_path`` may be a directory or a video-style path; if it carries a
    video suffix (``.mov``/``.mp4``/...), the suffix is stripped and the EXRs
    are written into a sibling directory of the same stem.

    Args:
        output_path: Output directory or video-style path used as a folder name.
        width, height: Recorded for compatibility; only ``output_path`` is
            used to set up the file system.
        target_primaries: ``'rec709'`` | ``'rec2020'`` | ``'aces2065-1'``.
        scene_white_nits: Nits value mapped to 1.0 in the EXR.
        half: Use 16-bit half-float EXR storage (default True).
        num_workers: Worker threads for parallel encode + write.
    """

    def __init__(
        self,
        output_path,
        width: int,
        height: int,
        *,
        target_primaries: str = "aces2065-1",
        scene_white_nits: float = SDR_PEAK_NITS,
        half: bool = True,
        num_workers: int = 4,
    ):
        if target_primaries not in EXR_PRIMARIES:
            raise ValueError(
                f"target_primaries must be one of {sorted(EXR_PRIMARIES)}; got {target_primaries!r}"
            )

        self.width = width
        self.height = height
        self.target_primaries = target_primaries
        self.scene_white_nits = float(scene_white_nits)
        self.half = bool(half)

        path = Path(output_path)
        if path.suffix.lower() in _VIDEO_SUFFIXES:
            self.output_dir = path.parent / path.stem
        else:
            self.output_dir = path
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._chromas, self._cs_tag = EXR_PRIMARIES[target_primaries]
        # Pre-warm conversion matrix so worker threads don't race on the cache.
        _get_bt2020_to_target_matrix(target_primaries)

        self.frame_idx = 0
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(num_workers)))
        self._futures: list = []

    def _process_and_save(self, pq_frame: np.ndarray, out_path: str) -> None:
        linear = _pq_bt2020_to_linear(
            pq_frame, self.target_primaries, self.scene_white_nits
        )
        _write_exr(linear, out_path, self.half, self._chromas, self._cs_tag)

    def write(self, frames) -> None:
        """Write a batch of PQ BT.2020 frames.

        Accepts a ``torch.Tensor`` ``[N, H, W, 3]`` (or ``[N, 3, H, W]``) in
        [0, 1], or any iterable of per-frame ``ndarray`` objects (``float`` or
        ``uint16`` in [0, 65535]).
        """
        if self._executor is None:
            raise RuntimeError("EXRSequenceWriter is closed; cannot accept more frames.")

        if isinstance(frames, torch.Tensor):
            arr = frames.detach().float().cpu().numpy()
            if arr.ndim != 4:
                raise ValueError(
                    f"Expected 4-D tensor [N,H,W,3] or [N,3,H,W]; got {arr.shape}"
                )
            if arr.shape[-1] != 3:
                if arr.shape[1] == 3:
                    arr = np.transpose(arr, (0, 2, 3, 1))
                else:
                    raise ValueError(f"Expected 3 channels; got shape {arr.shape}")
            iter_frames = (arr[i] for i in range(arr.shape[0]))
        else:
            iter_frames = frames

        for frame in iter_frames:
            f = np.asarray(frame)
            if f.dtype == np.uint16:
                f = f.astype(np.float32) / 65535.0
            else:
                f = f.astype(np.float32, copy=False)
            out_path = str(self.output_dir / f"frame_{self.frame_idx:06d}.exr")
            self.frame_idx += 1
            self._futures.append(
                self._executor.submit(self._process_and_save, f, out_path)
            )

    def close(self) -> None:
        if self._executor is not None:
            try:
                for fut in self._futures:
                    fut.result()
            finally:
                self._executor.shutdown(wait=True)
                self._executor = None
                self._futures = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# HDRVideoWriter to handle both EXR and ProRes output
# =============================================================================


def copy_audio_to_video(audio_source: str, video_target: str):
    """Mux the audio track from *audio_source* into *video_target* (in-place)."""
    video_dir = os.path.dirname(video_target) or "."
    file_ext = os.path.splitext(video_target)[1]

    with tempfile.NamedTemporaryFile(dir=video_dir, suffix=file_ext, delete=True) as tmp:
        temp_output = tmp.name
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-i", video_target,
            "-i", audio_source,
            "-c:v", "copy", "-c:a", "copy",
            "-map", "0:v:0", "-map", "1:a:0",
            "-movflags", "+faststart",
            temp_output, "-y",
        ]
        try:
            subprocess.run(command, check=True, timeout=300)
            shutil.copy(src=temp_output, dst=video_target)
            logger.info("Audio copied from source into %s", video_target)
        except Exception as e:
            logger.warning("Could not copy audio: %s", e)


class HDRVideoWriter:
    def __init__(self, output_path, width, height, fps, options=None, audio_source_path=None):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_source_path = audio_source_path
        self.writer = None
        self.output_is_exr = False
        
        options = options or dict()
        if options.get('output_codec', 'prores') == 'exr':
            if output_path.lower().endswith('.zip'):
                output_dir = os.path.dirname(output_path) or "."
                self.temp_exr_dir = tempfile.mkdtemp(dir=output_dir)
            else:
                self.temp_exr_dir = output_path
                os.makedirs(output_path, exist_ok=True)
            self.writer = EXRSequenceWriter(self.temp_exr_dir, width, height)
            self.output_is_exr = True
        else:
            self.writer = HDR10VideoWriter(output_path, width, height, fps)
        
    def write(self, frames_tensor):
        frames_tensor = tpdf_dither(frames_tensor, n_bit=8)
        frames_tensor = apply_highlight_curve_max_rgb(frames_tensor.float())
        if not self.output_is_exr:
            frames_tensor = (frames_tensor.float().cpu().numpy() * 65535.0).astype(np.uint16)
        self.writer.write(frames_tensor)
            
    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            if self.output_is_exr and self.output_path.lower().endswith('.zip'):
                archive_base = str(Path(self.output_path).with_suffix(''))
                try:
                    shutil.make_archive(archive_base, 'zip', self.temp_exr_dir)
                    shutil.rmtree(self.temp_exr_dir)
                except Exception as e:
                    logger.warning(f"Failed to create archive: {e}")
                    pass
                self.temp_exr_dir = None
            elif self.audio_source_path is not None:
                copy_audio_to_video(self.audio_source_path, self.output_path)
            
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

