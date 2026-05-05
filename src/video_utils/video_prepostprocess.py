"""Model-independent video pre- and post-processing facade.

Contains ``ResizeDivisible`` (shortest-edge resize with divisibility
adjustment), ``VideoWriter`` (lazy callback-driven writer), and the
``VideoPrePostProcess`` facade that composes them with ``ColorMatcher``,
``USMSharpener``, and ``TransformMemmap`` for disk-backed storage.

Typical usage::

    with VideoPrePostProcess(resolution=1080, source_bit_depth=8,
                             color_method="lab", usm_weight=0.5) as vpp:
        # Ingest
        vpp.ingest(reader_fn=reader_callback,
                   total_frames=121, padded_frames=121)

        # Open writer (factory called lazily on first frame)
        vpp.open_writer(writer_factory=my_factory,
                        total_frames=121, progress_fn=progress)

        # ... VAE encode / DiT / VAE decode ...

        # Post-process + write (streaming callback)
        frames = vpp.color_correct_and_finalize(decoded, reference=ref)
        vpp.write_frames(frames)

        vpp.close_writer()
    # memmaps + writer cleaned up automatically on __exit__
"""

import logging
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from .color_matcher import ColorFn, ColorMatcher
from .transform_memmap import TransformMemmap
from .usm_sharpener import USMSharpener

logger = logging.getLogger(__name__)


# ====================================================================== #
#  ResizeDivisible                                                      #
# ====================================================================== #

class ResizeDivisible:
    """Shortest-edge resize with divisibility adjustment.

    Computes the natural target dimensions (shortest edge = *resolution*),
    rounds both dimensions up to the nearest multiple of *factor* (e.g. 16),
    and adjusts the image to the rounded target using the chosen *strategy*.

    Strategies
    ----------
    ``"resize"`` (default)
        Forward:  single resize to padded dims (no double interpolation).
        Backward: bicubic resize from padded dims back to true dims.

    ``"pad"``
        Forward:  resize to true dims, then ``F.pad`` to padded dims.
        Backward: crop to true dims (lossless, no interpolation).

    After ``__call__``, the instance stores:
        * ``true_dims``         — (H, W) before div rounding  (e.g. 1080, 1920)
        * ``padded_dims``       — (H, W) after div rounding   (e.g. 1088, 1920)
        * ``needs_resize_back`` — whether post-processing must undo the adjustment

    Call ``resize_back(tensor)`` to reverse the adjustment (padded → true).
    """

    def __init__(
        self,
        resolution: int,
        max_resolution: int = 0,
        downsample_only: bool = False,
        factor: int = 16,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        strategy: Literal["resize", "pad"] = "resize",
        pad_mode: str = "replicate",
    ):
        self.resolution = resolution
        self.max_resolution = max_resolution
        self.downsample_only = downsample_only
        self.factor = factor
        self.interpolation = interpolation
        self.strategy = strategy
        self.pad_mode = pad_mode

        self.true_dims: Optional[Tuple[int, int]] = None
        self.padded_dims: Optional[Tuple[int, int]] = None
        self.needs_resize_back: bool = False
        self._pad_bottom: int = 0
        self._pad_right: int = 0

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _round_up(x: int, factor: int) -> int:
        return x + (factor - x % factor) % factor

    def _compute_dims(self, height: int, width: int) -> Tuple[int, int, int, int]:
        """Return (true_h, true_w, padded_h, padded_w)."""
        size = self.resolution
        if self.downsample_only and min(width, height) < size:
            size = min(width, height)

        if height <= width:
            true_h = size
            true_w = round(width * size / height)
        else:
            true_w = size
            true_h = round(height * size / width)

        if self.max_resolution > 0 and max(true_h, true_w) > self.max_resolution:
            scale = self.max_resolution / max(true_h, true_w)
            true_h, true_w = round(true_h * scale), round(true_w * scale)

        # Round to even for codec compatibility (libx264)
        true_h = (true_h // 2) * 2
        true_w = (true_w // 2) * 2

        padded_h = self._round_up(true_h, self.factor)
        padded_w = self._round_up(true_w, self.factor)

        return true_h, true_w, padded_h, padded_w

    @staticmethod
    def _is_hwc(tensor: torch.Tensor) -> bool:
        """Heuristic: [T, H, W, C] when last dim is a small channel count."""
        return tensor.ndim == 4 and tensor.shape[-1] <= 16 and tensor.shape[-2] > 16

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def __call__(self, image: Union[torch.Tensor, Image.Image]) -> Union[torch.Tensor, Image.Image]:
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        true_h, true_w, padded_h, padded_w = self._compute_dims(height, width)

        self.true_dims = (true_h, true_w)
        self.padded_dims = (padded_h, padded_w)
        self.needs_resize_back = (true_h != padded_h or true_w != padded_w)
        self._pad_bottom = padded_h - true_h
        self._pad_right = padded_w - true_w

        antialias = not (isinstance(image, torch.Tensor) and image.device.type == 'mps')

        if self.strategy == "pad":
            resized = TVF.resize(image, (true_h, true_w), self.interpolation, antialias=antialias)
            if self.needs_resize_back and isinstance(resized, torch.Tensor):
                resized = F.pad(resized, (0, self._pad_right, 0, self._pad_bottom),
                                mode=self.pad_mode)
            elif self.needs_resize_back and isinstance(resized, Image.Image):
                t = TVF.to_tensor(resized)  # [C, H, W]
                t = F.pad(t, (0, self._pad_right, 0, self._pad_bottom),
                          mode=self.pad_mode)
                resized = TVF.to_pil_image(t)
            return resized

        # strategy == "resize": single resize to padded dims
        return TVF.resize(image, (padded_h, padded_w), self.interpolation, antialias=antialias)

    # ------------------------------------------------------------------ #
    #  Backward                                                            #
    # ------------------------------------------------------------------ #

    def resize_back(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse the forward adjustment (padded -> true dims).

        * ``strategy="resize"``: bicubic interpolation back to true dims.
        * ``strategy="pad"``:    crop away the padding (lossless).

        No-op when the tensor already has the true dimensions or when
        forward did not change them.

        Accepts common layouts:
            [T, C, H, W]  or  [C, T, H, W]  — H, W at dims -2, -1
            [T, H, W, C]                     — H, W at dims -3, -2
        """
        if self.true_dims is None or not self.needs_resize_back:
            return tensor
        true_h, true_w = self.true_dims

        if self.strategy == "pad":
            return self._crop_back(tensor, true_h, true_w)
        return self._resize_back(tensor, true_h, true_w)

    # -- resize strategy ------------------------------------------------ #

    @staticmethod
    def _resize_back(tensor: torch.Tensor, true_h: int, true_w: int) -> torch.Tensor:
        h, w = tensor.shape[-2], tensor.shape[-1]

        if ResizeDivisible._is_hwc(tensor):
            if tensor.shape[-3] == true_h and tensor.shape[-2] == true_w:
                return tensor
            t = tensor.permute(0, 3, 1, 2)
            t = F.interpolate(t.float(), size=(true_h, true_w),
                              mode='bicubic', align_corners=False).to(tensor.dtype)
            return t.permute(0, 2, 3, 1)

        if h == true_h and w == true_w:
            return tensor

        return F.interpolate(tensor.float(), size=(true_h, true_w),
                             mode='bicubic', align_corners=False).to(tensor.dtype)

    # -- pad strategy --------------------------------------------------- #

    @staticmethod
    def _crop_back(tensor: torch.Tensor, true_h: int, true_w: int) -> torch.Tensor:
        if ResizeDivisible._is_hwc(tensor):
            if tensor.shape[-3] == true_h and tensor.shape[-2] == true_w:
                return tensor
            return tensor[:, :true_h, :true_w, :]

        h, w = tensor.shape[-2], tensor.shape[-1]
        if h == true_h and w == true_w:
            return tensor
        return tensor[..., :true_h, :true_w]


# ====================================================================== #
#  VideoWriter                                                             #
# ====================================================================== #

WriterFactory = Callable[[int, int], Tuple[
    Callable[[np.ndarray], None], Callable[[], None]]]
"""``(H, W) -> (write_fn, close_fn)`` factory signature."""


class VideoWriter:
    """Lazy, callback-driven video writer with progress tracking.

    The writer is initialised with a *factory* rather than a concrete writer
    because output dimensions are only known when the first decoded chunk
    arrives.  The factory is invoked once, on the first ``write_frames``
    call.

    Parameters
    ----------
    writer_factory : ``(H, W) -> (write_fn, close_fn)``
        Factory that receives the frame height and width and returns:

        * ``write_fn(frame: np.ndarray)`` — write one ``[H, W, C]`` frame.
        * ``close_fn()`` — release resources (e.g. close ffmpeg pipe).
    total_frames : int
        Expected frame count for progress reporting (0 = unknown).
    progress_fn : callable, optional
        ``progress_fn(fraction, message=...)`` called after each batch.
    """

    def __init__(
        self,
        writer_factory: WriterFactory,
        total_frames: int = 0,
        progress_fn: Optional[Callable[..., Any]] = None,
    ):
        self._factory = writer_factory
        self._write_fn: Optional[Callable[[np.ndarray], None]] = None
        self._close_fn: Optional[Callable[[], None]] = None
        self._total_frames = total_frames
        self._written = 0
        self._progress_fn = progress_fn

    # -- public API ------------------------------------------------------- #

    def write_frames(self, frames: np.ndarray) -> None:
        """Write a batch of finalised frames.

        Parameters
        ----------
        frames : np.ndarray
            ``[T, H, W, C]`` uint8 or uint16 — typically the output of
            ``VideoPrePostProcess.color_correct_and_finalize``.
        """
        if self._write_fn is None:
            H, W = frames.shape[1], frames.shape[2]
            self._write_fn, self._close_fn = self._factory(H, W)

        for i in range(frames.shape[0]):
            self._write_fn(frames[i])
        self._written += frames.shape[0]

        if self._progress_fn and self._total_frames > 0:
            frac = self._written / self._total_frames
            self._progress_fn(
                frac,
                message=f"Processed {self._written}"
                        f"/{self._total_frames} frames...",
            )
        logger.info("Saved %d/%s frames",
                     self._written, self._total_frames or "?")

    def close(self) -> None:
        """Finalise and release the writer.  Idempotent."""
        if self._close_fn is not None:
            try:
                self._close_fn()
            except Exception:
                logger.exception("Error closing writer")
            self._close_fn = None
        self._write_fn = None

    @property
    def frames_written(self) -> int:
        """Number of frames written so far."""
        return self._written

    def __del__(self):
        self.close()


# ====================================================================== #
#  VideoPrePostProcess facade                                              #
# ====================================================================== #


class VideoPrePostProcess:
    """Facade that orchestrates spatial resize, color correction, sharpening,
    and disk-backed reference storage for video pre/post-processing.

    Parameters
    ----------
    resolution : int
        Shortest-edge target size (e.g. 1080 for 1080p).
    max_resolution : int
        Upper-bound on the longest edge (0 = uncapped).
    factor : int
        Divisibility factor (default 16 for typical VAE/DiT).
    strategy : ``"resize"`` or ``"pad"``
        How to reach the div-aligned dimensions (see ``ResizeDivisible``).
    pad_mode : str
        Padding mode when ``strategy="pad"`` (default ``"replicate"``).
    source_bit_depth : int
        Bit depth of the source video (8 or 10+).  Controls memmap
        storage mode (uint8 for 8-bit, uint16 for >8-bit).
    output_bit_depth : int, optional
        Bit depth for output quantization.  Defaults to *source_bit_depth*
        when ``None``, but can be set independently (e.g. 16 for SDR→HDR).
    color_method : str
        Color correction method (default ``"lab"``).
    usm_weight : float
        USM sharpening strength (0 = disabled).
    usm_radius : int
        USM kernel radius.
    usm_threshold : int
        USM edge-strength threshold (0–255 scale).
    color_overrides : dict, optional
        Mapping of ``{method_name: callable}`` forwarded to
        ``ColorMatcher(overrides=...)``.  Use this to inject
        platform-specific backends (e.g. MLX wavelet) instead of
        monkey-patching ``color_fix`` at the module level.
    """

    def __init__(
        self,
        resolution: int,
        max_resolution: int = 0,
        factor: int = 16,
        strategy: Literal["resize", "pad"] = "resize",
        pad_mode: str = "replicate",
        source_bit_depth: int = 8,
        output_bit_depth: Optional[int] = None,
        color_method: str = "lab",
        usm_weight: float = 0.0,
        usm_radius: int = 51,
        usm_threshold: int = 10,
        color_overrides: Optional[Dict[str, ColorFn]] = None,
    ):
        self._resizer = ResizeDivisible(
            resolution=resolution,
            max_resolution=max_resolution,
            factor=factor,
            strategy=strategy,
            pad_mode=pad_mode,
        )
        self.source_bit_depth = source_bit_depth
        self.output_bit_depth = output_bit_depth if output_bit_depth is not None else source_bit_depth
        self._color_matcher = ColorMatcher(method=color_method, overrides=color_overrides)
        self._usm_sharpener = USMSharpener(
            weight=usm_weight,
            radius=usm_radius,
            threshold=usm_threshold,
        )
        self._memmaps: Dict[str, TransformMemmap] = {}
        self._writer: Optional[VideoWriter] = None

    # ------------------------------------------------------------------ #
    #  Runtime reconfiguration                                             #
    # ------------------------------------------------------------------ #

    def configure(
        self,
        *,
        color_method: Optional[str] = None,
        output_bit_depth: Optional[int] = None,
        usm_weight: Optional[float] = None,
        usm_radius: Optional[int] = None,
        usm_threshold: Optional[int] = None,
    ) -> None:
        """Update post-processing parameters in place.

        Only non-None arguments are applied, so callers can set just the
        subset they care about.
        """
        if color_method is not None:
            self._color_matcher.method = color_method
        if output_bit_depth is not None:
            self.output_bit_depth = output_bit_depth
        if usm_weight is not None:
            self._usm_sharpener.weight = usm_weight
        if usm_radius is not None:
            self._usm_sharpener.radius = usm_radius if usm_radius % 2 == 1 else usm_radius + 1
        if usm_threshold is not None:
            self._usm_sharpener.threshold = usm_threshold

    # ------------------------------------------------------------------ #
    #  Delegated properties                                                #
    # ------------------------------------------------------------------ #

    @property
    def true_dims(self) -> Optional[Tuple[int, int]]:
        """(H, W) before div rounding — set after ``resize()``."""
        return self._resizer.true_dims

    @property
    def padded_dims(self) -> Optional[Tuple[int, int]]:
        """(H, W) after div rounding — set after ``resize()``."""
        return self._resizer.padded_dims

    @property
    def needs_resize_back(self) -> bool:
        return self._resizer.needs_resize_back

    @property
    def memmap_mode(self) -> str:
        """Memmap storage mode derived from ``source_bit_depth``."""
        return "uint16" if self.source_bit_depth > 8 else "uint8"

    # ------------------------------------------------------------------ #
    #  Spatial resize                                                      #
    # ------------------------------------------------------------------ #

    def resize(self, image):
        """Forward spatial adjustment (wraps ``ResizeDivisible.__call__``)."""
        return self._resizer(image)

    def resize_back(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse the spatial adjustment (wraps ``ResizeDivisible.resize_back``)."""
        return self._resizer.resize_back(tensor)

    # ------------------------------------------------------------------ #
    #  Named memmaps — disk-backed storage for pipeline stages             #
    # ------------------------------------------------------------------ #
    #                                                                      #
    #  Standard names:                                                     #
    #    "input"   — resized input video (for VAE encode + color ref)      #
    #    "decoded" — decoded output tiles (for spatial blending)            #
    #                                                                      #
    #  Any string key is allowed; these are just conventions.              #
    # ------------------------------------------------------------------ #

    def create_memmap(
        self, name: str, T: int, H: int, W: int, C: int = 3,
        mode: Optional[str] = None,
    ) -> TransformMemmap:
        """Create a named memmap.

        Parameters
        ----------
        name : str
            Key for this memmap (e.g. ``"input"``, ``"decoded"``).
        T, H, W, C : int
            Shape of the memmap ``(T, H, W, C)``.
        mode : str, optional
            Storage mode (``"uint8"``, ``"uint16"``, ``"raw"``).
            If *None*, derived from ``source_bit_depth``.

        .. note:: Does NOT auto-cleanup a previous memmap with the same
           name.  Call ``cleanup_memmap(name)`` explicitly first if needed.

        Returns the created ``TransformMemmap``.
        """
        if mode is None:
            mode = self.memmap_mode
        mm = TransformMemmap(shape=(T, H, W, C), mode=mode)
        self._memmaps[name] = mm
        return mm

    def get_memmap(self, name: str) -> Optional[TransformMemmap]:
        """Return the named memmap, or None if it doesn't exist."""
        return self._memmaps.get(name)

    def ingest(
        self,
        reader_fn: Callable[[int, int], np.ndarray],
        total_frames: int,
        padded_frames: int,
        batch_size: int = 4,
        name: str = "input",
    ) -> TransformMemmap:
        """Read, resize, and store video frames to a named memmap.

        Parameters
        ----------
        reader_fn : ``(start, count) -> np.ndarray [B, H, W, C]``
            Callback that returns ``count`` frames starting at ``start``
            as raw uint8 or uint16 pixels.  The caller provides whatever
            platform-specific decoder they need (PyAV, decord, OpenCV, etc.).
        total_frames : int
            Number of actual source frames to read.
        padded_frames : int
            Total temporal length of the memmap (>= total_frames).
            Extra frames are mirror-padded.
        batch_size : int
            How many frames to process per iteration (default 4).
        name : str
            Memmap slot name (default ``"input"``).

        Returns
        -------
        TransformMemmap
            The populated memmap.
        """
        self.cleanup_memmap(name)

        # Probe first frame to infer bit depth and initialise resizer dims.
        probe = reader_fn(0, 1)  # np.ndarray [1, H, W, C] uint8/uint16
        C = probe.shape[-1]

        # Auto-detect source bit depth from reader dtype
        if probe.dtype == np.uint16:
            self.source_bit_depth = 16
        else:
            self.source_bit_depth = 8
        src_scale = 65535.0 if self.source_bit_depth > 8 else 255.0

        # Resize probe (float tensor) to set dimensions
        probe_float = torch.from_numpy(probe[0].astype(np.float32) / src_scale).permute(2, 0, 1)
        self._resizer(probe_float)  # sets true_dims / padded_dims
        H, W = self._resizer.padded_dims
        del probe, probe_float

        mm = self.create_memmap(name, padded_frames, H, W, C)

        for start in range(0, total_frames, batch_size):
            count = min(batch_size, total_frames - start)
            frames_np = reader_fn(start, count)  # [B, H, W, C] uint8/uint16

            # Convert to float [0,1] for resize, then back to integer
            frames_f = torch.from_numpy(frames_np.astype(np.float32) / src_scale)
            batch_chw = frames_f.permute(0, 3, 1, 2)  # [B, C, H, W]
            batch_out = torch.stack([self._resizer(f) for f in batch_chw])
            batch_out = batch_out.cpu().float().clamp(0.0, 1.0)
            # Quantize back to integer
            resized_np = (batch_out.permute(0, 2, 3, 1).numpy() * src_scale).clip(0, src_scale).astype(mm.dtype)
            mm.write_raw_batch(resized_np, start)
            del frames_np, frames_f, batch_chw, batch_out, resized_np

        if padded_frames > total_frames:
            mm.mirror_pad(total_frames)

        mm.flush()
        logger.info("Ingested %d frames to disk cache: %s", total_frames, mm)
        return mm

    def read_tile(
        self,
        start: int,
        count: int,
        top: int = 0,
        left: int = 0,
        tile_h: Optional[int] = None,
        tile_w: Optional[int] = None,
        name: str = "input",
    ) -> np.ndarray:
        """Read a spatial tile from the memmap, normalized to [-1, 1] float16.

        This is the single read entry-point for memmap data.
        ``read_reference`` is a convenience wrapper that returns a torch
        tensor.

        Parameters
        ----------
        start : int
            First frame index.
        count : int
            Number of frames.
        top, left : int
            Top-left corner of the spatial crop (default 0, 0).
        tile_h, tile_w : int, optional
            Height and width of the tile.  If *None*, reads the full
            spatial extent.
        name : str
            Memmap slot name (default ``"input"``).

        Returns
        -------
        np.ndarray
            ``[count, tile_h, tile_w, C]`` float16 in ``[-1, 1]``.
        """
        mm = self._memmaps.get(name)
        if mm is None:
            raise RuntimeError(f"No memmap[{name!r}] — call ingest() or create_memmap() first")

        if tile_h is None:
            tile_h = mm.shape[1]
        if tile_w is None:
            tile_w = mm.shape[2]

        mm._ensure_mapped()
        raw = np.array(mm._mm[start:start + count, top:top + tile_h, left:left + tile_w, :])
        scale = 65535.0 if self.source_bit_depth > 8 else 255.0
        return (raw.astype(np.float32) / scale * 2.0 - 1.0).astype(np.float16)

    def read_reference(
        self, start: int = 0, count: Optional[int] = None,
        as_cthw: bool = True, name: str = "input",
    ) -> torch.Tensor:
        """Read frames as ``[-1, 1]`` float16 torch tensor (delegates to ``read_tile``).

        When *count* is ``None``, reads all frames from *start* to the end.
        """
        mm = self._memmaps.get(name)
        if mm is None:
            raise RuntimeError(f"No memmap[{name!r}] — call ingest() or create_memmap() first")
        if count is None:
            count = mm.shape[0] - start
        tile_np = self.read_tile(start, count, name=name)
        t = torch.from_numpy(tile_np)
        if as_cthw:
            t = t.permute(3, 0, 1, 2)  # THWC → CTHW
        return t

    def cleanup_memmap(self, name: str = "input") -> None:
        """Delete a named memmap's backing file and remove it."""
        mm = self._memmaps.pop(name, None)
        if mm is not None:
            mm.cleanup()

    def cleanup_all_memmaps(self) -> None:
        """Delete all memmap backing files."""
        for name in list(self._memmaps):
            self.cleanup_memmap(name)

    # Backward compat: self.memmap points to the "input" slot
    @property
    def memmap(self) -> Optional[TransformMemmap]:
        return self._memmaps.get("input")

    @memmap.setter
    def memmap(self, value: Optional[TransformMemmap]) -> None:
        if value is None:
            self._memmaps.pop("input", None)
        else:
            self._memmaps["input"] = value

    # ------------------------------------------------------------------ #
    #  Color correction                                                    #
    # ------------------------------------------------------------------ #

    def color_correct(
        self,
        content: torch.Tensor,
        reference: torch.Tensor,
        debug: Optional[object] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Apply color correction."""
        return self._color_matcher.apply(content, reference, debug=debug, device=device)

    # ------------------------------------------------------------------ #
    #  Denormalization helper                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def denormalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
        """Convert from ``[-1, 1]`` → ``[0, 1]`` with clamping."""
        return tensor.clamp(-1, 1).mul(0.5).add(0.5)

    # ------------------------------------------------------------------ #
    #  Full postprocess pipeline                                           #
    # ------------------------------------------------------------------ #

    @property
    def output_dtype(self) -> np.dtype:
        """NumPy dtype for final pixel output based on ``output_bit_depth``."""
        return np.uint16 if self.output_bit_depth > 8 else np.uint8

    @property
    def output_scale(self) -> float:
        """Scale factor for ``[0, 1]`` → pixel value conversion."""
        return 65535.0 if self.output_bit_depth > 8 else 255.0

    def color_correct_and_finalize(
        self,
        decoded: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> np.ndarray:
        """Color-correct, denormalize, sharpen, and quantize decoded frames.

        Returns pixel values in the native integer format determined by
        ``output_bit_depth`` (uint8 for 8-bit, uint16 for 10/16-bit).

        Parameters
        ----------
        decoded : Tensor
            ``[T, H, W, C]`` in ``[-1, 1]`` on CPU.
        reference : Tensor, optional
            ``[C, T, H, W]`` in ``[-1, 1]`` on CPU (for color correction).
            Pass ``None`` to skip color correction.
        device : str
            GPU device for color correction / sharpening.

        Returns
        -------
        np.ndarray
            ``[T, H, W, C]`` uint8 ``[0, 255]`` or uint16 ``[0, 65535]``.
        """
        T = decoded.shape[0]
        if T == 0:
            return np.empty((0,) + tuple(decoded.shape[1:]),
                            dtype=self.output_dtype)

        if self._color_matcher.method != "none" and reference is not None:
            ref_cpu = reference.permute(1, 0, 2, 3)  # [C,T,H,W] → [T,C,H,W]
            T_ref = ref_cpu.shape[0]
            indices = (torch.linspace(0, T_ref - 1, T).long()
                       if T_ref != T else None)

            video = decoded.permute(0, 3, 1, 2).to(device)  # [T,H,W,C] → [T,C,H,W]

            H, W = video.shape[-2], video.shape[-1]
            pix = H * W
            if pix > 4_000_000:
                chunk_sz = max(1, min(4, T))
            elif pix > 2_000_000:
                chunk_sz = max(1, min(8, T))
            else:
                chunk_sz = max(1, min(32, T))

            result_chunks = []
            for s in range(0, T, chunk_sz):
                e = min(s + chunk_sz, T)
                vc = video[s:e]
                if indices is not None:
                    rc = ref_cpu[indices[s:e]].to(device)
                else:
                    rc = ref_cpu[s:e].to(device)
                vc = self._color_matcher.apply(vc, rc)
                vc.clamp_(-1, 1).mul_(0.5).add_(0.5)
                result_chunks.append(
                    vc.permute(0, 2, 3, 1).float().cpu()
                )
                del rc
            del video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            video_out = torch.cat(result_chunks, dim=0)
            del result_chunks, ref_cpu
        else:
            video_out = self.denormalize_to_01(decoded).float()

        if self._usm_sharpener.weight > 0:
            video_out = self._usm_sharpener.apply(video_out, device=device)

        scale = self.output_scale
        return (video_out.cpu().numpy() * scale).clip(0, scale).astype(self.output_dtype)

    # ------------------------------------------------------------------ #
    #  Video writer                                                        #
    # ------------------------------------------------------------------ #

    def open_writer(
        self,
        writer_factory: WriterFactory,
        total_frames: int = 0,
        progress_fn: Optional[Callable[..., Any]] = None,
    ) -> VideoWriter:
        """Create and register a :class:`VideoWriter` for streaming output.

        Parameters
        ----------
        writer_factory : ``(H, W) -> (write_fn, close_fn)``
            See :class:`VideoWriter` for the factory protocol.
        total_frames : int
            Expected frame count for progress reporting (0 = unknown).
        progress_fn : callable, optional
            ``progress_fn(fraction, message=...)`` called after each batch.

        Returns
        -------
        VideoWriter
            The newly created writer.
        """
        self.close_writer()
        self._writer = VideoWriter(writer_factory, total_frames, progress_fn)
        return self._writer

    def write_frames(self, frames: np.ndarray) -> None:
        """Write a batch of finalised frames via the registered writer.

        Parameters
        ----------
        frames : np.ndarray
            ``[T, H, W, C]`` uint8 or uint16 — typically the output of
            ``color_correct_and_finalize``.
        """
        if self._writer is None:
            raise RuntimeError("No writer. Call open_writer() first.")
        self._writer.write_frames(frames)

    def close_writer(self) -> None:
        """Close and discard the current writer.  Idempotent."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def frames_written(self) -> int:
        """Number of frames written so far (0 if no writer)."""
        return self._writer.frames_written if self._writer else 0

    # ------------------------------------------------------------------ #
    #  Resource management                                                 #
    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        """Release all managed resources (memmaps + writer)."""
        self.close_writer()
        self.cleanup_all_memmaps()

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
