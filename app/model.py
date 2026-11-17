"""RMBG model wrapper.

Holds the single loaded network and performs one forward pass at a time. The
pre/post-processing reproduces the reference RMBG-1.4 pipeline exactly:

    resize to 1024x1024 bilinear -> scale to 0..1 -> subtract mean 0.5
    -> forward -> take d1 -> bilinear resize to the original size
    -> min-max normalise -> 8-bit alpha

Nothing in here touches FastAPI, so it is reusable from the batch CLI.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.config import Settings

log = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    model_id: str
    revision: str | None
    device: str
    input_size: int
    half_precision: bool
    loaded_at: float
    load_seconds: float


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class BackgroundRemover:
    """Thread-safe facade over the segmentation network."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: torch.nn.Module | None = None
        self._device: str = "cpu"
        self._half = False
        self._info: ModelInfo | None = None
        # One forward pass at a time. Even with several worker threads this
        # keeps peak memory at a single activation graph, which is what makes
        # the memory ceiling on a small VPS predictable.
        self._infer_lock = threading.Lock()

    # ------------------------------------------------------------------ state
    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def info(self) -> ModelInfo | None:
        return self._info

    # ------------------------------------------------------------------- load
    def load(self) -> ModelInfo:
        if self._model is not None and self._info is not None:
            return self._info

        settings = self._settings
        started = time.perf_counter()

        if settings.torch_threads > 0:
            torch.set_num_threads(settings.torch_threads)
            log.info("torch threads pinned", extra={"threads": settings.torch_threads})

        self._device = resolve_device(settings.device)
        self._half = settings.use_half_precision and self._device == "cuda"

        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "loading model",
            extra={
                "model_id": settings.model_id,
                "revision": settings.model_revision,
                "device": self._device,
                "cache_dir": str(settings.model_cache_dir),
                "local_files_only": settings.local_files_only,
            },
        )

        # Imported lazily so `--help`, tests and config errors do not pay the
        # multi-second transformers import cost.
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            settings.model_id,
            revision=settings.model_revision,
            trust_remote_code=settings.trust_remote_code,
            cache_dir=str(settings.model_cache_dir),
            local_files_only=settings.local_files_only,
        )
        model.to(self._device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self._model = model
        elapsed = time.perf_counter() - started
        self._info = ModelInfo(
            model_id=settings.model_id,
            revision=settings.model_revision,
            device=self._device,
            input_size=settings.model_input_size,
            half_precision=self._half,
            loaded_at=time.time(),
            load_seconds=elapsed,
        )
        log.info("model loaded", extra={"seconds": round(elapsed, 2)})
        return self._info

    def warmup(self) -> float:
        """Run one throwaway pass so the first real request is not the slowest.

        Lazy kernel/allocator initialisation otherwise shows up as a multi-second
        outlier on request #1.
        """
        dummy = Image.new("RGB", (512, 512), (127, 127, 127))
        started = time.perf_counter()
        self.predict_mask(dummy)
        elapsed = time.perf_counter() - started
        log.info("warmup complete", extra={"seconds": round(elapsed, 2)})
        return elapsed

    def unload(self) -> None:
        self._model = None
        self._info = None
        if self._device == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------- inference
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        size = self._settings.model_input_size
        resized = image.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        # RMBG-1.4 normalisation: mean 0.5, std 1.0.
        tensor = tensor - 0.5
        return tensor.to(self._device)

    @staticmethod
    def _extract_prediction(output: object) -> torch.Tensor:
        """Pull the finest-scale sigmoid map (``d1``) out of the model output.

        The checkpoint returns a nested list of six side outputs plus encoder
        features; older/newer wrappers return a bare tensor or a dict.
        """
        candidate = output
        if isinstance(candidate, dict):
            for key in ("logits", "prediction", "out", "alphas"):
                if key in candidate:
                    candidate = candidate[key]
                    break
        # Descend into (list|tuple) nesting until we reach a 4-D tensor.
        depth = 0
        while isinstance(candidate, list | tuple) and candidate and depth < 4:
            candidate = candidate[0]
            depth += 1
        if not isinstance(candidate, torch.Tensor):
            raise RuntimeError(f"unexpected model output type: {type(output)!r}")
        if candidate.ndim == 3:
            candidate = candidate.unsqueeze(0)
        if candidate.ndim != 4:
            raise RuntimeError(f"unexpected prediction shape: {tuple(candidate.shape)}")
        return candidate

    def predict_mask(self, image: Image.Image) -> Image.Image:
        """Return an 8-bit ``L`` alpha mask at the input image's resolution."""
        if self._model is None:
            raise RuntimeError("model is not loaded; call load() first")

        width, height = image.size
        tensor = self._preprocess(image)

        with self._infer_lock, torch.inference_mode():
            if self._half:
                with torch.autocast("cuda", dtype=torch.float16):
                    raw = self._model(tensor)
            else:
                raw = self._model(tensor)

            prediction = self._extract_prediction(raw).float()
            prediction = F.interpolate(
                prediction, size=(height, width), mode="bilinear", align_corners=False
            )
            prediction = prediction[0]

            lo = torch.min(prediction)
            hi = torch.max(prediction)
            prediction = (prediction - lo) / torch.clamp(hi - lo, min=1e-8)
            alpha = (prediction.squeeze(0) * 255.0).clamp(0, 255).to(torch.uint8)
            alpha_np = alpha.cpu().numpy()

        return Image.fromarray(alpha_np, mode="L")

    def remove_background(
        self, image: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        """Return ``(cutout_rgba, mask_L)`` - the notebook's public behaviour."""
        from app.imaging import apply_mask

        mask = self.predict_mask(image)
        return apply_mask(image, mask), mask


_remover: BackgroundRemover | None = None


def get_remover() -> BackgroundRemover:
    """Process-wide singleton, created by the app lifespan or the CLI."""
    global _remover
    if _remover is None:
        from app.config import get_settings

        _remover = BackgroundRemover(get_settings())
    return _remover


def set_remover(remover: BackgroundRemover | None) -> None:
    """Injection point for tests."""
    global _remover
    _remover = remover
