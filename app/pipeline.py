"""The synchronous processing pipeline.

One function, no framework imports, so it can be driven by the HTTP layer, the
batch CLI, or a test. It runs entirely on a worker thread.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from PIL import Image

from app.config import Settings
from app.imaging import (
    apply_mask,
    composite_on_colour,
    decode_image,
    encode_image,
    refine_mask,
)
from app.model import BackgroundRemover
from app.schemas import ImageMeta, ProcessingOptions

log = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    payload: bytes
    media_type: str
    extension: str
    meta: ImageMeta
    timings_ms: dict[str, float] = field(default_factory=dict)


def _resize_output(image: Image.Image, longest_edge: int) -> Image.Image:
    if max(image.size) <= longest_edge:
        return image
    scale = longest_edge / max(image.size)
    target = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale)))
    )
    return image.resize(target, Image.Resampling.LANCZOS)


def process_image(
    data: bytes,
    options: ProcessingOptions,
    settings: Settings,
    remover: BackgroundRemover,
    *,
    declared_content_type: str | None = None,
) -> ProcessResult:
    """Decode -> segment -> refine -> composite -> encode."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    decoded = decode_image(
        data,
        max_pixels=settings.max_pixels,
        max_dimension=settings.max_input_dimension,
        declared_content_type=declared_content_type,
    )
    timings["decode"] = (time.perf_counter() - t0) * 1000

    source = decoded.image

    t0 = time.perf_counter()
    mask = remover.predict_mask(source)
    timings["inference"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if options.feather or options.threshold is not None or options.edge_contrast:
        mask = refine_mask(
            mask,
            feather=options.feather,
            threshold=options.threshold,
            contrast=options.edge_contrast,
        )

    output_format = options.output_format or settings.default_output_format
    background = options.background or settings.default_background

    if output_format == "mask":
        result = mask
    else:
        result = apply_mask(source, mask)
        # `jpeg` flattens inside encode_image; for png/webp a requested colour
        # is composited here so the alpha channel is preserved as fully opaque.
        if background and output_format != "jpeg":
            result = composite_on_colour(result, background)

    if options.max_output_dimension:
        result = _resize_output(result, options.max_output_dimension)
    timings["postprocess"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    payload, media_type, extension = encode_image(
        result,
        output_format,
        png_compress_level=settings.png_compress_level,
        webp_quality=settings.webp_quality,
        jpeg_quality=settings.jpeg_quality,
        background=background,
    )
    timings["encode"] = (time.perf_counter() - t0) * 1000

    meta = ImageMeta(
        width=result.width,
        height=result.height,
        original_width=decoded.original_size[0],
        original_height=decoded.original_size[1],
        source_format=decoded.source_format,
        was_downscaled=decoded.was_downscaled,
    )

    return ProcessResult(
        payload=payload,
        media_type=media_type,
        extension=extension,
        meta=meta,
        timings_ms={k: round(v, 2) for k, v in timings.items()},
    )
