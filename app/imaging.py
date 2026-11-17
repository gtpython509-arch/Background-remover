"""Image decoding, validation, mask refinement and encoding.

Everything here is pure CPU/PIL work and holds no model state, so it is safe
to call from the inference worker threads.
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass

from PIL import Image, ImageFile, ImageFilter, ImageOps, UnidentifiedImageError

from app.errors import PayloadTooLarge, UnsupportedMedia

log = logging.getLogger(__name__)

# Refuse truncated files outright rather than silently processing grey padding.
ImageFile.LOAD_TRUNCATED_IMAGES = False

#: Upper bound enforced by PIL itself; the per-request limit comes from config.
Image.MAX_IMAGE_PIXELS = 100_000_000

_CHECKER_LIGHT = (255, 255, 255)
_CHECKER_DARK = (204, 204, 204)


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image
    """RGB image, EXIF-rotated, downscaled to within ``max_input_dimension``."""
    original_size: tuple[int, int]
    source_format: str | None
    was_downscaled: bool


def parse_colour(value: str) -> tuple[int, int, int, int]:
    """Parse ``#rgb``, ``#rrggbb``, ``#rrggbbaa`` or a CSS colour name."""
    text = value.strip()
    if not text:
        raise ValueError("empty colour")
    if text.lower() in {"transparent", "none"}:
        return (0, 0, 0, 0)
    try:
        from PIL import ImageColor

        rgba = ImageColor.getcolor(text, "RGBA")
    except ValueError as exc:
        raise ValueError(f"unrecognised colour: {value!r}") from exc
    return rgba  # type: ignore[return-value]


def decode_image(
    data: bytes,
    *,
    max_pixels: int,
    max_dimension: int,
    declared_content_type: str | None = None,
) -> DecodedImage:
    """Decode untrusted bytes into a bounded RGB image.

    Validation order matters: the header is inspected *before* the pixel buffer
    is allocated, so a 50-byte file claiming to be 60000x60000 is rejected
    without ever reserving memory for it.
    """
    if not data:
        raise UnsupportedMedia("Empty request body - no image data received.")

    try:
        probe = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise UnsupportedMedia(
            "Could not decode the file as an image. Supported: PNG, JPEG, WebP, BMP, TIFF."
        ) from None
    except Image.DecompressionBombError as exc:
        raise PayloadTooLarge(f"Image rejected as a decompression bomb: {exc}") from None
    except Exception as exc:  # corrupt headers surface as assorted PIL errors
        raise UnsupportedMedia(f"Malformed image file: {exc}") from None

    source_format = probe.format
    width, height = probe.size
    if width <= 0 or height <= 0:
        raise UnsupportedMedia("Image reports zero width or height.")

    pixels = width * height
    if pixels > max_pixels:
        raise PayloadTooLarge(
            f"Image is {width}x{height} ({pixels / 1e6:.1f} MP); "
            f"the limit is {max_pixels / 1e6:.1f} MP."
        )

    try:
        probe.load()
        image = ImageOps.exif_transpose(probe) or probe
    except Image.DecompressionBombError as exc:
        raise PayloadTooLarge(f"Image rejected as a decompression bomb: {exc}") from None
    except OSError as exc:
        raise UnsupportedMedia(f"Image file is truncated or corrupt: {exc}") from None

    # Flatten any input alpha onto white: the network expects three channels,
    # and a transparent PNG would otherwise produce black fringing.
    if image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
        image = flat
    elif image.mode != "RGB":
        image = image.convert("RGB")

    original_size = image.size
    was_downscaled = False
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        target = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        image = image.resize(target, Image.Resampling.LANCZOS)
        was_downscaled = True
        log.info(
            "downscaled oversized input",
            extra={"from": original_size, "to": target},
        )

    if declared_content_type and source_format:
        # Informational only - the magic bytes are what we trust.
        log.debug(
            "decoded image",
            extra={"declared": declared_content_type, "actual": source_format},
        )

    return DecodedImage(
        image=image,
        original_size=original_size,
        source_format=source_format,
        was_downscaled=was_downscaled,
    )


def refine_mask(
    mask: Image.Image,
    *,
    feather: float = 0.0,
    threshold: float | None = None,
    contrast: float = 0.0,
) -> Image.Image:
    """Optional alpha clean-up.

    All three default to off, so the untouched mask is bit-identical to the
    reference implementation.

    * ``contrast`` pushes mid-grey alpha toward 0/255, cutting halo on hair.
    * ``threshold`` hard-cuts to a binary matte (useful for die-cut stickers).
    * ``feather`` blurs the edge by N pixels for softer compositing.
    """
    out = mask
    if contrast > 0:
        # Smoothstep-style curve around 0.5, strength controlled by `contrast`.
        gain = 1.0 + max(0.0, min(contrast, 10.0))
        lut = []
        for i in range(256):
            x = i / 255.0
            y = 1.0 / (1.0 + math.exp(-gain * 6.0 * (x - 0.5)))
            lut.append(int(round(max(0.0, min(1.0, y)) * 255)))
        out = out.point(lut)

    if threshold is not None:
        cut = int(round(max(0.0, min(1.0, threshold)) * 255))
        out = out.point(lambda v, c=cut: 255 if v >= c else 0)

    if feather > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=min(feather, 25.0)))

    return out


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Attach ``mask`` as the alpha channel of ``image``."""
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.BILINEAR)
    cutout = image.convert("RGBA")
    cutout.putalpha(mask.convert("L"))
    return cutout


def composite_on_colour(cutout: Image.Image, colour: str) -> Image.Image:
    """Flatten an RGBA cutout onto a solid colour."""
    rgba = cutout.convert("RGBA")
    r, g, b, a = parse_colour(colour)
    if a == 0:
        return rgba
    backdrop = Image.new("RGBA", rgba.size, (r, g, b, 255))
    return Image.alpha_composite(backdrop, rgba)


def checkerboard(size: tuple[int, int], square: int = 16) -> Image.Image:
    """Transparency backdrop for previews. Built by tiling, not per-pixel."""
    tile = Image.new("RGB", (square * 2, square * 2), _CHECKER_LIGHT)
    dark = Image.new("RGB", (square, square), _CHECKER_DARK)
    tile.paste(dark, (0, 0))
    tile.paste(dark, (square, square))

    board = Image.new("RGB", size)
    for y in range(0, size[1], tile.height):
        for x in range(0, size[0], tile.width):
            board.paste(tile, (x, y))
    return board


def encode_image(
    image: Image.Image,
    output_format: str,
    *,
    png_compress_level: int = 6,
    webp_quality: int = 90,
    jpeg_quality: int = 92,
    background: str | None = None,
) -> tuple[bytes, str, str]:
    """Serialise to bytes. Returns ``(payload, media_type, file_extension)``."""
    buffer = io.BytesIO()
    fmt = output_format.lower()

    if fmt == "png":
        image.convert("RGBA").save(
            buffer, "PNG", optimize=False, compress_level=png_compress_level
        )
        return buffer.getvalue(), "image/png", "png"

    if fmt == "webp":
        image.convert("RGBA").save(
            buffer, "WEBP", quality=webp_quality, method=4, lossless=False
        )
        return buffer.getvalue(), "image/webp", "webp"

    if fmt == "jpeg":
        # JPEG has no alpha, so a backdrop is mandatory.
        flat = composite_on_colour(image, background or "#ffffff").convert("RGB")
        flat.save(buffer, "JPEG", quality=jpeg_quality, optimize=True, progressive=True)
        return buffer.getvalue(), "image/jpeg", "jpg"

    if fmt == "mask":
        mask = image.convert("RGBA").split()[-1] if image.mode == "RGBA" else image.convert("L")
        mask.save(buffer, "PNG", optimize=False, compress_level=png_compress_level)
        return buffer.getvalue(), "image/png", "png"

    raise UnsupportedMedia(f"Unsupported output format: {output_format!r}")
