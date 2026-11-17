from __future__ import annotations

import io

import pytest
from PIL import Image

from app.errors import PayloadTooLarge, UnsupportedMedia
from app.imaging import (
    apply_mask,
    composite_on_colour,
    decode_image,
    encode_image,
    parse_colour,
    refine_mask,
)
from tests.conftest import make_image

DECODE_LIMITS = {"max_pixels": 4_000_000, "max_dimension": 512}


def test_decode_png():
    result = decode_image(make_image((128, 96)), **DECODE_LIMITS)
    assert result.image.size == (128, 96)
    assert result.image.mode == "RGB"
    assert result.source_format == "PNG"
    assert result.was_downscaled is False


def test_decode_rejects_empty_body():
    with pytest.raises(UnsupportedMedia):
        decode_image(b"", **DECODE_LIMITS)


def test_decode_rejects_non_image():
    with pytest.raises(UnsupportedMedia):
        decode_image(b"this is definitely not an image", **DECODE_LIMITS)


def test_decode_rejects_too_many_pixels():
    data = make_image((2500, 2500))
    with pytest.raises(PayloadTooLarge):
        decode_image(data, max_pixels=1_000_000, max_dimension=4096)


def test_decode_downscales_oversized_input():
    result = decode_image(make_image((1600, 800)), max_pixels=4_000_000, max_dimension=512)
    assert max(result.image.size) == 512
    assert result.was_downscaled is True
    # Metadata reports the pre-downscale dimensions.
    assert result.original_size == (1600, 800)


def test_decode_flattens_input_alpha_onto_white():
    source = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
    buffer = io.BytesIO()
    source.save(buffer, "PNG")
    result = decode_image(buffer.getvalue(), **DECODE_LIMITS)
    assert result.image.mode == "RGB"
    assert result.image.getpixel((0, 0)) == (255, 255, 255)


def test_decode_grayscale_converts_to_rgb():
    result = decode_image(make_image((64, 64), mode="L"), **DECODE_LIMITS)
    assert result.image.mode == "RGB"


@pytest.mark.parametrize("value", ["#fff", "#ffffff", "white", "#11223344"])
def test_parse_colour_accepts_common_forms(value):
    assert len(parse_colour(value)) == 4


def test_parse_colour_rejects_garbage():
    with pytest.raises(ValueError):
        parse_colour("not-a-colour")


def test_refine_mask_threshold_is_binary():
    mask = Image.linear_gradient("L")
    out = refine_mask(mask, threshold=0.5)
    assert set(out.getdata()) <= {0, 255}


def test_refine_mask_defaults_are_identity():
    mask = Image.linear_gradient("L")
    assert list(refine_mask(mask).getdata()) == list(mask.getdata())


def test_apply_mask_sets_alpha():
    image = Image.new("RGB", (16, 16), (10, 20, 30))
    mask = Image.new("L", (16, 16), 128)
    out = apply_mask(image, mask)
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0))[3] == 128


def test_apply_mask_resizes_mismatched_mask():
    image = Image.new("RGB", (32, 32))
    mask = Image.new("L", (8, 8), 255)
    assert apply_mask(image, mask).size == (32, 32)


def test_composite_on_colour_removes_transparency():
    cutout = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    out = composite_on_colour(cutout, "#0000ff")
    assert out.getpixel((0, 0)) == (0, 0, 255, 255)


@pytest.mark.parametrize(
    ("fmt", "expected_type"),
    [("png", "image/png"), ("webp", "image/webp"), ("jpeg", "image/jpeg"), ("mask", "image/png")],
)
def test_encode_image_formats(fmt, expected_type):
    image = Image.new("RGBA", (24, 24), (1, 2, 3, 200))
    payload, media_type, _ = encode_image(image, fmt)
    assert media_type == expected_type
    assert Image.open(io.BytesIO(payload)).size == (24, 24)


def test_encode_jpeg_has_no_alpha():
    image = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
    payload, _, _ = encode_image(image, "jpeg", background="#00ff00")
    assert Image.open(io.BytesIO(payload)).mode == "RGB"


def test_encode_rejects_unknown_format():
    with pytest.raises(UnsupportedMedia):
        encode_image(Image.new("RGBA", (8, 8)), "gif")
