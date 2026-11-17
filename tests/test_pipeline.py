from __future__ import annotations

import io

from PIL import Image

from app.pipeline import process_image
from app.schemas import ProcessingOptions
from tests.conftest import make_image


def test_pipeline_returns_png_by_default(settings, stub_remover):
    result = process_image(make_image((100, 50)), ProcessingOptions(), settings, stub_remover)
    assert result.media_type == "image/png"
    assert result.extension == "png"
    assert result.meta.width == 100 and result.meta.height == 50
    assert set(result.timings_ms) == {"decode", "inference", "postprocess", "encode"}


def test_pipeline_honours_output_format(settings, stub_remover):
    result = process_image(
        make_image((64, 64)), ProcessingOptions(output_format="webp"), settings, stub_remover
    )
    assert result.media_type == "image/webp"


def test_pipeline_records_downscaling_in_meta(settings, stub_remover):
    # settings.max_input_dimension is 1024 in the fixture.
    result = process_image(make_image((1600, 400)), ProcessingOptions(), settings, stub_remover)
    assert result.meta.was_downscaled is True
    assert result.meta.original_width == 1600
    assert max(result.meta.width, result.meta.height) == 1024


def test_pipeline_threshold_produces_binary_alpha(settings, stub_remover):
    result = process_image(
        make_image((64, 64)),
        ProcessingOptions(output_format="mask", threshold=0.5),
        settings,
        stub_remover,
    )
    mask = Image.open(io.BytesIO(result.payload))
    assert set(mask.getdata()) <= {0, 255}


def test_pipeline_calls_the_model_once_per_image(settings, stub_remover):
    before = stub_remover.calls
    process_image(make_image(), ProcessingOptions(), settings, stub_remover)
    assert stub_remover.calls == before + 1
