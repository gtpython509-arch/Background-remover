"""Test fixtures.

Two things make the suite fast and hermetic:

* Settings are driven through real environment variables and
  ``get_settings.cache_clear()``. Modules bind ``get_settings`` by reference at
  import time, so patching one module's name would leave the others stale -
  clearing the cache behind the single shared function is what actually works.
* The network is never loaded. ``StubRemover`` returns a deterministic
  elliptical matte, so tests run in about a second and need no weights.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

#: Applied to every test unless a fixture overrides a key.
BASE_ENV = {
    "RMBG_ENVIRONMENT": "development",
    "RMBG_LOG_JSON": "false",
    "RMBG_LOG_LEVEL": "WARNING",
    "RMBG_WARMUP_ON_STARTUP": "false",
    "RMBG_RATE_LIMIT_ENABLED": "false",
    "RMBG_SERVE_UI": "false",
    "RMBG_API_KEYS": "",
    "RMBG_STORAGE_ENABLED": "false",
    "RMBG_ALLOW_REMOTE_FETCH": "false",
    "RMBG_MAX_UPLOAD_BYTES": "2000000",
    "RMBG_MAX_PIXELS": "4000000",
    "RMBG_MAX_INPUT_DIMENSION": "1024",
    "RMBG_MODEL_CACHE_DIR": "/tmp/rmbg-test-models",
}


def _apply_env(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str]) -> None:
    import os

    from app.config import get_settings

    # Drop any RMBG_* left over from the developer's shell or a real .env, so
    # the suite behaves the same locally and in CI.
    for key in [k for k in os.environ if k.startswith("RMBG_")]:
        monkeypatch.delenv(key, raising=False)

    for key, value in {**BASE_ENV, **overrides}.items():
        monkeypatch.setenv(key, value)

    get_settings.cache_clear()


@pytest.fixture
def settings(monkeypatch):
    """Settings built from BASE_ENV, shared by every module in the process."""
    from app.config import get_settings

    _apply_env(monkeypatch, {})
    yield get_settings()
    get_settings.cache_clear()


class StubRemover:
    """Stands in for the real model: an elliptical subject matte."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def info(self):
        from app.model import ModelInfo

        return ModelInfo(
            model_id="stub",
            revision=None,
            device="cpu",
            input_size=1024,
            half_precision=False,
            loaded_at=0.0,
            load_seconds=0.0,
        )

    def load(self):
        return self.info

    def warmup(self) -> float:
        return 0.0

    def unload(self) -> None:
        return None

    def predict_mask(self, image: Image.Image) -> Image.Image:
        from PIL import ImageDraw

        self.calls += 1
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).ellipse(
            [
                image.width * 0.2,
                image.height * 0.2,
                image.width * 0.8,
                image.height * 0.8,
            ],
            fill=255,
        )
        return mask

    def remove_background(self, image: Image.Image):
        from app.imaging import apply_mask

        mask = self.predict_mask(image)
        return apply_mask(image, mask), mask


@pytest.fixture
def stub_remover():
    return StubRemover()


def build_client(monkeypatch, stub, env_overrides: dict[str, str] | None = None):
    """Build a TestClient with the model stubbed out. Returns a context manager."""
    from fastapi.testclient import TestClient

    import app.main as main_module
    import app.ratelimit as ratelimit_module
    from app.config import get_settings

    _apply_env(monkeypatch, env_overrides or {})
    settings = get_settings()

    # main.lifespan constructs the remover through this name.
    monkeypatch.setattr(main_module, "BackgroundRemover", lambda _settings: stub)
    # The limiter caches the old window/limit, so rebuild it per test.
    monkeypatch.setattr(ratelimit_module, "_limiter", None)

    return TestClient(main_module.create_app(settings))


@pytest.fixture
def client(monkeypatch, stub_remover):
    from app.config import get_settings

    with build_client(monkeypatch, stub_remover) as test_client:
        yield test_client
    get_settings.cache_clear()


def make_image(
    size: tuple[int, int] = (256, 256), fmt: str = "PNG", mode: str = "RGB"
) -> bytes:
    """A solid-colour test image, encoded to bytes."""
    image = Image.new(mode, size, 128 if mode == "L" else (90, 140, 200))
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()
