from __future__ import annotations

import pytest

from app.errors import ApiError
from app.ratelimit import SlidingWindowLimiter
from tests.conftest import build_client, make_image


@pytest.fixture
def auth_client(stub_remover, monkeypatch):
    with build_client(
        monkeypatch,
        stub_remover,
        {"RMBG_API_KEYS": "secret-key-one,secret-key-two"},
    ) as client:
        yield client


def test_request_without_key_is_401(auth_client):
    response = auth_client.post(
        "/v1/remove", files={"file": ("a.png", make_image(), "image/png")}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_request_with_wrong_key_is_401(auth_client):
    response = auth_client.post(
        "/v1/remove",
        files={"file": ("a.png", make_image(), "image/png")},
        headers={"X-API-Key": "nope"},
    )
    assert response.status_code == 401


def test_request_with_valid_key_succeeds(auth_client):
    response = auth_client.post(
        "/v1/remove",
        files={"file": ("a.png", make_image(), "image/png")},
        headers={"X-API-Key": "secret-key-two"},
    )
    assert response.status_code == 200


def test_bearer_token_is_accepted(auth_client):
    response = auth_client.post(
        "/v1/remove",
        files={"file": ("a.png", make_image(), "image/png")},
        headers={"Authorization": "Bearer secret-key-one"},
    )
    assert response.status_code == 200


def test_health_endpoints_stay_public(auth_client):
    assert auth_client.get("/healthz").status_code == 200
    assert auth_client.get("/readyz").status_code == 200


def test_limiter_allows_up_to_the_limit():
    limiter = SlidingWindowLimiter(limit=3, window_seconds=60)
    assert [limiter.check("a")[0] for _ in range(4)] == [True, True, True, False]


def test_limiter_buckets_are_independent():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=60)
    assert limiter.check("a")[0] is True
    assert limiter.check("b")[0] is True
    assert limiter.check("a")[0] is False


def test_limiter_reports_retry_after():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=30)
    limiter.check("a")
    allowed, remaining, retry_after = limiter.check("a")
    assert allowed is False and remaining == 0 and 0 < retry_after <= 30


def test_rate_limited_request_returns_429(stub_remover, monkeypatch):
    overrides = {
        "RMBG_RATE_LIMIT_ENABLED": "true",
        "RMBG_RATE_LIMIT_REQUESTS": "2",
        "RMBG_RATE_LIMIT_WINDOW_SECONDS": "60",
    }
    with build_client(monkeypatch, stub_remover, overrides) as client:
        statuses = [
            client.post(
                "/v1/remove",
                files={"file": ("a.png", make_image((32, 32)), "image/png")},
            ).status_code
            for _ in range(3)
        ]

    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429


def test_fetcher_blocks_private_addresses():
    from app.config import Settings
    from app.fetcher import fetch_image_bytes

    settings = Settings(allow_remote_fetch=True)
    for url in (
        "http://127.0.0.1/x.png",
        "http://localhost/x.png",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.5/x.png",
    ):
        with pytest.raises(ApiError) as excinfo:
            fetch_image_bytes(url, settings)
        assert excinfo.value.status_code == 400


def test_fetcher_rejects_non_http_schemes():
    from app.config import Settings
    from app.fetcher import fetch_image_bytes

    settings = Settings(allow_remote_fetch=True)
    with pytest.raises(ApiError):
        fetch_image_bytes("file:///etc/passwd", settings)
