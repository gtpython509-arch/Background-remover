from __future__ import annotations

import base64
import io

from PIL import Image

from tests.conftest import make_image


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_reports_ready(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_info(client):
    body = client.get("/v1/info").json()
    assert body["supported_output_formats"] == ["png", "webp", "jpeg", "mask"]
    assert body["auth_required"] is False


def test_metrics_exposes_counters(client):
    client.get("/healthz")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rmbg_http_requests_total" in response.text


def test_remove_returns_transparent_png(client):
    response = client.post(
        "/v1/remove", files={"file": ("photo.png", make_image((200, 150)), "image/png")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert 'filename="photo_cutout.png"' in response.headers["content-disposition"]

    result = Image.open(io.BytesIO(response.content))
    assert result.mode == "RGBA"
    assert result.size == (200, 150)
    # Corner is outside the stub's ellipse, centre is inside.
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((100, 75))[3] == 255


def test_remove_request_id_header(client):
    response = client.post(
        "/v1/remove", files={"file": ("a.png", make_image(), "image/png")}
    )
    assert response.headers["X-Request-ID"]


def test_remove_mask_output_is_grayscale(client):
    response = client.post(
        "/v1/remove",
        files={"file": ("a.png", make_image((64, 64)), "image/png")},
        data={"output_format": "mask"},
    )
    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.content)).mode == "L"


def test_remove_jpeg_output_flattens_on_background(client):
    response = client.post(
        "/v1/remove",
        files={"file": ("a.png", make_image((64, 64)), "image/png")},
        data={"output_format": "jpeg", "background": "#00ff00"},
    )
    assert response.status_code == 200
    out = Image.open(io.BytesIO(response.content))
    assert out.mode == "RGB"
    red, green, blue = out.getpixel((1, 1))
    assert green > 200 and red < 60 and blue < 60


def test_remove_rejects_non_image(client):
    response = client.post(
        "/v1/remove", files={"file": ("x.png", b"not an image", "image/png")}
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_remove_rejects_disallowed_content_type(client):
    response = client.post(
        "/v1/remove", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert response.status_code == 415


def test_remove_rejects_oversized_body(client):
    # Larger than the 2 MB test limit; blocked by Content-Length up front.
    response = client.post(
        "/v1/remove", files={"file": ("big.png", b"\x00" * 2_500_000, "image/png")}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_remove_rejects_too_many_pixels(client):
    response = client.post(
        "/v1/remove", files={"file": ("big.png", make_image((2100, 2100)), "image/png")}
    )
    assert response.status_code == 413


def test_remove_rejects_bad_options(client):
    response = client.post(
        "/v1/remove",
        files={"file": ("a.png", make_image(), "image/png")},
        data={"background": "octarine"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_options"


def test_remove_missing_file_is_422(client):
    response = client.post("/v1/remove", data={"output_format": "png"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_remove_json_returns_base64(client):
    response = client.post(
        "/v1/remove/json",
        files={"file": ("a.png", make_image((80, 80)), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["content_type"] == "image/png"
    assert body["meta"]["width"] == 80
    assert "inference" in body["timings_ms"]
    decoded = base64.b64decode(body["image_base64"])
    assert Image.open(io.BytesIO(decoded)).size == (80, 80)


def test_remove_json_can_omit_base64(client):
    response = client.post(
        "/v1/remove/json",
        files={"file": ("a.png", make_image(), "image/png")},
        data={"include_base64": "false"},
    )
    assert response.json()["image_base64"] is None


def test_remove_json_upload_without_storage_is_rejected(client):
    response = client.post(
        "/v1/remove/json",
        files={"file": ("a.png", make_image(), "image/png")},
        data={"upload": "true"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "storage_disabled"


def test_max_output_dimension_downscales_result(client):
    response = client.post(
        "/v1/remove",
        files={"file": ("a.png", make_image((400, 200)), "image/png")},
        data={"max_output_dimension": "100"},
    )
    assert Image.open(io.BytesIO(response.content)).size == (100, 50)


def test_batch_processes_several_files(client):
    response = client.post(
        "/v1/remove/batch",
        files=[
            ("files", ("a.png", make_image((64, 64)), "image/png")),
            ("files", ("b.png", make_image((48, 48)), "image/png")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["succeeded"] == 2


def test_batch_isolates_a_single_bad_file(client):
    response = client.post(
        "/v1/remove/batch",
        files=[
            ("files", ("good.png", make_image((32, 32)), "image/png")),
            ("files", ("bad.png", b"garbage", "image/png")),
        ],
    )
    body = response.json()
    assert body["succeeded"] == 1
    failures = [r for r in body["results"] if not r["ok"]]
    assert failures[0]["error"]["code"] == "unsupported_media_type"


def test_remote_fetch_disabled_by_default(client):
    response = client.post(
        "/v1/remove/url", json={"image_url": "https://example.com/a.png"}
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "remote_fetch_disabled"


def test_security_headers_present(client):
    headers = client.get("/healthz").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
