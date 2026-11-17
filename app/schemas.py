"""Request/response models and the shared set of processing options."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

OutputFormat = Literal["png", "webp", "jpeg", "mask"]


class ProcessingOptions(BaseModel):
    """Knobs a caller may tune per request. All optional; defaults reproduce
    the reference RMBG-1.4 output exactly."""

    output_format: OutputFormat | None = Field(
        default=None, description="png (RGBA), webp (RGBA), jpeg (flattened), mask (grayscale)."
    )
    background: str | None = Field(
        default=None,
        description="Hex colour or CSS name composited behind the subject, e.g. '#ffffff'.",
    )
    feather: float = Field(
        default=0.0, ge=0.0, le=25.0, description="Gaussian edge softening, in pixels."
    )
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Hard-cut the matte to binary at this alpha."
    )
    edge_contrast: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Push mid-grey alpha to 0/255 to reduce halo."
    )
    max_output_dimension: int | None = Field(
        default=None, ge=16, le=8192, description="Downscale the result's longest edge."
    )
    upload: bool = Field(
        default=False, description="Also store the result in the configured S3 bucket."
    )

    @field_validator("background")
    @classmethod
    def _valid_colour(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        from app.imaging import parse_colour

        parse_colour(value)
        return value


class RemoveUrlRequest(ProcessingOptions):
    image_url: HttpUrl = Field(description="Public http(s) URL of the source image.")


class ImageMeta(BaseModel):
    width: int
    height: int
    original_width: int
    original_height: int
    source_format: str | None = None
    was_downscaled: bool = False


class RemoveJsonResponse(BaseModel):
    request_id: str
    filename: str
    content_type: str
    bytes: int
    image_base64: str | None = Field(
        default=None, description="Base64 result; omitted when uploaded to storage."
    )
    storage_url: str | None = None
    meta: ImageMeta
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    uptime_seconds: float


class ReadyResponse(BaseModel):
    # `model_loaded` would otherwise collide with pydantic's reserved namespace.
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ready", "not_ready"]
    model_loaded: bool
    storage_ok: bool
    in_flight: int
    capacity: int


class InfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    app: str
    version: str
    model_id: str
    model_revision: str | None
    device: str
    input_size: int
    half_precision: bool
    model_load_seconds: float | None
    max_upload_bytes: int
    max_pixels: int
    max_input_dimension: int
    supported_output_formats: list[str]
    auth_required: bool
    storage_enabled: bool
    remote_fetch_enabled: bool

