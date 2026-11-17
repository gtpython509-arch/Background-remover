"""Application configuration.

Every value can be overridden with an environment variable prefixed `RMBG_`
(e.g. ``RMBG_MAX_UPLOAD_BYTES=20971520``) or placed in a ``.env`` file next to
the project root. See ``.env.example`` for a documented template.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Device = Literal["auto", "cpu", "cuda"]
OutputFormat = Literal["png", "webp", "jpeg", "mask"]

#: List settings read from the environment as plain comma-separated text.
#:
#: Without NoDecode, pydantic-settings JSON-decodes any env var whose field is
#: a complex type, and does it *before* field validators run - so
#: `RMBG_API_KEYS=key1,key2` would raise a SettingsError at startup rather than
#: reaching the splitter below. NoDecode hands the raw string to the validator
#: instead.
CsvList = Annotated[list[str], NoDecode]


def _split_csv(value: object) -> object:
    """Allow list-valued settings to be given as a comma-separated string."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RMBG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_id` / `model_dir` would otherwise collide with pydantic's
        # reserved `model_` namespace.
        protected_namespaces=(),
    )

    # ---------------------------------------------------------------- runtime
    app_name: str = "RMBG Background Removal API"
    environment: Literal["development", "staging", "production"] = "production"
    debug: bool = False

    host: str = "127.0.0.1"
    port: int = 8000
    root_path: str = ""
    """Set to e.g. ``/api`` when nginx mounts the app under a sub-path."""

    serve_ui: bool = True
    """Serve the bundled drag-and-drop test page at ``/``."""

    enable_docs: bool | None = None
    """Expose ``/docs`` and ``/openapi.json``.

    ``None`` means "on outside production" - the safe default. Set it to true
    in production only when you need the schema reachable, e.g. to import the
    API into Postman from a deployed host. It publishes your full endpoint and
    parameter list, so prefer importing the bundled collection instead.
    """

    # ------------------------------------------------------------------ model
    model_id: str = "briaai/RMBG-1.4"
    model_revision: str | None = None
    """Pin a git revision/commit of the model repo for reproducible deploys."""
    model_cache_dir: Path = Path("/var/lib/rmbg/models")
    local_files_only: bool = False
    """True in production once the model has been baked in / pre-downloaded."""
    trust_remote_code: bool = True
    """RMBG-1.4 ships custom modelling code; required for that checkpoint."""

    device: Device = "auto"
    model_input_size: int = 1024
    """Resolution the network runs at. 1024 matches the reference implementation."""
    use_half_precision: bool = True
    """fp16 inference. Only applied on CUDA; ignored on CPU."""
    torch_threads: int = 0
    """0 = let torch decide. On a 2-vCPU VPS, 2 is a sane explicit value."""
    warmup_on_startup: bool = True

    # -------------------------------------------------------------- inference
    inference_workers: int = 1
    """Concurrent forward passes. Keep at 1: torch already parallelises
    internally, and >1 multiplies peak memory without improving throughput."""
    max_queued_requests: int = 16
    """Requests waiting for the model. Beyond this the API returns 503 instead
    of silently building an unbounded backlog."""
    inference_timeout_seconds: float = 120.0

    # ------------------------------------------------------------------ input
    max_upload_bytes: int = 15 * 1024 * 1024
    max_pixels: int = 50_000_000
    """Decompression-bomb guard, checked before the image is fully decoded."""
    max_input_dimension: int = 4096
    """Inputs larger than this are downscaled first, bounding peak RAM."""
    allowed_content_types: CsvList = Field(
        default_factory=lambda: [
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/webp",
            "image/bmp",
            "image/tiff",
        ]
    )

    # ----------------------------------------------------------------- output
    default_output_format: OutputFormat = "png"
    png_compress_level: int = 6
    webp_quality: int = 90
    jpeg_quality: int = 92
    default_background: str | None = None
    """Hex colour (``#ffffff``) composited behind the cutout for jpeg output."""

    # ------------------------------------------------------------------- auth
    api_keys: CsvList = Field(default_factory=list)
    """Comma-separated list. Empty disables auth (dev only - logged as a warning)."""
    api_key_header: str = "X-API-Key"

    cors_origins: CsvList = Field(default_factory=list)
    cors_allow_credentials: bool = False

    # ------------------------------------------------------------- rate limit
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    trusted_proxy_count: int = 1
    """How many reverse proxies sit in front of the app. Used to read the real
    client IP out of X-Forwarded-For without letting clients spoof it."""

    # ---------------------------------------------------------- remote fetch
    allow_remote_fetch: bool = False
    """Enables ``POST /v1/remove/url``. Off by default - it is an SSRF vector."""
    remote_fetch_timeout_seconds: float = 10.0
    remote_fetch_allowed_hosts: CsvList = Field(default_factory=list)
    """Empty = any public host. Private/loopback/link-local IPs are always blocked."""

    # -------------------------------------------------------------- storage
    storage_enabled: bool = False
    s3_endpoint_url: str | None = None
    """e.g. https://s3.us-west-004.backblazeb2.com for Backblaze B2."""
    s3_region: str | None = None
    s3_bucket: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_key_prefix: str = "cutouts/"
    s3_public_base_url: str | None = None
    """Public CDN/bucket base URL used to build the returned link."""
    s3_presign_seconds: int = 0
    """If > 0 and no public base URL, return a presigned GET URL instead."""

    # -------------------------------------------------------------- logging
    log_level: str = "INFO"
    log_json: bool = True
    log_access: bool = True
    metrics_enabled: bool = True

    # ------------------------------------------------------------ validators
    @field_validator(
        "api_keys",
        "cors_origins",
        "allowed_content_types",
        "remote_fetch_allowed_hosts",
        mode="before",
    )
    @classmethod
    def _accept_csv(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"invalid log level: {value}")
        return level

    @field_validator("default_background")
    @classmethod
    def _check_colour(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        from app.imaging import parse_colour  # local import avoids a cycle

        parse_colour(value)  # raises ValueError on bad input
        return value

    @model_validator(mode="after")
    def _check_storage(self) -> Settings:
        if self.storage_enabled:
            missing = [
                name
                for name in (
                    "s3_bucket",
                    "s3_access_key_id",
                    "s3_secret_access_key",
                )
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(
                    "storage_enabled=true but missing: "
                    + ", ".join(f"RMBG_{m.upper()}" for m in missing)
                )
        return self

    # ---------------------------------------------------------------- helpers
    @property
    def auth_enabled(self) -> bool:
        """Auth is on only when at least one key is configured.

        With ``RMBG_API_KEYS`` empty the service is wide open; startup logs a
        warning, and that is the intended behaviour for local testing.
        """
        return bool(self.api_keys)

    @property
    def docs_enabled(self) -> bool:
        if self.enable_docs is None:
            return not self.is_production
        return self.enable_docs

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor so the env is parsed exactly once per process."""
    return Settings()
