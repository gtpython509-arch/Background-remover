"""ASGI application factory.

Run in production with:
    gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 1 -b 127.0.0.1:8000

Keep ``-w 1``: each worker loads its own ~1.5 GB copy of the model. Scale with
``inference_workers`` and horizontal replicas, not with gunicorn workers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import metrics
from app.config import Settings, get_settings
from app.errors import register_exception_handlers
from app.logging_config import configure_logging
from app.middleware import (
    MaxBodySizeMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.model import BackgroundRemover, set_remover
from app.routes import health, remove
from app.runner import InferenceRunner, get_runner, set_runner
from app.storage import ObjectStorage, set_storage
from app.version import __version__

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, tear the pool down on shutdown."""
    settings: Settings = app.state.settings

    set_storage(ObjectStorage(settings))
    set_runner(InferenceRunner(settings))

    remover = BackgroundRemover(settings)
    set_remover(remover)

    try:
        info = remover.load()
        metrics.set_gauge("rmbg_model_loaded", 1)
        log.info(
            "service ready",
            extra={
                "version": __version__,
                "device": info.device,
                "model_id": info.model_id,
                "inference_workers": settings.inference_workers,
                "auth": settings.auth_enabled,
            },
        )
        if settings.warmup_on_startup:
            remover.warmup()
    except Exception:
        metrics.set_gauge("rmbg_model_loaded", 0)
        # Fail loudly: a process that can never serve a request should not
        # keep a port open and pass liveness checks.
        log.exception("model failed to load - aborting startup")
        raise

    try:
        yield
    finally:
        log.info("shutting down")
        # RuntimeError here just means startup failed before the pool existed.
        with suppress(RuntimeError):
            get_runner().shutdown(wait=True)
        remover.unload()
        set_remover(None)
        metrics.set_gauge("rmbg_model_loaded", 0)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    if not settings.auth_enabled:
        log.warning(
            "API key auth is DISABLED - anyone who can reach this port can use "
            "the model. Fine for local/Postman testing; set RMBG_API_KEYS "
            "before exposing it to the internet."
        )
    if settings.docs_enabled and settings.is_production:
        log.warning(
            "/docs and /openapi.json are PUBLIC in production because "
            "RMBG_ENABLE_DOCS=true."
        )

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Production background-removal API built on RMBG-1.4. "
            "POST an image to /v1/remove and get a transparent PNG back."
        ),
        lifespan=lifespan,
        root_path=settings.root_path,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings

    # Middleware runs bottom-up on the request path, so the body-size guard
    # added last is checked first.
    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=settings.is_production)
    app.add_middleware(RequestContextMiddleware, settings=settings)
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_upload_bytes)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", settings.api_key_header, "Authorization"],
            expose_headers=[
                "X-Request-ID",
                "X-Image-Width",
                "X-Image-Height",
                "X-Inference-Ms",
            ],
            max_age=600,
        )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(remove.router)

    if settings.serve_ui and STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")
    else:

        @app.get("/", include_in_schema=False)
        def index_redirect() -> RedirectResponse:
            return RedirectResponse("/v1/info")

    return app


app = create_app()
