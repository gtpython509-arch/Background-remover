"""Liveness, readiness, info and metrics endpoints.

Kept unauthenticated so load balancers and uptime monitors can reach them.
They expose no image data and no secrets.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Response

from app import metrics
from app.config import get_settings
from app.model import get_remover
from app.runner import get_runner
from app.schemas import HealthResponse, InfoResponse, ReadyResponse
from app.storage import get_storage
from app.version import __version__

router = APIRouter(tags=["ops"])

_BOOTED_AT = time.time()


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
def healthz() -> HealthResponse:
    """Always 200 while the process is alive. Does not touch the model."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        uptime_seconds=round(time.time() - _BOOTED_AT, 3),
    )


@router.get("/readyz", response_model=ReadyResponse, summary="Readiness probe")
def readyz(response: Response) -> ReadyResponse:
    """503 until the model is loaded, so nginx/compose can gate traffic."""
    remover = get_remover()
    storage_ok = get_storage().health()

    try:
        runner = get_runner()
        in_flight, capacity = runner.in_flight, runner.capacity
    except RuntimeError:
        in_flight, capacity = 0, 0

    ready = remover.is_ready and storage_ok and capacity > 0
    if not ready:
        response.status_code = 503

    return ReadyResponse(
        status="ready" if ready else "not_ready",
        model_loaded=remover.is_ready,
        storage_ok=storage_ok,
        in_flight=in_flight,
        capacity=capacity,
    )


@router.get("/v1/info", response_model=InfoResponse, summary="Service capabilities")
def info() -> InfoResponse:
    settings = get_settings()
    model_info = get_remover().info
    return InfoResponse(
        app=settings.app_name,
        version=__version__,
        model_id=settings.model_id,
        model_revision=settings.model_revision,
        device=model_info.device if model_info else "unloaded",
        input_size=settings.model_input_size,
        half_precision=bool(model_info and model_info.half_precision),
        model_load_seconds=round(model_info.load_seconds, 3) if model_info else None,
        max_upload_bytes=settings.max_upload_bytes,
        max_pixels=settings.max_pixels,
        max_input_dimension=settings.max_input_dimension,
        supported_output_formats=["png", "webp", "jpeg", "mask"],
        auth_required=settings.auth_enabled,
        storage_enabled=settings.storage_enabled,
        remote_fetch_enabled=settings.allow_remote_fetch,
    )


@router.get("/metrics", include_in_schema=False, summary="Prometheus metrics")
def prometheus_metrics() -> Response:
    if not get_settings().metrics_enabled:
        return Response(status_code=404)
    return Response(
        content=metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
