"""Cross-cutting HTTP middleware: request ids, access logs, size guard,
security headers."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app import metrics
from app.config import Settings
from app.logging_config import request_id_var

log = logging.getLogger("app.access")


def _metric_path(request: Request) -> str:
    """Label value for metrics.

    The raw URL path must never become a label: an unauthenticated scanner
    hitting /a, /b, /c... would create a new counter series per request and
    grow the registry without bound. Only the matched route template, or the
    constant "unmatched", is used.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if template:
        return str(template)
    return "unmatched"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, logs the access line, records metrics."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id")
        # Never echo an unbounded client-supplied value back into logs.
        request_id = (
            incoming[:64]
            if incoming and incoming.isascii() and incoming.isprintable()
            else uuid.uuid4().hex
        )
        # The contextvar must stay set until after the access line is logged,
        # otherwise that line - the one you actually grep for - is the only log
        # in the request without a request_id.
        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            try:
                response = await call_next(request)
            except Exception:
                duration = time.perf_counter() - started
                metrics.inc(
                    "rmbg_http_requests_total",
                    {"path": _metric_path(request), "status": "500"},
                )
                log.exception(
                    "request failed",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": round(duration * 1000, 2),
                    },
                )
                raise

            duration = time.perf_counter() - started
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = f"{duration * 1000:.1f}"

            remaining = getattr(request.state, "rate_limit_remaining", None)
            if remaining is not None:
                response.headers["X-RateLimit-Limit"] = str(
                    self._settings.rate_limit_requests
                )
                response.headers["X-RateLimit-Remaining"] = str(remaining)

            metrics.inc(
                "rmbg_http_requests_total",
                {"path": _metric_path(request), "status": str(response.status_code)},
            )
            metrics.observe("rmbg_http_request_seconds", duration)

            # Probes and scrapes would otherwise dominate the log volume.
            if self._settings.log_access and request.url.path not in (
                "/healthz",
                "/readyz",
                "/metrics",
            ):
                log.info(
                    "request",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": round(duration * 1000, 2),
                    },
                )
            return response
        finally:
            request_id_var.reset(token)


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Rejects an oversized request before the body is buffered.

    nginx enforces this too (`client_max_body_size`); this layer keeps the
    guarantee when the app is exposed directly or run behind another proxy.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self._max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "payload_too_large",
                        "message": (
                            f"Request body exceeds the "
                            f"{self._max_bytes / 1e6:.1f} MB limit."
                        ),
                    }
                },
            )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, enable_hsts: bool = False) -> None:
        super().__init__(app)
        self._enable_hsts = enable_hsts

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        if self._enable_hsts:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response
