"""API-key authentication and client identification."""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import secrets

from fastapi import Request, status

from app.config import Settings, get_settings
from app.errors import ApiError

log = logging.getLogger(__name__)


def client_ip(request: Request, settings: Settings | None = None) -> str:
    """Resolve the real client IP.

    ``X-Forwarded-For`` is attacker-controlled on the left, proxy-controlled on
    the right. With N trusted proxies in front, the client address is the Nth
    entry counted from the right, so we only ever read that position.
    """
    settings = settings or get_settings()
    hops = settings.trusted_proxy_count
    if hops > 0:
        forwarded = request.headers.get("x-forwarded-for", "")
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if len(parts) >= hops:
            candidate = parts[-hops]
            try:
                return str(ipaddress.ip_address(candidate))
            except ValueError:
                log.warning("malformed x-forwarded-for entry", extra={"value": candidate})
    return request.client.host if request.client else "unknown"


def _fingerprint(api_key: str) -> str:
    """Short, non-reversible key id, safe to put in logs and metrics."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


def require_api_key(request: Request) -> str:
    """FastAPI dependency. Returns a stable identity string for the caller."""
    settings = get_settings()

    if not settings.auth_enabled:
        return f"ip:{client_ip(request, settings)}"

    header = request.headers.get(settings.api_key_header.lower())
    if not header:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            header = auth[7:].strip()

    if not header:
        raise ApiError(
            f"Missing credentials. Send the {settings.api_key_header} header.",
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="unauthorized",
        )

    # compare_digest against every configured key: constant time, no early exit.
    matched = False
    for candidate in settings.api_keys:
        if secrets.compare_digest(header, candidate):
            matched = True
            break

    if not matched:
        log.warning("rejected api key", extra={"ip": client_ip(request, settings)})
        raise ApiError(
            "Invalid API key.",
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="unauthorized",
        )

    request.state.api_key_id = _fingerprint(header)
    return f"key:{_fingerprint(header)}"
