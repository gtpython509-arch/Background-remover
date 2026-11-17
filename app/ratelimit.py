"""In-process sliding-window rate limiter.

Adequate for a single-instance deployment, which is what a Hostinger VPS runs.
If you scale to several containers or uvicorn workers, move this to Redis or
enforce it in nginx (`limit_req_zone`) instead - each process keeps its own
counters, so N processes allow N times the configured rate.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Request, status

from app.config import get_settings
from app.errors import ApiError
from app.security import client_ip


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def check(self, identity: str) -> tuple[bool, int, float]:
        """Return ``(allowed, remaining, retry_after_seconds)``."""
        now = time.monotonic()
        with self._lock:
            self._sweep(now)
            hits = self._hits[identity]
            cutoff = now - self.window
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = max(0.0, hits[0] + self.window - now)
                return False, 0, retry_after

            hits.append(now)
            return True, self.limit - len(hits), 0.0

    def _sweep(self, now: float) -> None:
        """Drop idle identities so the dict cannot grow without bound."""
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        cutoff = now - self.window
        for identity in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[identity]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._last_sweep = 0.0


_limiter: SlidingWindowLimiter | None = None


def get_limiter() -> SlidingWindowLimiter:
    global _limiter
    if _limiter is None:
        settings = get_settings()
        _limiter = SlidingWindowLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds
        )
    return _limiter


def enforce_rate_limit(request: Request, identity: str) -> None:
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return

    # Anonymous callers are bucketed by IP; authenticated ones by key, so one
    # tenant behind a shared NAT cannot exhaust another tenant's allowance.
    bucket = identity if identity.startswith("key:") else f"ip:{client_ip(request, settings)}"
    allowed, remaining, retry_after = get_limiter().check(bucket)

    request.state.rate_limit_remaining = remaining
    if not allowed:
        raise ApiError(
            "Rate limit exceeded.",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            code="rate_limited",
            headers={
                "Retry-After": str(max(1, int(retry_after) + 1)),
                "X-RateLimit-Limit": str(settings.rate_limit_requests),
                "X-RateLimit-Remaining": "0",
            },
        )
