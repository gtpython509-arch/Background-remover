"""Bounded async bridge to the synchronous model.

The event loop must never block on a forward pass, and the backlog of waiting
requests must never be unbounded - otherwise a traffic spike turns into an
out-of-memory kill instead of a few honest 503s.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from app import metrics
from app.config import Settings
from app.errors import ApiError, ServiceOverloaded

log = logging.getLogger(__name__)

T = TypeVar("T")


class InferenceRunner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.inference_workers),
            thread_name_prefix="rmbg-infer",
        )
        # Capacity = in-flight slots + queue depth.
        capacity = max(1, settings.inference_workers) + max(0, settings.max_queued_requests)
        self._slots = asyncio.Semaphore(capacity)
        self._capacity = capacity
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def capacity(self) -> int:
        return self._capacity

    async def run(self, fn: Callable[..., T], *args: object) -> T:
        """Execute ``fn`` on the inference pool, or fail fast when saturated."""
        if self._slots.locked():
            metrics.inc("rmbg_requests_rejected_total", {"reason": "overloaded"})
            raise ServiceOverloaded(
                f"All {self._capacity} inference slots are busy. Retry shortly."
            )

        await self._slots.acquire()
        self._in_flight += 1
        metrics.set_gauge("rmbg_inference_in_flight", self._in_flight)
        queued_at = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self._executor, fn, *args)
            result = await asyncio.wait_for(
                future, timeout=self._settings.inference_timeout_seconds
            )
            metrics.observe("rmbg_inference_seconds", time.perf_counter() - queued_at)
            return result
        except TimeoutError:
            metrics.inc("rmbg_requests_rejected_total", {"reason": "timeout"})
            log.error(
                "inference timed out",
                extra={"timeout": self._settings.inference_timeout_seconds},
            )
            raise ApiError(
                "Inference timed out. Try a smaller image.",
                status_code=504,
                code="inference_timeout",
            ) from None
        finally:
            self._in_flight -= 1
            metrics.set_gauge("rmbg_inference_in_flight", self._in_flight)
            self._slots.release()

    def shutdown(self, wait: bool = True) -> None:
        log.info("shutting down inference pool", extra={"wait": wait})
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


_runner: InferenceRunner | None = None


def get_runner() -> InferenceRunner:
    if _runner is None:
        raise RuntimeError("inference runner is not initialised")
    return _runner


def set_runner(runner: InferenceRunner | None) -> None:
    global _runner
    _runner = runner
