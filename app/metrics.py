"""Dependency-free Prometheus text metrics.

Deliberately not using prometheus_client: this keeps the image small and the
counter set is tiny. Values are per-process, so with more than one uvicorn
worker scrape each worker separately or put a real exporter in front.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

Labels = tuple[tuple[str, str], ...]
MetricKey = tuple[str, Labels]

_lock = threading.Lock()
# float, not int: `inc` accepts a float step and histogram sums are seconds.
_counters: dict[MetricKey, float] = defaultdict(float)
_gauges: dict[MetricKey, float] = {}
_hist_buckets: dict[str, dict[float, int]] = defaultdict(
    lambda: dict.fromkeys(_LATENCY_BUCKETS, 0)
)
_hist_sum: dict[str, float] = defaultdict(float)
_hist_count: dict[str, int] = defaultdict(int)

_started_at = time.time()


def _key(name: str, labels: dict[str, str] | None) -> MetricKey:
    return name, tuple(sorted((labels or {}).items()))


def inc(name: str, labels: dict[str, str] | None = None, value: float = 1) -> None:
    with _lock:
        _counters[_key(name, labels)] += value


def set_gauge(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    with _lock:
        _gauges[_key(name, labels)] = value


def observe(name: str, seconds: float) -> None:
    with _lock:
        buckets = _hist_buckets[name]
        for bound in _LATENCY_BUCKETS:
            if seconds <= bound:
                buckets[bound] += 1
        _hist_sum[name] += seconds
        _hist_count[name] += 1


def _format_labels(labels: Labels) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{str(v).replace(chr(34), "")}"' for k, v in labels)
    return "{" + inner + "}"


def render() -> str:
    """Render the registry in Prometheus text exposition format."""
    lines: list[str] = []
    with _lock:
        lines.append("# TYPE rmbg_uptime_seconds gauge")
        lines.append(f"rmbg_uptime_seconds {time.time() - _started_at:.3f}")

        seen: set[str] = set()
        for (name, labels), value in sorted(_counters.items()):
            if name not in seen:
                lines.append(f"# TYPE {name} counter")
                seen.add(name)
            lines.append(f"{name}{_format_labels(labels)} {value:g}")

        seen.clear()
        for (name, labels), value in sorted(_gauges.items()):
            if name not in seen:
                lines.append(f"# TYPE {name} gauge")
                seen.add(name)
            lines.append(f"{name}{_format_labels(labels)} {value:g}")

        for name, buckets in sorted(_hist_buckets.items()):
            lines.append(f"# TYPE {name} histogram")
            cumulative = 0
            for bound in _LATENCY_BUCKETS:
                cumulative = buckets[bound]
                lines.append(f'{name}_bucket{{le="{bound}"}} {cumulative}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {_hist_count[name]}')
            lines.append(f"{name}_sum {_hist_sum[name]:.6f}")
            lines.append(f"{name}_count {_hist_count[name]}")

    return "\n".join(lines) + "\n"


def reset() -> None:
    """Test helper."""
    with _lock:
        _counters.clear()
        _gauges.clear()
        _hist_buckets.clear()
        _hist_sum.clear()
        _hist_count.clear()
