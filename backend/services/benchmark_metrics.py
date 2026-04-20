import threading
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class _RouteStats:
    count: int = 0
    total_ms: float = 0.0


class TranslationBenchmarkStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._routes: dict[str, _RouteStats] = defaultdict(_RouteStats)
        self._error_count = 0
        self._total_count = 0

    def record_success(self, route: str, latency_ms: float) -> None:
        with self._lock:
            self._total_count += 1
            stats = self._routes[route]
            stats.count += 1
            stats.total_ms += latency_ms

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            routes: dict[str, dict[str, float | int]] = {}
            for route, stats in self._routes.items():
                avg_ms = stats.total_ms / stats.count if stats.count else 0.0
                routes[route] = {
                    "count": stats.count,
                    "avg_ms": round(avg_ms, 2),
                    "total_ms": round(stats.total_ms, 2),
                }
            return {
                "started_at": self._started_at,
                "uptime_seconds": round(time.time() - self._started_at, 2),
                "total_success_count": self._total_count,
                "error_count": self._error_count,
                "routes": routes,
            }

    def reset(self) -> None:
        with self._lock:
            self._started_at = time.time()
            self._routes = defaultdict(_RouteStats)
            self._error_count = 0
            self._total_count = 0


translation_benchmark_store = TranslationBenchmarkStore()
