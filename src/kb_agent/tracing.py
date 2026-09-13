"""轻量链路追踪：记录每个阶段的耗时，用于定位延迟瓶颈和线上排障。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from .schemas import TraceSpan


class Tracer:
    def __init__(self) -> None:
        self.spans: list[TraceSpan] = []

    @contextmanager
    def span(self, name: str, **meta) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - started) * 1000, **meta)

    def record(self, name: str, duration_ms: float, **meta) -> None:
        self.spans.append(TraceSpan(name=name, duration_ms=round(duration_ms, 3), meta=dict(meta)))

    def as_dicts(self) -> list[dict]:
        return [{"name": span.name, "duration_ms": span.duration_ms, **span.meta} for span in self.spans]
