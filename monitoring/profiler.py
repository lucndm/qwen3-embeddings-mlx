"""Inference profiler for stage-level timing."""

import time
from contextlib import contextmanager
from typing import Optional

from .prometheus_metrics import PrometheusMetrics


class InferenceProfiler:
    """Context manager for profiling inference stages."""

    def __init__(
        self,
        metrics: PrometheusMetrics,
        model: str,
        endpoint: str = "embeddings",
    ):
        self.metrics = metrics
        self.model = model
        self.endpoint = endpoint
        self._start_time: Optional[float] = None

    @contextmanager
    def stage(self, stage_name: str):
        """Profile a specific inference stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.metrics.inference_duration.labels(
                model=self.model,
                endpoint=self.endpoint,
                stage=stage_name,
            ).observe(duration)

    @contextmanager
    def total(self):
        """Profile total request time."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.metrics.request_duration.labels(
                model=self.model,
                endpoint=self.endpoint,
            ).observe(duration)
            self.metrics.requests_total.labels(
                model=self.model,
                endpoint=self.endpoint,
                status="success",
            ).inc()

    def record_tokens(self, token_count: int):
        """Record token count."""
        self.metrics.inference_tokens.labels(
            model=self.model,
            endpoint=self.endpoint,
        ).inc(amount=token_count)

    def record_cache_hit(self):
        """Record cache hit."""
        self.metrics.cache_hits.labels(model=self.model).inc()

    def record_cache_miss(self):
        """Record cache miss."""
        self.metrics.cache_misses.labels(model=self.model).inc()

    def update_cache_size(self, size: int):
        """Update cache size gauge."""
        self.metrics.cache_size.set(size)

    def record_model_load(self, duration: float):
        """Record model load duration."""
        self.metrics.model_load_duration.labels(model=self.model).observe(duration)

    def record_model_eviction(self):
        """Record model eviction."""
        self.metrics.model_evictions.labels(model=self.model).inc()
        self.metrics.loaded_models.dec()

    def update_loaded_models(self, count: int):
        """Update loaded models gauge."""
        self.metrics.loaded_models.set(count)
