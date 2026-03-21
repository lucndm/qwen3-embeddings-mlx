"""Prometheus metrics definitions for Qwen3 Embedding Server."""

import platform
from typing import Optional
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Separate registry to avoid conflicts with OpenTelemetry
REGISTRY = CollectorRegistry()

# Histogram buckets for latency (1ms to 1s)
INFERENCE_BUCKETS = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
]

# Global singleton instance
_global_metrics_instance: Optional["PrometheusMetrics"] = None


class PrometheusMetrics:
    """Prometheus metrics registry with singleton pattern for production."""

    def __new__(cls, registry: Optional[CollectorRegistry] = None):
        """Create or return singleton instance for production.

        When registry is None, returns the global singleton (for production use).
        When registry is provided, creates a new instance (for isolated testing).

        Args:
            registry: If None, returns singleton. If provided, creates new instance.
        """
        global _global_metrics_instance

        # For testing: if a specific registry is provided, create a new instance
        if registry is not None:
            instance = super(PrometheusMetrics, cls).__new__(cls)
            instance._initialized = False
            instance._is_singleton = False
            return instance

        # For production: return singleton instance
        if _global_metrics_instance is None:
            _global_metrics_instance = super(PrometheusMetrics, cls).__new__(cls)
            _global_metrics_instance._initialized = False
            _global_metrics_instance._is_singleton = True
        return _global_metrics_instance

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics.

        Args:
            registry: CollectorRegistry to use. If None, uses global REGISTRY (singleton).
                      Use a fresh registry for isolated testing.
        """
        # Skip initialization if already initialized (singleton case)
        if hasattr(self, "_initialized") and self._initialized:
            return

        if registry is None:
            registry = REGISTRY

        # Request metrics
        self.requests_total = Counter(
            "qwen3_requests_total",
            "Total number of requests",
            ["model", "endpoint", "status"],
            registry=registry,
        )

        self.request_duration = Histogram(
            "qwen3_request_duration_seconds",
            "Request duration",
            ["model", "endpoint"],
            buckets=INFERENCE_BUCKETS,
            registry=registry,
        )

        # Inference stage metrics
        self.inference_duration = Histogram(
            "qwen3_inference_duration_seconds",
            "Duration of inference stages",
            ["model", "endpoint", "stage"],
            buckets=INFERENCE_BUCKETS,
            registry=registry,
        )

        self.inference_tokens = Counter(
            "qwen3_inference_tokens_total",
            "Total tokens processed",
            ["model", "endpoint"],
            registry=registry,
        )

        # System metrics (updated by SystemMetricsCollector)
        self.cpu_percent = Gauge(
            "qwen3_cpu_percent",
            "CPU utilization %",
            registry=registry,
        )

        self.memory_rss = Gauge(
            "qwen3_memory_rss_bytes",
            "RSS memory in bytes",
            registry=registry,
        )

        self.gpu_active = Gauge(
            "qwen3_gpu_active_percent",
            "GPU active ratio %",
            registry=registry,
        )

        self.gpu_freq = Gauge(
            "qwen3_gpu_freq_mhz",
            "GPU frequency MHz",
            registry=registry,
        )

        self.gpu_temp = Gauge(
            "qwen3_gpu_temp_celsius",
            "GPU temperature",
            registry=registry,
        )

        # Cache metrics
        self.cache_hits = Counter(
            "qwen3_cache_hits_total",
            "Cache hits",
            ["model"],
            registry=registry,
        )

        self.cache_misses = Counter(
            "qwen3_cache_misses_total",
            "Cache misses",
            ["model"],
            registry=registry,
        )

        self.cache_size = Gauge(
            "qwen3_cache_size",
            "Current cache size",
            registry=registry,
        )

        self.cache_evictions = Counter(
            "qwen3_cache_evictions_total",
            "Cache evictions",
            registry=registry,
        )

        # Model management metrics
        self.model_load_duration = Histogram(
            "qwen3_model_load_duration_seconds",
            "Model load duration",
            ["model"],
            buckets=INFERENCE_BUCKETS,
            registry=registry,
        )

        self.model_evictions = Counter(
            "qwen3_model_evictions_total",
            "Model evictions",
            ["model"],
            registry=registry,
        )

        self.loaded_models = Gauge(
            "qwen3_loaded_models",
            "Number of loaded models",
            registry=registry,
        )

        # Mark as initialized
        self._initialized = True

    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        return (
            platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin"
        )


def get_metrics_handler():
    """Return Prometheus metrics handler for FastAPI."""

    from fastapi import Response

    def metrics_endpoint():
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    return metrics_endpoint
