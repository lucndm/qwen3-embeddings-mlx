"""Prometheus metrics definitions for Qwen3 Embedding Server."""

import platform
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


class PrometheusMetrics:
    """Prometheus metrics registry."""

    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            "qwen3_requests_total",
            "Total number of requests",
            ["model", "endpoint", "status"],
            registry=REGISTRY,
        )

        self.request_duration = Histogram(
            "qwen3_request_duration_seconds",
            "Request duration",
            ["model", "endpoint"],
            buckets=INFERENCE_BUCKETS,
            registry=REGISTRY,
        )

        # Inference stage metrics
        self.inference_duration = Histogram(
            "qwen3_inference_duration_seconds",
            "Duration of inference stages",
            ["model", "endpoint", "stage"],
            buckets=INFERENCE_BUCKETS,
            registry=REGISTRY,
        )

        self.inference_tokens = Counter(
            "qwen3_inference_tokens_total",
            "Total tokens processed",
            ["model", "endpoint"],
            registry=REGISTRY,
        )

        # System metrics (updated by SystemMetricsCollector)
        self.cpu_percent = Gauge(
            "qwen3_cpu_percent",
            "CPU utilization %",
            registry=REGISTRY,
        )

        self.memory_rss = Gauge(
            "qwen3_memory_rss_bytes",
            "RSS memory in bytes",
            registry=REGISTRY,
        )

        self.gpu_active = Gauge(
            "qwen3_gpu_active_percent",
            "GPU active ratio %",
            registry=REGISTRY,
        )

        self.gpu_freq = Gauge(
            "qwen3_gpu_freq_mhz",
            "GPU frequency MHz",
            registry=REGISTRY,
        )

        self.gpu_temp = Gauge(
            "qwen3_gpu_temp_celsius",
            "GPU temperature",
            registry=REGISTRY,
        )

        # Cache metrics
        self.cache_hits = Counter(
            "qwen3_cache_hits_total",
            "Cache hits",
            ["model"],
            registry=REGISTRY,
        )

        self.cache_misses = Counter(
            "qwen3_cache_misses_total",
            "Cache misses",
            ["model"],
            registry=REGISTRY,
        )

        self.cache_size = Gauge(
            "qwen3_cache_size",
            "Current cache size",
            registry=REGISTRY,
        )

        self.cache_evictions = Counter(
            "qwen3_cache_evictions_total",
            "Cache evictions",
            registry=REGISTRY,
        )

        # Model management metrics
        self.model_load_duration = Histogram(
            "qwen3_model_load_duration_seconds",
            "Model load duration",
            ["model"],
            buckets=INFERENCE_BUCKETS,
            registry=REGISTRY,
        )

        self.model_evictions = Counter(
            "qwen3_model_evictions_total",
            "Model evictions",
            ["model"],
            registry=REGISTRY,
        )

        self.loaded_models = Gauge(
            "qwen3_loaded_models",
            "Number of loaded models",
            registry=REGISTRY,
        )

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
