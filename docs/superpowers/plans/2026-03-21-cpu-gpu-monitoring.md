# CPU/GPU Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Prometheus-compatible CPU/GPU monitoring and inference profiling to the Qwen3 Embeddings Server.

**Architecture:** Three new modules (`monitoring/prometheus_metrics.py`, `monitoring/system_collector.py`, `monitoring/profiler.py`) integrated into `server.py` via lifespan context manager, exposing `/metrics/prometheus` endpoint.

**Tech Stack:** prometheus_client, psutil (existing), powermetrics (macOS), asyncio background tasks.

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add prometheus_client dependency**

```toml
[project]
name = "qwen3-embeddings-mlx"
version = "0.1.0"
description = "High-performance text embedding server for Apple Silicon"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.116.1",
    "mlx-lm>=0.26.3",
    "numpy>=2.3.2",
    "opentelemetry-api>=1.40.0",
    "opentelemetry-exporter-otlp>=1.40.0",
    "opentelemetry-sdk>=1.40.0",
    "prometheus_client>=0.21.0",
    "psutil>=7.0.0",
    "pydantic>=2.11.7",
    "rich>=13.0.0",
    "uvicorn[standard]>=0.35.0",
]
```

- [ ] **Step 2: Sync dependencies with uv**

```bash
uv sync
```

Expected: `prometheus_client` installed

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add prometheus_client for metrics export"
```

---

## Task 2: Create Monitoring Module Structure

**Files:**
- Create: `monitoring/__init__.py`
- Create: `monitoring/prometheus_metrics.py`
- Create: `monitoring/system_collector.py`
- Create: `monitoring/profiler.py`

- [ ] **Step 1: Create monitoring/__init__.py**

```python
"""Monitoring module for CPU/GPU metrics and inference profiling."""

from .prometheus_metrics import PrometheusMetrics, get_metrics_handler
from .system_collector import SystemMetricsCollector
from .profiler import InferenceProfiler

__all__ = [
    "PrometheusMetrics",
    "get_metrics_handler",
    "SystemMetricsCollector",
    "InferenceProfiler",
]
```

- [ ] **Step 2: Create monitoring/prometheus_metrics.py**

```python
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
            platform.machine() in ("arm64", "aarch64")
            and platform.system() == "Darwin"
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
```

- [ ] **Step 3: Create monitoring/system_collector.py**

```python
"""System metrics collector using psutil and powermetrics."""

import asyncio
import logging
import platform
import re
import subprocess
from typing import Optional, Dict, Any

import psutil

from .prometheus_metrics import PrometheusMetrics

logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    """Background collector for system metrics."""

    def __init__(
        self,
        metrics: PrometheusMetrics,
        interval: float = 2.0,
        gpu_enabled: bool = True,
    ):
        self.metrics = metrics
        self.interval = interval
        self.gpu_enabled = gpu_enabled and metrics.is_apple_silicon()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._retry_count = 0
        self._max_retries = 3

    async def start(self):
        """Start the background collection task."""
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        logger.info("System metrics collector started")

    async def stop(self):
        """Stop the background collection task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("System metrics collector stopped")

    async def _collect_loop(self):
        """Main collection loop with retry logic."""
        while self._running:
            try:
                await self._collect_once()
                self._retry_count = 0
            except Exception as e:
                logger.warning(f"Metrics collection error: {e}")
                self._retry_count += 1
                if self._retry_count >= self._max_retries:
                    logger.error("Max retries reached, disabling GPU metrics")
                    self.gpu_enabled = False
                    self._retry_count = 0
                else:
                    # Exponential backoff
                    await asyncio.sleep(self.interval * (2 ** self._retry_count))
                    continue

            await asyncio.sleep(self.interval)

    async def _collect_once(self):
        """Collect all metrics once."""
        # CPU and memory (always available)
        self._collect_cpu_memory()

        # GPU (Apple Silicon only)
        if self.gpu_enabled:
            await self._collect_gpu_metrics()

    def _collect_cpu_memory(self):
        """Collect CPU and memory metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics.cpu_percent.set(cpu_percent)

            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics.memory_rss.set(memory_info.rss)
        except Exception as e:
            logger.warning(f"CPU/memory collection failed: {e}")

    async def _collect_gpu_metrics(self):
        """Collect GPU metrics using powermetrics."""
        try:
            # Run powermetrics with timeout
            proc = await asyncio.create_subprocess_exec(
                "sudo",
                "powermetrics",
                "--samplers",
                "gpu_power",
                "-i",
                "2000",
                "-n",
                "1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=5.0
                )
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning("powermetrics timed out")
                return

            if proc.returncode != 0:
                error_msg = stderr.decode().strip()
                if "permission" in error_msg.lower():
                    logger.warning(
                        "powermetrics permission denied. "
                        "Run scripts/setup-gpu-metrics.sh"
                    )
                    self.gpu_enabled = False
                else:
                    logger.warning(f"powermetrics failed: {error_msg}")
                return

            # Parse output
            self._parse_powermetrics(stdout.decode())

        except FileNotFoundError:
            logger.warning("powermetrics not found, disabling GPU metrics")
            self.gpu_enabled = False
        except Exception as e:
            logger.warning(f"GPU metrics collection failed: {e}")

    def _parse_powermetrics(self, output: str):
        """Parse powermetrics output and update gauges."""
        # Example output:
        # GPU Power: 2345 mW
        # GPU Active Residency: 78%
        # GPU Frequency: 1000 MHz

        # Active residency
        match = re.search(r"GPU Active Residency:\s*([\d.]+)%", output)
        if match:
            self.metrics.gpu_active.set(float(match.group(1)))

        # Frequency
        match = re.search(r"GPU Frequency:\s*([\d.]+)\s*MHz", output)
        if match:
            self.metrics.gpu_freq.set(float(match.group(1)))

        # Temperature (if available)
        match = re.search(r"GPU Temperature:\s*([\d.]+)", output)
        if match:
            self.metrics.gpu_temp.set(float(match.group(1)))
```

- [ ] **Step 4: Create monitoring/profiler.py**

```python
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
        ).add(token_count)

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
```

- [ ] **Step 5: Commit**

```bash
git add monitoring/
git commit -m "feat: add monitoring module with Prometheus metrics"
```

---

## Task 3: Create Setup Script for GPU Metrics

**Files:**
- Create: `scripts/setup-gpu-metrics.sh`

- [ ] **Step 1: Create setup script**

```bash
#!/bin/bash
# Setup sudoers for powermetrics GPU monitoring

set -e

USERNAME=$(whoami)
SUDOERS_FILE="/etc/sudoers.d/qwen3-powermetrics"

echo "Setting up GPU metrics permissions..."
echo ""
echo "This script will configure sudo to allow powermetrics without password."
echo "The rule is restricted to safe intervals (1000-60000ms) to prevent DoS."
echo ""

# Create sudoers file with restricted interval pattern
echo "$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/powermetrics --samplers gpu_power -i [1-6][0-9][0-9][0-9]" | sudo tee $SUDOERS_FILE
sudo chmod 440 $SUDOERS_FILE

echo ""
echo "✓ GPU metrics configured successfully."
echo ""
echo "You can now run the server and GPU metrics will be collected automatically."
echo "To verify, run: sudo powermetrics --samplers gpu_power -i 2000 -n 1"
```

- [ ] **Step 2: Make script executable**

```bash
chmod +x scripts/setup-gpu-metrics.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/setup-gpu-metrics.sh
git commit -m "feat: add GPU metrics setup script for macOS"
```

---

## Task 4: Integrate Monitoring into Server

**Files:**
- Modify: `server.py`

- [ ] **Step 1: Add monitoring imports at top of server.py**

```python
# Monitoring imports
from monitoring import PrometheusMetrics, SystemMetricsCollector, InferenceProfiler
```

- [ ] **Step 2: Initialize monitoring in global scope**

```python
# Initialize model managers
model_manager = ModelManager(config)
rerank_model_manager = RerankModelManager(config)

# Initialize monitoring
prometheus_metrics = PrometheusMetrics()
system_collector = SystemMetricsCollector(
    metrics=prometheus_metrics,
    interval=float(os.getenv("METRICS_COLLECTION_INTERVAL", "2.0")),
    gpu_enabled=os.getenv("GPU_METRICS_ENABLED", "true").lower() == "true",
)
```

- [ ] **Step 3: Update lifespan to start/stop collector**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info(f"Starting Qwen3 Embedding Server v{app.version}")
    logger.info(f"Configuration: {config}")
    logger.info(f"Available models: {list(AVAILABLE_MODELS.keys())}")

    # Start system metrics collector
    await system_collector.start()

    try:
        # Load default model at startup
        await model_manager.load_model(config.model_name)
    except Exception as e:
        logger.error(f"Failed to initialize server with default model: {e}")
        # Server can still start, models will be loaded on demand

    app.state.start_time = time.time()

    yield

    # Shutdown
    logger.info("Shutting down server...")
    await system_collector.stop()
```

- [ ] **Step 4: Add Prometheus endpoint**

```python
@app.get("/metrics/prometheus", tags=["Monitoring"])
async def prometheus_metrics():
    """
    Get Prometheus-format metrics.

    Scrape this endpoint with Prometheus or compatible tools.
    """
    from fastapi import Response
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    return Response(
        content=generate_latest(prometheus_metrics.REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
```

- [ ] **Step 5: Commit**

```bash
git add server.py
git commit -m "feat: integrate monitoring into server with /metrics/prometheus"
```

---

## Task 6: Add Profiling to Inference Endpoints

**Files:**
- Modify: `server.py` (embedding and rerank endpoints)

- [ ] **Step 1: Update create_embeddings to use profiler**

Find the `create_embeddings` function and wrap with profiler:

```python
async def create_embeddings(request: OpenAIEmbeddingRequest):
    """Create embeddings for input text(s)."""
    try:
        start_time = time.time()

        # Normalize input to list
        texts = [request.input] if isinstance(request.input, str) else request.input

        # Resolve model name (OpenAI -> Qwen)
        model_resolved = resolve_model(request.model)

        # Create profiler
        profiler = InferenceProfiler(
            metrics=prometheus_metrics,
            model=model_resolved,
            endpoint="embeddings",
        )

        with profiler.total():
            # Generate embeddings using existing ModelManager
            embeddings, model_used, embedding_dim = (
                await model_manager.generate_embeddings(
                    texts, model_name=model_resolved, normalize=True
                )
            )

        # Count tokens for usage info
        model_tuple = model_manager.models.get(model_used)
        if model_tuple:
            _, tokenizer = model_tuple
            total_tokens = count_tokens_batch(texts, tokenizer)
            profiler.record_tokens(total_tokens)

        # ... rest of function unchanged
```

- [ ] **Step 2: Update rerank endpoint similarly**

```python
async def rerank_documents(request: RerankRequest):
    """Rerank documents by relevance to a query."""
    try:
        start_time = time.time()

        # Resolve model name
        model_resolved = resolve_rerank_model(request.model)

        # Create profiler
        profiler = InferenceProfiler(
            metrics=prometheus_metrics,
            model=model_resolved,
            endpoint="rerank",
        )

        with profiler.total():
            # Compute relevance scores
            scores = await rerank_model_manager.compute_scores(
                query=request.query,
                documents=request.documents,
                model_name=model_resolved,
                max_tokens=request.max_tokens_per_doc,
            )

        # ... rest of function unchanged
```

- [ ] **Step 3: Commit**

```bash
git add server.py
git commit -m "feat: add profiling to embedding and rerank endpoints"
```

---

## Task 7: Write Tests

**Files:**
- Create: `tests/test_monitoring.py`

- [ ] **Step 1: Create test file**

```python
"""Tests for monitoring module."""

import pytest
from unittest.mock import patch, MagicMock
import asyncio

from monitoring.prometheus_metrics import PrometheusMetrics
from monitoring.profiler import InferenceProfiler


class TestPrometheusMetrics:
    """Test Prometheus metrics registration."""

    def test_metrics_initialization(self):
        """Test that all metrics are registered."""
        metrics = PrometheusMetrics()

        assert metrics.requests_total is not None
        assert metrics.request_duration is not None
        assert metrics.inference_duration is not None
        assert metrics.cpu_percent is not None
        assert metrics.memory_rss is not None

    def test_is_apple_silicon(self):
        """Test platform detection."""
        metrics = PrometheusMetrics()

        with patch("platform.machine", return_value="arm64"):
            with patch("platform.system", return_value="Darwin"):
                assert metrics.is_apple_silicon() is True

        with patch("platform.machine", return_value="x86_64"):
            assert metrics.is_apple_silicon() is False


class TestInferenceProfiler:
    """Test inference profiler."""

    def test_profiler_stage_timing(self):
        """Test that profiler records stage timing."""
        metrics = PrometheusMetrics()
        profiler = InferenceProfiler(metrics, "test-model", "embeddings")

        with profiler.stage("tokenization"):
            time.sleep(0.01)  # 10ms

        # Metric should be recorded (no exception)

    def test_profiler_total_timing(self):
        """Test that profiler records total timing."""
        metrics = PrometheusMetrics()
        profiler = InferenceProfiler(metrics, "test-model", "embeddings")

        with profiler.total():
            time.sleep(0.01)

        # Metric should be recorded

    def test_profiler_token_recording(self):
        """Test token count recording."""
        metrics = PrometheusMetrics()
        profiler = InferenceProfiler(metrics, "test-model", "embeddings")

        profiler.record_tokens(100)

        # Counter should be incremented (no exception)


class TestPrometheusEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint_format(self):
        """Test that endpoint returns valid Prometheus format."""
        import requests

        try:
            response = requests.get("http://localhost:8000/metrics/prometheus")
            assert response.status_code == 200

            # Check Prometheus format
            content = response.text
            assert "qwen3_requests_total" in content
            assert "qwen3_cpu_percent" in content
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running")
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_monitoring.py -v
```

Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_monitoring.py
git commit -m "test: add monitoring module tests"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add GPU Monitoring section to README**

Add after "OpenTelemetry Metrics" section:

```markdown
## 🔧 CPU/GPU Monitoring (Prometheus)

Built-in Prometheus metrics for performance profiling on Apple Silicon.

### Quick Setup (Required for GPU metrics)

GPU metrics require `powermetrics` which needs elevated permissions. Run once:

```bash
./scripts/setup-gpu-metrics.sh
```

Or manually:

```bash
sudo bash -c 'echo "$(whoami) ALL=(ALL) NOPASSWD: /usr/bin/powermetrics --samplers gpu_power -i [1-6][0-9][0-9][0-9]" > /etc/sudoers.d/qwen3-powermetrics'
sudo chmod 440 /etc/sudoers.d/qwen3-powermetrics
```

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `qwen3_requests_total` | Counter | Total requests |
| `qwen3_request_duration_seconds` | Histogram | Request latency |
| `qwen3_inference_duration_seconds` | Histogram | Stage-level timing |
| `qwen3_cpu_percent` | Gauge | CPU utilization % |
| `qwen3_gpu_active_percent` | Gauge | GPU active ratio % |
| `qwen3_memory_rss_bytes` | Gauge | Memory usage |
| `qwen3_cache_*` | Various | Cache metrics |

### Prometheus Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'qwen3-embeddings'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/prometheus'
    scrape_interval: 15s
```

### Grafana Dashboard

Import the dashboard from `grafana/dashboards/qwen3-embeddings.json`.

```

- [ ] **Step 2: Update endpoints table**

Add to "Other Endpoints" table:
| `GET /metrics/prometheus` | Prometheus-format metrics |

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Prometheus monitoring documentation"
```

---

## Task 9: Create Grafana Dashboard

**Files:**
- Create: `grafana/dashboards/qwen3-embeddings.json`

- [ ] **Step 1: Create directory**

```bash
mkdir -p grafana/dashboards
```

- [ ] **Step 2: Create dashboard JSON**

Create a Grafana 10.x compatible dashboard with panels for:
- Inference latency heatmap (by stage)
- GPU utilization over time
- CPU and memory usage
- Request rate and error rate
- Cache hit ratio

- [ ] **Step 3: Commit**

```bash
git add grafana/
git commit -m "feat: add Grafana dashboard for monitoring"
```

---

## Task 10: Final Testing & Verification

**Files:**
- All modified files

- [ ] **Step 1: Start server**

```bash
uv run python server.py
```

- [ ] **Step 2: Verify health endpoint**

```bash
curl http://localhost:8000/health
```

Expected: `{"status": "healthy", ...}`

- [ ] **Step 3: Verify Prometheus endpoint**

```bash
curl http://localhost:8000/metrics/prometheus | head -50
```

Expected: Prometheus format metrics including `qwen3_*`

- [ ] **Step 4: Make a request and verify metrics update**

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "small"}'

curl http://localhost:8000/metrics/prometheus | grep qwen3_requests
```

Expected: Counter incremented

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: final testing and verification"
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-03-21-cpu-gpu-monitoring.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
