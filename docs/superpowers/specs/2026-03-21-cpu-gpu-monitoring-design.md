# CPU/GPU Monitoring Design

**Date:** 2026-03-21
**Status:** Draft
**Author:** Claude Opus 4.6

## Overview

Add Prometheus-compatible CPU/GPU monitoring to the Qwen3 Embeddings Server for performance profiling on Apple Silicon Macs.

## Goals

- Profile inference performance at stage level (tokenization, forward pass, pooling, normalization)
- Monitor system resources: CPU utilization, GPU active ratio, memory usage
- Integrate with Prometheus/Grafana ecosystem for visualization
- Provide clear setup instructions for macOS permissions

## Non-Goals

- Real-time alerting (handled by external tools)
- Multi-node monitoring (single Mac deployment only)
- Historical data storage (Prometheus handles this)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
├─────────────────────────────────────────────────────────────┤
│  /v1/embeddings    /v1/rerank    /health    /metrics        │
│        │               │                          │          │
│        ▼               ▼                          ▼          │
│  ┌─────────────────────────┐    ┌──────────────────────┐    │
│  │   Inference Profiler    │    │  Prometheus Metrics  │    │
│  │  - tokenization_ms      │    │  - Counter           │    │
│  │  - forward_pass_ms      │    │  - Histogram         │    │
│  │  - pooling_ms           │    │  - Gauge             │    │
│  │  - total_ms             │    │                      │    │
│  └─────────────────────────┘    └──────────▲───────────┘    │
│                                            │                 │
├────────────────────────────────────────────┼─────────────────┤
│  Background Task (asyncio)                 │                 │
│  ┌─────────────────────────────┐           │                 │
│  │  System Metrics Collector   │───────────┘                 │
│  │  - powermetrics (GPU %)     │                             │
│  │  - psutil (CPU %, memory)   │                             │
│  │  - interval: 2s             │                             │
│  └─────────────────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Prometheus/Grafana
```

## Components

### 1. Inference Profiler

Measures timing of each inference stage using context managers.

**Stages:**
- `tokenization` - Text to token IDs
- `forward_pass` - Transformer forward pass
- `pooling` - Mean pooling across sequence
- `normalization` - L2 normalization
- `total` - End-to-end request time

### 2. System Metrics Collector

Background asyncio task collecting system metrics every 2 seconds.

**Data Sources:**
- `psutil` - CPU percent, RSS memory
- `powermetrics` - GPU active ratio, frequency, temperature

**Lifecycle:**
- Started in FastAPI `lifespan()` context manager on startup
- Properly cancelled on shutdown via `asyncio.Task.cancel()`
- Auto-restarts on failure with exponential backoff (max 3 retries, then disable)

### 3. Prometheus Endpoint

New endpoint `/metrics/prometheus` exposing metrics in Prometheus format.

**Relationship with OpenTelemetry:**
- Existing OTLP metrics remain unchanged for backward compatibility
- Prometheus endpoint uses `prometheus_client` with separate registry
- Both systems can coexist; users choose based on their infrastructure

## Metrics Schema

### Inference Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `qwen3_inference_duration_seconds` | Histogram | model, endpoint, stage | Duration of each inference stage |
| `qwen3_inference_tokens_total` | Counter | model, endpoint | Total tokens processed |

**Histogram Buckets (seconds):** `[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]`

### System Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `qwen3_cpu_percent` | Gauge | - | CPU utilization % |
| `qwen3_memory_rss_bytes` | Gauge | - | RSS memory in bytes |
| `qwen3_gpu_active_percent` | Gauge | - | GPU active ratio % |
| `qwen3_gpu_freq_mhz` | Gauge | - | GPU frequency |
| `qwen3_gpu_temp_celsius` | Gauge | - | GPU temperature |

### Request Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `qwen3_requests_total` | Counter | model, endpoint, status | Total requests |
| `qwen3_request_duration_seconds` | Histogram | model, endpoint | Request latency |

### Cache Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `qwen3_cache_hits_total` | Counter | model | Cache hits |
| `qwen3_cache_misses_total` | Counter | model | Cache misses |
| `qwen3_cache_size` | Gauge | - | Current cache entries |
| `qwen3_cache_evictions_total` | Counter | model | Cache evictions |

### Model Management Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `qwen3_model_load_duration_seconds` | Histogram | model | Time to load model |
| `qwen3_model_evictions_total` | Counter | model | Model evictions from memory |
| `qwen3_loaded_models` | Gauge | - | Currently loaded models count |

### Model Management Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `qwen3_model_load_duration_seconds` | Histogram | model | Time to load a model |
| `qwen3_model_evictions_total` | Counter | model | Number of model evictions |
| `qwen3_loaded_models` | Gauge | - | Number of models currently loaded |

### Cache Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `qwen3_cache_hits_total` | Counter | model | Embedding cache hits |
| `qwen3_cache_misses_total` | Counter | model | Embedding cache misses |
| `qwen3_cache_size` | Gauge | - | Current cache size |
| `qwen3_cache_evictions_total` | Counter | - | Cache entry evictions |

### Histogram Buckets

```python
INFERENCE_BUCKETS = [0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
# Covers: 1ms (0.6B fast) to 1s (8B slow + batching)
```

## File Structure

```
qwen3-embeddings-mlx/
├── server.py                 # Add profiler integration
├── monitoring/
│   ├── __init__.py
│   ├── prometheus_metrics.py # Prometheus metrics definitions
│   ├── system_collector.py   # Background CPU/GPU collector
│   └── profiler.py           # Inference stage profiler
├── scripts/
│   └── setup-gpu-metrics.sh  # Sudoers setup script
└── grafana/
    └── dashboard.json        # Pre-built Grafana dashboard
```

## Sudoers Setup

GPU metrics require `powermetrics` which needs elevated permissions.

### Security Note

The sudoers rule restricts the interval to a safe range (1000-60000ms) to prevent DoS attacks through excessive system calls.

### Setup Script (`scripts/setup-gpu-metrics.sh`)

```bash
#!/bin/bash
USERNAME=$(whoami)
SUDOERS_FILE="/etc/sudoers.d/qwen3-powermetrics"

echo "Setting up GPU metrics permissions..."
# Restrict to interval 1000-60000ms to prevent DoS
echo "$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/powermetrics --samplers gpu_power -i [1-6][0-9][0-9][0-9]" | sudo tee $SUDOERS_FILE
sudo chmod 440 $SUDOERS_FILE

echo "✓ GPU metrics configured."
```

### Manual Setup

```bash
sudo bash -c 'echo "$(whoami) ALL=(ALL) NOPASSWD: /usr/bin/powermetrics --samplers gpu_power -i [1-6][0-9][0-9][0-9]" > /etc/sudoers.d/qwen3-powermetrics'
sudo chmod 440 /etc/sudoers.d/qwen3-powermetrics
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ENABLE_PROMETHEUS_METRICS` | `true` | Enable Prometheus endpoint |
| `METRICS_COLLECTION_INTERVAL` | `2.0` | System metrics interval (seconds) |
| `GPU_METRICS_ENABLED` | `true` | Enable GPU metrics collection |

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `powermetrics` not available | GPU metrics return `NaN`, CPU/memory still work |
| sudoers not configured | Log warning once, disable GPU collection |
| `prometheus_client` not installed | `/metrics/prometheus` returns 503, system still works |
| `powermetrics` hangs (>5s) | Timeout and skip collection, log warning |
| Non-Apple Silicon detected | Disable GPU metrics automatically, CPU/memory still work |
| Collector task crashes | Auto-restart with exponential backoff (max 3 retries) |

### Platform Detection

```python
def is_apple_silicon() -> bool:
    return (
        platform.machine() in ('arm64', 'aarch64') and
        platform.system() == 'Darwin'
    )
```

## Dependencies

- `prometheus_client` - Prometheus metrics library (add to pyproject.toml)

## Testing Strategy

### Unit Tests

- `test_profiler.py` - Test InferenceProfiler context manager, timing accuracy
- `test_system_collector.py` - Test powermetrics parsing, psutil integration
- `test_prometheus_metrics.py` - Test metric registration, label validation

### Integration Tests

- Test `/metrics/prometheus` endpoint returns valid Prometheus format
- Test metrics are updated after inference requests
- Test graceful degradation when powermetrics unavailable

### Mocking Strategy

```python
# Mock powermetrics for CI
@pytest.fixture
def mock_powermetrics(monkeypatch):
    async def fake_collect(*args, **kwargs):
        return {"gpu_active_percent": 50.0, "gpu_freq_mhz": 1000}
    monkeypatch.setattr(SystemMetricsCollector, "_collect_gpu", fake_collect)
```

## Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'qwen3-embeddings'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/prometheus'
    scrape_interval: 15s
```

## Grafana Dashboard

Dashboard file: `grafana/dashboards/qwen3-embeddings.json` (Grafana 10.x compatible)

Panels:
- Inference latency by stage (histogram heatmap)
- GPU utilization over time
- CPU and memory usage
- Request rate and error rate
- Cache hit ratio

## Open Questions

- None at this time.
