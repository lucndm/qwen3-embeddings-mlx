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

### 3. Prometheus Endpoint

New endpoint `/metrics/prometheus` exposing metrics in Prometheus format.

## Metrics Schema

### Inference Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `qwen3_inference_duration_seconds` | Histogram | model, endpoint, stage | Duration of each inference stage |
| `qwen3_inference_tokens_total` | Counter | model, endpoint | Total tokens processed |

### System Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `qwen3_cpu_percent` | Gauge | - | CPU utilization % |
| `qwen3_memory_rss_bytes` | Gauge | - | RSS memory in bytes |
| `qwen3_gpu_active_percent` | Gauge | - | GPU active ratio % |
| `qwen3_gpu_freq_mhz` | Gauge | - | GPU frequency |
| `qwen3_gpu_temp_celsius` | Gauge | - | GPU temperature |

### Request Metrics

| Metric Name | Type | Labels |
|-------------|------|--------|
| `qwen3_requests_total` | Counter | model, endpoint, status |
| `qwen3_request_duration_seconds` | Histogram | model, endpoint |

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

### Setup Script (`scripts/setup-gpu-metrics.sh`)

```bash
#!/bin/bash
USERNAME=$(whoami)
SUDOERS_FILE="/etc/sudoers.d/qwen3-powermetrics"

echo "Setting up GPU metrics permissions..."
echo "$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/powermetrics --samplers gpu_power -i *" | sudo tee $SUDOERS_FILE
sudo chmod 440 $SUDOERS_FILE

echo "✓ GPU metrics configured."
```

### Manual Setup

```bash
sudo bash -c 'echo "$(whoami) ALL=(ALL) NOPASSWD: /usr/bin/powermetrics --samplers gpu_power -i *" > /etc/sudoers.d/qwen3-powermetrics'
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
| `psutil` not available | Fallback to `resource` module for memory |
| Prometheus scrape fails | Metrics accumulate in memory (bounded) |

## Dependencies

- `prometheus_client` - Prometheus metrics library

## Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'qwen3-embeddings'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/prometheus'
```

## Open Questions

- None at this time.
