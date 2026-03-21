# Qwen3 Embeddings Server for Mac

**OpenAI/Cohere-compatible embeddings & reranking on your Mac.** Drop-in replacement for OpenAI embeddings and Cohere rerank APIs. Works with LiteLLM, LangChain, and more. 🚀

![Version](https://img.shields.io/badge/version-2.1.0-blue)
![0.6B Speed](https://img.shields.io/badge/0.6B-44K_tokens/sec-green)
![4B Speed](https://img.shields.io/badge/4B-18K_tokens/sec-blue)
![8B Speed](https://img.shields.io/badge/8B-11K_tokens/sec-purple)
![Platform](https://img.shields.io/badge/Platform-Apple_Silicon-black)

## ✨ Features

- **OpenAI-Compatible Embeddings** - Drop-in replacement for `text-embedding-3-*` models
- **Cohere-Compatible Reranking** - `/v1/rerank` endpoint for document reranking
- **LiteLLM Integration** - Works seamlessly with LiteLLM proxy
- **3 Embedding Sizes** - Small (0.6B), Medium (4B), Large (8B)
- **2 Reranker Sizes** - Small (0.6B), Large (4B)
- **Dimension Truncation** - Reduce embedding size on the fly
- **OpenTelemetry Metrics** - Built-in OTLP observability
- **Apple Silicon Optimized** - Leverages MLX framework for M1/M2/M3/M4 chips

## 🏃 Quick Start

### Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- 1-5GB free space (depending on model)

### Install & Run

```bash
# Clone
git clone https://github.com/lucndm/qwen3-embeddings-mlx.git
cd qwen3-embeddings-mlx

# Install with uv
uv sync

# Run
uv run python server.py
```

Server runs at `http://localhost:8000`

### Test

```bash
# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "text-embedding-3-small"}'

# Rerank
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "small",
    "query": "What is machine learning?",
    "documents": ["ML is AI", "Weather is nice", "Neural networks"],
    "top_n": 2
  }'
```

## 🔌 LiteLLM Integration (M4 Mac Use Case)

This server is designed to run locally on M4 Macs with LiteLLM proxy for RAG pipelines.

### Embeddings Config

Add to your `litellm_config.yaml`:

```yaml
model_list:
  - model_name: text-embedding-3-small
    litellm_params:
      model: openai/text-embedding-3-small
      api_base: http://localhost:8000/v1
      api_key: dummy

  - model_name: text-embedding-3-medium
    litellm_params:
      model: openai/text-embedding-3-medium
      api_base: http://localhost:8000/v1
      api_key: dummy

  - model_name: text-embedding-3-large
    litellm_params:
      model: openai/text-embedding-3-large
      api_base: http://localhost:8000/v1
      api_key: dummy
```

### Rerank via Direct API

LiteLLM doesn't natively support rerank, so call directly:

```python
import requests

# Rerank documents for RAG
response = requests.post(
    "http://localhost:8000/v1/rerank",
    json={
        "model": "small",  # or "large" for 4B model
        "query": "What is the revenue?",
        "documents": retrieved_chunks,
        "top_n": 5
    }
)

ranked_results = response.json()["results"]
# Use top results for LLM context
```

### Full RAG Pipeline Example

```python
from litellm import embedding
import requests

# 1. Embed query
query = "What is machine learning?"
emb = embedding(model="text-embedding-3-small", input=[query])

# 2. Vector search (your vector DB here)
# candidates = vector_db.search(emb.data[0]["embedding"], top_k=20)

# 3. Rerank candidates
rerank_resp = requests.post(
    "http://localhost:8000/v1/rerank",
    json={
        "model": "small",
        "query": query,
        "documents": [c["text"] for c in candidates],
        "top_n": 5
    }
)

# 4. Use top-ranked docs for LLM
top_docs = [candidates[r["index"]] for r in rerank_resp.json()["results"]]
```

## 📖 API Reference

### POST /v1/embeddings

OpenAI-compatible embeddings endpoint.

**Request:**

```json
{
  "input": "string or array of strings",
  "model": "text-embedding-3-small",
  "encoding_format": "float",
  "dimensions": 512
}
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
```

### POST /v1/rerank

Cohere-compatible rerank endpoint. Takes a query and documents, returns sorted by relevance.

**Request:**

```json
{
  "model": "small",
  "query": "What is machine learning?",
  "documents": ["Doc 1 text", "Doc 2 text", "Doc 3 text"],
  "top_n": 2,
  "max_tokens_per_doc": 4096
}
```

**Response:**

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.95},
    {"index": 2, "relevance_score": 0.72}
  ],
  "id": "uuid-request-id"
}
```

### Model Mapping

**Embeddings:**

| OpenAI Model | Qwen Model | Dimensions | Speed |
|--------------|------------|------------|-------|
| `text-embedding-3-small` | Qwen3 0.6B | 1024 | ⚡⚡⚡ 44K tok/s |
| `text-embedding-3-medium` | Qwen3 4B | 2560 | ⚡⚡ 18K tok/s |
| `text-embedding-3-large` | Qwen3 8B | 4096 | ⚡ 11K tok/s |
| `text-embedding-ada-002` | Qwen3 0.6B | 1024 | ⚡⚡⚡ |

**Rerank:**

| Alias | Full Model | Size |
|-------|------------|------|
| `small`, `0.6b`, `default` | Qwen3-Reranker-0.6B | ~900MB |
| `large`, `4b` | Qwen3-Reranker-4B | ~2.5GB |
| `rerank-v3.5`, `rerank-v4.0` | → small | - |
| `rerank-v4.0-pro` | → large | - |

### Other Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /models` | List available embedding models |
| `GET /metrics` | Server metrics |
| `GET /metrics/prometheus` | Prometheus-format metrics |
| `GET /` | API info |

## 💻 Usage Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Required but unused
)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Hello world",
    dimensions=512  # Optional: truncate to 512 dims
)

embedding = response.data[0].embedding
print(f"Dimensions: {len(embedding)}")  # 512
```

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "input": ["Hello", "World"],
        "model": "text-embedding-3-medium"
    }
)

data = response.json()
for item in data["data"]:
    print(f"Index {item['index']}: {len(item['embedding'])} dims")
```

### LangChain

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="dummy"
)

vector = embeddings.embed_query("Hello world")
```

### JavaScript

```javascript
const response = await fetch("http://localhost:8000/v1/embeddings", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    input: "Hello world",
    model: "text-embedding-3-small"
  })
});

const data = await response.json();
console.log(`Dimensions: ${data.data[0].embedding.length}`);
```

### cURL

```bash
# Single embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "text-embedding-3-small"}'

# Batch embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello", "World"], "model": "text-embedding-3-medium"}'

# With dimension truncation
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "text-embedding-3-small", "dimensions": 256}'
```

## 📊 OpenTelemetry Metrics

Built-in OTLP metrics support for observability:

| Metric | Type | Description |
|--------|------|-------------|
| `embedding_requests_total` | Counter | Total requests (embeddings + rerank) |
| `embedding_latency_ms` | Histogram | Request latency (ms) |
| `tokens_total` | Counter | Total tokens processed |
| `embedding_errors_total` | Counter | Total errors |

All metrics include `model` and `endpoint` (embeddings/rerank) attributes.

**Configuration:**

```bash
OTEL_SERVICE_NAME=qwen3-embedding-server \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
uv run python server.py
```

Works with any OTLP-compatible backend (Grafana, Datadog, Honeycomb, etc.).

### CPU/GPU Monitoring (Prometheus)

Built-in Prometheus metrics for monitoring CPU/GPU utilization and system health on Apple Silicon.

**Quick Setup:**

```bash
# Install dependencies and setup monitoring
./scripts/setup-gpu-metrics.sh

# Start server with metrics enabled
ENABLE_PROMETHEUS_METRICS=true uv run python server.py

# Verify metrics endpoint
curl http://localhost:8000/metrics/prometheus
```

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `server_cpu_percent` | Gauge | CPU utilization percentage |
| `server_memory_percent` | Gauge | Memory utilization percentage |
| `server_memory_used_mb` | Gauge | Memory used in MB |
| `server_memory_available_mb` | Gauge | Available memory in MB |
| `gpu_utilization_percent` | Gauge | GPU utilization percentage |
| `gpu_memory_used_mb` | Gauge | GPU memory used in MB |
| `gpu_memory_total_mb` | Gauge | Total GPU memory in MB |
| `gpu_memory_percent` | Gauge | GPU memory utilization percentage |
| `gpu_temperature_celsius` | Gauge | GPU temperature in Celsius |
| `gpu_power_watts` | Gauge | GPU power consumption in watts |
| `up` | Gauge | Server health status (1=healthy) |
| `uptime_seconds` | Gauge | Server uptime in seconds |

**Prometheus Configuration:**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'qwen3-embeddings'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'qwen3-embedding-server'
```

**Grafana Dashboard:**

Import the provided Grafana dashboard for visualization:

```bash
# Import dashboard JSON from docs/monitoring/grafana-dashboard.json
# Navigate to Grafana UI → Dashboards → Import → Upload JSON
```

## ⚙️ Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | Qwen3-0.6B | Default model |
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Server host |
| `MAX_BATCH_SIZE` | 1024 | Max texts per batch |
| `MAX_TEXT_LENGTH` | 8192 | Max tokens per text |
| `LOG_LEVEL` | INFO | Logging level |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | - | OTLP collector endpoint |
| `ENABLE_PROMETHEUS_METRICS` | true | Enable Prometheus endpoint |
| `METRICS_COLLECTION_INTERVAL` | 2.0 | System metrics interval (seconds) |
| `GPU_METRICS_ENABLED` | true | Enable GPU metrics collection |

## 🛠️ Make Commands

```bash
make run          # Start server
make dev          # Development mode with auto-reload
make test         # Run tests
make benchmark    # Performance benchmark
make health       # Health check
make help         # Show all commands
```

## 📈 Performance

Benchmarks from M4 Mac:

| Model | Throughput | Latency | Memory |
|-------|------------|---------|--------|
| 0.6B (small) | 44K tok/s | ~1.3ms | 900MB |
| 4B (medium) | 18K tok/s | ~3-5ms | 2.5GB |
| 8B (large) | 11K tok/s | ~8-12ms | 4.5GB |

**Rerank:**

| Model | Latency (per doc) | Memory |
|-------|-------------------|--------|
| 0.6B (small) | ~2ms | 900MB |
| 4B (large) | ~5ms | 2.5GB |

## 🚀 Why Use This?

- **Drop-in Replacement** - Works with existing OpenAI/LiteLLM code
- **Privacy** - Data never leaves your machine
- **Speed** - No network latency, local inference
- **Cost** - Free after setup, no API fees
- **Quality** - State-of-the-art Qwen3 models

## 📦 Project Structure

```
qwen3-embeddings-mlx/
├── server.py              # Main FastAPI server
├── pyproject.toml         # uv package config
├── models/
│   ├── __init__.py
│   ├── openai_compat.py   # OpenAI-compatible Pydantic models
│   └── rerank_compat.py   # Cohere-compatible rerank models
├── utils/
│   ├── __init__.py
│   ├── model_mapping.py   # OpenAI -> Qwen model mapping
│   └── rerank_utils.py    # Cohere -> Qwen rerank mapping
├── tests/
│   ├── test_api.py        # API tests
│   └── benchmark.py       # Performance benchmarks
└── examples/              # Usage examples
```

## 🤝 Contributing

1. Fork the repo
2. Create feature branch
3. Run tests: `make test`
4. Submit PR

## 📄 License

MIT License

## 🙏 Credits

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Qwen](https://github.com/QwenLM/Qwen) - Embedding models
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

---

**Ready to start?** `python server.py` 🎉
