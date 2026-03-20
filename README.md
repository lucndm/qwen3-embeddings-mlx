# Qwen3 Embeddings Server for Mac

**OpenAI-compatible text embeddings on your Mac.** Drop-in replacement for OpenAI embeddings API. Works with LiteLLM, LangChain, and more. 🚀

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![0.6B Speed](https://img.shields.io/badge/0.6B-44K_tokens/sec-green)
![4B Speed](https://img.shields.io/badge/4B-18K_tokens/sec-blue)
![8B Speed](https://img.shields.io/badge/8B-11K_tokens/sec-purple)
![Platform](https://img.shields.io/badge/Platform-Apple_Silicon-black)

## ✨ Features

- **OpenAI-Compatible API** - Drop-in replacement for `text-embedding-3-*` models
- **LiteLLM Integration** - Works seamlessly with LiteLLM proxy
- **3 Model Sizes** - Small (0.6B), Medium (4B), Large (8B)
- **Dimension Truncation** - Reduce embedding size on the fly
- **OpenTelemetry Metrics** - Built-in observability support
- **Apple Silicon Optimized** - Leverages MLX framework for M1/M2/M3/M4 chips

## 🏃 Quick Start

### Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.9+
- 1-5GB free space (depending on model)

### Install & Run

```bash
# Clone
git clone https://github.com/lucndm/qwen3-embeddings-mlx.git
cd qwen3-embeddings-mlx

# Install
pip install -r requirements.txt

# Run
python server.py
```

Server runs at `http://localhost:8000`

### Test

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "text-embedding-3-small"}'
```

## 🔌 LiteLLM Integration

This server is fully compatible with LiteLLM. Add to your `litellm_config.yaml`:

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

Then use with any LiteLLM-compatible client:

```python
from litellm import embedding

response = embedding(
    model="text-embedding-3-small",
    input=["Hello world", "Test embedding"],
)
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

### Model Mapping

| OpenAI Model | Qwen Model | Dimensions | Speed |
|--------------|------------|------------|-------|
| `text-embedding-3-small` | Qwen3 0.6B | 1024 | ⚡⚡⚡ 44K tok/s |
| `text-embedding-3-medium` | Qwen3 4B | 2560 | ⚡⚡ 18K tok/s |
| `text-embedding-3-large` | Qwen3 8B | 4096 | ⚡ 11K tok/s |
| `text-embedding-ada-002` | Qwen3 0.6B | 1024 | ⚡⚡⚡ |

You can also use Qwen names directly: `small`, `medium`, `large`

### Other Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /models` | List available models |
| `GET /metrics` | Server metrics |
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
| `embedding_requests_total` | Counter | Total embedding requests |
| `embedding_latency_ms` | Histogram | Request latency (ms) |
| `tokens_total` | Counter | Total tokens processed |
| `embedding_errors_total` | Counter | Total errors |

**Configuration:**

```bash
OTEL_SERVICE_NAME=qwen3-embedding-server \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
python server.py
```

Install OTEL dependencies:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
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

Benchmarks from M2 Max (32GB):

| Model | Throughput | Latency | Memory |
|-------|------------|---------|--------|
| 0.6B (small) | 44K tok/s | ~1.3ms | 900MB |
| 4B (medium) | 18K tok/s | ~3-5ms | 2.5GB |
| 8B (large) | 11K tok/s | ~8-12ms | 4.5GB |

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
├── models/
│   ├── __init__.py
│   └── openai_compat.py   # OpenAI-compatible Pydantic models
├── utils/
│   ├── __init__.py
│   └── model_mapping.py   # OpenAI -> Qwen model mapping
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
