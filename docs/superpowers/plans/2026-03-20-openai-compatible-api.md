# OpenAI-Compatible Embeddings API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace existing embedding endpoints with OpenAI-compatible `/v1/embeddings` endpoint

**Architecture:** Clean separation with dedicated modules - Pydantic models in `models/openai_compat.py`, model mapping logic in `utils/model_mapping.py`, updated server.py with single endpoint

**Tech Stack:** FastAPI, Pydantic, NumPy, MLX

**Note:** This plan follows implementation-first order (not strict TDD) because it's an API refactoring where tests can only run after the endpoint exists. Tests are updated in Task 4, immediately before the integration test in Task 5.

---

## File Structure

```
qwen3-embeddings-mlx/
├── models/
│   ├── __init__.py           # Exports OpenAI models
│   └── openai_compat.py      # OpenAI-compatible Pydantic models
├── utils/
│   ├── __init__.py           # Exports helpers
│   └── model_mapping.py      # Model name mapping + truncate helper
├── server.py                 # Updated with /v1/embeddings
└── tests/
    └── test_api.py           # Updated tests for OpenAI format
```

---

## Task 1: Create OpenAI-Compatible Pydantic Models

**Files:**
- Create: `models/__init__.py`
- Create: `models/openai_compat.py`

- [ ] **Step 1: Create models directory and __init__.py**

```bash
mkdir -p models
```

```python
# models/__init__.py
from .openai_compat import (
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    EmbeddingObject,
    UsageInfo,
    OpenAIError,
    OpenAIErrorResponse,
)

__all__ = [
    "OpenAIEmbeddingRequest",
    "OpenAIEmbeddingResponse",
    "EmbeddingObject",
    "UsageInfo",
    "OpenAIError",
    "OpenAIErrorResponse",
]
```

- [ ] **Step 2: Create OpenAI-compatible Pydantic models**

```python
# models/openai_compat.py
"""
OpenAI-compatible Pydantic models for embeddings API.
Matches OpenAI's embedding API schema for drop-in compatibility.
"""

from typing import List, Union, Optional
from pydantic import BaseModel, Field, field_validator


class OpenAIEmbeddingRequest(BaseModel):
    """Request model matching OpenAI's /v1/embeddings endpoint"""

    input: Union[str, List[str]] = Field(
        ...,
        description="Input text to embed, can be a string or array of strings"
    )
    model: str = Field(
        ...,
        description="Model to use for embedding (OpenAI name or Qwen name/alias)"
    )
    encoding_format: str = Field(
        default="float",
        description="Format for embeddings (only 'float' supported)"
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="Truncate embeddings to this dimension (optional)"
    )

    @field_validator('encoding_format')
    @classmethod
    def validate_encoding_format(cls, v):
        if v != "float":
            raise ValueError("Only 'float' encoding_format is supported")
        return v

    @field_validator('input')
    @classmethod
    def validate_input(cls, v):
        if isinstance(v, str):
            if not v or v.isspace():
                raise ValueError("Input cannot be empty or whitespace only")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Input array cannot be empty")
            for i, text in enumerate(v):
                if not text or (isinstance(text, str) and text.isspace()):
                    raise ValueError(f"Input at index {i} cannot be empty or whitespace only")
        return v


class EmbeddingObject(BaseModel):
    """Single embedding object in response"""

    object: str = Field(default="embedding", description="Object type")
    embedding: List[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index in the input array")


class UsageInfo(BaseModel):
    """Token usage information"""

    prompt_tokens: int = Field(..., description="Number of tokens in prompt")
    total_tokens: int = Field(..., description="Total tokens used")


class OpenAIEmbeddingResponse(BaseModel):
    """Response model matching OpenAI's embedding response"""

    object: str = Field(default="list", description="Object type")
    data: List[EmbeddingObject] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used")
    usage: UsageInfo = Field(..., description="Token usage info")


class OpenAIError(BaseModel):
    """OpenAI-style error object"""

    message: str = Field(..., description="Error message")
    type: str = Field(default="invalid_request_error", description="Error type")
    param: Optional[str] = Field(default=None, description="Parameter that caused error")
    code: Optional[str] = Field(default=None, description="Error code")


class OpenAIErrorResponse(BaseModel):
    """OpenAI-style error response wrapper"""

    error: OpenAIError = Field(..., description="Error details")
```

- [ ] **Step 3: Commit models**

```bash
git add models/__init__.py models/openai_compat.py
git commit -m "feat: add OpenAI-compatible Pydantic models for embeddings API"
```

---

## Task 2: Create Model Mapping Utilities

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/model_mapping.py`

- [ ] **Step 1: Create utils directory and __init__.py**

```bash
mkdir -p utils
```

```python
# utils/__init__.py
from .model_mapping import resolve_model, truncate_embedding, count_tokens_batch

__all__ = ["resolve_model", "truncate_embedding", "count_tokens_batch"]
```

- [ ] **Step 2: Create model mapping module**

```python
# utils/model_mapping.py
"""
Model mapping utilities for OpenAI-to-Qwen compatibility.
Maps OpenAI model names to Qwen equivalents and provides helper functions.
"""

from typing import List, Optional, Tuple, Any


# Mapping from OpenAI model names to Qwen aliases
MODEL_MAPPING = {
    # OpenAI embedding models -> Qwen equivalents
    "text-embedding-3-small": "small",
    "text-embedding-3-large": "large",
    "text-embedding-ada-002": "small",
    "text-embedding-v1": "small",
    # Common aliases
    "ada-002": "small",
    "ada": "small",
}


def resolve_model(model: str) -> str:
    """
    Resolve model name to Qwen model alias or name.

    Args:
        model: Model name (OpenAI name, Qwen alias, or full Qwen name)

    Returns:
        Resolved model name/alias for Qwen
    """
    # Check if it's an OpenAI model name
    if model in MODEL_MAPPING:
        return MODEL_MAPPING[model]

    # Pass through Qwen names/aliases as-is
    return model


def truncate_embedding(embedding: List[float], dimensions: int) -> List[float]:
    """
    Truncate embedding vector to specified dimensions.

    Simple truncation by taking first N elements.
    This works because embedding vectors typically have
    information concentrated in earlier dimensions.

    Args:
        embedding: Full embedding vector
        dimensions: Target dimension count

    Returns:
        Truncated embedding vector

    Raises:
        ValueError: If dimensions > embedding length or <= 0
    """
    if dimensions <= 0:
        raise ValueError(f"Dimensions must be positive, got {dimensions}")

    if dimensions > len(embedding):
        raise ValueError(
            f"Requested dimensions ({dimensions}) exceeds embedding size ({len(embedding)})"
        )

    return embedding[:dimensions]


def count_tokens(text: str, tokenizer: Any) -> int:
    """
    Count tokens in text using tokenizer.

    Args:
        text: Text to count tokens for
        tokenizer: MLX tokenizer instance

    Returns:
        Number of tokens
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


def count_tokens_batch(texts: List[str], tokenizer: Any) -> int:
    """
    Count total tokens across multiple texts.

    Args:
        texts: List of texts
        tokenizer: MLX tokenizer instance

    Returns:
        Total token count
    """
    return sum(count_tokens(text, tokenizer) for text in texts)
```

- [ ] **Step 3: Commit utils**

```bash
git add utils/__init__.py utils/model_mapping.py
git commit -m "feat: add model mapping utilities for OpenAI compatibility"
```

---

## Task 3: Update Server with OpenAI Endpoint

**Files:**
- Modify: `server.py`

- [ ] **Step 1: Add imports at top of server.py (after existing imports)**

Add after line 28 (after `import uvicorn`):

```python
# OpenAI compatibility imports
from models import (
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    EmbeddingObject,
    UsageInfo,
    OpenAIError,
    OpenAIErrorResponse,
)
from utils import resolve_model, truncate_embedding, count_tokens_batch
```

- [ ] **Step 2: Remove legacy Pydantic models**

Delete lines 373-441 (EmbedRequest, EmbedResponse, BatchEmbedRequest, BatchEmbedResponse classes):

Remove these classes:
- `EmbedRequest` (lines 373-396)
- `EmbedResponse` (lines 398-404)
- `BatchEmbedRequest` (lines 406-432)
- `BatchEmbedResponse` (lines 434-441)

**Important:** Do NOT delete `HealthResponse` which starts at line 443.

- [ ] **Step 3: Replace existing ValueError error handler**

Find and replace the existing `value_error_handler` function (search for `@app.exception_handler(ValueError)`):

**Replace this:**

```python
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )
```

**With this:**

```python
@app.exception_handler(ValueError)
async def openai_error_handler(request: Request, exc: ValueError):
    """Handle validation errors with OpenAI-style response"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=OpenAIErrorResponse(
            error=OpenAIError(
                message=str(exc),
                type="invalid_request_error",
            )
        ).model_dump()
    )
```

- [ ] **Step 4: Replace /embed and /embed_batch endpoints with /v1/embeddings**

Replace the `/embed` and `/embed_batch` endpoints (lines 545-624) with:

```python
@app.post(
    "/v1/embeddings",
    response_model=OpenAIEmbeddingResponse,
    tags=["Embeddings"],
    status_code=status.HTTP_200_OK
)
async def create_embeddings(request: OpenAIEmbeddingRequest):
    """
    Create embeddings for input text(s).

    OpenAI-compatible endpoint that accepts single text or batch of texts.
    Supports model name mapping (OpenAI names -> Qwen models) and
    optional dimension truncation.
    """
    try:
        start_time = time.time()

        # Normalize input to list
        texts = [request.input] if isinstance(request.input, str) else request.input

        # Resolve model name (OpenAI -> Qwen)
        model_resolved = resolve_model(request.model)

        # Generate embeddings using existing ModelManager
        embeddings, model_used, embedding_dim = await model_manager.generate_embeddings(
            texts,
            model_name=model_resolved,
            normalize=True
        )

        # Count tokens for usage info
        model_tuple = model_manager.models.get(model_used)
        if model_tuple:
            _, tokenizer = model_tuple
            total_tokens = count_tokens_batch(texts, tokenizer)
        else:
            total_tokens = 0  # Fallback if model not in cache

        # Build response with optional truncation
        data = []
        for i, emb in enumerate(embeddings):
            emb_list = emb.tolist()

            # Apply dimension truncation if requested
            if request.dimensions:
                emb_list = truncate_embedding(emb_list, request.dimensions)

            data.append(EmbeddingObject(
                object="embedding",
                embedding=emb_list,
                index=i
            ))

        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Generated {len(data)} embeddings in {processing_time:.2f}ms")

        return OpenAIEmbeddingResponse(
            object="list",
            data=data,
            model=request.model,  # Return original model name for compatibility
            usage=UsageInfo(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=OpenAIErrorResponse(
                error=OpenAIError(
                    message=str(e),
                    type="invalid_request_error",
                )
            ).model_dump()
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=OpenAIErrorResponse(
                error=OpenAIError(
                    message=f"Embedding generation failed: {str(e)}",
                    type="server_error",
                )
            ).model_dump()
        )
```

- [ ] **Step 5: Update root endpoint info**

Update the `/` endpoint (around line 527) to reflect new endpoint:

```python
@app.get("/", tags=["General"])
async def root():
    """Get API information"""
    return {
        "service": "Qwen3 Embedding Server",
        "version": app.version,
        "default_model": config.model_name,
        "available_models": list(AVAILABLE_MODELS.keys()),
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "health": "/health",
            "metrics": "/metrics",
            "models": "/models",
            "documentation": "/docs"
        }
    }
```

- [ ] **Step 6: Commit server changes**

```bash
git add server.py
git commit -m "feat: replace legacy endpoints with OpenAI-compatible /v1/embeddings"
```

---

## Task 4: Update Tests for OpenAI Format

**Files:**
- Modify: `tests/test_api.py`

- [ ] **Step 1: Update test constants and model config**

Replace lines 16-28:

```python
# Configuration
BASE_URL = "http://localhost:8000"
TOLERANCE = 0.01  # For float comparisons

# Model configurations (Qwen models)
MODELS = {
    "small": {"dim": 1024, "alias": "small"},
    "medium": {"dim": 2560, "alias": "medium"},
    "large": {"dim": 4096, "alias": "large"}
}

# OpenAI model mapping for tests
OPENAI_MODEL_MAPPING = {
    "text-embedding-3-small": "small",
    "text-embedding-3-large": "large",
    "text-embedding-ada-002": "small",
}
```

- [ ] **Step 2: Replace test methods with OpenAI format tests**

Replace `test_single_embedding` method (lines 75-109):

```python
    def test_single_embedding(self, model: str = "small") -> Dict[str, Any]:
        """Test single text embedding with OpenAI format"""
        test_text = "Machine learning is transforming the world"

        payload = {
            "input": test_text,
            "model": model,
            "encoding_format": "float"
        }

        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload
        )

        assert response.status_code == 200, f"Embedding failed: {response.text}"

        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["index"] == 0
        assert "embedding" in data["data"][0]
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

        # Get expected dimension for model
        expected_dim = MODELS.get(model, {"dim": 1024})["dim"]

        # Validate embedding
        embedding = np.array(data["data"][0]["embedding"])
        assert embedding.shape == (expected_dim,), f"Wrong dimension: {embedding.shape}"

        # Check normalization (Qwen models return normalized embeddings)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < TOLERANCE, f"Not normalized: norm={norm}"

        return data
```

- [ ] **Step 3: Replace batch embedding test**

Replace `test_batch_embedding` method (lines 111-151):

```python
    def test_batch_embedding(self, model: str = "small") -> Dict[str, Any]:
        """Test batch embedding with OpenAI format"""
        test_texts = [
            "Python is a great programming language",
            "FastAPI makes building APIs easy",
            "MLX is optimized for Apple Silicon"
        ]

        payload = {
            "input": test_texts,
            "model": model,
            "encoding_format": "float"
        }

        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload
        )

        assert response.status_code == 200, f"Batch embedding failed: {response.text}"

        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == len(test_texts)
        assert "usage" in data

        # Get expected dimension for model
        expected_dim = MODELS.get(model, {"dim": 1024})["dim"]

        # Validate embeddings
        for i, emb_obj in enumerate(data["data"]):
            assert emb_obj["object"] == "embedding"
            assert emb_obj["index"] == i
            embedding = np.array(emb_obj["embedding"])
            assert embedding.shape == (expected_dim,), f"Wrong shape at index {i}"

            # Check normalization
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < TOLERANCE, f"Not normalized at index {i}: norm={norm}"

        return data
```

- [ ] **Step 4: Add dimension truncation test**

Add new method after `test_large_batch`:

```python
    def test_dimension_truncation(self) -> None:
        """Test dimension truncation feature"""
        test_text = "Testing dimension truncation"
        target_dim = 512

        payload = {
            "input": test_text,
            "model": "small",  # 1024 dimensions
            "dimensions": target_dim
        }

        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload
        )

        assert response.status_code == 200, f"Truncation failed: {response.text}"

        data = response.json()
        embedding = np.array(data["data"][0]["embedding"])
        assert embedding.shape == (target_dim,), f"Wrong truncated dimension: {embedding.shape}"
```

- [ ] **Step 5: Add model mapping test**

Add new method after `test_dimension_truncation`:

```python
    def test_model_mapping(self) -> None:
        """Test OpenAI model name mapping to Qwen models"""
        test_text = "Testing model mapping"

        # Test OpenAI model name -> Qwen mapping
        for openai_model, expected_alias in OPENAI_MODEL_MAPPING.items():
            payload = {
                "input": test_text,
                "model": openai_model
            }

            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload
            )

            assert response.status_code == 200, f"Model mapping failed for {openai_model}: {response.text}"

            data = response.json()
            # Model name in response should be the original (OpenAI) name
            assert data["model"] == openai_model

            # Embedding dimension should match expected Qwen model
            expected_dim = MODELS[expected_alias]["dim"]
            embedding = np.array(data["data"][0]["embedding"])
            assert embedding.shape == (expected_dim,), f"Wrong dim for {openai_model}"
```

- [ ] **Step 6: Update empty text test**

Replace `test_empty_text` method (lines 153-160):

```python
    def test_empty_input(self) -> None:
        """Test handling of empty input"""
        # Empty string
        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": "", "model": "small"}
        )
        assert response.status_code == 422, "Empty input should be rejected"

        # Empty array
        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": [], "model": "small"}
        )
        assert response.status_code == 422, "Empty array should be rejected"
```

- [ ] **Step 7: Update large batch test**

Replace `test_large_batch` method (lines 162-172):

```python
    def test_large_batch(self) -> None:
        """Test handling of large batch"""
        large_texts = ["Test text"] * 100

        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": large_texts, "model": "small"}
        )

        # Should succeed (server allows up to 1024 batch size)
        assert response.status_code == 200, f"Large batch failed: {response.text}"
        assert len(response.json()["data"]) == 100
```

- [ ] **Step 8: Update similarity test**

Replace `test_similarity` method (lines 174-196):

```python
    def test_similarity(self) -> None:
        """Test semantic similarity"""
        pairs = [
            ("dog", "puppy", 0.3),  # Should be similar
            ("dog", "car", 0.1),     # Should be dissimilar
            ("AI", "artificial intelligence", 0.2),  # Should be similar
        ]

        for text1, text2, min_similarity in pairs:
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": [text1, text2], "model": "small"}
            )

            assert response.status_code == 200
            data = response.json()
            embeddings = np.array([d["embedding"] for d in data["data"]])

            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1])

            if min_similarity > 0:
                assert similarity >= min_similarity, \
                    f"'{text1}' and '{text2}' similarity {similarity:.3f} < {min_similarity}"
```

- [ ] **Step 9: Update performance test**

Replace `test_performance` method (lines 198-229):

```python
    def test_performance(self) -> Dict[str, float]:
        """Test performance metrics"""
        metrics = {}

        # Single embedding latency
        times = []
        for _ in range(5):
            start = time.time()
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": "Performance test", "model": "small"}
            )
            times.append((time.time() - start) * 1000)
            assert response.status_code == 200

        metrics["single_embed_ms"] = np.mean(times[1:])  # Skip first (warmup)

        # Batch embedding latency
        times = []
        for _ in range(3):
            start = time.time()
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": ["Test"] * 10, "model": "small"}
            )
            times.append((time.time() - start) * 1000)
            assert response.status_code == 200

        metrics["batch_10_ms"] = np.mean(times)
        metrics["throughput_per_sec"] = 10000 / metrics["batch_10_ms"]

        return metrics
```

- [ ] **Step 10: Update run_tests function**

Replace `run_tests` function (lines 231-287):

```python
def run_tests():
    """Run all tests"""
    print("🧪 Qwen3 Embedding Server - OpenAI Compatible API Tests")
    print("=" * 50)

    client = TestClient()

    # Check server
    if not client.check_server():
        print("❌ Server is not running. Start with: python server.py")
        return False

    results = {"passed": 0, "failed": 0}

    # Test suite
    tests = [
        ("Health Check", client.test_health),
        ("Models Endpoint", client.test_models_endpoint),
        ("Single Embedding (small)", lambda: client.test_single_embedding("small")),
        ("Single Embedding (medium)", lambda: client.test_single_embedding("medium")),
        ("Batch Embedding (small)", lambda: client.test_batch_embedding("small")),
        ("Batch Embedding (medium)", lambda: client.test_batch_embedding("medium")),
        ("Empty Input Validation", client.test_empty_input),
        ("Large Batch Handling", client.test_large_batch),
        ("Dimension Truncation", client.test_dimension_truncation),
        ("Model Mapping (OpenAI names)", client.test_model_mapping),
        ("Semantic Similarity", client.test_similarity),
        ("Performance Metrics", client.test_performance),
    ]

    for test_name, test_func in tests:
        try:
            print(f"\n📋 {test_name}")
            result = test_func()

            if result:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, float):
                            print(f"  ✓ {key}: {value:.2f}")
                        else:
                            print(f"  ✓ {key}: {value}")

            print(f"  ✅ Passed")
            results["passed"] += 1

        except AssertionError as e:
            print(f"  ❌ Failed: {e}")
            results["failed"] += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["failed"] += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"Results: {results['passed']} passed, {results['failed']} failed")

    return results["failed"] == 0
```

- [ ] **Step 11: Commit test updates**

```bash
git add tests/test_api.py
git commit -m "test: update tests for OpenAI-compatible /v1/embeddings endpoint"
```

---

## Task 5: Integration Test

- [ ] **Step 1: Start server and run tests**

```bash
# Terminal 1: Start server
python server.py

# Terminal 2: Run tests
python tests/test_api.py
```

Expected: All 12 tests pass

- [ ] **Step 2: Test with curl (OpenAI format)**

```bash
# Single embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "text-embedding-3-small"}'

# Batch embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello", "World"], "model": "small"}'

# With dimension truncation
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "small", "dimensions": 512}'
```

Expected: All return valid OpenAI-format responses

---

## Task 6: Final Commit and Cleanup

- [ ] **Step 1: Update version in server.py**

Update line 479:

```python
version="2.0.0",  # Major version bump for API change
```

- [ ] **Step 2: Final commit**

```bash
git add server.py
git commit -m "chore: bump version to 2.0.0 for OpenAI-compatible API"
```

---

## Summary

| Task | Files | Description |
|------|-------|-------------|
| 1 | `models/*.py` | OpenAI-compatible Pydantic models |
| 2 | `utils/*.py` | Model mapping + truncate helpers |
| 3 | `server.py` | Replace endpoints with /v1/embeddings |
| 4 | `tests/test_api.py` | Updated tests for OpenAI format |
| 5 | - | Integration testing |
| 6 | `server.py` | Version bump |
