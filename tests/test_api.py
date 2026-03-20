#!/usr/bin/env python3
"""
API Tests for Qwen3 Embedding Server - OpenAI Compatible

Run with: python tests/test_api.py
Or with pytest: pytest tests/test_api.py -v
"""

import sys
import time
from typing import Dict, Any
import requests
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"
TOLERANCE = 0.01  # For float comparisons

# Model configurations (Qwen models)
MODELS = {
    "small": {"dim": 1024, "alias": "small"},
    "medium": {"dim": 2560, "alias": "medium"},
    "large": {"dim": 4096, "alias": "large"},
}

# OpenAI model mapping for tests
OPENAI_MODEL_MAPPING = {
    "text-embedding-3-small": "small",
    "text-embedding-3-large": "large",
    "text-embedding-ada-002": "small",
}


class TestClient:
    """Test client for Qwen3 Embedding Server"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def check_server(self) -> bool:
        """Check if server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        response = self.session.get(f"{self.base_url}/health")
        assert response.status_code == 200, (
            f"Health check failed: {response.status_code}"
        )

        data = response.json()
        assert "status" in data
        assert "model_status" in data
        assert "embedding_dim" in data
        # Default model should be small (1024 dim)
        assert data["embedding_dim"] == 1024

        return data

    def test_models_endpoint(self) -> Dict[str, Any]:
        """Test models listing endpoint"""
        response = self.session.get(f"{self.base_url}/models")
        assert response.status_code == 200, (
            f"Models endpoint failed: {response.status_code}"
        )

        data = response.json()
        assert "models" in data
        assert "default_model" in data
        assert "loaded_models" in data

        # Check that all expected models are listed
        models = data["models"]
        assert len(models) >= 3, "Should have at least 3 models available"

        return data

    def test_single_embedding(self, model: str = "small") -> Dict[str, Any]:
        """Test single text embedding with OpenAI format"""
        test_text = "Machine learning is transforming the world"

        payload = {"input": test_text, "model": model, "encoding_format": "float"}

        response = self.session.post(f"{self.base_url}/v1/embeddings", json=payload)

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

    def test_batch_embedding(self, model: str = "small") -> Dict[str, Any]:
        """Test batch embedding with OpenAI format"""
        test_texts = [
            "Python is a great programming language",
            "FastAPI makes building APIs easy",
            "MLX is optimized for Apple Silicon",
        ]

        payload = {"input": test_texts, "model": model, "encoding_format": "float"}

        response = self.session.post(f"{self.base_url}/v1/embeddings", json=payload)

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
            assert abs(norm - 1.0) < TOLERANCE, (
                f"Not normalized at index {i}: norm={norm}"
            )

        return data

    def test_empty_input(self) -> None:
        """Test handling of empty input"""
        # Empty string
        response = self.session.post(
            f"{self.base_url}/v1/embeddings", json={"input": "", "model": "small"}
        )
        assert response.status_code == 422, "Empty input should be rejected"

        # Empty array
        response = self.session.post(
            f"{self.base_url}/v1/embeddings", json={"input": [], "model": "small"}
        )
        assert response.status_code == 422, "Empty array should be rejected"

    def test_large_batch(self) -> None:
        """Test handling of large batch"""
        large_texts = ["Test text"] * 100

        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": large_texts, "model": "small"},
        )

        # Should succeed (server allows up to 1024 batch size)
        assert response.status_code == 200, f"Large batch failed: {response.text}"
        assert len(response.json()["data"]) == 100

    def test_dimension_truncation(self) -> None:
        """Test dimension truncation feature"""
        test_text = "Testing dimension truncation"
        target_dim = 512

        payload = {"input": test_text, "model": "small", "dimensions": target_dim}

        response = self.session.post(f"{self.base_url}/v1/embeddings", json=payload)

        assert response.status_code == 200, f"Truncation failed: {response.text}"

        data = response.json()
        embedding = np.array(data["data"][0]["embedding"])
        assert embedding.shape == (target_dim,), (
            f"Wrong truncated dimension: {embedding.shape}"
        )

    def test_model_mapping(self) -> None:
        """Test OpenAI model name mapping to Qwen models"""
        test_text = "Testing model mapping"

        # Test OpenAI model name -> Qwen mapping
        for openai_model, expected_alias in OPENAI_MODEL_MAPPING.items():
            payload = {"input": test_text, "model": openai_model}

            response = self.session.post(f"{self.base_url}/v1/embeddings", json=payload)

            assert response.status_code == 200, (
                f"Model mapping failed for {openai_model}: {response.text}"
            )

            data = response.json()
            # Model name in response should be the original (OpenAI) name
            assert data["model"] == openai_model

            # Embedding dimension should match expected Qwen model
            expected_dim = MODELS[expected_alias]["dim"]
            embedding = np.array(data["data"][0]["embedding"])
            assert embedding.shape == (expected_dim,), f"Wrong dim for {openai_model}"

    def test_similarity(self) -> None:
        """Test semantic similarity"""
        pairs = [
            ("dog", "puppy", 0.3),  # Should be similar
            ("dog", "car", 0.1),  # Should be dissimilar
            ("AI", "artificial intelligence", 0.2),  # Should be similar
        ]

        for text1, text2, min_similarity in pairs:
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": [text1, text2], "model": "small"},
            )

            assert response.status_code == 200
            data = response.json()
            embeddings = np.array([d["embedding"] for d in data["data"]])

            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1])

            if min_similarity > 0:
                assert similarity >= min_similarity, (
                    f"'{text1}' and '{text2}' similarity {similarity:.3f} "
                    f"< {min_similarity}"
                )

    def test_performance(self) -> Dict[str, float]:
        """Test performance metrics"""
        metrics = {}

        # Single embedding latency
        times = []
        for _ in range(5):
            start = time.time()
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": "Performance test", "model": "small"},
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
                json={"input": ["Test"] * 10, "model": "small"},
            )
            times.append((time.time() - start) * 1000)
            assert response.status_code == 200

        metrics["batch_10_ms"] = np.mean(times)
        metrics["throughput_per_sec"] = 10000 / metrics["batch_10_ms"]

        return metrics


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


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
