#!/usr/bin/env python3
"""
Qwen3 Embedding Server using MLX on Apple Silicon

A high-performance text embedding server optimized for Apple Silicon Macs,
providing REST API access to Qwen3 embedding models via the MLX framework.
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from functools import lru_cache
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

import numpy as np
import mlx
import mlx.core as mx
from mlx_lm import load
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator
import uvicorn

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

# Constants
DEFAULT_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

# Available models configuration
AVAILABLE_MODELS = {
    "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ": {
        "alias": ["small", "0.6b", "default"],
        "embedding_dim": 1024,
        "description": "Small 0.6B parameter model, fast and efficient",
    },
    "mlx-community/Qwen3-Embedding-4B-4bit-DWQ": {
        "alias": ["medium", "4b"],
        "embedding_dim": 2560,
        "description": "Medium 4B parameter model, balanced performance",
    },
    "mlx-community/Qwen3-Embedding-8B-4bit-DWQ": {
        "alias": ["large", "8b"],
        "embedding_dim": 4096,
        "description": "Large 8B parameter model, higher quality embeddings",
    },
}

# Build alias mapping
MODEL_ALIASES = {}
for model_name, config in AVAILABLE_MODELS.items():
    for alias in config.get("alias", []):
        MODEL_ALIASES[alias.lower()] = model_name
MIN_BATCH_SIZE = 1
DEFAULT_MAX_BATCH = 1024  # Increased for stress testing
DEFAULT_MAX_LENGTH = 8192
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"


# Configure logging
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure application logging"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


# Initialize logger
logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))


# Configuration
@dataclass
class ServerConfig:
    """Server configuration"""

    model_name: str = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", str(DEFAULT_MAX_BATCH)))
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", str(DEFAULT_MAX_LENGTH)))
    port: int = int(os.getenv("PORT", str(DEFAULT_PORT)))
    host: str = os.getenv("HOST", DEFAULT_HOST)
    enable_cors: bool = os.getenv("ENABLE_CORS", "true").lower() == "true"
    cors_origins: List[str] = None

    def __post_init__(self):
        """Validate configuration"""
        if self.cors_origins is None:
            self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        if self.max_batch_size < MIN_BATCH_SIZE:
            raise ValueError(f"max_batch_size must be at least {MIN_BATCH_SIZE}")
        if self.max_text_length < 1:
            raise ValueError("max_text_length must be positive")
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")


# Load configuration
config = ServerConfig()


class ModelStatus(str, Enum):
    """Model status enumeration"""

    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"


class ModelManager:
    """
    Manages MLX model loading, caching, and inference.

    This class handles the lifecycle of multiple embedding models,
    including loading, warming up, and generating embeddings.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.models: Dict[str, Tuple[Any, Any]] = {}  # model_name -> (model, tokenizer)
        self.model_status: Dict[str, ModelStatus] = {}  # model_name -> status
        self.model_load_times: Dict[str, float] = {}  # model_name -> load_time
        self._locks: Dict[str, asyncio.Lock] = {}  # model_name -> lock
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._global_lock = asyncio.Lock()  # For managing model dict
        self.max_loaded_models = 2  # Maximum models to keep in memory

    def _resolve_model_name(self, model_identifier: Optional[str] = None) -> str:
        """Resolve model identifier to actual model name"""
        if not model_identifier:
            return self.config.model_name

        # Check if it's an alias
        model_lower = model_identifier.lower()
        if model_lower in MODEL_ALIASES:
            return MODEL_ALIASES[model_lower]

        # Check if it's a valid model name
        if model_identifier in AVAILABLE_MODELS:
            return model_identifier

        # Invalid model
        raise ValueError(
            f"Unknown model: {model_identifier}. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    async def load_model(self, model_name: Optional[str] = None) -> str:
        """Load and initialize the specified embedding model

        Args:
            model_name: Model name or alias. If None, uses default.

        Returns:
            The resolved model name
        """
        model_name = self._resolve_model_name(model_name)

        # Check if already loaded
        if (
            model_name in self.models
            and self.model_status.get(model_name) == ModelStatus.READY
        ):
            return model_name

        # Get or create lock for this model
        async with self._global_lock:
            if model_name not in self._locks:
                self._locks[model_name] = asyncio.Lock()

        async with self._locks[model_name]:
            # Double-check after acquiring lock
            if (
                model_name in self.models
                and self.model_status.get(model_name) == ModelStatus.READY
            ):
                return model_name

            self.model_status[model_name] = ModelStatus.LOADING
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()

            try:
                # Check if we need to evict a model
                await self._manage_memory(model_name)

                # Load model and tokenizer
                model, tokenizer = load(model_name)

                # Validate model architecture
                if not hasattr(model, "model"):
                    raise ValueError(
                        "Invalid model architecture: missing 'model' attribute"
                    )

                # Store the model
                self.models[model_name] = (model, tokenizer)

                # Warm up the model
                logger.info(f"Warming up model {model_name}...")
                await self._warmup(model_name)

                self.model_load_times[model_name] = time.time() - start_time
                self.model_status[model_name] = ModelStatus.READY
                logger.info(
                    f"Model {model_name} loaded successfully in {self.model_load_times[model_name]:.2f}s"
                )

                return model_name

            except Exception as e:
                self.model_status[model_name] = ModelStatus.ERROR
                logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
                raise RuntimeError(f"Model loading failed: {e}") from e

    async def _manage_memory(self, new_model: str) -> None:
        """Manage memory by evicting models if necessary"""
        if len(self.models) >= self.max_loaded_models:
            # Find least recently used model (simple strategy)
            # In production, you'd want proper LRU tracking
            models_to_evict = [m for m in self.models.keys() if m != new_model]
            if models_to_evict:
                evict_model = models_to_evict[0]  # Simple: evict first
                logger.info(
                    f"Evicting model {evict_model} to make room for {new_model}"
                )
                del self.models[evict_model]
                self.model_status[evict_model] = ModelStatus.UNLOADED
                # Clear cache entries for this model
                cache_keys_to_remove = [
                    k
                    for k in self._embedding_cache.keys()
                    if k.startswith(f"{evict_model}:")
                ]
                for key in cache_keys_to_remove:
                    del self._embedding_cache[key]

    async def _warmup(self, model_name: str) -> None:
        """Warm up model to compile Metal kernels"""
        try:
            # Don't call generate_embeddings as it will call load_model again
            # Instead, directly process test data
            test_texts = ["warmup", "test"]
            model, tokenizer = self.models[model_name]

            for text in test_texts:
                tokens = tokenizer.encode(text)
                if len(tokens) > self.config.max_text_length:
                    tokens = tokens[: self.config.max_text_length]

                input_ids = mx.array([tokens])
                hidden_states = self._get_hidden_states(input_ids, model)
                pooled = mx.mean(hidden_states, axis=1)
                mx.eval(pooled)  # Force evaluation to compile kernels

        except Exception as e:
            logger.warning(f"Warmup failed for {model_name} (non-critical): {e}")

    def _get_hidden_states(self, input_ids: mx.array, model: Any) -> mx.array:
        """
        Extract hidden states from the model before output projection.

        Args:
            input_ids: Token IDs as MLX array [batch_size, seq_len]

        Returns:
            Hidden states [batch_size, seq_len, hidden_dim]
        """
        # Get token embeddings
        h = model.model.embed_tokens(input_ids)

        # Pass through transformer layers
        for layer in model.model.layers:
            h = layer(h, mask=None, cache=None)

        # Apply final layer normalization
        h = model.model.norm(h)

        return h

    async def generate_embeddings(
        self, texts: List[str], model_name: Optional[str] = None, normalize: bool = True
    ) -> Tuple[np.ndarray, str, int]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of input texts
            model_name: Model to use (name or alias)
            normalize: Whether to L2-normalize embeddings

        Returns:
            Tuple of (embeddings, model_name, embedding_dim)
        """
        # Resolve and load model if needed
        model_name = await self.load_model(model_name)

        if self.model_status.get(model_name) != ModelStatus.READY:
            raise RuntimeError(
                f"Model {model_name} not ready (status: {self.model_status.get(model_name)})"
            )

        if not texts:
            embedding_dim = AVAILABLE_MODELS[model_name]["embedding_dim"]
            return np.array([]), model_name, embedding_dim

        model, tokenizer = self.models[model_name]
        embedding_dim = AVAILABLE_MODELS[model_name]["embedding_dim"]

        embeddings = []

        for text in texts:
            # Check cache if enabled
            cache_key = f"{model_name}:{text}:{normalize}"
            if cache_key in self._embedding_cache:
                embeddings.append(self._embedding_cache[cache_key])
                continue

            # Tokenize text
            tokens = tokenizer.encode(text)

            # Truncate if necessary
            if len(tokens) > self.config.max_text_length:
                logger.warning(
                    f"Truncating text from {len(tokens)} to {self.config.max_text_length} tokens"
                )
                tokens = tokens[: self.config.max_text_length]

            # Convert to MLX array with batch dimension
            input_ids = mx.array([tokens])

            # Get hidden states
            hidden_states = self._get_hidden_states(input_ids, model)

            # Mean pooling across sequence dimension
            pooled = mx.mean(hidden_states, axis=1)  # [1, hidden_dim]

            # Normalize if requested
            if normalize:
                norm = mx.linalg.norm(pooled, axis=1, keepdims=True)
                pooled = pooled / mx.maximum(norm, 1e-9)

            # Force evaluation and convert to numpy
            mx.eval(pooled)
            embedding = np.array(pooled.tolist()[0], dtype=np.float32)

            # Cache the result (with size limit)
            if len(self._embedding_cache) < 1000:  # Simple cache size limit
                self._embedding_cache[cache_key] = embedding

            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32), model_name, embedding_dim

    def get_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current model status and information"""
        if model_name:
            model_name = self._resolve_model_name(model_name)
            return {
                "status": self.model_status.get(model_name, ModelStatus.UNLOADED).value,
                "model_name": model_name,
                "embedding_dim": AVAILABLE_MODELS[model_name]["embedding_dim"],
                "load_time": self.model_load_times.get(model_name),
                "description": AVAILABLE_MODELS[model_name]["description"],
            }

        # Return status for all models
        models_status = {}
        for name in AVAILABLE_MODELS:
            models_status[name] = {
                "status": self.model_status.get(name, ModelStatus.UNLOADED).value,
                "embedding_dim": AVAILABLE_MODELS[name]["embedding_dim"],
                "load_time": self.model_load_times.get(name),
                "aliases": AVAILABLE_MODELS[name]["alias"],
                "description": AVAILABLE_MODELS[name]["description"],
            }

        return {
            "loaded_models": list(self.models.keys()),
            "default_model": self.config.model_name,
            "max_batch_size": self.config.max_batch_size,
            "max_text_length": self.config.max_text_length,
            "cache_size": len(self._embedding_cache),
            "models": models_status,
        }


# Initialize model manager
model_manager = ModelManager(config)


# Legacy Pydantic models removed - now using OpenAI-compatible models from models/


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service health status")
    model_status: str = Field(..., description="Model status")
    model_name: str = Field(..., description="Model name")
    embedding_dim: int = Field(..., description="Embedding dimension")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info(f"Starting Qwen3 Embedding Server v{app.version}")
    logger.info(f"Configuration: {config}")
    logger.info(f"Available models: {list(AVAILABLE_MODELS.keys())}")

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


# Create FastAPI application
app = FastAPI(
    title="Qwen3 Embedding Server",
    description="High-performance text embedding service using MLX on Apple Silicon",
    version="2.0.0",  # Major version bump for OpenAI-compatible API
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware if enabled
if config.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests with timing"""
    start_time = time.time()

    # Process request
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000

        # Log successful requests
        logger.info(
            f"{request.method} {request.url.path} "
            f"- Status: {response.status_code} "
            f"- Time: {process_time:.2f}ms"
        )

        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        return response

    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            f"{request.method} {request.url.path} "
            f"- Error: {e} "
            f"- Time: {process_time:.2f}ms"
        )
        raise


# API Routes
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
            "documentation": "/docs",
        },
    }


@app.post(
    "/v1/embeddings",
    response_model=OpenAIEmbeddingResponse,
    tags=["Embeddings"],
    status_code=status.HTTP_200_OK,
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
            texts, model_name=model_resolved, normalize=True
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

            data.append(
                EmbeddingObject(object="embedding", embedding=emb_list, index=i)
            )

        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Generated {len(data)} embeddings in {processing_time:.2f}ms")

        return OpenAIEmbeddingResponse(
            object="list",
            data=data,
            model=request.model,  # Return original model name for compatibility
            usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=OpenAIErrorResponse(
                error=OpenAIError(message=str(e), type="invalid_request_error")
            ).model_dump(),
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
            ).model_dump(),
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    status_code=status.HTTP_200_OK,
)
async def health_check():
    """
    Health check endpoint.

    Returns the current health status of the service,
    including model readiness and resource usage.
    """
    try:
        # Get memory usage if available
        memory_mb = None
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass

        uptime = (
            time.time() - app.state.start_time
            if hasattr(app.state, "start_time")
            else 0
        )
        model_status = model_manager.get_status()

        # Check default model status
        default_model_status = model_manager.model_status.get(
            config.model_name, ModelStatus.UNLOADED
        )

        return HealthResponse(
            status="healthy"
            if default_model_status == ModelStatus.READY
            else "degraded",
            model_status=default_model_status.value,
            model_name=config.model_name,
            embedding_dim=AVAILABLE_MODELS[config.model_name]["embedding_dim"],
            memory_usage_mb=memory_mb,
            uptime_seconds=uptime,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get detailed metrics and configuration.

    Returns comprehensive information about the service,
    including configuration, model status, and performance metrics.
    """
    return {
        "models": model_manager.get_status(),
        "config": {
            "host": config.host,
            "port": config.port,
            "max_batch_size": config.max_batch_size,
            "max_text_length": config.max_text_length,
            "cors_enabled": config.enable_cors,
        },
        "version": app.version,
    }


@app.get("/models", tags=["Models"])
async def list_models():
    """
    List available models and their status.

    Returns information about all available models,
    their aliases, and current loading status.
    """
    return model_manager.get_status()


# Error handlers
@app.exception_handler(ValueError)
async def openai_error_handler(request: Request, exc: ValueError):
    """Handle validation errors with OpenAI-style response"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=OpenAIErrorResponse(
            error=OpenAIError(message=str(exc), type="invalid_request_error")
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"},
    )


# Main entry point
def main():
    """Run the server"""
    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=os.getenv("DEV_MODE", "false").lower() == "true",
        access_log=True,
    )


if __name__ == "__main__":
    main()
