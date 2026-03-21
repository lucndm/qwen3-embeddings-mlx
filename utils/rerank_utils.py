"""
Rerank utilities for Cohere-compatible reranking API.
Maps Cohere model names to Qwen reranker models and provides helper functions.
"""

from typing import List, Dict, Any


# Available rerank models configuration
AVAILABLE_RERANK_MODELS = {
    "samdotci/Qwen3-Reranker-0.6B-mlx-4Bit": {
        "alias": ["small", "0.6b", "default"],
        "description": "Small 0.6B reranker model, fast and efficient",
    },
    "mlx-community/Qwen3-Reranker-4B-mxfp8": {
        "alias": ["large", "4b"],
        "description": "Large 4B reranker model, higher accuracy",
    },
}

# Build alias mapping for rerank models
RERANK_MODEL_ALIASES: Dict[str, str] = {}
for model_name, config in AVAILABLE_RERANK_MODELS.items():
    for alias in config.get("alias", []):
        RERANK_MODEL_ALIASES[alias.lower()] = model_name

# Mapping from Cohere model names to Qwen aliases
RERANK_MODEL_MAPPING = {
    "rerank-v3.5": "small",
    "rerank-v4.0": "small",
    "rerank-english-v3.0": "small",
    "rerank-multilingual-v3.0": "small",
    "rerank-v4.0-pro": "large",
    "rerank-v3.5-pro": "large",
}


def resolve_rerank_model(model: str) -> str:
    """
    Resolve rerank model name to Qwen model name or alias.

    Args:
        model: Model name (Cohere name, Qwen alias, or full Qwen name)

    Returns:
        Resolved model name/alias for Qwen reranker
    """
    # Check if it's a Cohere model name
    if model in RERANK_MODEL_MAPPING:
        return RERANK_MODEL_MAPPING[model]

    # Check if it's an alias
    model_lower = model.lower()
    if model_lower in RERANK_MODEL_ALIASES:
        return RERANK_MODEL_ALIASES[model_lower]

    # Check if it's a valid full model name
    if model in AVAILABLE_RERANK_MODELS:
        return model

    # Pass through as-is (will be validated by model manager)
    return model


def get_rerank_model_full_name(model_identifier: str) -> str:
    """
    Get the full model name from an identifier (alias or name).

    Args:
        model_identifier: Model alias or name

    Returns:
        Full model name

    Raises:
        ValueError: If model not found
    """
    # Check if it's an alias
    model_lower = model_identifier.lower()
    if model_lower in RERANK_MODEL_ALIASES:
        return RERANK_MODEL_ALIASES[model_lower]

    # Check if it's a full name
    if model_identifier in AVAILABLE_RERANK_MODELS:
        return model_identifier

    raise ValueError(
        f"Unknown rerank model: {model_identifier}. "
        f"Available: {list(AVAILABLE_RERANK_MODELS.keys())}"
    )
