"""
Model mapping utilities for OpenAI-to-Qwen compatibility.
Maps OpenAI model names to Qwen equivalents and provides helper functions.
"""

from typing import List, Any


# Mapping from OpenAI model names to Qwen aliases
MODEL_MAPPING = {
    # OpenAI embedding models -> Qwen equivalents
    "text-embedding-3-small": "small",
    "text-embedding-3-medium": "medium",
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
