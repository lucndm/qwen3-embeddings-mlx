"""
OpenAI-compatible Pydantic models for embeddings API.
Matches OpenAI's embedding API schema for drop-in compatibility.
"""

from typing import List, Union, Optional
from pydantic import BaseModel, Field, field_validator


class OpenAIEmbeddingRequest(BaseModel):
    """Request model matching OpenAI's /v1/embeddings endpoint"""

    input: Union[str, List[str]] = Field(
        ..., description="Input text to embed, can be a string or array of strings"
    )
    model: str = Field(
        ..., description="Model to use for embedding (OpenAI name or Qwen name/alias)"
    )
    encoding_format: Optional[str] = Field(
        default="float", description="Format for embeddings (only 'float' supported)"
    )
    dimensions: Optional[int] = Field(
        default=None, description="Truncate embeddings to this dimension (optional)"
    )

    @field_validator("encoding_format", mode="before")
    @classmethod
    def validate_encoding_format(cls, v):
        if v is None:
            return "float"
        if v != "float":
            raise ValueError("Only 'float' encoding_format is supported")
        return v

    @field_validator("input")
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
                    raise ValueError(
                        f"Input at index {i} cannot be empty or whitespace only"
                    )
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
    param: Optional[str] = Field(
        default=None, description="Parameter that caused error"
    )
    code: Optional[str] = Field(default=None, description="Error code")


class OpenAIErrorResponse(BaseModel):
    """OpenAI-style error response wrapper"""

    error: OpenAIError = Field(..., description="Error details")
