"""
Cohere-compatible Pydantic models for rerank API.
Matches Cohere's rerank API schema for drop-in compatibility.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


class RerankRequest(BaseModel):
    """Request model matching Cohere's /v1/rerank endpoint"""

    model: str = Field(
        ..., description="Model to use for reranking (Cohere name or Qwen name/alias)"
    )
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(
        ..., description="List of documents to rerank", min_length=1, max_length=1000
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Number of top results to return. If None, returns all.",
        ge=1,
    )
    max_tokens_per_doc: int = Field(
        default=4096,
        description="Maximum tokens per document. Long docs will be truncated.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v or v.isspace():
            raise ValueError("Query cannot be empty or whitespace only")
        return v

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, v):
        if not v:
            raise ValueError("Documents list cannot be empty")
        for i, doc in enumerate(v):
            if not doc or (isinstance(doc, str) and doc.isspace()):
                raise ValueError(
                    f"Document at index {i} cannot be empty or whitespace only"
                )
        return v

    @field_validator("top_n", mode="before")
    @classmethod
    def validate_top_n(cls, v, info):
        if v is not None:
            docs = info.data.get("documents", [])
            if docs and v > len(docs):
                # Clamp to number of documents
                return len(docs)
        return v


class RerankResult(BaseModel):
    """Single rerank result"""

    index: int = Field(..., description="Original index of the document")
    relevance_score: float = Field(
        ..., description="Relevance score (higher is more relevant)"
    )


class RerankResponse(BaseModel):
    """Response model matching Cohere's rerank response"""

    results: List[RerankResult] = Field(
        ..., description="Ordered list of reranked documents"
    )
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request ID")
