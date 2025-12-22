"""Pydantic models for API request/response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field(..., description="API version")


class MemoryQueryRequest(BaseModel):
    """Request model for querying memories."""

    query: str = Field(..., description="Query text for memory retrieval")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of memories to retrieve")
    persona_id: str | None = Field(default=None, description="Optional persona ID filter")
    domain: str | None = Field(default=None, description="Optional domain filter")


class MemoryStoreRequest(BaseModel):
    """Request model for storing a memory."""

    content: str = Field(..., description="Memory content to store")
    domain: str = Field(default="conversation", description="Domain category")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    persona_id: str | None = Field(default=None, description="Optional persona ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MemoryResponse(BaseModel):
    """Response model for a single memory."""

    content: str
    domain: str
    timestamp: datetime
    importance: float
    metadata: dict[str, Any]


class MemoryQueryResponse(BaseModel):
    """Response model for memory query results."""

    memories: list[MemoryResponse] = Field(..., description="Retrieved memories")
    count: int = Field(..., description="Number of memories returned")


class MemoryStoreResponse(BaseModel):
    """Response model for memory storage operation."""

    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Operation result message")


class AugmentPromptRequest(BaseModel):
    """Request model for prompt augmentation with memories."""

    prompt: str = Field(..., description="Original prompt to augment")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum memories to include")
    persona_id: str | None = Field(default=None, description="Optional persona ID filter")
    domain: str | None = Field(default=None, description="Optional domain filter")


class AugmentPromptResponse(BaseModel):
    """Response model for augmented prompt."""

    original_prompt: str = Field(..., description="Original prompt")
    augmented_prompt: str = Field(..., description="Prompt with memory context")
    memories_used: int = Field(..., description="Number of memories included")


class GenerateRequest(BaseModel):
    """Request model for provider-agnostic generation."""

    prompt: str = Field(..., description="Prompt text")
    model: str | None = Field(default=None, description="Model name (provider-specific)")
    provider: str = Field(default="ollama", description="Provider type (ollama, openai, custom)")
    use_memory: bool = Field(default=True, description="Whether to augment with memories")
    persona_id: str | None = Field(default=None, description="Optional persona ID")
    stream: bool = Field(default=False, description="Whether to stream the response")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int | None = Field(default=None, ge=1, description="Maximum tokens to generate")


class GenerateResponse(BaseModel):
    """Response model for generation."""

    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used for generation")
    provider: str = Field(..., description="Provider used")
    memories_used: int = Field(default=0, description="Number of memories used in augmentation")
