"""Memory operation endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from memory_gate.api.dependencies import get_memory_gateway
from memory_gate.api.models import (
    AugmentPromptRequest,
    AugmentPromptResponse,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
)
from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import LearningContext

router = APIRouter()


@router.post("/memory/query", response_model=MemoryQueryResponse)
async def query_memories(
    request: MemoryQueryRequest,
    gateway: MemoryGateway = Depends(get_memory_gateway),
) -> MemoryQueryResponse:
    """Query memories based on text similarity.

    Args:
        request: Query parameters including text and filters.
        gateway: Memory gateway instance (injected).

    Returns:
        MemoryQueryResponse: Retrieved memories.
    """
    try:
        memories = await gateway.store.retrieve_context(
            query=request.query,
            limit=request.limit,
        )

        # Convert to response models
        memory_responses = [
            MemoryResponse(
                content=memory.content,
                domain=memory.domain,
                timestamp=memory.timestamp,
                importance=memory.importance,
                metadata=memory.metadata,
            )
            for memory in memories
        ]

        return MemoryQueryResponse(
            memories=memory_responses,
            count=len(memory_responses),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory query failed: {e!s}") from e


@router.post("/memory/store", response_model=MemoryStoreResponse)
async def store_memory(
    request: MemoryStoreRequest,
    gateway: MemoryGateway = Depends(get_memory_gateway),
) -> MemoryStoreResponse:
    """Store a new memory.

    Args:
        request: Memory data to store.
        gateway: Memory gateway instance (injected).

    Returns:
        MemoryStoreResponse: Operation result.
    """
    try:
        # Create learning context
        context = LearningContext(
            content=request.content,
            domain=request.domain,
            timestamp=datetime.now(),
            importance=request.importance,
            metadata=request.metadata,
        )

        # Store in memory gateway
        await gateway.store.store_context(context)

        return MemoryStoreResponse(
            success=True,
            message="Memory stored successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory storage failed: {e!s}") from e


@router.post("/memory/augment", response_model=AugmentPromptResponse)
async def augment_prompt(
    request: AugmentPromptRequest,
    gateway: MemoryGateway = Depends(get_memory_gateway),
) -> AugmentPromptResponse:
    """Augment a prompt with relevant memories.

    Args:
        request: Prompt and augmentation parameters.
        gateway: Memory gateway instance (injected).

    Returns:
        AugmentPromptResponse: Augmented prompt with memory context.
    """
    try:
        # Retrieve relevant memories
        memories = await gateway.store.retrieve_context(
            query=request.prompt,
            limit=request.limit,
        )

        # Build augmented prompt
        if memories:
            memory_context = "\n".join(
                f"- {memory.content} (importance: {memory.importance:.2f})"
                for memory in memories
            )
            augmented = f"""Based on relevant memories:
{memory_context}

Current request: {request.prompt}"""
        else:
            augmented = request.prompt

        return AugmentPromptResponse(
            original_prompt=request.prompt,
            augmented_prompt=augmented,
            memories_used=len(memories),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prompt augmentation failed: {e!s}",
        ) from e
