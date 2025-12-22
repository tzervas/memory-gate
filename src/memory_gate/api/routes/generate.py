"""Generation endpoints using provider framework."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from memory_gate.api.dependencies import get_memory_gateway
from memory_gate.api.models import GenerateRequest, GenerateResponse
from memory_gate.memory_gateway import MemoryGateway
from memory_gate.providers import ProviderType, get_provider
from memory_gate.providers.base import GenerationConfig

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    gateway: MemoryGateway = Depends(get_memory_gateway),
) -> GenerateResponse | StreamingResponse:
    """Generate a response using specified provider with optional memory augmentation.

    Args:
        request: Generation parameters including prompt, provider, and options.
        gateway: Memory gateway instance (injected).

    Returns:
        GenerateResponse: Generated content or StreamingResponse if streaming.
    """
    try:
        # Determine provider type
        provider_type = ProviderType.OLLAMA
        if request.provider.lower() == "openai":
            provider_type = ProviderType.OPENAI
        elif request.provider.lower() == "custom":
            provider_type = ProviderType.CUSTOM

        # Prepare the prompt (augment with memories if requested)
        prompt = request.prompt
        memories_count = 0

        if request.use_memory:
            # Retrieve relevant memories
            memories = await gateway.store.retrieve_context(
                query=request.prompt,
                limit=5,
            )
            memories_count = len(memories)

            # Augment prompt with memories
            if memories:
                memory_context = "\n".join(
                    f"- {memory.content} (importance: {memory.importance:.2f})"
                    for memory in memories
                )
                prompt = f"""Based on relevant memories:
{memory_context}

Current request: {request.prompt}"""

        # Build generation config
        config = GenerationConfig(
            temperature=request.temperature if request.temperature is not None else 0.7,
            max_tokens=request.max_tokens,
        )

        # Generate using provider
        async with get_provider(provider_type) as provider:
            response = await provider.generate(
                prompt=prompt,
                model=request.model,
                config=config,
            )

            return GenerateResponse(
                content=response.content,
                model=response.model or request.model or "unknown",
                provider=request.provider,
                memories_used=memories_count,
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {e!s}",
        ) from e
