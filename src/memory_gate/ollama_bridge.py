"""Ollama Memory Bridge: Integration layer between MemoryGate and Ollama.

This module provides a loosely-coupled bridge that augments Ollama prompts
with relevant memories from the MemoryGate system, enabling persistent
learning without modifying model weights.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import LearningContext
from memory_gate.metrics import record_memory_operation

logger = logging.getLogger(__name__)

# Error message constants
ERROR_MSG_OLLAMA_CONNECTION = "Failed to connect to Ollama at {url}: {error}"
ERROR_MSG_OLLAMA_GENERATE = "Ollama generate request failed: {error}"
ERROR_MSG_OLLAMA_CHAT = "Ollama chat request failed: {error}"
ERROR_MSG_OLLAMA_EMBEDDINGS = "Ollama embeddings request failed: {error}"


@dataclass
class OllamaBridgeConfig:
    """Configuration for the Ollama Memory Bridge.

    Args:
        ollama_base_url: Base URL for Ollama API (default: http://localhost:11434)
        default_model: Default model to use for generation (default: llama3.2)
        memory_retrieval_limit: Maximum memories to retrieve for context (default: 5)
        timeout_seconds: HTTP timeout for Ollama requests (default: 120)
        stream_by_default: Whether to stream responses by default (default: True)
        memory_injection_template: Template for injecting memories into prompts
        store_interactions: Whether to store interactions as memories (default: True)
        interaction_domain: Domain to use when storing interactions (default: "conversation")
    """

    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2"
    memory_retrieval_limit: int = 5
    timeout_seconds: int = 120
    stream_by_default: bool = True
    memory_injection_template: str = field(default="")
    store_interactions: bool = True
    interaction_domain: str = "conversation"

    def __post_init__(self) -> None:
        if not self.memory_injection_template:
            self.memory_injection_template = (
                "You have access to the following relevant memories from previous "
                "interactions:\n\n{memories}\n\n"
                "Use these memories to inform your response when relevant, but don't "
                "explicitly mention that you're using memories unless asked.\n\n"
                "Current request: {prompt}"
            )


@dataclass
class OllamaResponse:
    """Structured response from Ollama API.

    Args:
        model: The model used for generation
        response: The generated text response
        done: Whether generation is complete
        context: Token context for conversation continuity (optional)
        total_duration: Total time taken in nanoseconds (optional)
        load_duration: Model load time in nanoseconds (optional)
        prompt_eval_count: Number of tokens in the prompt (optional)
        eval_count: Number of tokens generated (optional)
        eval_duration: Generation time in nanoseconds (optional)
    """

    model: str
    response: str
    done: bool
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


@dataclass
class ChatMessage:
    """A message in a chat conversation.

    Args:
        role: The role of the message sender (system, user, assistant)
        content: The message content
        images: Optional list of base64-encoded images (for multimodal models)
    """

    role: str
    content: str
    images: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API requests."""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            result["images"] = self.images
        return result


class OllamaBridgeError(Exception):
    """Base exception for Ollama bridge errors."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class OllamaConnectionError(OllamaBridgeError):
    """Raised when connection to Ollama fails."""


class OllamaGenerationError(OllamaBridgeError):
    """Raised when Ollama generation fails."""


class OllamaMemoryBridge:
    """Bridge between MemoryGate and Ollama for memory-augmented generation.

    This class provides the core integration layer that:
    1. Retrieves relevant memories from the MemoryGate system
    2. Augments prompts with contextual memory
    3. Sends requests to Ollama
    4. Stores interactions for future learning

    The bridge operates as a loosely-coupled layer that can be easily
    attached or detached without affecting the base Ollama model.
    """

    def __init__(
        self,
        memory_gateway: MemoryGateway[LearningContext],
        config: OllamaBridgeConfig | None = None,
    ) -> None:
        """Initialize the Ollama Memory Bridge.

        Args:
            memory_gateway: The MemoryGateway instance for memory operations
            config: Configuration for the bridge (uses defaults if not provided)
        """
        self.memory_gateway = memory_gateway
        self.config = config or OllamaBridgeConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client for Ollama requests."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.ollama_base_url,
                timeout=httpx.Timeout(self.config.timeout_seconds),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if Ollama is available and responding.

        Returns:
            True if Ollama is healthy, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except httpx.HTTPError as e:
            logger.warning("Ollama health check failed: %s", e)
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from Ollama.

        Returns:
            List of model information dictionaries

        Raises:
            OllamaConnectionError: If connection to Ollama fails
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except httpx.HTTPError as e:
            msg = ERROR_MSG_OLLAMA_CONNECTION.format(
                url=self.config.ollama_base_url, error=str(e)
            )
            raise OllamaConnectionError(msg) from e

    def _format_memories_for_prompt(
        self, memories: list[LearningContext]
    ) -> str:
        """Format retrieved memories for injection into prompts.

        Args:
            memories: List of LearningContext objects to format

        Returns:
            Formatted string representation of memories
        """
        if not memories:
            return ""

        formatted_parts = []
        for i, memory in enumerate(memories, 1):
            age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
            age_str = (
                f"{age_hours:.1f} hours ago"
                if age_hours < 24
                else f"{age_hours / 24:.1f} days ago"
            )

            formatted_parts.append(
                f"[Memory {i}] ({age_str}, importance: {memory.importance:.2f})\n"
                f"{memory.content}"
            )

        return "\n\n".join(formatted_parts)

    async def _retrieve_relevant_memories(
        self,
        query: str,
        domain_filter: str | None = None,
        limit: int | None = None,
    ) -> list[LearningContext]:
        """Retrieve memories relevant to the query.

        Args:
            query: The query to search for relevant memories
            domain_filter: Optional domain to filter memories
            limit: Maximum number of memories to retrieve

        Returns:
            List of relevant LearningContext objects
        """
        retrieval_limit = limit or self.config.memory_retrieval_limit
        try:
            memories = await self.memory_gateway.store.retrieve_context(
                query=query,
                limit=retrieval_limit,
                domain_filter=domain_filter,
            )
            record_memory_operation(
                operation_type="ollama_bridge_memory_retrieval", success=True
            )
            return memories
        except Exception as e:
            logger.warning("Failed to retrieve memories: %s", e)
            record_memory_operation(
                operation_type="ollama_bridge_memory_retrieval", success=False
            )
            return []

    def _build_augmented_prompt(
        self,
        prompt: str,
        memories: list[LearningContext],
    ) -> str:
        """Build a prompt augmented with memory context.

        Args:
            prompt: The original user prompt
            memories: Retrieved memories to inject

        Returns:
            Augmented prompt with memory context
        """
        if not memories:
            return prompt

        formatted_memories = self._format_memories_for_prompt(memories)
        return self.config.memory_injection_template.format(
            memories=formatted_memories,
            prompt=prompt,
        )

    async def _store_interaction(
        self,
        prompt: str,
        response: str,
        domain: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Store an interaction as a memory for future retrieval.

        Args:
            prompt: The user's prompt
            response: The model's response
            domain: Domain for the memory (uses config default if not provided)
            importance: Importance score for the memory
            metadata: Additional metadata for the memory
        """
        if not self.config.store_interactions:
            return

        interaction_domain = domain or self.config.interaction_domain
        interaction_metadata = metadata or {}
        interaction_metadata["source"] = "ollama_bridge"
        interaction_metadata["type"] = "interaction"

        # Create a condensed representation of the interaction
        interaction_content = (
            f"User: {prompt}\n\nAssistant: {response[:500]}"
            + ("..." if len(response) > 500 else "")
        )

        learning_context = LearningContext(
            content=interaction_content,
            domain=interaction_domain,
            timestamp=datetime.now(),
            importance=importance,
            metadata=interaction_metadata,
        )

        try:
            await self.memory_gateway.learn_from_interaction(
                learning_context, feedback=importance
            )
            record_memory_operation(
                operation_type="ollama_bridge_store_interaction", success=True
            )
        except Exception as e:
            logger.warning("Failed to store interaction: %s", e)
            record_memory_operation(
                operation_type="ollama_bridge_store_interaction", success=False
            )

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        use_memory: bool = True,
        domain_filter: str | None = None,
        store_interaction: bool | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> OllamaResponse:
        """Generate a response with memory augmentation.

        This is the primary method for memory-augmented generation. It:
        1. Retrieves relevant memories for the prompt
        2. Augments the prompt with memory context
        3. Sends the augmented prompt to Ollama
        4. Optionally stores the interaction for future learning

        Args:
            prompt: The user's prompt
            model: Model to use (defaults to config.default_model)
            use_memory: Whether to augment with memories (default: True)
            domain_filter: Optional domain filter for memory retrieval
            store_interaction: Whether to store this interaction (defaults to config)
            stream: Whether to stream the response (ignored, use generate_stream)
            **kwargs: Additional parameters passed to Ollama

        Returns:
            OllamaResponse with the generated text

        Raises:
            OllamaConnectionError: If connection to Ollama fails
            OllamaGenerationError: If generation fails
        """
        model_name = model or self.config.default_model
        should_store = (
            store_interaction
            if store_interaction is not None
            else self.config.store_interactions
        )

        # Retrieve and inject memories
        augmented_prompt = prompt
        memories: list[LearningContext] = []
        if use_memory:
            memories = await self._retrieve_relevant_memories(
                query=prompt, domain_filter=domain_filter
            )
            augmented_prompt = self._build_augmented_prompt(prompt, memories)
            logger.debug(
                "Augmented prompt with %d memories for generation", len(memories)
            )

        # Build request payload
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": augmented_prompt,
            "stream": False,
            **kwargs,
        }

        try:
            client = await self._get_client()
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

            result = OllamaResponse(
                model=data.get("model", model_name),
                response=data.get("response", ""),
                done=data.get("done", True),
                context=data.get("context"),
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_eval_count=data.get("prompt_eval_count"),
                eval_count=data.get("eval_count"),
                eval_duration=data.get("eval_duration"),
            )

            # Store interaction asynchronously
            if should_store and result.response:
                asyncio.create_task(
                    self._store_interaction(
                        prompt=prompt,
                        response=result.response,
                        domain=domain_filter,
                        metadata={"model": model_name, "memories_used": str(len(memories))},
                    )
                )

            record_memory_operation(
                operation_type="ollama_bridge_generate", success=True
            )
            return result

        except httpx.HTTPStatusError as e:
            msg = ERROR_MSG_OLLAMA_GENERATE.format(error=str(e))
            record_memory_operation(
                operation_type="ollama_bridge_generate", success=False
            )
            raise OllamaGenerationError(msg) from e
        except httpx.HTTPError as e:
            msg = ERROR_MSG_OLLAMA_CONNECTION.format(
                url=self.config.ollama_base_url, error=str(e)
            )
            record_memory_operation(
                operation_type="ollama_bridge_generate", success=False
            )
            raise OllamaConnectionError(msg) from e

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        use_memory: bool = True,
        domain_filter: str | None = None,
        store_interaction: bool | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response with memory augmentation.

        Similar to generate(), but yields response tokens as they're generated.

        Args:
            prompt: The user's prompt
            model: Model to use (defaults to config.default_model)
            use_memory: Whether to augment with memories (default: True)
            domain_filter: Optional domain filter for memory retrieval
            store_interaction: Whether to store this interaction (defaults to config)
            **kwargs: Additional parameters passed to Ollama

        Yields:
            Response text chunks as they're generated

        Raises:
            OllamaConnectionError: If connection to Ollama fails
            OllamaGenerationError: If generation fails
        """
        model_name = model or self.config.default_model
        should_store = (
            store_interaction
            if store_interaction is not None
            else self.config.store_interactions
        )

        # Retrieve and inject memories
        augmented_prompt = prompt
        memories: list[LearningContext] = []
        if use_memory:
            memories = await self._retrieve_relevant_memories(
                query=prompt, domain_filter=domain_filter
            )
            augmented_prompt = self._build_augmented_prompt(prompt, memories)
            logger.debug(
                "Augmented prompt with %d memories for streaming", len(memories)
            )

        # Build request payload
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": augmented_prompt,
            "stream": True,
            **kwargs,
        }

        full_response_parts: list[str] = []

        try:
            client = await self._get_client()
            async with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                full_response_parts.append(chunk)
                                yield chunk
                        except json.JSONDecodeError:
                            continue

            # Store interaction after streaming completes
            full_response = "".join(full_response_parts)
            if should_store and full_response:
                asyncio.create_task(
                    self._store_interaction(
                        prompt=prompt,
                        response=full_response,
                        domain=domain_filter,
                        metadata={"model": model_name, "memories_used": str(len(memories))},
                    )
                )

            record_memory_operation(
                operation_type="ollama_bridge_generate_stream", success=True
            )

        except httpx.HTTPStatusError as e:
            msg = ERROR_MSG_OLLAMA_GENERATE.format(error=str(e))
            record_memory_operation(
                operation_type="ollama_bridge_generate_stream", success=False
            )
            raise OllamaGenerationError(msg) from e
        except httpx.HTTPError as e:
            msg = ERROR_MSG_OLLAMA_CONNECTION.format(
                url=self.config.ollama_base_url, error=str(e)
            )
            record_memory_operation(
                operation_type="ollama_bridge_generate_stream", success=False
            )
            raise OllamaConnectionError(msg) from e

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        use_memory: bool = True,
        domain_filter: str | None = None,
        store_interaction: bool | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> OllamaResponse | AsyncGenerator[str, None]:
        """Chat with memory augmentation.

        Supports multi-turn conversations with memory context injection.
        Memory is retrieved based on the last user message.

        Args:
            messages: List of ChatMessage objects representing the conversation
            model: Model to use (defaults to config.default_model)
            use_memory: Whether to augment with memories (default: True)
            domain_filter: Optional domain filter for memory retrieval
            store_interaction: Whether to store this interaction (defaults to config)
            stream: Whether to stream the response
            **kwargs: Additional parameters passed to Ollama

        Returns:
            OllamaResponse if stream=False, AsyncGenerator if stream=True

        Raises:
            OllamaConnectionError: If connection to Ollama fails
            OllamaGenerationError: If chat fails
        """
        model_name = model or self.config.default_model
        should_store = (
            store_interaction
            if store_interaction is not None
            else self.config.store_interactions
        )

        # Find the last user message for memory retrieval
        last_user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_message = msg.content
                break

        # Retrieve memories and create system message
        augmented_messages = [msg.to_dict() for msg in messages]
        memories: list[LearningContext] = []

        if use_memory and last_user_message:
            memories = await self._retrieve_relevant_memories(
                query=last_user_message, domain_filter=domain_filter
            )

            if memories:
                formatted_memories = self._format_memories_for_prompt(memories)
                memory_system_message = {
                    "role": "system",
                    "content": (
                        "You have access to the following relevant memories from "
                        "previous interactions. Use them to inform your response "
                        "when relevant:\n\n" + formatted_memories
                    ),
                }
                # Insert memory context as system message at the beginning
                augmented_messages.insert(0, memory_system_message)
                logger.debug(
                    "Augmented chat with %d memories", len(memories)
                )

        # Build request payload
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": augmented_messages,
            "stream": stream,
            **kwargs,
        }

        if stream:
            return self._chat_stream(
                payload=payload,
                last_user_message=last_user_message,
                model_name=model_name,
                memories=memories,
                should_store=should_store,
                domain_filter=domain_filter,
            )
        else:
            return await self._chat_sync(
                payload=payload,
                last_user_message=last_user_message,
                model_name=model_name,
                memories=memories,
                should_store=should_store,
                domain_filter=domain_filter,
            )

    async def _chat_sync(
        self,
        payload: dict[str, Any],
        last_user_message: str,
        model_name: str,
        memories: list[LearningContext],
        should_store: bool,
        domain_filter: str | None,
    ) -> OllamaResponse:
        """Handle synchronous chat request."""
        try:
            client = await self._get_client()
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            message_content = data.get("message", {}).get("content", "")

            result = OllamaResponse(
                model=data.get("model", model_name),
                response=message_content,
                done=data.get("done", True),
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_eval_count=data.get("prompt_eval_count"),
                eval_count=data.get("eval_count"),
                eval_duration=data.get("eval_duration"),
            )

            # Store interaction asynchronously
            if should_store and message_content and last_user_message:
                asyncio.create_task(
                    self._store_interaction(
                        prompt=last_user_message,
                        response=message_content,
                        domain=domain_filter,
                        metadata={"model": model_name, "memories_used": str(len(memories))},
                    )
                )

            record_memory_operation(
                operation_type="ollama_bridge_chat", success=True
            )
            return result

        except httpx.HTTPStatusError as e:
            msg = ERROR_MSG_OLLAMA_CHAT.format(error=str(e))
            record_memory_operation(
                operation_type="ollama_bridge_chat", success=False
            )
            raise OllamaGenerationError(msg) from e
        except httpx.HTTPError as e:
            msg = ERROR_MSG_OLLAMA_CONNECTION.format(
                url=self.config.ollama_base_url, error=str(e)
            )
            record_memory_operation(
                operation_type="ollama_bridge_chat", success=False
            )
            raise OllamaConnectionError(msg) from e

    async def _chat_stream(
        self,
        payload: dict[str, Any],
        last_user_message: str,
        model_name: str,
        memories: list[LearningContext],
        should_store: bool,
        domain_filter: str | None,
    ) -> AsyncGenerator[str, None]:
        """Handle streaming chat request."""
        full_response_parts: list[str] = []

        try:
            client = await self._get_client()
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                full_response_parts.append(chunk)
                                yield chunk
                        except json.JSONDecodeError:
                            continue

            # Store interaction after streaming completes
            full_response = "".join(full_response_parts)
            if should_store and full_response and last_user_message:
                asyncio.create_task(
                    self._store_interaction(
                        prompt=last_user_message,
                        response=full_response,
                        domain=domain_filter,
                        metadata={"model": model_name, "memories_used": str(len(memories))},
                    )
                )

            record_memory_operation(
                operation_type="ollama_bridge_chat_stream", success=True
            )

        except httpx.HTTPStatusError as e:
            msg = ERROR_MSG_OLLAMA_CHAT.format(error=str(e))
            record_memory_operation(
                operation_type="ollama_bridge_chat_stream", success=False
            )
            raise OllamaGenerationError(msg) from e
        except httpx.HTTPError as e:
            msg = ERROR_MSG_OLLAMA_CONNECTION.format(
                url=self.config.ollama_base_url, error=str(e)
            )
            record_memory_operation(
                operation_type="ollama_bridge_chat_stream", success=False
            )
            raise OllamaConnectionError(msg) from e

    async def generate_without_memory(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> OllamaResponse:
        """Generate a response without memory augmentation.

        Useful for baseline comparisons or when memory context is not desired.

        Args:
            prompt: The user's prompt
            model: Model to use (defaults to config.default_model)
            **kwargs: Additional parameters passed to Ollama

        Returns:
            OllamaResponse with the generated text
        """
        return await self.generate(
            prompt=prompt,
            model=model,
            use_memory=False,
            store_interaction=False,
            **kwargs,
        )

    async def get_embeddings(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """Get embeddings for text using Ollama.

        Args:
            text: Text to generate embeddings for
            model: Model to use for embeddings

        Returns:
            List of embedding floats

        Raises:
            OllamaConnectionError: If connection to Ollama fails
            OllamaGenerationError: If embedding generation fails
        """
        model_name = model or self.config.default_model

        payload = {
            "model": model_name,
            "prompt": text,
        }

        try:
            client = await self._get_client()
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding", [])
            record_memory_operation(
                operation_type="ollama_bridge_embeddings", success=True
            )
            return embedding

        except httpx.HTTPStatusError as e:
            msg = ERROR_MSG_OLLAMA_EMBEDDINGS.format(error=str(e))
            record_memory_operation(
                operation_type="ollama_bridge_embeddings", success=False
            )
            raise OllamaGenerationError(msg) from e
        except httpx.HTTPError as e:
            msg = ERROR_MSG_OLLAMA_CONNECTION.format(
                url=self.config.ollama_base_url, error=str(e)
            )
            record_memory_operation(
                operation_type="ollama_bridge_embeddings", success=False
            )
            raise OllamaConnectionError(msg) from e


# Convenience function for quick setup
async def create_memory_bridge(
    memory_gateway: MemoryGateway[LearningContext],
    ollama_url: str = "http://localhost:11434",
    default_model: str = "llama3.2",
) -> OllamaMemoryBridge:
    """Create and initialize an OllamaMemoryBridge.

    Args:
        memory_gateway: The MemoryGateway instance
        ollama_url: Ollama API base URL
        default_model: Default model for generation

    Returns:
        Configured OllamaMemoryBridge instance
    """
    config = OllamaBridgeConfig(
        ollama_base_url=ollama_url,
        default_model=default_model,
    )
    bridge = OllamaMemoryBridge(memory_gateway=memory_gateway, config=config)

    # Verify connection
    if not await bridge.health_check():
        logger.warning(
            "Ollama is not responding at %s. Bridge created but may not function.",
            ollama_url,
        )

    return bridge
