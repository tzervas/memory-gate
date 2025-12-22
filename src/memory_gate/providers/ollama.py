"""Ollama provider implementation.

This module implements the BaseModelProvider interface for Ollama,
enabling MemoryGate to use Ollama as a model backend.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import httpx

from memory_gate.metrics import record_memory_operation

from .base import (
    BaseModelProvider,
    ChatMessage,
    GenerationConfig,
    ModelProviderConfig,
    ProviderConnectionError,
    ProviderGenerationError,
    ProviderResponse,
    ProviderType,
)

logger = logging.getLogger(__name__)


@dataclass
class OllamaProviderConfig(ModelProviderConfig):
    """Configuration for Ollama provider.

    Args:
        base_url: Ollama API base URL
        default_model: Default model for generation
        timeout_seconds: Request timeout
        embedding_model: Model to use for embeddings (if different from default)
    """

    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2"
    timeout_seconds: int = 120
    embedding_model: str | None = None
    provider_type: ProviderType = field(default=ProviderType.OLLAMA, init=False)

    def __post_init__(self) -> None:
        """Ensure provider_type is set correctly."""
        object.__setattr__(self, "provider_type", ProviderType.OLLAMA)


class OllamaProvider(BaseModelProvider):
    """Ollama model provider implementation.

    Provides integration with Ollama for:
    - Text generation (generate, generate_stream)
    - Chat completions (chat, chat_stream)
    - Embeddings (get_embeddings)

    Example:
        ```python
        config = OllamaProviderConfig(
            base_url="http://localhost:11434",
            default_model="llama3.2",
        )
        async with OllamaProvider(config) as provider:
            response = await provider.generate("Hello, world!")
            print(response.content)
        ```
    """

    def __init__(self, config: OllamaProviderConfig | None = None) -> None:
        """Initialize Ollama provider.

        Args:
            config: Provider configuration (uses defaults if not provided)
        """
        self._ollama_config = config or OllamaProviderConfig()
        super().__init__(self._ollama_config)
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        """Get the Ollama base URL."""
        return self._ollama_config.base_url

    async def connect(self) -> None:
        """Establish connection to Ollama."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._config.timeout_seconds),
            )
        self._connected = True
        logger.info("Connected to Ollama at %s", self.base_url)

    async def disconnect(self) -> None:
        """Close connection to Ollama."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info("Disconnected from Ollama")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client, creating if necessary."""
        if self._client is None or self._client.is_closed:
            await self.connect()
        return self._client  # type: ignore[return-value]

    async def health_check(self) -> bool:
        """Check if Ollama is healthy.

        Returns:
            True if Ollama is responding, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            healthy = response.status_code == 200
            record_memory_operation(
                operation_type="ollama_provider_health_check",
                success=healthy,
            )
            return healthy
        except httpx.HTTPError as e:
            logger.warning("Ollama health check failed: %s", e)
            record_memory_operation(
                operation_type="ollama_provider_health_check",
                success=False,
            )
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from Ollama.

        Returns:
            List of model information dictionaries

        Raises:
            ProviderConnectionError: If connection fails
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            record_memory_operation(
                operation_type="ollama_provider_list_models",
                success=True,
            )
            return models
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="ollama_provider_list_models",
                success=False,
            )
            raise ProviderConnectionError(
                f"Failed to list models: {e}",
                provider=ProviderType.OLLAMA,
            ) from e

    def _build_ollama_options(self, config: GenerationConfig) -> dict[str, Any]:
        """Convert GenerationConfig to Ollama options format."""
        options: dict[str, Any] = {}

        if config.temperature != 0.7:  # Only include if not default
            options["temperature"] = config.temperature
        if config.top_p != 1.0:
            options["top_p"] = config.top_p
        if config.top_k is not None:
            options["top_k"] = config.top_k
        if config.seed is not None:
            options["seed"] = config.seed
        if config.stop_sequences:
            options["stop"] = config.stop_sequences

        # Add any extra options
        options.update(config.extra)

        return options

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt
            model: Model to use (defaults to config.default_model)
            config: Generation configuration
            **kwargs: Additional Ollama-specific parameters

        Returns:
            ProviderResponse with generated text

        Raises:
            ProviderConnectionError: If connection fails
            ProviderGenerationError: If generation fails
        """
        model_name = self._get_model(model)
        gen_config = self._build_generation_config(config)

        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": self._build_ollama_options(gen_config),
            **kwargs,
        }

        if gen_config.max_tokens:
            payload["options"]["num_predict"] = gen_config.max_tokens

        start_time = time.time()

        try:
            client = await self._get_client()
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            result = ProviderResponse(
                content=data.get("response", ""),
                model=data.get("model", model_name),
                provider=ProviderType.OLLAMA,
                done=data.get("done", True),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                latency_ms=latency_ms,
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "context": data.get("context"),
                },
            )

            record_memory_operation(
                operation_type="ollama_provider_generate",
                success=True,
            )
            return result

        except httpx.HTTPStatusError as e:
            record_memory_operation(
                operation_type="ollama_provider_generate",
                success=False,
            )
            raise ProviderGenerationError(
                f"Generation failed: {e}",
                provider=ProviderType.OLLAMA,
            ) from e
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="ollama_provider_generate",
                success=False,
            )
            raise ProviderConnectionError(
                f"Connection failed: {e}",
                provider=ProviderType.OLLAMA,
            ) from e

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming output.

        Args:
            prompt: The input prompt
            model: Model to use
            config: Generation configuration
            **kwargs: Additional Ollama-specific parameters

        Yields:
            Text chunks as they're generated

        Raises:
            ProviderConnectionError: If connection fails
            ProviderGenerationError: If generation fails
        """
        model_name = self._get_model(model)
        gen_config = self._build_generation_config(config)

        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": self._build_ollama_options(gen_config),
            **kwargs,
        }

        if gen_config.max_tokens:
            payload["options"]["num_predict"] = gen_config.max_tokens

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
                                yield chunk
                        except json.JSONDecodeError:
                            continue

            record_memory_operation(
                operation_type="ollama_provider_generate_stream",
                success=True,
            )

        except httpx.HTTPStatusError as e:
            record_memory_operation(
                operation_type="ollama_provider_generate_stream",
                success=False,
            )
            raise ProviderGenerationError(
                f"Streaming generation failed: {e}",
                provider=ProviderType.OLLAMA,
            ) from e
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="ollama_provider_generate_stream",
                success=False,
            )
            raise ProviderConnectionError(
                f"Connection failed: {e}",
                provider=ProviderType.OLLAMA,
            ) from e

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Chat with the model.

        Args:
            messages: List of chat messages
            model: Model to use
            config: Generation configuration
            **kwargs: Additional Ollama-specific parameters

        Returns:
            ProviderResponse with assistant's reply

        Raises:
            ProviderConnectionError: If connection fails
            ProviderGenerationError: If chat fails
        """
        model_name = self._get_model(model)
        gen_config = self._build_generation_config(config)

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [msg.to_dict() for msg in messages],
            "stream": False,
            "options": self._build_ollama_options(gen_config),
            **kwargs,
        }

        if gen_config.max_tokens:
            payload["options"]["num_predict"] = gen_config.max_tokens

        start_time = time.time()

        try:
            client = await self._get_client()
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000
            message_content = data.get("message", {}).get("content", "")

            result = ProviderResponse(
                content=message_content,
                model=data.get("model", model_name),
                provider=ProviderType.OLLAMA,
                done=data.get("done", True),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                latency_ms=latency_ms,
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                },
            )

            record_memory_operation(
                operation_type="ollama_provider_chat",
                success=True,
            )
            return result

        except httpx.HTTPStatusError as e:
            record_memory_operation(
                operation_type="ollama_provider_chat",
                success=False,
            )
            raise ProviderGenerationError(
                f"Chat failed: {e}",
                provider=ProviderType.OLLAMA,
            ) from e
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="ollama_provider_chat",
                success=False,
            )
            raise ProviderConnectionError(
                f"Connection failed: {e}",
                provider=ProviderType.OLLAMA,
            ) from e

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Chat with streaming output.

        Args:
            messages: List of chat messages
            model: Model to use
            config: Generation configuration
            **kwargs: Additional Ollama-specific parameters

        Yields:
            Text chunks as they're generated

        Raises:
            ProviderConnectionError: If connection fails
            ProviderGenerationError: If chat fails
        """
        model_name = self._get_model(model)
        gen_config = self._build_generation_config(config)

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [msg.to_dict() for msg in messages],
            "stream": True,
            "options": self._build_ollama_options(gen_config),
            **kwargs,
        }

        if gen_config.max_tokens:
            payload["options"]["num_predict"] = gen_config.max_tokens

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
                                yield chunk
                        except json.JSONDecodeError:
                            continue

            record_memory_operation(
                operation_type="ollama_provider_chat_stream",
                success=True,
            )

        except httpx.HTTPStatusError as e:
            record_memory_operation(
                operation_type="ollama_provider_chat_stream",
                success=False,
            )
            raise ProviderGenerationError(
                f"Streaming chat failed: {e}",
                provider=ProviderType.OLLAMA,
            ) from e
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="ollama_provider_chat_stream",
                success=False,
            )
            raise ProviderConnectionError(
                f"Connection failed: {e}",
                provider=ProviderType.OLLAMA,
            ) from e

    async def get_embeddings(
        self,
        text: str | list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Get embeddings for text.

        Args:
            text: Single text or list of texts
            model: Embedding model to use

        Returns:
            List of embedding vectors

        Raises:
            ProviderConnectionError: If connection fails
            ProviderGenerationError: If embedding fails
        """
        import asyncio

        # Use embedding model if specified, otherwise default
        model_name = model or self._ollama_config.embedding_model or self.default_model

        # Normalize to list
        texts = [text] if isinstance(text, str) else text

        async def get_single_embedding(t: str) -> list[float]:
            """Get embedding for a single text."""
            payload = {
                "model": model_name,
                "prompt": t,
            }
            client = await self._get_client()
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])

        try:
            # Use asyncio.gather for concurrent requests
            embeddings = await asyncio.gather(
                *[get_single_embedding(t) for t in texts]
            )

            record_memory_operation(
                operation_type="ollama_provider_embeddings",
                success=True,
            )
            return list(embeddings)

        except httpx.HTTPStatusError as e:
            record_memory_operation(
                operation_type="ollama_provider_embeddings",
                success=False,
            )
            raise ProviderGenerationError(
                f"Embedding failed: {e}",
                provider=ProviderType.OLLAMA,
            ) from e
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="ollama_provider_embeddings",
                success=False,
            )
            raise ProviderConnectionError(
                f"Connection failed: {e}",
                provider=ProviderType.OLLAMA,
            ) from e
