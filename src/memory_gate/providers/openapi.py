"""Universal OpenAPI-compliant provider.

This module provides a configurable provider that can connect to any
OpenAPI-compliant LLM API, making it easy for users to integrate custom
backends or fork the project for their own provider implementations.

The provider supports:
- Configurable endpoints for generate, chat, and embeddings
- Custom header injection (for auth, API keys, etc.)
- Response field mapping for different API response formats
- Request body templating for different API request formats
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
class EndpointConfig:
    """Configuration for a single API endpoint.

    Args:
        path: URL path for the endpoint (e.g., "/v1/completions")
        method: HTTP method (GET, POST, etc.)
        request_template: Template for request body with placeholders
        response_content_path: JSON path to extract content from response
        response_usage_path: JSON path to extract usage stats (optional)
        stream_content_path: JSON path for streaming content extraction
    """

    path: str
    method: str = "POST"
    request_template: dict[str, Any] = field(default_factory=dict)
    response_content_path: str = "choices.0.message.content"
    response_usage_path: str | None = "usage"
    stream_content_path: str = "choices.0.delta.content"


@dataclass
class OpenAPIProviderConfig(ModelProviderConfig):
    """Configuration for OpenAPI-compliant provider.

    This configuration allows users to define how to communicate with
    any OpenAPI-compliant LLM API by specifying:
    - Base URL and authentication
    - Endpoint paths and methods
    - Request/response field mappings

    Args:
        base_url: Base URL of the API
        default_model: Default model identifier
        api_key: API key for authentication (optional)
        api_key_header: Header name for API key (default: "Authorization")
        api_key_prefix: Prefix for API key value (default: "Bearer ")
        custom_headers: Additional headers to include in requests
        generate_endpoint: Configuration for text generation endpoint
        chat_endpoint: Configuration for chat endpoint
        embeddings_endpoint: Configuration for embeddings endpoint
        timeout_seconds: Request timeout
    """

    base_url: str = "http://localhost:8000"
    default_model: str = "default"
    api_key: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    custom_headers: dict[str, str] = field(default_factory=dict)

    # Endpoint configurations with OpenAI-compatible defaults
    generate_endpoint: EndpointConfig = field(
        default_factory=lambda: EndpointConfig(
            path="/v1/completions",
            method="POST",
            request_template={
                "model": "{model}",
                "prompt": "{prompt}",
                "max_tokens": "{max_tokens}",
                "temperature": "{temperature}",
                "stream": "{stream}",
            },
            response_content_path="choices.0.text",
            stream_content_path="choices.0.text",
        )
    )

    chat_endpoint: EndpointConfig = field(
        default_factory=lambda: EndpointConfig(
            path="/v1/chat/completions",
            method="POST",
            request_template={
                "model": "{model}",
                "messages": "{messages}",
                "max_tokens": "{max_tokens}",
                "temperature": "{temperature}",
                "stream": "{stream}",
            },
            response_content_path="choices.0.message.content",
            stream_content_path="choices.0.delta.content",
        )
    )

    embeddings_endpoint: EndpointConfig = field(
        default_factory=lambda: EndpointConfig(
            path="/v1/embeddings",
            method="POST",
            request_template={
                "model": "{model}",
                "input": "{input}",
            },
            response_content_path="data.0.embedding",
        )
    )

    timeout_seconds: int = 120
    provider_type: ProviderType = field(default=ProviderType.CUSTOM, init=False)

    def __post_init__(self) -> None:
        """Ensure provider_type is set correctly."""
        object.__setattr__(self, "provider_type", ProviderType.CUSTOM)


class OpenAPIProvider(BaseModelProvider):
    """Universal OpenAPI-compliant model provider.

    This provider can connect to any API that follows OpenAPI/OpenAI-style
    conventions, with configurable endpoint paths and response mappings.

    It's designed to be:
    - Easy to configure for different APIs
    - A base for custom provider implementations
    - Compatible with OpenAI-style APIs out of the box

    Example:
        ```python
        # Connect to an OpenAI-compatible API
        config = OpenAPIProviderConfig(
            base_url="https://api.example.com",
            api_key="your-api-key",
            default_model="gpt-4",
        )
        async with OpenAPIProvider(config) as provider:
            response = await provider.chat([
                ChatMessage.user("Hello!")
            ])
            print(response.content)

        # Connect to a custom API with different endpoints
        custom_config = OpenAPIProviderConfig(
            base_url="https://custom-api.example.com",
            default_model="custom-model",
            chat_endpoint=EndpointConfig(
                path="/api/v2/generate",
                response_content_path="result.text",
            ),
        )
        ```
    """

    def __init__(self, config: OpenAPIProviderConfig | None = None) -> None:
        """Initialize OpenAPI provider.

        Args:
            config: Provider configuration (uses defaults if not provided)
        """
        self._openapi_config = config or OpenAPIProviderConfig()
        super().__init__(self._openapi_config)
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        """Get the API base URL."""
        return self._openapi_config.base_url

    def _build_headers(self) -> dict[str, str]:
        """Build request headers including authentication."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **self._openapi_config.custom_headers,
        }

        if self._openapi_config.api_key:
            headers[self._openapi_config.api_key_header] = (
                f"{self._openapi_config.api_key_prefix}{self._openapi_config.api_key}"
            )

        return headers

    async def connect(self) -> None:
        """Establish connection to the API."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._config.timeout_seconds),
                headers=self._build_headers(),
            )
        self._connected = True
        logger.info("Connected to OpenAPI provider at %s", self.base_url)

    async def disconnect(self) -> None:
        """Close connection to the API."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info("Disconnected from OpenAPI provider")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client, creating if necessary."""
        if self._client is None or self._client.is_closed:
            await self.connect()
        return self._client  # type: ignore[return-value]

    def _extract_from_path(self, data: dict[str, Any], path: str) -> Any:
        """Extract a value from nested dict using dot notation path.

        Args:
            data: Dictionary to extract from
            path: Dot-separated path (e.g., "choices.0.message.content")

        Returns:
            Extracted value or None if not found
        """
        parts = path.split(".")
        current = data

        for part in parts:
            if current is None:
                return None

            # Handle array index
            if part.isdigit():
                idx = int(part)
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None
            # Handle dict key
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current

    def _build_request_body(
        self,
        template: dict[str, Any],
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Build request body from template and values.

        Args:
            template: Request template with placeholders
            values: Values to substitute

        Returns:
            Built request body
        """
        result: dict[str, Any] = {}

        for key, value in template.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # It's a placeholder
                placeholder = value[1:-1]
                if placeholder in values and values[placeholder] is not None:
                    result[key] = values[placeholder]
            elif isinstance(value, dict):
                # Recurse for nested dicts
                result[key] = self._build_request_body(value, values)
            else:
                result[key] = value

        return result

    async def health_check(self) -> bool:
        """Check if the API is healthy.

        Returns:
            True if API is responding, False otherwise
        """
        try:
            client = await self._get_client()
            # Try to hit the models endpoint or a simple GET
            response = await client.get("/v1/models")
            healthy = response.status_code in (200, 401, 403)  # Auth errors still mean it's up
            record_memory_operation(
                operation_type="openapi_provider_health_check",
                success=healthy,
            )
            return healthy
        except httpx.HTTPError as e:
            logger.warning("OpenAPI provider health check failed: %s", e)
            record_memory_operation(
                operation_type="openapi_provider_health_check",
                success=False,
            )
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from the API.

        Returns:
            List of model information dictionaries
        """
        try:
            client = await self._get_client()
            response = await client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            record_memory_operation(
                operation_type="openapi_provider_list_models",
                success=True,
            )
            return models
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="openapi_provider_list_models",
                success=False,
            )
            raise ProviderConnectionError(
                f"Failed to list models: {e}",
                provider=ProviderType.CUSTOM,
            ) from e

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
            model: Model to use
            config: Generation configuration
            **kwargs: Additional parameters

        Returns:
            ProviderResponse with generated text
        """
        model_name = self._get_model(model)
        gen_config = self._build_generation_config(config)
        endpoint = self._openapi_config.generate_endpoint

        values = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": gen_config.max_tokens,
            "temperature": gen_config.temperature,
            "stream": False,
            **kwargs,
        }

        body = self._build_request_body(endpoint.request_template, values)
        start_time = time.time()

        try:
            client = await self._get_client()
            response = await client.request(
                endpoint.method,
                endpoint.path,
                json=body,
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000
            content = self._extract_from_path(data, endpoint.response_content_path) or ""

            usage = None
            if endpoint.response_usage_path:
                usage_data = self._extract_from_path(data, endpoint.response_usage_path)
                if usage_data:
                    usage = {
                        "prompt_tokens": usage_data.get("prompt_tokens", 0),
                        "completion_tokens": usage_data.get("completion_tokens", 0),
                    }

            result = ProviderResponse(
                content=str(content),
                model=model_name,
                provider=ProviderType.CUSTOM,
                done=True,
                usage=usage,
                latency_ms=latency_ms,
                metadata={"raw_response": data},
            )

            record_memory_operation(
                operation_type="openapi_provider_generate",
                success=True,
            )
            return result

        except httpx.HTTPStatusError as e:
            record_memory_operation(
                operation_type="openapi_provider_generate",
                success=False,
            )
            raise ProviderGenerationError(
                f"Generation failed: {e}",
                provider=ProviderType.CUSTOM,
            ) from e
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="openapi_provider_generate",
                success=False,
            )
            raise ProviderConnectionError(
                f"Connection failed: {e}",
                provider=ProviderType.CUSTOM,
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
            **kwargs: Additional parameters

        Yields:
            Text chunks as they're generated
        """
        model_name = self._get_model(model)
        gen_config = self._build_generation_config(config)
        endpoint = self._openapi_config.generate_endpoint

        values = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": gen_config.max_tokens,
            "temperature": gen_config.temperature,
            "stream": True,
            **kwargs,
        }

        body = self._build_request_body(endpoint.request_template, values)

        try:
            client = await self._get_client()
            async with client.stream(
                endpoint.method,
                endpoint.path,
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            chunk = self._extract_from_path(
                                data, endpoint.stream_content_path
                            )
                            if chunk:
                                yield str(chunk)
                        except json.JSONDecodeError:
                            continue

            record_memory_operation(
                operation_type="openapi_provider_generate_stream",
                success=True,
            )

        except httpx.HTTPStatusError as e:
            record_memory_operation(
                operation_type="openapi_provider_generate_stream",
                success=False,
            )
            raise ProviderGenerationError(
                f"Streaming generation failed: {e}",
                provider=ProviderType.CUSTOM,
            ) from e
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="openapi_provider_generate_stream",
                success=False,
            )
            raise ProviderConnectionError(
                f"Connection failed: {e}",
                provider=ProviderType.CUSTOM,
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
            **kwargs: Additional parameters

        Returns:
            ProviderResponse with assistant's reply
        """
        model_name = self._get_model(model)
        gen_config = self._build_generation_config(config)
        endpoint = self._openapi_config.chat_endpoint

        values = {
            "model": model_name,
            "messages": [msg.to_dict() for msg in messages],
            "max_tokens": gen_config.max_tokens,
            "temperature": gen_config.temperature,
            "stream": False,
            **kwargs,
        }

        body = self._build_request_body(endpoint.request_template, values)
        start_time = time.time()

        try:
            client = await self._get_client()
            response = await client.request(
                endpoint.method,
                endpoint.path,
                json=body,
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000
            content = self._extract_from_path(data, endpoint.response_content_path) or ""

            usage = None
            if endpoint.response_usage_path:
                usage_data = self._extract_from_path(data, endpoint.response_usage_path)
                if usage_data:
                    usage = {
                        "prompt_tokens": usage_data.get("prompt_tokens", 0),
                        "completion_tokens": usage_data.get("completion_tokens", 0),
                    }

            result = ProviderResponse(
                content=str(content),
                model=model_name,
                provider=ProviderType.CUSTOM,
                done=True,
                usage=usage,
                latency_ms=latency_ms,
                metadata={"raw_response": data},
            )

            record_memory_operation(
                operation_type="openapi_provider_chat",
                success=True,
            )
            return result

        except httpx.HTTPStatusError as e:
            record_memory_operation(
                operation_type="openapi_provider_chat",
                success=False,
            )
            raise ProviderGenerationError(
                f"Chat failed: {e}",
                provider=ProviderType.CUSTOM,
            ) from e
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="openapi_provider_chat",
                success=False,
            )
            raise ProviderConnectionError(
                f"Connection failed: {e}",
                provider=ProviderType.CUSTOM,
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
            **kwargs: Additional parameters

        Yields:
            Text chunks as they're generated
        """
        model_name = self._get_model(model)
        gen_config = self._build_generation_config(config)
        endpoint = self._openapi_config.chat_endpoint

        values = {
            "model": model_name,
            "messages": [msg.to_dict() for msg in messages],
            "max_tokens": gen_config.max_tokens,
            "temperature": gen_config.temperature,
            "stream": True,
            **kwargs,
        }

        body = self._build_request_body(endpoint.request_template, values)

        try:
            client = await self._get_client()
            async with client.stream(
                endpoint.method,
                endpoint.path,
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            chunk = self._extract_from_path(
                                data, endpoint.stream_content_path
                            )
                            if chunk:
                                yield str(chunk)
                        except json.JSONDecodeError:
                            continue

            record_memory_operation(
                operation_type="openapi_provider_chat_stream",
                success=True,
            )

        except httpx.HTTPStatusError as e:
            record_memory_operation(
                operation_type="openapi_provider_chat_stream",
                success=False,
            )
            raise ProviderGenerationError(
                f"Streaming chat failed: {e}",
                provider=ProviderType.CUSTOM,
            ) from e
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="openapi_provider_chat_stream",
                success=False,
            )
            raise ProviderConnectionError(
                f"Connection failed: {e}",
                provider=ProviderType.CUSTOM,
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
        """
        model_name = self._get_model(model)
        endpoint = self._openapi_config.embeddings_endpoint

        # Normalize to list
        texts = [text] if isinstance(text, str) else text

        values = {
            "model": model_name,
            "input": texts,
        }

        body = self._build_request_body(endpoint.request_template, values)

        try:
            client = await self._get_client()
            response = await client.request(
                endpoint.method,
                endpoint.path,
                json=body,
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings from response
            embeddings: list[list[float]] = []
            data_list = data.get("data", [])

            for item in data_list:
                embedding = item.get("embedding", [])
                embeddings.append(embedding)

            record_memory_operation(
                operation_type="openapi_provider_embeddings",
                success=True,
            )
            return embeddings

        except httpx.HTTPStatusError as e:
            record_memory_operation(
                operation_type="openapi_provider_embeddings",
                success=False,
            )
            raise ProviderGenerationError(
                f"Embedding failed: {e}",
                provider=ProviderType.CUSTOM,
            ) from e
        except httpx.HTTPError as e:
            record_memory_operation(
                operation_type="openapi_provider_embeddings",
                success=False,
            )
            raise ProviderConnectionError(
                f"Connection failed: {e}",
                provider=ProviderType.CUSTOM,
            ) from e
