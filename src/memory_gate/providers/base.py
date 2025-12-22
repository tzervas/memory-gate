"""Base classes and protocols for model providers.

This module defines the abstract interfaces that all model providers must implement,
ensuring a consistent API across different backends (Ollama, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ProviderType(Enum):
    """Supported provider types."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class ChatMessage:
    """A message in a chat conversation.

    This is the universal message format used across all providers.

    Args:
        role: The role of the message sender (system, user, assistant)
        content: The message content
        images: Optional list of base64-encoded images (for multimodal models)
        metadata: Optional additional metadata
    """

    role: str
    content: str
    images: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API requests."""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            result["images"] = self.images
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        """Create an assistant message."""
        return cls(role="assistant", content=content)

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Create a system message."""
        return cls(role="system", content=content)


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Universal configuration that providers translate to their specific formats.

    Args:
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop_sequences: Sequences that stop generation
        presence_penalty: Penalty for token presence
        frequency_penalty: Penalty for token frequency
        seed: Random seed for reproducibility
        extra: Provider-specific extra parameters
    """

    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    """Standardized response from any model provider.

    Args:
        content: The generated text content
        model: The model used for generation
        provider: The provider type that generated this response
        done: Whether generation is complete
        usage: Token usage statistics (if available)
        latency_ms: Response latency in milliseconds
        metadata: Additional provider-specific metadata
    """

    content: str
    model: str
    provider: ProviderType
    done: bool = True
    usage: dict[str, int] | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int | None:
        """Get prompt token count if available."""
        if self.usage:
            return self.usage.get("prompt_tokens")
        return None

    @property
    def completion_tokens(self) -> int | None:
        """Get completion token count if available."""
        if self.usage:
            return self.usage.get("completion_tokens")
        return None


class ModelProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, provider: ProviderType | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider


class ProviderConnectionError(ModelProviderError):
    """Raised when connection to provider fails."""


class ProviderGenerationError(ModelProviderError):
    """Raised when generation fails."""


class ProviderAuthenticationError(ModelProviderError):
    """Raised when authentication fails."""


class ProviderRateLimitError(ModelProviderError):
    """Raised when rate limit is exceeded."""


@dataclass
class ModelProviderConfig:
    """Base configuration for model providers.

    Subclass this for provider-specific configurations.

    Args:
        provider_type: Type of the provider
        default_model: Default model to use
        timeout_seconds: Request timeout
        max_retries: Maximum retry attempts
        retry_delay_seconds: Delay between retries
    """

    provider_type: ProviderType
    default_model: str
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@runtime_checkable
class ModelProviderProtocol(Protocol):
    """Protocol defining the interface all providers must implement."""

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        ...

    async def connect(self) -> None:
        """Establish connection to the provider."""
        ...

    async def disconnect(self) -> None:
        """Close connection to the provider."""
        ...

    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        ...

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        ...

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate text from a prompt."""
        ...

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        ...

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Chat with the model."""
        ...

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Chat with streaming."""
        ...

    async def get_embeddings(
        self,
        text: str | list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Get embeddings for text."""
        ...


class BaseModelProvider(ABC):
    """Abstract base class for model providers.

    All concrete providers (Ollama, OpenAI, etc.) should inherit from this class
    and implement the abstract methods.

    This provides:
    - Common initialization patterns
    - Shared utility methods
    - Consistent error handling
    - Connection lifecycle management
    """

    def __init__(self, config: ModelProviderConfig) -> None:
        """Initialize the provider with configuration.

        Args:
            config: Provider-specific configuration
        """
        self._config = config
        self._connected = False

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return self._config.provider_type

    @property
    def default_model(self) -> str:
        """Return the default model."""
        return self._config.default_model

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the provider.

        Should set self._connected = True on success.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the provider.

        Should set self._connected = False.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and responding.

        Returns:
            True if healthy, False otherwise
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from the provider.

        Returns:
            List of model information dictionaries
        """
        ...

    @abstractmethod
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
            **kwargs: Provider-specific parameters

        Returns:
            ProviderResponse with generated text
        """
        ...

    @abstractmethod
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
            **kwargs: Provider-specific parameters

        Yields:
            Text chunks as they're generated
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Chat with the model using message history.

        Args:
            messages: List of chat messages
            model: Model to use
            config: Generation configuration
            **kwargs: Provider-specific parameters

        Returns:
            ProviderResponse with assistant's reply
        """
        ...

    @abstractmethod
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
            **kwargs: Provider-specific parameters

        Yields:
            Text chunks as they're generated
        """
        ...

    @abstractmethod
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
        ...

    def _get_model(self, model: str | None) -> str:
        """Get model name, using default if not specified."""
        return model or self.default_model

    def _build_generation_config(
        self, config: GenerationConfig | None
    ) -> GenerationConfig:
        """Build generation config with defaults."""
        return config or GenerationConfig()

    async def __aenter__(self) -> "BaseModelProvider":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
