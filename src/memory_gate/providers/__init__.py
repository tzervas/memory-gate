"""Model Provider Framework for MemoryGate.

This module defines the provider/connector pattern for integrating with
various model backends (Ollama, OpenAI, Anthropic, local models, etc.).

The architecture follows a provider-consumer pattern:
- Providers: Implement the base protocol for specific backends
- Connectors: Handle the actual communication with backend APIs
- Consumer: The MemoryGate system that uses providers uniformly

This design enables:
- Easy addition of new model providers
- Consistent interface across all backends
- Centralized configuration management
- Simplified testing through mock providers

Available Providers:
- OllamaProvider: For Ollama local models
- OpenAPIProvider: Universal provider for any OpenAPI-compliant API
- (Future) OpenAIProvider, AnthropicProvider, etc.
"""

from .base import (
    BaseModelProvider,
    ChatMessage,
    GenerationConfig,
    ModelProviderConfig,
    ModelProviderError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderGenerationError,
    ProviderRateLimitError,
    ProviderResponse,
    ProviderType,
)
from .ollama import OllamaProvider, OllamaProviderConfig
from .openapi import EndpointConfig, OpenAPIProvider, OpenAPIProviderConfig
from .registry import (
    ProviderRegistry,
    get_provider,
    get_registry,
    list_available_providers,
    register_provider,
)

__all__ = [
    # Base classes and protocols
    "BaseModelProvider",
    "ChatMessage",
    "GenerationConfig",
    "ModelProviderConfig",
    "ModelProviderError",
    "ProviderAuthenticationError",
    "ProviderConnectionError",
    "ProviderGenerationError",
    "ProviderRateLimitError",
    "ProviderResponse",
    "ProviderType",
    # Ollama provider
    "OllamaProvider",
    "OllamaProviderConfig",
    # OpenAPI provider (universal)
    "EndpointConfig",
    "OpenAPIProvider",
    "OpenAPIProviderConfig",
    # Registry
    "ProviderRegistry",
    "get_provider",
    "get_registry",
    "list_available_providers",
    "register_provider",
]
