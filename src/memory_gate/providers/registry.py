"""Provider registry for managing model providers.

This module provides a registry pattern for managing model providers,
enabling dynamic provider registration and lookup.
"""

import logging
from typing import Any

from .base import (
    BaseModelProvider,
    ModelProviderConfig,
    ModelProviderError,
    ProviderType,
)
from .ollama import OllamaProvider, OllamaProviderConfig
from .openapi import OpenAPIProvider, OpenAPIProviderConfig

logger = logging.getLogger(__name__)

# Type alias for provider factory functions
ProviderFactory = type[BaseModelProvider]


class ProviderRegistry:
    """Registry for model providers.

    Manages registration and instantiation of model providers.
    Supports both built-in and custom providers.

    Example:
        ```python
        # Register a custom provider
        registry = ProviderRegistry()
        registry.register(ProviderType.CUSTOM, MyCustomProvider)

        # Get a provider instance
        provider = registry.get(ProviderType.OLLAMA, config)
        ```
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in providers."""
        self._providers: dict[ProviderType, ProviderFactory] = {}
        self._default_configs: dict[ProviderType, type[ModelProviderConfig]] = {}
        self._register_builtin_providers()

    def _register_builtin_providers(self) -> None:
        """Register built-in providers."""
        # Ollama provider
        self._providers[ProviderType.OLLAMA] = OllamaProvider
        self._default_configs[ProviderType.OLLAMA] = OllamaProviderConfig

        # OpenAPI universal provider
        self._providers[ProviderType.CUSTOM] = OpenAPIProvider
        self._default_configs[ProviderType.CUSTOM] = OpenAPIProviderConfig

        # Future providers will be registered here:
        # self._providers[ProviderType.OPENAI] = OpenAIProvider
        # self._default_configs[ProviderType.OPENAI] = OpenAIProviderConfig
        # self._providers[ProviderType.ANTHROPIC] = AnthropicProvider
        # self._default_configs[ProviderType.ANTHROPIC] = AnthropicProviderConfig

    def register(
        self,
        provider_type: ProviderType,
        provider_class: ProviderFactory,
    ) -> None:
        """Register a provider class.

        Args:
            provider_type: Type identifier for the provider
            provider_class: Provider class to register
        """
        if provider_type in self._providers:
            logger.warning(
                "Overwriting existing provider registration for %s",
                provider_type.value,
            )
        self._providers[provider_type] = provider_class
        logger.info("Registered provider: %s", provider_type.value)

    def unregister(self, provider_type: ProviderType) -> bool:
        """Unregister a provider.

        Args:
            provider_type: Type identifier of provider to remove

        Returns:
            True if provider was removed, False if not found
        """
        if provider_type in self._providers:
            del self._providers[provider_type]
            logger.info("Unregistered provider: %s", provider_type.value)
            return True
        return False

    def get(
        self,
        provider_type: ProviderType,
        config: ModelProviderConfig | None = None,
    ) -> BaseModelProvider:
        """Get a provider instance.

        Args:
            provider_type: Type of provider to get
            config: Optional configuration for the provider

        Returns:
            Configured provider instance

        Raises:
            ModelProviderError: If provider type is not registered
        """
        if provider_type not in self._providers:
            raise ModelProviderError(
                f"Provider not registered: {provider_type.value}",
                provider=provider_type,
            )

        provider_class = self._providers[provider_type]

        # Use default config if none provided
        if config is None:
            if provider_type in self._default_configs:
                config = self._default_configs[provider_type]()
            else:
                raise ModelProviderError(
                    f"No default config for provider: {provider_type.value}",
                    provider=provider_type,
                )

        return provider_class(config)  # type: ignore[arg-type]

    def list_providers(self) -> list[ProviderType]:
        """List all registered provider types.

        Returns:
            List of registered provider types
        """
        return list(self._providers.keys())

    def is_registered(self, provider_type: ProviderType) -> bool:
        """Check if a provider type is registered.

        Args:
            provider_type: Type to check

        Returns:
            True if registered, False otherwise
        """
        return provider_type in self._providers


# Global registry instance
_global_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry.

    Returns:
        Global ProviderRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ProviderRegistry()
    return _global_registry


def register_provider(
    provider_type: ProviderType,
    provider_class: ProviderFactory,
) -> None:
    """Register a provider in the global registry.

    Args:
        provider_type: Type identifier for the provider
        provider_class: Provider class to register
    """
    get_registry().register(provider_type, provider_class)


def get_provider(
    provider_type: ProviderType,
    config: ModelProviderConfig | None = None,
    **config_kwargs: Any,
) -> BaseModelProvider:
    """Get a provider from the global registry.

    Args:
        provider_type: Type of provider to get
        config: Optional configuration object
        **config_kwargs: If config is None, these are used to create config

    Returns:
        Configured provider instance

    Example:
        ```python
        # Using config object
        provider = get_provider(
            ProviderType.OLLAMA,
            config=OllamaProviderConfig(base_url="http://localhost:11434")
        )

        # Using kwargs (creates default config with overrides)
        provider = get_provider(
            ProviderType.OLLAMA,
            base_url="http://localhost:11434",
            default_model="mistral"
        )

        # Using the universal OpenAPI provider
        provider = get_provider(
            ProviderType.CUSTOM,
            config=OpenAPIProviderConfig(
                base_url="https://api.example.com",
                api_key="your-key",
            )
        )
        ```
    """
    registry = get_registry()

    # If no config but kwargs provided, create appropriate config
    if config is None and config_kwargs:
        if provider_type == ProviderType.OLLAMA:
            config = OllamaProviderConfig(**config_kwargs)
        elif provider_type == ProviderType.CUSTOM:
            config = OpenAPIProviderConfig(**config_kwargs)
        # Add other provider configs here as they're implemented

    return registry.get(provider_type, config)


def list_available_providers() -> list[str]:
    """List names of all available providers.

    Returns:
        List of provider type names
    """
    return [p.value for p in get_registry().list_providers()]
