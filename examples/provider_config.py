"""Provider configuration example for MemoryGate.

This script demonstrates how to configure and use different model providers:
- Ollama provider
- OpenAI-compatible providers
- Custom provider configuration
"""

import asyncio

from memory_gate.providers.base import GenerationConfig
from memory_gate.providers.ollama import OllamaProvider
from memory_gate.providers.openapi import OpenAPIProvider
from memory_gate.providers.registry import ProviderRegistry


async def demonstrate_ollama_provider() -> None:
    """Demonstrate Ollama provider usage."""
    print("\n" + "=" * 60)
    print("1. Ollama Provider Example")
    print("=" * 60)

    # Create Ollama provider
    provider = OllamaProvider(base_url="http://localhost:11434")

    # Check if Ollama is available
    try:
        models = await provider.list_models()
        print(f"✓ Connected to Ollama")
        print(
            f"  Available models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}"
        )

        # Generate a response
        config = GenerationConfig(
            model="llama3",
            temperature=0.7,
            max_tokens=100,
        )

        print(f"\n  Generating response with model: {config.model}")
        response = await provider.generate(
            prompt="What is Python in one sentence?",
            config=config,
        )
        print(f"  Response: {response.content}")

    except Exception as e:
        print(f"✗ Could not connect to Ollama: {e}")
        print("  Please ensure Ollama is running: ollama serve")


async def demonstrate_openai_provider() -> None:
    """Demonstrate OpenAI-compatible provider usage."""
    print("\n" + "=" * 60)
    print("2. OpenAI-Compatible Provider Example")
    print("=" * 60)

    # Create OpenAI provider (or any OpenAI-compatible API)
    # Note: This requires a valid API key
    provider = OpenAPIProvider(
        base_url="https://api.openai.com/v1",
        api_key="your-api-key-here",  # Set via environment variable in production
    )

    print("  OpenAI-compatible provider created")
    print("  Note: Set OPENAI_API_KEY environment variable for actual usage")

    # Example configuration
    config = GenerationConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=100,
    )
    print(f"  Configuration: model={config.model}, temp={config.temperature}")


async def demonstrate_provider_registry() -> None:
    """Demonstrate provider registry for managing multiple providers."""
    print("\n" + "=" * 60)
    print("3. Provider Registry Example")
    print("=" * 60)

    # Create registry
    registry = ProviderRegistry()

    # Register Ollama provider
    ollama = OllamaProvider(base_url="http://localhost:11434")
    registry.register("ollama", ollama)
    print("  ✓ Registered 'ollama' provider")

    # Register OpenAI provider
    openai = OpenAPIProvider(
        base_url="https://api.openai.com/v1",
        api_key="your-api-key-here",
    )
    registry.register("openai", openai)
    print("  ✓ Registered 'openai' provider")

    # Set default provider
    registry.set_default("ollama")
    print("  ✓ Set 'ollama' as default provider")

    # List registered providers
    providers = registry.list_providers()
    print(f"\n  Registered providers: {', '.join(providers)}")
    print(f"  Default provider: {registry.default_provider_name}")

    # Get provider by name
    provider = registry.get_provider("ollama")
    print(f"\n  Retrieved provider: {provider.__class__.__name__}")


async def demonstrate_custom_configuration() -> None:
    """Demonstrate custom provider configuration."""
    print("\n" + "=" * 60)
    print("4. Custom Configuration Example")
    print("=" * 60)

    # Custom Ollama configuration for local model
    print("\n  Example 1: Local Ollama with custom model")
    config1 = GenerationConfig(
        model="mistral",
        temperature=0.5,
        max_tokens=500,
        top_p=0.9,
    )
    print(f"    Model: {config1.model}")
    print(f"    Temperature: {config1.temperature}")
    print(f"    Max tokens: {config1.max_tokens}")

    # OpenAI configuration for GPT-4
    print("\n  Example 2: OpenAI GPT-4 configuration")
    config2 = GenerationConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        top_p=0.95,
    )
    print(f"    Model: {config2.model}")
    print(f"    Temperature: {config2.temperature}")
    print(f"    Max tokens: {config2.max_tokens}")

    # Local LLM configuration
    print("\n  Example 3: Custom local API configuration")
    custom_provider = OpenAPIProvider(
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
    )
    print(f"    Base URL: http://localhost:8080/v1")
    print(f"    Provider: OpenAPI-compatible")


async def main() -> None:
    """Run all provider examples."""
    print("MemoryGate Provider Configuration Examples")

    await demonstrate_ollama_provider()
    await demonstrate_openai_provider()
    await demonstrate_provider_registry()
    await demonstrate_custom_configuration()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Configure providers in config.yaml or config.toml")
    print("2. Set environment variables for API keys")
    print("3. Use providers with MemoryGate API for memory-augmented generation")


if __name__ == "__main__":
    asyncio.run(main())
