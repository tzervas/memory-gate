"""MemoryGate: Dynamic memory learning layer for AI agents."""

import logging

from . import agents  # Import the agents submodule
from . import providers  # Import the providers submodule
from .agent_interface import AgentDomain, BaseMemoryEnabledAgent, SimpleEchoAgent
from .memory_gateway import MemoryGateway
from .memory_protocols import KnowledgeStore, LearningContext, MemoryAdapter
from .ollama_bridge import (
    ChatMessage,
    OllamaBridgeConfig,
    OllamaBridgeError,
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaMemoryBridge,
    OllamaResponse,
    create_memory_bridge,
)

# Re-export commonly used provider classes at top level for convenience
from .providers import (
    BaseModelProvider,
    GenerationConfig,
    OllamaProvider,
    OllamaProviderConfig,
    OpenAPIProvider,
    OpenAPIProviderConfig,
    ProviderResponse,
    ProviderType,
    get_provider,
)

__version__ = "0.1.0"

# Configure logging for consistent output across the package
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO,
)

__all__ = [
    # Agent system
    "AgentDomain",
    "BaseMemoryEnabledAgent",
    "SimpleEchoAgent",
    "agents",
    # Memory system
    "KnowledgeStore",
    "LearningContext",
    "MemoryAdapter",
    "MemoryGateway",
    # Ollama bridge (backward compatible)
    "ChatMessage",
    "OllamaBridgeConfig",
    "OllamaBridgeError",
    "OllamaConnectionError",
    "OllamaGenerationError",
    "OllamaMemoryBridge",
    "OllamaResponse",
    "create_memory_bridge",
    # Provider framework
    "BaseModelProvider",
    "GenerationConfig",
    "OllamaProvider",
    "OllamaProviderConfig",
    "OpenAPIProvider",
    "OpenAPIProviderConfig",
    "ProviderResponse",
    "ProviderType",
    "get_provider",
    "providers",
]
