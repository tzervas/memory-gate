"""MemoryGate: Dynamic memory learning layer for AI agents."""

import logging

from . import agents  # Import the agents submodule
from .agent_interface import AgentDomain, BaseMemoryEnabledAgent, SimpleEchoAgent
from .memory_gateway import MemoryGateway
from .memory_protocols import KnowledgeStore, LearningContext, MemoryAdapter

__version__ = "0.1.0"

# Configure logging for consistent output across the package
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO,
)

__all__ = [
    "AgentDomain",
    "BaseMemoryEnabledAgent",
    "KnowledgeStore",
    "LearningContext",
    "MemoryAdapter",
    "MemoryGateway",
    "SimpleEchoAgent",
    "agents",
]
