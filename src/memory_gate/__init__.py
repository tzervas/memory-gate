"""MemoryGate: Dynamic memory learning layer for AI agents."""

__version__ = "0.1.0"

from .memory_protocols import LearningContext, KnowledgeStore, MemoryAdapter
from .memory_gateway import MemoryGateway
from .agent_interface import BaseMemoryEnabledAgent, AgentDomain, SimpleEchoAgent
from . import agents  # Import the agents submodule

__all__ = [
    "LearningContext",
    "KnowledgeStore",
    "MemoryAdapter",
    "MemoryGateway",
    "BaseMemoryEnabledAgent",
    "AgentDomain",
    "SimpleEchoAgent",
    "agents",
]
