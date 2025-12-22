"""FastAPI dependency injection utilities."""

from collections.abc import AsyncGenerator

from memory_gate.memory_gateway import MemoryGateway
from memory_gate.storage.in_memory import InMemoryKnowledgeStore

# Global gateway instance (will be configured on app startup)
_gateway: MemoryGateway | None = None


def configure_gateway(gateway: MemoryGateway) -> None:
    """Configure the global memory gateway instance.

    Args:
        gateway: Memory gateway instance to use.
    """
    global _gateway
    _gateway = gateway


async def get_memory_gateway() -> AsyncGenerator[MemoryGateway]:
    """Dependency that provides a memory gateway instance.

    Yields:
        MemoryGateway: Memory gateway instance.

    Raises:
        RuntimeError: If gateway is not configured.
    """
    if _gateway is None:
        # Create default gateway with in-memory store for development
        gateway = MemoryGateway(store=InMemoryKnowledgeStore())
        yield gateway
    else:
        yield _gateway
