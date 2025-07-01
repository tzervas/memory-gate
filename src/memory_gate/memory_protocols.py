from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeVar

T = TypeVar("T")


class MemoryAdapter(Protocol[T]):
    """Protocol for memory adaptation strategies."""

    async def adapt_knowledge(self, context: T, feedback: float | None = None) -> T:
        """Adapt knowledge based on context and feedback."""
        ...


class KnowledgeStore(Protocol[T]):
    """Protocol for knowledge persistence."""

    async def store_experience(self, key: str, experience: T) -> None:
        """Store learning experience."""
        ...

    async def retrieve_context(
        self, query: str, limit: int = 10, domain_filter: str | None = None
    ) -> list[T]:
        """Retrieve relevant context."""
        ...


@dataclass
class LearningContext:
    """Container for learning context data."""

    content: str
    domain: str
    timestamp: datetime
    importance: float = 1.0
    metadata: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
