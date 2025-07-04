from memory_gate.memory_protocols import KnowledgeStore, LearningContext


class InMemoryKnowledgeStore(KnowledgeStore[LearningContext]):
    """An in-memory implementation of the KnowledgeStore protocol."""

    def __init__(self) -> None:
        self._store: dict[str, LearningContext] = {}

    async def store_experience(self, key: str, experience: LearningContext) -> None:
        """Stores a learning experience in the in-memory dictionary."""
        self._store[key] = experience

    async def retrieve_context(
        self, query: str, limit: int = 10, domain_filter: str | None = None
    ) -> list[LearningContext]:
        """Retrieves relevant context from the in-memory store."""
        # This is a naive implementation for demonstration purposes.
        # A real implementation would use a more sophisticated search.
        results = [
            exp
            for exp in self._store.values()
            if query.lower() in exp.content.lower()
            and (domain_filter is None or exp.domain == domain_filter)
        ]
        return results[:limit]
