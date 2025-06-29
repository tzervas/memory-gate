import asyncio
from typing import Dict, List, Optional

from memory_gate.memory_protocols import KnowledgeStore, LearningContext


class InMemoryKnowledgeStore(KnowledgeStore[LearningContext]):
    """
    Basic in-memory knowledge store using a dictionary.
    Implements domain filtering and sorting for retrieval.
    """

    def __init__(self) -> None:
        self.storage: Dict[str, LearningContext] = {}
        self._lock = asyncio.Lock()

    async def store_experience(self, key: str, experience: LearningContext) -> None:
        """Store learning experience in memory."""
        async with self._lock:
            self.storage[key] = experience

    async def retrieve_context(
        self, query: str, limit: int = 10, domain_filter: Optional[str] = None
    ) -> List[LearningContext]:
        """
        Retrieve relevant context from memory.
        Filters by domain (if provided) and then by naive query string matching in content.
        Sorts results by importance (descending) and then by timestamp (descending).
        """
        async with self._lock:
            results: List[LearningContext] = []

            candidate_experiences = list(self.storage.values())

            if domain_filter:
                candidate_experiences = [
                    exp for exp in candidate_experiences if exp.domain == domain_filter
                ]

            results.extend(
                exp
                for exp in candidate_experiences
                if query.lower() in exp.content.lower()
            )
            results.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)

            return results[:limit]

    async def get_experience_by_id(self, key: str) -> Optional[LearningContext]:
        """Retrieve a specific experience by its id (for testing/debugging)."""
        async with self._lock:
            return self.storage.get(key)

    async def get_all_experiences(self) -> List[LearningContext]:
        """Retrieve all experiences (mainly for debugging/testing)."""
        async with self._lock:
            return list(self.storage.values())

    async def clear_storage(self) -> None:
        """Clears all experiences from the store (mainly for testing)."""
        async with self._lock:
            self.storage.clear()
