import asyncio
import hashlib
from typing import Generic, TypeVar, Optional

from memory_gate.memory_protocols import MemoryAdapter, KnowledgeStore
from memory_gate.metrics import record_memory_operation

T = TypeVar("T")

from typing import Generic, TypeVar, Optional, Tuple # Keep this one

class MemoryGateway(Generic[T]): # Keep this enhanced definition
    """Central memory management system."""

    def __init__(self, adapter: MemoryAdapter[T], store: KnowledgeStore[T]) -> None:
        self.adapter = adapter
        self.store = store
        self._consolidation_task: Optional[asyncio.Task[None]] = None

    async def learn_from_interaction(
        self, context: T, feedback: Optional[float] = None
    ) -> Tuple[T, asyncio.Task[None]]:
        """
        Process interaction, update knowledge, and return adapted context and storage task.

        Args:
            context: The learning context or data to process.
            feedback: Optional feedback score for adaptation.

        Returns:
            A tuple containing the adapted context and the asyncio.Task for the storage operation.
        """
        try:
            adapted_context = await self.adapter.adapt_knowledge(context, feedback)
            key = self._generate_key(adapted_context)

            # Always create a task for storage. The caller can choose to await it.
            storage_task = asyncio.create_task(self.store.store_experience(key, adapted_context))

            record_memory_operation(
                operation_type="gateway_learn_interaction", success=True
            )
            return adapted_context, storage_task
        except Exception as e:
            # If exception occurs before task creation, task won't exist.
            # If it occurs during task creation (unlikely for create_task itself),
            # it's a different scenario. Assume exceptions are from adapt_knowledge or _generate_key.
            record_memory_operation(
                operation_type="gateway_learn_interaction", success=False
            )
            print(f"Error in MemoryGateway learn_from_interaction: {e}")
            raise

    def _generate_key(self, context: T) -> str:
        """Generate unique key for context storage."""
        content_str = str(context)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def get_context_key(self, context: T) -> str:
        """
        Generates and returns a unique key for the given context.
        This is a public wrapper around the internal _generate_key method.
        """
        return self._generate_key(context)
