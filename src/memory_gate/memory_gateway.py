import asyncio
from typing import Generic, TypeVar

from memory_gate.memory_protocols import KnowledgeStore, MemoryAdapter
from memory_gate.metrics import (
    record_memory_operation,
)  # Assuming this is a general operation counter

T = TypeVar(
    "T"
)  # If T is LearningContext, this is fine. If T can be other things, metrics might need adjustment.
# For now, assume T is typically LearningContext or similar enough that str(context) is meaningful.


class MemoryGateway(Generic[T]):
    """Central memory management system."""

    def __init__(self, adapter: MemoryAdapter[T], store: KnowledgeStore[T]) -> None:
        self.adapter = adapter
        self.store = store
        self._consolidation_task: asyncio.Task[None] | None = None

    async def learn_from_interaction(
        self, context: T, feedback: float | None = None
    ) -> T:
        """Process interaction and update knowledge."""
        try:
            # The adapter might have its own metrics if it becomes complex
            adapted_context = await self.adapter.adapt_knowledge(context, feedback)

            # Async storage to prevent blocking
            key = self._generate_key(adapted_context)
            # The actual store_experience in VectorMemoryStore now handles its own detailed metrics
            # Here, we might record a higher-level "gateway_learn_operation" if desired,
            # but for now, VectorMemoryStore's metrics are quite thorough.
            asyncio.create_task(self.store.store_experience(key, adapted_context))

            # Record a successful learn operation at the gateway level
            record_memory_operation(
                operation_type="gateway_learn_interaction", success=True
            )
            return adapted_context
        except Exception as e:
            record_memory_operation(
                operation_type="gateway_learn_interaction", success=False
            )
            print(f"Error in MemoryGateway learn_from_interaction: {e}")
            raise  # Re-raise for now

    def _generate_key(self, context: T) -> str:
        """Generate unique key for context storage."""
        import hashlib

        content_str = str(context)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
