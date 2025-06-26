import asyncio
from typing import Generic, TypeVar, Optional

from memory_gate.memory_protocols import MemoryAdapter, KnowledgeStore, LearningContext

T = TypeVar('T')

class MemoryGateway(Generic[T]):
    """Central memory management system."""
    
    def __init__(
        self, 
        adapter: MemoryAdapter[T],
        store: KnowledgeStore[T]
    ) -> None:
        self.adapter = adapter
        self.store = store
        self._consolidation_task: Optional[asyncio.Task] = None
    
    async def learn_from_interaction(
        self, 
        context: T, 
        feedback: Optional[float] = None
    ) -> T:
        """Process interaction and update knowledge."""
        adapted_context = await self.adapter.adapt_knowledge(context, feedback)
        
        # Async storage to prevent blocking
        key = self._generate_key(adapted_context)
        asyncio.create_task(self.store.store_experience(key, adapted_context))
        
        return adapted_context
    
    def _generate_key(self, context: T) -> str:
        """Generate unique key for context storage."""
        import hashlib
        content_str = str(context)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
