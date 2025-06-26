import pytest
from datetime import datetime

from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.in_memory import InMemoryKnowledgeStore

@pytest.fixture
def in_memory_store() -> InMemoryKnowledgeStore:
    """Create an in-memory knowledge store."""
    return InMemoryKnowledgeStore()

@pytest.mark.asyncio
async def test_store_and_retrieve(in_memory_store: InMemoryKnowledgeStore) -> None:
    """Test storing and retrieving an experience."""
    context = LearningContext(
        content="Test content",
        domain="test",
        timestamp=datetime.now()
    )

    await in_memory_store.store_experience("test_key", context)
    results = await in_memory_store.retrieve_context("content")

    assert len(results) == 1
    assert results[0] == context

@pytest.mark.asyncio
async def test_retrieve_miss(in_memory_store: InMemoryKnowledgeStore) -> None:
    """Test retrieving a non-existent experience."""
    results = await in_memory_store.retrieve_context("non_existent")
    assert len(results) == 0

@pytest.mark.asyncio
async def test_retrieve_limit(in_memory_store: InMemoryKnowledgeStore) -> None:
    """Test the limit parameter of retrieve_context."""
    for i in range(5):
        context = LearningContext(
            content=f"Test content {i}",
            domain="test",
            timestamp=datetime.now()
        )
        await in_memory_store.store_experience(f"test_key_{i}", context)

    results = await in_memory_store.retrieve_context("content", limit=3)
    assert len(results) == 3

