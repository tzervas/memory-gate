import asyncio
import pytest
from unittest.mock import AsyncMock
from datetime import datetime

from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import LearningContext, MemoryAdapter, KnowledgeStore


@pytest.fixture
def memory_gateway() -> MemoryGateway[LearningContext]:
    """Create a test memory gateway."""
    adapter = AsyncMock(spec=MemoryAdapter)
    store = AsyncMock(spec=KnowledgeStore)
    return MemoryGateway(adapter, store)


@pytest.mark.asyncio
async def test_learn_from_interaction(
    memory_gateway: MemoryGateway[LearningContext],
) -> None:
    """Test learning from an interaction."""
    context = LearningContext(
        content="Test learning content", domain="test", timestamp=datetime.now()
    )

    # Mock adapter behavior
    memory_gateway.adapter.adapt_knowledge.return_value = context

    result = await memory_gateway.learn_from_interaction(context, 0.8)

    # Verify adapter was called
    memory_gateway.adapter.adapt_knowledge.assert_called_once_with(context, 0.8)

    # Verify store was called (asynchronously)
    await asyncio.sleep(0.1)  # Allow async task to complete
    memory_gateway.store.store_experience.assert_called_once()

    assert result == context


@pytest.mark.asyncio
async def test_key_generation(memory_gateway: MemoryGateway[LearningContext]) -> None:
    """Test key generation consistency."""
    context1 = LearningContext(
        content="identical content",
        domain="test",
        timestamp=datetime(2023, 1, 1, 12, 0, 0),
    )
    context2 = LearningContext(
        content="identical content",
        domain="test",
        timestamp=datetime(2023, 1, 1, 12, 0, 0),
    )

    key1 = memory_gateway._generate_key(context1)
    key2 = memory_gateway._generate_key(context2)

    # Keys should be deterministic for the same content
    assert key1 == key2
    assert len(key1) == 16
