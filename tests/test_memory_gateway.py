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
async def test_learn_from_interaction_async_store(
    memory_gateway: MemoryGateway[LearningContext],
) -> None:
    """Test learning from an interaction with asynchronous storage (default)."""
    fixed_timestamp = datetime(2023, 1, 1, 12, 0, 0)
    context = LearningContext(
        content="Test async learning", domain="test_async", timestamp=fixed_timestamp
    )
    feedback_score = 0.8

    memory_gateway.adapter.adapt_knowledge.return_value = context
    memory_gateway.store.store_experience = AsyncMock()

    result = await memory_gateway.learn_from_interaction(context, feedback_score, sync_store=False)

    memory_gateway.adapter.adapt_knowledge.assert_called_once_with(context, feedback_score)
    await asyncio.sleep(0.01)
    expected_key = memory_gateway._generate_key(context) # Use internal method for verification design
    memory_gateway.store.store_experience.assert_called_once_with(expected_key, context)
    assert result == context


@pytest.mark.asyncio
async def test_learn_from_interaction_sync_store(
    memory_gateway: MemoryGateway[LearningContext],
) -> None:
    """Test learning from an interaction with synchronous storage."""
    fixed_timestamp = datetime(2023, 1, 1, 12, 0, 1) # Slightly different timestamp
    context = LearningContext(
        content="Test sync learning", domain="test_sync", timestamp=fixed_timestamp
    )
    feedback_score = 0.9

    memory_gateway.adapter.adapt_knowledge.return_value = context
    memory_gateway.store.store_experience = AsyncMock()

    result = await memory_gateway.learn_from_interaction(context, feedback_score, sync_store=True)

    memory_gateway.adapter.adapt_knowledge.assert_called_once_with(context, feedback_score)
    expected_key = memory_gateway._generate_key(context)
    memory_gateway.store.store_experience.assert_called_once_with(expected_key, context)
    assert result == context


@pytest.mark.asyncio
async def test_internal_key_generation(memory_gateway: MemoryGateway[LearningContext]) -> None:
    """Test _generate_key for consistency and format (internal method)."""
    fixed_timestamp = datetime(2023, 1, 1, 12, 0, 0)

    context1 = LearningContext(
        content="identical content for keygen", domain="test_keygen", timestamp=fixed_timestamp
    )
    context2 = LearningContext(
        content="identical content for keygen", domain="test_keygen", timestamp=fixed_timestamp
    )
    context3 = LearningContext(
        content="different content for keygen", domain="test_keygen", timestamp=fixed_timestamp
    )

    key1 = memory_gateway._generate_key(context1)
    key2 = memory_gateway._generate_key(context2)
    key3 = memory_gateway._generate_key(context3)

    assert key1 == key2, f"Keys for identical contexts should match. Key1: {key1}, Key2: {key2}"
    assert len(key1) == 16
    assert key1 != key3, f"Keys for different contexts should not match. Key1: {key1}, Key3: {key3}"
    assert len(key3) == 16


def test_public_get_context_key(memory_gateway: MemoryGateway[LearningContext]) -> None:
    """Test the public get_context_key method."""
    fixed_timestamp = datetime(2023, 1, 1, 12, 0, 2) # Different timestamp
    context = LearningContext(
        content="content for public key", domain="test_public_key", timestamp=fixed_timestamp
    )

    key = memory_gateway.get_context_key(context)
    expected_key = memory_gateway._generate_key(context)

    assert key == expected_key
    assert len(key) == 16

    context_alt = LearningContext(
        content="alternative content for public key", domain="test_public_key", timestamp=fixed_timestamp
    )
    key_alt = memory_gateway.get_context_key(context_alt)
    assert key_alt != key
    assert len(key_alt) == 16
