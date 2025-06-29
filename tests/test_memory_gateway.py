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
async def test_learn_from_interaction_returns_context_and_task(
    memory_gateway: MemoryGateway[LearningContext],
) -> None:
    """Test learn_from_interaction returns adapted context and a storage task."""
    fixed_timestamp = datetime(2023, 1, 1, 12, 0, 0)
    context = LearningContext(
        content="Test interaction learning", domain="test_interaction", timestamp=fixed_timestamp
    )
    feedback_score = 0.8

    # Mock adapter behavior
    memory_gateway.adapter.adapt_knowledge.return_value = context
    # Mock store_experience to be an awaitable async function
    mock_store_experience = AsyncMock()
    memory_gateway.store.store_experience = mock_store_experience

    adapted_context, storage_task = await memory_gateway.learn_from_interaction(context, feedback_score)

    # Verify adapter was called
    memory_gateway.adapter.adapt_knowledge.assert_called_once_with(context, feedback_score)

    # Assert the returned context is what the adapter returned
    assert adapted_context == context

    # Assert a task was returned and it's an asyncio.Task
    assert isinstance(storage_task, asyncio.Task)

    # Allow the task to complete
    await storage_task

    # Verify store_experience was called by the task
    expected_key = memory_gateway.get_context_key(context) # Use public method now
    mock_store_experience.assert_called_once_with(expected_key, context)


@pytest.mark.asyncio
async def test_learn_from_interaction_task_actually_stores(
    memory_gateway: MemoryGateway[LearningContext],
) -> None:
    """Test that the task returned by learn_from_interaction correctly stores the data."""
    fixed_timestamp = datetime(2023, 1, 1, 12, 0, 1)
    context = LearningContext(
        content="Test storage via task", domain="test_storage_task", timestamp=fixed_timestamp
    )

    # Real store mock that we can inspect after the task
    actual_storage_dict = {}
    async def fake_store_experience(key: str, exp: LearningContext):
        await asyncio.sleep(0.01) # simulate some io
        actual_storage_dict[key] = exp

    memory_gateway.store.store_experience = fake_store_experience
    memory_gateway.adapter.adapt_knowledge.return_value = context

    _, storage_task = await memory_gateway.learn_from_interaction(context)
    await storage_task # Ensure task completion

    expected_key = memory_gateway.get_context_key(context)
    assert expected_key in actual_storage_dict
    assert actual_storage_dict[expected_key] == context


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
