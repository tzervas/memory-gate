import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from memory_gate.consolidation import ConsolidationWorker
from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.vector_store import VectorMemoryStore, VectorStoreConfig


@pytest_asyncio.fixture
async def vector_store_for_consolidation(
    temp_chroma_directory: str,
) -> VectorMemoryStore:
    """Create a VectorMemoryStore instance for consolidation tests, with persistence."""
    config = VectorStoreConfig(
        collection_name="consolidation_test_collection",
        persist_directory=temp_chroma_directory,
    )
    return VectorMemoryStore(config=config)


@pytest.fixture
def consolidation_worker(
    vector_store_for_consolidation: VectorMemoryStore,
) -> ConsolidationWorker:
    """Create a ConsolidationWorker instance for testing."""
    return ConsolidationWorker(
        store=vector_store_for_consolidation, consolidation_interval=0.1
    )


@pytest.mark.asyncio
async def test_consolidation_worker_instantiation(
    consolidation_worker: ConsolidationWorker,
) -> None:
    """Test that the ConsolidationWorker can be instantiated."""
    assert consolidation_worker is not None
    assert consolidation_worker.store is not None


@pytest.mark.asyncio
async def test_consolidation_loop_calls_perform_consolidation(
    consolidation_worker: ConsolidationWorker,
) -> None:
    """Test that the consolidation loop calls _perform_consolidation."""
    # Mock the actual _perform_consolidation to check if it's called
    consolidation_worker._perform_consolidation = AsyncMock()

    await consolidation_worker.start()
    await asyncio.sleep(0.25)  # Allow time for at least one or two calls
    await consolidation_worker.stop()

    assert consolidation_worker._perform_consolidation.call_count >= 1


@pytest.mark.asyncio
async def test_perform_consolidation_deletes_old_low_importance_items(
    vector_store_for_consolidation: VectorMemoryStore,  # Use the store directly to setup
    consolidation_worker: ConsolidationWorker,  # Worker will use the same store
) -> None:
    """Test that _perform_consolidation correctly deletes old, low-importance items."""
    store = vector_store_for_consolidation
    now = datetime.now()

    # Item 1: Old and low importance (should be deleted)
    context1 = LearningContext(
        content="Old low importance",
        domain="test",
        timestamp=now - timedelta(days=40),
        importance=0.1,
    )
    await store.store_experience("key1", context1)

    # Item 2: New but low importance (should NOT be deleted by age criteria)
    context2 = LearningContext(
        content="New low importance",
        domain="test",
        timestamp=now - timedelta(days=10),
        importance=0.1,
    )
    await store.store_experience("key2", context2)

    # Item 3: Old but high importance (should NOT be deleted by importance criteria)
    context3 = LearningContext(
        content="Old high importance",
        domain="test",
        timestamp=now - timedelta(days=40),
        importance=0.9,
    )
    await store.store_experience("key3", context3)

    # Item 4: New and high importance (should NOT be deleted)
    context4 = LearningContext(
        content="New high importance",
        domain="test",
        timestamp=now - timedelta(days=10),
        importance=0.9,
    )
    await store.store_experience("key4", context4)

    # Item 5: Old and borderline low importance (should be deleted)
    context5 = LearningContext(
        content="Old borderline low importance",
        domain="test",
        timestamp=now - timedelta(days=31),
        importance=0.19,
    )
    await store.store_experience("key5", context5)

    assert store.get_collection_size() == 5

    # Manually trigger consolidation for testing this specific logic
    # The ConsolidationWorker's loop is tested separately.
    await consolidation_worker._perform_consolidation()

    assert store.get_collection_size() == 3  # key1 and key5 should be deleted

    # Verify remaining items
    item2_retrieved = await store.get_experience_by_id("key2")
    assert item2_retrieved is not None
    assert item2_retrieved.content == context2.content

    item3_retrieved = await store.get_experience_by_id("key3")
    assert item3_retrieved is not None
    assert item3_retrieved.content == context3.content

    item4_retrieved = await store.get_experience_by_id("key4")
    assert item4_retrieved is not None
    assert item4_retrieved.content == context4.content

    # Verify deleted items are gone
    item1_retrieved = await store.get_experience_by_id("key1")
    assert item1_retrieved is None
    item5_retrieved = await store.get_experience_by_id("key5")
    assert item5_retrieved is None


@pytest.mark.asyncio
async def test_consolidation_with_empty_store(
    consolidation_worker: ConsolidationWorker,
) -> None:
    """Test that consolidation runs without errors on an empty store."""
    # Store is already empty as it's fresh for this worker
    try:
        await consolidation_worker._perform_consolidation()
    except Exception as e:
        pytest.fail(f"Consolidation on empty store raised an exception: {e}")

    # Check collection size (should still be 0)
    # Need access to the store from the worker
    assert consolidation_worker.store.get_collection_size() == 0


@pytest.mark.asyncio
async def test_consolidation_with_no_matching_items_for_deletion(
    vector_store_for_consolidation: VectorMemoryStore,
    consolidation_worker: ConsolidationWorker,
) -> None:
    """Test consolidation when no items meet deletion criteria."""
    store = vector_store_for_consolidation
    now = datetime.now()

    # Add items that should NOT be deleted
    context1 = LearningContext(
        content="New high importance item",
        domain="test",
        timestamp=now - timedelta(days=5),
        importance=0.8,
    )
    await store.store_experience("no_delete_key1", context1)
    context2 = LearningContext(
        content="Old but very high importance item",
        domain="test",
        timestamp=now - timedelta(days=100),
        importance=0.99,
    )
    await store.store_experience("no_delete_key2", context2)

    initial_size = store.get_collection_size()
    assert initial_size == 2

    await consolidation_worker._perform_consolidation()

    assert store.get_collection_size() == initial_size  # No items should be deleted
