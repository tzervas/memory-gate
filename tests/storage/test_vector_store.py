"""This module contains tests for the VectorMemoryStore, covering functionality such as initialization, data persistence, experience lifecycle, and error handling."""

import logging
from dataclasses import replace
from datetime import datetime

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.extra.pytz import timezones

from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.vector_store import VectorMemoryStore

logger = logging.getLogger(__name__)


@pytest.fixture(params=["persistent_vector_store", "in_memory_vector_store"])
def vector_store(request):
    """Parameterized fixture to test both persistent and in-memory vector stores."""
    return request.getfixturevalue(request.param)


@pytest.mark.asyncio
async def test_initialization(
    persistent_vector_store: VectorMemoryStore,
    in_memory_vector_store: VectorMemoryStore,
):
    """Test store initialization with different configurations."""
    assert persistent_vector_store.collection is not None
    assert persistent_vector_store.collection_name == "test_persistent_collection"

    assert in_memory_vector_store.collection is not None
    assert in_memory_vector_store.collection_name == "test_in_memory_collection"


@pytest.mark.asyncio
async def test_store_and_retrieve_single_experience(vector_store: VectorMemoryStore):
    """Test storing and retrieving a single learning experience."""
    logger.info("Starting single experience store and retrieve test")

    context = LearningContext(
        content="Test infrastructure issue: high CPU on web-01",
        domain="infrastructure",
        timestamp=datetime.now(),
        importance=0.85,
        metadata={"server": "web-01", "metric": "cpu_utilization"},
    )
    key = "infra_cpu_web01"

    logger.info(f"Storing experience with key: {key}")
    await vector_store.store_experience(key, context)
    logger.info(
        f"Experience stored. Collection size: {vector_store.get_collection_size()}"
    )
    assert vector_store.get_collection_size() == 1

    logger.info("Retrieving context with similarity search")
    retrieved_contexts = await vector_store.retrieve_context(
        query="CPU issue on web server", limit=1
    )
    logger.info(f"Retrieved {len(retrieved_contexts)} contexts")

    assert len(retrieved_contexts) == 1
    retrieved_context = retrieved_contexts[0]
    logger.info("Validating retrieved context matches original")
    assert retrieved_context.content == context.content
    assert retrieved_context.domain == context.domain
    assert retrieved_context.importance == context.importance
    assert retrieved_context.metadata == context.metadata
    logger.info("Single experience test completed successfully")


@pytest.mark.asyncio
async def test_experience_lifecycle(vector_store: VectorMemoryStore):
    """Test full lifecycle of an experience (store, retrieve, update, delete)."""
    key = "lifecycle_key"
    context = LearningContext(
        content="Initial content for lifecycle test",
        domain="lifecycle",
        timestamp=datetime.now(),
        importance=0.5,
        metadata={"stage": "initial"},
    )

    # Store
    await vector_store.store_experience(key, context)
    stored = await vector_store.get_experience_by_id(key)
    assert stored is not None
    assert stored.content == context.content

    # Update
    update_data = {"content": "Updated content", "importance": 0.7}
    updated_context = replace(context, **update_data)
    await vector_store.store_experience(key, updated_context)
    retrieved = await vector_store.get_experience_by_id(key)
    assert retrieved is not None
    assert retrieved.content == "Updated content"
    assert retrieved.importance == 0.7

    # Delete
    await vector_store.delete_experience(key)
    deleted = await vector_store.get_experience_by_id(key)
    assert deleted is None


@st.composite
def learning_contexts(draw):
    """A Hypothesis strategy for generating LearningContext objects."""
    content = draw(st.text(min_size=1, max_size=200))
    domain = draw(st.text(min_size=1, max_size=30))
    timestamp = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31),
            timezones=timezones(),
        )
    )
    importance = draw(st.floats(min_value=0.0, max_value=1.0))
    metadata = draw(
        st.dictionaries(
            keys=st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=20
            ).filter(lambda x: x not in ["domain", "timestamp", "importance"]),
            values=st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=50
            ),
            max_size=5,
        )
    )
    return LearningContext(
        content=content,
        domain=domain,
        timestamp=timestamp,
        importance=importance,
        metadata=metadata,
    )


@given(key=st.text(min_size=1, max_size=50), context=learning_contexts())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.asyncio
async def test_property_based_store_and_retrieve(
    vector_store: VectorMemoryStore, key: str, context: LearningContext
):
    """Test that storing and retrieving a LearningContext preserves its properties."""
    await vector_store.store_experience(key, context)
    retrieved = await vector_store.get_experience_by_id(key)
    assert retrieved is not None
    assert retrieved.content == context.content
    assert retrieved.domain == context.domain
    assert retrieved.timestamp.replace(microsecond=0) == context.timestamp.replace(
        microsecond=0
    )
    assert abs(retrieved.importance - context.importance) < 1e-6
    for k, v in context.metadata.items():
        assert k in retrieved.metadata
        assert retrieved.metadata[k] == v
