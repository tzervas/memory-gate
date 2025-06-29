import pytest
from datetime import datetime
import shutil # Keep if other fixtures in this file might need it, otherwise remove.
import tempfile # Keep if other fixtures in this file might need it, otherwise remove.
from pathlib import Path # Keep if other fixtures in this file might need it, otherwise remove.
import pytest_asyncio # Keep if other async fixtures remain, otherwise remove.

from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.vector_store import VectorMemoryStore

# Fixtures `temp_chroma_directory`, `persistent_vector_store`,
# and `in_memory_vector_store` are now in tests/conftest.py


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_store_fixture_name", ["persistent_vector_store", "in_memory_vector_store"]
)
async def test_store_and_retrieve_single_experience(
    vector_store_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test storing and retrieving a single learning experience."""
    vector_store: VectorMemoryStore = request.getfixturevalue(vector_store_fixture_name)

    context = LearningContext(
        content="Test infrastructure issue: high CPU on web-01",
        domain="infrastructure",
        timestamp=datetime.now(),
        importance=0.85,
        metadata={"server": "web-01", "metric": "cpu_utilization"},
    )
    key = "infra_cpu_web01"

    await vector_store.store_experience(key, context)
    assert vector_store.get_collection_size() == 1

    retrieved_contexts = await vector_store.retrieve_context(
        query="CPU issue on web server", limit=1
    )

    assert len(retrieved_contexts) == 1
    retrieved_context = retrieved_contexts[0]
    assert retrieved_context.content == context.content
    assert retrieved_context.domain == context.domain
    assert retrieved_context.importance == context.importance
    assert retrieved_context.metadata == context.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_store_fixture_name", ["persistent_vector_store", "in_memory_vector_store"]
)
async def test_retrieve_context_with_domain_filter(
    vector_store_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test retrieving context with a domain filter."""
    vector_store: VectorMemoryStore = request.getfixturevalue(vector_store_fixture_name)

    context_infra = LearningContext(
        content="Server reboot procedure",
        domain="infrastructure",
        timestamp=datetime.now(),
    )
    context_code = LearningContext(
        content="Python style guide", domain="code_review", timestamp=datetime.now()
    )
    await vector_store.store_experience("infra_reboot", context_infra)
    await vector_store.store_experience("code_python_style", context_code)

    retrieved_contexts = await vector_store.retrieve_context(
        query="procedure style guide", limit=5, domain_filter="infrastructure"
    )
    assert len(retrieved_contexts) == 1
    assert retrieved_contexts[0].domain == "infrastructure"
    assert retrieved_contexts[0].content == context_infra.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_store_fixture_name", ["persistent_vector_store", "in_memory_vector_store"]
)
async def test_get_experience_by_id(
    vector_store_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test retrieving a specific experience by its ID."""
    vector_store: VectorMemoryStore = request.getfixturevalue(vector_store_fixture_name)

    context = LearningContext(
        content="Unique content for ID test", domain="test", timestamp=datetime.now()
    )
    key = "unique_id_123"
    await vector_store.store_experience(key, context)

    retrieved_context = await vector_store.get_experience_by_id(key)
    assert retrieved_context is not None
    assert retrieved_context.content == context.content

    non_existent_context = await vector_store.get_experience_by_id("non_existent_key")
    assert non_existent_context is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_store_fixture_name", ["persistent_vector_store", "in_memory_vector_store"]
)
async def test_delete_experience(
    vector_store_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test deleting an experience."""
    vector_store: VectorMemoryStore = request.getfixturevalue(vector_store_fixture_name)

    context = LearningContext(
        content="Content to be deleted", domain="test", timestamp=datetime.now()
    )
    key = "delete_me"
    await vector_store.store_experience(key, context)
    assert vector_store.get_collection_size() == 1

    retrieved_before_delete = await vector_store.get_experience_by_id(key)
    assert retrieved_before_delete is not None

    await vector_store.delete_experience(key)
    assert vector_store.get_collection_size() == 0

    retrieved_after_delete = await vector_store.get_experience_by_id(key)
    assert retrieved_after_delete is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_store_fixture_name", ["persistent_vector_store", "in_memory_vector_store"]
)
async def test_store_multiple_and_limit_retrieval(
    vector_store_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test storing multiple experiences and limiting retrieval."""
    vector_store: VectorMemoryStore = request.getfixturevalue(vector_store_fixture_name)

    for i in range(5):
        ctx = LearningContext(
            content=f"Common content item {i}",
            domain="test_limit",
            timestamp=datetime.now(),
            importance=0.1 * i,
        )
        await vector_store.store_experience(f"limit_key_{i}", ctx)

    assert vector_store.get_collection_size() == 5

    # Test limit
    retrieved_limit_3 = await vector_store.retrieve_context("Common content", limit=3)
    assert len(retrieved_limit_3) == 3

    # Test limit greater than available
    retrieved_limit_10 = await vector_store.retrieve_context("Common content", limit=10)
    assert len(retrieved_limit_10) == 5  # Max available that match


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_store_fixture_name", ["persistent_vector_store", "in_memory_vector_store"]
)
async def test_empty_query_retrieval(
    vector_store_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test behavior with an empty query string."""
    vector_store: VectorMemoryStore = request.getfixturevalue(vector_store_fixture_name)
    # Depending on ChromaDB's behavior with empty query embeddings, this might return all or error.
    # SentenceTransformer will likely produce a valid embedding for an empty string.
    # This test is to document and verify the behavior.
    context = LearningContext(
        content="Some content", domain="test", timestamp=datetime.now()
    )
    await vector_store.store_experience("some_key", context)

    # It's expected that an empty query might not be very useful,
    # but it shouldn't crash the system.
    # The result might be an empty list or some arbitrary documents.
    try:
        retrieved_contexts = await vector_store.retrieve_context(query="", limit=5)
        # Assert that it doesn't raise an exception and returns a list (possibly empty)
        assert isinstance(retrieved_contexts, list)
    except Exception as e:
        pytest.fail(f"Empty query string retrieval raised an exception: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_store_fixture_name", ["persistent_vector_store", "in_memory_vector_store"]
)
async def test_no_results_retrieval(
    vector_store_fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test retrieval when no documents match the query."""
    vector_store: VectorMemoryStore = request.getfixturevalue(vector_store_fixture_name)
    context = LearningContext(
        content="Specific content", domain="test", timestamp=datetime.now()
    )
    await vector_store.store_experience("specific_key", context)

    retrieved_contexts = await vector_store.retrieve_context(
        query="completely unrelated query string", limit=5
    )
    assert len(retrieved_contexts) == 0


# It might be beneficial to add a test for metadata_filter once its usage is more defined.
# For example:
# @pytest.mark.asyncio
# async def test_retrieve_context_with_metadata_filter(vector_store: VectorMemoryStore):
#     context1 = LearningContext(content="Content A", domain="filter_test", timestamp=datetime.now(), metadata={"type": "alpha"})
#     context2 = LearningContext(content="Content B", domain="filter_test", timestamp=datetime.now(), metadata={"type": "beta"})
#     await vector_store.store_experience("key_A", context1)
#     await vector_store.store_experience("key_B", context2)
#
#     # Example: find where metadata.type == "alpha"
#     results = await vector_store.retrieve_context("Content", limit=5, metadata_filter={"type": "alpha"})
#     assert len(results) == 1
#     assert results[0].metadata["type"] == "alpha"

# Note: Running tests that download models (like sentence-transformers) can be slow on first run.
# For CI, consider pre-caching these models or using mocked embedding functions for pure unit tests
# if speed becomes an issue. These tests are more integration-oriented.
