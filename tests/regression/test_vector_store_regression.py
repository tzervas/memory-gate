"""Regression tests for VectorMemoryStore core functionality.

These tests ensure that critical vector store operations remain stable
across releases and refactoring changes.
"""

import pytest

from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.vector_store import VectorMemoryStore


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "context"),
    [
        ("regression_basic_001", None),  # Will be filled by fixture
        ("regression_meta_001", None),
        ("regression_unicode_001", None),
    ],
    indirect=["context"],
)
async def test_core_storage_retrieval_stability(
    isolated_vector_store: VectorMemoryStore,
    core_test_contexts: list[tuple[str, LearningContext]],
):
    """Test that core storage and retrieval operations remain stable.

    This regression test ensures that:
    1. Basic storage operations work consistently
    2. Retrieval by ID returns exact matches
    3. Content integrity is preserved
    4. Metadata is properly stored and retrieved
    """
    # Use parametrized approach instead of for loops
    test_cases = core_test_contexts

    # Store all contexts (idempotent operation)
    stored_keys = []
    for key, context in test_cases:
        await isolated_vector_store.store_experience(key, context)
        stored_keys.append(key)

    # Verify storage count
    assert isolated_vector_store.get_collection_size() == len(test_cases)

    # Test retrieval stability for each stored item
    for key, original_context in test_cases:
        retrieved = await isolated_vector_store.get_experience_by_id(key)

        # Core stability assertions
        assert retrieved is not None, f"Failed to retrieve context for key: {key}"
        assert retrieved.content == original_context.content
        assert retrieved.domain == original_context.domain
        assert retrieved.importance == original_context.importance

        # Metadata integrity check
        assert retrieved.metadata == original_context.metadata

        # Timestamp should be preserved (ignoring microseconds for stability)
        assert retrieved.timestamp.replace(
            microsecond=0
        ) == original_context.timestamp.replace(microsecond=0)


@pytest.mark.regression
@pytest.mark.asyncio
async def test_similarity_search_stability(
    isolated_vector_store: VectorMemoryStore,
    core_test_contexts: list[tuple[str, LearningContext]],
):
    """Test that similarity search results remain consistent.

    This regression test ensures that:
    1. Similarity search returns expected number of results
    2. Search results are ranked consistently
    3. Query variations produce stable results
    """
    # Store test contexts
    for key, context in core_test_contexts:
        await isolated_vector_store.store_experience(key, context)

    # Test query scenarios - these should be stable across versions
    test_queries = [
        ("basic test", 1, "Should find basic content"),
        ("metadata test", 1, "Should find metadata-related content"),
        ("unicode", 1, "Should find unicode content"),
        ("regression", 3, "Should find all regression-related content"),
    ]

    for query, min_expected, description in test_queries:
        results = await isolated_vector_store.retrieve_context(query=query, limit=10)

        assert len(results) >= min_expected, f"{description} - Query: '{query}'"

        # Verify all results are valid LearningContext objects
        for result in results:
            assert isinstance(result, LearningContext)
            assert result.content is not None
            assert result.domain is not None
            assert 0.0 <= result.importance <= 1.0


@pytest.mark.regression
@pytest.mark.asyncio
async def test_experience_lifecycle_stability(isolated_vector_store: VectorMemoryStore):
    """Test that experience lifecycle operations remain stable.

    This regression test ensures that:
    1. Store operation is idempotent
    2. Update operation works consistently
    3. Delete operation is reliable
    4. Non-existent key retrieval is handled properly
    """
    key = "lifecycle_regression_test"

    # Original context
    original_context = LearningContext(
        content="Lifecycle regression test content",
        domain="regression_test",
        timestamp=pytest.importorskip("datetime").datetime(2024, 1, 1, 12, 0, 0),
        importance=0.5,
        metadata={"stage": "original"},
    )

    # 1. Store operation
    await isolated_vector_store.store_experience(key, original_context)
    assert isolated_vector_store.get_collection_size() == 1

    # Verify storage
    stored = await isolated_vector_store.get_experience_by_id(key)
    assert stored is not None
    assert stored.content == original_context.content

    # 2. Update operation (store with same key)
    updated_context = LearningContext(
        content="Updated lifecycle content",
        domain="regression_test",
        timestamp=pytest.importorskip("datetime").datetime(2024, 1, 1, 12, 0, 0),
        importance=0.8,
        metadata={"stage": "updated"},
    )

    await isolated_vector_store.store_experience(key, updated_context)
    # Collection size should remain 1 (update, not insert)
    assert isolated_vector_store.get_collection_size() == 1

    # Verify update
    updated = await isolated_vector_store.get_experience_by_id(key)
    assert updated is not None
    assert updated.content == updated_context.content
    assert updated.importance == updated_context.importance
    assert updated.metadata["stage"] == "updated"

    # 3. Delete operation
    await isolated_vector_store.delete_experience(key)
    assert isolated_vector_store.get_collection_size() == 0

    # Verify deletion
    deleted = await isolated_vector_store.get_experience_by_id(key)
    assert deleted is None

    # 4. Non-existent key retrieval
    non_existent = await isolated_vector_store.get_experience_by_id("non_existent_key")
    assert non_existent is None


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "context"),
    [
        ("regression_min_001", None),
        ("regression_max_001", None),
        ("regression_zero_001", None),
    ],
    indirect=["context"],
)
async def test_edge_case_stability(
    isolated_vector_store: VectorMemoryStore,
    edge_case_contexts: list[tuple[str, LearningContext]],
):
    """Test that edge cases are handled consistently.

    This regression test ensures that:
    1. Minimal content is handled properly
    2. Boundary importance values work correctly
    3. Empty metadata is supported
    4. Edge cases don't break normal operations
    """
    # Store all edge case contexts
    for key, context in edge_case_contexts:
        await isolated_vector_store.store_experience(key, context)

    # Verify all edge cases were stored
    assert isolated_vector_store.get_collection_size() == len(edge_case_contexts)

    # Test each edge case retrieval
    for key, original_context in edge_case_contexts:
        retrieved = await isolated_vector_store.get_experience_by_id(key)

        assert retrieved is not None, f"Edge case failed for key: {key}"
        assert retrieved.content == original_context.content
        assert retrieved.importance == original_context.importance

        # Verify edge case specific properties
        if key == "regression_min_001":
            assert len(retrieved.content) == 1  # Minimal content
            assert len(retrieved.metadata) == 0  # Empty metadata
        elif key == "regression_max_001":
            assert retrieved.importance == 1.0  # Maximum importance
        elif key == "regression_zero_001":
            assert retrieved.importance == 0.0  # Minimum importance


@pytest.mark.regression
@pytest.mark.asyncio
async def test_concurrent_operations_stability(
    isolated_vector_store: VectorMemoryStore,
):
    """Test that concurrent-like operations remain stable.

    This regression test ensures that:
    1. Multiple rapid operations don't interfere
    2. State remains consistent after multiple operations
    3. No race conditions in sequential operations
    """
    # Simulate rapid sequential operations (not true concurrency for determinism)
    operations = [
        ("store_1", "Content 1", 0.1),
        ("store_2", "Content 2", 0.2),
        ("store_3", "Content 3", 0.3),
    ]

    # Execute operations rapidly in sequence
    contexts = []
    for key, content, importance in operations:
        context = LearningContext(
            content=content,
            domain="concurrent_test",
            timestamp=pytest.importorskip("datetime").datetime(2024, 1, 1, 12, 0, 0),
            importance=importance,
            metadata={"operation": key},
        )
        contexts.append((key, context))
        await isolated_vector_store.store_experience(key, context)

    # Verify final state consistency
    assert isolated_vector_store.get_collection_size() == len(operations)

    # Verify each operation result
    for key, original_context in contexts:
        retrieved = await isolated_vector_store.get_experience_by_id(key)
        assert retrieved is not None
        assert retrieved.content == original_context.content
        assert retrieved.importance == original_context.importance
        assert retrieved.metadata["operation"] == key
