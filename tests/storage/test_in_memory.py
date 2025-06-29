import pytest
import pytest_asyncio # Correct import for async fixtures
import asyncio
from datetime import datetime, timedelta

from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.in_memory import InMemoryKnowledgeStore


@pytest_asyncio.fixture # Use pytest_asyncio.fixture for async fixtures
async def in_memory_store() -> InMemoryKnowledgeStore:
    """Create an empty in-memory knowledge store for each test."""
    store = InMemoryKnowledgeStore()
    await store.clear_storage()
    return store


@pytest.mark.asyncio
async def test_store_and_get_experience_by_key(in_memory_store: InMemoryKnowledgeStore) -> None:
    context = LearningContext(
        content="Test content for key retrieval", domain="test_key", timestamp=datetime.now()
    )
    await in_memory_store.store_experience("test_key_123", context)
    retrieved_context = await in_memory_store.get_experience_by_key("test_key_123")
    assert retrieved_context is not None
    assert retrieved_context.content == "Test content for key retrieval"
    assert await in_memory_store.get_experience_by_key("non_existent_key") is None


@pytest.mark.asyncio
async def test_retrieve_context_basic(in_memory_store: InMemoryKnowledgeStore) -> None:
    ts = datetime.now()
    context1 = LearningContext(content="Alpha project details", domain="projects", timestamp=ts)
    context2 = LearningContext(content="Beta project status", domain="projects", timestamp=ts)
    await in_memory_store.store_experience("alpha1", context1)
    await in_memory_store.store_experience("beta1", context2)

    results = await in_memory_store.retrieve_context("project")
    assert len(results) == 2
    results_alpha = await in_memory_store.retrieve_context("Alpha")
    assert len(results_alpha) == 1
    assert results_alpha[0].content == "Alpha project details"
    results_none = await in_memory_store.retrieve_context("non_existent_query")
    assert len(results_none) == 0


@pytest.mark.asyncio
async def test_retrieve_context_with_limit(in_memory_store: InMemoryKnowledgeStore) -> None:
    now = datetime.now()
    for i in range(5):
        context = LearningContext(
            content=f"Common content item {i}", domain="test_limit", timestamp=now - timedelta(minutes=i)
        )
        await in_memory_store.store_experience(f"limit_key_{i}", context)

    results = await in_memory_store.retrieve_context("Common content", limit=3)
    assert len(results) == 3
    assert "item 0" in results[0].content
    assert "item 1" in results[1].content
    assert "item 2" in results[2].content


@pytest.mark.asyncio
async def test_retrieve_context_with_domain_filter(in_memory_store: InMemoryKnowledgeStore) -> None:
    ts = datetime.now()
    context_infra = LearningContext(content="Server CPU high", domain="infrastructure", timestamp=ts)
    context_app = LearningContext(content="Application bug report", domain="application", timestamp=ts)
    context_infra2 = LearningContext(content="Network latency issue", domain="infrastructure", timestamp=ts)

    await in_memory_store.store_experience("infra1", context_infra)
    await in_memory_store.store_experience("app1", context_app)
    await in_memory_store.store_experience("infra2", context_infra2)

    results_infra = await in_memory_store.retrieve_context("issue", domain_filter="infrastructure")
    assert len(results_infra) == 1
    infra_contents = {res.content for res in results_infra}
    assert "Network latency issue" in infra_contents

    results_app = await in_memory_store.retrieve_context("bug", domain_filter="application")
    assert len(results_app) == 1
    assert results_app[0].content == "Application bug report"
    results_none = await in_memory_store.retrieve_context("issue", domain_filter="non_existent_domain")
    assert len(results_none) == 0


@pytest.mark.asyncio
async def test_retrieve_context_sorting_by_importance_and_timestamp(in_memory_store: InMemoryKnowledgeStore) -> None:
    now = datetime.now()
    s1 = LearningContext(content="Sort item 1", domain="sort_test", timestamp=now - timedelta(hours=1), importance=0.9)
    s2 = LearningContext(content="Sort item 2", domain="sort_test", timestamp=now, importance=0.5)
    s3 = LearningContext(content="Sort item 3", domain="sort_test", timestamp=now - timedelta(minutes=30), importance=0.9)
    s4 = LearningContext(content="Sort item 4", domain="sort_test", timestamp=now - timedelta(minutes=15), importance=0.2)

    await in_memory_store.store_experience("s1", s1)
    await in_memory_store.store_experience("s2", s2)
    await in_memory_store.store_experience("s3", s3)
    await in_memory_store.store_experience("s4", s4)

    results = await in_memory_store.retrieve_context("Sort item", limit=4)
    assert len(results) == 4
    assert results[0].content == "Sort item 3"
    assert results[1].content == "Sort item 1"
    assert results[2].content == "Sort item 2"
    assert results[3].content == "Sort item 4"

@pytest.mark.asyncio
async def test_get_all_experiences_and_clear(in_memory_store: InMemoryKnowledgeStore) -> None:
    assert len(await in_memory_store.get_all_experiences()) == 0
    ts = datetime.now()
    context1 = LearningContext(content="Content A", domain="all", timestamp=ts)
    context2 = LearningContext(content="Content B", domain="all", timestamp=ts)
    await in_memory_store.store_experience("keyA", context1)
    await in_memory_store.store_experience("keyB", context2)
    all_exps = await in_memory_store.get_all_experiences()
    assert len(all_exps) == 2
    await in_memory_store.clear_storage()
    assert len(await in_memory_store.get_all_experiences()) == 0
    retrieved_after_clear = await in_memory_store.get_experience_by_key("keyA")
    assert retrieved_after_clear is None

@pytest.mark.asyncio
async def test_concurrent_storage(in_memory_store: InMemoryKnowledgeStore) -> None:
    num_items = 100
    now = datetime.now()
    tasks = []
    for i in range(num_items):
        context = LearningContext(
            content=f"Concurrent item {i}",
            domain="concurrent_test",
            timestamp=now + timedelta(seconds=i)
        )
        tasks.append(in_memory_store.store_experience(f"concurrent_key_{i}", context))
    await asyncio.gather(*tasks)
    all_experiences = await in_memory_store.get_all_experiences()
    assert len(all_experiences) == num_items
    one_item = await in_memory_store.get_experience_by_key("concurrent_key_50")
    assert one_item is not None
    assert one_item.content == "Concurrent item 50"
