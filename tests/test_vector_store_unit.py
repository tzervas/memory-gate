"""Unit tests for vector_store uncovered paths."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.vector_store import (
    ChromaDBMetadata,
    VectorMemoryStore,
    VectorStoreConfig,
    VectorStoreError,
    VectorStoreInitError,
    VectorStoreOperationError,
)


class TestChromaDBMetadata:
    """Test ChromaDBMetadata validation helpers."""

    def test_valid_metadata(self) -> None:
        meta = ChromaDBMetadata(
            domain="infra",
            timestamp=datetime.now().isoformat(),
            importance=0.5,
            custom_field="value",
        )
        filtered = meta.to_filtered_dict()
        assert "custom_field" in filtered
        assert "domain" not in filtered

    def test_invalid_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            ChromaDBMetadata(timestamp="not-a-date")

    def test_invalid_importance_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid importance value"):
            ChromaDBMetadata(timestamp=datetime.now().isoformat(), importance="bad")  # type: ignore[arg-type]


class TestVectorStoreExceptions:
    """Test custom exception classes."""

    def test_vector_store_error(self) -> None:
        err = VectorStoreError("base error")
        assert err.message == "base error"

    def test_vector_store_init_error(self) -> None:
        err = VectorStoreInitError("init failed")
        assert err.message == "init failed"

    def test_vector_store_operation_error(self) -> None:
        err = VectorStoreOperationError("op failed")
        assert err.message == "op failed"


class TestVectorStoreInitFailures:
    """Test initialization error paths."""

    def test_chromadb_init_failure(self) -> None:
        config = VectorStoreConfig(collection_name="fail", persist_directory="/tmp/x")
        with patch(
            "memory_gate.storage.vector_store.chromadb.PersistentClient",
            side_effect=RuntimeError("chroma down"),
        ):
            with pytest.raises(VectorStoreInitError, match="Failed to initialize ChromaDB"):
                VectorMemoryStore(config=config)

    def test_embedding_model_init_failure(self) -> None:
        config = VectorStoreConfig(collection_name="fail", persist_directory=None)
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = MagicMock()
        with (
            patch("memory_gate.storage.vector_store.chromadb.Client", return_value=mock_client),
            patch(
                "memory_gate.storage.vector_store.SentenceTransformer",
                side_effect=RuntimeError("model missing"),
            ),
        ):
            with pytest.raises(VectorStoreInitError, match="Failed to initialize embedding model"):
                VectorMemoryStore(config=config)


@pytest.fixture
def mocked_vector_store() -> VectorMemoryStore:
    """Vector store with mocked ChromaDB and embedding model."""
    config = VectorStoreConfig(
        collection_name="unit_test_collection",
        persist_directory=None,
    )
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

    with (
        patch("memory_gate.storage.vector_store.chromadb.Client", return_value=mock_client),
        patch(
            "memory_gate.storage.vector_store.SentenceTransformer",
            return_value=mock_model,
        ),
    ):
        store = VectorMemoryStore(config=config)
    store.collection = mock_collection
    store.embedding_model = mock_model
    return store


class TestVectorStoreOperations:
    """Test vector store operation branches with mocks."""

    def test_validate_query_results_invalid(self, mocked_vector_store: VectorMemoryStore) -> None:
        assert mocked_vector_store._validate_query_results({}) is False
        assert mocked_vector_store._validate_query_results({"ids": [[]]}) is False

    def test_parse_metadata_fallback(self, mocked_vector_store: VectorMemoryStore) -> None:
        context = mocked_vector_store._parse_metadata(
            "id-1",
            {"timestamp": "invalid", "domain": "test"},
            "document body",
        )
        assert context.domain == "unknown"
        assert context.content == "document body"

    @pytest.mark.asyncio
    async def test_store_experience_failure(
        self, mocked_vector_store: VectorMemoryStore
    ) -> None:
        mocked_vector_store.collection.upsert.side_effect = RuntimeError("upsert failed")
        context = LearningContext(
            content="fail store",
            domain="test",
            timestamp=datetime.now(),
        )
        with pytest.raises(RuntimeError, match="upsert failed"):
            await mocked_vector_store.store_experience("key", context)

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(
        self, mocked_vector_store: VectorMemoryStore
    ) -> None:
        ts = datetime.now().isoformat()
        mocked_vector_store.collection.query.return_value = {
            "ids": [["id-1"]],
            "documents": [["filtered doc"]],
            "metadatas": [[{"domain": "infra", "timestamp": ts, "importance": 0.7}]],
        }
        results = await mocked_vector_store.retrieve_context(
            query="search",
            domain_filter="infra",
            metadata_filter={"importance": {"$gte": 0.5}},
        )
        assert len(results) == 1
        assert results[0].domain == "infra"

    @pytest.mark.asyncio
    async def test_retrieve_invalid_results_returns_empty(
        self, mocked_vector_store: VectorMemoryStore
    ) -> None:
        mocked_vector_store.collection.query.return_value = {"ids": [[]]}
        results = await mocked_vector_store.retrieve_context(query="empty")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_failure_raises(
        self, mocked_vector_store: VectorMemoryStore
    ) -> None:
        mocked_vector_store.collection.query.side_effect = RuntimeError("query failed")
        with pytest.raises(RuntimeError, match="query failed"):
            await mocked_vector_store.retrieve_context(query="boom")

    @pytest.mark.asyncio
    async def test_get_experience_missing_documents(
        self, mocked_vector_store: VectorMemoryStore
    ) -> None:
        mocked_vector_store.collection.get.return_value = {
            "ids": ["id-1"],
            "documents": [],
            "metadatas": [],
        }
        result = await mocked_vector_store.get_experience_by_id("id-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_experience_failure_raises(
        self, mocked_vector_store: VectorMemoryStore
    ) -> None:
        mocked_vector_store.collection.get.side_effect = RuntimeError("get failed")
        with pytest.raises(RuntimeError, match="get failed"):
            await mocked_vector_store.get_experience_by_id("id-1")

    @pytest.mark.asyncio
    async def test_delete_experience_failure(
        self, mocked_vector_store: VectorMemoryStore
    ) -> None:
        mocked_vector_store.collection.delete.side_effect = RuntimeError("delete failed")
        with pytest.raises(RuntimeError, match="delete failed"):
            await mocked_vector_store.delete_experience("id-1")

    @pytest.mark.asyncio
    async def test_get_experiences_by_metadata_filter(
        self, mocked_vector_store: VectorMemoryStore
    ) -> None:
        ts = datetime.now().isoformat()
        mocked_vector_store.collection.get.return_value = {
            "ids": ["id-1"],
            "documents": ["doc"],
            "metadatas": [{"domain": "infra", "timestamp": ts, "importance": 0.2}],
        }
        items = await mocked_vector_store.get_experiences_by_metadata_filter(
            {"importance": {"$lt": 0.5}}
        )
        assert len(items) == 1
        assert items[0][0] == "id-1"

    @pytest.mark.asyncio
    async def test_get_experiences_by_metadata_filter_failure(
        self, mocked_vector_store: VectorMemoryStore
    ) -> None:
        mocked_vector_store.collection.get.side_effect = RuntimeError("filter failed")
        with pytest.raises(RuntimeError, match="filter failed"):
            await mocked_vector_store.get_experiences_by_metadata_filter({"domain": "x"})