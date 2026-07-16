"""Tests for memory_gate.embedding_catalog."""

import pytest

from memory_gate.embedding_catalog import (
    SUPPORTED_STABLE_IDS,
    CatalogEntry,
    resolve_model,
)


class TestResolveModel:
    def test_stable_id_bge_small(self) -> None:
        entry = resolve_model("bge-small-en-v1.5")
        assert entry.st_name == "BAAI/bge-small-en-v1.5"
        assert entry.stable_id == "bge-small-en-v1.5"
        assert entry.dimension == 384

    def test_stable_id_minilm(self) -> None:
        entry = resolve_model("all-minilm-l6-v2")
        assert entry.st_name == "all-MiniLM-L6-v2"
        assert entry.dimension == 384

    def test_sentence_transformers_name_minilm(self) -> None:
        entry = resolve_model("all-MiniLM-L6-v2")
        assert entry.stable_id == "all-minilm-l6-v2"

    def test_hf_prefixed_bge(self) -> None:
        entry = resolve_model("BAAI/bge-base-en-v1.5")
        assert entry.stable_id == "bge-base-en-v1.5"
        assert entry.dimension == 768

    def test_sentence_transformers_prefix(self) -> None:
        entry = resolve_model("sentence-transformers/all-MiniLM-L6-v2")
        assert entry.stable_id == "all-minilm-l6-v2"

    def test_aliases(self) -> None:
        assert resolve_model("minilm").stable_id == "all-minilm-l6-v2"
        assert resolve_model("bge-small").stable_id == "bge-small-en-v1.5"
        assert resolve_model("bge-base").stable_id == "bge-base-en-v1.5"

    def test_unknown_lists_supported_ids(self) -> None:
        with pytest.raises(ValueError, match="unknown embedding model"):
            resolve_model("not-a-model")
        with pytest.raises(ValueError) as exc_info:
            resolve_model("not-a-model")
        for sid in SUPPORTED_STABLE_IDS:
            assert sid in str(exc_info.value)

    def test_catalog_entry_frozen(self) -> None:
        entry = resolve_model("all-minilm-l6-v2")
        assert isinstance(entry, CatalogEntry)