"""Regression test fixtures and utilities."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pytest

from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.vector_store import VectorMemoryStore, VectorStoreConfig

# Test data sets for regression testing
REGRESSION_TEST_DATA = {
    "core_scenarios": [
        {
            "id": "basic_storage",
            "key": "regression_basic_001",
            "context": {
                "content": "Basic storage regression test content",
                "domain": "regression",
                "importance": 0.8,
                "metadata": {"test_type": "basic", "version": "1.0"},
            },
        },
        {
            "id": "complex_metadata",
            "key": "regression_meta_001",
            "context": {
                "content": "Complex metadata test with various field types",
                "domain": "metadata_test",
                "importance": 0.9,
                "metadata": {
                    "string_field": "test_value",
                    "numeric_field": "42",
                    "boolean_field": "true",
                    "category": "regression_test",
                },
            },
        },
        {
            "id": "unicode_content",
            "key": "regression_unicode_001",
            "context": {
                "content": "Unicode test: 你好世界 🌍 Здравствуй мир",
                "domain": "unicode_test",
                "importance": 0.7,
                "metadata": {"encoding": "utf-8", "test_type": "unicode"},
            },
        },
    ],
    "edge_cases": [
        {
            "id": "minimal_content",
            "key": "regression_min_001",
            "context": {
                "content": "X",  # Minimal valid content
                "domain": "minimal",
                "importance": 0.1,
                "metadata": {},
            },
        },
        {
            "id": "max_importance",
            "key": "regression_max_001",
            "context": {
                "content": "Maximum importance test case",
                "domain": "edge_case",
                "importance": 1.0,
                "metadata": {"boundary": "max"},
            },
        },
        {
            "id": "zero_importance",
            "key": "regression_zero_001",
            "context": {
                "content": "Zero importance test case",
                "domain": "edge_case",
                "importance": 0.0,
                "metadata": {"boundary": "min"},
            },
        },
    ],
}


@pytest.fixture
def regression_data() -> dict[str, Any]:
    """Provide standardized regression test data."""
    return REGRESSION_TEST_DATA


@pytest.fixture
def core_test_contexts(
    regression_data: dict[str, Any],
) -> list[tuple[str, LearningContext]]:
    """Generate LearningContext objects for core regression scenarios."""
    contexts = []
    for scenario in regression_data["core_scenarios"]:
        context_data = scenario["context"]
        context = LearningContext(
            content=context_data["content"],
            domain=context_data["domain"],
            timestamp=datetime(
                2024, 1, 1, 12, 0, 0
            ),  # Fixed timestamp for reproducibility
            importance=context_data["importance"],
            metadata=context_data["metadata"],
        )
        contexts.append((scenario["key"], context))
    return contexts


@pytest.fixture
def edge_case_contexts(
    regression_data: dict[str, Any],
) -> list[tuple[str, LearningContext]]:
    """Generate LearningContext objects for edge case regression scenarios."""
    contexts = []
    for scenario in regression_data["edge_cases"]:
        context_data = scenario["context"]
        context = LearningContext(
            content=context_data["content"],
            domain=context_data["domain"],
            timestamp=datetime(
                2024, 1, 1, 12, 0, 0
            ),  # Fixed timestamp for reproducibility
            importance=context_data["importance"],
            metadata=context_data["metadata"],
        )
        contexts.append((scenario["key"], context))
    return contexts


@pytest.fixture
async def isolated_vector_store() -> VectorMemoryStore:
    """Create an isolated vector store for regression testing."""
    config = VectorStoreConfig(
        collection_name="regression_test_collection",
        persist_directory=None,  # Always use in-memory for isolation
        embedding_model_name="all-MiniLM-L6-v2",
    )
    return VectorMemoryStore(config=config)
    # Cleanup happens automatically with in-memory store


@pytest.fixture
def regression_results_path(tmp_path: Path) -> Path:
    """Provide path for regression test results."""
    results_dir = tmp_path / "regression_results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def save_regression_baseline(
    test_name: str, results: dict[str, Any], results_path: Path
) -> None:
    """Save regression test baseline results."""
    baseline_file = results_path / f"{test_name}_baseline.json"
    with baseline_file.open("w") as f:
        json.dump(results, f, indent=2, default=str)


def load_regression_baseline(
    test_name: str, results_path: Path
) -> dict[str, Any] | None:
    """Load regression test baseline results if they exist."""
    baseline_file = results_path / f"{test_name}_baseline.json"
    if baseline_file.exists():
        with baseline_file.open() as f:
            return json.load(f)
    return None
