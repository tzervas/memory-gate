"""
Idempotent pytest tests for logging using caplog fixture.

Tests all logger.info, logger.warning, and logger.error calls across the codebase
with parametrization to avoid for-loops and ensure stability across changes.
"""

import asyncio
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_gate.agent_interface import (
    AgentDomain,
    BaseMemoryEnabledAgent,
    SimpleEchoAgent,
)
from memory_gate.consolidation import ConsolidationWorker
from memory_gate.main import main_async, shutdown_handler
from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import KnowledgeStore, LearningContext, MemoryAdapter

# Test data for parametrized logging tests
MAIN_MODULE_LOG_SCENARIOS = (
    # (log_level, expected_message_pattern, test_description)
    ("info", "Initializing MemoryGate System", "system initialization log"),
    (
        "info",
        "Initializing VectorMemoryStore with ChromaDB",
        "vector store initialization log",
    ),
    ("info", "MemoryGateway initialized", "memory gateway initialization log"),
    (
        "info",
        "Initializing ConsolidationWorker",
        "consolidation worker initialization log",
    ),
    ("info", "ConsolidationWorker started", "consolidation worker start log"),
    ("info", "Example agents initialized", "agents initialization log"),
    ("info", "Starting Prometheus metrics server", "metrics server start log"),
    ("info", "MemoryGate System is running", "system ready log"),
    ("info", "Main task cancelled", "main task cancellation log"),
)

CONSOLIDATION_LOG_SCENARIOS = (
    # (log_level, expected_message_pattern, test_description)
    ("info", "Starting consolidation cycle", "consolidation cycle start log"),
    ("debug", "Retrieved", "items retrieval log"),
    ("debug", "Deleted", "batch deletion log"),
    ("info", "Consolidation cycle completed", "consolidation cycle completion log"),
    ("error", "Error during consolidation cycle", "consolidation cycle error log"),
)

AGENT_INTERFACE_LOG_SCENARIOS = (
    # (log_level, expected_message_pattern, test_description)
    ("warning", "Learning failed for agent", "agent learning failure warning"),
    ("error", "Critical learning failure for agent", "agent learning critical error"),
)


class MockAgent(BaseMemoryEnabledAgent):
    """Mock agent for testing logging scenarios."""

    def __init__(self, memory_gateway: MemoryGateway[LearningContext]) -> None:
        super().__init__(
            agent_name="MockAgent",
            domain=AgentDomain.GENERAL,
            memory_gateway=memory_gateway,
        )

    async def _execute_task(
        self, enhanced_context: dict[str, Any]
    ) -> tuple[str, float]:
        """Simple mock task execution."""
        return "Mock result", 0.8


@pytest.fixture
def mock_memory_gateway() -> MemoryGateway[LearningContext]:
    """Create a mock memory gateway for testing."""
    adapter = AsyncMock(spec=MemoryAdapter)
    store = AsyncMock(spec=KnowledgeStore)

    # Create a real MemoryGateway instance but with mocked adapter and store
    gateway = MemoryGateway(adapter, store)

    # Mock the learn_from_interaction method to be an AsyncMock
    gateway.learn_from_interaction = AsyncMock()

    return gateway


@pytest.fixture
def mock_consolidation_worker() -> ConsolidationWorker:
    """Create a mock consolidation worker for testing."""
    store = AsyncMock()
    # Add required methods to the mock store
    store.get_experiences_by_metadata_filter = AsyncMock(return_value=[])
    store.delete_experience = AsyncMock()
    store.store_experience = AsyncMock()
    store.get_collection_size = MagicMock(return_value=0)

    return ConsolidationWorker(store=store, consolidation_interval=1)


@pytest.mark.parametrize(
    ("log_level", "expected_pattern", "description"), MAIN_MODULE_LOG_SCENARIOS
)
@pytest.mark.asyncio
async def test_main_module_logging(
    caplog: pytest.LogCaptureFixture,
    log_level: str,
    expected_pattern: str,
    description: str,
) -> None:
    """
    Test that main module emits expected log messages under correct conditions.

    Args:
        caplog: Pytest fixture for capturing log records
        log_level: Expected log level (info, warning, error)
        expected_pattern: Pattern to match in log message
        description: Test description for clarity
    """
    caplog.set_level(getattr(logging, log_level.upper()))

    # Mock all external dependencies to isolate logging behavior
    with (
        patch("memory_gate.main.VectorMemoryStore") as mock_store,
        patch("memory_gate.main.ConsolidationWorker") as mock_worker,
        patch("memory_gate.main.start_metrics_server"),
        patch("asyncio.sleep", side_effect=asyncio.CancelledError()),
        patch("asyncio.create_task") as mock_create_task,
        patch("memory_gate.main.CONSOLIDATION_ENABLED", new=True),
    ):
        # Configure mocks for different scenarios
        mock_store.return_value = AsyncMock()
        mock_worker.return_value = AsyncMock()
        mock_create_task.return_value = AsyncMock()

        import contextlib

        with contextlib.suppress(asyncio.CancelledError):
            await main_async()

    # Assert that the expected log message was emitted
    matching_records = [
        record
        for record in caplog.records
        if record.levelname.lower() == log_level.lower()
        and expected_pattern.lower() in record.message.lower()
    ]

    assert matching_records, (
        f"Expected {log_level} log with pattern '{expected_pattern}' not found. Description: {description}. Available {log_level} logs: {[r.message for r in caplog.records if r.levelname.lower() == log_level.lower()]}"
    )


@pytest.mark.parametrize(
    ("log_level", "expected_pattern", "description"), CONSOLIDATION_LOG_SCENARIOS
)
@pytest.mark.asyncio
async def test_consolidation_logging(
    caplog: pytest.LogCaptureFixture,
    mock_consolidation_worker: ConsolidationWorker,
    log_level: str,
    expected_pattern: str,
    description: str,
) -> None:
    """
    Test that consolidation module emits expected log messages.

    Args:
        caplog: Pytest fixture for capturing log records
        mock_consolidation_worker: Mock consolidation worker
        log_level: Expected log level (info, warning, error, debug)
        expected_pattern: Pattern to match in log message
        description: Test description for clarity
    """
    caplog.set_level(getattr(logging, log_level.upper()))

    # Configure mock store behavior for different scenarios
    if "error" in description.lower():
        # Simulate error conditions
        mock_consolidation_worker.store.get_experiences_by_metadata_filter.side_effect = Exception(
            "Test error"
        )
    else:
        # Normal operation
        mock_consolidation_worker.store.get_experiences_by_metadata_filter.return_value = [
            (
                "key1",
                LearningContext(
                    content="Test content",
                    domain="test",
                    timestamp=datetime.now() - timedelta(days=35),
                    importance=0.1,
                ),
            )
        ]

    import contextlib

    with contextlib.suppress(Exception):
        await mock_consolidation_worker._perform_consolidation()

    # Assert that the expected log message was emitted
    matching_records = [
        record
        for record in caplog.records
        if record.levelname.lower() == log_level.lower()
        and any(
            pattern in record.message.lower()
            for pattern in expected_pattern.lower().split("\\d+")
        )
    ]

    assert matching_records, (
        f"Expected {log_level} log with pattern '{expected_pattern}' not found. Description: {description}. Available {log_level} logs: {[r.message for r in caplog.records if r.levelname.lower() == log_level.lower()]}"
    )


@pytest.mark.parametrize(
    ("log_level", "expected_pattern", "description"), AGENT_INTERFACE_LOG_SCENARIOS
)
@pytest.mark.asyncio
async def test_agent_interface_logging(
    caplog: pytest.LogCaptureFixture,
    mock_memory_gateway: MemoryGateway[LearningContext],
    log_level: str,
    expected_pattern: str,
    description: str,
) -> None:
    """
    Test that agent interface emits expected log messages under error conditions.

    Args:
        caplog: Pytest fixture for capturing log records
        mock_memory_gateway: Mock memory gateway
        log_level: Expected log level (warning, error)
        expected_pattern: Pattern to match in log message
        description: Test description for clarity
    """
    caplog.set_level(getattr(logging, log_level.upper()))

    agent = MockAgent(mock_memory_gateway)

    # Configure mock to simulate learning failures
    if "critical" in description.lower():
        mock_memory_gateway.learn_from_interaction.side_effect = Exception(
            "Critical learning error"
        )
    else:
        mock_memory_gateway.learn_from_interaction.side_effect = ValueError(
            "Learning failed"
        )

    # Process a task to trigger learning and potential logging
    await agent.process_task("Test task input", store_interaction_memory=True)

    # Assert that the expected log message was emitted
    matching_records = [
        record
        for record in caplog.records
        if record.levelname.lower() == log_level.lower()
        and expected_pattern.lower() in record.message.lower()
    ]

    assert matching_records, (
        f"Expected {log_level} log with pattern '{expected_pattern}' not found. Description: {description}. Available {log_level} logs: {[r.message for r in caplog.records if r.levelname.lower() == log_level.lower()]}"
    )


@pytest.mark.asyncio
async def test_shutdown_handler_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Test that shutdown handler emits expected log messages."""
    caplog.set_level(logging.INFO)

    mock_loop = AsyncMock()
    mock_worker = AsyncMock()
    mock_worker.is_running = MagicMock(return_value=True)

    # Create a test task to cancel
    task = asyncio.create_task(asyncio.sleep(1))

    with patch("memory_gate.main.background_tasks", [task]):
        await shutdown_handler(mock_loop, mock_worker)

    # Check for shutdown initiation log
    shutdown_logs = [
        record
        for record in caplog.records
        if record.levelname == "INFO" and "graceful shutdown" in record.message.lower()
    ]
    assert shutdown_logs, "Expected shutdown initiation log not found"

    # Check for worker stop log
    worker_stop_logs = [
        record
        for record in caplog.records
        if record.levelname == "INFO"
        and "stopping consolidationworker" in record.message.lower()
    ]
    assert worker_stop_logs, "Expected worker stop log not found"

    # Check for background tasks stopped log
    tasks_stopped_logs = [
        record
        for record in caplog.records
        if record.levelname == "INFO"
        and "background tasks stopped" in record.message.lower()
    ]
    assert tasks_stopped_logs, "Expected background tasks stopped log not found"


@pytest.mark.asyncio
async def test_consolidation_error_logging_scenarios(
    caplog: pytest.LogCaptureFixture,
    mock_consolidation_worker: ConsolidationWorker,
) -> None:
    """Test specific error scenarios in consolidation to ensure error logging."""
    caplog.set_level(logging.ERROR)

    # Test scenario where store methods are missing
    mock_store = AsyncMock()
    # Intentionally don't add required methods to simulate missing methods
    worker_no_methods = ConsolidationWorker(store=mock_store, consolidation_interval=1)

    # This should complete without error but log appropriately
    await worker_no_methods._perform_consolidation()

    # Test scenario with exception during consolidation
    mock_consolidation_worker.store.get_experiences_by_metadata_filter.side_effect = (
        RuntimeError("Database error")
    )

    with pytest.raises(RuntimeError):
        await mock_consolidation_worker._perform_consolidation()

    error_logs = [
        record
        for record in caplog.records
        if record.levelname == "ERROR" and "consolidation" in record.message.lower()
    ]
    assert error_logs, "Expected consolidation error log not found"


@pytest.mark.asyncio
async def test_agent_task_execution_logging_scenarios(
    caplog: pytest.LogCaptureFixture,
    mock_memory_gateway: MemoryGateway[LearningContext],
) -> None:
    """Test logging scenarios during agent task execution."""
    caplog.set_level(logging.WARNING)

    # Test successful learning (should not produce warning/error logs)
    mock_memory_gateway.learn_from_interaction.return_value = LearningContext(
        content="test", domain="test", timestamp=datetime.now()
    )

    agent = MockAgent(mock_memory_gateway)
    result, confidence = await agent.process_task("Test successful task")

    assert confidence > 0, "Expected successful task execution"

    # Explicitly assert that no warning or error logs are present for successful execution
    assert all(record.levelno < logging.WARNING for record in caplog.records), (
        "No warning or error logs should be present during successful learning"
    )

    # Clear logs for next test
    caplog.clear()

    # Test learning failure scenarios
    test_scenarios = [
        (ValueError("Learning value error"), "WARNING", "learning failed"),
        (RuntimeError("Learning runtime error"), "WARNING", "learning failed"),
        (Exception("Critical learning error"), "ERROR", "critical learning failure"),
    ]

    for exception, expected_level, expected_message in test_scenarios:
        caplog.clear()
        mock_memory_gateway.learn_from_interaction.side_effect = exception

        await agent.process_task("Test task with learning failure")

        matching_logs = [
            record
            for record in caplog.records
            if record.levelname == expected_level
            and expected_message in record.message.lower()
        ]

        assert matching_logs, (
            f"Expected {expected_level} log with '{expected_message}' not found for {exception.__class__.__name__}"
        )


@pytest.mark.asyncio
async def test_consolidation_loop_error_logging(
    caplog: pytest.LogCaptureFixture,
    mock_consolidation_worker: ConsolidationWorker,
) -> None:
    """Test error logging in consolidation loop."""
    caplog.set_level(logging.ERROR)

    # Mock the consolidation to raise an exception
    mock_consolidation_worker._perform_consolidation = AsyncMock(
        side_effect=Exception("Loop error")
    )

    # Start the worker and let it run briefly
    await mock_consolidation_worker.start()
    await asyncio.sleep(0.1)  # Allow time for error to occur
    await mock_consolidation_worker.stop()

    # Check for loop error log
    loop_error_logs = [
        record
        for record in caplog.records
        if record.levelname == "ERROR"
        and "consolidation loop" in record.message.lower()
    ]
    assert loop_error_logs, "Expected consolidation loop error log not found"


# Integration test to ensure logging works across component interactions
@pytest.mark.asyncio
async def test_logging_integration_stability(
    caplog: pytest.LogCaptureFixture,
    temp_chroma_directory: Path,
) -> None:
    """
    Integration test to ensure logging remains stable across component interactions.

    This test verifies that logging works correctly when components interact,
    ensuring the tests remain stable across codebase changes.
    """
    caplog.set_level(logging.INFO)

    # Test with real components to ensure logging integration works
    from memory_gate.main import PassthroughAdapter
    from memory_gate.memory_gateway import MemoryGateway
    from memory_gate.storage.vector_store import VectorMemoryStore, VectorStoreConfig

    # Create real components
    config = VectorStoreConfig(
        collection_name="test_logging_integration",
        persist_directory=str(temp_chroma_directory),
    )
    store = VectorMemoryStore(config=config)
    adapter = PassthroughAdapter()
    gateway = MemoryGateway(adapter=adapter, store=store)

    # Create and use an agent
    agent = SimpleEchoAgent(gateway)

    # Process a task - this should generate logs
    result, confidence = await agent.process_task("Integration test task")

    # Verify we got expected logs without being too specific about exact messages
    # This ensures test stability across code changes
    info_logs = [record for record in caplog.records if record.levelname == "INFO"]

    # Should have some info logs from the process
    assert info_logs, "Expected some info logs during integration test"

    # Verify the agent actually worked
    assert result is not None
    assert confidence >= 0

    # Clean up
    if hasattr(store, "client") and hasattr(store.client, "stop"):
        store.client.stop()


# Test to ensure log message consistency and format stability
@pytest.mark.parametrize(
    ("module_name", "logger_name"),
    [
        ("memory_gate.main", "memory_gate.main"),
        ("memory_gate.consolidation", "memory_gate.consolidation"),
        ("memory_gate.agent_interface", "memory_gate.agent_interface"),
    ],
)
def test_logger_configuration_stability(logger_name: str) -> None:
    """
    Test that loggers are properly configured and maintain consistent naming.

    This ensures that logger configuration remains stable across changes.
    """
    logger = logging.getLogger(logger_name)

    # Verify logger exists and has expected name
    assert logger.name == logger_name

    # Verify logger can handle different levels (test with INFO and above since that's the default)
    # In production, loggers are typically set to INFO level by default
    assert logger.isEnabledFor(logging.INFO)
    assert logger.isEnabledFor(logging.WARNING)
    assert logger.isEnabledFor(logging.ERROR)

    # Verify logger has proper hierarchy
    assert logger.parent is not None or logger.name == "root"
