import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest

from memory_gate.main import main_async, shutdown_handler


@pytest.mark.asyncio
@pytest.mark.timeout(10)  # 10 second timeout for initialization test
@patch("memory_gate.main.VectorMemoryStore")
@patch("memory_gate.main.ConsolidationWorker")
@patch("memory_gate.main.start_metrics_server")
@patch("asyncio.sleep")  # Mock the infinite sleep loop
async def test_main_async_initialization(
    mock_sleep,
    mock_start_metrics_server,
    mock_consolidation_worker,
    mock_vector_memory_store,
):
    """Test that main_async initializes all components as expected."""
    # Mock the infinite sleep to raise CancelledError after initialization
    mock_sleep.side_effect = asyncio.CancelledError()

    with patch("asyncio.create_task") as mock_create_task:
        logging.info("Starting main_async initialization test")
        try:
            await main_async()
        except asyncio.CancelledError:
            logging.info("main_async cancelled as expected (infinite loop interrupted)")
        logging.info("Completed main_async initialization test")

    mock_vector_memory_store.assert_called_once()
    mock_consolidation_worker.assert_called_once()
    mock_create_task.assert_called_once()
    mock_start_metrics_server.assert_called_once()
    logging.info("All mock objects were called as expected")


@pytest.mark.asyncio
async def test_shutdown_handler():
    """Test that the shutdown handler correctly cancels tasks."""
    logging.info("Starting shutdown handler test")
    mock_loop = AsyncMock()
    mock_worker = AsyncMock()
    task = asyncio.create_task(asyncio.sleep(1))

    with patch("memory_gate.main.background_tasks", [task]):
        await shutdown_handler(mock_loop, mock_worker)

    assert task.cancelled()
    mock_worker.stop.assert_awaited_once()
    logging.info("Shutdown handler test passed")
