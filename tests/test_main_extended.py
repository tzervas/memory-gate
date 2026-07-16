"""Extended unit tests for memory_gate.main uncovered paths."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_gate.main import PassthroughAdapter, main, main_async
from memory_gate.memory_protocols import LearningContext


class TestPassthroughAdapter:
    """Test PassthroughAdapter feedback handling."""

    @pytest.mark.asyncio
    async def test_adapt_with_normalized_feedback(self) -> None:
        adapter = PassthroughAdapter()
        context = LearningContext(
            content="test",
            domain="general",
            timestamp=__import__("datetime").datetime.now(),
            importance=0.8,
        )
        result = await adapter.adapt_knowledge(context, feedback=0.6)
        assert result.importance == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_adapt_with_out_of_range_feedback(self) -> None:
        adapter = PassthroughAdapter()
        context = LearningContext(
            content="test",
            domain="general",
            timestamp=__import__("datetime").datetime.now(),
            importance=0.8,
        )
        result = await adapter.adapt_knowledge(context, feedback=1.5)
        assert result.importance == 1.5


class TestMainAsyncPaths:
    """Test main_async branches not covered by existing tests."""

    @pytest.mark.asyncio
    @patch("memory_gate.main.VectorMemoryStore")
    @patch("memory_gate.main.start_metrics_server")
    @patch("asyncio.sleep", side_effect=asyncio.CancelledError())
    async def test_main_async_consolidation_disabled(
        self,
        mock_sleep: MagicMock,
        mock_start_metrics: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        mock_vector_store.return_value = AsyncMock()
        with patch("memory_gate.main.CONSOLIDATION_ENABLED", new=False):
            await main_async()
        mock_start_metrics.assert_called_once()


class TestMainEntrypoint:
    """Test synchronous main() entrypoint."""

    @patch("memory_gate.main.asyncio.get_event_loop")
    def test_main_keyboard_interrupt(self, mock_get_loop: MagicMock) -> None:
        loop = MagicMock()
        main_task = MagicMock()
        main_task.done.return_value = False
        loop.create_task.return_value = main_task
        loop.run_until_complete.side_effect = KeyboardInterrupt()
        mock_get_loop.return_value = loop

        main()

        main_task.cancel.assert_called_once()