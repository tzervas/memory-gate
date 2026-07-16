"""Unit tests for memory_gate.metrics Prometheus helpers."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from memory_gate.metrics import (
    _should_include_exception_info,
    record_agent_memory_learned,
    record_agent_task_processed,
    record_consolidation_items_processed,
    record_consolidation_run,
    record_memory_operation,
    start_metrics_server,
)


class TestMetricsHelpers:
    """Test metric recording helper functions."""

    def test_record_memory_operation_success(self) -> None:
        record_memory_operation("store_experience", success=True)
        record_memory_operation("retrieve_context", success=False)

    def test_record_consolidation_helpers(self) -> None:
        record_consolidation_run(success=True)
        record_consolidation_run(success=False)
        record_consolidation_items_processed(3, action="deleted")

    def test_record_agent_helpers(self) -> None:
        record_agent_task_processed("TestAgent", "general", success=True)
        record_agent_task_processed("TestAgent", "general", success=False)
        record_agent_memory_learned("TestAgent", "general")


class TestStartMetricsServer:
    """Test Prometheus metrics server startup."""

    @patch("memory_gate.metrics.start_http_server")
    def test_start_metrics_server_success(self, mock_start: MagicMock) -> None:
        start_metrics_server(port=18008, addr="127.0.0.1")
        mock_start.assert_called_once()

    @patch("memory_gate.metrics.start_http_server", side_effect=OSError("port in use"))
    def test_start_metrics_server_failure(
        self, mock_start: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            start_metrics_server(port=18009, addr="127.0.0.1")
        assert mock_start.called
        assert any("Prometheus metrics server" in r.message for r in caplog.records)

    @patch("memory_gate.metrics.start_http_server", side_effect=OSError("port in use"))
    def test_start_metrics_server_failure_debug_logging(
        self, mock_start: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            start_metrics_server(port=18010, addr="127.0.0.1")
        assert mock_start.called


class TestShouldIncludeExceptionInfo:
    """Test exception info logging helper."""

    def test_returns_false_when_debug_disabled(self) -> None:
        assert _should_include_exception_info() is False

    def test_returns_true_when_debug_enabled(self) -> None:
        test_logger = logging.getLogger("memory_gate.metrics.test")
        with patch.object(test_logger, "isEnabledFor", return_value=True):
            with patch("memory_gate.metrics.logger", test_logger):
                assert _should_include_exception_info() is True