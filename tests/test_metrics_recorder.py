"""Unit tests for MetricsRecorder class.

Tests cover recording, saving, loading, and reporting metrics,
including edge cases like file I/O errors and malformed files.
Follows DRY, SRP, and KISS principles with extensive use of fixtures.
"""

from pathlib import Path

from metrics_recorder import MetricsRecorder
import pytest


@pytest.fixture
def metrics_file(tmp_path: Path) -> str:
    """Provide temporary metrics file path."""
    return str(tmp_path / "test_metrics.json")


@pytest.fixture
def recorder(metrics_file: str) -> MetricsRecorder:
    """Create a MetricsRecorder instance with test file."""
    return MetricsRecorder(metrics_file=metrics_file)


@pytest.fixture
def populated_recorder(recorder: MetricsRecorder) -> MetricsRecorder:
    """Create recorder with sample data."""
    recorder.record_performance_metric("accuracy", 0.95, "decimal")
    recorder.record_quality_metric("test_count", 100)
    recorder.record_timing_metric("duration", 2.5)
    recorder.record_metadata("version", "1.0.0")
    return recorder


@pytest.fixture
def malformed_json_file(metrics_file: str) -> str:
    """Create file with malformed JSON."""
    Path(metrics_file).write_text("not valid json")
    return metrics_file


@pytest.fixture
def mock_io_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock file operations to raise IOError."""

    def mock_open(*args, **kwargs):  # noqa: ARG001
        msg = "Mock I/O error"
        raise OSError(msg)

    monkeypatch.setattr("builtins.open", mock_open)


class TestMetricsRecorderInit:
    """Test MetricsRecorder initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        recorder = MetricsRecorder()
        assert recorder.metrics_file.name == "test_metrics.json"
        assert recorder.display_config["show_performance"] is True
        assert recorder.previous_runs == []

    def test_custom_initialization(self, metrics_file: str) -> None:
        """Test initialization with custom parameters."""
        config = {"show_performance": False, "max_history_runs": 5}
        recorder = MetricsRecorder(metrics_file=metrics_file, display_config=config)
        assert recorder.display_config["show_performance"] is False
        assert recorder.display_config["max_history_runs"] == 5


class TestMetricsRecording:
    """Test metric recording functionality."""

    def test_record_performance_metric(self, recorder: MetricsRecorder) -> None:
        """Test performance metric recording."""
        recorder.record_performance_metric("accuracy", 0.95, "decimal")

        perf_metric = recorder.metrics["performance"]["accuracy"]
        assert perf_metric["value"] == 0.95
        assert perf_metric["unit"] == "decimal"

    def test_record_quality_metric(self, recorder: MetricsRecorder) -> None:
        """Test quality metric recording."""
        recorder.record_quality_metric("errors", 5)
        assert recorder.metrics["quality"]["errors"] == 5

    def test_record_timing_metric(self, recorder: MetricsRecorder) -> None:
        """Test timing metric recording."""
        recorder.record_timing_metric("duration", 2.5)
        assert recorder.metrics["timing"]["duration"] == 2.5

    def test_record_error(self, recorder: MetricsRecorder) -> None:
        """Test error recording."""
        recorder.record_error("test_func", "ValueError", "Invalid input")

        error = recorder.metrics["errors"][0]
        assert error["test"] == "test_func"
        assert error["type"] == "ValueError"
        assert error["message"] == "Invalid input"
        assert "timestamp" in error

    def test_record_metadata(self, recorder: MetricsRecorder) -> None:
        """Test metadata recording."""
        recorder.record_metadata("version", "1.0.0")
        assert recorder.metrics["metadata"]["version"] == "1.0.0"


class TestMetricsPersistence:
    """Test metrics saving and loading."""

    def test_save_and_load_cycle(
        self, populated_recorder: MetricsRecorder, metrics_file: str
    ) -> None:
        """Test complete save and load cycle."""
        populated_recorder.save_metrics()

        new_recorder = MetricsRecorder(metrics_file=metrics_file)
        assert len(new_recorder.previous_runs) == 1

        saved_run = new_recorder.previous_runs[0]
        assert saved_run["performance"]["accuracy"]["value"] == 0.95
        assert saved_run["quality"]["test_count"] == 100
        assert "timestamp" in saved_run

    def test_load_nonexistent_file(self, recorder: MetricsRecorder) -> None:
        """Test loading when file doesn't exist."""
        assert recorder.previous_runs == []

    def test_load_malformed_json(
        self, malformed_json_file: str, capsys: pytest.CaptureFixture
    ) -> None:
        """Test graceful handling of malformed JSON with warning verification."""
        recorder = MetricsRecorder(metrics_file=malformed_json_file)

        # Verify previous_runs becomes empty
        assert recorder.previous_runs == []

        # Verify warning is logged
        captured = capsys.readouterr()
        assert "Warning: Could not load previous metrics:" in captured.out
        assert "JSONDecodeError" in captured.out or "Expecting" in captured.out

    def test_max_history_runs_limit(self, metrics_file: str) -> None:
        """Test history size limitation."""
        recorder = MetricsRecorder(
            metrics_file=metrics_file, display_config={"max_history_runs": 2}
        )

        # Create 3 runs
        for i in range(3):
            recorder.record_timing_metric("test_time", float(i))
            recorder.save_metrics()
            recorder.metrics["timing"].clear()

        # Should only keep last 2
        new_recorder = MetricsRecorder(
            metrics_file=metrics_file, display_config={"max_history_runs": 2}
        )
        assert len(new_recorder.previous_runs) == 2


class TestMetricsErrorHandling:
    """Test error handling in metrics operations."""

    def test_save_io_error(
        self,
        populated_recorder: MetricsRecorder,
        mock_io_error: None,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test graceful handling of save I/O errors with warning verification."""
        # Should not raise exception, just print warning
        populated_recorder.save_metrics()

        # Verify warning is logged
        captured = capsys.readouterr()
        assert "Warning: Could not save metrics:" in captured.out
        assert "Mock I/O error" in captured.out

    def test_load_io_error(
        self,
        metrics_file: str,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test graceful handling of load I/O errors with warning verification."""
        # First create a file so it exists
        Path(metrics_file).write_text('[{"test": "data"}]')

        # Then mock open to raise IOError
        def mock_open(*args, **kwargs):  # noqa: ARG001
            msg = "Mock I/O error"
            raise OSError(msg)

        monkeypatch.setattr("builtins.open", mock_open)

        # Now create recorder which should trigger the I/O error
        recorder = MetricsRecorder(metrics_file=metrics_file)

        # Verify previous_runs becomes empty
        assert recorder.previous_runs == []

        # Verify warning is logged
        captured = capsys.readouterr()
        assert "Warning: Could not load previous metrics:" in captured.out
        assert "Mock I/O error" in captured.out


class TestMetricsAnalysis:
    """Test metrics analysis and reporting."""

    def test_calculate_statistics(self, metrics_file: str) -> None:
        """Test statistics calculation across runs."""
        recorder = MetricsRecorder(metrics_file=metrics_file)

        # Create runs with known values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            recorder.record_timing_metric("test_time", value)
            recorder.save_metrics()
            recorder.metrics["timing"].clear()

        new_recorder = MetricsRecorder(metrics_file=metrics_file)
        stats = new_recorder.calculate_statistics("timing", "test_time")

        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_trend_analysis(self, metrics_file: str) -> None:
        """Test trend detection."""
        recorder = MetricsRecorder(metrics_file=metrics_file)

        # Create increasing trend
        for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
            recorder.record_timing_metric("test_time", value)
            recorder.save_metrics()
            recorder.metrics["timing"].clear()

        new_recorder = MetricsRecorder(metrics_file=metrics_file)
        trend = new_recorder.get_trend("timing", "test_time")
        assert trend == "increasing"

    def test_generate_report(self, populated_recorder: MetricsRecorder) -> None:
        """Test report generation."""
        report = populated_recorder.generate_report()

        assert "TEST METRICS REPORT" in report
        assert "accuracy" in report
        assert "test_count" in report
        assert "duration" in report

    def test_format_duration(self, recorder: MetricsRecorder) -> None:
        """Test duration formatting utility."""
        assert recorder.format_duration(0.5) == "500.0ms"
        assert recorder.format_duration(1.5) == "1.50s"
        assert recorder.format_duration(65.0) == "1m 5.0s"


class TestMetricsConfiguration:
    """Test configuration and utility methods."""

    def test_toggle_display_config(self, recorder: MetricsRecorder) -> None:
        """Test display configuration updates."""
        original = recorder.display_config["show_performance"]
        recorder.toggle_display_config(show_performance=not original)
        assert recorder.display_config["show_performance"] != original

    def test_get_summary_metrics(self, recorder: MetricsRecorder) -> None:
        """Test summary metrics generation."""
        recorder.record_timing_metric("Total Run Time", 10.5)
        recorder.record_error("test1", "Error", "msg1")
        recorder.record_error("test2", "Error", "msg2")
        recorder.record_quality_metric("metric1", "value1")

        summary = recorder.get_summary_metrics()

        assert summary["total_time"] == "10.50s"
        assert summary["error_count"] == 2
        assert summary["quality_metrics_count"] == 1
