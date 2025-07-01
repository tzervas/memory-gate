import asyncio
import gc
import logging
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

from metrics_recorder import MetricsRecorder
import pytest
import pytest_asyncio

from memory_gate.storage.vector_store import VectorMemoryStore, VectorStoreConfig

# Initialize metrics recorder with configurable display
metrics_recorder = MetricsRecorder(
    metrics_file="test_metrics.json",
    display_config={
        "show_performance": True,
        "show_quality": True,
        "show_timing": True,
        "show_comparison": True,
        "show_trends": True,
        "max_history_runs": 10,
        "detailed_breakdown": False,
    },
)

# Global counters
passed_tests = 0
failed_tests = 0
skipped_tests = 0
error_tests = 0


def pytest_sessionstart(session):
    """Record session start time and metadata."""
    session.config.start_time = time.time()

    # Record metadata about the test environment
    metrics_recorder.record_metadata(
        "python_version",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    metrics_recorder.record_metadata("platform", platform.system())
    metrics_recorder.record_metadata("architecture", platform.machine())
    metrics_recorder.record_metadata("pytest_args", " ".join(session.config.args))


def pytest_sessionfinish(session, exitstatus):
    """Record final metrics and generate report."""
    total_time = time.time() - session.config.start_time

    # Record timing metrics
    metrics_recorder.record_timing_metric("Total Run Time", total_time)

    # Record quality metrics
    total_tests = passed_tests + failed_tests + skipped_tests + error_tests
    if total_tests > 0:
        pass_rate = (passed_tests / total_tests) * 100
        metrics_recorder.record_quality_metric("Pass Rate", f"{pass_rate:.1f}%")
        metrics_recorder.record_quality_metric("Total Tests", total_tests)
        metrics_recorder.record_quality_metric("Passed Tests", passed_tests)
        metrics_recorder.record_quality_metric("Failed Tests", failed_tests)
        metrics_recorder.record_quality_metric("Skipped Tests", skipped_tests)
        metrics_recorder.record_quality_metric("Error Tests", error_tests)

    # Record performance metrics
    if total_tests > 0:
        avg_test_time = total_time / total_tests
        metrics_recorder.record_performance_metric(
            "Average Test Duration", avg_test_time, "seconds"
        )

    metrics_recorder.record_metadata("exit_status", exitstatus)

    # Save and display report
    metrics_recorder.save_metrics()
    print(metrics_recorder.generate_report())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Record individual test timing."""
    start = time.time()
    _outcome = yield  # TODO: Use outcome for future test result analysis and metrics
    duration = time.time() - start
    metrics_recorder.record_timing_metric(f"{item.name}_duration", duration)
    # Future enhancement: outcome.get_result() can be used for detailed test analysis


def pytest_runtest_logreport(report):
    """Track test outcomes for quality metrics."""
    global passed_tests, failed_tests, skipped_tests, error_tests

    if report.when == "call":
        if report.outcome == "passed":
            passed_tests += 1
        elif report.outcome == "failed":
            failed_tests += 1
            # Record the failure
            metrics_recorder.record_error(
                test_name=report.nodeid,
                error_type="test_failure",
                message=str(report.longrepr) if report.longrepr else "Unknown failure",
            )
        elif report.outcome == "skipped":
            skipped_tests += 1
    elif report.when == "setup" and report.outcome == "failed":
        error_tests += 1
        metrics_recorder.record_error(
            test_name=report.nodeid,
            error_type="setup_error",
            message=str(report.longrepr) if report.longrepr else "Setup failed",
        )


@pytest.fixture
def temp_chroma_directory() -> Path:
    """Create a temporary directory for ChromaDB persistence for a test that cleans up, even on Windows."""
    path = tempfile.mkdtemp()
    yield Path(path)
    # HACK: Force cleanup on Windows, which can have file locking issues.
    if (
        hasattr(sys, "_getframe")
        and sys.platform == "win32"
        and sys._getframe(1).f_code.co_name == "finish"
    ):
        gc.collect()
        time.sleep(1)

    shutil.rmtree(path, ignore_errors=True)


@pytest_asyncio.fixture
async def persistent_vector_store(
    temp_chroma_directory: Path,
) -> VectorMemoryStore:
    """Create a VectorMemoryStore with persistence for testing."""
    config = VectorStoreConfig(
        collection_name="test_persistent_collection",
        persist_directory=str(temp_chroma_directory),
        embedding_model_name="all-MiniLM-L6-v2",
    )
    store = VectorMemoryStore(config=config)
    yield store
    if hasattr(store, "client") and hasattr(store.client, "stop"):
        store.client.stop()
    del store
    gc.collect()
    if sys.platform == "win32":
        await asyncio.sleep(0.1)


@pytest_asyncio.fixture
async def in_memory_vector_store() -> VectorMemoryStore:
    """Create an in-memory VectorMemoryStore for testing."""
    config = VectorStoreConfig(
        collection_name="test_in_memory_collection",
        persist_directory=None,  # In-memory
        embedding_model_name="all-MiniLM-L6-v2",
    )
    store = VectorMemoryStore(config=config)
    yield store
    if hasattr(store, "client") and hasattr(store.client, "stop"):
        store.client.stop()
    del store
    gc.collect()
    if sys.platform == "win32":
        await asyncio.sleep(0.1)
