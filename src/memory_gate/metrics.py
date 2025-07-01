from prometheus_client import (  # type: ignore[import-not-found]
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    start_http_server,
)
import time

# Create a custom registry (optional, but good practice for managing metrics)
# If not using a custom registry, prometheus_client uses a global default registry.
REGISTRY = CollectorRegistry()

# --- Define Metrics ---

# Memory Operations
MEMORY_OPERATIONS_TOTAL = Counter(
    "memory_gate_operations_total",
    "Total number of memory operations.",
    [
        "operation_type",
        "status",
    ],  # e.g., operation_type="store_experience", status="success"
    registry=REGISTRY,
)

MEMORY_STORE_LATENCY_SECONDS = Histogram(
    "memory_gate_store_latency_seconds",
    "Latency of storing experiences in the memory store.",
    ["store_type"],  # e.g., store_type="vector_store"
    registry=REGISTRY,
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
    ),  # Buckets in seconds
)

MEMORY_RETRIEVAL_LATENCY_SECONDS = Histogram(
    "memory_gate_retrieval_latency_seconds",
    "Latency of retrieving context from the memory store.",
    ["store_type"],
    registry=REGISTRY,
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
    ),
)

MEMORY_ITEMS_COUNT = Gauge(
    "memory_gate_items_count",
    "Number of items currently in the memory store.",
    ["store_type", "collection_name"],
    registry=REGISTRY,
)

# Consolidation Worker Metrics
CONSOLIDATION_RUNS_TOTAL = Counter(
    "memory_gate_consolidation_runs_total",
    "Total number of consolidation runs.",
    ["status"],  # e.g., status="success", status="failure"
    registry=REGISTRY,
)

CONSOLIDATION_DURATION_SECONDS = Histogram(
    "memory_gate_consolidation_duration_seconds",
    "Duration of memory consolidation runs.",
    registry=REGISTRY,
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),  # Up to 1 hour
)

CONSOLIDATION_ITEMS_PROCESSED_TOTAL = Counter(
    "memory_gate_consolidation_items_processed_total",
    "Total number of items processed during consolidation.",
    ["action"],  # e.g., action="deleted", action="merged", action="updated_importance"
    registry=REGISTRY,
)

# Agent Interaction Metrics
AGENT_TASKS_PROCESSED_TOTAL = Counter(
    "memory_gate_agent_tasks_processed_total",
    "Total number of tasks processed by agents.",
    ["agent_name", "agent_domain", "status"],  # status="success", "failure"
    registry=REGISTRY,
)

AGENT_TASK_DURATION_SECONDS = Histogram(
    "memory_gate_agent_task_duration_seconds",
    "Duration of agent task processing.",
    ["agent_name", "agent_domain"],
    registry=REGISTRY,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

AGENT_MEMORY_LEARNED_TOTAL = Counter(
    "memory_gate_agent_memory_learned_total",
    "Total number of memories learned by agents.",
    ["agent_name", "agent_domain"],
    registry=REGISTRY,
)

# --- Helper Functions to Update Metrics ---
# These can be called from relevant parts of the application code.


def record_memory_operation(operation_type: str, success: bool = True) -> None:
    status = "success" if success else "failure"
    MEMORY_OPERATIONS_TOTAL.labels(operation_type=operation_type, status=status).inc()


def record_consolidation_run(success: bool = True) -> None:
    status = "success" if success else "failure"
    CONSOLIDATION_RUNS_TOTAL.labels(status=status).inc()


def record_consolidation_items_processed(count: int, action: str) -> None:
    CONSOLIDATION_ITEMS_PROCESSED_TOTAL.labels(action=action).inc(count)


def record_agent_task_processed(
    agent_name: str, agent_domain: str, success: bool = True
) -> None:
    status = "success" if success else "failure"
    AGENT_TASKS_PROCESSED_TOTAL.labels(
        agent_name=agent_name, agent_domain=agent_domain, status=status
    ).inc()


def record_agent_memory_learned(agent_name: str, agent_domain: str) -> None:
    AGENT_MEMORY_LEARNED_TOTAL.labels(
        agent_name=agent_name, agent_domain=agent_domain
    ).inc()


# --- Utility to start HTTP server for metrics ---
def start_metrics_server(port: int = 8008, addr: str = "0.0.0.0") -> None:
    """Starts an HTTP server to expose the metrics on /metrics endpoint."""
    try:
        start_http_server(port, addr=addr, registry=REGISTRY)
        print(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        print(f"Error starting Prometheus metrics server: {e}")


