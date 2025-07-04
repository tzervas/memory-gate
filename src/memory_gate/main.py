"""Main entrypoint for MemoryGate: initializes storage, memory gateway, agents, and metrics server.
Handles graceful shutdown and background task management.
"""

import asyncio
import logging
import os

from memory_gate.agent_interface import SimpleEchoAgent
from memory_gate.agents import InfrastructureAgent
from memory_gate.consolidation import ConsolidationWorker
from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import LearningContext, MemoryAdapter
from memory_gate.metrics import start_metrics_server
from memory_gate.storage.vector_store import VectorMemoryStore, VectorStoreConfig

# --- Configuration ---
# These could be loaded from environment variables, config files, etc.
# For Helm, these would typically be passed as env vars or mounted config files.

# Metrics Server Configuration
METRICS_PORT = int(os.getenv("METRICS_PORT", "8008"))
METRICS_HOST = os.getenv("METRICS_HOST", "127.0.0.1")

# ChromaDB Configuration (used by VectorMemoryStore)
CHROMA_PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY", "./data/production_chromadb_store"
)
CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME", "memory_gate_prod_collection"
)
# Ensure the persist directory exists if it's local
if CHROMA_PERSIST_DIRECTORY.startswith(("./", "/")):
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)


# Consolidation Worker Configuration
CONSOLIDATION_ENABLED = os.getenv("CONSOLIDATION_ENABLED", "true").lower() == "true"
CONSOLIDATION_INTERVAL_SECONDS = int(
    os.getenv("CONSOLIDATION_INTERVAL_SECONDS", "3600")
)  # 1 hour

# Global list to keep track of background tasks for graceful shutdown
background_tasks: list[asyncio.Task[None]] = []

# Initialize module-level logger
logger = logging.getLogger(__name__)


class PassthroughAdapter(MemoryAdapter[LearningContext]):
    """A simple adapter that passes through the context, optionally adjusting importance."""

    async def adapt_knowledge(
        self, context: LearningContext, feedback: float | None = None
    ) -> LearningContext:
        if feedback is not None and 0.0 <= feedback <= 1.0:
            # Example: average current importance with feedback score
            context.importance = (context.importance + feedback) / 2.0
        elif feedback is not None:  # e.g. if feedback can be any score
            context.importance = feedback  # directly set importance from feedback
        return context


async def main_async() -> None:
    """Initializes and starts the MemoryGate components."""
    logger.info("Initializing MemoryGate System...")

    # 1. Initialize Storage
    logger.info(
        "Initializing VectorMemoryStore with ChromaDB: directory='%s', collection='%s'",
        CHROMA_PERSIST_DIRECTORY,
        CHROMA_COLLECTION_NAME,
    )
    vector_config = VectorStoreConfig(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )
    knowledge_store = VectorMemoryStore(config=vector_config)
    # Test storage (optional, for quick verification during startup)
    # logger.debug("Initial collection size: %s", knowledge_store.get_collection_size())

    # 2. Initialize Memory Adapter
    memory_adapter = PassthroughAdapter()

    # 3. Initialize Memory Gateway
    memory_gateway = MemoryGateway(adapter=memory_adapter, store=knowledge_store)
    logger.info("MemoryGateway initialized.")

    # 4. Initialize and Start Consolidation Worker (if enabled)
    consolidation_worker = None
    if CONSOLIDATION_ENABLED:
        logger.info(
            "Initializing ConsolidationWorker with interval %ds.",
            CONSOLIDATION_INTERVAL_SECONDS,
        )
        consolidation_worker = ConsolidationWorker(
            store=knowledge_store,  # ConsolidationWorker expects a store that can provide keys
            consolidation_interval=CONSOLIDATION_INTERVAL_SECONDS,
        )
        consolidation_task = asyncio.create_task(consolidation_worker.start())
        background_tasks.append(consolidation_task)
        logger.info("ConsolidationWorker started.")
    else:
        logger.info("ConsolidationWorker is disabled.")

    # 5. Initialize Agents (example)
    # In a real scenario, agents might be managed differently (e.g., separate processes,
    # dynamically loaded)
    # For now, we can instantiate them to show they can use the gateway.
    echo_agent = SimpleEchoAgent(memory_gateway)
    infra_agent = InfrastructureAgent(memory_gateway)
    logger.info(
        "Example agents initialized: %s, %s",
        echo_agent.agent_name,
        infra_agent.agent_name,
    )

    # --- Example Agent Usage (optional, for testing if run directly) ---
    # This part would typically be driven by external events or an API
    # For a K8s service, the app would just run and wait for API calls or other triggers.
    # if __name__ == "__main__": # or some dev mode flag
    #     logger.info("Running example agent interaction")
    #     task_result, confidence = await infra_agent.process_task(
    #         "The main application server is reporting high CPU usage after the last deployment."
    #     )
    #     logger.info("InfraAgent Task Result:\n%s\nConfidence: %.2f", task_result, confidence)
    #     logger.info("Example agent interaction finished")

    # 6. Start Prometheus Metrics Server
    logger.info(
        "Starting Prometheus metrics server on %s:%d...", METRICS_HOST, METRICS_PORT
    )
    start_metrics_server(port=METRICS_PORT, addr=METRICS_HOST)

    logger.info("MemoryGate System is running. Press Ctrl+C to exit.")

    # Keep the main coroutine alive. In a real app, this might be an API server loop.
    # For now, just wait until cancellation.
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for a long time, or handle other tasks
    except asyncio.CancelledError:
        logger.info("Main task cancelled. Shutting down...")


async def shutdown_handler(
    loop: asyncio.AbstractEventLoop, consolidation_worker: ConsolidationWorker | None
) -> None:
    """
    Gracefully shuts down background tasks and the consolidation worker, then stops the event
    loop. Ensures all background tasks are cancelled and awaited before stopping the event loop.
    """
    logger.info("Initiating graceful shutdown...")

    # Cancel all background tasks
    for task in background_tasks:
        if not task.done():
            task.cancel()

    # Wait for tasks to complete cancellation
    await asyncio.gather(
        *[task for task in background_tasks if not task.done()], return_exceptions=True
    )

    if (
        consolidation_worker
        and getattr(consolidation_worker, "is_running", lambda: False)()
    ):
        logger.info("Stopping ConsolidationWorker...")
        await consolidation_worker.stop()  # Ensure its specific stop logic is called

    logger.info("Background tasks stopped.")

    # Properly stop the event loop after all cleanup
    loop.stop()


def main() -> None:
    """Entrypoint for running the MemoryGate application and event loop."""
    loop = asyncio.get_event_loop()

    # Need to pass consolidation_worker to shutdown_handler.
    # This is tricky as it's created inside main_async.
    # A simpler way for now: main_async handles its own cleanup of consolidation_worker on
    # CancelledError. Or, make consolidation_worker accessible globally or via a class.
    # For this example, let's rely on main_async's try/except for its own task cleanup.
    # The signal handler will cancel main_async.

    # This is a simplified signal handling. For robust production, consider more complex patterns.
    # The shutdown_handler is not directly used here because main_async is the primary task.
    # When main_async is cancelled, its finally block (if any) or except CancelledError should
    # handle cleanup.

    main_task: asyncio.Task[None] | None = None
    try:
        main_task = loop.create_task(main_async())  # main_async returns None implicitly
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    finally:
        if main_task and not main_task.done():
            main_task.cancel()
            # Give it a moment to cancel
            # This might need to be run within the loop if main_task.cancel() doesn't block
            # loop.run_until_complete(main_task) # This can cause issues if loop is already stopping

            # A more robust shutdown for tasks created within main_async:
            # main_async should catch CancelledError and await its own created tasks.
            # For now, cancelling main_task and letting its internal structure handle it.
            logger.info("Main task cancellation requested.")
            # Allow some time for cleanup within main_async's CancelledError block
            # This is still a bit simplistic.
            # loop.run_until_complete(asyncio.sleep(1)) # Not ideal here.

        # Additional cleanup for other resources if necessary
        logger.info("MemoryGate application stopped.")
