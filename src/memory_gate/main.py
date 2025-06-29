"""Main entrypoint for MemoryGate: initializes storage, memory gateway, agents, and metrics server.
Handles graceful shutdown and background task management.
"""

import asyncio
import os
<<<<<<< HEAD
import argparse
from datetime import datetime
=======
>>>>>>> 58ed5e323d40cd90c3089e434c046b08405c3878
from typing import List, Optional

from memory_gate.memory_protocols import LearningContext, MemoryAdapter, KnowledgeStore
from memory_gate.storage.in_memory import InMemoryKnowledgeStore # Using InMemory for CLI
from memory_gate.memory_gateway import MemoryGateway
# Import other components if the full system needs to run, or keep minimal for CLI
# from memory_gate.consolidation import ConsolidationWorker
# from memory_gate.agent_interface import SimpleEchoAgent
# from memory_gate.agents import InfrastructureAgent
# from memory_gate.metrics import start_metrics_server


# Global list to keep track of background tasks for graceful shutdown (if any)
background_tasks: List[asyncio.Task[None]] = []


class PassthroughAdapter(MemoryAdapter[LearningContext]):
    """A simple adapter that passes through the context, optionally adjusting importance."""

    async def adapt_knowledge(
        self, context: LearningContext, feedback: float | None = None
    ) -> LearningContext:
        if feedback is not None and 0.0 <= feedback <= 1.0:
            context.importance = (context.importance + feedback) / 2.0
        elif feedback is not None:
            context.importance = feedback
        return context


# Import CLI handlers from the new module
from memory_gate.cli_handlers import cli_store_experience, cli_retrieve_context

<<<<<<< HEAD
=======
    # 1. Initialize Storage
    print(
        f"Initializing VectorMemoryStore with ChromaDB: directory='"
        f"{CHROMA_PERSIST_DIRECTORY}', collection='{CHROMA_COLLECTION_NAME}'"
    )
    knowledge_store = VectorMemoryStore(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )
    # Test storage (optional, for quick verification during startup)
    # print(f"Initial collection size: {knowledge_store.get_collection_size()}")
>>>>>>> 58ed5e323d40cd90c3089e434c046b08405c3878

async def main_cli_handler(args: argparse.Namespace) -> None:
    """Initializes gateway and handles CLI commands."""
    # For CLI, using InMemoryKnowledgeStore for simplicity
    knowledge_store: KnowledgeStore[LearningContext] = InMemoryKnowledgeStore()
    memory_adapter: MemoryAdapter[LearningContext] = PassthroughAdapter()
    gateway = MemoryGateway(adapter=memory_adapter, store=knowledge_store)

    if args.command == "store":
        await cli_store_experience(gateway, args)
    elif args.command == "retrieve":
        await cli_retrieve_context(gateway, args)
    else:
        print(f"Unknown command: {args.command}")
        # This path should ideally not be reached if argparse is set up correctly.

<<<<<<< HEAD
=======
    # 5. Initialize Agents (example)
    # In a real scenario, agents might be managed differently (e.g., separate processes,
    # dynamically loaded)
    # For now, we can instantiate them to show they can use the gateway.
    echo_agent = SimpleEchoAgent(memory_gateway)
    infra_agent = InfrastructureAgent(memory_gateway)
    print(
        f"Example agents initialized: {echo_agent.agent_name}, {infra_agent.agent_name}"
    )
>>>>>>> 58ed5e323d40cd90c3089e434c046b08405c3878

# --- Full System Startup (can be kept for non-CLI execution) ---
async def main_async_server() -> None:
    """Initializes and starts the MemoryGate components for server mode."""
    # This is the original main_async, renamed and potentially simplified if CLI is primary
    print("Initializing MemoryGate System (Server Mode)...")
    # ... (original server setup code from previous main_async)
    # For now, let's keep it minimal as the focus is CLI
    # Metrics Server Configuration
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8008"))
    METRICS_HOST = os.getenv("METRICS_HOST", "0.0.0.0")

    # Initialize Storage (e.g. VectorMemoryStore for server mode)
    # knowledge_store = VectorMemoryStore(...)
    # memory_adapter = PassthroughAdapter()
    # memory_gateway = MemoryGateway(adapter=memory_adapter, store=knowledge_store)
    # print("MemoryGateway initialized for server mode.")
    # Start metrics server, consolidation worker etc.
    # start_metrics_server(port=METRICS_PORT, addr=METRICS_HOST)
    print("Server mode components would be initialized here.")
    print("For this iteration, server mode is placeholder.")

    print("\nMemoryGate System (Server Mode) would be running. Press Ctrl+C to exit.")
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
<<<<<<< HEAD
        print("Server task cancelled. Shutting down...")


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoryGate CLI and Server Control.")
    subparsers = parser.add_subparsers(dest="mode", help="Run mode: 'cli' or 'server'")
    subparsers.required = True # Make mode selection mandatory

    # --- CLI Mode Parser ---
    cli_parser = subparsers.add_parser("cli", help="Run in CLI mode for direct interaction.")
    cli_subparsers = cli_parser.add_subparsers(dest="command", help="CLI command to execute")
    cli_subparsers.required = True

    # Store command
    store_parser = cli_subparsers.add_parser("store", help="Store a new learning experience.")
    store_parser.add_argument("content", type=str, help="Content of the learning experience.")
    store_parser.add_argument("--domain", type=str, default="general", help="Domain of the experience.")
    store_parser.add_argument("--importance", type=float, default=1.0, help="Importance score (0.0-1.0).")
    store_parser.add_argument("--feedback", type=float, help="Feedback score for adaptation (optional).")
    store_parser.add_argument("--metadata", nargs='*', help="Metadata key-value pairs (e.g., key1=value1 key2=value2).")

    # Retrieve command
    retrieve_parser = cli_subparsers.add_parser("retrieve", help="Retrieve relevant learning context.")
    retrieve_parser.add_argument("query", type=str, help="Query string to search for.")
    retrieve_parser.add_argument("--limit", type=int, default=5, help="Maximum number of contexts to retrieve.")
    retrieve_parser.add_argument("--domain-filter", type=str, help="Filter by domain (optional).")

    # --- Server Mode Parser ---
    server_parser = subparsers.add_parser("server", help="Run in Server mode (e.g., with metrics, consolidation).")
    # Add server-specific arguments if any, e.g., --port, --config-file

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
=======
        print("Main task cancelled. Shutting down...")


async def shutdown_handler(
    loop: asyncio.AbstractEventLoop, consolidation_worker: ConsolidationWorker | None
) -> None:
    """
    Gracefully shuts down background tasks and the consolidation worker, then stops the event
    loop. Ensures all background tasks are cancelled and awaited before stopping the event loop.
    """
    print("Initiating graceful shutdown...")

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
        print("Stopping ConsolidationWorker...")
        await consolidation_worker.stop()  # Ensure its specific stop logic is called

    print("Background tasks stopped.")

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

>>>>>>> 58ed5e323d40cd90c3089e434c046b08405c3878
    main_task: Optional[asyncio.Task[None]] = None

    try:
<<<<<<< HEAD
        if args.mode == "cli":
            main_task = loop.create_task(main_cli_handler(args))
            loop.run_until_complete(main_task)
        elif args.mode == "server":
            # For now, server mode is a placeholder
            print("Server mode selected. This is currently a placeholder.")
            # To run the full server:
            # main_task = loop.create_task(main_async_server())
            # loop.run_until_complete(main_task)
        else:
            # Should not happen if subparsers.required = True
            parser.print_help()

=======
        main_task = loop.create_task(main_async())  # main_async returns None implicitly
        loop.run_until_complete(main_task)
>>>>>>> 58ed5e323d40cd90c3089e434c046b08405c3878
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Shutting down...")
    finally:
        if main_task and not main_task.done():
            main_task.cancel()
            # Ensure the task has time to process cancellation
            # loop.run_until_complete(main_task) # This can lead to "Cannot cancel a completed task"
            # A simple way to allow cleanup:
            if not loop.is_closed():
                # Gather remaining tasks, including the cancelled main_task
                # This might be overly complex for simple CLI; for server it's more relevant
                pending = asyncio.all_tasks(loop=loop)
        print("MemoryGate application finished.")


if __name__ == "__main__":
    main()
