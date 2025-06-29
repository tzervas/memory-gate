import asyncio
import os
import argparse
from datetime import datetime
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


# Placeholder for server mode startup.
# Actual server initialization (e.g., with VectorMemoryStore, metrics, etc.)
# would go into a separate module or function called when mode == 'server'.
async def start_server_mode() -> None:
    print("Server mode selected. Full server initialization is not yet implemented in this entrypoint.")
    print("To run the full application with all components (VectorStore, Agents, Metrics, etc.),")
    print("you would typically have a dedicated server startup script or use a managed deployment (e.g., Docker with a different CMD).")

    # Example of what might be here:
    # from memory_gate.server import run_application_server
    # await run_application_server()

    # For now, just keep it alive as a placeholder
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        print("Server mode placeholder task cancelled.")


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
    main_task: Optional[asyncio.Task[None]] = None

    try:
        if args.mode == "cli":
            main_task = loop.create_task(main_cli_handler(args))
            loop.run_until_complete(main_task)
        elif args.mode == "server":
            main_task = loop.create_task(start_server_mode())
            loop.run_until_complete(main_task)
        else:
            # Should not happen if subparsers.required = True
            parser.print_help()

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
