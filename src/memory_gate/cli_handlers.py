import argparse
from datetime import datetime

from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import LearningContext


async def cli_store_experience(gateway: MemoryGateway[LearningContext], args: argparse.Namespace) -> None:
    """Handles the 'store' CLI command."""
    metadata = {}
    if args.metadata:
        for item in args.metadata:
            try:
                key, value = item.split('=', 1)
                metadata[key] = value
            except ValueError:
                print(f"Warning: Malformed metadata item '{item}' ignored. Expected format: key=value")

    context = LearningContext(
        content=args.content,
        domain=args.domain,
        timestamp=datetime.now(),
        importance=args.importance,
        metadata=metadata
    )
    # learn_from_interaction now returns (adapted_context, storage_task)
    adapted_context, storage_task = await gateway.learn_from_interaction(context, args.feedback)

    # Await the storage task to ensure completion for CLI
    await storage_task

    # Use the public method to get the key
    key = gateway.get_context_key(adapted_context)
    print(f"Stored experience with key: {key}. Content: '{adapted_context.content}', Importance: {adapted_context.importance}")


async def cli_retrieve_context(gateway: MemoryGateway[LearningContext], args: argparse.Namespace) -> None:
    """Handles the 'retrieve' CLI command."""
    contexts = await gateway.store.retrieve_context(
        query=args.query,
        limit=args.limit,
        domain_filter=args.domain_filter
    )
    if contexts:
        print(f"Retrieved {len(contexts)} context(s):")
        for i, ctx in enumerate(contexts):
            print(f"  {i+1}. Content: '{ctx.content}'")
            print(f"     Domain: {ctx.domain}, Timestamp: {ctx.timestamp.isoformat()}, Importance: {ctx.importance}")
            if ctx.metadata:
                print(f"     Metadata: {ctx.metadata}")
    else:
        print("No relevant context found.")
