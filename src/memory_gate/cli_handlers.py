import argparse
from datetime import datetime

from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import LearningContext


async def cli_store_experience(gateway: MemoryGateway[LearningContext], args: argparse.Namespace) -> None:
    """Handles the 'store' CLI command."""
    metadata = {}
    if args.metadata:
        for item in args.metadata:
            key, value = item.split('=', 1)
            metadata[key] = value

    context = LearningContext(
        content=args.content,
        domain=args.domain,
        timestamp=datetime.now(),
        importance=args.importance,
        metadata=metadata
    )
    # Call with sync_store=True for CLI to ensure operation completes.
    adapted_context = await gateway.learn_from_interaction(context, args.feedback, sync_store=True)

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
