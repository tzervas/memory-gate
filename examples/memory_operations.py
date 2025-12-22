"""Memory operations example for MemoryGate.

This script demonstrates advanced memory operations:
- Batch storage of learning contexts
- Filtering by domain and importance
- Memory metadata usage
- Knowledge lifecycle management
"""

import asyncio
from datetime import datetime, timedelta

from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.in_memory import InMemoryKnowledgeStore


async def demonstrate_batch_storage() -> None:
    """Demonstrate batch memory storage."""
    print("\n1. Batch Memory Storage")
    print("-" * 60)

    store = InMemoryKnowledgeStore()

    # Create multiple memories from a conversation
    conversation = [
        ("User asked about Python features", 0.7),
        ("Explained list comprehensions with examples", 0.8),
        ("User asked about async/await syntax", 0.9),
        ("Provided async programming examples", 0.8),
        ("User successfully implemented async function", 1.0),
    ]

    print("  Storing conversation memories...")
    for i, (content, importance) in enumerate(conversation, 1):
        context = LearningContext(
            content=content,
            domain="conversation",
            importance=importance,
            timestamp=datetime.now() - timedelta(minutes=len(conversation) - i),
            metadata={"turn": i, "type": "interaction"},
        )
        await store.store_experience(f"conv_{i}", context)
        print(f"    {i}. {content}")

    print(f"\n  ✓ Stored {len(conversation)} conversation memories")


async def demonstrate_domain_filtering() -> None:
    """Demonstrate filtering memories by domain."""
    print("\n2. Domain-Based Filtering")
    print("-" * 60)

    store = InMemoryKnowledgeStore()

    # Store memories in different domains
    domains_data = {
        "programming": [
            "Python uses dynamic typing",
            "JavaScript is single-threaded",
            "Go has built-in concurrency",
        ],
        "devops": [
            "Docker containers provide isolation",
            "Kubernetes orchestrates containers",
            "Terraform manages infrastructure as code",
        ],
        "database": [
            "PostgreSQL supports ACID transactions",
            "Redis is an in-memory data store",
            "MongoDB is a document database",
        ],
    }

    print("  Storing memories across different domains...")
    context_id = 0
    for domain, contents in domains_data.items():
        for content in contents:
            context = LearningContext(
                content=content,
                domain=domain,
                importance=0.8,
                timestamp=datetime.now(),
            )
            await store.store_experience(f"ctx_{context_id}", context)
            context_id += 1
        print(f"    ✓ Stored {len(contents)} {domain} memories")

    # Query specific domain
    print("\n  Querying 'devops' domain...")
    results = await store.retrieve_context(
        "containers deployment", domain_filter="devops", limit=10
    )
    for ctx in results:
        print(f"    - {ctx.content}")


async def demonstrate_importance_filtering() -> None:
    """Demonstrate filtering by importance level."""
    print("\n3. Importance-Based Filtering")
    print("-" * 60)

    store = InMemoryKnowledgeStore()

    # Store memories with varying importance
    memories_with_importance = [
        ("Critical bug fix in production", 1.0),
        ("New feature request from customer", 0.9),
        ("Code review comment addressed", 0.7),
        ("Documentation typo fixed", 0.4),
        ("Coffee break chat about weather", 0.1),
    ]

    print("  Storing memories with different importance levels...")
    for i, (content, importance) in enumerate(memories_with_importance):
        context = LearningContext(
            content=content,
            domain="work",
            importance=importance,
            timestamp=datetime.now(),
        )
        await store.store_experience(f"work_{i}", context)
        print(f"    - {content} (importance: {importance})")

    # Query and filter by importance
    print("\n  Querying high-importance memories (>= 0.8)...")
    results = await store.retrieve_context("work activities", limit=10)
    high_importance = [ctx for ctx in results if ctx.importance >= 0.8]
    for ctx in high_importance:
        print(f"    - {ctx.content} ({ctx.importance})")


async def demonstrate_metadata_usage() -> None:
    """Demonstrate using metadata for rich context."""
    print("\n4. Metadata Usage")
    print("-" * 60)

    store = InMemoryKnowledgeStore()

    # Store memories with rich metadata
    print("  Storing memories with metadata...")
    memories = [
        LearningContext(
            content="Implemented user authentication with JWT",
            domain="development",
            importance=0.9,
            timestamp=datetime.now(),
            metadata={
                "project": "api-service",
                "author": "developer1",
                "commit": "abc123",
                "files_changed": 5,
                "lines_added": 150,
            },
        ),
        LearningContext(
            content="Deployed version 2.0 to production",
            domain="deployment",
            importance=1.0,
            timestamp=datetime.now(),
            metadata={
                "project": "api-service",
                "environment": "production",
                "version": "2.0.0",
                "deployer": "automated",
            },
        ),
        LearningContext(
            content="Fixed memory leak in data processing",
            domain="development",
            importance=0.95,
            timestamp=datetime.now(),
            metadata={
                "project": "data-pipeline",
                "author": "developer2",
                "issue": "PROJ-456",
                "severity": "high",
            },
        ),
    ]

    for i, context in enumerate(memories):
        await store.store_experience(f"dev_{i}", context)
        print(f"    ✓ {context.content}")
        print(f"      Metadata: {context.metadata}")

    # Query and inspect metadata
    print("\n  Querying and examining metadata...")
    results = await store.retrieve_context(
        "development activities", domain_filter="development", limit=10
    )
    for ctx in results:
        print(f"    - {ctx.content}")
        if ctx.metadata and "project" in ctx.metadata:
            print(f"      Project: {ctx.metadata['project']}")


async def main() -> None:
    """Run all memory operation examples."""
    print("MemoryGate Memory Operations Examples")
    print("=" * 60)

    await demonstrate_batch_storage()
    await demonstrate_domain_filtering()
    await demonstrate_importance_filtering()
    await demonstrate_metadata_usage()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("\nKey Takeaways:")
    print("- Memories can be organized by domain for efficient retrieval")
    print("- Importance levels help prioritize critical information")
    print("- Metadata enables rich contextual information storage")
    print("- Batch operations efficiently handle multiple memories")


if __name__ == "__main__":
    asyncio.run(main())
