"""Basic usage example for MemoryGate.

This script demonstrates the core functionality of MemoryGate:
- Creating a knowledge store
- Storing learning contexts
- Querying stored knowledge
- Working with memory protocols
"""

import asyncio
from datetime import datetime

from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.in_memory import InMemoryKnowledgeStore


async def main() -> None:
    """Demonstrate basic MemoryGate usage."""
    print("MemoryGate Basic Usage Example")
    print("=" * 60)

    # 1. Create an in-memory knowledge store
    print("\n1. Creating in-memory knowledge store...")
    store = InMemoryKnowledgeStore()

    # 2. Store some learning contexts
    print("\n2. Storing learning contexts...")
    contexts = [
        LearningContext(
            content="Python is a versatile programming language used for web development, data science, and automation",
            domain="programming",
            importance=0.9,
            timestamp=datetime.now(),
            metadata={"category": "languages", "level": "beginner"},
        ),
        LearningContext(
            content="FastAPI is a modern, fast web framework for building APIs with Python based on type hints",
            domain="programming",
            importance=0.8,
            timestamp=datetime.now(),
            metadata={"category": "frameworks", "type": "web"},
        ),
        LearningContext(
            content="Vector databases store and retrieve data based on semantic similarity using embeddings",
            domain="ai",
            importance=0.9,
            timestamp=datetime.now(),
            metadata={"category": "databases", "type": "vector"},
        ),
        LearningContext(
            content="MemoryGate enables AI systems to retain and build upon knowledge across sessions",
            domain="ai",
            importance=1.0,
            timestamp=datetime.now(),
            metadata={"category": "memory", "project": "memorygate"},
        ),
    ]

    for i, context in enumerate(contexts):
        await store.store_experience(f"context_{i}", context)
        print(f"   ✓ Stored: {context.content[:60]}...")

    # 3. Query knowledge by content
    print("\n3. Querying knowledge about 'Python frameworks'...")
    results = await store.retrieve_context("Python frameworks", limit=2)
    print(f"   Found {len(results)} relevant contexts:")
    for ctx in results:
        print(f"   - {ctx.content[:70]}...")
        print(f"     Domain: {ctx.domain}, Importance: {ctx.importance}")

    # 4. Query by domain
    print("\n4. Querying 'programming' domain...")
    results = await store.retrieve_context(
        "programming", domain_filter="programming", limit=5
    )
    print(f"   Found {len(results)} programming contexts:")
    for ctx in results:
        print(f"   - {ctx.content[:70]}...")

    # 5. Demonstrate metadata access
    print("\n5. Examining metadata...")
    ai_results = await store.retrieve_context("AI", domain_filter="ai", limit=10)
    print(f"   Found {len(ai_results)} AI-related contexts:")
    for ctx in ai_results:
        print(f"   - {ctx.content[:60]}...")
        if ctx.metadata:
            print(f"     Metadata: {ctx.metadata}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("\nNote: This example uses in-memory storage.")
    print("For persistent storage, use ChromaDB or Qdrant backends.")


if __name__ == "__main__":
    asyncio.run(main())
