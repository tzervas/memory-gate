"""Example usage of the MemoryGate API.

This script demonstrates how to interact with the MemoryGate REST API
for memory storage, retrieval, and provider-agnostic generation.
"""

import asyncio
import json

import httpx


async def main() -> None:
    """Demonstrate MemoryGate API usage."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        # 1. Check API health
        print("1. Checking API health...")
        response = await client.get(f"{base_url}/health")
        print(f"   Status: {response.json()['status']}\n")

        # 2. Store some memories
        print("2. Storing memories...")
        memories = [
            {
                "content": "Python is a high-level programming language",
                "domain": "programming",
                "importance": 0.8,
            },
            {
                "content": "FastAPI is a modern web framework for Python",
                "domain": "programming",
                "importance": 0.9,
            },
            {
                "content": "MemoryGate provides persistent memory for AI systems",
                "domain": "ai",
                "importance": 1.0,
            },
        ]

        for memory in memories:
            response = await client.post(
                f"{base_url}/api/v1/memory/store",
                json=memory,
            )
            print(f"   Stored: {memory['content'][:50]}...")

        print()

        # 3. Query memories
        print("3. Querying memories about 'Python'...")
        response = await client.post(
            f"{base_url}/api/v1/memory/query",
            json={"query": "Python", "limit": 5},
        )
        results = response.json()
        print(f"   Found {results['count']} memories:")
        for mem in results["memories"]:
            print(f"   - {mem['content'][:60]}... (importance: {mem['importance']})")

        print()

        # 4. Augment a prompt with memories
        print("4. Augmenting prompt with memories...")
        response = await client.post(
            f"{base_url}/api/v1/memory/augment",
            json={"prompt": "Tell me about Python web frameworks", "limit": 3},
        )
        result = response.json()
        print(f"   Original prompt: {result['original_prompt']}")
        print(f"   Memories used: {result['memories_used']}")
        print(f"   Augmented prompt:\n{result['augmented_prompt'][:200]}...\n")

        # 5. Get OpenAPI documentation
        print("5. OpenAPI documentation available at:")
        print(f"   - Swagger UI: {base_url}/docs")
        print(f"   - ReDoc: {base_url}/redoc")
        print(f"   - OpenAPI JSON: {base_url}/openapi.json")


if __name__ == "__main__":
    print("MemoryGate API Example")
    print("=" * 60)
    print("Make sure the API server is running:")
    print("  python -m memory_gate.api.cli --port 8000")
    print("=" * 60)
    print()

    try:
        asyncio.run(main())
    except httpx.ConnectError:
        print("\nError: Could not connect to API server.")
        print("Please start the server first:")
        print("  python -m memory_gate.api.cli --port 8000")
