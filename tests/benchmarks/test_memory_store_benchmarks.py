"""Benchmark tests for memory store performance."""

from datetime import datetime
import logging

import pytest

from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.vector_store import VectorMemoryStore

logger = logging.getLogger(__name__)


@pytest.fixture
def benchmark_data() -> list[tuple[str, LearningContext]]:
    """Generate benchmark test data."""
    return [
        (
            f"benchmark_key_{i}",
            LearningContext(
                content=f"Benchmark test content {i} with sufficient length to test real-world scenarios",
                domain="benchmark_domain",
                timestamp=datetime.now(),
                importance=0.8,
                metadata={"benchmark": "true", "index": str(i)},
            ),
        )
        for i in range(100)  # Generate 100 test items
    ]


@pytest.mark.benchmark
@pytest.mark.asyncio
@pytest.mark.timeout(60)  # 60 second timeout for individual benchmark tests
async def test_store_experience_performance(
    benchmark,
    persistent_vector_store: VectorMemoryStore,
    benchmark_data: list[tuple[str, LearningContext]],
):
    """Benchmark storing experiences."""
    logger.info("Starting store experience performance benchmark")
    key, context = benchmark_data[0]
    logger.info(f"Benchmarking storage of experience with key: {key}")

    async def store_async():
        logger.debug("Executing store operation")
        await persistent_vector_store.store_experience(key, context)
        logger.debug("Store operation completed")

    logger.info("Running benchmark...")
    await benchmark(store_async)
    logger.info("Benchmark completed")

    # Verification step after benchmark
    logger.info("Verifying stored experience")
    result = await persistent_vector_store.get_experience_by_id(key)
    assert result is not None
    assert result.content == context.content
    logger.info("Store experience performance test completed successfully")


@pytest.mark.benchmark
@pytest.mark.asyncio
@pytest.mark.timeout(120)  # 2 minute timeout for batch operations
async def test_batch_store_performance(
    benchmark,
    persistent_vector_store: VectorMemoryStore,
    benchmark_data: list[tuple[str, LearningContext]],
):
    """Benchmark batch storing of experiences."""
    logger.info("Starting batch store performance benchmark")
    batch_size = 10
    logger.info(f"Benchmarking batch storage of {batch_size} experiences")

    async def store_batch_async():
        logger.debug("Starting batch store operation")
        for i, (key, context) in enumerate(benchmark_data[:batch_size]):
            if i % 5 == 0:  # Log progress every 5 items
                logger.debug(f"Storing item {i + 1}/{batch_size}")
            await persistent_vector_store.store_experience(key, context)
        logger.debug("Batch store operation completed")

    logger.info("Running batch benchmark...")
    await benchmark(store_batch_async)
    logger.info("Batch benchmark completed")

    # Verification step after benchmark
    logger.info("Verifying batch store results")
    collection_size = persistent_vector_store.get_collection_size()
    logger.info(f"Collection size after batch store: {collection_size}")
    assert collection_size >= batch_size
    logger.info("Batch store performance test completed successfully")


@pytest.mark.benchmark
@pytest.mark.asyncio
@pytest.mark.timeout(60)  # 60 second timeout for GPU tests
async def test_gpu_embedding_performance(benchmark, temp_chroma_directory):
    """Benchmark embedding generation with GPU acceleration."""
    logger.info("Starting GPU embedding performance benchmark")
    # This test is designed to run on a machine with a GPU.
    # It will be skipped if no GPU is available.
    try:
        logger.info("Initializing VectorMemoryStore for GPU test")
        store = VectorMemoryStore(
            collection_name="benchmark_gpu_collection",
            persist_directory=str(temp_chroma_directory),
        )
        logger.info("VectorMemoryStore initialized successfully")
    except ImportError:
        logger.warning("Skipping GPU test: PyTorch with CUDA not available")
        pytest.skip("Skipping GPU test: PyTorch with CUDA not available.")

    test_text = (
        "This is a test content for benchmarking GPU embedding generation performance"
    )
    logger.info(
        f"Benchmarking embedding generation for text of length: {len(test_text)}"
    )

    async def generate_gpu_embedding_async():
        logger.debug("Generating embedding...")
        result = await store._generate_embedding(test_text)
        logger.debug(f"Embedding generated with dimension: {len(result)}")
        return result

    logger.info("Running GPU embedding benchmark...")
    await benchmark(generate_gpu_embedding_async)
    logger.info("GPU embedding performance test completed successfully")
