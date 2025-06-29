import pytest
import pytest_asyncio
import shutil
import tempfile
from pathlib import Path
import asyncio # Added for get_event_loop

from memory_gate.storage.vector_store import VectorMemoryStore
# from memory_gate.memory_protocols import LearningContext


@pytest.fixture(scope="module") # Factory can remain module scoped
def temp_chroma_directory_factory():
    """A factory to create temporary directories for ChromaDB persistence (module-scoped)."""
    created_dirs = []
    def _create_temp_dir(name_prefix="chroma_module_") -> Path:
        temp_dir = Path(tempfile.mkdtemp(prefix=name_prefix))
        created_dirs.append(temp_dir)
        return temp_dir

    yield _create_temp_dir

    for directory in created_dirs:
        shutil.rmtree(directory)

@pytest.fixture # Function scope for the actual directory path
def temp_chroma_directory(temp_chroma_directory_factory) -> Path:
    """Create a function-scoped temporary directory for ChromaDB."""
    return temp_chroma_directory_factory("chroma_func_")


def _init_vector_store_sync(collection_name: str, persist_directory: str | None, model_name: str) -> VectorMemoryStore:
    """Synchronous helper to initialize VectorMemoryStore (used by run_in_executor)."""
    return VectorMemoryStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model_name=model_name,
    )

@pytest_asyncio.fixture # Function scope
async def persistent_vector_store(temp_chroma_directory: Path) -> VectorMemoryStore:
    """Create a VectorMemoryStore with persistence for testing, initialized in executor."""
    loop = asyncio.get_event_loop()
    store = await loop.run_in_executor(
        None, # Default executor (ThreadPoolExecutor)
        _init_vector_store_sync, # Callable
        "test_persistent_collection", # args for _init_vector_store_sync
        str(temp_chroma_directory),
        "all-MiniLM-L6-v2"
    )
    # If the store has an async clear method that needs awaiting:
    # await store.clear_all_data_dangerous()
    # Or, if clearing is sync but should happen after init for safety:
    # await loop.run_in_executor(None, store.client.delete_collection, store.collection_name)
    # await loop.run_in_executor(None, setattr, store, 'collection', store.client.get_or_create_collection(store.collection_name))
    # For now, assuming tests will handle their own specific data setup/cleanup after getting the store.
    return store


@pytest_asyncio.fixture # Function scope
async def in_memory_vector_store() -> VectorMemoryStore:
    """Create an in-memory VectorMemoryStore for testing, initialized in executor."""
    loop = asyncio.get_event_loop()
    store = await loop.run_in_executor(
        None,
        _init_vector_store_sync,
        "test_in_memory_collection",
        None, # No persistence
        "all-MiniLM-L6-v2"
    )
    # Similar clearing considerations as above if needed.
    return store
