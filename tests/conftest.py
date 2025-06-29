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


@pytest_asyncio.fixture # Function scope
async def persistent_vector_store(temp_chroma_directory: Path) -> VectorMemoryStore:
    """Create a VectorMemoryStore with persistence for testing."""
    # Direct initialization (synchronous) within the async fixture.
    # pytest-asyncio will run this fixture setup in its managed event loop.
    # If VectorMemoryStore.__init__ is truly blocking or misbehaves with asyncio,
    # this might still cause issues. This change is for diagnosis.
    store = VectorMemoryStore(
        collection_name="test_persistent_collection",
        persist_directory=str(temp_chroma_directory),
        embedding_model_name="all-MiniLM-L6-v2",
        device="cpu"  # Explicitly use CPU for tests
    )
    # If tests require a clean state and the store is re-used (e.g. module scope, though this is function scope)
    # await clear_vector_store_collection_dangerous(store) # If a clear method were available and needed
    return store


@pytest_asyncio.fixture # Function scope
async def in_memory_vector_store() -> VectorMemoryStore:
    """Create an in-memory VectorMemoryStore for testing."""
    store = VectorMemoryStore(
        collection_name="test_in_memory_collection",
        persist_directory=None, # In-memory
        embedding_model_name="all-MiniLM-L6-v2",
        device="cpu"  # Explicitly use CPU for tests
    )
    # await clear_vector_store_collection_dangerous(store) # If needed for state management across uses
    return store
