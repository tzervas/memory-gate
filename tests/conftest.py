import pytest
import pytest_asyncio
import shutil
import tempfile
from pathlib import Path

from memory_gate.storage.vector_store import VectorMemoryStore
# If LearningContext or other common types are needed for fixture type hints:
# from memory_gate.memory_protocols import LearningContext


@pytest.fixture(scope="module")
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

@pytest.fixture # Reverted to function scope
def temp_chroma_directory(temp_chroma_directory_factory) -> Path:
    """Create a function-scoped temporary directory for ChromaDB."""
    return temp_chroma_directory_factory("chroma_func_") # Use a prefix for clarity


@pytest_asyncio.fixture # Reverted to function scope
async def persistent_vector_store(temp_chroma_directory: Path) -> VectorMemoryStore:
    """Create a VectorMemoryStore with persistence for testing."""
    store = VectorMemoryStore(
        collection_name="test_persistent_collection",
        persist_directory=str(temp_chroma_directory),
        embedding_model_name="all-MiniLM-L6-v2",
    )
    # Optional: clear data if collection might persist in some unexpected way
    # await store.clear_all_data() # Assuming such a method exists or implement one
    return store


# Pre-initialize for in_memory_vector_store to separate sync init from async fixture execution
_pre_in_memory_vector_store = VectorMemoryStore(
    collection_name="test_in_memory_pre_init_collection",
    persist_directory=None,
    embedding_model_name="all-MiniLM-L6-v2",
)

@pytest_asyncio.fixture # Reverted to function scope
async def in_memory_vector_store() -> VectorMemoryStore:
    """Create an in-memory VectorMemoryStore for testing (no persistence)."""
    # Here, we would ideally clear the state of _pre_in_memory_vector_store
    # For now, tests using it must be aware it might share state if not cleared in tests.
    # Or, we re-initialize it here, which defeats the purpose of pre-initialization.
    # Let's try returning the pre-initialized one directly for diagnostics.
    # This means tests using this fixture will share the same instance if not careful.
    # A better pre-init would involve a factory or ensuring clear_all_data_dangerous is called.
    # For diagnostic purposes:
    # await _pre_in_memory_vector_store.clear_all_data_dangerous() # if clear method is async and works
    # This pre-initialization was problematic. Reverting to standard fixture for now.
    # Tests will need to manage clearing this if it's shared or use it carefully.
    # For function scope, this should be fine as it's recreated.
    store = VectorMemoryStore(
        collection_name="test_in_memory_collection", # Use a distinct name per call if needed
        persist_directory=None,
        embedding_model_name="all-MiniLM-L6-v2",
    )
    return store
