from typing import Any, List, Optional, Dict, cast
import asyncio
import chromadb  # type: ignore[import-not-found]
from chromadb.config import Settings  # type: ignore[import-not-found]
from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
from datetime import datetime

from memory_gate.memory_protocols import KnowledgeStore, LearningContext
from memory_gate.metrics import (
    MEMORY_STORE_LATENCY_SECONDS,
    MEMORY_RETRIEVAL_LATENCY_SECONDS,
    MEMORY_ITEMS_COUNT,
    record_memory_operation,
)


class VectorMemoryStore(KnowledgeStore[LearningContext]):
    """
    Production vector storage with ChromaDB backend.
    Implements the KnowledgeStore protocol for LearningContext.
    """

    def __init__(
        self,
        collection_name: str = "memory_gate_default_collection",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = "./data/chromadb_store",
        chroma_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the VectorMemoryStore.

        Args:
            collection_name: Name of the ChromaDB collection to use.
            embedding_model_name: Name of the sentence-transformer model for embeddings.
            persist_directory: Directory to persist ChromaDB data. If None, uses in-memory.
            chroma_settings: Additional settings for ChromaDB client.
        """
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)

        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    **(chroma_settings or {"anonymized_telemetry": False})
                ),
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(
                    **(chroma_settings or {"anonymized_telemetry": False})
                )
            )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "MemoryGate learning storage"},
            # embedding_function is not set here, we generate embeddings manually
        )
        # Initialize item count gauge
        MEMORY_ITEMS_COUNT.labels(
            store_type="vector_store", collection_name=self.collection_name
        ).set_function(
            lambda: self.collection.count()  # Periodically update with current count
        )

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generates embedding for a given text using sentence-transformer model."""
        loop = asyncio.get_event_loop()
        # SentenceTransformer.encode is CPU-bound, run in executor
        embedding = await loop.run_in_executor(None, self.embedding_model.encode, text)
        return cast(List[float], embedding.tolist())

    async def store_experience(self, key: str, experience: LearningContext) -> None:
        """
        Stores a learning experience with its vector embedding.

        Args:
            key: Unique identifier for the experience.
            experience: The LearningContext object to store.
        """
        try:
            with MEMORY_STORE_LATENCY_SECONDS.labels(store_type="vector_store").time():
                embedding = await self._generate_embedding(experience.content)

                metadata_to_store = {
                    "domain": experience.domain,
                    "timestamp": experience.timestamp.isoformat(),
                    "importance": experience.importance,
                    **(experience.metadata or {}),
                }

                self.collection.upsert(
                    ids=[key],
                    embeddings=[embedding],
                    documents=[experience.content],
                    metadatas=[metadata_to_store],
                )
            record_memory_operation(operation_type="store_experience", success=True)
        except Exception as e:
            record_memory_operation(operation_type="store_experience", success=False)
            # Optionally re-raise the exception or handle it
            print(f"Error storing experience {key}: {e}")
            raise  # Re-raise for now

    async def retrieve_context(
        self,
        query: str,
        limit: int = 10,
        domain_filter: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[LearningContext]:
        """
        Retrieves relevant context using vector similarity search.

        Args:
            query: The query string to search for.
            limit: Maximum number of results to return.
            domain_filter: Optional filter for the 'domain' metadata field.
            metadata_filter: Optional dictionary for more complex metadata filtering.

        Returns:
            A list of LearningContext objects matching the query.
        """
        try:
            with MEMORY_RETRIEVAL_LATENCY_SECONDS.labels(
                store_type="vector_store"
            ).time():
                query_embedding = await self._generate_embedding(query)

                where_clause: Dict[str, Any] = {}
                if domain_filter:
                    where_clause["domain"] = domain_filter

                if metadata_filter:
                    if where_clause and metadata_filter:
                        where_clause = {"$and": [where_clause, metadata_filter]}
                    elif metadata_filter:
                        where_clause = metadata_filter

                query_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=where_clause if where_clause else None,
                    include=["metadatas", "documents", "distances"],
                )

            contexts: List[LearningContext] = []
            if (
                query_results["ids"] and query_results["ids"][0]
            ):  # Check if there are any results
                for i, doc_content in enumerate(query_results["documents"][0]):
                    metadata = query_results["metadatas"][0][i]

                    # Reconstruct original metadata, excluding ChromaDB specific or already mapped fields
                    original_metadata = {
                        k: v
                        for k, v in metadata.items()
                        if k not in ["domain", "timestamp", "importance"]
                    }

                    contexts.append(
                        LearningContext(
                            content=doc_content,
                            domain=str(
                                metadata.get("domain", "unknown")
                            ),  # handle missing domain
                            timestamp=datetime.fromisoformat(
                                str(metadata["timestamp"])
                            ),
                            importance=float(
                                str(metadata.get("importance", 1.0))
                            ),  # handle missing importance
                            metadata=original_metadata,
                        )
                    )
            record_memory_operation(operation_type="retrieve_context", success=True)
            return contexts
        except Exception as e:
            record_memory_operation(operation_type="retrieve_context", success=False)
            print(f"Error retrieving context for query '{query}': {e}")
            raise  # Re-raise for now

    async def get_experience_by_id(self, key: str) -> Optional[LearningContext]:
        """Retrieves a specific experience by its key."""
        try:
            with MEMORY_RETRIEVAL_LATENCY_SECONDS.labels(
                store_type="vector_store"
            ).time():
                result = self.collection.get(
                    ids=[key], include=["metadatas", "documents"]
                )

            if not result["ids"] or not result["documents"]:
                record_memory_operation(
                    operation_type="get_experience_by_id", success=True
                )  # Success, but not found
                return None

            doc_content = result["documents"][0]
            metadata = result["metadatas"][0]

            original_metadata = {
                k: v
                for k, v in metadata.items()
                if k not in ["domain", "timestamp", "importance"]
            }

            lc = LearningContext(
                content=doc_content,
                domain=str(metadata.get("domain", "unknown")),
                timestamp=datetime.fromisoformat(str(metadata["timestamp"])),
                importance=float(str(metadata.get("importance", 1.0))),
                metadata=original_metadata,
            )
            record_memory_operation(operation_type="get_experience_by_id", success=True)
            return lc
        except Exception as e:
            record_memory_operation(
                operation_type="get_experience_by_id", success=False
            )
            print(f"Error retrieving experience by ID '{key}': {e}")
            raise

    async def delete_experience(self, key: str) -> None:
        """Deletes an experience by its key."""
        try:
            # No specific latency metric for delete, but count as an operation
            self.collection.delete(ids=[key])
            record_memory_operation(operation_type="delete_experience", success=True)
        except Exception as e:
            record_memory_operation(operation_type="delete_experience", success=False)
            print(f"Error deleting experience '{key}': {e}")
            raise

    def get_collection_size(self) -> int:
        """Returns the number of items in the collection."""
        # This is now handled by the MEMORY_ITEMS_COUNT gauge's set_function
        return cast(int, self.collection.count())

    async def get_experiences_by_metadata_filter(
        self,
        metadata_filter: Dict[str, Any],
        limit: int = 100,  # Default limit to avoid fetching too much
        offset: int = 0,
    ) -> List[tuple[str, LearningContext]]:
        """
        Retrieves experiences (ID and LearningContext) based on a metadata filter.
        Uses ChromaDB's get() method which can filter on metadata.

        Args:
            metadata_filter: The filter to apply (e.g., {"importance": {"$lt": 0.5}}).
            limit: Maximum number of results to return.
            offset: Offset for pagination.

        Returns:
            A list of tuples, where each tuple is (id, LearningContext).
        """
        try:
            with MEMORY_RETRIEVAL_LATENCY_SECONDS.labels(
                store_type="vector_store"
            ).time():  # Classify as retrieval
                results = self.collection.get(
                    where=metadata_filter,
                    limit=limit,
                    offset=offset,
                    include=["metadatas", "documents"],  # IDs are included by default
                )

            items: List[tuple[str, LearningContext]] = []
            if (
                results["ids"] and results["documents"] and results["metadatas"]
            ):  # Ensure all lists are present and non-empty
                for i, item_id in enumerate(results["ids"]):
                    doc_content = results["documents"][i]
                    metadata = results["metadatas"][i]

                    original_metadata = {
                        k: v
                        for k, v in metadata.items()
                        if k not in ["domain", "timestamp", "importance"]
                    }
                    context = LearningContext(
                        content=doc_content,
                        domain=str(metadata.get("domain", "unknown")),
                        timestamp=datetime.fromisoformat(str(metadata["timestamp"])),
                        importance=float(str(metadata.get("importance", 1.0))),
                        metadata=original_metadata,
                    )
                    items.append((item_id, context))
            record_memory_operation(
                operation_type="get_experiences_by_metadata_filter", success=True
            )
            return items
        except Exception as e:
            record_memory_operation(
                operation_type="get_experiences_by_metadata_filter", success=False
            )
            print(f"Error in get_experiences_by_metadata_filter: {e}")
            raise  # Re-raise for now

    # The following methods (get_experience_by_id, delete_experience, get_collection_size)
    # were duplicated in the file content provided by read_files.
    # I am assuming the version with metrics (earlier in the file) is the correct one.
    # I will remove the duplicated, non-metric versions if they appear after this fixed method.
    # This tool does not allow me to delete the duplicated methods directly without another read.
    # For now, I'm only fixing the identified syntax error location.
    # The next read_files or grep will confirm if duplicates exist and need removal.

    # Duplicated get_experience_by_id was here in my mental model of the file.
    # Duplicated delete_experience was here.
    # Duplicated get_collection_size was here.
    # Duplicated get_experiences_by_metadata_filter was here.


# Example of how to use (for testing or direct script execution)
async def main_test() -> None:
    print("Initializing VectorMemoryStore (in-memory ChromaDB)...")
    # Using persist_directory=None for in-memory for this test
    vector_store = VectorMemoryStore(
        collection_name="test_run_collection", persist_directory=None
    )

    print(f"Initial collection size: {vector_store.get_collection_size()}")

    # Example LearningContext objects
    context1 = LearningContext(
        content="User prefers dark mode in applications.",
        domain="preferences",
        timestamp=datetime.now(),
        importance=0.8,
        metadata={"user_id": "user123", "app_version": "1.2.0"},
    )
    context2 = LearningContext(
        content="Common error in deployment script: file not found.",
        domain="devops",
        timestamp=datetime.now(),
        importance=0.95,
        metadata={"script_name": "deploy.sh", "error_code": "ENOENT"},
    )
    context3 = LearningContext(
        content="User frequently asks about Python syntax.",
        domain="support",
        timestamp=datetime.now(),
        importance=0.7,
        metadata={"user_id": "user456", "language": "python"},
    )

    print("\nStoring experiences...")
    await vector_store.store_experience("pref_dark_mode", context1)
    await vector_store.store_experience("devops_error_enoent", context2)
    await vector_store.store_experience("support_python_syntax", context3)

    print(f"Collection size after storing: {vector_store.get_collection_size()}")

    print("\nRetrieving context for 'python error':")
    retrieved_contexts = await vector_store.retrieve_context("python error", limit=2)
    for ctx in retrieved_contexts:
        print(
            f"  - Content: {ctx.content}, Domain: {ctx.domain}, Importance: {ctx.importance}, Meta: {ctx.metadata}"
        )

    print("\nRetrieving context for 'deployment script' with domain 'devops':")
    retrieved_devops = await vector_store.retrieve_context(
        "deployment script", limit=1, domain_filter="devops"
    )
    for ctx in retrieved_devops:
        print(
            f"  - Content: {ctx.content}, Domain: {ctx.domain}, Importance: {ctx.importance}"
        )

    print("\nRetrieving experience by ID 'pref_dark_mode':")
    specific_context = await vector_store.get_experience_by_id("pref_dark_mode")
    if specific_context:
        print(f"  - Found: {specific_context.content}")

    print("\nDeleting experience 'support_python_syntax'...")
    await vector_store.delete_experience("support_python_syntax")
    print(f"Collection size after deletion: {vector_store.get_collection_size()}")

    print("\nAttempting to retrieve deleted experience:")
    deleted_retrieval = await vector_store.get_experience_by_id("support_python_syntax")
    print(f"  - Found: {'Yes' if deleted_retrieval else 'No'}")

    print("\nTest finished.")


if __name__ == "__main__":
    # This allows running the file directly for quick tests if needed.
    # Note: top-level await is Python 3.8+, ensure your environment supports it if you run this directly
    # or use asyncio.run(main_test())
    asyncio.run(main_test())
