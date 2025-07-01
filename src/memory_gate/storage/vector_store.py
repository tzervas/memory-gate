import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, cast

import chromadb  # type: ignore[import-not-found]
from chromadb.config import Settings  # type: ignore[import-not-found]

# from chromadb.api import Collection  # type: ignore[import-not-found] - Reserved for future use
from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

from memory_gate.memory_protocols import KnowledgeStore, LearningContext
from memory_gate.metrics import (
    MEMORY_ITEMS_COUNT,
    MEMORY_RETRIEVAL_LATENCY_SECONDS,
    MEMORY_STORE_LATENCY_SECONDS,
    record_memory_operation,
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for VectorMemoryStore.

    Args:
        collection_name: Name of the ChromaDB collection to use
        embedding_model_name: Name of the sentence-transformer model for embeddings
        persist_directory: Directory to persist ChromaDB data. If None, uses in-memory
        collection_metadata: Additional metadata for the collection
        chroma_settings: Additional settings for ChromaDB client
        max_batch_size: Maximum number of items to process in a single batch
        embedding_device: Device to use for embedding generation ('cpu' or 'cuda')
    """

    collection_name: str = "memory_gate_default_collection"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    persist_directory: str | None = "./data/chromadb_store"
    collection_metadata: dict[str, Any] | None = None
    chroma_settings: dict[str, Any] | None = None
    max_batch_size: int = 100
    embedding_device: str = "cpu"  # or "cuda" for GPU support


class VectorStoreError(Exception):
    """Base exception for VectorMemoryStore errors."""


class VectorStoreInitError(VectorStoreError):
    """Raised when VectorMemoryStore initialization fails."""


class VectorStoreOperationError(VectorStoreError):
    """Raised when a VectorMemoryStore operation fails."""


class VectorMemoryStore(KnowledgeStore[LearningContext]):
    """
    Production vector storage with ChromaDB backend.
    Implements the KnowledgeStore protocol for LearningContext.
    """

    def __init__(
        self,
        config: VectorStoreConfig,
    ) -> None:
        """
        Initializes the VectorMemoryStore.

        Args:
            config: Configuration object for the vector store.
        """
        self.config = config

        try:
            if self.config.persist_directory:
                self.client = chromadb.PersistentClient(
                    path=self.config.persist_directory,
                    settings=Settings(
                        **(
                            self.config.chroma_settings
                            or {"anonymized_telemetry": False}
                        )
                    ),
                )
            else:
                self.client = chromadb.Client(
                    settings=Settings(
                        **(
                            self.config.chroma_settings
                            or {"anonymized_telemetry": False}
                        )
                    )
                )
        except Exception as e:
            raise VectorStoreInitError(
                f"Failed to initialize ChromaDB client: {e}"
            ) from e

        try:
            self.embedding_model = SentenceTransformer(
                self.config.embedding_model_name, device=self.config.embedding_device
            )
        except Exception as e:
            raise VectorStoreInitError(
                f"Failed to initialize embedding model: {e}"
            ) from e

        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata=self.config.collection_metadata
            or {"description": "MemoryGate learning storage"},
        )
        # Initialize item count gauge
        MEMORY_ITEMS_COUNT.labels(
            store_type="vector_store", collection_name=self.config.collection_name
        ).set_function(
            lambda: self.collection.count()  # Periodically update with current count
        )

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generates embedding for a given text using sentence-transformer model."""
        loop = asyncio.get_event_loop()
        # SentenceTransformer.encode is CPU-bound, run in executor
        embedding = await loop.run_in_executor(None, self.embedding_model.encode, text)
        return cast("list[float]", embedding.tolist())

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
        domain_filter: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[LearningContext]:
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

                where_clause: dict[str, Any] = {}
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

            contexts: list[LearningContext] = []
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
                                metadata.get("importance", 1.0)
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

    async def get_experience_by_id(self, key: str) -> LearningContext | None:
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
                importance=float(metadata.get("importance", 1.0)),
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
        return cast("int", self.collection.count())

    async def get_experiences_by_metadata_filter(
        self,
        metadata_filter: dict[str, Any],
        limit: int = 100,  # Default limit to avoid fetching too much
        offset: int = 0,
    ) -> list[tuple[str, LearningContext]]:
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

            items: list[tuple[str, LearningContext]] = []
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
                        importance=float(metadata.get("importance", 1.0)),
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
