from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory_gate.storage.vector_store import VectorMemoryStore # To avoid circular dependency if VectorMemoryStore imports from helpers
    from memory_gate.metrics import MEMORY_ITEMS_COUNT, record_memory_operation # Assuming these are needed

async def clear_vector_store_collection_dangerous(store: "VectorMemoryStore") -> None:
    """
    Deletes all data in the specified VectorMemoryStore's collection by deleting
    and recreating it. Intended for testing purposes. Use with caution.
    Updates the store's collection attribute in place.
    """
    # These imports are here to avoid them at global scope if this helper is imported widely
    # and to ensure they are available when this function is called.
    # However, for type checking, they might be needed at the top with TYPE_CHECKING.
    # Let's assume they are available in the environment where this helper is used.
    # from memory_gate.metrics import MEMORY_ITEMS_COUNT, record_memory_operation

    # For now, directly use the metrics if they are globally available or part of the store object
    # This might need adjustment based on actual metric accessibility.
    # For simplicity, I'll assume the metrics part needs to be handled carefully
    # or that the store object itself can re-initialize its metrics link.

    # To access MEMORY_ITEMS_COUNT and record_memory_operation, they should be imported
    # or passed, or the store should handle its own metric reset.
    # Let's make them available for now.
    from memory_gate.metrics import MEMORY_ITEMS_COUNT, record_memory_operation

    try:
        store.client.delete_collection(name=store.collection_name)
        store.collection = store.client.get_or_create_collection(
            name=store.collection_name,
            metadata={"description": "MemoryGate learning storage"}
        )
        # Reset the metric function for the new collection instance
        # This assumes MEMORY_ITEMS_COUNT is accessible here.
        MEMORY_ITEMS_COUNT.labels(store_type="vector_store", collection_name=store.collection_name).set_function(
            lambda: store.collection.count()
        )
        record_memory_operation(operation_type="clear_vector_store_collection_dangerous", success=True)
        # print(f"Test Helper: Collection '{store.collection_name}' cleared and recreated.")
    except Exception as e:
        record_memory_operation(operation_type="clear_vector_store_collection_dangerous", success=False)
        print(f"Test Helper: Error clearing collection '{store.collection_name}': {e}")
        try:
            store.collection = store.client.get_or_create_collection(
                name=store.collection_name,
                metadata={"description": "MemoryGate learning storage"}
            )
            MEMORY_ITEMS_COUNT.labels(store_type="vector_store", collection_name=store.collection_name).set_function(
                lambda: store.collection.count()
            )
        except Exception as e_recreate:
            print(f"Test Helper: Failed to recreate collection '{store.collection_name}' after delete error: {e_recreate}")
            raise e_recreate
