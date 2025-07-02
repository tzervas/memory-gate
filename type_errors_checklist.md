# Type Errors Checklist for vector_store.py

## MyPy Audit Results
**Date:** Current analysis  
**File:** `/home/vector_weight/Documents/projects/memory-gate/src/memory_gate/storage/vector_store.py`  
**Total Errors Found:** 16

---

## 1. Embeddings Type Errors

### ❌ Line 164: Incompatible embeddings type in upsert()
- **Error:** `Argument "embeddings" to "upsert" of "Collection" has incompatible type "list[list[float]]"; expected "ndarray[Any, dtype[signedinteger[_32Bit] | floating[_32Bit]]] | list[ndarray[Any, dtype[signedinteger[_32Bit] | floating[_32Bit]]]] | Sequence[float] | Sequence[int] | list[Sequence[float] | Sequence[int]] | None"`
- **Issue:** ChromaDB expects numpy arrays or specific sequence types, but we're passing `list[list[float]]`
- **Location:** `store_experience()` method - upsert call
- **Fix Required:** Convert embedding to proper numpy array or compatible sequence type

### ❌ Line 210: Incompatible query_embeddings type in query()
- **Error:** `Argument "query_embeddings" to "query" of "Collection" has incompatible type "list[list[float]]"; expected "ndarray[Any, dtype[signedinteger[_32Bit] | floating[_32Bit]]] | list[ndarray[Any, dtype[signedinteger[_32Bit] | floating[_32Bit]]]] | Sequence[float] | Sequence[int] | list[Sequence[float] | Sequence[int]] | None"`
- **Issue:** Same embedding type issue in query operation
- **Location:** `retrieve_context()` method - query call
- **Fix Required:** Convert query embedding to proper type

---

## 2. Metadata Types Errors

### ❌ Line 166: Incompatible metadata type in upsert()
- **Error:** `List item 0 has incompatible type "dict[str, object]"; expected "Mapping[str, str | int | float | bool]"`
- **Issue:** ChromaDB metadata expects specific value types, but we have `object` type
- **Location:** `store_experience()` method - metadata parameter
- **Fix Required:** Ensure metadata values are properly typed as str | int | float | bool

### ❌ Line 242: Incompatible metadata type in LearningContext constructor
- **Error:** `Argument "metadata" to "LearningContext" has incompatible type "dict[str | Any, str | int | float | Any]"; expected "dict[str, str] | None"`
- **Issue:** LearningContext expects `dict[str, str] | None` but getting broader types
- **Location:** `retrieve_context()` method - LearningContext creation
- **Fix Required:** Convert metadata to expected string-only values or make LearningContext more flexible

### ❌ Line 281: Incompatible metadata type in LearningContext constructor (duplicate)
- **Error:** `Argument "metadata" to "LearningContext" has incompatible type "dict[str | Any, str | int | float | Any]"; expected "dict[str, str] | None"`
- **Issue:** Same metadata type issue in different method
- **Location:** `get_experience_by_id()` method - LearningContext creation
- **Fix Required:** Same as above

### ❌ Line 353: Incompatible metadata type in LearningContext constructor (third occurrence)
- **Error:** `Argument "metadata" to "LearningContext" has incompatible type "dict[str, str | int | float]"; expected "dict[str, str] | None"`
- **Issue:** Similar metadata type mismatch
- **Location:** `get_experiences_by_metadata_filter()` method - LearningContext creation
- **Fix Required:** Same as above

---

## 3. Include Parameters Errors

### ❌ Line 213: Incompatible include parameter types (3 errors)
- **Error 1:** `List item 0 has incompatible type "str"; expected "IncludeEnum"`
- **Error 2:** `List item 1 has incompatible type "str"; expected "IncludeEnum"`
- **Error 3:** `List item 2 has incompatible type "str"; expected "IncludeEnum"`
- **Issue:** ChromaDB expects `IncludeEnum` types instead of string literals
- **Location:** `retrieve_context()` method - query call with `include=["metadatas", "documents", "distances"]`
- **Fix Required:** Import and use proper `IncludeEnum` values

### ❌ Line 258: Incompatible include parameter types (2 errors)
- **Error 1:** `List item 0 has incompatible type "str"; expected "IncludeEnum"`
- **Error 2:** `List item 1 has incompatible type "str"; expected "IncludeEnum"`
- **Issue:** Same include enum issue
- **Location:** `get_experience_by_id()` method - get call with `include=["metadatas", "documents"]`
- **Fix Required:** Same as above

### ❌ Line 332: Incompatible include parameter types (2 errors)
- **Error 1:** `List item 0 has incompatible type "str"; expected "IncludeEnum"`
- **Error 2:** `List item 1 has incompatible type "str"; expected "IncludeEnum"`
- **Issue:** Same include enum issue
- **Location:** `get_experiences_by_metadata_filter()` method - get call with `include=["metadatas", "documents"]`
- **Fix Required:** Same as above

---

## 4. None-Indexing Issues

### ❌ Line 220: Value not indexable
- **Error:** `Value of type "list[list[str]] | None" is not indexable`
- **Issue:** Attempting to index `query_results["documents"][0]` without checking if it's None
- **Location:** `retrieve_context()` method - accessing documents results
- **Fix Required:** Add proper None checks before indexing

### ❌ Line 221: Value not indexable
- **Error:** `Value of type "list[list[Mapping[str, str | int | float | bool]]] | None" is not indexable`
- **Issue:** Attempting to index `query_results["metadatas"][0]` without checking if it's None
- **Location:** `retrieve_context()` method - accessing metadata results
- **Fix Required:** Add proper None checks before indexing

### ❌ Line 268: Value not indexable
- **Error:** `Value of type "list[Mapping[str, str | int | float | bool]] | None" is not indexable`
- **Issue:** Attempting to index `result["metadatas"][0]` without checking if it's None
- **Location:** `get_experience_by_id()` method - accessing metadata results
- **Fix Required:** Add proper None checks before indexing

---

## Summary by Category

| Category | Error Count | Critical Impact |
|----------|-------------|-----------------|
| **Embeddings Types** | 2 | High - Core functionality |
| **Metadata Types** | 4 | Medium - Data integrity |
| **Include Parameters** | 6 | Medium - API compatibility |
| **None-Indexing** | 3 | High - Runtime safety |
| **Other** | 1 | Low |

---

## Priority Fix Order

1. **High Priority - Runtime Safety**
   - None-indexing issues (lines 220, 221, 268)
   - Embeddings type compatibility (lines 164, 210)

2. **Medium Priority - API Compatibility**
   - Include parameter enum types (lines 213, 258, 332)
   - Metadata type compatibility (lines 166, 242, 281, 353)

3. **Low Priority**
   - Review and address remaining type inconsistencies

---

## Required Imports/Dependencies

- Import `IncludeEnum` from ChromaDB
- Import `numpy` for proper array handling
- Review `LearningContext` type definitions for metadata flexibility
