"""Shared embedding model catalog aligned with memory-gate-rs (mg/embed-catalog@STABLE)."""

from dataclasses import dataclass

# Stable catalog entries: id -> (sentence_transformers_load_name, dimension)
_CATALOG: dict[str, tuple[str, int]] = {
    "all-minilm-l6-v2": ("all-MiniLM-L6-v2", 384),
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", 384),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", 768),
}

SUPPORTED_STABLE_IDS: tuple[str, ...] = tuple(_CATALOG.keys())


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Resolved embedding model from the shared catalog."""

    stable_id: str
    st_name: str
    dimension: int


def _normalize_key(name: str) -> str:
    normalized = name.strip().lower()
    stripped = normalized.removeprefix("sentence-transformers/")
    if stripped.startswith("baai/"):
        stripped = stripped[len("baai/") :]
    return stripped


def resolve_model(name: str) -> CatalogEntry:
    """Resolve a stable ID, SentenceTransformers name, or alias to a catalog entry.

    Args:
        name: Catalog stable ID, HF/ST load name, or short alias (case-insensitive).

    Returns:
        CatalogEntry with stable_id, st_name, and dimension.

    Raises:
        ValueError: If the name is not in the catalog, with supported stable IDs listed.
    """
    key = _normalize_key(name)

    stable_id: str | None = key if key in _CATALOG else None
    if stable_id is None:
        match key:
            case "allminilml6v2" | "minilm" | "minilm-l6":
                stable_id = "all-minilm-l6-v2"
            case "bgesmallenv15" | "bge-small" | "bge_small_en_v1.5":
                stable_id = "bge-small-en-v1.5"
            case "bgebaseenv15" | "bge-base" | "bge_base_en_v1.5":
                stable_id = "bge-base-en-v1.5"
            case _:
                stable_id = None

    if stable_id is None:
        supported = ", ".join(SUPPORTED_STABLE_IDS)
        raise ValueError(
            f"unknown embedding model '{name}'; supported stable IDs: {supported}"
        )

    st_name, dimension = _CATALOG[stable_id]
    return CatalogEntry(stable_id=stable_id, st_name=st_name, dimension=dimension)