#!/usr/bin/env python3
"""Optional startup guard for ChromaDB CVE-2026-45829 (client-only enforcement).

Invoke before starting memory-gate when remote Chroma server mode might be configured:

    uv run python docs/security/chromadb_startup_guard.py

Environment variables:
    MEMORY_GATE_CHROMA_SERVER_WARN: Set to "0" to disable warnings (not recommended).
    MEMORY_GATE_ENFORCE_CLIENT_ONLY: Set to "1" to exit with code 1 on remote-server signals.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("memory_gate.security.chromadb")

# Signals that memory-gate or Chroma may be configured for remote HTTP server mode.
_REMOTE_CHROMA_ENV_VARS: tuple[str, ...] = (
    "CHROMA_SERVER_HOST",
    "CHROMA_HTTP_HOST",
    "CHROMA_HTTP_PORT",
    "CHROMA_HOST",
    "CHROMA_PORT",
    "CHROMA_CLIENT_AUTH_PROVIDER",
    "CHROMA_CLIENT_AUTH_CREDENTIALS",
)

_CVE_REFERENCE = "CVE-2026-45829 (unpatched upstream; see SECURITY.md)"


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def detect_remote_chroma_configuration() -> list[str]:
    """Return environment variable names indicating remote Chroma server usage."""
    return [name for name in _REMOTE_CHROMA_ENV_VARS if os.getenv(name)]


def emit_startup_warning(detected: list[str]) -> None:
    """Log a critical warning when remote Chroma server configuration is present."""
    if not detected:
        logger.info(
            "ChromaDB startup guard: embedded client-only configuration detected "
            "(no remote server env vars). %s mitigations apply to server mode only.",
            _CVE_REFERENCE,
        )
        return

    logger.critical(
        "ChromaDB REMOTE SERVER MODE detected via environment: %s. "
        "%s allows pre-auth RCE on Python Chroma servers and client-side RCE via "
        "poisoned collections. Memory-gate defaults to embedded PersistentClient; "
        "unset these variables or complete formal risk acceptance (SECURITY.md).",
        ", ".join(detected),
        _CVE_REFERENCE,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    warn_enabled = os.getenv("MEMORY_GATE_CHROMA_SERVER_WARN", "1")
    enforce = _truthy(os.getenv("MEMORY_GATE_ENFORCE_CLIENT_ONLY"))

    detected = detect_remote_chroma_configuration()

    if _truthy(warn_enabled) or enforce:
        emit_startup_warning(detected)

    if enforce and detected:
        logger.critical(
            "MEMORY_GATE_ENFORCE_CLIENT_ONLY=1: refusing to start with remote Chroma "
            "configuration."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())