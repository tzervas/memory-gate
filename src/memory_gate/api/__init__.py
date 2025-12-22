"""REST API module for MemoryGate.

This module provides a FastAPI-based REST API for memory operations,
prompt augmentation, and provider-agnostic model generation.
"""

from memory_gate.api.app import create_app

__all__ = ["create_app"]
