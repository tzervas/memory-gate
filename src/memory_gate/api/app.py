"""FastAPI application factory for MemoryGate REST API."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from memory_gate.api.routes import generate, health, memory

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Lifespan context manager for application startup and shutdown."""
    logger.info("Starting MemoryGate API")
    yield
    logger.info("Shutting down MemoryGate API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(
        title="MemoryGate API",
        description="Dynamic memory learning layer for AI systems",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(memory.router, prefix="/api/v1", tags=["Memory"])
    app.include_router(generate.router, prefix="/api/v1", tags=["Generation"])

    return app
