"""Health check endpoints."""

from datetime import datetime

from fastapi import APIRouter

from memory_gate.api.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse: Service health status.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="0.1.0",
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check() -> HealthResponse:
    """Readiness check endpoint.

    Returns:
        HealthResponse: Service readiness status.
    """
    return HealthResponse(
        status="ready",
        timestamp=datetime.now(),
        version="0.1.0",
    )
