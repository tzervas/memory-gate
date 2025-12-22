"""Tests for the MemoryGate REST API.

Tests all API endpoints including health checks, memory operations,
and provider-agnostic generation.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from memory_gate.api.app import create_app
from memory_gate.api.dependencies import configure_gateway
from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import LearningContext
from memory_gate.providers.base import ProviderResponse, ProviderType


@pytest.fixture
def mock_memory_gateway() -> MagicMock:
    """Create a mock MemoryGateway for API testing."""
    gateway = MagicMock(spec=MemoryGateway)
    gateway.store = MagicMock()
    gateway.store.retrieve_context = AsyncMock(return_value=[])
    gateway.store.store_context = AsyncMock()
    return gateway


@pytest.fixture
def client(mock_memory_gateway: MagicMock) -> TestClient:
    """Create a test client with mocked dependencies."""
    configure_gateway(mock_memory_gateway)
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_memory() -> LearningContext:
    """Create a sample memory for testing."""
    return LearningContext(
        content="Test memory content",
        domain="test",
        timestamp=datetime.now(),
        importance=0.8,
        metadata={"source": "test"},
    )


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"

    def test_readiness_check(self, client: TestClient) -> None:
        """Test the readiness check endpoint."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "timestamp" in data


class TestMemoryEndpoints:
    """Test memory operation endpoints."""

    def test_query_memories_empty(
        self,
        client: TestClient,
        mock_memory_gateway: MagicMock,
    ) -> None:
        """Test querying memories with no results."""
        mock_memory_gateway.store.retrieve_context.return_value = []

        response = client.post(
            "/api/v1/memory/query",
            json={"query": "test query", "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["memories"] == []

    def test_query_memories_with_results(
        self,
        client: TestClient,
        mock_memory_gateway: MagicMock,
        sample_memory: LearningContext,
    ) -> None:
        """Test querying memories with results."""
        mock_memory_gateway.store.retrieve_context.return_value = [sample_memory]

        response = client.post(
            "/api/v1/memory/query",
            json={"query": "test query", "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["memories"]) == 1
        assert data["memories"][0]["content"] == "Test memory content"
        assert data["memories"][0]["importance"] == 0.8

    def test_query_memories_with_limit(
        self,
        client: TestClient,
        mock_memory_gateway: MagicMock,
    ) -> None:
        """Test that query limit is respected."""
        response = client.post(
            "/api/v1/memory/query",
            json={"query": "test query", "limit": 5},
        )

        assert response.status_code == 200
        mock_memory_gateway.store.retrieve_context.assert_called_once()
        call_kwargs = mock_memory_gateway.store.retrieve_context.call_args.kwargs
        assert call_kwargs["limit"] == 5

    def test_store_memory_success(
        self,
        client: TestClient,
        mock_memory_gateway: MagicMock,
    ) -> None:
        """Test storing a memory successfully."""
        response = client.post(
            "/api/v1/memory/store",
            json={
                "content": "New memory content",
                "domain": "test",
                "importance": 0.7,
                "metadata": {"key": "value"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "successfully" in data["message"]
        mock_memory_gateway.store.store_context.assert_called_once()

    def test_store_memory_with_defaults(
        self,
        client: TestClient,
        mock_memory_gateway: MagicMock,
    ) -> None:
        """Test storing a memory with default values."""
        response = client.post(
            "/api/v1/memory/store",
            json={"content": "Simple memory"},
        )

        assert response.status_code == 200
        mock_memory_gateway.store.store_context.assert_called_once()
        stored_context = mock_memory_gateway.store.store_context.call_args.args[0]
        assert stored_context.content == "Simple memory"
        assert stored_context.domain == "conversation"
        assert stored_context.importance == 0.5

    def test_augment_prompt_no_memories(
        self,
        client: TestClient,
        mock_memory_gateway: MagicMock,
    ) -> None:
        """Test augmenting a prompt with no memories."""
        mock_memory_gateway.store.retrieve_context.return_value = []

        response = client.post(
            "/api/v1/memory/augment",
            json={"prompt": "Test prompt", "limit": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["original_prompt"] == "Test prompt"
        assert data["augmented_prompt"] == "Test prompt"
        assert data["memories_used"] == 0

    def test_augment_prompt_with_memories(
        self,
        client: TestClient,
        mock_memory_gateway: MagicMock,
        sample_memory: LearningContext,
    ) -> None:
        """Test augmenting a prompt with memories."""
        mock_memory_gateway.store.retrieve_context.return_value = [sample_memory]

        response = client.post(
            "/api/v1/memory/augment",
            json={"prompt": "Test prompt", "limit": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["original_prompt"] == "Test prompt"
        assert "Based on relevant memories:" in data["augmented_prompt"]
        assert "Test memory content" in data["augmented_prompt"]
        assert data["memories_used"] == 1


class TestGenerateEndpoints:
    """Test generation endpoints."""

    @patch("memory_gate.api.routes.generate.get_provider")
    def test_generate_without_memory(
        self,
        mock_get_provider: MagicMock,
        client: TestClient,
        mock_memory_gateway: MagicMock,
    ) -> None:
        """Test generation without memory augmentation."""
        # Mock provider
        mock_provider = MagicMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock()
        mock_provider.generate = AsyncMock(
            return_value=ProviderResponse(
                content="Generated response",
                model="test-model",
                provider=ProviderType.OLLAMA,
            ),
        )
        mock_get_provider.return_value = mock_provider

        response = client.post(
            "/api/v1/generate",
            json={
                "prompt": "Test prompt",
                "provider": "ollama",
                "use_memory": False,
            },
        )

        # Debug: print response details if it fails
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Generated response"
        assert data["provider"] == "ollama"
        assert data["memories_used"] == 0

    @patch("memory_gate.api.routes.generate.get_provider")
    def test_generate_with_memory(
        self,
        mock_get_provider: MagicMock,
        client: TestClient,
        mock_memory_gateway: MagicMock,
        sample_memory: LearningContext,
    ) -> None:
        """Test generation with memory augmentation."""
        # Mock memories
        mock_memory_gateway.store.retrieve_context.return_value = [sample_memory]

        # Mock provider
        mock_provider = MagicMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock()
        mock_provider.generate = AsyncMock(
            return_value=ProviderResponse(
                content="Generated with memory",
                model="test-model",
                provider=ProviderType.OLLAMA,
            ),
        )
        mock_get_provider.return_value = mock_provider

        response = client.post(
            "/api/v1/generate",
            json={
                "prompt": "Test prompt",
                "provider": "ollama",
                "use_memory": True,
                "model": "test-model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Generated with memory"
        assert data["memories_used"] == 1

        # Verify the prompt was augmented
        call_args = mock_provider.generate.call_args
        augmented_prompt = call_args.kwargs["prompt"]
        assert "Based on relevant memories:" in augmented_prompt
        assert "Test memory content" in augmented_prompt


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation generation."""

    def test_openapi_schema(self, client: TestClient) -> None:
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "MemoryGate API"
        assert schema["info"]["version"] == "0.1.0"
        assert "paths" in schema

    def test_docs_endpoint(self, client: TestClient) -> None:
        """Test that Swagger UI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_endpoint(self, client: TestClient) -> None:
        """Test that ReDoc docs are available."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
