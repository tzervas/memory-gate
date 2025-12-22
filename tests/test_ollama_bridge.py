"""Unit tests for the Ollama Memory Bridge.

Tests the OllamaMemoryBridge class with mocked Ollama API responses.
"""

from collections.abc import AsyncGenerator
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import LearningContext
from memory_gate.ollama_bridge import (
    ChatMessage,
    OllamaBridgeConfig,
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaMemoryBridge,
    OllamaResponse,
    create_memory_bridge,
)


@pytest.fixture
def mock_memory_gateway() -> MagicMock:
    """Create a mock MemoryGateway for testing."""
    gateway = MagicMock(spec=MemoryGateway)
    gateway.store = MagicMock()
    gateway.store.retrieve_context = AsyncMock(return_value=[])
    gateway.learn_from_interaction = AsyncMock()
    return gateway


@pytest.fixture
def sample_memories() -> list[LearningContext]:
    """Create sample memories for testing."""
    return [
        LearningContext(
            content="Previous discussion about Python async programming",
            domain="conversation",
            timestamp=datetime.now(),
            importance=0.8,
            metadata={"source": "test"},
        ),
        LearningContext(
            content="User prefers detailed explanations with examples",
            domain="conversation",
            timestamp=datetime.now(),
            importance=0.9,
            metadata={"source": "test"},
        ),
    ]


@pytest.fixture
def bridge_config() -> OllamaBridgeConfig:
    """Create a test configuration."""
    return OllamaBridgeConfig(
        ollama_base_url="http://localhost:11434",
        default_model="llama3.2",
        memory_retrieval_limit=5,
        timeout_seconds=30,
        store_interactions=True,
    )


@pytest.fixture
def ollama_bridge(
    mock_memory_gateway: MagicMock, bridge_config: OllamaBridgeConfig
) -> OllamaMemoryBridge:
    """Create an OllamaMemoryBridge instance for testing."""
    return OllamaMemoryBridge(
        memory_gateway=mock_memory_gateway,
        config=bridge_config,
    )


class TestOllamaBridgeConfig:
    """Tests for OllamaBridgeConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = OllamaBridgeConfig()
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.default_model == "llama3.2"
        assert config.memory_retrieval_limit == 5
        assert config.timeout_seconds == 120
        assert config.stream_by_default is True
        assert config.store_interactions is True
        assert config.interaction_domain == "conversation"
        assert "{memories}" in config.memory_injection_template
        assert "{prompt}" in config.memory_injection_template

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = OllamaBridgeConfig(
            ollama_base_url="http://custom:8080",
            default_model="mistral",
            memory_retrieval_limit=10,
            timeout_seconds=60,
            store_interactions=False,
        )
        assert config.ollama_base_url == "http://custom:8080"
        assert config.default_model == "mistral"
        assert config.memory_retrieval_limit == 10
        assert config.timeout_seconds == 60
        assert config.store_interactions is False


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_basic_message(self) -> None:
        """Test basic message creation."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.images is None

    def test_to_dict_without_images(self) -> None:
        """Test conversion to dict without images."""
        msg = ChatMessage(role="user", content="Hello")
        result = msg.to_dict()
        assert result == {"role": "user", "content": "Hello"}
        assert "images" not in result

    def test_to_dict_with_images(self) -> None:
        """Test conversion to dict with images."""
        msg = ChatMessage(role="user", content="Hello", images=["base64data"])
        result = msg.to_dict()
        assert result == {"role": "user", "content": "Hello", "images": ["base64data"]}


class TestOllamaResponse:
    """Tests for OllamaResponse dataclass."""

    def test_minimal_response(self) -> None:
        """Test response with minimal data."""
        response = OllamaResponse(
            model="llama3.2",
            response="Hello!",
            done=True,
        )
        assert response.model == "llama3.2"
        assert response.response == "Hello!"
        assert response.done is True
        assert response.context is None
        assert response.total_duration is None

    def test_full_response(self) -> None:
        """Test response with all fields."""
        response = OllamaResponse(
            model="llama3.2",
            response="Hello!",
            done=True,
            context=[1, 2, 3],
            total_duration=1000000,
            load_duration=500000,
            prompt_eval_count=10,
            eval_count=5,
            eval_duration=400000,
        )
        assert response.context == [1, 2, 3]
        assert response.total_duration == 1000000


class TestOllamaMemoryBridge:
    """Tests for OllamaMemoryBridge."""

    @pytest.mark.asyncio
    async def test_format_memories_empty(
        self, ollama_bridge: OllamaMemoryBridge
    ) -> None:
        """Test formatting empty memory list."""
        result = ollama_bridge._format_memories_for_prompt([])
        assert result == ""

    @pytest.mark.asyncio
    async def test_format_memories_with_content(
        self, ollama_bridge: OllamaMemoryBridge, sample_memories: list[LearningContext]
    ) -> None:
        """Test formatting memories with content."""
        result = ollama_bridge._format_memories_for_prompt(sample_memories)
        assert "[Memory 1]" in result
        assert "[Memory 2]" in result
        assert "Python async programming" in result
        assert "detailed explanations" in result
        assert "importance:" in result

    @pytest.mark.asyncio
    async def test_build_augmented_prompt_no_memories(
        self, ollama_bridge: OllamaMemoryBridge
    ) -> None:
        """Test building prompt without memories."""
        prompt = "What is Python?"
        result = ollama_bridge._build_augmented_prompt(prompt, [])
        assert result == prompt

    @pytest.mark.asyncio
    async def test_build_augmented_prompt_with_memories(
        self, ollama_bridge: OllamaMemoryBridge, sample_memories: list[LearningContext]
    ) -> None:
        """Test building prompt with memories."""
        prompt = "What is Python?"
        result = ollama_bridge._build_augmented_prompt(prompt, sample_memories)
        assert prompt in result
        assert "memories" in result.lower()
        assert "Python async programming" in result

    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories(
        self,
        ollama_bridge: OllamaMemoryBridge,
        mock_memory_gateway: MagicMock,
        sample_memories: list[LearningContext],
    ) -> None:
        """Test memory retrieval."""
        mock_memory_gateway.store.retrieve_context.return_value = sample_memories

        result = await ollama_bridge._retrieve_relevant_memories(
            query="Python programming",
            domain_filter="conversation",
            limit=5,
        )

        assert len(result) == 2
        mock_memory_gateway.store.retrieve_context.assert_called_once_with(
            query="Python programming",
            limit=5,
            domain_filter="conversation",
        )

    @pytest.mark.asyncio
    async def test_retrieve_memories_handles_error(
        self,
        ollama_bridge: OllamaMemoryBridge,
        mock_memory_gateway: MagicMock,
    ) -> None:
        """Test that memory retrieval errors are handled gracefully."""
        mock_memory_gateway.store.retrieve_context.side_effect = Exception("DB error")

        result = await ollama_bridge._retrieve_relevant_memories(
            query="Python programming",
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_store_interaction_disabled(
        self,
        mock_memory_gateway: MagicMock,
    ) -> None:
        """Test that interaction storage can be disabled."""
        config = OllamaBridgeConfig(store_interactions=False)
        bridge = OllamaMemoryBridge(memory_gateway=mock_memory_gateway, config=config)

        await bridge._store_interaction(
            prompt="test",
            response="response",
        )

        mock_memory_gateway.learn_from_interaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_interaction_enabled(
        self,
        ollama_bridge: OllamaMemoryBridge,
        mock_memory_gateway: MagicMock,
    ) -> None:
        """Test storing interaction when enabled."""
        await ollama_bridge._store_interaction(
            prompt="test prompt",
            response="test response",
            domain="test_domain",
            importance=0.7,
        )

        mock_memory_gateway.learn_from_interaction.assert_called_once()
        call_args = mock_memory_gateway.learn_from_interaction.call_args
        context = call_args[0][0]
        assert isinstance(context, LearningContext)
        assert "test prompt" in context.content
        assert "test response" in context.content
        assert context.domain == "test_domain"

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, ollama_bridge: OllamaMemoryBridge
    ) -> None:
        """Test health check with successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(
            ollama_bridge, "_get_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            result = await ollama_bridge.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, ollama_bridge: OllamaMemoryBridge
    ) -> None:
        """Test health check with failed response."""
        import httpx

        with patch.object(
            ollama_bridge, "_get_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            result = await ollama_bridge.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_generate_without_memory(
        self, ollama_bridge: OllamaMemoryBridge, mock_memory_gateway: MagicMock
    ) -> None:
        """Test generation without memory augmentation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.2",
            "response": "Hello, I'm an AI assistant.",
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            ollama_bridge, "_get_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            result = await ollama_bridge.generate_without_memory(prompt="Hello")

        assert isinstance(result, OllamaResponse)
        assert result.response == "Hello, I'm an AI assistant."
        # Memory retrieval should not be called
        mock_memory_gateway.store.retrieve_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_with_memory(
        self,
        ollama_bridge: OllamaMemoryBridge,
        mock_memory_gateway: MagicMock,
        sample_memories: list[LearningContext],
    ) -> None:
        """Test generation with memory augmentation."""
        mock_memory_gateway.store.retrieve_context.return_value = sample_memories

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.2",
            "response": "Based on our previous discussion...",
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            ollama_bridge, "_get_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            result = await ollama_bridge.generate(
                prompt="Tell me about Python",
                use_memory=True,
            )

        assert isinstance(result, OllamaResponse)
        mock_memory_gateway.store.retrieve_context.assert_called_once()
        # Verify the prompt was augmented (check the call to post)
        call_args = mock_client.return_value.post.call_args
        payload = call_args[1]["json"]
        assert "memories" in payload["prompt"].lower()

    @pytest.mark.asyncio
    async def test_generate_connection_error(
        self, ollama_bridge: OllamaMemoryBridge
    ) -> None:
        """Test generate raises OllamaConnectionError on connection failure."""
        import httpx

        with patch.object(
            ollama_bridge, "_get_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            with pytest.raises(OllamaConnectionError):
                await ollama_bridge.generate(prompt="Hello", use_memory=False)

    @pytest.mark.asyncio
    async def test_generate_api_error(
        self, ollama_bridge: OllamaMemoryBridge
    ) -> None:
        """Test generate raises OllamaGenerationError on API error."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server error", request=MagicMock(), response=mock_response
            )
        )

        with patch.object(
            ollama_bridge, "_get_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)

            with pytest.raises(OllamaGenerationError):
                await ollama_bridge.generate(prompt="Hello", use_memory=False)

    @pytest.mark.asyncio
    async def test_chat_without_memory(
        self, ollama_bridge: OllamaMemoryBridge, mock_memory_gateway: MagicMock
    ) -> None:
        """Test chat without memory augmentation."""
        messages = [
            ChatMessage(role="user", content="Hello"),
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hi there!"},
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            ollama_bridge, "_get_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            result = await ollama_bridge.chat(
                messages=messages,
                use_memory=False,
            )

        assert isinstance(result, OllamaResponse)
        assert result.response == "Hi there!"
        mock_memory_gateway.store.retrieve_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_with_memory(
        self,
        ollama_bridge: OllamaMemoryBridge,
        mock_memory_gateway: MagicMock,
        sample_memories: list[LearningContext],
    ) -> None:
        """Test chat with memory augmentation."""
        mock_memory_gateway.store.retrieve_context.return_value = sample_memories

        messages = [
            ChatMessage(role="user", content="Tell me about Python"),
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Python is great!"},
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            ollama_bridge, "_get_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            result = await ollama_bridge.chat(
                messages=messages,
                use_memory=True,
            )

        assert isinstance(result, OllamaResponse)
        mock_memory_gateway.store.retrieve_context.assert_called_once()
        # Verify memory context was added as system message
        call_args = mock_client.return_value.post.call_args
        payload = call_args[1]["json"]
        messages_sent = payload["messages"]
        assert any(msg["role"] == "system" for msg in messages_sent)

    @pytest.mark.asyncio
    async def test_list_models(self, ollama_bridge: OllamaMemoryBridge) -> None:
        """Test listing available models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2", "size": 1000000},
                {"name": "mistral", "size": 2000000},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            ollama_bridge, "_get_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            models = await ollama_bridge.list_models()

        assert len(models) == 2
        assert models[0]["name"] == "llama3.2"

    @pytest.mark.asyncio
    async def test_get_embeddings(self, ollama_bridge: OllamaMemoryBridge) -> None:
        """Test getting embeddings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            ollama_bridge, "_get_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            embeddings = await ollama_bridge.get_embeddings("Hello world")

        assert embeddings == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_close(self, ollama_bridge: OllamaMemoryBridge) -> None:
        """Test closing the bridge."""
        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        ollama_bridge._client = mock_client

        await ollama_bridge.close()

        mock_client.aclose.assert_called_once()


class TestCreateMemoryBridge:
    """Tests for the create_memory_bridge convenience function."""

    @pytest.mark.asyncio
    async def test_create_bridge_success(
        self, mock_memory_gateway: MagicMock
    ) -> None:
        """Test creating a bridge with successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("memory_gate.ollama_bridge.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.is_closed = False
            mock_client_class.return_value = mock_client

            bridge = await create_memory_bridge(
                memory_gateway=mock_memory_gateway,
                ollama_url="http://localhost:11434",
                default_model="llama3.2",
            )

        assert isinstance(bridge, OllamaMemoryBridge)
        assert bridge.config.default_model == "llama3.2"

    @pytest.mark.asyncio
    async def test_create_bridge_unhealthy_ollama(
        self, mock_memory_gateway: MagicMock
    ) -> None:
        """Test creating a bridge when Ollama is not responding."""
        import httpx

        with patch("memory_gate.ollama_bridge.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_client.is_closed = False
            mock_client_class.return_value = mock_client

            bridge = await create_memory_bridge(
                memory_gateway=mock_memory_gateway,
            )

        # Bridge should still be created even if Ollama is not responding
        assert isinstance(bridge, OllamaMemoryBridge)
