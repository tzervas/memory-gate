"""Tests for configuration management."""

import os
from pathlib import Path
import tempfile

import pytest

from memory_gate.config import (
    MemoryConfig,
    MemoryGateConfig,
    MetricsConfig,
    ProviderConfig,
    ServerConfig,
    StorageConfig,
)


def test_server_config_defaults() -> None:
    """Test ServerConfig default values."""
    config = ServerConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.reload is False
    assert config.workers == 1
    assert config.log_level == "info"


def test_storage_config_defaults() -> None:
    """Test StorageConfig default values."""
    config = StorageConfig()
    assert config.backend == "memory"
    assert config.persist_directory is None
    assert config.collection_name == "memory_gate"


def test_provider_config_defaults() -> None:
    """Test ProviderConfig default values."""
    config = ProviderConfig()
    assert config.default_provider == "ollama"
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.ollama_default_model == "llama3"


def test_memory_config_defaults() -> None:
    """Test MemoryConfig default values."""
    config = MemoryConfig()
    assert config.max_context_length == 4000
    assert config.retrieval_limit == 10
    assert config.similarity_threshold == 0.7
    assert config.enable_consolidation is True
    assert config.consolidation_interval == 3600


def test_metrics_config_defaults() -> None:
    """Test MetricsConfig default values."""
    config = MetricsConfig()
    assert config.enable_metrics is True
    assert config.metrics_port == 9090


def test_memory_gate_config_defaults() -> None:
    """Test MemoryGateConfig default values."""
    config = MemoryGateConfig()
    assert isinstance(config.server, ServerConfig)
    assert isinstance(config.storage, StorageConfig)
    assert isinstance(config.provider, ProviderConfig)
    assert isinstance(config.memory, MemoryConfig)
    assert isinstance(config.metrics, MetricsConfig)


def test_config_from_yaml() -> None:
    """Test loading configuration from YAML file."""
    yaml_content = """
server:
  host: "127.0.0.1"
  port: 9000
  log_level: "debug"

storage:
  backend: "chroma"
  collection_name: "test_collection"

provider:
  default_provider: "openai"
  openai_default_model: "gpt-4"

memory:
  max_context_length: 5000
  retrieval_limit: 20

metrics:
  enable_metrics: false
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = MemoryGateConfig.from_yaml(temp_path)

        # Check server config
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000
        assert config.server.log_level == "debug"

        # Check storage config
        assert config.storage.backend == "chroma"
        assert config.storage.collection_name == "test_collection"

        # Check provider config
        assert config.provider.default_provider == "openai"
        assert config.provider.openai_default_model == "gpt-4"

        # Check memory config
        assert config.memory.max_context_length == 5000
        assert config.memory.retrieval_limit == 20

        # Check metrics config
        assert config.metrics.enable_metrics is False

    finally:
        Path(temp_path).unlink()


def test_config_from_toml() -> None:
    """Test loading configuration from TOML file."""
    toml_content = """
[server]
host = "127.0.0.1"
port = 9000
log_level = "debug"

[storage]
backend = "chroma"
collection_name = "test_collection"

[provider]
default_provider = "openai"
openai_default_model = "gpt-4"

[memory]
max_context_length = 5000
retrieval_limit = 20

[metrics]
enable_metrics = false
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as f:
        f.write(toml_content)
        temp_path = f.name

    try:
        config = MemoryGateConfig.from_toml(temp_path)

        # Check server config
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000
        assert config.server.log_level == "debug"

        # Check storage config
        assert config.storage.backend == "chroma"
        assert config.storage.collection_name == "test_collection"

        # Check provider config
        assert config.provider.default_provider == "openai"
        assert config.provider.openai_default_model == "gpt-4"

        # Check memory config
        assert config.memory.max_context_length == 5000
        assert config.memory.retrieval_limit == 20

        # Check metrics config
        assert config.metrics.enable_metrics is False

    finally:
        Path(temp_path).unlink()


def test_config_file_not_found() -> None:
    """Test error handling for non-existent config file."""
    with pytest.raises(FileNotFoundError):
        MemoryGateConfig.from_yaml("/nonexistent/config.yaml")

    with pytest.raises(FileNotFoundError):
        MemoryGateConfig.from_toml("/nonexistent/config.toml")


def test_config_load_with_path() -> None:
    """Test loading configuration with explicit path."""
    yaml_content = """
server:
  port: 7000
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = MemoryGateConfig.load(temp_path)
        assert config.server.port == 7000

    finally:
        Path(temp_path).unlink()


def test_config_load_unsupported_format() -> None:
    """Test error handling for unsupported config format."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported config format"):
            MemoryGateConfig.load(temp_path)
    finally:
        Path(temp_path).unlink()


def test_config_load_default() -> None:
    """Test loading default configuration when no file exists."""
    # Load from non-existent directory
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            config = MemoryGateConfig.load()
            # Should return default configuration
            assert config.server.port == 8000
            assert config.storage.backend == "memory"
        finally:
            os.chdir(original_cwd)


def test_server_config_validation() -> None:
    """Test ServerConfig validation."""
    # Valid port
    config = ServerConfig(port=8080)
    assert config.port == 8080

    # Invalid port (too low)
    with pytest.raises(ValueError):
        ServerConfig(port=0)

    # Invalid port (too high)
    with pytest.raises(ValueError):
        ServerConfig(port=70000)


def test_memory_config_validation() -> None:
    """Test MemoryConfig validation."""
    # Valid similarity threshold
    config = MemoryConfig(similarity_threshold=0.5)
    assert config.similarity_threshold == 0.5

    # Invalid similarity threshold (too low)
    with pytest.raises(ValueError):
        MemoryConfig(similarity_threshold=-0.1)

    # Invalid similarity threshold (too high)
    with pytest.raises(ValueError):
        MemoryConfig(similarity_threshold=1.5)


def test_config_partial_override() -> None:
    """Test partial configuration override."""
    yaml_content = """
server:
  port: 9000

storage:
  backend: "chroma"
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = MemoryGateConfig.from_yaml(temp_path)

        # Overridden values
        assert config.server.port == 9000
        assert config.storage.backend == "chroma"

        # Default values should still be present
        assert config.server.host == "0.0.0.0"
        assert config.provider.default_provider == "ollama"
        assert config.memory.retrieval_limit == 10

    finally:
        Path(temp_path).unlink()
