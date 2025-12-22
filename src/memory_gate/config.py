"""Configuration management for MemoryGate.

This module provides a unified configuration system supporting:
- Environment variables
- YAML configuration files
- TOML configuration files
- Programmatic defaults

Configuration priority (highest to lowest):
1. Environment variables
2. Configuration file (YAML/TOML)
3. Default values
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    """API server configuration."""

    host: str = Field(
        default="0.0.0.0",
        description="Host address to bind the server to",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port number to bind the server to",
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload on code changes (development only)",
    )
    workers: int = Field(
        default=1,
        ge=1,
        description="Number of worker processes",
    )
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        description="Logging level",
    )


class StorageConfig(BaseSettings):
    """Storage backend configuration."""

    backend: Literal["memory", "chroma", "qdrant"] = Field(
        default="memory",
        description="Storage backend type",
    )
    persist_directory: Path | None = Field(
        default=None,
        description="Directory for persistent storage (ChromaDB)",
    )
    collection_name: str = Field(
        default="memory_gate",
        description="Collection name for vector store",
    )
    # ChromaDB specific
    chroma_host: str | None = Field(
        default=None,
        description="ChromaDB host (for remote ChromaDB)",
    )
    chroma_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="ChromaDB port (for remote ChromaDB)",
    )
    # Qdrant specific
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant server URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key",
    )


class ProviderConfig(BaseSettings):
    """Default provider configuration."""

    default_provider: str = Field(
        default="ollama",
        description="Default model provider",
    )
    # Ollama configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    ollama_default_model: str = Field(
        default="llama3",
        description="Default Ollama model",
    )
    # OpenAI configuration
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    openai_default_model: str = Field(
        default="gpt-4",
        description="Default OpenAI model",
    )


class MemoryConfig(BaseSettings):
    """Memory management configuration."""

    max_context_length: int = Field(
        default=4000,
        ge=100,
        description="Maximum tokens for memory context",
    )
    retrieval_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default number of memories to retrieve",
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for memory retrieval",
    )
    enable_consolidation: bool = Field(
        default=True,
        description="Enable automatic memory consolidation",
    )
    consolidation_interval: int = Field(
        default=3600,
        ge=60,
        description="Memory consolidation interval in seconds",
    )


class MetricsConfig(BaseSettings):
    """Metrics and monitoring configuration."""

    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    metrics_port: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Port for Prometheus metrics endpoint",
    )


class MemoryGateConfig(BaseSettings):
    """Main MemoryGate configuration.

    This class aggregates all configuration sections and provides
    methods to load configuration from various sources.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORYGATE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Configuration sections
    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "MemoryGateConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            MemoryGateConfig instance with loaded configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the YAML file is invalid.
        """
        import yaml

        config_path = Path(path)
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        with config_path.open() as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            config_data = {}

        return cls(**config_data)

    @classmethod
    def from_toml(cls, path: Path | str) -> "MemoryGateConfig":
        """Load configuration from a TOML file.

        Args:
            path: Path to TOML configuration file.

        Returns:
            MemoryGateConfig instance with loaded configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the TOML file is invalid.
        """
        import tomllib  # Python 3.11+ built-in

        config_path = Path(path)
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        with config_path.open("rb") as f:
            config_data = tomllib.load(f)

        return cls(**config_data)

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> "MemoryGateConfig":
        """Load configuration with automatic format detection.

        Attempts to load configuration from:
        1. Specified config_path if provided
        2. ./config.yaml or ./config.yml
        3. ./config.toml
        4. Default configuration if no file found

        Args:
            config_path: Optional path to configuration file.

        Returns:
            MemoryGateConfig instance.
        """
        if config_path:
            config_path = Path(config_path)
            if config_path.suffix in {".yaml", ".yml"}:
                return cls.from_yaml(config_path)
            if config_path.suffix == ".toml":
                return cls.from_toml(config_path)
            msg = f"Unsupported config format: {config_path.suffix}"
            raise ValueError(msg)

        # Try default locations
        default_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path("config.toml"),
        ]

        for path in default_paths:
            if path.exists():
                if path.suffix in {".yaml", ".yml"}:
                    return cls.from_yaml(path)
                if path.suffix == ".toml":
                    return cls.from_toml(path)

        # Return default configuration if no file found
        return cls()


# Create a global config instance
_config: MemoryGateConfig | None = None


def get_config() -> MemoryGateConfig:
    """Get the global configuration instance.

    Returns:
        The global MemoryGateConfig instance.
    """
    global _config
    if _config is None:
        _config = MemoryGateConfig.load()
    return _config


def set_config(config: MemoryGateConfig) -> None:
    """Set the global configuration instance.

    Args:
        config: Configuration instance to set as global.
    """
    global _config
    _config = config
