## Local Setup

To set up the project locally for development, follow these steps:

### Prerequisites
- Python 3.12 or higher
- UV package manager (recommended) or pip

### Installation with UV (Recommended)

1. **Install UV** (if not already installed):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Create and activate virtual environment with dependencies**:
   ```bash
   # Create project environment and install dependencies
   uv sync
   
   # Activate the environment
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```

3. **Install development dependencies**:
   ```bash
   uv sync --extra dev
   ```

4. **Install optional dependencies** (as needed):
   ```bash
   # For GPU support
   uv sync --extra gpu
   
   # For advanced storage backends
   uv sync --extra storage
   
   # Install all extras
   uv sync --all-extras
   ```

### Alternative Installation with pip/venv

1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   # Install core dependencies
   pip install -e .
   
   # Install with development dependencies
   pip install -e ".[dev]"
   
   # Install with all optional dependencies
   pip install -e ".[dev,gpu,storage]"
   ```

### Verify Installation

```bash
# Run tests to verify everything is working

<!-- FLEET-BADGES:BEGIN -->
[![CI](https://github.com/tzervas/memory-gate/actions/workflows/fleet-ci.yml/badge.svg?branch=main)](https://github.com/tzervas/memory-gate/actions/workflows/fleet-ci.yml?query=branch%3Amain)
[![Security](https://github.com/tzervas/memory-gate/actions/workflows/fleet-security.yml/badge.svg?branch=main)](https://github.com/tzervas/memory-gate/actions/workflows/fleet-security.yml?query=branch%3Amain)
<!-- FLEET-BADGES:END -->

pytest

# Check code formatting
black --check src tests

# Run type checking
mypy src
```

### Cross-Platform Compatibility

This setup supports:
- **Windows** (including WSL)
- **macOS** 
- **Linux**
- **Dev Containers** for isolated development

## Versioning

This project follows [Conventional Commits](https://www.conventionalcommits.org/) and uses [Commitizen](https://commitizen-tools.github.io/commitizen/) for release versioning. Version is tracked in `pyproject.toml` (`project.version` and `[tool.commitizen]`). Before release, dispatch the **Commitizen** workflow (Actions → Commitizen → Run workflow) to verify commits on the current branch.

## Contact

Tyler Zervas
- GitHub: [tzervas](https://github.com/tzervas)
- X: [@vec_wt_tech](https://twitter.com/vec_wt_tech)
- Email: [tz-dev@vectorweight.com](mailto:tz-dev@vectorweight.com)

# MemoryGate

A dynamic memory learning layer for AI agents, designed for DevOps automation and homelab AI R&D.

## Supported embedding models

`VectorStoreConfig.embedding_model_name` accepts a **stable catalog ID** (shared with [memory-gate-rs](https://github.com/tzervas/memory-gate-rs)) or a **catalog-listed** SentenceTransformers / Hugging Face load name. Off-catalog model strings are rejected at init. Default remains `all-MiniLM-L6-v2` for existing deployments; cross-port parity experiments should pin `all-minilm-l6-v2`.

### One collection per embedding model

Each Chroma collection is bound to exactly one catalog model and dimension. On `VectorMemoryStore` init, collection metadata is stamped (or validated):

| Metadata key | Value |
|--------------|-------|
| `memory_gate_embedding_model` | Stable catalog ID (e.g. `all-minilm-l6-v2`) |
| `memory_gate_embedding_dim` | Embedding dimension as a string (e.g. `384`) |

If an existing collection was created with a different model or dimension, initialization fails with `VectorStoreInitError`. Legacy collections that already contain vectors but lack these keys also fail closed—use a new `collection_name` or migrate. Empty collections without stamps are updated automatically. User experience metadata cannot override binding: keys prefixed with `memory_gate_` or `embedding_` are stripped on store.

| Stable ID | SentenceTransformers / HF | Dim |
|-----------|---------------------------|-----|
| `all-minilm-l6-v2` | `all-MiniLM-L6-v2` | 384 |
| `bge-small-en-v1.5` | `BAAI/bge-small-en-v1.5` | 384 |
| `bge-base-en-v1.5` | `BAAI/bge-base-en-v1.5` | 768 |

Resolve programmatically via `memory_gate.embedding_catalog.resolve_model(id)`.

## Key Features

- **Persistent Learning**: Enables AI agents to retain and build upon operational knowledge across sessions.
- **Context-Aware Adaptation**: Dynamically adjusts responses based on accumulated experience patterns.
- **Cluster Focus**: Designed for seamless deployment in containerized environments with proper monitoring and scaling.
- **DevOps Integration**: Provides native integration with existing CI/CD pipelines and infrastructure tools.
