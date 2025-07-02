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

## Contact

Tyler Zervas
- GitHub: [tzervas](https://github.com/tzervas)
- X: [@vec_wt_tech](https://twitter.com/vec_wt_tech)
- Email: [tz-dev@vectorweight.com](mailto:tz-dev@vectorweight.com)

# MemoryGate

A dynamic memory learning layer for AI agents, designed for DevOps automation and homelab AI R&D.

## Key Features

- **Persistent Learning**: Enables AI agents to retain and build upon operational knowledge across sessions.
- **Context-Aware Adaptation**: Dynamically adjusts responses based on accumulated experience patterns.
- **Cluster Focus**: Designed for seamless deployment in containerized environments with proper monitoring and scaling.
- **DevOps Integration**: Provides native integration with existing CI/CD pipelines and infrastructure tools.
