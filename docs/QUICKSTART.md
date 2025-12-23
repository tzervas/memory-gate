# MemoryGate Quickstart Guide

Welcome to MemoryGate! This guide will help you get started with MemoryGate in minutes.

## What is MemoryGate?

MemoryGate is a dynamic memory learning layer for AI systems that enables:
- **Persistent Learning**: AI agents retain knowledge across sessions
- **Context-Aware Responses**: Dynamically adapts based on accumulated experience
- **Multi-Provider Support**: Works with Ollama, OpenAI, and other LLM providers
- **Flexible Deployment**: From pip-installable packages to Kubernetes clusters

## Installation

### Prerequisites
- Python 3.12 or higher
- UV package manager (recommended) or pip

### Quick Install with UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/tzervas/memory-gate.git
cd memory-gate

# Create environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install with MCP support (includes FastAPI for REST API)
uv sync --extra mcp
```

### Alternative: Install with pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install MemoryGate
pip install -e .

# Install with API support
pip install -e ".[mcp]"
```

## Quick Start: 5-Minute Tutorial

### 1. Basic Usage (Python)

Create a file `quick_test.py`:

```python
import asyncio
from datetime import datetime
from memory_gate.memory_protocols import LearningContext
from memory_gate.storage.in_memory import InMemoryKnowledgeStore

async def main():
    # Create knowledge store with in-memory storage
    store = InMemoryKnowledgeStore()
    
    # Store a memory
    context = LearningContext(
        content="Python is great for AI and data science",
        domain="programming",
        importance=0.9,
        timestamp=datetime.now(),
    )
    await store.store_experience("ctx_1", context)
    
    # Query memories
    results = await store.retrieve_context("Python programming", limit=5)
    for memory in results:
        print(f"Found: {memory.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python quick_test.py
```

### 2. REST API Usage

Start the API server:

```bash
# Start the server
memory-gate-serve --port 8000 --verbose

# Or with Python module
python -m memory_gate.api.cli --port 8000
```

In another terminal, test the API:

```python
# test_api.py
import httpx
import asyncio

async def main():
    async with httpx.AsyncClient() as client:
        # Store a memory
        response = await client.post(
            "http://localhost:8000/api/v1/memory/store",
            json={
                "content": "FastAPI is a modern web framework",
                "domain": "programming",
                "importance": 0.9
            }
        )
        print(f"Stored: {response.json()}")
        
        # Query memories
        response = await client.post(
            "http://localhost:8000/api/v1/memory/query",
            json={"query": "web framework", "limit": 5}
        )
        print(f"Found: {response.json()}")

asyncio.run(main())
```

### 3. With Ollama Integration

First, ensure Ollama is running:

```bash
# Install Ollama from https://ollama.ai
# Start Ollama server
ollama serve

# Pull a model
ollama pull llama3
```

Create `ollama_test.py`:

```python
import asyncio
from memory_gate.providers.ollama import OllamaProvider
from memory_gate.providers.base import GenerationConfig

async def main():
    # Create Ollama provider
    provider = OllamaProvider(base_url="http://localhost:11434")
    
    # Generate a response
    config = GenerationConfig(model="llama3", temperature=0.7)
    response = await provider.generate(
        prompt="What is Python in one sentence?",
        config=config
    )
    print(f"Response: {response.content}")

asyncio.run(main())
```

## Configuration

MemoryGate supports configuration through YAML, TOML files, or environment variables.

### Using Config Files

**Option 1: YAML** (`config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"

storage:
  backend: "memory"  # or "chroma" for persistent storage
  collection_name: "my_memories"

provider:
  default_provider: "ollama"
  ollama_base_url: "http://localhost:11434"
  ollama_default_model: "llama3"

memory:
  max_context_length: 4000
  retrieval_limit: 10
  similarity_threshold: 0.7
```

**Option 2: TOML** (`config.toml`)

```toml
[server]
host = "0.0.0.0"
port = 8000
log_level = "info"

[storage]
backend = "memory"
collection_name = "my_memories"

[provider]
default_provider = "ollama"
ollama_base_url = "http://localhost:11434"
ollama_default_model = "llama3"

[memory]
max_context_length = 4000
retrieval_limit = 10
similarity_threshold = 0.7
```

Copy example configs:

```bash
cp config.example.yaml config.yaml
# or
cp config.example.toml config.toml
```

### Using Environment Variables

```bash
# Server configuration
export MEMORYGATE_SERVER__HOST="0.0.0.0"
export MEMORYGATE_SERVER__PORT=8000

# Provider configuration
export MEMORYGATE_PROVIDER__DEFAULT_PROVIDER="ollama"
export MEMORYGATE_PROVIDER__OLLAMA_BASE_URL="http://localhost:11434"

# Start server with environment config
memory-gate-serve
```

### Using Config File with CLI

```bash
# Use specific config file
memory-gate-serve --config config.yaml

# Override config with CLI arguments
memory-gate-serve --config config.yaml --port 9000 --verbose
```

## Examples

The `examples/` directory contains several example scripts:

1. **basic_usage.py** - Core memory operations
2. **api_usage.py** - REST API client examples
3. **provider_config.py** - Model provider configuration
4. **memory_operations.py** - Advanced memory operations

Run any example:

```bash
# Basic usage
python examples/basic_usage.py

# API usage (requires server running)
python examples/api_usage.py

# Provider configuration
python examples/provider_config.py

# Memory operations
python examples/memory_operations.py
```

## API Documentation

Once the server is running, access interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Common Operations

### Health Check

```bash
curl http://localhost:8000/health
```

### Store a Memory

```bash
curl -X POST http://localhost:8000/api/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Docker containers provide application isolation",
    "domain": "devops",
    "importance": 0.9
  }'
```

### Query Memories

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "containerization",
    "limit": 5
  }'
```

### Augment Prompt with Memories

```bash
curl -X POST http://localhost:8000/api/v1/memory/augment \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain container orchestration",
    "limit": 3
  }'
```

## Storage Backends

MemoryGate supports multiple storage backends:

### In-Memory (Default)
- Fast, no setup required
- Data lost when process stops
- Good for development and testing

```yaml
storage:
  backend: "memory"
```

### ChromaDB (Persistent)
- Vector database for semantic search
- Persists data to disk
- Good for production use

```bash
# Install ChromaDB support
uv sync --extra storage
# or
pip install -e ".[storage]"
```

```yaml
storage:
  backend: "chroma"
  persist_directory: "./data/chroma"
  collection_name: "memories"
```

### Qdrant (Scalable)
- High-performance vector search
- Cloud and self-hosted options
- Good for enterprise deployments

```yaml
storage:
  backend: "qdrant"
  qdrant_url: "http://localhost:6333"
  qdrant_api_key: "your-api-key"
```

## Testing Your Setup

Run the test suite to verify your installation:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest --cov=src/memory_gate
```

## Next Steps

1. **Explore Examples**: Check out the `examples/` directory for more use cases
2. **Read API Docs**: Visit the `/docs` endpoint for detailed API reference
3. **Configure Storage**: Set up ChromaDB for persistent memory storage
4. **Integrate with Your App**: Use the REST API or Python SDK in your application
5. **Production Deployment**: See deployment guides for Docker/Kubernetes

## Troubleshooting

### Server won't start

**Issue**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Install MCP dependencies:
```bash
uv sync --extra mcp
# or
pip install -e ".[mcp]"
```

### Can't connect to Ollama

**Issue**: `httpx.ConnectError`

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

### Configuration not loading

**Issue**: Config file not found

**Solution**: 
- Place `config.yaml` or `config.toml` in the current directory
- Or specify path: `memory-gate-serve --config /path/to/config.yaml`

### Memory not persisting

**Issue**: Memories lost after restart

**Solution**: Use persistent storage:
```yaml
storage:
  backend: "chroma"
  persist_directory: "./data"
```

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: https://github.com/tzervas/memory-gate/issues
- **Email**: tz-dev@vectorweight.com
- **Twitter**: @vec_wt_tech

## What's Next?

Check out the [Project Tracker](docs/project_tracker.md) to see what features are coming next:
- Persona Manager (multi-context support)
- Memory Culling (intelligent cleanup)
- Open WebUI Integration
- And more!

---

**Ready to build something amazing? Let's go! 🚀**
