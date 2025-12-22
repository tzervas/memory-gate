# MemoryGate REST API

A FastAPI-based REST API for the MemoryGate dynamic memory learning layer.

## Features

- **Memory Operations**: Store, retrieve, and query memories
- **Prompt Augmentation**: Enhance prompts with relevant memories
- **Provider-Agnostic Generation**: Generate responses using any supported provider (Ollama, OpenAI, custom)
- **Health Checks**: Health and readiness probes for container orchestration
- **Auto-generated Documentation**: Swagger UI and ReDoc interfaces

## Quick Start

### Starting the Server

```bash
# Using the CLI entry point
memory-gate-serve --host 0.0.0.0 --port 8000

# Or with development auto-reload
memory-gate-serve --port 8000 --reload --verbose

# Or directly with Python
python -m memory_gate.api.cli --port 8000
```

### API Documentation

Once the server is running, access the documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### Health Checks

#### GET /health
Check if the service is running.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-22T17:00:00Z",
  "version": "0.1.0"
}
```

#### GET /ready
Check if the service is ready to accept requests.

```bash
curl http://localhost:8000/ready
```

### Memory Operations

#### POST /api/v1/memory/query
Query memories by similarity.

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python programming",
    "limit": 10,
    "persona_id": null,
    "domain": null
  }'
```

Response:
```json
{
  "memories": [
    {
      "content": "Python is a high-level programming language",
      "domain": "programming",
      "timestamp": "2025-12-22T17:00:00Z",
      "importance": 0.8,
      "metadata": {}
    }
  ],
  "count": 1
}
```

#### POST /api/v1/memory/store
Store a new memory.

```bash
curl -X POST http://localhost:8000/api/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "content": "FastAPI is a modern Python web framework",
    "domain": "programming",
    "importance": 0.9,
    "metadata": {"source": "manual"}
  }'
```

Response:
```json
{
  "success": true,
  "message": "Memory stored successfully"
}
```

#### POST /api/v1/memory/augment
Augment a prompt with relevant memories.

```bash
curl -X POST http://localhost:8000/api/v1/memory/augment \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me about Python web frameworks",
    "limit": 5
  }'
```

Response:
```json
{
  "original_prompt": "Tell me about Python web frameworks",
  "augmented_prompt": "Based on relevant memories:\n- FastAPI is a modern Python web framework (importance: 0.90)\n\nCurrent request: Tell me about Python web frameworks",
  "memories_used": 1
}
```

### Generation

#### POST /api/v1/generate
Generate a response using any provider with optional memory augmentation.

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is FastAPI?",
    "model": "llama3",
    "provider": "ollama",
    "use_memory": true,
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

Response:
```json
{
  "content": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+...",
  "model": "llama3",
  "provider": "ollama",
  "memories_used": 2
}
```

## Configuration

### Environment Variables

```bash
# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Memory Store (for production)
CHROMA_PERSIST_DIRECTORY=/app/data/chromadb
CHROMA_COLLECTION_NAME=memory_gate_collection

# Provider Configuration
OLLAMA_BASE_URL=http://localhost:11434
```

## Python Client Example

See `examples/api_usage.py` for a complete example:

```python
import httpx
import asyncio

async def query_memories():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/memory/query",
            json={"query": "Python", "limit": 5}
        )
        return response.json()

# Run example
asyncio.run(query_memories())
```

## Development

### Running Tests

```bash
# Run API tests
pytest tests/test_api.py -v

# Run all tests
pytest -v

# Run with coverage
pytest --cov=src/memory_gate/api tests/test_api.py
```

### Linting and Formatting

```bash
# Format code
ruff format src/memory_gate/api

# Check linting
ruff check src/memory_gate/api

# Type checking
mypy src/memory_gate/api
```

## Integration

### Open WebUI Integration

The API is designed to integrate with Open WebUI and similar frontends:

1. Configure Open WebUI to use MemoryGate's `/api/v1/memory/augment` endpoint
2. Use `/api/v1/generate` for memory-augmented generation
3. Store conversation history via `/api/v1/memory/store`

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[mcp]"

EXPOSE 8000

CMD ["memory-gate-serve", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

See `helm/memory-gate/` for Kubernetes deployment manifests.

## Architecture

The API follows a modular design:

```
src/memory_gate/api/
├── __init__.py        # Module exports
├── app.py             # FastAPI application factory
├── cli.py             # CLI entry point
├── dependencies.py    # Dependency injection
├── models.py          # Pydantic request/response models
└── routes/
    ├── __init__.py
    ├── health.py      # Health check endpoints
    ├── memory.py      # Memory operation endpoints
    └── generate.py    # Generation endpoints
```

## Support

- **Documentation**: See main README and POC specification
- **Issues**: Report at https://github.com/tzervas/memory-gate/issues
- **Examples**: Check the `examples/` directory
