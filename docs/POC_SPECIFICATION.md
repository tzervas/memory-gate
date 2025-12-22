# MemoryGate POC Specification

## Document Information
- **Version**: 1.1.0
- **Created**: 2025-12-22
- **Last Updated**: 2025-12-22
- **Status**: Draft
- **Author**: Development Team

---

## Executive Summary

MemoryGate is a **loosely-coupled dynamic memory and learning layer** designed to provide persistent, context-aware memory capabilities for AI models and agents. The POC focuses on creating a standalone component that can integrate with Ollama and web UIs (Open WebUI, ComfyUI) to enable active learning without modifying base model weights.

### Core Innovation

Unlike traditional fine-tuning approaches, MemoryGate operates as an **offset layer** to underlying model weights and training data, making it:
- Easy to attach/detach from any compatible model
- Simple to version, swap, and rollback
- Secure (can be removed without affecting base model)
- Lightweight through differential versioning

### Scaling Philosophy

MemoryGate is architected to scale from **research prototype to enterprise deployment**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MemoryGate Scaling Trajectory                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIER 1: Module/Package          TIER 2: Service              TIER 3: Enterprise
│  ════════════════════           ════════════════             ══════════════════
│                                                                             │
│  ┌─────────────────┐           ┌─────────────────┐         ┌─────────────────┐
│  │  pip install    │           │  Docker/Compose │         │  Helm Chart     │
│  │  memory-gate    │    ──►    │  Deployment     │   ──►   │  K8s Cluster    │
│  └─────────────────┘           └─────────────────┘         └─────────────────┘
│                                                                             │
│  • Single-user R&D             • Team/Department           • Enterprise-wide │
│  • Agentic systems             • Multi-user                • High availability│
│  • Local Ollama                • Shared Ollama             • GPU clusters     │
│  • In-memory/SQLite            • PostgreSQL/Redis          • Distributed DB   │
│  • No auth required            • Basic auth/RBAC           • SSO/OIDC/mTLS    │
│  • Manual backup               • Scheduled backup          • HA replication   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

This document focuses on **Tier 1 (POC)** with architectural decisions that enable smooth progression through all tiers.

---

## Project Vision

### Origin & Philosophy

> **Important**: MemoryGate is an **original, independent implementation** based on concepts developed over several years, predating recent industry publications on active learning (including Google's Titans paper). While similar ideas have emerged in academic research, this project intentionally follows its own architectural vision and design philosophy rather than aligning with or deriving from external research.

The core concepts behind MemoryGate emerged from practical observations about the limitations of current AI systems:
- Models forget context between sessions
- Fine-tuning is expensive and risks catastrophic forgetting  
- There's no clean separation between "base knowledge" and "learned experience"
- Users lack control over what AI systems remember or forget

**This project implements solutions to these problems according to its own design principles**, which often differ significantly from mainstream industry approaches. Contributors should implement features based on this project's documentation and specifications, not external papers or frameworks.

### Design Principles (Project-Specific)

1. **Memory as Offset, Not Weight Modification**: Memories exist as retrievable context that augments prompts, not as changes to model parameters. This is fundamentally different from approaches that modify embeddings or weights.

2. **Explicit Over Implicit**: Memory operations should be observable, controllable, and reversible. No "black box" learning.

3. **User Sovereignty**: Users control their memory layer completely - what's stored, what's forgotten, what personas exist.

4. **Clean Decoupling**: The memory layer must be completely removable without any trace on the base model.

5. **Differential Efficiency**: Store changes, not snapshots. This applies to memory versions, persona deltas, and consolidation.

### Goals
1. **Persistent Learning**: Enable AI agents/models to retain knowledge across sessions
2. **Active Memory Management**: Intelligent culling of contextually irrelevant memories
3. **Persona Management**: Support for multiple memory profiles/personas
4. **Lightweight Storage**: Differential versioning for efficient memory storage
5. **Loose Coupling**: Easy attachment/detachment from base models
6. **Integration Ready**: Native support for Ollama, Open WebUI, ComfyUI ecosystems

---

## POC Scope and Deliverables

### POC Phase (Current Focus)

The POC demonstrates the core memory layer functionality with Ollama integration.

#### POC Deliverables

| ID | Deliverable | Description | Priority |
|----|-------------|-------------|----------|
| P1 | Memory Protocol Interface | Standardized interface for memory operations | Critical |
| P2 | Vector Store Backend | ChromaDB-based persistent memory storage | Critical |
| P3 | Memory Consolidation Engine | Background process for memory management | High |
| P4 | Ollama Integration Layer | API adapter for Ollama model interaction | Critical |
| P5 | Persona Manager | Basic persona/profile management | Medium |
| P6 | Memory Differential System | Lightweight delta-based memory versioning | Medium |
| P7 | Web UI Integration | REST API for Open WebUI/ComfyUI | High |

#### POC Success Criteria

- [ ] Memory retrieval latency < 100ms for contextual queries
- [ ] Successfully store and retrieve 10,000+ memories
- [ ] Demonstrate memory attachment/detachment from Ollama model
- [ ] Show memory culling reduces storage by 30%+ over time
- [ ] Persona switching within 50ms
- [ ] Memory differential reduces version storage by 80%+

---

## Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Layer                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Open WebUI  │  │   ComfyUI    │  │  Custom Apps │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                           │
│         └─────────────────┼─────────────────┘                           │
│                           │                                             │
│                    ┌──────▼───────┐                                     │
│                    │   REST API   │                                     │
│                    └──────┬───────┘                                     │
└──────────────────────────│──────────────────────────────────────────────┘
                           │
┌──────────────────────────│──────────────────────────────────────────────┐
│                    MemoryGate Core                                       │
│                           │                                             │
│  ┌────────────────────────▼────────────────────────────────────┐        │
│  │                   Memory Gateway                             │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │        │
│  │  │   Adapter   │  │  Retrieval  │  │ Context Injector    │  │        │
│  │  │   Layer     │  │   Engine    │  │ (Prompt Augmentation)│  │        │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │        │
│  └──────────────────────────┬──────────────────────────────────┘        │
│                             │                                           │
│  ┌──────────────────────────▼──────────────────────────────────┐        │
│  │                  Persona Manager                             │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │        │
│  │  │  Persona    │  │  Memory     │  │  Context            │  │        │
│  │  │  Profiles   │  │  Isolation  │  │  Boundaries         │  │        │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │        │
│  └──────────────────────────┬──────────────────────────────────┘        │
│                             │                                           │
│  ┌──────────────────────────▼──────────────────────────────────┐        │
│  │              Memory Consolidation Engine                     │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │        │
│  │  │ Importance  │  │  Memory     │  │  Differential       │  │        │
│  │  │  Scoring    │  │  Culling    │  │  Versioning         │  │        │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │        │
│  └──────────────────────────┬──────────────────────────────────┘        │
│                             │                                           │
│  ┌──────────────────────────▼──────────────────────────────────┐        │
│  │                   Storage Layer                              │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │        │
│  │  │ Vector Store│  │  Metadata   │  │  Version Store      │  │        │
│  │  │  (ChromaDB) │  │  Index      │  │  (Differentials)    │  │        │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                           │
                           │ Loosely Coupled Interface
                           │
┌──────────────────────────▼──────────────────────────────────────────────┐
│                      Model Layer                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Ollama Runtime                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │
│  │  │ Base Model  │  │  Context    │  │  Response Generation    │   │   │
│  │  │  Weights    │  │  Window     │  │                         │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

#### 1. Loose Coupling via Prompt Augmentation
MemoryGate does NOT modify model weights. Instead, it:
- Intercepts prompts before they reach the model
- Retrieves relevant memories via vector similarity
- Augments the prompt with contextual memory
- Passes the enhanced prompt to the model
- Captures responses for future memory storage

```python
# Conceptual flow
user_prompt = "How do I fix the nginx configuration?"

# MemoryGate retrieves relevant memories
relevant_memories = await memory_gateway.retrieve_context(user_prompt)

# Augment prompt with memory context
augmented_prompt = f"""
Based on your previous experiences:
{format_memories(relevant_memories)}

Current question: {user_prompt}
"""

# Send to Ollama (base model unchanged)
response = await ollama.generate(model="llama3", prompt=augmented_prompt)

# Store interaction for future learning
await memory_gateway.learn_from_interaction(user_prompt, response)
```

#### 2. Differential Memory Versioning
Instead of storing complete memory snapshots, store deltas:

```python
@dataclass
class MemoryDelta:
    """Represents a change to a memory rather than the full memory."""
    base_version: str           # Reference to base memory version
    delta_type: DeltaType       # ADD, MODIFY, DELETE, MERGE
    changes: dict[str, Any]     # Only the changed fields
    timestamp: datetime
    reason: str                 # Why this change was made
    
# Storage is dramatically reduced:
# Full memory: 1KB average
# Delta: 50-100 bytes average
# 80-95% storage reduction for version history
```

#### 3. Memory Culling Strategy

```python
class MemoryCullingStrategy:
    """Determines which memories to archive or delete."""
    
    def calculate_relevance_score(self, memory: LearningContext) -> float:
        """
        Score based on:
        - Recency: How recently was this memory accessed?
        - Frequency: How often is this memory retrieved?
        - Importance: User/system assigned importance
        - Contextual Fit: How well does it fit current persona/domain?
        """
        recency_score = self._recency_decay(memory.last_accessed)
        frequency_score = self._access_frequency(memory.access_count)
        importance_score = memory.importance
        context_score = self._context_relevance(memory, current_persona)
        
        return (
            0.25 * recency_score +
            0.25 * frequency_score +
            0.30 * importance_score +
            0.20 * context_score
        )
    
    def cull_memories(self, threshold: float = 0.3) -> list[str]:
        """Archive memories below relevance threshold."""
        # Memories aren't deleted - they're archived as differentials
        # Can be restored if needed
```

#### 4. Persona Management

```python
@dataclass
class Persona:
    """A memory profile/persona for the AI."""
    id: str
    name: str
    description: str
    memory_collection: str      # Isolated ChromaDB collection
    domain_filters: list[str]   # Which domains this persona cares about
    personality_context: str    # Base personality injection
    created_at: datetime
    is_active: bool
    
class PersonaManager:
    """Manages multiple personas and their memory isolation."""
    
    async def switch_persona(self, persona_id: str) -> None:
        """Switch active persona - memories are isolated per persona."""
        # Hot-swap the memory context
        # No model reload needed - just context switching
        
    async def merge_personas(
        self, 
        source_id: str, 
        target_id: str,
        merge_strategy: MergeStrategy
    ) -> None:
        """Selectively merge memories between personas."""
```

---

## Integration Specifications

### Ollama Integration

```python
class OllamaMemoryBridge:
    """Bridge between MemoryGate and Ollama."""
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        memory_gateway: MemoryGateway = None,
    ):
        self.ollama_url = ollama_base_url
        self.memory_gateway = memory_gateway
        
    async def generate_with_memory(
        self,
        model: str,
        prompt: str,
        persona_id: str | None = None,
        stream: bool = False,
    ) -> AsyncGenerator[str, None] | str:
        """Generate response with memory augmentation."""
        
        # 1. Retrieve relevant memories
        memories = await self.memory_gateway.retrieve_context(
            query=prompt,
            limit=5,
            persona_filter=persona_id,
        )
        
        # 2. Build augmented prompt
        augmented = self._build_memory_prompt(prompt, memories)
        
        # 3. Call Ollama
        response = await self._ollama_generate(model, augmented, stream)
        
        # 4. Store interaction (async, non-blocking)
        asyncio.create_task(
            self._store_interaction(prompt, response, persona_id)
        )
        
        return response
```

### Open WebUI Integration

MemoryGate exposes a REST API compatible with Open WebUI's plugin/extension system:

```yaml
# OpenAPI Spec (partial)
openapi: 3.0.0
info:
  title: MemoryGate API
  version: 1.0.0
  
paths:
  /api/v1/memory/query:
    post:
      summary: Query memories
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                query: {type: string}
                limit: {type: integer, default: 10}
                persona_id: {type: string}
                
  /api/v1/memory/store:
    post:
      summary: Store a memory
      
  /api/v1/persona/switch:
    post:
      summary: Switch active persona
      
  /api/v1/memory/augment:
    post:
      summary: Augment a prompt with relevant memories
      description: Primary endpoint for UI integration
```

---

## Current Implementation Status

### Implemented ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Memory Protocols | ✅ Complete | `memory_protocols.py` - Core interfaces |
| Learning Context | ✅ Complete | Dataclass for memory storage |
| Memory Gateway | ✅ Complete | Central memory management |
| Vector Store (ChromaDB) | ✅ Complete | With Pydantic validation |
| In-Memory Store | ✅ Complete | For testing/development |
| Consolidation Worker | ✅ Partial | Basic implementation, needs enhancement |
| Agent Interface | ✅ Complete | Base classes for memory-enabled agents |
| Metrics (Prometheus) | ✅ Complete | Full observability |
| Docker/Compose | ✅ Complete | Development environment |
| Helm Charts | ✅ Partial | Basic K8s deployment |

### Needs Implementation 🚧

| Component | Priority | Estimated Effort |
|-----------|----------|------------------|
| Ollama Integration Bridge | Critical | 16 hours |
| REST API Layer (FastAPI) | Critical | 12 hours |
| Persona Manager | High | 16 hours |
| Memory Differential System | High | 20 hours |
| Memory Culling Strategy | High | 12 hours |
| Open WebUI Plugin | Medium | 8 hours |
| ComfyUI Node | Medium | 8 hours |
| Security/Detachment Layer | Medium | 12 hours |

### Needs Enhancement 🔄

| Component | Enhancement Needed |
|-----------|-------------------|
| Consolidation Worker | Implement merging, decay, better scoring |
| Vector Store | Add versioning support |
| Agent Interface | Add persona awareness |

---

## Task Decomposition

### Phase 1: POC Core (Weeks 1-2)

#### Sprint 1: Ollama Integration
- [ ] **TASK-001**: Create OllamaMemoryBridge class
  - [ ] Implement Ollama API client
  - [ ] Add memory retrieval integration
  - [ ] Build prompt augmentation logic
  - [ ] Add async response handling
  - [ ] Unit tests

- [ ] **TASK-002**: Create REST API Layer
  - [ ] Set up FastAPI application
  - [ ] Implement memory CRUD endpoints
  - [ ] Add prompt augmentation endpoint
  - [ ] Add health/readiness endpoints
  - [ ] OpenAPI documentation
  - [ ] Integration tests

#### Sprint 2: Persona Management
- [ ] **TASK-003**: Implement Persona data model
  - [ ] Define Persona dataclass/model
  - [ ] Create persona storage schema
  - [ ] Add persona validation

- [ ] **TASK-004**: Implement PersonaManager
  - [ ] Create persona CRUD operations
  - [ ] Implement persona switching
  - [ ] Add memory isolation per persona
  - [ ] Add context boundary enforcement

### Phase 2: Memory Enhancement (Weeks 3-4)

#### Sprint 3: Differential Versioning
- [ ] **TASK-005**: Design differential storage schema
  - [ ] Define MemoryDelta model
  - [ ] Create delta calculation algorithm
  - [ ] Implement delta application logic

- [ ] **TASK-006**: Implement version store
  - [ ] Create version tracking system
  - [ ] Implement delta storage
  - [ ] Add version reconstruction
  - [ ] Add rollback capability

#### Sprint 4: Memory Culling
- [ ] **TASK-007**: Implement relevance scoring
  - [ ] Recency decay function
  - [ ] Frequency tracking
  - [ ] Context relevance calculation
  - [ ] Combined scoring algorithm

- [ ] **TASK-008**: Implement culling engine
  - [ ] Background culling worker
  - [ ] Archive vs delete logic
  - [ ] Configurable thresholds
  - [ ] Metrics and monitoring

### Phase 3: UI Integration (Weeks 5-6)

#### Sprint 5: Open WebUI Integration
- [ ] **TASK-009**: Create Open WebUI plugin
  - [ ] Plugin manifest
  - [ ] Memory sidebar component
  - [ ] Persona selector
  - [ ] Memory search interface

#### Sprint 6: Security & Polish
- [ ] **TASK-010**: Implement detachment layer
  - [ ] Clean detachment protocol
  - [ ] Memory export/import
  - [ ] Base model fallback mode
  - [ ] Security audit

---

## MVP Phase Definition

Following successful POC validation, the MVP expands to:

### MVP Scope
1. **Production-ready Ollama integration**
2. **Full persona lifecycle management**
3. **Advanced memory consolidation**
4. **Multi-user support**
5. **Comprehensive API documentation**
6. **Performance optimization (sub-50ms retrieval)**
7. **Kubernetes production deployment**

### MVP Timeline: 8-12 weeks post-POC

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Ollama API changes | Medium | High | Abstract API layer, version pinning |
| Memory retrieval latency | Medium | High | Caching, index optimization |
| Storage growth | High | Medium | Aggressive culling, compression |
| Persona isolation leak | Low | High | Strict collection separation |
| Model context overflow | Medium | Medium | Smart memory selection algorithm |

---

## Appendix

### A. Existing File Structure

```
memory-gate/
├── src/memory_gate/
│   ├── __init__.py
│   ├── main.py                 # Application entrypoint
│   ├── memory_gateway.py       # Core memory management
│   ├── memory_protocols.py     # Protocol definitions
│   ├── consolidation.py        # Memory consolidation worker
│   ├── agent_interface.py      # Agent base classes
│   ├── metrics.py              # Prometheus metrics
│   ├── agents/
│   │   ├── __init__.py
│   │   └── infrastructure_agent.py
│   └── storage/
│       ├── __init__.py
│       ├── in_memory.py        # In-memory store
│       └── vector_store.py     # ChromaDB vector store
├── tests/
├── docs/
├── helm/memory-gate/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

### B. Configuration Reference

Environment variables for deployment:

```bash
# Metrics
METRICS_PORT=8008
METRICS_HOST=0.0.0.0

# ChromaDB
CHROMA_PERSIST_DIRECTORY=/app/data/chromadb
CHROMA_COLLECTION_NAME=memory_gate_collection

# Consolidation
CONSOLIDATION_ENABLED=true
CONSOLIDATION_INTERVAL_SECONDS=3600

# Ollama (new for POC)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama3

# API (new for POC)
API_HOST=0.0.0.0
API_PORT=8000
```

### C. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Memory retrieval latency | < 100ms p95 | Prometheus histogram |
| Storage efficiency | 80%+ reduction via differentials | Before/after comparison |
| Memory culling effectiveness | 30%+ storage reduction | Collection size monitoring |
| Persona switch time | < 50ms | Prometheus histogram |
| API response time | < 200ms p95 | Prometheus histogram |
| Test coverage | > 85% | pytest-cov |

---

*Document End*
