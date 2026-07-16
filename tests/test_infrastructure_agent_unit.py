"""Unit tests for infrastructure agent suggestion branches."""

from datetime import datetime
from typing import Any

import pytest

from memory_gate.agent_interface import AgentDomain
from memory_gate.agents.infrastructure_agent import InfrastructureAgent
from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import LearningContext, MemoryAdapter


class PassThroughAdapter(MemoryAdapter[LearningContext]):
    async def adapt_knowledge(
        self, context: LearningContext, feedback: float | None = None
    ) -> LearningContext:
        return context


class TestInfrastructureAgentBranches:
    """Cover infrastructure agent suggestion code paths."""

    @pytest.fixture
    def infra_agent(self) -> InfrastructureAgent:
        from unittest.mock import AsyncMock

        store = AsyncMock()
        gateway = MemoryGateway(adapter=PassThroughAdapter(), store=store)
        return InfrastructureAgent(gateway)

    @pytest.mark.asyncio
    async def test_high_importance_with_solution_keyword(
        self, infra_agent: InfrastructureAgent
    ) -> None:
        context = {
            "task_input": "server alpha slow",
            "retrieved_memories": [
                {
                    "content": "Task: alpha issue\nSolution: restart service",
                    "importance": 0.9,
                    "domain": AgentDomain.INFRASTRUCTURE.value,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {},
                    "age_hours": 1.0,
                }
            ],
        }
        result, confidence = await infra_agent._execute_task(context)
        assert "Suggestions based on past experiences" in result
        assert confidence == 0.75

    @pytest.mark.asyncio
    async def test_keyword_match_without_solution(
        self, infra_agent: InfrastructureAgent
    ) -> None:
        context = {
            "task_input": "database latency spike",
            "retrieved_memories": [
                {
                    "content": "database connection pool exhausted last week",
                    "importance": 0.5,
                    "domain": AgentDomain.INFRASTRUCTURE.value,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {},
                    "age_hours": 2.0,
                }
            ],
        }
        result, confidence = await infra_agent._execute_task(context)
        assert "Keyword match past event" in result
        assert confidence == 0.75

    @pytest.mark.asyncio
    async def test_memories_found_but_not_relevant(
        self, infra_agent: InfrastructureAgent
    ) -> None:
        context: dict[str, Any] = {
            "task_input": "network latency spike",
            "retrieved_memories": [
                {
                    "content": "storage alert on backup volume",
                    "importance": 0.3,
                    "domain": AgentDomain.INFRASTRUCTURE.value,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {},
                    "age_hours": 3.0,
                }
            ],
        }
        result, confidence = await infra_agent._execute_task(context)
        assert "none seem highly relevant" in result
        assert confidence == 0.55