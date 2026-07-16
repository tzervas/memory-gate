"""Unit tests for agent_interface uncovered paths."""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory_gate.agent_interface import (
    ERROR_MSG_EMPTY_AGENT_NAME,
    ERROR_MSG_INVALID_AGENT_DOMAIN,
    ERROR_MSG_INVALID_MEMORY_GATEWAY,
    ERROR_MSG_SUBCLASS_MUST_IMPLEMENT,
    AgentDomain,
    BaseMemoryEnabledAgent,
    SimpleEchoAgent,
)
from memory_gate.memory_gateway import MemoryGateway
from memory_gate.memory_protocols import LearningContext, MemoryAdapter


class PassThroughAdapter(MemoryAdapter[LearningContext]):
    async def adapt_knowledge(
        self, context: LearningContext, feedback: float | None = None
    ) -> LearningContext:
        return context


def _make_gateway(
    retrieve_results: list[LearningContext] | None = None,
    learn_side_effect: Exception | None = None,
) -> MemoryGateway[LearningContext]:
    store = AsyncMock()
    store.retrieve_context = AsyncMock(return_value=retrieve_results or [])
    gateway = MemoryGateway(adapter=PassThroughAdapter(), store=store)
    if learn_side_effect is not None:
        gateway.learn_from_interaction = AsyncMock(side_effect=learn_side_effect)
    else:
        gateway.learn_from_interaction = AsyncMock()
    return gateway


class FailingAgent(BaseMemoryEnabledAgent):
    def __init__(
        self,
        memory_gateway: MemoryGateway[LearningContext],
        error: Exception,
    ) -> None:
        super().__init__(
            agent_name="FailingAgent",
            domain=AgentDomain.GENERAL,
            memory_gateway=memory_gateway,
        )
        self._error = error

    async def _execute_task(
        self, enhanced_context: dict[str, Any]
    ) -> tuple[str, float]:
        raise self._error


class TestBaseMemoryEnabledAgentValidation:
    """Test constructor validation errors."""

    def test_empty_agent_name_raises(self) -> None:
        gateway = _make_gateway()
        with pytest.raises(ValueError, match=ERROR_MSG_EMPTY_AGENT_NAME):
            BaseMemoryEnabledAgent("", AgentDomain.GENERAL, gateway)

    def test_invalid_domain_raises(self) -> None:
        gateway = _make_gateway()
        with pytest.raises(ValueError, match=ERROR_MSG_INVALID_AGENT_DOMAIN):
            BaseMemoryEnabledAgent("Agent", "not-a-domain", gateway)  # type: ignore[arg-type]

    def test_invalid_gateway_raises(self) -> None:
        with pytest.raises(ValueError, match=ERROR_MSG_INVALID_MEMORY_GATEWAY):
            BaseMemoryEnabledAgent("Agent", AgentDomain.GENERAL, MagicMock())  # type: ignore[arg-type]


class TestBaseMemoryEnabledAgentExecution:
    """Test task execution error and edge-case paths."""

    @pytest.mark.asyncio
    async def test_runtime_error_during_task_records_failure(self) -> None:
        gateway = _make_gateway()
        agent = FailingAgent(gateway, RuntimeError("task boom"))
        result, confidence = await agent.process_task("fail task")
        assert "Error processing task" in result
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_unexpected_error_during_task(self) -> None:
        gateway = _make_gateway()
        agent = FailingAgent(gateway, Exception("unexpected"))
        result, confidence = await agent.process_task("unexpected fail")
        assert "Unexpected error processing task" in result
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_failure_without_storing_memory_reraises(self) -> None:
        gateway = _make_gateway()
        agent = FailingAgent(gateway, ValueError("no store"))
        with pytest.raises(ValueError, match="no store"):
            await agent.process_task("fail", store_interaction_memory=False)

    @pytest.mark.asyncio
    async def test_failure_summary_includes_failure_reason(self) -> None:
        gateway = _make_gateway()
        agent = FailingAgent(gateway, RuntimeError("disk full"))
        await agent.process_task("disk issue", store_interaction_memory=True)
        learned = gateway.learn_from_interaction.await_args
        assert learned is not None
        context = learned.args[0]
        assert "FailureReason: disk full" in context.content

    @pytest.mark.asyncio
    async def test_execute_task_not_implemented(self) -> None:
        gateway = _make_gateway()
        agent = BaseMemoryEnabledAgent("Bare", AgentDomain.GENERAL, gateway)
        with pytest.raises(NotImplementedError, match=ERROR_MSG_SUBCLASS_MUST_IMPLEMENT):
            await agent._execute_task({"task_input": "x"})


class TestSimpleEchoAgent:
    """Test SimpleEchoAgent memory display paths."""

    @pytest.mark.asyncio
    async def test_echo_with_retrieved_memories_boosts_confidence(self) -> None:
        memory = LearningContext(
            content="Prior incident resolved by restart",
            domain="general",
            timestamp=datetime.now(),
            importance=0.9,
        )
        gateway = _make_gateway(retrieve_results=[memory])
        agent = SimpleEchoAgent(gateway)
        result, confidence = await agent.process_task("check server")
        assert "Retrieved memories:" in result
        assert confidence == 0.6

    def test_feedback_with_new_importance(self, capsys: pytest.CaptureFixture[str]) -> None:
        gateway = _make_gateway()
        agent = SimpleEchoAgent(gateway)
        agent.provide_feedback_on_memory("key-1", 0.8, new_importance=0.95)
        captured = capsys.readouterr()
        assert "Suggested new importance: 0.95" in captured.out