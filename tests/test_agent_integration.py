import pytest
import asyncio
from datetime import datetime, timedelta  # Added timedelta here

from memory_gate import MemoryGateway, LearningContext, AgentDomain, SimpleEchoAgent
from memory_gate.agents import InfrastructureAgent
from memory_gate.storage.vector_store import VectorMemoryStore
from memory_gate.memory_protocols import MemoryAdapter  # For mocking

# Reuse the persistent vector store fixture from test_vector_store


class MockMemoryAdapter(MemoryAdapter[LearningContext]):
    async def adapt_knowledge(
        self, context: LearningContext, feedback: float | None = None
    ) -> LearningContext:
        # Simple pass-through adapter for testing
        if feedback is not None and 0.0 <= feedback <= 1.0:
            # Simulate feedback influencing importance if needed
            context.importance = (context.importance + feedback) / 2
        return context # This was the mis-indented line, now correctly part of adapt_knowledge

import pytest_asyncio # Add this import


@pytest_asyncio.fixture # Changed decorator
async def memory_gateway_for_agent(
    persistent_vector_store: VectorMemoryStore, # This comes from conftest.py
) -> MemoryGateway[LearningContext]:
    """Create a MemoryGateway with a persistent VectorMemoryStore and MockMemoryAdapter."""
    adapter = MockMemoryAdapter()
    # persistent_vector_store is already a KnowledgeStore[LearningContext]
    # Ensure persistent_vector_store is awaited if it's an async fixture itself,
    # but here it's passed as an already resolved value by pytest.
    return MemoryGateway(adapter=adapter, store=persistent_vector_store)


@pytest_asyncio.fixture # Changed decorator
async def echo_agent( # Changed to async def
    memory_gateway_for_agent: MemoryGateway[LearningContext], # This will be awaited by pytest-asyncio
) -> SimpleEchoAgent:
    return SimpleEchoAgent(memory_gateway=memory_gateway_for_agent)


@pytest_asyncio.fixture # Changed decorator
async def infra_agent( # Changed to async def
    memory_gateway_for_agent: MemoryGateway[LearningContext], # This will be awaited by pytest-asyncio
) -> InfrastructureAgent:
    return InfrastructureAgent(memory_gateway=memory_gateway_for_agent)


@pytest.mark.asyncio
async def test_simple_echo_agent_process_task_and_learn(
    echo_agent: SimpleEchoAgent,
    memory_gateway_for_agent: MemoryGateway[LearningContext],
) -> None:
    """Test SimpleEchoAgent processing a task and storing the interaction."""
    task_input = "Hello, Echo!"
    initial_mem_count = memory_gateway_for_agent.store.get_collection_size()

    result_str, confidence = await echo_agent.process_task(task_input)

    assert task_input in result_str
    assert confidence >= 0.5

    # Allow time for async storage in MemoryGateway.learn_from_interaction
    await asyncio.sleep(0.1)

    assert memory_gateway_for_agent.store.get_collection_size() == initial_mem_count + 1

    # Verify the learned memory content
    # The query for retrieve_context should be specific enough
    retrieved_memories = await memory_gateway_for_agent.store.retrieve_context(
        query=f"Agent: {echo_agent.agent_name}\nTask: {task_input}",
        limit=1,
        domain_filter=echo_agent.domain.value,
    )
    assert len(retrieved_memories) == 1
    learned_memory = retrieved_memories[0]
    assert f"Task: {task_input}" in learned_memory.content
    assert f"Result: {result_str}" in learned_memory.content
    assert learned_memory.domain == echo_agent.domain.value
    assert learned_memory.metadata["agent_name"] == echo_agent.agent_name


@pytest.mark.asyncio
async def test_infra_agent_process_task_no_prior_memory(
    infra_agent: InfrastructureAgent,
    memory_gateway_for_agent: MemoryGateway[LearningContext],
) -> None:
    """Test InfrastructureAgent with no relevant prior memories."""
    task_input = "Web server is down, showing 503 errors."
    initial_mem_count = memory_gateway_for_agent.store.get_collection_size()

    result_str, confidence = await infra_agent.process_task(task_input)

    assert "No directly relevant past experiences found" in result_str
    assert confidence < 0.6  # Should be lower confidence without relevant memories

    await asyncio.sleep(0.1)
    assert memory_gateway_for_agent.store.get_collection_size() == initial_mem_count + 1

    # Check that the interaction itself was learned
    retrieved = await memory_gateway_for_agent.store.retrieve_context(
        query=task_input, domain_filter=AgentDomain.INFRASTRUCTURE.value
    )
    assert any(f"Task: {task_input}" in mem.content for mem in retrieved)


@pytest.mark.asyncio
async def test_infra_agent_uses_relevant_memory(
    infra_agent: InfrastructureAgent,
    memory_gateway_for_agent: MemoryGateway[LearningContext],
) -> None:
    """Test InfrastructureAgent using a relevant past memory."""
    store = memory_gateway_for_agent.store

    # Pre-populate a relevant memory
    past_task = "Server alpha unresponsive"
    past_solution = "Solution: Rebooted server alpha, confirmed network config."
    relevant_memory_content = (
        f"Task: {past_task}\nResult: Resolved by operator.\n{past_solution}"
    )

    past_lc = LearningContext(
        content=relevant_memory_content,
        domain=AgentDomain.INFRASTRUCTURE.value,
        timestamp=datetime.now() - timedelta(days=5),
        importance=0.85,  # High importance
        metadata={"incident_id": "INC001", "resolved_by": "ops_team"},
    )
    # Generate key for storage (MemoryGateway does this internally, here we do it for direct store access)
    key = memory_gateway_for_agent._generate_key(past_lc)
    await store.store_experience(key, past_lc)

    initial_mem_count = store.get_collection_size()
    assert initial_mem_count >= 1

    current_task_input = "Server alpha is very slow and occasionally unresponsive."
    result_str, confidence = await infra_agent.process_task(current_task_input)

    assert "Suggestions based on past experiences:" in result_str
    assert (
        past_solution[:50] in result_str
    )  # Check if part of the solution is suggested
    assert confidence > 0.6  # Should be higher confidence with relevant memories

    await asyncio.sleep(0.1)
    assert (
        store.get_collection_size() == initial_mem_count + 1
    )  # Current interaction also learned


@pytest.mark.asyncio
async def test_agent_does_not_learn_if_flagged(
    echo_agent: SimpleEchoAgent,
    memory_gateway_for_agent: MemoryGateway[LearningContext],
) -> None:
    """Test that agent does not store memory if store_interaction_memory is False."""
    task_input = "Confidential task, do not record."
    initial_mem_count = memory_gateway_for_agent.store.get_collection_size()

    await echo_agent.process_task(task_input, store_interaction_memory=False)

    await asyncio.sleep(0.1)  # Ensure async tasks would have completed

    assert memory_gateway_for_agent.store.get_collection_size() == initial_mem_count


# Test for the feedback placeholder
def test_base_agent_feedback_placeholder(echo_agent: SimpleEchoAgent, capsys) -> None:
    """Test the placeholder feedback mechanism prints expected message."""
    echo_agent.provide_feedback_on_memory("some_key", 0.9, 0.95)
    captured = capsys.readouterr()
    assert "Feedback received for memory 'some_key'" in captured.out
    assert "Actual feedback processing logic needs to be implemented" in captured.out


# TODO: Add tests for other DevOps agents once they are implemented.
# TODO: Add tests for multi-agent memory sharing (e.g., ensuring domain filtering works as expected
#       when multiple agents populate the same store with memories from different domains).
