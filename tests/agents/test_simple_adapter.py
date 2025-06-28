import pytest
from datetime import datetime

from memory_gate.memory_protocols import LearningContext
from memory_gate.agents.simple_adapter import SimpleMemoryAdapter


@pytest.fixture
def simple_adapter() -> SimpleMemoryAdapter:
    """Fixture to create a SimpleMemoryAdapter instance."""
    return SimpleMemoryAdapter()


@pytest.fixture
def sample_context() -> LearningContext:
    """Fixture to create a sample LearningContext instance."""
    return LearningContext(
        content="Sample content for adapter test",
        domain="adapter_test",
        timestamp=datetime.now(),
        importance=0.75  # Initial importance
    )


@pytest.mark.asyncio
async def test_adapt_knowledge_no_feedback(
    simple_adapter: SimpleMemoryAdapter, sample_context: LearningContext
) -> None:
    """Test adapt_knowledge when no feedback is provided."""
    original_importance = sample_context.importance
    adapted_context = await simple_adapter.adapt_knowledge(sample_context)

    assert adapted_context == sample_context  # Should be the same instance
    assert adapted_context.content == sample_context.content
    assert adapted_context.importance == original_importance  # Importance should not change


@pytest.mark.asyncio
async def test_adapt_knowledge_with_valid_feedback_0_to_1(
    simple_adapter: SimpleMemoryAdapter, sample_context: LearningContext
) -> None:
    """Test adapt_knowledge with feedback within the 0.0 to 1.0 range."""
    feedback_score = 0.5  # Example feedback score
    original_importance = sample_context.importance

    adapted_context = await simple_adapter.adapt_knowledge(sample_context, feedback=feedback_score)

    assert adapted_context == sample_context
    expected_importance = (original_importance + feedback_score) / 2.0
    assert adapted_context.importance == pytest.approx(expected_importance)


@pytest.mark.asyncio
async def test_adapt_knowledge_with_feedback_outside_0_to_1(
    simple_adapter: SimpleMemoryAdapter, sample_context: LearningContext
) -> None:
    """Test adapt_knowledge with feedback outside the 0.0 to 1.0 range (should directly assign)."""
    feedback_score = 5.0  # Example feedback score outside 0-1

    adapted_context = await simple_adapter.adapt_knowledge(sample_context, feedback=feedback_score)

    assert adapted_context == sample_context
    assert adapted_context.importance == feedback_score


@pytest.mark.asyncio
async def test_adapt_knowledge_with_zero_feedback(
    simple_adapter: SimpleMemoryAdapter, sample_context: LearningContext
) -> None:
    """Test adapt_knowledge with feedback score of 0.0."""
    feedback_score = 0.0
    original_importance = sample_context.importance

    adapted_context = await simple_adapter.adapt_knowledge(sample_context, feedback=feedback_score)

    assert adapted_context == sample_context
    expected_importance = (original_importance + feedback_score) / 2.0
    assert adapted_context.importance == pytest.approx(expected_importance)


@pytest.mark.asyncio
async def test_adapt_knowledge_with_one_feedback(
    simple_adapter: SimpleMemoryAdapter, sample_context: LearningContext
) -> None:
    """Test adapt_knowledge with feedback score of 1.0."""
    feedback_score = 1.0
    original_importance = sample_context.importance

    adapted_context = await simple_adapter.adapt_knowledge(sample_context, feedback=feedback_score)

    assert adapted_context == sample_context
    expected_importance = (original_importance + feedback_score) / 2.0
    assert adapted_context.importance == pytest.approx(expected_importance)


@pytest.mark.asyncio
async def test_adapt_knowledge_context_instance_unchanged_identity(
    simple_adapter: SimpleMemoryAdapter, sample_context: LearningContext
) -> None:
    """Test that the adapter returns the same context instance."""
    adapted_context = await simple_adapter.adapt_knowledge(sample_context, feedback=0.9)
    assert adapted_context is sample_context # Check for instance identity
