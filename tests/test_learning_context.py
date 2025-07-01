from datetime import datetime

from memory_gate.memory_protocols import LearningContext


def test_learning_context_post_init() -> None:
    """Test the __post_init__ method of the LearningContext dataclass."""
    context = LearningContext(
        content="Test content", domain="test", timestamp=datetime.now()
    )
    assert context.metadata == {}
