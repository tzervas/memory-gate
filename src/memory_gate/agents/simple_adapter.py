from typing import Optional

from memory_gate.memory_protocols import LearningContext, MemoryAdapter


class SimpleMemoryAdapter(MemoryAdapter[LearningContext]):
    """
    A simple memory adapter that passes through the learning context.
    If feedback is provided, it can adjust the importance of the context.
    """

    async def adapt_knowledge(
        self, context: LearningContext, feedback: Optional[float] = None
    ) -> LearningContext:
        """
        Adapts the learning context. If feedback is provided,
        it updates the context's importance.

        Args:
            context: The LearningContext object to adapt.
            feedback: An optional float score (e.g., 0.0 to 1.0)
                      representing the perceived value or correctness of the context.

        Returns:
            The adapted LearningContext. This implementation returns the same
            context instance, potentially modified.
        """
        if feedback is not None:
            # Example: Simple averaging of current importance and feedback
            # Ensure feedback is within a reasonable range if it has specific bounds
            # For instance, if feedback is a score from 0 to 1:
            if 0.0 <= feedback <= 1.0:
                context.importance = (context.importance + feedback) / 2.0
            else:
                # Handle feedback outside expected range, e.g., clamp or log
                # For this simple adapter, we might just assign it if it's a raw score
                context.importance = feedback

        # The adapter could do more, like transforming the content,
        # adding metadata, or categorizing the context based on feedback.
        # For now, it's a passthrough with optional importance adjustment.
        return context
