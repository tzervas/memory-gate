from typing import Any, Dict, Tuple, List
import asyncio

from memory_gate.agent_interface import BaseMemoryEnabledAgent, AgentDomain
from memory_gate.memory_protocols import LearningContext
from memory_gate.memory_gateway import MemoryGateway


class InfrastructureAgent(BaseMemoryEnabledAgent):
    """
    An agent specialized in diagnosing and suggesting solutions for
    infrastructure-related tasks and issues.
    """

    def __init__(self, memory_gateway: MemoryGateway[LearningContext]) -> None:
        super().__init__(
            agent_name="InfraTroubleshooterAgent",
            domain=AgentDomain.INFRASTRUCTURE,
            memory_gateway=memory_gateway,
            retrieval_limit=5,  # Retrieve up to 5 relevant past incidents/solutions
        )

    async def _execute_task(
        self, enhanced_context: Dict[str, Any]
    ) -> Tuple[str, float]:  # (result_string, confidence_score)
        """
        Executes an infrastructure-related task, such as diagnosing an issue.

        Args:
            enhanced_context: Context dictionary with task_input and retrieved_memories.

        Returns:
            A tuple containing a diagnostic message or solution suggestion, and a confidence score.
        """
        task_input: str = enhanced_context.get(
            "task_input", "No specific task provided."
        )
        retrieved_memories: List[Dict[str, Any]] = enhanced_context.get(
            "retrieved_memories", []
        )

        response_parts: List[str] = []
        confidence: float = 0.3  # Base confidence for a new, unanalyzed issue

        response_parts.append(f"Analyzing infrastructure task: '{task_input}'")

        if not retrieved_memories:
            response_parts.append(
                "No directly relevant past experiences found in memory."
            )
            response_parts.append(
                "Recommendation: Investigate as a new issue. Check standard monitoring tools and logs."
            )
            confidence = (
                0.4  # Slightly higher confidence if we explicitly know it's new
            )
        else:
            response_parts.append(
                f"Found {len(retrieved_memories)} potentially relevant past experiences:"
            )

            # Simple strategy: check for high-importance memories or very similar content
            # A more advanced agent would do deeper analysis, NLP matching, etc.
            highly_relevant_suggestions = []
            for mem in retrieved_memories:
                # Example: if a past memory has high importance and mentions a similar term
                # This is a very naive similarity check.
                if mem["importance"] >= 0.75:
                    # Simulate extracting a solution from the memory content
                    # Real solution extraction would be more complex
                    if "solution:" in mem["content"].lower():
                        solution_part = mem["content"][
                            mem["content"].lower().find("solution:") :
                        ]
                        highly_relevant_suggestions.append(
                            f"High-importance similar past event (Importance: {mem['importance']}): {mem['content'][:150]}...\n  -> Suggested from past: {solution_part[:100]}..."
                        )
                    else:
                        highly_relevant_suggestions.append(
                            f"High-importance similar past event (Importance: {mem['importance']}): {mem['content'][:150]}..."
                        )
                elif (
                    task_input.split(" ")[0].lower() in mem["content"].lower()
                ):  # very basic keyword match
                    highly_relevant_suggestions.append(
                        f"Keyword match past event (Importance: {mem['importance']}): {mem['content'][:150]}..."
                    )

            if highly_relevant_suggestions:
                response_parts.append("\nSuggestions based on past experiences:")
                response_parts.extend([f"  - {s}" for s in highly_relevant_suggestions])
                confidence = (
                    0.75  # Higher confidence if relevant memories provide suggestions
                )
            else:
                response_parts.append(
                    "Past experiences found, but none seem highly relevant or offer clear solutions."
                )
                response_parts.append(
                    "Recommendation: Review the retrieved memories for general patterns, but treat as a new variant of issue."
                )
                confidence = 0.55

        # Simulate some processing time
        await asyncio.sleep(0.05 + 0.01 * len(retrieved_memories))

        return "\n".join(response_parts), confidence
