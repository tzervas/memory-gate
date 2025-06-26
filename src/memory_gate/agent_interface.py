"""Agent interface and base classes for memory-enabled AI agents in MemoryGate."""
from typing import Any, Optional, Tuple, List, Dict
import asyncio
from enum import Enum
from datetime import datetime

from memory_gate.memory_protocols import LearningContext
from memory_gate.memory_gateway import MemoryGateway
from memory_gate.metrics import (
    AGENT_TASK_DURATION_SECONDS,
    record_agent_task_processed,
    record_agent_memory_learned,
)


class AgentDomain(Enum):
    """Supported agent domains for categorizing memories and tasks."""

    INFRASTRUCTURE = "infrastructure"
    CODE_REVIEW = "code_review"
    DEPLOYMENT = "deployment"
    INCIDENT_RESPONSE = "incident_response"
    GENERAL = "general"  # Default or non-specific domain


class BaseMemoryEnabledAgent:
    """
    Base class for AI agents enhanced with memory capabilities.
    It provides a common framework for interacting with the MemoryGateway.
    """

    def __init__(
        self,
        agent_name: str,
        domain: AgentDomain,
        memory_gateway: MemoryGateway[LearningContext],
        retrieval_limit: int = 5,
    ) -> None:
        """
        Initializes the MemoryEnabledAgent.

        Args:
            agent_name: The name of the agent.
            domain: The primary operational domain of the agent.
            memory_gateway: An instance of MemoryGateway for learning and recall.
            retrieval_limit: Default number of memories to retrieve for context.
        """
        if not agent_name:
            raise ValueError("Agent name cannot be empty.")
        if not isinstance(domain, AgentDomain):
            raise ValueError("Invalid agent domain.")
        if not isinstance(memory_gateway, MemoryGateway):
            raise ValueError("Invalid memory_gateway instance.")

        self.agent_name = agent_name
        self.domain = domain
        self.memory_gateway = memory_gateway
        self.retrieval_limit = retrieval_limit
        self._interaction_count = 0  # Simple interaction counter for metadata

    async def process_task(
        self,
        task_input: str,
        task_specific_context: Optional[Dict[str, Any]] = None,
        store_interaction_memory: bool = True,
    ) -> Tuple[str, float]:  # Returns (result_string, confidence_score)
        """
        Processes a given task, leveraging the memory system, and learns from the interaction.

        Args:
            task_input: The primary input or query for the task.
            task_specific_context: Optional dictionary with additional context for the task \
                execution.
            store_interaction_memory: If True, the interaction and its result are stored as a \
                memory.

        Returns:
            A tuple containing the task's result (string) and a confidence score (float).
        """
        # 1. Retrieve relevant memories
        relevant_memories = await self.memory_gateway.store.retrieve_context(
            query=task_input,
            limit=self.retrieval_limit,
            domain_filter=self.domain.value,  # Filter memories by agent's domain
        )

        # 2. Build enhanced context for task execution
        enhanced_context = self._build_enhanced_context(
            task_input, relevant_memories, task_specific_context or {}
        )

        # 3. Execute the task using the enhanced context
        # Subclasses must implement _execute_task
        task_failed_exception = None
        try:
            with AGENT_TASK_DURATION_SECONDS.labels(
                agent_name=self.agent_name, agent_domain=self.domain.value
            ).time():
                result_str, confidence = await self._execute_task(enhanced_context)
            record_agent_task_processed(
                self.agent_name, self.domain.value, success=True
            )
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            task_failed_exception = e
            result_str = f"Error processing task: {e}"
            confidence = 0.0  # No confidence if task execution failed
            record_agent_task_processed(
                self.agent_name, self.domain.value, success=False
            )
            print(f"Agent {self.agent_name} failed to execute task '{task_input}': {e}")
        except Exception as e:  # pylint: disable=broad-except  # fallback for truly unexpected errors
            task_failed_exception = e
            result_str = f"Unexpected error processing task: {e}"
            confidence = 0.0
            record_agent_task_processed(
                self.agent_name, self.domain.value, success=False
            )
            print(f"Agent {self.agent_name} encountered an unexpected error: {e}")

        # 4. Learn from this interaction (optional)
        if store_interaction_memory:
            # Even if the task failed, we might want to record the failure event.
            # The content of the learned memory should reflect the outcome.
            interaction_summary = (
                f"Agent: {self.agent_name}\nDomain: {self.domain.value}\nTask: {task_input}\n"
                f"Result: {result_str}"
            )
            if task_failed_exception:
                interaction_summary += (
                    f"\nFailureReason: {str(task_failed_exception)}"
                )

            learning_ctx = LearningContext(
                content=interaction_summary,
                domain=self.domain.value,  # Storing with agent's primary domain
                timestamp=datetime.now(),
                importance=confidence,  # Confidence from task execution, or 0.0 for failure
                metadata={
                    "agent_name": self.agent_name,
                    "interaction_id": str(self._interaction_count),
                    "retrieved_memories_count": str(len(relevant_memories)),
                    "task_input_length": str(len(task_input)),
                    "task_execution_status": (
                        "success" if task_failed_exception is None else "failure"
                    ),
                },
            )
            try:
                await self.memory_gateway.learn_from_interaction(
                    learning_ctx, feedback=confidence
                )
                record_agent_memory_learned(self.agent_name, self.domain.value)
            except (RuntimeError, ValueError, KeyError, TypeError) as learn_e:
                print(
                    f"Agent {self.agent_name} failed to learn from interaction: "
                    f"{learn_e}"
                )
            except Exception as learn_e:  # pylint: disable=broad-except
                print(
                    f"Agent {self.agent_name} encountered an unexpected error during "
                    f"learning: {learn_e}"
                )

        self._interaction_count += 1

        if task_failed_exception and not store_interaction_memory:
            # If task failed and we are not storing the interaction, re-raise the original exception
            # so the caller is aware of the task failure directly.
            raise task_failed_exception

        return result_str, confidence

    def _build_enhanced_context(
        self,
        task_input: str,
        retrieved_memories: List[LearningContext],
        base_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Constructs an enhanced context dictionary for task execution.
        This can be overridden by subclasses for more specific context building.

        Args:
            task_input: The primary input for the task.
            retrieved_memories: A list of LearningContext objects retrieved from memory.
            base_context: Initial context dictionary for the task.

        Returns:
            An enhanced context dictionary.
        """
        enhanced = base_context.copy()
        enhanced.update(
            {
                "task_input": task_input,
                "agent_name": self.agent_name,
                "agent_domain": self.domain.value,
                "retrieved_memories": [
                    {
                        "content": mem.content,
                        "domain": mem.domain,
                        "timestamp": mem.timestamp.isoformat(),
                        "importance": mem.importance,
                        "metadata": mem.metadata,
                        # Calculate age for easier use in task logic
                        "age_hours": round(
                            (datetime.now() - mem.timestamp).total_seconds() / 3600, 2
                        ),
                    }
                    for mem in retrieved_memories
                ],
            }
        )
        return enhanced

    async def _execute_task(
        self, enhanced_context: Dict[str, Any]
    ) -> Tuple[str, float]:  # (result_string, confidence_score)
        """
        Executes the agent's specific task logic using the enhanced context.
        This method MUST be implemented by subclasses.

        Args:
            enhanced_context: The context dictionary, enriched with retrieved memories.

        Returns:
            A tuple containing the task's result (string) and a confidence score (float).
        """
        raise NotImplementedError("Subclasses must implement the _execute_task method.")

    def provide_feedback_on_memory(
        self,
        memory_key: str,  # Assuming memories can be identified by a key
        feedback_score: float,  # e.g., 1.0 for very useful, -1.0 for not useful
        new_importance: Optional[float] = None,
    ) -> None:
        """
        Allows the agent or an external system to provide feedback on a specific memory.
        This is a placeholder for a more complex feedback mechanism.
        The MemoryGateway and KnowledgeStore would need to support updating memories.

        Args:
            memory_key: The key of the memory to provide feedback on.
            feedback_score: A score indicating the usefulness or correctness of the memory.
            new_importance: Optionally, a new importance score for the memory.
        """
        # This functionality would require methods in MemoryGateway/KnowledgeStore
        # like `update_memory_importance(key, new_importance)` or
        # `record_feedback(key, score)`.
        print(
            f"Feedback received for memory '{memory_key}': score={feedback_score}, "
            f"new_importance={new_importance}"
        )
        print(
            "Note: Actual feedback processing logic needs to be implemented in "
            "MemoryGateway/KnowledgeStore."
        )
        # Example: self.memory_gateway.update_experience_importance(memory_key, new_importance)


# Example of a concrete agent (will be moved to a separate file or section later)
class SimpleEchoAgent(BaseMemoryEnabledAgent):
    """A simple agent that echoes the task input and any retrieved memories."""

    def __init__(self, memory_gateway: MemoryGateway[LearningContext]) -> None:
        super().__init__(
            agent_name="EchoAgent",
            domain=AgentDomain.GENERAL,
            memory_gateway=memory_gateway,
        )

    async def _execute_task(
        self, enhanced_context: Dict[str, Any]
    ) -> Tuple[str, float]:
        task_input = enhanced_context.get("task_input", "No input provided.")
        memories_retrieved = enhanced_context.get("retrieved_memories", [])

        response_parts = [f"EchoAgent processed: {task_input}"]
        if memories_retrieved:
            response_parts.append("Retrieved memories:")
            for mem in memories_retrieved:
                response_parts.append(
                    f"  - {mem['content'][:100]}... (Importance: {mem['importance']})"
                )

        result_str = "\n".join(response_parts)
        confidence = 0.5  # Default confidence for a simple echo
        if (
            memories_retrieved
        ):  # Slightly higher confidence if memories were found (even if not used)
            confidence = 0.6

        await asyncio.sleep(0.01)  # Simulate some async work
        return result_str, confidence
