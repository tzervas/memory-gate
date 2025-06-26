import pytest
import asyncio
from unittest.mock import AsyncMock

from memory_gate.consolidation import ConsolidationWorker
from memory_gate.memory_protocols import LearningContext, KnowledgeStore

@pytest.fixture
def consolidation_worker() -> ConsolidationWorker:
    """Create a consolidation worker."""
    store = AsyncMock(spec=KnowledgeStore)
    return ConsolidationWorker(store, consolidation_interval=0.1)

@pytest.mark.asyncio
async def test_consolidation_loop(consolidation_worker: ConsolidationWorker) -> None:
    """Test the consolidation loop."""
    # Mock the perform_consolidation method
    consolidation_worker._perform_consolidation = AsyncMock()

    await consolidation_worker.start()
    await asyncio.sleep(0.2)
    await consolidation_worker.stop()

    # Check that perform_consolidation was called
    assert consolidation_worker._perform_consolidation.call_count > 0

