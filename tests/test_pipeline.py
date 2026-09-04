import pytest
import asyncio
import torch
from backend.pipeline import StreamingPipeline
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.fixture
def mock_pipeline():
    with patch('backend.pipeline.CICIDSLoader'), \
         patch('backend.pipeline.GraphPruner'), \
         patch('backend.pipeline.FeaturePipeline'), \
         patch('backend.pipeline.GraphBuilder'), \
         patch('backend.pipeline.MemoryModule'), \
         patch('backend.pipeline.CyberTGN'), \
         patch('backend.pipeline.DeltaPredictor'), \
         patch('backend.pipeline.DecisionEngine'):
        pipeline = StreamingPipeline("dummy_path")
        
        # Add basic mocks for ML flow
        pipeline.graph_builder.build_temporal_data = MagicMock(return_value=MagicMock(
            src=torch.tensor([0]), dst=torch.tensor([1]), t=torch.tensor([0.0]), msg=torch.tensor([[0.0]])
        ))
        
        return pipeline

@pytest.mark.asyncio
async def test_pipeline_initialization(mock_pipeline):
    assert mock_pipeline.event_queue is not None
    assert mock_pipeline.loader is not None
    assert mock_pipeline.pruner is not None
    assert mock_pipeline.feature_pipeline is not None

@pytest.mark.asyncio
async def test_ingest_stage(mock_pipeline):
    mock_pipeline.loader.load = MagicMock(return_value=["event1", "event2"])
    
    await mock_pipeline.ingest_stage()
    
    assert mock_pipeline.event_queue.qsize() == 2
    event1 = await mock_pipeline.event_queue.get()
    event2 = await mock_pipeline.event_queue.get()
    
    assert event1 == "event1"
    assert event2 == "event2"

@pytest.mark.asyncio
async def test_process_stage(mock_pipeline):
    # Setup mock returns
    mock_pipeline.pruner.prune = MagicMock(return_value=["pruned_event"])
    mock_pipeline.feature_pipeline.process_window = MagicMock(return_value=["feature"])
    
    # Pre-fill queue with 1000 events to trigger processing
    for i in range(1000):
        await mock_pipeline.event_queue.put(f"event{i}")
        
    # Run process_stage as a task so it doesn't block infinitely
    task = asyncio.create_task(mock_pipeline.process_stage())
    
    # Give it a moment to process the first batch
    await asyncio.sleep(0.1)
    
    # Assertions
    mock_pipeline.pruner.prune.assert_called_once()
    mock_pipeline.feature_pipeline.process_window.assert_called_once()
    
    # Cancel the infinite loop task
    task.cancel()
