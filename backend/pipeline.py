import asyncio
from typing import List
from ingestion.schema import NetworkEvent
from ingestion.csv_loader import CICIDSLoader
from pruning.graph_pruner import GraphPruner
from features.feature_pipeline import FeaturePipeline

class StreamingPipeline:
    def __init__(self, data_path: str):
        self.loader = CICIDSLoader(data_path)
        self.pruner = GraphPruner("configs/pruning_rules.yaml")
        self.feature_pipeline = FeaturePipeline(window_size_sec=15.0)
        self.event_queue = asyncio.Queue()
        
    async def ingest_stage(self):
        for event in self.loader.load():
            await self.event_queue.put(event)
            
    async def process_stage(self):
        batch = []
        while True:
            event = await self.event_queue.get()
            batch.append(event)
            
            if len(batch) >= 1000:
                pruned = list(self.pruner.prune(iter(batch)))
                features = list(self.feature_pipeline.process_window(iter(pruned)))
                # Integrates with TGN, Forecasting, and Decision Engine
                batch = []
