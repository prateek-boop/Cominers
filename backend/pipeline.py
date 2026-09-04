import asyncio
import logging
from typing import List
import torch
from ingestion.schema import NetworkEvent
from ingestion.csv_loader import CICIDSLoader
from pruning.graph_pruner import GraphPruner
from features.feature_pipeline import FeaturePipeline
from graph.graph_builder import GraphBuilder
from models.memory import MemoryModule
from models.tgn import CyberTGN
from forecasting.delta_predictor import DeltaPredictor
from response.decision_engine import DecisionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingPipeline:
    def __init__(self, data_path: str):
        self.loader = CICIDSLoader(data_path)
        self.pruner = GraphPruner("configs/pruning_rules.yaml")
        self.feature_pipeline = FeaturePipeline(window_size_sec=15.0)
        self.graph_builder = GraphBuilder()
        self.event_queue = asyncio.Queue()
        
        # Initialize ML Components (Mocking trained weights for now)
        logger.info("Initializing ML models...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.memory = MemoryModule(num_nodes=10000, raw_msg_dim=16, memory_dim=128, time_dim=128).to(self.device)
        self.tgn = CyberTGN(self.memory, in_channels=128, out_channels=128, msg_dim=16, time_dim=128).to(self.device)
        self.forecaster = DeltaPredictor(state_dim=128).to(self.device)
        self.decision_engine = DecisionEngine("configs/response_policy.yaml")
        
    async def ingest_stage(self):
        try:
            for event in self.loader.load():
                await self.event_queue.put(event)
            logger.info("Ingestion stage completed.")
        except Exception as e:
            logger.error(f"Error during ingestion: {e}", exc_info=True)
            
    async def process_stage(self):
        batch = []
        while True:
            try:
                event = await self.event_queue.get()
                batch.append(event)
                
                if len(batch) >= 1000:
                    logger.info(f"Processing batch of {len(batch)} events")
                    pruned = list(self.pruner.prune(iter(batch)))
                    features = list(self.feature_pipeline.process_window(iter(pruned)))
                    
                    if not features:
                        batch = []
                        continue

                    # 1. Build Temporal Graph Data
                    temporal_data = self.graph_builder.build_temporal_data(features).to(self.device)
                    
                    # 2. Extract batch attributes
                    src, dst, t, msg = temporal_data.src, temporal_data.dst, temporal_data.t, temporal_data.msg
                    edge_index = torch.stack([src, dst], dim=0)
                    
                    # All unique nodes in this batch
                    n_id = torch.cat([src, dst]).unique()
                    
                    logger.info(f"Forwarding {len(features)} windows to TGN (Nodes: {len(n_id)})...")
                    
                    # 3. TGN Forward Pass
                    self.tgn.eval()
                    self.forecaster.eval()
                    with torch.no_grad():
                        z = self.tgn(n_id, edge_index, t, msg)
                        
                        # 4. Forecaster Forward Pass (Expects shape: batch, seq_len, state_dim)
                        # We use the embeddings `z` as the current state.
                        # For simplicity in this streaming setup, we treat the current embeddings as seq_len=1
                        z_seq = z.unsqueeze(1) 
                        forecasts = self.forecaster(z_seq)
                        
                        # Update Memory
                        self.memory.update_state(src, dst, t, msg)
                    
                    # 5. Decision Engine Evaluation (Simulated for the first node's forecast)
                    if len(z) > 0:
                        mock_risk_prob = 0.7  # In reality, from AttackProbabilityHead
                        mock_mitre_stage = 4  # In reality, from MitreStageClassifier
                        action = self.decision_engine.evaluate(
                            risk_prob=mock_risk_prob, 
                            mitre_stage=mock_mitre_stage, 
                            conformal_confirmed=True
                        )
                        logger.info(f"Decision Engine Action: {action}")
                    
                    batch = []
            except asyncio.CancelledError:
                logger.info("Process stage cancelled.")
                break
            except Exception as e:
                logger.error(f"Error during processing batch: {e}", exc_info=True)
                # Clear batch to avoid getting stuck on bad data
                batch = []
