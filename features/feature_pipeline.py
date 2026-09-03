import numpy as np
from typing import Iterator, Tuple
from ingestion.schema import NetworkEvent
from .flow_features import FlowFeatureExtractor

class FeaturePipeline:
    def __init__(self, window_size_sec: float = 15.0):
        self.flow_extractor = FlowFeatureExtractor(window_size_sec)
        
    def process_window(self, events: Iterator[NetworkEvent]) -> Iterator[Tuple[Tuple[str, str, int, int, str], np.ndarray, float, str]]:
        for event in events:
            self.flow_extractor.process_event(event)
            
        for key in self.flow_extractor.flows.keys():
            flow_feats = self.flow_extractor.extract_features(key)
            events_list = self.flow_extractor.flows[key]
            
            # Use timestamp of the first packet in the flow window
            timestamp = events_list[0].timestamp
            # Use label of the first packet 
            label = events_list[0].label
            
            vector = np.concatenate([
                np.array([
                    flow_feats.duration,
                    flow_feats.packet_count,
                    flow_feats.byte_count,
                    flow_feats.mean_iat,
                    flow_feats.std_iat,
                    flow_feats.syn_count,
                    flow_feats.ack_count
                ], dtype=np.float32),
                flow_feats.packet_features_mean
            ])
            yield key, vector, timestamp, label
            
        self.flow_extractor.flows.clear()
