import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from ingestion.schema import NetworkEvent
from .packet_features import PacketFeatureExtractor

@dataclass
class FlowFeatures:
    duration: float = 0.0
    packet_count: int = 0
    byte_count: int = 0
    mean_iat: float = 0.0
    std_iat: float = 0.0
    syn_count: int = 0
    ack_count: int = 0
    packet_features_mean: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float32))

class FlowFeatureExtractor:
    def __init__(self, window_size_sec: float = 15.0):
        self.window_size_sec = window_size_sec
        self.flows: Dict[Tuple[str, str, int, int, str], List[NetworkEvent]] = defaultdict(list)
        self.packet_extractor = PacketFeatureExtractor()

    def _get_flow_key(self, event: NetworkEvent) -> Tuple[str, str, int, int, str]:
        if event.src_ip < event.dst_ip:
            return (event.src_ip, event.dst_ip, event.src_port, event.dst_port, event.protocol)
        return (event.dst_ip, event.src_ip, event.dst_port, event.src_port, event.protocol)

    def process_event(self, event: NetworkEvent):
        key = self._get_flow_key(event)
        self.flows[key].append(event)
        
    def extract_features(self, key: Tuple[str, str, int, int, str]) -> FlowFeatures:
        events = self.flows.get(key, [])
        if not events:
            return FlowFeatures()
            
        events.sort(key=lambda x: x.timestamp)
        duration = events[-1].timestamp - events[0].timestamp
        
        iats = [events[i].timestamp - events[i-1].timestamp for i in range(1, len(events))]
        mean_iat = np.mean(iats) if iats else 0.0
        std_iat = np.std(iats) if iats else 0.0
        
        packet_feats = np.stack([self.packet_extractor.extract(e) for e in events])
        mean_feats = np.mean(packet_feats, axis=0)
        
        syn_count = int(np.sum(packet_feats[:, 4]))
        ack_count = int(np.sum(packet_feats[:, 5]))
        
        return FlowFeatures(
            duration=duration,
            packet_count=len(events),
            byte_count=sum(e.packet_size for e in events),
            mean_iat=float(mean_iat),
            std_iat=float(std_iat),
            syn_count=syn_count,
            ack_count=ack_count,
            packet_features_mean=mean_feats
        )
