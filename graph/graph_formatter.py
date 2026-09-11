"""Graph Formatter.

Maps IP addresses to integer node IDs and transforms aggregated flows
into dynamic temporal graph edges (src, dst, timestamp, message features)
for the CyberTGN engine.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch

from features.flow_aggregator import FlowRecord


@dataclass
class FormattedGraphBatch:
    src_nodes: torch.Tensor
    dst_nodes: torch.Tensor
    timestamps: torch.Tensor
    edge_features: torch.Tensor
    flows: List[FlowRecord]
    raw_features: np.ndarray


class GraphFormatter:
    """Manages IP-to-node ID mapping and prepares tensor batches for CyberTGN."""

    def __init__(self, max_nodes: int = 10000):
        self.max_nodes = max_nodes
        self.ip_to_id: Dict[str, int] = {}
        self.id_to_ip: Dict[int, str] = {}

    def get_or_create_node_id(self, ip: str) -> int:
        if ip not in self.ip_to_id:
            if len(self.ip_to_id) >= self.max_nodes:
                raise ValueError(f"Exceeded maximum node capacity: {self.max_nodes}")
            node_id = len(self.ip_to_id)
            self.ip_to_id[ip] = node_id
            self.id_to_ip[node_id] = ip
        return self.ip_to_id[ip]

    def format_flows(self, flows: List[FlowRecord]) -> FormattedGraphBatch:
        """Format a batch of FlowRecords into CyberTGN input tensors."""
        if not flows:
            raise ValueError("Flow list cannot be empty")

        src_indices = [self.get_or_create_node_id(f.src_ip) for f in flows]
        dst_indices = [self.get_or_create_node_id(f.dst_ip) for f in flows]
        times = [f.last_time for f in flows]

        # Extract 16 features for each flow
        raw_feature_matrix = np.array([f.to_16_features() for f in flows], dtype=np.float64)

        # Apply standard CyberTGN feature normalization: sign(x) * log1p(abs(x))
        transformed = np.sign(raw_feature_matrix) * np.log1p(np.abs(raw_feature_matrix))
        # Clip numerical outliers to match training bounds
        transformed = np.clip(transformed, -100.0, 100.0)

        # Build tensors matching model_runtime convention
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        dst_tensor = torch.tensor(dst_indices, dtype=torch.long)
        # Match training timestamp conversion (float32 rounding before long)
        t_tensor = torch.tensor(times, dtype=torch.float32).long()
        msg_tensor = torch.tensor(transformed, dtype=torch.float32)

        return FormattedGraphBatch(
            src_nodes=src_tensor,
            dst_nodes=dst_tensor,
            timestamps=t_tensor,
            edge_features=msg_tensor,
            flows=flows,
            raw_features=raw_feature_matrix,
        )

    def reset(self):
        """Clear node mapping if graph needs complete reinitialization."""
        self.ip_to_id.clear()
        self.id_to_ip.clear()
