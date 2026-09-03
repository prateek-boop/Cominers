import torch
from torch_geometric.data import TemporalData
from typing import List, Tuple
import numpy as np
from ingestion.normalizer import Normalizer
from ingestion.label_mapper import LabelMapper

class GraphBuilder:
    def __init__(self):
        self.normalizer = Normalizer()
        self.label_mapper = LabelMapper()

    def build_temporal_data(self, flow_features: List[Tuple[Tuple[str, str, int, int, str], np.ndarray, float, str]]) -> TemporalData:
        src_nodes = []
        dst_nodes = []
        timestamps = []
        edge_attrs = []
        labels = []
        
        flow_features.sort(key=lambda x: x[2])
        
        for key, feat, t, label in flow_features:
            src_ip, dst_ip = key[0], key[1]
            src_id = self.normalizer.get_node_id(src_ip)
            dst_id = self.normalizer.get_node_id(dst_ip)
            
            src_nodes.append(src_id)
            dst_nodes.append(dst_id)
            timestamps.append(t)
            edge_attrs.append(feat)
            labels.append(self.label_mapper.map_to_int(label))
            
        return TemporalData(
            src=torch.tensor(src_nodes, dtype=torch.long),
            dst=torch.tensor(dst_nodes, dtype=torch.long),
            t=torch.tensor(timestamps, dtype=torch.float),
            msg=torch.tensor(np.stack(edge_attrs), dtype=torch.float) if edge_attrs else torch.empty((0, 16), dtype=torch.float),
            y=torch.tensor(labels, dtype=torch.long)
        )
