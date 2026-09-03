import torch
from collections import defaultdict
from typing import Dict
from torch_geometric.data import TemporalData

class NodeFeatureCalculator:
    def __init__(self):
        self.in_degree: Dict[int, int] = defaultdict(int)
        self.out_degree: Dict[int, int] = defaultdict(int)

    def calculate_static_features(self, data: TemporalData, num_nodes: int) -> torch.Tensor:
        features = torch.zeros((num_nodes, 2), dtype=torch.float)
        
        for src, dst in zip(data.src.tolist(), data.dst.tolist()):
            self.out_degree[src] += 1
            self.in_degree[dst] += 1
            
        for i in range(num_nodes):
            features[i, 0] = self.in_degree[i]
            features[i, 1] = self.out_degree[i]
            
        return features
