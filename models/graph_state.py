import torch
from torch_geometric.nn import global_mean_pool

class GraphStateGenerator:
    def __call__(self, z: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        return global_mean_pool(z, batch)
