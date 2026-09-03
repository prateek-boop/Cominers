import torch
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import LastAggregator, IdentityMessage

class MemoryModule(torch.nn.Module):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator()
        )
        
    def forward(self, n_id):
        return self.memory(n_id)

    def reset_state(self):
        self.memory.reset_state()

    def detach(self):
        self.memory.detach()

    def update_state(self, src, dst, t, raw_msg):
        self.memory.update_state(src, dst, t, raw_msg)
