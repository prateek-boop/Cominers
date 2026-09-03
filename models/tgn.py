import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_dim: int):
        super().__init__()
        self.time_enc = TimeEncoder(time_dim)
        self.conv = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            heads=2,
            edge_dim=msg_dim + time_dim,
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([msg, rel_t_enc], dim=-1)
        return self.conv(x, edge_index, edge_attr)

class CyberTGN(torch.nn.Module):
    def __init__(self, memory_module, in_channels, out_channels, msg_dim, time_dim):
        super().__init__()
        self.memory = memory_module
        self.embedding = GraphAttentionEmbedding(in_channels, out_channels, msg_dim, time_dim)

    def forward(self, n_id, edge_index, t, msg):
        z_mem, last_update = self.memory(n_id)
        z = self.embedding(z_mem, last_update, edge_index, t, msg)
        return z
