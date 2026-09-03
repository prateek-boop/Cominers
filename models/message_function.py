import torch
from torch import nn
from torch_geometric.nn.models.tgn import MessageModule

class MLPMessageFunction(MessageModule):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        in_dim = 2 * memory_dim + raw_msg_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, raw_msg: torch.Tensor, t_enc: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)
        return self.mlp(x)
