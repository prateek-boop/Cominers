import torch
from torch import nn

class DeltaPredictor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.head_15s = nn.Linear(hidden_dim, state_dim)
        self.head_30s = nn.Linear(hidden_dim, state_dim)
        self.head_45s = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, x: torch.Tensor) -> dict:
        out, hidden = self.gru(x)
        last_out = out[:, -1, :]
        
        return {
            '15s': self.head_15s(last_out),
            '30s': self.head_30s(last_out),
            '45s': self.head_45s(last_out)
        }
