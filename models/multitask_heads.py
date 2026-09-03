import torch
from torch import nn

class AttackProbabilityHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )
        
    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_src, z_dst], dim=-1)
        out = self.mlp(z)
        return torch.sigmoid(out).squeeze(-1)

class MitreStageClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_classes)
        )
        
    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(z)
