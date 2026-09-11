"""Checkpoint-declared time transforms; legacy weights keep identity behavior."""
import torch
from torch import nn


class TransformedTimeEncoder(nn.Module):
    def __init__(self, encoder, transform='identity'):
        super().__init__()
        if transform not in ('identity','signed_log1p'):
            raise ValueError('Unsupported time transform')
        self.lin=encoder.lin
        self.out_channels=encoder.out_channels
        self.transform=transform

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self,t):
        if self.transform=='signed_log1p':
            t=torch.sign(t)*torch.log1p(torch.abs(t))
        return self.lin(t.view(-1,1)).cos()
