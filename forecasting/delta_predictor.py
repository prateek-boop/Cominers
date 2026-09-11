"""Phase 4: Multi-Step Residual Latent Delta Predictor.

Instead of naive compounding autoregression (which diverges within 3 steps),
predicts zero-centered, bounded residual latent deltas:
Delta S_hat_t = g(S_t) ==> S_hat_{t+k|t} = S_t + Delta S_hat_t

Rollout horizon is strictly bounded to K <= 3 snapshot windows:
- Step 1: +15 seconds (k = 1)
- Step 2: +30 seconds (k = 2)
- Step 3: +45 seconds (k = 3)
"""
import torch
from torch import nn
from typing import Dict


class DeltaPredictor(nn.Module):
    """Predicts future state transition deltas across K=3 snapshot horizons (15s, 30s, 45s)."""

    def __init__(self, state_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Specialized heads for bounded horizon deltas
        self.head_15s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.head_30s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.head_45s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_sequence: [batch_size, seq_len, state_dim]
        Returns:
            Dict containing predicted deltas for '15s', '30s', '45s' horizons.
        """
        if state_sequence.dim() == 2:
            state_sequence = state_sequence.unsqueeze(1)

        out, _ = self.gru(state_sequence)
        last_hidden = out[:, -1, :]

        delta_15s = self.head_15s(last_hidden)
        delta_30s = self.head_30s(last_hidden)
        delta_45s = self.head_45s(last_hidden)

        # Baseline current state S_t
        current_state = state_sequence[:, -1, :]

        return {
            "delta_15s": delta_15s,
            "delta_30s": delta_30s,
            "delta_45s": delta_45s,
            "forecast_15s": current_state + delta_15s,
            "forecast_30s": current_state + delta_30s,
            "forecast_45s": current_state + delta_45s,
        }
