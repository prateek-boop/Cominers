"""Optional v2 current-flow logits; legacy checkpoints remain history-only."""
import torch
from torch import nn


class CurrentFlowHead(nn.Module):
    def __init__(self, msg_dim=16, hidden_dim=64, num_classes=8):
        super().__init__()
        self.register_buffer('mean', torch.zeros(msg_dim))
        self.register_buffer('scale', torch.ones(msg_dim))
        self.net = nn.Sequential(nn.Linear(msg_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1 + num_classes))

    def forward(self, msg):
        logits = self.net((msg - self.mean) / self.scale.clamp_min(1e-6))
        return logits[:, 0], logits[:, 1:]


def load_current_flow(checkpoint):
    state = checkpoint.get('current_flow_head')
    if state is None:
        return None
    head = CurrentFlowHead(state['mean'].numel(), state['net.0.weight'].shape[0],
                           state['net.2.weight'].shape[0] - 1)
    head.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise ValueError('Current-flow checkpoint contains non-finite values')
    head.eval()
    return head


def combined_logits(prob_head, stage_head, current_flow_head, src, dst, msg):
    attack, stage = prob_head(src, dst), stage_head(src, dst)
    if current_flow_head is not None:
        direct_attack, direct_stage = current_flow_head(msg)
        attack, stage = attack + direct_attack, stage + direct_stage
    return attack, stage


def encode_timestamps(values, encoding='legacy_float32'):
    if encoding not in ('legacy_float32', 'int64_seconds'):
        raise ValueError(f'Unsupported timestamp encoding: {encoding}')
    return torch.as_tensor(values, dtype=torch.float64 if encoding == 'int64_seconds' else torch.float32).long()
