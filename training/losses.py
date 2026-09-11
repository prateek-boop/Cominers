"""Stable classification objectives with explicit masking of unknown stages."""
import torch
from torch.nn import functional as F


def supervised_loss(attack_logits, stage_logits, labels, stages, positive_weight, stage_weights, stage_weight):
    binary = F.binary_cross_entropy_with_logits(attack_logits,labels,pos_weight=positive_weight)
    mask = stages >= 0
    if mask.any() and stage_weight > 0:
        stage = F.cross_entropy(stage_logits[mask],stages[mask],weight=stage_weights)
        stage_denominator = stage_weights[stages[mask]].sum().detach()
    else:
        stage = binary.new_zeros(())
        stage_denominator = binary.new_zeros(())
    return binary + stage_weight*stage, binary, stage, stage_denominator
