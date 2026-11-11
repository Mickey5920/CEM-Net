import torch
import torch.nn as nn


def build_attention_pooling(pooling, hidden_dim):
    """Create an attention scorer only when attention pooling is requested."""
    if pooling == 'attention':
        return nn.Linear(int(hidden_dim), 1)
    return None


def pool_sequence(sequence_hidden, non_pad_mask, pooling='last', attention=None):
    """Pool valid sequence states for clinical prediction."""
    if non_pad_mask.dim() == 2:
        non_pad_mask = non_pad_mask.unsqueeze(-1)
    non_pad_mask = non_pad_mask.float()

    if pooling == 'mean':
        lengths = non_pad_mask.sum(dim=1).clamp_min(1.0)
        return sequence_hidden.sum(dim=1) / lengths

    if pooling == 'attention':
        if attention is None:
            raise ValueError('attention pooling requires an attention layer')
        scores = attention(sequence_hidden).squeeze(-1)
        valid = non_pad_mask.squeeze(-1).bool()
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return (sequence_hidden * weights * non_pad_mask).sum(dim=1)

    if pooling != 'last':
        raise ValueError(f'Unsupported pooling: {pooling}')
    lengths = non_pad_mask.squeeze(-1).sum(dim=1).long().clamp_min(1)
    gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, sequence_hidden.size(-1))
    return sequence_hidden.gather(dim=1, index=gather_index).squeeze(1)
