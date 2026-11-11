import torch
import torch.nn as nn

import transformer.Constants as Constants
from models.clinical_pooling import build_attention_pooling, pool_sequence


class S2P2StateLayer(nn.Module):
    """Continuous-time diagonal state-space layer matching the S2P2 adapter role."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.decay_log_rate = nn.Parameter(torch.zeros(dim))
        self.impulse = nn.Linear(dim, dim)
        self.depth = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward_step(self, state, impulse, delta_t):
        decay = torch.exp(-torch.nn.functional.softplus(self.decay_log_rate).view(1, -1) * delta_t)
        state = state * decay + self.impulse(impulse)
        return self.norm(state + self.depth(state))


class ClinicalS2P2Encoder(nn.Module):
    """S2P2-style continuous-time state-space adapter for marked event sequences."""

    def __init__(
        self,
        num_types,
        d_model=64,
        n_layers=2,
        dropout=0.1,
        pooling='last',
        time_scale_hours=24.0,
    ):
        super().__init__()
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.layers = nn.ModuleList(S2P2StateLayer(d_model, dropout=dropout) for _ in range(max(int(n_layers), 1)))
        self.dropout = nn.Dropout(dropout)
        self.attention = build_attention_pooling(pooling, d_model)

    def forward(self, event_type, event_time, time_gap=None):
        non_pad = event_type.ne(Constants.PAD)
        if time_gap is None:
            time_gap = torch.zeros_like(event_time)
            time_gap[:, 1:] = torch.clamp(event_time[:, 1:] - event_time[:, :-1], min=0.0)
        impulses = self.dropout(self.event_emb(event_type))
        delta = (time_gap / self.time_scale_hours).clamp_min(0.0).unsqueeze(-1)
        states = [impulses.new_zeros((event_type.size(0), impulses.size(-1))) for _ in self.layers]
        outputs = []
        for index in range(event_type.size(1)):
            mask = non_pad[:, index].unsqueeze(-1).float()
            hidden = impulses[:, index, :]
            dt = delta[:, index, :]
            next_states = []
            for layer_index, layer in enumerate(self.layers):
                updated = layer.forward_step(states[layer_index], hidden, dt)
                state = torch.where(mask.bool(), updated, states[layer_index])
                next_states.append(state)
                hidden = state
            states = next_states
            outputs.append(hidden * mask)
        sequence_hidden = torch.stack(outputs, dim=1)
        pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        return pooled, sequence_hidden
