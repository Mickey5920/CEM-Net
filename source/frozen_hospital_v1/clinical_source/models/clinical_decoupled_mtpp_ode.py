import torch
import torch.nn as nn

import transformer.Constants as Constants
from models.clinical_pooling import build_attention_pooling, pool_sequence


class ODEDynamics(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, state):
        return self.net(state)


class ClinicalDecoupledMTPPODEEncoder(nn.Module):
    """Decoupled marked TPP neural-ODE adapter: time evolution then mark jump."""

    def __init__(
        self,
        num_types,
        d_model=64,
        hidden_dim=64,
        n_layers=2,
        dropout=0.1,
        pooling='last',
        time_scale_hours=24.0,
        ode_steps=2,
    ):
        super().__init__()
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.ode_steps = max(int(ode_steps), 1)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.in_proj = nn.Linear(d_model, hidden_dim)
        self.dynamics = nn.ModuleList(ODEDynamics(hidden_dim, dropout=dropout) for _ in range(max(int(n_layers), 1)))
        self.jump = nn.Sequential(
            nn.LayerNorm(hidden_dim + d_model),
            nn.Linear(hidden_dim + d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = build_attention_pooling(pooling, hidden_dim)

    def evolve(self, state, delta_t):
        step = delta_t / float(self.ode_steps)
        for _ in range(self.ode_steps):
            for dynamics in self.dynamics:
                k1 = dynamics(state)
                k2 = dynamics(state + 0.5 * step * k1)
                k3 = dynamics(state + 0.5 * step * k2)
                k4 = dynamics(state + step * k3)
                state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state

    def forward(self, event_type, event_time, time_gap=None):
        non_pad = event_type.ne(Constants.PAD)
        if time_gap is None:
            time_gap = torch.zeros_like(event_time)
            time_gap[:, 1:] = torch.clamp(event_time[:, 1:] - event_time[:, :-1], min=0.0)
        marks = self.dropout(self.event_emb(event_type))
        delta = (time_gap / self.time_scale_hours).clamp_min(0.0).unsqueeze(-1)
        state = marks.new_zeros((event_type.size(0), self.in_proj.out_features))
        outputs = []
        for index in range(event_type.size(1)):
            mask = non_pad[:, index].unsqueeze(-1).float()
            evolved = self.evolve(state, delta[:, index, :])
            jumped = self.norm(evolved + self.jump(torch.cat([evolved, marks[:, index, :]], dim=-1)))
            state = torch.where(mask.bool(), jumped, state)
            outputs.append(state * mask)
        sequence_hidden = torch.stack(outputs, dim=1)
        pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        return pooled, sequence_hidden
