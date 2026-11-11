import torch
import torch.nn as nn

import transformer.Constants as Constants
from models.clinical_pooling import build_attention_pooling, pool_sequence


class ResidualTemporalConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.1):
        super().__init__()
        padding = int(kernel_size) // 2
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        y = self.norm(x).transpose(1, 2)
        y = self.net(y).transpose(1, 2)
        return (x + y) * mask.unsqueeze(-1).float()


class ClinicalResCNNEncoder(nn.Module):
    """1D residual convolutional baseline over event tokens and time features."""

    def __init__(
        self,
        num_types,
        d_model=64,
        n_layers=2,
        kernel_size=5,
        dropout=0.1,
        pooling='mean',
        time_scale_hours=24.0,
    ):
        super().__init__()
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.time_proj = nn.Sequential(nn.Linear(2, d_model), nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            ResidualTemporalConvBlock(d_model, kernel_size=kernel_size, dropout=dropout)
            for _ in range(max(int(n_layers), 1))
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.attention = build_attention_pooling(pooling, d_model)

    def forward(self, event_type, event_time, time_gap=None):
        non_pad = event_type.ne(Constants.PAD)
        if time_gap is None:
            time_gap = torch.zeros_like(event_time)
            time_gap[:, 1:] = torch.clamp(event_time[:, 1:] - event_time[:, :-1], min=0.0)
        time_features = torch.stack(
            [event_time / self.time_scale_hours, time_gap / self.time_scale_hours],
            dim=-1,
        )
        sequence_hidden = self.dropout(self.event_emb(event_type) + self.time_proj(time_features))
        sequence_hidden = sequence_hidden * non_pad.unsqueeze(-1).float()
        for block in self.blocks:
            sequence_hidden = block(sequence_hidden, non_pad)
        sequence_hidden = self.out_norm(sequence_hidden) * non_pad.unsqueeze(-1).float()
        pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        return pooled, sequence_hidden
