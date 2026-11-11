import torch
import torch.nn as nn

import transformer.Constants as Constants
from models.clinical_pooling import build_attention_pooling, pool_sequence


class ClinicalTransformerEncoder(nn.Module):
    """Bidirectional Transformer encoder over the fully observed landmark window."""

    def __init__(
        self,
        num_types,
        d_model=64,
        d_inner=256,
        n_layers=2,
        n_head=4,
        dropout=0.1,
        pooling='mean',
        time_scale_hours=24.0,
    ):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f'd_model={d_model} must be divisible by n_head={n_head}')
        self.d_model = int(d_model)
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.time_proj = nn.Sequential(nn.Linear(2, d_model), nn.Tanh())
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_inner,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
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
        x = self.dropout(self.event_emb(event_type) + self.time_proj(time_features))
        sequence_hidden = self.encoder(x, src_key_padding_mask=~non_pad)
        sequence_hidden = sequence_hidden * non_pad.unsqueeze(-1).float()
        pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        return pooled, sequence_hidden
