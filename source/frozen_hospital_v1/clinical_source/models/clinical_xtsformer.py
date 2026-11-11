import math

import torch
import torch.nn as nn

import transformer.Constants as Constants
from models.clinical_pooling import build_attention_pooling, pool_sequence


class ContinuousTimeEncoding(nn.Module):
    """Sinusoidal continuous-time encoding used by the XTSFormer adapter."""

    def __init__(self, d_model, scale=1.0):
        super().__init__()
        self.d_model = int(d_model)
        self.scale = float(scale)
        half = max(self.d_model // 2, 1)
        frequencies = torch.exp(
            torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1))
        )
        self.register_buffer('frequencies', frequencies, persistent=False)

    def forward(self, event_time):
        values = event_time.unsqueeze(-1) / max(self.scale, 1e-6) * self.frequencies
        encoded = torch.cat([torch.sin(values), torch.cos(values)], dim=-1)
        if encoded.size(-1) < self.d_model:
            encoded = torch.nn.functional.pad(encoded, (0, self.d_model - encoded.size(-1)))
        return encoded[..., : self.d_model]


class XTSFormerScaleBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_layers, n_head, dropout, scale):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f'd_model={d_model} must be divisible by n_head={n_head}')
        self.time_encoding = ContinuousTimeEncoding(d_model, scale=scale)
        self.gap_proj = nn.Sequential(nn.Linear(1, d_model), nn.Tanh())
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
            num_layers=max(int(n_layers), 1),
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def forward(self, base, event_time, time_gap, non_pad):
        x = base + self.time_encoding(event_time) + self.gap_proj(time_gap.unsqueeze(-1))
        hidden = self.encoder(x, src_key_padding_mask=~non_pad)
        return hidden * non_pad.unsqueeze(-1).float()


class ClinicalXTSFormerEncoder(nn.Module):
    """Cross-temporal-scale Transformer adapter for irregular clinical events."""

    def __init__(
        self,
        num_types,
        d_model=64,
        d_inner=256,
        n_layers=1,
        n_head=4,
        dropout=0.1,
        pooling='mean',
        time_scale_hours=24.0,
        scales=(1.0, 4.0, 24.0),
    ):
        super().__init__()
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            XTSFormerScaleBlock(
                d_model=d_model,
                d_inner=d_inner,
                n_layers=n_layers,
                n_head=n_head,
                dropout=dropout,
                scale=float(scale),
            )
            for scale in scales
        )
        self.scale_score = nn.Linear(d_model, 1)
        self.out_norm = nn.LayerNorm(d_model)
        self.attention = build_attention_pooling(pooling, d_model)

    def forward(self, event_type, event_time, time_gap=None):
        non_pad = event_type.ne(Constants.PAD)
        if time_gap is None:
            time_gap = torch.zeros_like(event_time)
            time_gap[:, 1:] = torch.clamp(event_time[:, 1:] - event_time[:, :-1], min=0.0)
        scaled_time = event_time / self.time_scale_hours
        scaled_gap = time_gap / self.time_scale_hours
        base = self.dropout(self.event_emb(event_type))
        scale_hidden = [
            block(base, scaled_time, scaled_gap, non_pad)
            for block in self.blocks
        ]
        scale_stack = torch.stack(scale_hidden, dim=2)
        scale_logits = self.scale_score(scale_stack).squeeze(-1)
        scale_weights = torch.softmax(scale_logits, dim=2).unsqueeze(-1)
        sequence_hidden = (scale_stack * scale_weights).sum(dim=2)
        sequence_hidden = self.out_norm(sequence_hidden) * non_pad.unsqueeze(-1).float()
        pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        return pooled, sequence_hidden
