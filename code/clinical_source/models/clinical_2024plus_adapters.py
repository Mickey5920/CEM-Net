import math

import torch
import torch.nn as nn

import transformer.Constants as Constants
from models.clinical_pooling import build_attention_pooling, pool_sequence
from transformer.mambapy.mamba import Mamba, MambaConfig


def _make_time_gap(event_time, time_gap):
    if time_gap is not None:
        return time_gap
    computed_gap = torch.zeros_like(event_time)
    computed_gap[:, 1:] = torch.clamp(event_time[:, 1:] - event_time[:, :-1], min=0.0)
    return computed_gap


def _gap_buckets(time_gap, boundaries):
    buckets = torch.bucketize(time_gap.detach(), boundaries.to(device=time_gap.device, dtype=time_gap.dtype))
    return buckets.clamp_min(0).clamp_max(int(boundaries.numel()))


class ClinicalEHRMambaAdapter(nn.Module):
    """EHRMamba-style bidirectional selective-state adapter for Exp2/Exp4 event inputs."""

    def __init__(
        self,
        num_types,
        d_model=64,
        n_layers=2,
        dropout=0.1,
        pooling='mean',
        time_scale_hours=24.0,
        d_state=16,
        expand_factor=2,
    ):
        super().__init__()
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.time_proj = nn.Sequential(nn.Linear(3, d_model), nn.Tanh())
        self.gap_bucket_emb = nn.Embedding(8, d_model)
        self.register_buffer(
            'gap_boundaries',
            torch.tensor([0.25, 1.0, 4.0, 12.0, 24.0, 72.0, 168.0], dtype=torch.float32),
            persistent=False,
        )
        config = MambaConfig(
            d_model=d_model,
            n_layers=max(int(n_layers), 1),
            d_state=d_state,
            expand_factor=expand_factor,
        )
        self.forward_mamba = Mamba(config)
        self.backward_mamba = Mamba(config)
        self.input_norm = nn.LayerNorm(d_model)
        self.fuse_gate = nn.Linear(d_model * 3, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention = build_attention_pooling(pooling, d_model)

    def forward(self, event_type, event_time, time_gap=None):
        non_pad = event_type.ne(Constants.PAD)
        time_gap = _make_time_gap(event_time, time_gap)
        scaled_time = event_time / self.time_scale_hours
        scaled_gap = time_gap / self.time_scale_hours
        log_gap = torch.log1p(time_gap.clamp_min(0.0)) / math.log1p(max(self.time_scale_hours, 1.0))
        time_features = torch.stack([scaled_time, scaled_gap, log_gap], dim=-1)
        bucket_features = self.gap_bucket_emb(_gap_buckets(time_gap, self.gap_boundaries))
        x = self.event_emb(event_type) + self.time_proj(time_features) + bucket_features
        x = self.dropout(self.input_norm(x)) * non_pad.unsqueeze(-1).float()

        forward_hidden = self.forward_mamba(x)
        reverse_hidden = self.backward_mamba(torch.flip(x, dims=[1]))
        reverse_hidden = torch.flip(reverse_hidden, dims=[1])
        gate = torch.sigmoid(self.fuse_gate(torch.cat([forward_hidden, reverse_hidden, x], dim=-1)))
        sequence_hidden = self.out_norm(gate * forward_hidden + (1.0 - gate) * reverse_hidden + x)
        sequence_hidden = sequence_hidden * non_pad.unsqueeze(-1).float()
        pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        return pooled, sequence_hidden


class ClinicalCOREBEHRTAdapter(nn.Module):
    """CORE-BEHRT-style bidirectional clinical Transformer adapter with CLS pooling."""

    def __init__(
        self,
        num_types,
        d_model=64,
        d_inner=256,
        n_layers=2,
        n_head=4,
        dropout=0.1,
        pooling='last',
        time_scale_hours=24.0,
    ):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f'd_model={d_model} must be divisible by n_head={n_head}')
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.time_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.gap_bucket_emb = nn.Embedding(8, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer(
            'gap_boundaries',
            torch.tensor([0.25, 1.0, 4.0, 12.0, 24.0, 72.0, 168.0], dtype=torch.float32),
            persistent=False,
        )
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
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention = build_attention_pooling(pooling, d_model)

    def forward(self, event_type, event_time, time_gap=None):
        non_pad = event_type.ne(Constants.PAD)
        time_gap = _make_time_gap(event_time, time_gap)
        scaled_time = event_time / self.time_scale_hours
        scaled_gap = time_gap / self.time_scale_hours
        log_gap = torch.log1p(time_gap.clamp_min(0.0)) / math.log1p(max(self.time_scale_hours, 1.0))
        time_features = torch.stack([scaled_time, scaled_gap, log_gap], dim=-1)
        x = self.event_emb(event_type)
        x = x + self.time_proj(time_features) + self.gap_bucket_emb(_gap_buckets(time_gap, self.gap_boundaries))
        x = self.dropout(self.input_norm(x)) * non_pad.unsqueeze(-1).float()

        cls = self.cls_token.expand(event_type.size(0), 1, -1)
        encoded_input = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones((event_type.size(0), 1), dtype=torch.bool, device=event_type.device)
        encoded_mask = torch.cat([cls_mask, non_pad], dim=1)
        encoded = self.encoder(encoded_input, src_key_padding_mask=~encoded_mask)
        cls_hidden = encoded[:, 0, :]
        sequence_hidden = encoded[:, 1:, :] * non_pad.unsqueeze(-1).float()
        if self.pooling == 'last':
            pooled = cls_hidden
        else:
            pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        return pooled, sequence_hidden


class ClinicalHyMaTEAdapter(nn.Module):
    """HyMaTE-style hybrid Mamba/Transformer event-time encoder for clinical labels."""

    def __init__(
        self,
        num_types,
        d_model=64,
        d_inner=256,
        n_layers=2,
        n_head=4,
        dropout=0.1,
        pooling='attention',
        time_scale_hours=24.0,
        d_state=16,
        expand_factor=2,
    ):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f'd_model={d_model} must be divisible by n_head={n_head}')
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.time_proj = nn.Sequential(nn.Linear(3, d_model), nn.Tanh())
        self.input_norm = nn.LayerNorm(d_model)
        config = MambaConfig(
            d_model=d_model,
            n_layers=max(int(n_layers), 1),
            d_state=d_state,
            expand_factor=expand_factor,
        )
        self.mamba = Mamba(config)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_inner,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=max(int(n_layers), 1),
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.fuse_gate = nn.Linear(d_model * 3, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention = build_attention_pooling(pooling, d_model)

    def forward(self, event_type, event_time, time_gap=None):
        non_pad = event_type.ne(Constants.PAD)
        time_gap = _make_time_gap(event_time, time_gap)
        scaled_time = event_time / self.time_scale_hours
        scaled_gap = time_gap / self.time_scale_hours
        log_gap = torch.log1p(time_gap.clamp_min(0.0)) / math.log1p(max(self.time_scale_hours, 1.0))
        time_features = torch.stack([scaled_time, scaled_gap, log_gap], dim=-1)
        base = self.event_emb(event_type) + self.time_proj(time_features)
        base = self.dropout(self.input_norm(base)) * non_pad.unsqueeze(-1).float()

        mamba_hidden = self.mamba(base)
        transformer_hidden = self.transformer(base, src_key_padding_mask=~non_pad)
        gate = torch.sigmoid(self.fuse_gate(torch.cat([mamba_hidden, transformer_hidden, base], dim=-1)))
        sequence_hidden = self.out_norm(gate * mamba_hidden + (1.0 - gate) * transformer_hidden + base)
        sequence_hidden = sequence_hidden * non_pad.unsqueeze(-1).float()
        pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        return pooled, sequence_hidden
