"""Explicit temporal/dense branches for the WWW development controls.

Uses the historical rich aggregate feature builder without dropping information.
No sparse tokens, latent attention, or endpoint query in these simple controls.
"""
import torch
from torch import nn

from models.clinical_hawkes_mamba import ClinicalHawkesMambaEncoder
from models.clinical_static import ClinicalStaticMLPEncoder
from models.clinical_heads import ClinicalPredictionModel, BinaryMultiTaskHead


class TemporalDenseEncoder(nn.Module):
    uses_time_gap = True
    uses_source_features = True

    def __init__(self, num_types, d_model=96, n_layers=2, dropout=0.3,
                 time_scale_hours=24., source_dim=0, hidden_dim=192,
                 fusion='concat', temporal_enabled=True, dense_enabled=True):
        super().__init__()
        if fusion not in {'concat', 'additive'}:
            raise ValueError(f'Unknown fusion {fusion}')
        if not temporal_enabled and not dense_enabled:
            raise ValueError('At least one branch is required')
        self.d_model = int(d_model)
        self.source_dim = int(source_dim)
        self.fusion = fusion
        self.temporal_encoder = ClinicalHawkesMambaEncoder(
            num_types=num_types, d_model=d_model, n_layers=n_layers,
            dropout=dropout, pooling='attention', time_scale_hours=time_scale_hours,
            hawkes_context_scale=1., static_context_scale=0.,
            sequence_backbone='mamba', sequence_n_layers=n_layers,
            decayed_history_enabled=True,
        ) if temporal_enabled else None
        self.dense_encoder = ClinicalStaticMLPEncoder(
            num_types=num_types, d_model=d_model, hidden_dim=hidden_dim,
            n_layers=2, dropout=dropout, residual=True,
            time_scale_hours=time_scale_hours, feature_set='rich',
            source_dim=source_dim, time_decay_hours=12.,
        ) if dense_enabled else None
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2*d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.LayerNorm(d_model),
        ) if fusion == 'concat' else nn.Identity()

    def encode_branches(self, event_type, event_time, time_gap=None, source_features=None):
        if source_features is None or source_features.shape != (event_type.size(0), self.source_dim):
            raise ValueError('Source features must have the declared training-derived width')
        if self.temporal_encoder is None:
            temporal = event_time.new_zeros((event_type.size(0), self.d_model))
            hidden = event_time.new_zeros((*event_type.shape, self.d_model))
        else:
            temporal, hidden = self.temporal_encoder(event_type, event_time, time_gap=time_gap)
        if self.dense_encoder is None:
            dense = torch.zeros_like(temporal)
        else:
            dense, _ = self.dense_encoder(event_type, event_time, source_features=source_features)
        return temporal, dense, hidden

    def forward(self, event_type, event_time, time_gap=None, source_features=None):
        temporal, dense, hidden = self.encode_branches(
            event_type,event_time,time_gap,source_features)
        return self.fusion_mlp(torch.cat([temporal,dense],dim=-1)), hidden


class AdditiveEndpointHead(nn.Module):
    """Two independently parameterized linear readouts, with summed logits."""
    def __init__(self,d_model,num_tasks,dropout):
        super().__init__()
        self.d_model=d_model
        self.temporal_head=BinaryMultiTaskHead(d_model,num_tasks,dropout)
        self.dense_head=BinaryMultiTaskHead(d_model,num_tasks,dropout)

    def forward(self,pooled):
        temporal,dense=pooled.split(self.d_model,dim=-1)
        return self.temporal_head(temporal)+self.dense_head(dense)


def build_simple_model(args,num_types,num_tasks,fusion):
    encoder=TemporalDenseEncoder(
        num_types,d_model=args.d_model,n_layers=args.sequence_n_layers or args.n_layers,
        dropout=args.dropout,time_scale_hours=args.time_scale_hours,
        source_dim=args.static_source_dim,hidden_dim=args.trace_anchor_static_hidden_dim,
        fusion=fusion,
    )
    head=(AdditiveEndpointHead(args.d_model,num_tasks,args.dropout) if fusion=='additive'
          else BinaryMultiTaskHead(args.d_model,num_tasks,args.dropout))
    return ClinicalPredictionModel(encoder,head)
