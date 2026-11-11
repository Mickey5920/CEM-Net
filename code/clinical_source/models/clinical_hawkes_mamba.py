import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from models.clinical_pooling import build_attention_pooling, pool_sequence
from models.clinical_static import ClinicalStaticMLPEncoder
from transformer.mambapy.mamba import Mamba, MambaConfig


_HERA_FULL_INTERVENTION = {
    'sequence_backbone': 'mamba',
    'hawkes_fusion_gate': 1.0,
    'rich_static_fusion_gate': 1.0,
    'ranking_objective_gate': 1.0,
    'transformer_ffn_dim': 192,
}

_HERA_SINGLE_FACTOR_CHANGES = {
    'full': {},
    'wo_hawkes': {'hawkes_fusion_gate': 0.0},
    'wo_rich_static': {'rich_static_fusion_gate': 0.0},
    'wo_ranking': {'ranking_objective_gate': 0.0},
    'transformer_backbone': {'sequence_backbone': 'transformer'},
    # Main TRACE-Mamba configuration centered on the temporal-trajectory and
    # landmark-conditioned clinical-state views.
    'clean_core': {
        'hawkes_fusion_gate': 0.0,
        'ranking_objective_gate': 0.0,
    },
    # Seed-41 screening ablations for the clean TRACE-Mamba core.  Each
    # variant retains weighted BCE and permanently removes decayed history.
    'clean_core_trajectory_only': {
        'hawkes_fusion_gate': 0.0,
        'rich_static_fusion_gate': 0.0,
        'ranking_objective_gate': 0.0,
    },
    'clean_core_state_only': {
        'trajectory_fusion_gate': 0.0,
        'hawkes_fusion_gate': 0.0,
        'ranking_objective_gate': 0.0,
    },
    'clean_core_transformer': {
        'sequence_backbone': 'transformer',
        'hawkes_fusion_gate': 0.0,
        'ranking_objective_gate': 0.0,
    },
    'clean_core_basic_state': {
        'hawkes_fusion_gate': 0.0,
        'ranking_objective_gate': 0.0,
    },
}

_TRACE_CLEAN_CORE_INTERVENTIONS = frozenset(
    name for name in _HERA_SINGLE_FACTOR_CHANGES if name.startswith('clean_core')
)

_TRACE_FUSION_MODES = frozenset({'fixed_add', 'normalized_residual', 'adaptive_gate'})


def resolve_hera_intervention(name):
    if name not in _HERA_SINGLE_FACTOR_CHANGES:
        raise ValueError(f'Unknown HERA intervention: {name}')
    contract = dict(_HERA_FULL_INTERVENTION)
    contract.update(_HERA_SINGLE_FACTOR_CHANGES[name])
    return contract


def is_trace_clean_core_intervention(name):
    return name in _TRACE_CLEAN_CORE_INTERVENTIONS


class ClinicalHawkesMambaEncoder(nn.Module):
    """Mamba trajectory encoder with optional history and clinical-state fusion."""

    def __init__(
        self,
        num_types,
        d_model=64,
        n_layers=2,
        dropout=0.1,
        pooling='attention',
        time_scale_hours=24.0,
        d_state=16,
        expand_factor=2,
        hawkes_context_scale=1.0,
        static_context_scale=0.0,
        static_feature_set='basic',
        static_source_dim=0,
        static_time_decay_hours=12.0,
        static_hidden_dim=None,
        initial_decay_hours=None,
        sequence_backbone='mamba',
        trajectory_fusion_gate=1.0,
        hawkes_fusion_gate=1.0,
        rich_static_fusion_gate=1.0,
        decayed_history_enabled=True,
        transformer_n_head=4,
        transformer_ffn_dim=192,
        sequence_n_layers=None,
        state_n_layers=None,
        fusion_mode='fixed_add',
        trajectory_residual_init=0.25,
    ):
        super().__init__()
        if sequence_backbone not in {'mamba', 'transformer'}:
            raise ValueError(f'Unsupported sequence_backbone={sequence_backbone}')
        if sequence_backbone == 'transformer' and int(d_model) % int(transformer_n_head) != 0:
            raise ValueError('d_model must be divisible by transformer_n_head')
        if fusion_mode not in _TRACE_FUSION_MODES:
            raise ValueError(
                f'Unsupported fusion_mode={fusion_mode!r}; expected one of {sorted(_TRACE_FUSION_MODES)}'
            )
        sequence_n_layers = int(n_layers if sequence_n_layers is None else sequence_n_layers)
        state_n_layers = int(n_layers if state_n_layers is None else state_n_layers)
        if sequence_n_layers <= 0:
            raise ValueError('sequence_n_layers must be positive')
        if state_n_layers <= 0:
            raise ValueError('state_n_layers must be positive')
        if not 0.0 < float(trajectory_residual_init) < 1.0:
            raise ValueError('trajectory_residual_init must be strictly between 0 and 1')
        self.num_types = int(num_types)
        self.d_model = int(d_model)
        self.sequence_n_layers = sequence_n_layers
        self.state_n_layers = state_n_layers
        self.pooling = pooling
        self.time_scale_hours = float(time_scale_hours)
        self.hawkes_context_scale = float(hawkes_context_scale)
        self.static_context_scale = float(static_context_scale)
        self.sequence_backbone = sequence_backbone
        self.trajectory_fusion_gate = float(trajectory_fusion_gate)
        self.hawkes_fusion_gate = float(hawkes_fusion_gate)
        self.rich_static_fusion_gate = float(rich_static_fusion_gate)
        self.decayed_history_enabled = bool(decayed_history_enabled)
        self.transformer_n_head = int(transformer_n_head)
        self.transformer_ffn_dim = int(transformer_ffn_dim)
        self.fusion_mode = fusion_mode
        self.trajectory_residual_init = float(trajectory_residual_init)
        self.static_feature_set = static_feature_set
        self.static_source_dim = int(static_source_dim)
        self.static_time_decay_hours = float(static_time_decay_hours)
        self.static_hidden_dim = None if static_hidden_dim is None else int(static_hidden_dim)
        if self.trajectory_fusion_gate == 0.0 and (
            self.static_context_scale <= 0.0 or self.rich_static_fusion_gate == 0.0
        ):
            raise ValueError('state-only configuration requires an enabled static branch')
        self.uses_source_features = self.static_context_scale > 0 and self.static_feature_set == 'rich'
        self.event_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
        self.time_proj = nn.Sequential(nn.Linear(2, d_model), nn.Tanh())
        if self.decayed_history_enabled:
            self.hawkes_source_emb = nn.Embedding(int(num_types) + 1, d_model, padding_idx=Constants.PAD)
            self.hawkes_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Tanh())
        else:
            self.hawkes_source_emb = None
            self.hawkes_proj = None
        if self.static_context_scale > 0:
            self.static_encoder = ClinicalStaticMLPEncoder(
                num_types=num_types,
                d_model=d_model,
                hidden_dim=self.static_hidden_dim or d_model,
                n_layers=max(self.state_n_layers, 2),
                dropout=dropout,
                residual=True,
                time_scale_hours=time_scale_hours,
                feature_set=static_feature_set,
                source_dim=static_source_dim,
                time_decay_hours=static_time_decay_hours,
            )
            self.fusion_norm = nn.LayerNorm(d_model)
            if self.fusion_mode == 'fixed_add':
                self.trajectory_view_norm = None
                self.state_view_norm = None
                self.trajectory_residual_logit = None
                self.adaptive_fusion_gate = None
            else:
                self.state_view_norm = nn.LayerNorm(d_model)
                if self.trajectory_fusion_gate != 0.0:
                    self.trajectory_view_norm = nn.LayerNorm(d_model)
                    initial_logit = math.log(
                        self.trajectory_residual_init / (1.0 - self.trajectory_residual_init)
                    )
                    if self.fusion_mode == 'normalized_residual':
                        self.trajectory_residual_logit = nn.Parameter(torch.tensor(initial_logit))
                        self.adaptive_fusion_gate = None
                    else:
                        self.trajectory_residual_logit = None
                        self.adaptive_fusion_gate = nn.Linear(2 * d_model, d_model)
                        nn.init.zeros_(self.adaptive_fusion_gate.weight)
                        nn.init.constant_(self.adaptive_fusion_gate.bias, initial_logit)
                else:
                    self.trajectory_view_norm = None
                    self.trajectory_residual_logit = None
                    self.adaptive_fusion_gate = None
        else:
            self.static_encoder = None
            self.fusion_norm = None
            self.trajectory_view_norm = None
            self.state_view_norm = None
            self.trajectory_residual_logit = None
            self.adaptive_fusion_gate = None
        if self.decayed_history_enabled:
            if initial_decay_hours is None:
                initial_decay_hours = max(self.time_scale_hours / 4.0, 1.0)
            inverse_softplus = math.log(math.expm1(float(initial_decay_hours)))
            self.log_decay_hours = nn.Parameter(torch.full((d_model,), inverse_softplus))
        else:
            self.register_parameter('log_decay_hours', None)
        if self.sequence_backbone == 'mamba':
            config = MambaConfig(
                d_model=d_model,
                n_layers=self.sequence_n_layers,
                d_state=d_state,
                expand_factor=expand_factor,
            )
            self.encoder = Mamba(config)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=self.transformer_n_head,
                dim_feedforward=self.transformer_ffn_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=self.sequence_n_layers,
                norm=None,
                enable_nested_tensor=False,
            )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention = build_attention_pooling(pooling, d_model)
        if self.trajectory_fusion_gate == 0.0:
            temporal_modules = (
                self.event_emb,
                self.time_proj,
                self.hawkes_source_emb,
                self.hawkes_proj,
                self.encoder,
                self.norm,
                self.attention,
            )
            for module in temporal_modules:
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
            if self.log_decay_hours is not None:
                self.log_decay_hours.requires_grad_(False)

    def fuse_views(self, trajectory_pooled, static_pooled):
        """Fuse trajectory and state views while preserving the historical fixed-add path."""
        static_scale = self.rich_static_fusion_gate * self.static_context_scale
        if self.fusion_mode == 'fixed_add':
            if trajectory_pooled is None:
                combined = static_scale * static_pooled
            else:
                combined = (
                    self.trajectory_fusion_gate * trajectory_pooled
                    + static_scale * static_pooled
                )
            return self.fusion_norm(combined)

        state_view = self.state_view_norm(static_pooled)
        state_term = static_scale * state_view
        if trajectory_pooled is None or self.trajectory_fusion_gate == 0.0:
            return self.fusion_norm(state_term)

        trajectory_view = self.trajectory_view_norm(trajectory_pooled)
        if self.fusion_mode == 'normalized_residual':
            residual_weight = torch.sigmoid(self.trajectory_residual_logit)
        else:
            residual_weight = torch.sigmoid(
                self.adaptive_fusion_gate(torch.cat([state_view, trajectory_view], dim=-1))
            )
        combined = state_term + self.trajectory_fusion_gate * residual_weight * trajectory_view
        return self.fusion_norm(combined)

    def hawkes_context(self, event_type, event_time):
        if not self.decayed_history_enabled:
            raise RuntimeError('Decayed history is disabled for this encoder configuration')
        batch_size, seq_len = event_type.shape
        non_pad = event_type.ne(Constants.PAD)
        tau = F.softplus(self.log_decay_hours).clamp_min(1e-3).view(1, -1)
        context = event_time.new_zeros((batch_size, self.d_model))
        previous_time = event_time.new_zeros((batch_size,))
        outputs = []
        source = self.hawkes_source_emb(event_type)
        for index in range(seq_len):
            current_time = event_time[:, index]
            valid = non_pad[:, index]
            delta = torch.clamp(current_time - previous_time, min=0.0).unsqueeze(-1)
            context = context * torch.exp(-delta / tau)
            outputs.append(context)
            update = source[:, index, :] * valid.unsqueeze(-1).float()
            context = context + update
            previous_time = torch.where(valid, current_time, previous_time)
        return torch.stack(outputs, dim=1)

    def forward(self, event_type, event_time, time_gap=None, source_features=None):
        non_pad = event_type.ne(Constants.PAD)
        if self.trajectory_fusion_gate == 0.0:
            static_pooled, _ = self.static_encoder(
                event_type,
                event_time,
                source_features=source_features,
            )
            pooled = self.fuse_views(None, static_pooled)
            sequence_hidden = static_pooled.new_zeros(
                (event_type.size(0), event_type.size(1), self.d_model)
            )
            return pooled, sequence_hidden

        if time_gap is None:
            time_gap = torch.zeros_like(event_time)
            time_gap[:, 1:] = torch.clamp(event_time[:, 1:] - event_time[:, :-1], min=0.0)
        time_features = torch.stack(
            [event_time / self.time_scale_hours, time_gap / self.time_scale_hours],
            dim=-1,
        )
        x = self.event_emb(event_type) + self.time_proj(time_features)
        if self.decayed_history_enabled:
            hawkes_features = self.hawkes_proj(self.hawkes_context(event_type, event_time))
            x = x + self.hawkes_fusion_gate * self.hawkes_context_scale * hawkes_features
        x = self.dropout(x)
        if self.sequence_backbone == 'transformer':
            encoded = self.encoder(x, src_key_padding_mask=~non_pad)
        else:
            encoded = self.encoder(x)
        sequence_hidden = self.norm(encoded)
        sequence_hidden = sequence_hidden * non_pad.unsqueeze(-1).float()
        pooled = pool_sequence(sequence_hidden, non_pad, pooling=self.pooling, attention=self.attention)
        if self.static_encoder is not None:
            static_pooled, _ = self.static_encoder(event_type, event_time, source_features=source_features)
            pooled = self.fuse_views(pooled, static_pooled)
        return pooled, sequence_hidden
