import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from models.clinical_hawkes_mamba import ClinicalHawkesMambaEncoder


def _inverse_softplus(value):
    value = torch.as_tensor(value, dtype=torch.float32).clamp_min(1e-4)
    return torch.log(torch.expm1(value))


class CausalMultiScaleEvidenceAccumulator(nn.Module):
    """Build sparse event-type evidence tokens from an observed EHR prefix.

    The accumulator exposes burden, coverage, last-observation timing, and
    exponentially decayed activity at ordered time scales.  Event types are
    sorted only to create a deterministic sparse set; downstream set encoding
    is invariant to that storage order.
    """

    def __init__(
        self,
        num_types,
        time_scale_hours,
        num_scales=4,
        scale_mode='multiscale',
        single_scale_hours=12.0,
        feature_group_mask='none',
        max_retained_events=512,
    ):
        super().__init__()
        if int(num_types) <= 0:
            raise ValueError('num_types must be positive')
        if float(time_scale_hours) <= 0:
            raise ValueError('time_scale_hours must be positive')
        if int(num_scales) <= 0:
            raise ValueError('num_scales must be positive')
        if scale_mode not in {'multiscale', 'single'}:
            raise ValueError("scale_mode must be 'multiscale' or 'single'")
        if float(single_scale_hours) <= 0:
            raise ValueError('single_scale_hours must be positive')
        if feature_group_mask not in {
            'none',
            'burden_coverage',
            'recency_activity',
            'care_source',
            'observation_intensity',
        }:
            raise ValueError('unknown accumulated-evidence feature-group mask')

        self.num_types = int(num_types)
        self.time_scale_hours = float(time_scale_hours)
        self.num_scales = int(num_scales)
        self.scale_mode = str(scale_mode)
        self.single_scale_hours = float(single_scale_hours)
        self.feature_group_mask = str(feature_group_mask)
        self.max_retained_events = int(max_retained_events)

        minimum = max(self.time_scale_hours / 24.0, 0.5)
        targets = torch.logspace(
            math.log10(minimum),
            math.log10(self.time_scale_hours),
            steps=self.num_scales,
            dtype=torch.float32,
        )
        increments = torch.cat([targets[:1], targets[1:] - targets[:-1]])
        self.raw_tau_increments = nn.Parameter(_inverse_softplus(increments))

    @property
    def token_feature_dim(self):
        return 4 + self.num_scales

    @property
    def summary_feature_dim(self):
        return 6

    def tau_hours(self):
        if self.scale_mode == 'single':
            # Keep the same parameter tensor in the ablation while removing
            # multi-scale information and scale learning from the forward map.
            return self.raw_tau_increments * 0.0 + self.single_scale_hours
        return torch.cumsum(F.softplus(self.raw_tau_increments).clamp_min(1e-3), dim=0)

    def forward(self, event_type, event_time):
        if event_type.ndim != 2 or event_time.shape != event_type.shape:
            raise ValueError('event_type and event_time must have matching [batch, length] shapes')
        if event_type.numel() and int(event_type.max()) > self.num_types:
            raise ValueError('event_type contains an id larger than num_types')

        batch_size, seq_len = event_type.shape
        non_pad = event_type.ne(Constants.PAD)
        if torch.any(non_pad & (event_time < 0)):
            raise ValueError('observed event times must be non-negative')
        if torch.any(non_pad & (event_time > self.time_scale_hours + 1e-4)):
            raise ValueError('event_time exceeds the configured prediction-time boundary')
        valid_float = non_pad.to(event_time.dtype)
        lengths = valid_float.sum(dim=1).clamp_min(1.0)

        counts = event_time.new_zeros((batch_size, self.num_types + 1))
        counts.scatter_add_(1, event_type, valid_float)

        last_time = event_time.new_zeros((batch_size, self.num_types + 1))
        last_time.scatter_reduce_(
            1,
            event_type,
            event_time * valid_float,
            reduce='amax',
            include_self=True,
        )

        tau = self.tau_hours().to(device=event_time.device, dtype=event_time.dtype)
        age = (self.time_scale_hours - event_time).clamp_min(0.0)
        per_event_activity = torch.exp(-age.unsqueeze(-1) / tau.view(1, 1, -1))
        per_event_activity = per_event_activity * valid_float.unsqueeze(-1)
        activity = event_time.new_zeros(
            (batch_size, self.num_types + 1, self.num_scales)
        )
        activity.scatter_add_(
            1,
            event_type.unsqueeze(-1).expand(-1, -1, self.num_scales),
            per_event_activity,
        )

        sorted_types, _ = torch.sort(event_type, dim=1)
        starts_group = torch.ones_like(sorted_types, dtype=torch.bool)
        if seq_len > 1:
            starts_group[:, 1:] = sorted_types[:, 1:].ne(sorted_types[:, :-1])
        token_mask = sorted_types.ne(Constants.PAD) & starts_group

        token_counts = torch.gather(counts, 1, sorted_types)
        token_last = torch.gather(last_time, 1, sorted_types)
        activity_index = sorted_types.unsqueeze(-1).expand(-1, -1, self.num_scales)
        token_activity = torch.gather(activity, 1, activity_index)

        count_fraction = token_counts / lengths.unsqueeze(1)
        last_fraction = token_last / self.time_scale_hours
        age_fraction = (self.time_scale_hours - token_last).clamp_min(0.0) / self.time_scale_hours
        token_features = torch.cat(
            [
                torch.log1p(token_counts).unsqueeze(-1),
                count_fraction.unsqueeze(-1),
                last_fraction.unsqueeze(-1),
                age_fraction.unsqueeze(-1),
                torch.log1p(token_activity),
            ],
            dim=-1,
        )
        token_features = token_features * token_mask.unsqueeze(-1).to(token_features.dtype)

        observed_time = event_time * valid_float
        maximum_time = observed_time.max(dim=1).values
        mean_time = observed_time.sum(dim=1) / lengths
        unique_count = token_mask.sum(dim=1).to(event_time.dtype).clamp_min(1.0)
        density = lengths / maximum_time.clamp_min(1.0)
        summary = torch.stack(
            [
                lengths / float(self.max_retained_events),
                unique_count / float(self.num_types),
                unique_count / lengths,
                maximum_time / self.time_scale_hours,
                mean_time / self.time_scale_hours,
                torch.log1p(density),
            ],
            dim=-1,
        )
        if self.feature_group_mask != 'none':
            token_features = token_features.clone()
            summary = summary.clone()
            if self.feature_group_mask == 'burden_coverage':
                token_features[..., :2] = 0.0
                summary[..., 1:3] = 0.0
            elif self.feature_group_mask == 'recency_activity':
                token_features[..., 2:] = 0.0
                summary[..., 3:5] = 0.0
            elif self.feature_group_mask == 'observation_intensity':
                summary[..., 0] = 0.0
                summary[..., 5] = 0.0
        return sorted_types, token_features, token_mask, summary


class _ResidualFeedForward(nn.Module):
    def __init__(self, d_model, dropout, hidden_dim=None):
        super().__init__()
        hidden_dim = int(hidden_dim or 2 * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, value):
        return value + self.net(self.norm(value))


class EndpointConditionedRelationalEvidenceEncoder(nn.Module):
    """Encode accumulated evidence with latent set interactions and task queries."""

    def __init__(
        self,
        num_types,
        num_tasks,
        d_model=96,
        dropout=0.2,
        time_scale_hours=24.0,
        source_dim=0,
        num_scales=4,
        scale_mode='multiscale',
        single_scale_hours=12.0,
        mixer_mode='relational',
        endpoint_conditioning='task_specific',
        num_latents=16,
        n_head=4,
        ffn_dim=None,
        feature_group_mask='none',
        evidence_input_mode='observed',
    ):
        super().__init__()
        if int(num_tasks) <= 0:
            raise ValueError('num_tasks must be positive')
        if mixer_mode not in {'relational', 'deepset'}:
            raise ValueError("mixer_mode must be 'relational' or 'deepset'")
        if endpoint_conditioning not in {'task_specific', 'shared'}:
            raise ValueError("endpoint_conditioning must be 'task_specific' or 'shared'")
        if int(d_model) % int(n_head) != 0:
            raise ValueError('d_model must be divisible by n_head')
        if int(num_latents) <= 0:
            raise ValueError('num_latents must be positive')
        if evidence_input_mode not in {'observed', 'trajectory_tiled'}:
            raise ValueError("evidence_input_mode must be 'observed' or 'trajectory_tiled'")

        self.num_types = int(num_types)
        self.num_tasks = int(num_tasks)
        self.d_model = int(d_model)
        self.source_dim = int(source_dim)
        self.mixer_mode = str(mixer_mode)
        self.endpoint_conditioning = str(endpoint_conditioning)
        self.num_latents = int(num_latents)
        self.ffn_dim = int(ffn_dim or 2 * d_model)
        self.feature_group_mask = str(feature_group_mask)
        self.evidence_input_mode = str(evidence_input_mode)

        self.accumulator = CausalMultiScaleEvidenceAccumulator(
            num_types=num_types,
            time_scale_hours=time_scale_hours,
            num_scales=num_scales,
            scale_mode=scale_mode,
            single_scale_hours=single_scale_hours,
            feature_group_mask=feature_group_mask,
        )
        self.type_embedding = nn.Embedding(
            int(num_types) + 1,
            d_model,
            padding_idx=Constants.PAD,
        )
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(self.accumulator.token_feature_dim),
            nn.Linear(self.accumulator.token_feature_dim, d_model),
            nn.GELU(),
        )
        self.token_norm = nn.LayerNorm(d_model)
        self.token_ffn = _ResidualFeedForward(d_model, dropout, self.ffn_dim)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(self.accumulator.summary_feature_dim + self.source_dim),
            nn.Linear(self.accumulator.summary_feature_dim + self.source_dim, d_model),
            nn.GELU(),
        )

        self.latent_queries = nn.Parameter(torch.empty(self.num_latents, d_model))
        self.task_queries = nn.Parameter(torch.empty(self.num_tasks, d_model))
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.task_queries, mean=0.0, std=0.02)

        self.latent_cross_attention = nn.MultiheadAttention(
            d_model,
            int(n_head),
            dropout=dropout,
            batch_first=True,
        )
        self.latent_cross_norm = nn.LayerNorm(d_model)
        self.latent_ffn = _ResidualFeedForward(d_model, dropout, self.ffn_dim)
        self.latent_self_attention = nn.MultiheadAttention(
            d_model,
            int(n_head),
            dropout=dropout,
            batch_first=True,
        )
        self.latent_self_norm = nn.LayerNorm(d_model)
        self.latent_self_ffn = _ResidualFeedForward(d_model, dropout, self.ffn_dim)
        self.endpoint_attention = nn.MultiheadAttention(
            d_model,
            int(n_head),
            dropout=dropout,
            batch_first=True,
        )
        self.endpoint_norm = nn.LayerNorm(d_model)
        self.endpoint_ffn = _ResidualFeedForward(d_model, dropout, self.ffn_dim)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _tile_last_dim(value, width):
        repeats = math.ceil(int(width) / int(value.size(-1)))
        return value.repeat(1, repeats)[..., :width]

    def _prepare_tokens(
        self,
        event_type,
        event_time,
        source_features,
        replacement_vector=None,
    ):
        token_types, token_features, token_mask, summary = self.accumulator(
            event_type,
            event_time,
        )
        if self.evidence_input_mode == 'trajectory_tiled':
            if replacement_vector is None:
                raise ValueError('trajectory_tiled evidence requires a trajectory vector')
            if replacement_vector.shape != (event_type.size(0), self.d_model):
                raise ValueError('replacement trajectory must have shape [batch, d_model]')
            replacement_vector = replacement_vector.to(
                device=event_time.device,
                dtype=event_time.dtype,
            )
            token_types = torch.ones_like(event_type)
            token_mask = torch.ones_like(event_type, dtype=torch.bool)
            tiled_features = self._tile_last_dim(
                replacement_vector,
                self.accumulator.token_feature_dim,
            )
            token_features = tiled_features.unsqueeze(1).expand(
                -1,
                event_type.size(1),
                -1,
            )
        token_values = self.type_embedding(token_types) + self.feature_projection(token_features)
        token_values = self.token_ffn(self.token_norm(token_values))
        token_values = token_values * token_mask.unsqueeze(-1).to(token_values.dtype)

        if self.evidence_input_mode == 'trajectory_tiled':
            global_input = self._tile_last_dim(
                replacement_vector,
                self.accumulator.summary_feature_dim + self.source_dim,
            )
        elif self.source_dim:
            if source_features is None:
                source_features = summary.new_zeros((summary.size(0), self.source_dim))
            else:
                source_features = source_features.to(device=summary.device, dtype=summary.dtype)
                if source_features.size(1) != self.source_dim:
                    raise ValueError(
                        f'source feature width {source_features.size(1)} does not match {self.source_dim}'
                    )
            if self.feature_group_mask == 'care_source':
                source_features = torch.zeros_like(source_features)
            global_input = torch.cat([summary, source_features], dim=-1)
        else:
            global_input = summary
        global_token = self.global_projection(global_input).unsqueeze(1)
        token_values = torch.cat([global_token, token_values], dim=1)
        global_mask = torch.ones(
            (token_mask.size(0), 1),
            dtype=torch.bool,
            device=token_mask.device,
        )
        token_mask = torch.cat([global_mask, token_mask], dim=1)
        return token_values, token_mask

    def _task_query_batch(self, batch_size, task_query_context=None):
        if self.endpoint_conditioning == 'shared':
            queries = self.task_queries[:1].expand(self.num_tasks, -1)
        else:
            queries = self.task_queries
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        if task_query_context is None:
            return queries
        if task_query_context.ndim == 2:
            if task_query_context.shape != (batch_size, self.d_model):
                raise ValueError('task query context must have shape [batch, d_model]')
            task_query_context = task_query_context.unsqueeze(1)
        if task_query_context.shape != (batch_size, self.num_tasks, self.d_model):
            raise ValueError('task query context must match [batch, task, d_model]')
        return queries + task_query_context

    def forward(
        self,
        event_type,
        event_time,
        source_features=None,
        replacement_vector=None,
        task_query_context=None,
    ):
        tokens, token_mask = self._prepare_tokens(
            event_type,
            event_time,
            source_features,
            replacement_vector=replacement_vector,
        )
        mask_float = token_mask.unsqueeze(-1).to(tokens.dtype)
        shared = (tokens * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        task_queries = self._task_query_batch(
            tokens.size(0),
            task_query_context=task_query_context,
        )

        if self.mixer_mode == 'deepset':
            endpoint = shared.unsqueeze(1) + task_queries
            endpoint = self.endpoint_ffn(self.endpoint_norm(endpoint))
            return self.output_norm(endpoint)

        latent = self.latent_queries.unsqueeze(0).expand(tokens.size(0), -1, -1)
        cross, _ = self.latent_cross_attention(
            latent,
            tokens,
            tokens,
            key_padding_mask=~token_mask,
            need_weights=False,
        )
        latent = self.latent_cross_norm(latent + self.dropout(cross))
        latent = self.latent_ffn(latent)
        mixed, _ = self.latent_self_attention(latent, latent, latent, need_weights=False)
        latent = self.latent_self_norm(latent + self.dropout(mixed))
        latent = self.latent_self_ffn(latent)

        endpoint, _ = self.endpoint_attention(
            task_queries,
            latent,
            latent,
            need_weights=False,
        )
        # The shared DeepSets path is an optimization-stable residual.  The
        # relational ablation retains it and removes only latent interactions.
        endpoint = self.endpoint_norm(task_queries + self.dropout(endpoint))
        endpoint = self.endpoint_ffn(endpoint + shared.unsqueeze(1))
        return self.output_norm(endpoint)


class ClinicalTRACEHREncoder(nn.Module):
    """TRACEHR temporal/evidence encoder and guarded complete-model upgrades."""

    uses_source_features = True
    uses_time_gap = True
    task_conditioned_output = True

    UPGRADE_VARIANTS = {
        'current',
        'u1_guarded',
        'u2_anchored',
        'u3_hierarchical',
        'u4_delta',
    }

    def __init__(
        self,
        num_types,
        num_tasks,
        d_model=96,
        n_layers=2,
        dropout=0.2,
        pooling='attention',
        time_scale_hours=24.0,
        source_dim=0,
        num_scales=4,
        scale_mode='multiscale',
        single_scale_hours=12.0,
        mixer_mode='relational',
        endpoint_conditioning='task_specific',
        num_latents=16,
        n_head=4,
        ffn_dim=None,
        evidence_scale=1.25,
        temporal_mode='mamba',
        evidence_enabled=True,
        evidence_input_mode='observed',
        evidence_feature_group_mask='none',
        time_gap_mode='observed',
        upgrade_variant='current',
        anchor_static_scale=1.25,
        anchor_static_hidden_dim=None,
        hawkes_context_scale=1.0,
        initial_decay_hours=None,
        sequence_n_layers=None,
    ):
        super().__init__()
        if temporal_mode not in {'mamba', 'none'}:
            raise ValueError("temporal_mode must be 'mamba' or 'none'")
        if upgrade_variant not in self.UPGRADE_VARIANTS:
            raise ValueError(f'unknown TRACEHR upgrade variant: {upgrade_variant}')
        if not bool(evidence_enabled) and temporal_mode == 'none':
            raise ValueError('at least one TRACEHR pathway must be enabled')
        if float(evidence_scale) <= 0:
            raise ValueError('evidence_scale must be positive')
        if float(anchor_static_scale) <= 0:
            raise ValueError('anchor_static_scale must be positive')
        if evidence_input_mode not in {'observed', 'trajectory_tiled'}:
            raise ValueError('unknown TRACEHR evidence input mode')
        if time_gap_mode not in {'observed', 'zero'}:
            raise ValueError('unknown TRACEHR time-gap mode')
        if evidence_input_mode == 'trajectory_tiled' and (
            temporal_mode == 'none' or not bool(evidence_enabled)
        ):
            raise ValueError('trajectory_tiled evidence requires both TRACEHR pathways')

        self.d_model = int(d_model)
        self.output_dim = 2 * int(d_model) if upgrade_variant == 'u4_delta' else int(d_model)
        self.num_tasks = int(num_tasks)
        self.temporal_mode = str(temporal_mode)
        self.evidence_enabled = bool(evidence_enabled)
        self.evidence_scale = float(evidence_scale)
        self.evidence_input_mode = str(evidence_input_mode)
        self.evidence_feature_group_mask = str(evidence_feature_group_mask)
        self.time_gap_mode = str(time_gap_mode)
        self.upgrade_variant = str(upgrade_variant)
        self.anchor_static_scale = float(anchor_static_scale)
        self.anchor_static_hidden_dim = int(anchor_static_hidden_dim or d_model)

        anchored_variant = self.upgrade_variant in {
            'u2_anchored',
            'u3_hierarchical',
            'u4_delta',
        }
        anchor_evidence_active = (
            anchored_variant
            and self.evidence_enabled
            and self.evidence_input_mode == 'observed'
            and self.temporal_mode == 'mamba'
        )
        if self.temporal_mode == 'mamba':
            self.temporal_encoder = ClinicalHawkesMambaEncoder(
                num_types=num_types,
                d_model=d_model,
                n_layers=n_layers,
                dropout=dropout,
                pooling=pooling,
                time_scale_hours=time_scale_hours,
                hawkes_context_scale=hawkes_context_scale,
                static_context_scale=(self.anchor_static_scale if anchor_evidence_active else 0.0),
                static_feature_set=('rich' if anchor_evidence_active else 'basic'),
                static_source_dim=(int(source_dim) if anchor_evidence_active else 0),
                static_time_decay_hours=12.0,
                static_hidden_dim=(self.anchor_static_hidden_dim if anchor_evidence_active else None),
                initial_decay_hours=initial_decay_hours,
                sequence_backbone='mamba',
                trajectory_fusion_gate=1.0,
                hawkes_fusion_gate=1.0,
                rich_static_fusion_gate=(1.0 if anchor_evidence_active else 0.0),
                decayed_history_enabled=True,
                transformer_n_head=n_head,
                sequence_n_layers=sequence_n_layers,
                fusion_mode='fixed_add',
            )
        else:
            self.temporal_encoder = None

        self.evidence_encoder = EndpointConditionedRelationalEvidenceEncoder(
            num_types=num_types,
            num_tasks=num_tasks,
            d_model=d_model,
            dropout=dropout,
            time_scale_hours=time_scale_hours,
            source_dim=source_dim,
            num_scales=num_scales,
            scale_mode=scale_mode,
            single_scale_hours=single_scale_hours,
            mixer_mode=mixer_mode,
            endpoint_conditioning=endpoint_conditioning,
            num_latents=num_latents,
            n_head=n_head,
            ffn_dim=ffn_dim,
            feature_group_mask=evidence_feature_group_mask,
            evidence_input_mode=evidence_input_mode,
        )
        self.fusion_norm = (
            nn.LayerNorm(d_model)
            if self.temporal_encoder is not None and self.upgrade_variant == 'current'
            else None
        )
        if self.upgrade_variant in {'u1_guarded', 'u2_anchored', 'u3_hierarchical'}:
            self.guard_network = nn.Linear(2 * d_model, d_model)
            nn.init.zeros_(self.guard_network.weight)
            nn.init.zeros_(self.guard_network.bias)
        else:
            self.guard_network = None
        if self.upgrade_variant == 'u3_hierarchical':
            self.task_context_projection = nn.Linear(d_model, d_model, bias=False)
            nn.init.xavier_uniform_(self.task_context_projection.weight)
        else:
            self.task_context_projection = None

    def forward(self, event_type, event_time, time_gap=None, source_features=None):
        if time_gap is not None:
            if time_gap.shape != event_time.shape:
                raise ValueError('time_gap and event_time must have matching shapes')
            non_pad = event_type.ne(Constants.PAD)
            time_gap = time_gap.to(device=event_time.device, dtype=event_time.dtype)
            time_gap = torch.where(non_pad, time_gap.clamp_min(0.0), torch.zeros_like(time_gap))
            time_gap = time_gap.clone()
            time_gap[:, 0] = 0.0
            if self.time_gap_mode == 'zero':
                time_gap.zero_()

        trajectory = None
        if self.temporal_encoder is not None:
            trajectory, sequence_hidden = self.temporal_encoder(
                event_type,
                event_time,
                time_gap=time_gap,
                source_features=source_features,
            )
        else:
            sequence_hidden = event_time.new_zeros(
                (event_type.size(0), event_type.size(1), self.d_model)
            )

        evidence = None
        if self.evidence_enabled:
            task_context = None
            if self.task_context_projection is not None and trajectory is not None:
                task_context = self.task_context_projection(trajectory)
                task_context = task_context.unsqueeze(1).expand(-1, self.num_tasks, -1)
            evidence = self.evidence_encoder(
                event_type,
                event_time,
                source_features=source_features,
                replacement_vector=(
                    trajectory if self.evidence_input_mode == 'trajectory_tiled' else None
                ),
                task_query_context=task_context,
            )

        if self.upgrade_variant == 'current':
            if trajectory is None:
                pooled = evidence
            elif evidence is None:
                pooled = trajectory.unsqueeze(1).expand(-1, self.num_tasks, -1)
                pooled = self.fusion_norm(pooled)
            else:
                pooled = self.fusion_norm(
                    trajectory.unsqueeze(1) + self.evidence_scale * evidence
                )
            return pooled, sequence_hidden

        base = (
            trajectory.unsqueeze(1).expand(-1, self.num_tasks, -1)
            if trajectory is not None
            else event_time.new_zeros((event_type.size(0), self.num_tasks, self.d_model))
        )
        evidence_view = evidence if evidence is not None else torch.zeros_like(base)
        if self.upgrade_variant == 'u4_delta':
            return torch.cat([base, evidence_view], dim=-1), sequence_hidden
        if evidence is None:
            return base, sequence_hidden
        guard = torch.tanh(self.guard_network(torch.cat([base, evidence_view], dim=-1)))
        pooled = base + self.evidence_scale * guard * evidence_view
        return pooled, sequence_hidden
