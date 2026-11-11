"""Paper-revision controls that preserve the archived TRACE-Mamba Full implementation.

The canonical Full encoder remains in ``clinical_hawkes_mamba.py``. This module adds
single-factor representation controls without editing that archived implementation.
"""

from __future__ import annotations

import torch

import transformer.Constants as Constants
from models.clinical_hawkes_mamba import ClinicalHawkesMambaEncoder, resolve_hera_intervention
from models.clinical_pooling import pool_sequence
from models.clinical_static import ClinicalStaticMLPEncoder


_TRACE_REPRESENTATION_CHANGES = {
    "state_only": {"trajectory_fusion_gate": 0.0},
    "trajectory_reversed_order": {"trajectory_order_mode": "reverse_identity"},
    "trajectory_no_time": {"trajectory_time_mode": "zero"},
    "wo_state_burden": {"static_feature_group_mask": "burden_coverage"},
    "wo_state_recency": {"static_feature_group_mask": "recency_activity"},
    "wo_state_source": {"static_feature_group_mask": "care_source"},
    "wo_state_intensity": {"static_feature_group_mask": "observation_intensity"},
}

TRACE_REPRESENTATION_INTERVENTIONS = frozenset(_TRACE_REPRESENTATION_CHANGES)
STATIC_FEATURE_GROUP_MASKS = frozenset(
    {"none", "burden_coverage", "recency_activity", "care_source", "observation_intensity"}
)


def is_trace_representation_intervention(name: str | None) -> bool:
    return name in TRACE_REPRESENTATION_INTERVENTIONS


def resolve_trace_intervention(name: str) -> dict[str, object]:
    if name in TRACE_REPRESENTATION_INTERVENTIONS:
        contract = dict(resolve_hera_intervention("full"))
        contract.update(_TRACE_REPRESENTATION_CHANGES[name])
        return contract
    return resolve_hera_intervention(name)


class ControlledClinicalStaticMLPEncoder(ClinicalStaticMLPEncoder):
    """The canonical rich state encoder with one prespecified input family zeroed."""

    feature_group_mask = "none"

    def build_static_features(self, event_type, event_time, source_features=None):
        features = super().build_static_features(
            event_type,
            event_time,
            source_features=source_features,
        )
        group = getattr(self, "feature_group_mask", "none")
        if group == "none":
            return features
        if self.feature_set != "rich":
            raise ValueError("Feature-group controls require the rich state feature set")
        if group not in STATIC_FEATURE_GROUP_MASKS:
            raise ValueError(f"Unknown static feature-group mask: {group}")

        n_types = self.num_types
        source_start = 4 * n_types
        summary_start = source_start + self.source_dim
        slices = {
            "burden_coverage": slice(0, 2 * n_types),
            "recency_activity": slice(2 * n_types, 4 * n_types),
            "care_source": slice(source_start, summary_start),
            "observation_intensity": slice(summary_start, summary_start + 4),
        }
        masked = features.clone()
        masked[:, slices[group]] = 0.0
        return masked


class TraceRepresentationControlledEncoder(ClinicalHawkesMambaEncoder):
    """TRACE-Mamba with trajectory-only perturbations and state-family masks."""

    def __init__(
        self,
        *args,
        trajectory_order_mode: str = "original",
        trajectory_time_mode: str = "original",
        static_feature_group_mask: str = "none",
        **kwargs,
    ):
        if trajectory_order_mode not in {"original", "reverse_identity"}:
            raise ValueError(f"Unsupported trajectory_order_mode={trajectory_order_mode!r}")
        if trajectory_time_mode not in {"original", "zero"}:
            raise ValueError(f"Unsupported trajectory_time_mode={trajectory_time_mode!r}")
        if static_feature_group_mask not in STATIC_FEATURE_GROUP_MASKS:
            raise ValueError(
                f"Unsupported static_feature_group_mask={static_feature_group_mask!r}"
            )
        super().__init__(*args, **kwargs)
        self.trajectory_order_mode = trajectory_order_mode
        self.trajectory_time_mode = trajectory_time_mode
        self.static_feature_group_mask = static_feature_group_mask
        if static_feature_group_mask != "none":
            if self.static_encoder is None:
                raise ValueError("State feature-group controls require an enabled state branch")
            # Reuse the already initialized canonical network without consuming a second
            # RNG stream or changing parameter names/state-dict structure.
            self.static_encoder.__class__ = ControlledClinicalStaticMLPEncoder
            self.static_encoder.feature_group_mask = static_feature_group_mask

    @staticmethod
    def _reverse_valid_identities(event_type: torch.Tensor, non_pad: torch.Tensor) -> torch.Tensor:
        reversed_type = event_type.clone()
        lengths = non_pad.sum(dim=1).tolist()
        for row, length in enumerate(lengths):
            length = int(length)
            if length > 1:
                reversed_type[row, :length] = event_type[row, :length].flip(0)
        return reversed_type

    def forward(self, event_type, event_time, time_gap=None, source_features=None):
        # The state-only control is already implemented by the canonical encoder and
        # intentionally receives the original prefix.
        if self.trajectory_fusion_gate == 0.0:
            return super().forward(
                event_type,
                event_time,
                time_gap=time_gap,
                source_features=source_features,
            )

        original_event_type = event_type
        original_event_time = event_time
        non_pad = original_event_type.ne(Constants.PAD)

        trajectory_event_type = original_event_type
        if self.trajectory_order_mode == "reverse_identity":
            trajectory_event_type = self._reverse_valid_identities(original_event_type, non_pad)

        trajectory_event_time = original_event_time
        trajectory_time_gap = time_gap
        if self.trajectory_time_mode == "zero":
            trajectory_event_time = torch.zeros_like(original_event_time)
            trajectory_time_gap = torch.zeros_like(original_event_time)
        elif trajectory_time_gap is None:
            trajectory_time_gap = torch.zeros_like(trajectory_event_time)
            trajectory_time_gap[:, 1:] = torch.clamp(
                trajectory_event_time[:, 1:] - trajectory_event_time[:, :-1], min=0.0
            )

        time_features = torch.stack(
            [
                trajectory_event_time / self.time_scale_hours,
                trajectory_time_gap / self.time_scale_hours,
            ],
            dim=-1,
        )
        x = self.event_emb(trajectory_event_type) + self.time_proj(time_features)
        if self.decayed_history_enabled:
            history = self.hawkes_context(trajectory_event_type, trajectory_event_time)
            x = x + self.hawkes_fusion_gate * self.hawkes_context_scale * self.hawkes_proj(history)
        x = self.dropout(x)
        if self.sequence_backbone == "transformer":
            encoded = self.encoder(x, src_key_padding_mask=~non_pad)
        else:
            encoded = self.encoder(x)
        sequence_hidden = self.norm(encoded) * non_pad.unsqueeze(-1).float()
        pooled = pool_sequence(
            sequence_hidden,
            non_pad,
            pooling=self.pooling,
            attention=self.attention,
        )
        if self.static_encoder is not None:
            # Crucially, state is always computed from the unperturbed prefix.
            static_pooled, _ = self.static_encoder(
                original_event_type,
                original_event_time,
                source_features=source_features,
            )
            pooled = self.fuse_views(pooled, static_pooled)
        return pooled, sequence_hidden

