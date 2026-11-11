"""Versioned representation controls for the AAAI paper revision.

This module intentionally leaves the archived V1 control implementation
untouched.  Each V2 intervention changes one *logical* contract field, while
``resolved_trace_v2_contract`` records the concrete runtime configuration.
"""

from __future__ import annotations

import math
from typing import Any

import torch

import transformer.Constants as Constants
from models.clinical_hawkes_mamba import resolve_hera_intervention
from models.clinical_pooling import pool_sequence
from models.trace_representation_controls import TraceRepresentationControlledEncoder


PARAMETER_MATCHED_TRANSFORMER_FFN_DIM = 158

PERMUTATION_SEEDS = {
    "trajectory_permuted_order_p0": 1009,
    "trajectory_permuted_order_p1": 2027,
    "trajectory_permuted_order_p2": 4099,
}

_TRACE_REPRESENTATION_V2_CHANGES: dict[str, dict[str, object]] = {
    "trajectory_capacity_matched": {
        "static_feature_group_mask": "trajectory_tiled"
    },
    "bias_only": {"static_feature_group_mask": "bias_only"},
    **{
        name: {"trajectory_order_mode": f"permute_identity_seed_{seed}"}
        for name, seed in PERMUTATION_SEEDS.items()
    },
    "trajectory_no_absolute_time": {
        "trajectory_time_mode": "zero_absolute_feature"
    },
    "trajectory_no_gap": {"trajectory_time_mode": "zero_gap_feature"},
    # This is a logical single-field replacement.  The wrapper resolves it to
    # PyTorch Transformer + FFN=158 and records both resolved values in artifacts.
    "transformer_param_matched": {
        "sequence_backbone": "transformer_param_matched"
    },
}

TRACE_REPRESENTATION_V2_INTERVENTIONS = frozenset(
    _TRACE_REPRESENTATION_V2_CHANGES
)
# Queue-facing compatibility names used by the versioned claim-control runner.
TRACE_CLAIM_INTERVENTIONS = TRACE_REPRESENTATION_V2_INTERVENTIONS
TRACE_CLAIM_INTERVENTION_CHANGES = {
    name: dict(change) for name, change in _TRACE_REPRESENTATION_V2_CHANGES.items()
}

_PERMUTATION_MODE_TO_SEED = {
    f"permute_identity_seed_{seed}": seed for seed in PERMUTATION_SEEDS.values()
}


def is_trace_representation_v2_intervention(name: str | None) -> bool:
    return name in TRACE_REPRESENTATION_V2_INTERVENTIONS


def is_trace_claim_intervention(name: str | None) -> bool:
    return is_trace_representation_v2_intervention(name)


def resolve_trace_v2_intervention(name: str) -> dict[str, object]:
    if name not in TRACE_REPRESENTATION_V2_INTERVENTIONS:
        raise ValueError(f"Unknown TRACE V2 intervention: {name}")
    contract = dict(resolve_hera_intervention("full"))
    contract.update(_TRACE_REPRESENTATION_V2_CHANGES[name])
    return contract


def resolve_trace_claim_intervention(name: str) -> dict[str, object]:
    return resolve_trace_v2_intervention(name)


def resolved_trace_v2_contract(name: str) -> dict[str, Any]:
    """Return concrete runtime settings in addition to the logical contract."""

    logical = resolve_trace_v2_intervention(name)
    logical_backbone = str(logical.get("sequence_backbone", "mamba"))
    logical_order = str(logical.get("trajectory_order_mode", "original"))
    logical_time = str(logical.get("trajectory_time_mode", "original"))
    logical_static = str(logical.get("static_feature_group_mask", "none"))
    bias_only = logical_static == "bias_only"
    trajectory_tiled = logical_static == "trajectory_tiled"

    return {
        "control_schema_version": 2,
        "resolved_sequence_backbone": (
            "transformer"
            if logical_backbone == "transformer_param_matched"
            else logical_backbone
        ),
        "resolved_transformer_ffn_dim": (
            PARAMETER_MATCHED_TRANSFORMER_FFN_DIM
            if logical_backbone == "transformer_param_matched"
            else int(logical.get("transformer_ffn_dim", 192))
        ),
        "resolved_trajectory_permutation_seed": _PERMUTATION_MODE_TO_SEED.get(
            logical_order
        ),
        "resolved_trajectory_order_mode": (
            "permuted_identity"
            if logical_order in _PERMUTATION_MODE_TO_SEED
            else logical_order
        ),
        "resolved_permutation_key": (
            "vectorized_modular_hash(control_seed,valid_time_grid,positions)"
            if logical_order in _PERMUTATION_MODE_TO_SEED
            else None
        ),
        "resolved_trajectory_time_mode": logical_time,
        "resolved_absolute_time_projection_enabled": (
            not bias_only and logical_time != "zero_absolute_feature"
        ),
        "resolved_gap_time_projection_enabled": (
            not bias_only and logical_time != "zero_gap_feature"
        ),
        "resolved_hawkes_clock_mode": (
            "unused" if bias_only else "original_event_time"
        ),
        "resolved_state_time_features_mode": (
            "unused"
            if bias_only or trajectory_tiled
            else "original_event_time_derived_recency_and_intensity"
        ),
        "resolved_static_input_mode": (
            "trajectory_tiled"
            if trajectory_tiled
            else ("unused" if bias_only else "record_state")
        ),
        "resolved_representation_mode": (
            "bias_only" if bias_only else "predictive"
        ),
        "resolved_trajectory_information_enabled": not bias_only,
        "resolved_record_state_information_enabled": not (
            bias_only or trajectory_tiled
        ),
        "resolved_state_network_enabled": not bias_only,
        "resolved_hawkes_history_enabled": not bias_only,
        "resolved_ranking_signal_enabled": not bias_only,
        "resolved_encoder_frozen": bias_only,
        "resolved_trainable_scope": (
            "prediction_head_bias_only" if bias_only else "encoder_and_prediction_head"
        ),
    }


class TraceRepresentationControlledEncoderV2(TraceRepresentationControlledEncoder):
    """TRACE encoder with capacity, permutation, timing, and bias controls.

    The default constructor remains bitwise identical to the archived V1
    wrapper.  New logical values are normalized before V1 construction and are
    applied only in this subclass's forward path.
    """

    def __init__(
        self,
        *args,
        trajectory_order_mode: str = "original",
        trajectory_time_mode: str = "original",
        static_feature_group_mask: str = "none",
        sequence_backbone: str = "mamba",
        transformer_ffn_dim: int = 192,
        **kwargs,
    ):
        logical_order_mode = str(trajectory_order_mode)
        logical_time_mode = str(trajectory_time_mode)
        logical_static_mode = str(static_feature_group_mask)
        logical_backbone = str(sequence_backbone)

        if logical_order_mode not in {"original", "reverse_identity"} | set(
            _PERMUTATION_MODE_TO_SEED
        ):
            raise ValueError(
                f"Unsupported V2 trajectory_order_mode={logical_order_mode!r}"
            )
        if logical_time_mode not in {
            "original",
            "zero",
            "zero_absolute_feature",
            "zero_gap_feature",
        }:
            raise ValueError(
                f"Unsupported V2 trajectory_time_mode={logical_time_mode!r}"
            )
        if logical_static_mode not in {
            "none",
            "burden_coverage",
            "recency_activity",
            "care_source",
            "observation_intensity",
            "trajectory_tiled",
            "bias_only",
        }:
            raise ValueError(
                f"Unsupported V2 static_feature_group_mask={logical_static_mode!r}"
            )
        if logical_backbone not in {
            "mamba",
            "transformer",
            "transformer_param_matched",
        }:
            raise ValueError(f"Unsupported V2 sequence_backbone={logical_backbone!r}")

        resolved_backbone = logical_backbone
        resolved_ffn_dim = int(transformer_ffn_dim)
        if logical_backbone == "transformer_param_matched":
            resolved_backbone = "transformer"
            resolved_ffn_dim = PARAMETER_MATCHED_TRANSFORMER_FFN_DIM

        base_order_mode = (
            "original"
            if logical_order_mode in _PERMUTATION_MODE_TO_SEED
            else logical_order_mode
        )
        base_time_mode = (
            "original"
            if logical_time_mode
            in {"zero_absolute_feature", "zero_gap_feature"}
            else logical_time_mode
        )
        base_static_mode = (
            "none"
            if logical_static_mode in {"trajectory_tiled", "bias_only"}
            else logical_static_mode
        )

        super().__init__(
            *args,
            trajectory_order_mode=base_order_mode,
            trajectory_time_mode=base_time_mode,
            static_feature_group_mask=base_static_mode,
            sequence_backbone=resolved_backbone,
            transformer_ffn_dim=resolved_ffn_dim,
            **kwargs,
        )

        self.logical_sequence_backbone = logical_backbone
        self.logical_trajectory_order_mode = logical_order_mode
        self.logical_trajectory_time_mode = logical_time_mode
        self.logical_static_feature_group_mask = logical_static_mode
        self.resolved_sequence_backbone = resolved_backbone
        self.resolved_transformer_ffn_dim = resolved_ffn_dim
        self.trajectory_permutation_seed = _PERMUTATION_MODE_TO_SEED.get(
            logical_order_mode
        )
        self.static_input_mode = (
            "trajectory_tiled"
            if logical_static_mode == "trajectory_tiled"
            else ("unused" if logical_static_mode == "bias_only" else "record_state")
        )
        self.representation_mode = (
            "bias_only" if logical_static_mode == "bias_only" else "predictive"
        )

        if self.static_input_mode == "trajectory_tiled" and self.static_encoder is None:
            raise ValueError(
                "trajectory_capacity_matched requires the Full state-network architecture"
            )
        if self.representation_mode == "bias_only":
            # The prediction head lives outside this encoder. Returning an all-zero
            # representation makes its intercept the only input-dependent-free term.
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _deterministic_permutation_indices(
        event_type: torch.Tensor,
        non_pad: torch.Tensor,
        seed: int,
        event_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build sample-specific within-prefix permutations entirely in torch.

        The modular hash is a deterministic function of the control seed, the
        valid time grid, and positions. Event identities are deliberately absent
        from the keys, so their original pairing with time cannot leak into the
        permutation. It consumes no RNG stream and performs no device-to-host
        synchronization. Padding is sorted after the valid prefix; accidental
        identity permutations are replaced by a one-position cyclic shift for
        rows of length greater than one.
        """

        if event_type.ndim != 2 or non_pad.shape != event_type.shape:
            raise ValueError("event_type and non_pad must have identical [B,L] shapes")
        batch_size, sequence_length = event_type.shape
        device = event_type.device
        valid = non_pad.to(dtype=torch.bool)
        positions_zero = torch.arange(
            sequence_length, dtype=torch.int64, device=device
        ).view(1, -1)
        positions = positions_zero + 1
        if event_time is None:
            quantized_time = torch.zeros_like(event_type, dtype=torch.int64)
        else:
            quantized_time = torch.round(
                event_time.to(dtype=torch.float64) * 1_000_000.0
            ).to(dtype=torch.int64)

        prime = 2_147_483_647
        row_terms = (
            quantized_time * 97
            + positions * 10_007
        ) * valid.to(dtype=torch.int64)
        row_checksum = torch.remainder(row_terms.sum(dim=1), prime)
        row_multiplier = (
            torch.remainder(
                row_checksum * 1_009 + int(seed) * 9_176 + 1,
                prime - 1,
            )
            + 1
        )
        mixed = torch.remainder(
            row_multiplier.view(batch_size, 1) * positions
            + int(seed) * 65_537,
            prime,
        )
        keys = torch.remainder(
            mixed.square() + mixed * 3 + positions * 104_729,
            prime,
        )
        # Multiplication by L+1 leaves room for a deterministic position tie-break.
        valid_sort_keys = keys * (sequence_length + 1) + positions_zero
        padding_sort_keys = (
            (prime + 1) * (sequence_length + 1) + positions_zero
        )
        sort_keys = torch.where(valid, valid_sort_keys, padding_sort_keys)
        permutation = torch.argsort(sort_keys, dim=1)

        lengths = valid.sum(dim=1, dtype=torch.int64)
        valid_output = positions_zero < lengths.view(-1, 1)
        identity = torch.all(
            torch.where(valid_output, permutation.eq(positions_zero), True), dim=1
        ) & lengths.gt(1)
        safe_lengths = lengths.clamp_min(1).view(-1, 1)
        rolled = torch.where(
            valid_output,
            torch.remainder(positions_zero + 1, safe_lengths),
            positions_zero,
        )
        permutation = torch.where(identity.view(-1, 1), rolled, permutation)
        return permutation

    @classmethod
    def _permute_valid_identities(
        cls,
        event_type: torch.Tensor,
        non_pad: torch.Tensor,
        seed: int,
        event_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        permutation = cls._deterministic_permutation_indices(
            event_type, non_pad, seed, event_time=event_time
        )
        return event_type.gather(1, permutation)

    @staticmethod
    def _tile_trajectory_for_state_net(
        trajectory_pooled: torch.Tensor, input_dim: int
    ) -> torch.Tensor:
        repeats = int(math.ceil(float(input_dim) / float(trajectory_pooled.size(1))))
        return trajectory_pooled.repeat(1, repeats)[:, :input_dim]

    def _requires_v2_forward(self) -> bool:
        return bool(
            self.trajectory_permutation_seed is not None
            or self.logical_trajectory_time_mode
            in {"zero_absolute_feature", "zero_gap_feature"}
            or self.static_input_mode == "trajectory_tiled"
            or self.representation_mode == "bias_only"
        )

    def forward(self, event_type, event_time, time_gap=None, source_features=None):
        if not self._requires_v2_forward():
            return super().forward(
                event_type,
                event_time,
                time_gap=time_gap,
                source_features=source_features,
            )

        if self.representation_mode == "bias_only":
            pooled = event_time.new_zeros((event_type.size(0), self.d_model))
            sequence_hidden = event_time.new_zeros(
                (event_type.size(0), event_type.size(1), self.d_model)
            )
            return pooled, sequence_hidden

        original_event_type = event_type
        original_event_time = event_time
        non_pad = original_event_type.ne(Constants.PAD)

        trajectory_event_type = original_event_type
        if self.trajectory_permutation_seed is not None:
            trajectory_event_type = self._permute_valid_identities(
                original_event_type,
                non_pad,
                self.trajectory_permutation_seed,
                event_time=original_event_time,
            )

        original_gap = time_gap
        if original_gap is None:
            original_gap = torch.zeros_like(original_event_time)
            original_gap[:, 1:] = torch.clamp(
                original_event_time[:, 1:] - original_event_time[:, :-1], min=0.0
            )

        absolute_feature = original_event_time
        gap_feature = original_gap
        if self.logical_trajectory_time_mode == "zero_absolute_feature":
            absolute_feature = torch.zeros_like(original_event_time)
        elif self.logical_trajectory_time_mode == "zero_gap_feature":
            gap_feature = torch.zeros_like(original_gap)

        time_features = torch.stack(
            [
                absolute_feature / self.time_scale_hours,
                gap_feature / self.time_scale_hours,
            ],
            dim=-1,
        )
        x = self.event_emb(trajectory_event_type) + self.time_proj(time_features)
        if self.decayed_history_enabled:
            # Split timing controls remove only one explicit time-projection column.
            # The pre-update history clock deliberately remains the original clock.
            history = self.hawkes_context(
                trajectory_event_type, original_event_time
            )
            x = x + (
                self.hawkes_fusion_gate
                * self.hawkes_context_scale
                * self.hawkes_proj(history)
            )
        x = self.dropout(x)
        if self.sequence_backbone == "transformer":
            encoded = self.encoder(x, src_key_padding_mask=~non_pad)
        else:
            encoded = self.encoder(x)
        sequence_hidden = self.norm(encoded) * non_pad.unsqueeze(-1).float()
        trajectory_pooled = pool_sequence(
            sequence_hidden,
            non_pad,
            pooling=self.pooling,
            attention=self.attention,
        )

        pooled = trajectory_pooled
        if self.static_encoder is not None:
            if self.static_input_mode == "trajectory_tiled":
                capacity_input = self._tile_trajectory_for_state_net(
                    trajectory_pooled, self.static_encoder.input_dim
                )
                static_pooled = self.static_encoder.net(capacity_input)
            else:
                static_pooled, _ = self.static_encoder(
                    original_event_type,
                    original_event_time,
                    source_features=source_features,
                )
            pooled = self.fuse_views(trajectory_pooled, static_pooled)
        return pooled, sequence_hidden
