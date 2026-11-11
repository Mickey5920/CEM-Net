"""Frozen-checkpoint sensitivity operations for TRACE-Mamba.

This module is intentionally separate from the paper-grade training sources.  It
does not alter the archived Full encoder or the frozen representation-control
queue.  The operations here are distribution-shift diagnostics for an already
trained model; they are not retrained component ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn

import transformer.Constants as Constants
from models.clinical_hawkes_mamba import ClinicalHawkesMambaEncoder
from models.clinical_heads import ClinicalPredictionModel
from models.clinical_pooling import pool_sequence


@dataclass(frozen=True)
class PerturbedPrefix:
    """A padded perturbed prefix and the number of retained valid events."""

    event_type: torch.Tensor
    event_time: torch.Tensor
    source_features: torch.Tensor | None
    retained_counts: torch.Tensor
    original_counts: torch.Tensor
    event_source: torch.Tensor | None = None

    @property
    def retained_fraction(self) -> torch.Tensor:
        return self.retained_counts.float() / self.original_counts.clamp_min(1).float()


def _validate_prefix(event_type: torch.Tensor, event_time: torch.Tensor) -> None:
    if event_type.ndim != 2 or event_time.ndim != 2:
        raise ValueError("event_type and event_time must both be rank-2 tensors")
    if event_type.shape != event_time.shape:
        raise ValueError("event_type and event_time must have identical shapes")
    if event_type.size(1) == 0:
        raise ValueError("prefix tensors must contain at least one sequence position")
    valid = event_type.ne(Constants.PAD)
    after_first_pad = event_type.eq(Constants.PAD).cumsum(dim=1).gt(0)
    if torch.any(valid & after_first_pad):
        raise ValueError("valid events must be left-aligned before suffix padding")


def _pack_selected_prefix(
    event_type: torch.Tensor,
    event_time: torch.Tensor,
    keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack selected valid events to the left while preserving their order."""

    _validate_prefix(event_type, event_time)
    if keep.shape != event_type.shape or keep.dtype != torch.bool:
        raise ValueError("keep must be a boolean tensor matching the prefix shape")
    valid = event_type.ne(Constants.PAD)
    if torch.any(keep & ~valid):
        raise ValueError("keep selects one or more padding positions")

    packed_type = torch.full_like(event_type, Constants.PAD)
    packed_time = torch.zeros_like(event_time)
    original_counts = valid.sum(dim=1)
    retained_counts = keep.sum(dim=1)
    for row in range(event_type.size(0)):
        selected_type = event_type[row, keep[row]]
        selected_time = event_time[row, keep[row]]
        length = int(selected_type.numel())
        if length:
            packed_type[row, :length] = selected_type
            packed_time[row, :length] = selected_time
    return packed_type, packed_time, retained_counts, original_counts


def _pack_selected_sources(event_source: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    if event_source.shape != keep.shape:
        raise ValueError("event_source and keep must have identical shapes")
    packed = torch.full_like(event_source, -1)
    for row in range(event_source.size(0)):
        selected = event_source[row, keep[row]]
        if selected.numel():
            packed[row, : selected.numel()] = selected
    return packed


def event_sources_from_token_namespace(
    event_type: torch.Tensor,
    id_to_token: Mapping[str | int, str],
    source_names: Sequence[str],
    source_aliases: Mapping[str, str] | None = None,
) -> torch.Tensor:
    """Recover per-event source IDs from the packaged token namespace.

    Model event IDs are one greater than the zero-based keys in
    ``id_to_token.json``.  eICU uses both ``infusion`` and ``infusionDrug`` token
    prefixes for the same stored source-count family, so that alias is enabled by
    default.  Other unknown prefixes fail closed instead of being guessed.
    """

    if event_type.ndim != 2:
        raise ValueError("event_type must be rank 2")
    lookup = token_source_lookup(id_to_token, source_names, source_aliases)
    return event_sources_from_lookup(event_type, lookup)


def token_source_lookup(
    id_to_token: Mapping[str | int, str],
    source_names: Sequence[str],
    source_aliases: Mapping[str, str] | None = None,
) -> dict[int, int]:
    """Map zero-based token IDs to the frozen source-summary column order."""

    names = [str(name) for name in source_names]
    if len(set(names)) != len(names):
        raise ValueError("source_names must be unique")
    source_index = {name: index for index, name in enumerate(names)}
    aliases = {"infusion": "infusionDrug"}
    if source_aliases is not None:
        aliases.update({str(key): str(value) for key, value in source_aliases.items()})
    lookup: dict[int, int] = {}
    missing: set[str] = set()
    for raw_id, raw_token in id_to_token.items():
        token = str(raw_token)
        prefix = token.split("::", 1)[0]
        normalized_source = aliases.get(prefix, prefix)
        index = source_index.get(normalized_source)
        if index is None:
            missing.add(f"source:{normalized_source}")
            continue
        lookup[int(raw_id)] = int(index)
    if missing:
        raise ValueError(f"unresolved event-source namespace entries: {sorted(missing)}")
    return lookup


def event_sources_from_lookup(
    event_type: torch.Tensor,
    token_source_by_id: Mapping[int, int],
) -> torch.Tensor:
    """Resolve model event tensors using a prevalidated zero-based lookup."""

    if event_type.ndim != 2:
        raise ValueError("event_type must be rank 2")
    event_source = torch.full_like(event_type, -1)
    missing: set[int] = set()
    for raw_model_id in torch.unique(event_type[event_type.ne(Constants.PAD)]).tolist():
        model_id = int(raw_model_id)
        index = token_source_by_id.get(model_id - 1)
        if index is None:
            missing.add(model_id - 1)
            continue
        event_source[event_type.eq(model_id)] = int(index)
    if missing:
        raise ValueError(f"unresolved zero-based token IDs: {sorted(missing)}")
    return event_source


def source_summaries_from_event_sources(
    event_source: torch.Tensor,
    keep: torch.Tensor,
    num_sources: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Recompute log-count and fraction summaries from retained source IDs."""

    if event_source.shape != keep.shape or keep.dtype != torch.bool:
        raise ValueError("event_source and boolean keep tensors must have identical shapes")
    num_sources = int(num_sources)
    if num_sources <= 0:
        raise ValueError("num_sources must be positive")
    selected = event_source[keep]
    if selected.numel() and (selected.min() < 0 or selected.max() >= num_sources):
        raise ValueError("event_source contains an index outside num_sources")
    counts = torch.zeros(
        (event_source.size(0), num_sources), dtype=dtype, device=event_source.device
    )
    for source_index in range(num_sources):
        counts[:, source_index] = (keep & event_source.eq(source_index)).sum(dim=1)
    totals = counts.sum(dim=1, keepdim=True)
    fractions = torch.where(totals > 0.0, counts / totals.clamp_min(1.0), counts)
    return torch.cat([torch.log1p(counts), fractions], dim=1)


def retain_recent_trajectory(
    event_type: torch.Tensor,
    event_time: torch.Tensor,
    context_length: int,
) -> PerturbedPrefix:
    """Keep the most recent ``context_length`` valid trajectory events.

    Absolute times remain measured from the original landmark prefix origin.  A
    caller can therefore use the returned tensors for the trajectory path while
    continuing to compute the aggregate-state path from the unmodified prefix.
    """

    _validate_prefix(event_type, event_time)
    context_length = int(context_length)
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    valid = event_type.ne(Constants.PAD)
    keep = torch.zeros_like(valid)
    for row, length in enumerate(valid.sum(dim=1).tolist()):
        length = int(length)
        start = max(0, length - context_length)
        keep[row, start:length] = True
    packed_type, packed_time, retained, original = _pack_selected_prefix(
        event_type, event_time, keep
    )
    return PerturbedPrefix(packed_type, packed_time, None, retained, original)


def _stable_patient_seed(global_seed: int, patient_id: torch.Tensor) -> int:
    values = ",".join(str(int(value)) for value in patient_id.detach().cpu().tolist())
    digest = hashlib.blake2b(
        f"{int(global_seed)}|{values}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def nested_retention_scores(
    event_type: torch.Tensor,
    patient_ids: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Create patient-stable event scores shared by all retention rates.

    Reusing one score tensor for 75/50/25% makes every lower-rate mask a subset
    of the higher-rate mask (the masks can coincide for short prefixes).  The
    scores depend on stable patient IDs, not batch order or batch size.
    """

    if event_type.ndim != 2:
        raise ValueError("event_type must be rank 2")
    if patient_ids.ndim != 2 or patient_ids.size(0) != event_type.size(0):
        raise ValueError("patient_ids must have shape [batch, id_fields]")
    scores = torch.ones(event_type.shape, dtype=torch.float32, device=event_type.device)
    valid = event_type.ne(Constants.PAD)
    for row, length in enumerate(valid.sum(dim=1).tolist()):
        length = int(length)
        if not length:
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_patient_seed(seed, patient_ids[row]))
        row_scores = torch.rand(length, generator=generator, dtype=torch.float32)
        scores[row, :length] = row_scores.to(event_type.device)
    return scores


def fixed_retention_mask(
    event_type: torch.Tensor,
    scores: torch.Tensor,
    retention_rate: float,
) -> torch.Tensor:
    """Select a fixed nested fraction of each patient's valid events."""

    retention_rate = float(retention_rate)
    if not 0.0 < retention_rate <= 1.0:
        raise ValueError("retention_rate must lie in (0, 1]")
    if scores.shape != event_type.shape:
        raise ValueError("scores must match the prefix shape")
    valid = event_type.ne(Constants.PAD)
    keep = torch.zeros_like(valid)
    for row, length in enumerate(valid.sum(dim=1).tolist()):
        length = int(length)
        if not length:
            continue
        retained = max(1, int(math.ceil(retention_rate * length)))
        selected = torch.argsort(scores[row, :length], stable=True)[:retained]
        keep[row, selected] = True
    return keep


def retain_uniform_random_events(
    event_type: torch.Tensor,
    event_time: torch.Tensor,
    source_features: torch.Tensor | None,
    scores: torch.Tensor,
    retention_rate: float,
    event_source: torch.Tensor | None = None,
) -> PerturbedPrefix:
    """Uniformly thin a prefix using prespecified nested event scores.

    Exactly ``max(1, ceil(rate * n))`` events are selected from each nonempty
    prefix.  All rates share the same random ordering, so the 25% selection is a
    subset of 50%, which is a subset of 75%, while selected events are repacked
    in their original chronological order.
    """

    _validate_prefix(event_type, event_time)
    keep = fixed_retention_mask(event_type, scores, retention_rate)
    packed_type, packed_time, retained, original = _pack_selected_prefix(
        event_type, event_time, keep
    )
    packed_source = None
    if event_source is not None:
        if event_source.shape != event_type.shape:
            raise ValueError("event_source must match the prefix shape")
        packed_source = _pack_selected_sources(event_source, keep)
    if source_features is not None and event_source is not None:
        if source_features.size(1) % 2:
            raise ValueError("source_features must have 2 * num_sources columns")
        if torch.equal(retained, original):
            adjusted_sources = source_features.clone()
        else:
            adjusted_sources = source_summaries_from_event_sources(
                event_source,
                keep,
                source_features.size(1) // 2,
                dtype=source_features.dtype,
            )
    elif source_features is not None:
        raise ValueError(
            "event_source is required to recompute source summaries exactly"
        )
    else:
        adjusted_sources = None
    return PerturbedPrefix(
        packed_type,
        packed_time,
        adjusted_sources,
        retained,
        original,
        packed_source,
    )


def remove_source_events(
    event_type: torch.Tensor,
    event_time: torch.Tensor,
    event_source: torch.Tensor,
    source_features: torch.Tensor,
    source_index: int,
) -> PerturbedPrefix:
    """Remove all events from one source when per-event source IDs are available.

    ``event_source`` is reconstructed exactly from the packaged token namespace
    with :func:`event_sources_from_token_namespace`.  Rows emptied by removal are
    returned with zero retained events so a runner can report or prespecify their
    handling instead of silently imputing.
    """

    _validate_prefix(event_type, event_time)
    if event_source.shape != event_type.shape:
        raise ValueError("event_source must match the prefix shape")
    if source_features.ndim != 2 or source_features.size(1) % 2:
        raise ValueError("source_features must have shape [batch, 2 * num_sources]")
    n_sources = source_features.size(1) // 2
    source_index = int(source_index)
    if not 0 <= source_index < n_sources:
        raise ValueError("source_index is outside the source summary dimension")

    valid = event_type.ne(Constants.PAD)
    keep = valid & event_source.ne(source_index)
    packed_type, packed_time, retained, original = _pack_selected_prefix(
        event_type, event_time, keep
    )

    adjusted_sources = source_summaries_from_event_sources(
        event_source, keep, n_sources, dtype=source_features.dtype
    )
    packed_source = _pack_selected_sources(event_source, keep)
    return PerturbedPrefix(
        packed_type,
        packed_time,
        adjusted_sources,
        retained,
        original,
        packed_source,
    )


class FrozenTraceSensitivityAdapter(nn.Module):
    """Evaluate trajectory and state views from different versions of a prefix."""

    def __init__(self, model: ClinicalPredictionModel):
        super().__init__()
        if not isinstance(model, ClinicalPredictionModel):
            raise TypeError("model must be a ClinicalPredictionModel")
        if not isinstance(model.encoder, ClinicalHawkesMambaEncoder):
            raise TypeError("model encoder must be ClinicalHawkesMambaEncoder")
        encoder = model.encoder
        if encoder.trajectory_fusion_gate != 1.0:
            raise ValueError("frozen sensitivity analysis requires the enabled Full trajectory")
        if encoder.static_encoder is None or encoder.rich_static_fusion_gate != 1.0:
            raise ValueError("frozen sensitivity analysis requires the enabled Full state path")
        self.model = model

    @property
    def encoder(self) -> ClinicalHawkesMambaEncoder:
        return self.model.encoder

    def _encode_trajectory(
        self, event_type: torch.Tensor, event_time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_prefix(event_type, event_time)
        encoder = self.encoder
        non_pad = event_type.ne(Constants.PAD)
        time_gap = torch.zeros_like(event_time)
        time_gap[:, 1:] = torch.clamp(event_time[:, 1:] - event_time[:, :-1], min=0.0)
        time_features = torch.stack(
            [
                event_time / encoder.time_scale_hours,
                time_gap / encoder.time_scale_hours,
            ],
            dim=-1,
        )
        hidden = encoder.event_emb(event_type) + encoder.time_proj(time_features)
        if encoder.decayed_history_enabled:
            history = encoder.hawkes_proj(encoder.hawkes_context(event_type, event_time))
            hidden = hidden + (
                encoder.hawkes_fusion_gate * encoder.hawkes_context_scale * history
            )
        hidden = encoder.dropout(hidden)
        if encoder.sequence_backbone == "transformer":
            encoded = encoder.encoder(hidden, src_key_padding_mask=~non_pad)
        else:
            encoded = encoder.encoder(hidden)
        sequence_hidden = encoder.norm(encoded) * non_pad.unsqueeze(-1).float()
        trajectory_pooled = pool_sequence(
            sequence_hidden,
            non_pad,
            pooling=encoder.pooling,
            attention=encoder.attention,
        )
        return trajectory_pooled, sequence_hidden

    def forward_views(
        self,
        trajectory_event_type: torch.Tensor,
        trajectory_event_time: torch.Tensor,
        state_event_type: torch.Tensor,
        state_event_time: torch.Tensor,
        state_source_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory_pooled, sequence_hidden = self._encode_trajectory(
            trajectory_event_type, trajectory_event_time
        )
        static_pooled, _ = self.encoder.static_encoder(
            state_event_type,
            state_event_time,
            source_features=state_source_features,
        )
        pooled = self.encoder.fuse_views(trajectory_pooled, static_pooled)
        logits = self.model.head(pooled)
        return logits, sequence_hidden, pooled

    def forward_context_length(
        self,
        event_type: torch.Tensor,
        event_time: torch.Tensor,
        source_features: torch.Tensor | None,
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory = retain_recent_trajectory(event_type, event_time, context_length)
        return self.forward_views(
            trajectory.event_type,
            trajectory.event_time,
            event_type,
            event_time,
            source_features,
        )

    def forward_retained_prefix(
        self,
        event_type: torch.Tensor,
        event_time: torch.Tensor,
        source_features: torch.Tensor | None,
        scores: torch.Tensor,
        retention_rate: float,
        event_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PerturbedPrefix]:
        retained = retain_uniform_random_events(
            event_type,
            event_time,
            source_features,
            scores,
            retention_rate,
            event_source=event_source,
        )
        logits, sequence_hidden, pooled = self.forward_views(
            retained.event_type,
            retained.event_time,
            retained.event_type,
            retained.event_time,
            retained.source_features,
        )
        return logits, sequence_hidden, pooled, retained
