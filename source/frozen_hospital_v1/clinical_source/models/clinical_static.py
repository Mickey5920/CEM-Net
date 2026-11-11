import torch
import torch.nn as nn

import transformer.Constants as Constants


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ClinicalStaticMLPEncoder(nn.Module):
    """Bag-of-events neural baseline built from the fixed event-token PKL."""

    def __init__(
        self,
        num_types,
        d_model=64,
        hidden_dim=64,
        n_layers=2,
        dropout=0.1,
        residual=False,
        time_scale_hours=24.0,
        feature_set='basic',
        source_dim=0,
        time_decay_hours=12.0,
    ):
        super().__init__()
        if feature_set not in {'basic', 'rich'}:
            raise ValueError(f'Unknown static feature_set={feature_set!r}')
        self.num_types = int(num_types)
        self.hidden_dim = int(hidden_dim)
        self.residual = bool(residual)
        self.time_scale_hours = float(time_scale_hours)
        self.feature_set = feature_set
        self.source_dim = int(source_dim)
        self.time_decay_hours = float(time_decay_hours)
        if self.feature_set == 'rich':
            self.input_dim = int(num_types) * 4 + self.source_dim + 4
        else:
            self.input_dim = int(num_types) + 4
        layers = [
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        if self.residual:
            layers.extend(ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(max(int(n_layers) - 1, 1)))
        else:
            for _ in range(max(int(n_layers) - 1, 0)):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, d_model))
        layers.append(nn.LayerNorm(d_model))
        self.net = nn.Sequential(*layers)

    def _event_counts(self, event_type):
        non_pad = event_type.ne(Constants.PAD)
        clipped = event_type.clamp(min=0, max=self.num_types)
        counts = event_type.new_zeros((event_type.size(0), self.num_types + 1), dtype=torch.float32)
        counts.scatter_add_(1, clipped, non_pad.float())
        return counts[:, 1:], non_pad, clipped

    def _last_time_by_type(self, clipped, event_time, non_pad):
        last_time = event_time.new_zeros((event_time.size(0), self.num_types + 1))
        values = event_time * non_pad.float()
        if hasattr(last_time, 'scatter_reduce_'):
            last_time.scatter_reduce_(1, clipped, values, reduce='amax', include_self=True)
        else:
            rows = torch.arange(event_time.size(0), device=event_time.device)
            for position in range(event_time.size(1)):
                index = clipped[:, position]
                current = last_time[rows, index]
                last_time[rows, index] = torch.maximum(current, values[:, position])
        return last_time[:, 1:]

    def build_static_features(self, event_type, event_time, source_features=None):
        counts, non_pad, clipped = self._event_counts(event_type)
        length = non_pad.sum(dim=1, keepdim=True).float().clamp_min(1.0)

        masked_time = event_time * non_pad.float()
        max_time = masked_time.max(dim=1, keepdim=True).values
        mean_time = masked_time.sum(dim=1, keepdim=True) / length
        density = length / max_time.clamp_min(1.0)
        summary = torch.cat(
            [
                length / 512.0,
                max_time / self.time_scale_hours,
                mean_time / self.time_scale_hours,
                density,
            ],
            dim=1,
        )

        if self.feature_set == 'basic':
            log_counts = torch.log1p(counts) / torch.log1p(length)
            return torch.cat([log_counts, summary], dim=1)

        present = counts.gt(0).float()
        window_hours = event_time.new_full((event_time.size(0), 1), max(self.time_scale_hours, 1e-6))
        last_time = self._last_time_by_type(clipped, event_time, non_pad)
        last_time_fraction = last_time / window_hours

        decay_scale = max(self.time_decay_hours, 1e-6)
        decay_values = torch.exp(-torch.clamp(window_hours - event_time, min=0.0) / decay_scale)
        decay_values = decay_values * non_pad.float()
        decay_count = event_time.new_zeros((event_time.size(0), self.num_types + 1))
        decay_count.scatter_add_(1, clipped, decay_values)
        decay_count = decay_count[:, 1:]

        if self.source_dim > 0:
            if source_features is None:
                source_features = event_time.new_zeros((event_time.size(0), self.source_dim))
            else:
                source_features = source_features.to(device=event_time.device, dtype=event_time.dtype)
                if source_features.size(1) != self.source_dim:
                    adjusted = event_time.new_zeros((event_time.size(0), self.source_dim))
                    width = min(source_features.size(1), self.source_dim)
                    adjusted[:, :width] = source_features[:, :width]
                    source_features = adjusted
        else:
            source_features = event_time.new_zeros((event_time.size(0), 0))

        return torch.cat([counts, present, last_time_fraction, decay_count, source_features, summary], dim=1)

    def forward(self, event_type, event_time, source_features=None):
        pooled = self.net(self.build_static_features(event_type, event_time, source_features=source_features))
        sequence_hidden = pooled.unsqueeze(1).expand(-1, event_type.size(1), -1)
        sequence_hidden = sequence_hidden * event_type.ne(Constants.PAD).unsqueeze(-1).float()
        return pooled, sequence_hidden
