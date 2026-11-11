"""Opt-in shared statistics for the frozen C + statistical residual contract."""
import torch
from torch import nn
from models.clinical_dense_optimization import DenseRepresentation


class SharedStatisticsResidual(nn.Module):
    """Reuse modules without reinitializing an already-trained residual head.

    Only the currently audited C/rich contract is accepted. State-dict keys are
    unchanged, so the old event implementation remains an independent oracle.
    """
    def __init__(self, original):
        super().__init__()
        if original.aux_weight != 0:
            raise ValueError('Shared path only supports the statistical residual')
        a, r = original.anchor.encoder.active, original.residual.encoder.active
        for rep in (a, r):
            if not isinstance(rep, DenseRepresentation) or rep.variant != 'C':
                raise ValueError('Requires C representation on both branches')
            if rep.base.feature_set != 'rich':
                raise ValueError('Requires rich statistics')
        for name in ('num_types', 'source_dim', 'time_scale_hours', 'time_decay_hours', 'input_dim'):
            if getattr(a.base, name) != getattr(r.base, name):
                raise ValueError('Incompatible feature contract: '+name)
        self.anchor = original.anchor.requires_grad_(False).eval()
        self.residual = original.residual
        self.aux_weight = 0.
        self.aux_loss = None

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward_features(self, features):
        with torch.no_grad():
            s = self.anchor.encoder.active.base.net(features)
            z = self.anchor.head(s)
        t = self.residual.encoder.active.base.net(features)
        return z + self.residual.head(t), t

    def forward(self, event_type, event_time, time_gap=None, source_features=None):
        features = self.anchor.encoder.active.unscaled_features(event_type, event_time, source_features)
        logits, pooled = self.forward_features(features)
        hidden = pooled.unsqueeze(1).expand(-1, event_type.size(1), -1) * event_type.ne(0).unsqueeze(-1)
        return logits, hidden, pooled
