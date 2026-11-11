"""Diagnostic additions to an immutable C + statistical residual predictor."""
import torch
from torch import nn
from models.clinical_static import ResidualMLPBlock
from models.clinical_tracehr_fullstream import parameter_count


class SeparateSummaryHead(nn.Module):
    def __init__(self, phi_dim, extra_dim, width):
        super().__init__()
        self.phi_norm = nn.LayerNorm(phi_dim)
        self.phi_linear = nn.Linear(phi_dim, width)
        self.extra_norm = nn.LayerNorm(extra_dim) if extra_dim else None
        self.extra_linear = nn.Linear(extra_dim, width, bias=False) if extra_dim else None
        self.body = nn.Sequential(nn.GELU(), nn.Dropout(.15), ResidualMLPBlock(width, .15),
                                  nn.Linear(width, 96), nn.LayerNorm(96))
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(96, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, phi, extra):
        h = self.phi_linear(self.phi_norm(phi))
        if self.extra_linear is not None:
            h = h + self.extra_linear(self.extra_norm(extra))
        return self.head(self.body(h))


class ProtectedResidual(nn.Module):
    def __init__(self, anchor, phi_dim, extra_dim, base_width, mean, scale):
        super().__init__()
        self.anchor = anchor.requires_grad_(False).eval()
        target = parameter_count(phi_dim, base_width)
        self.width = min(range(2, 193), key=lambda w: abs(parameter_count(phi_dim + extra_dim, w)-target))
        self.residual = SeparateSummaryHead(phi_dim, extra_dim, self.width)
        count = sum(p.numel() for p in self.residual.parameters())
        self.match_error = abs(count / target - 1)
        assert self.match_error < .03
        self.register_buffer('extra_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('extra_scale', torch.as_tensor(scale, dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(self, phi, extra=None):
        with torch.no_grad():
            base = self.anchor(phi)
        if self.extra_mean.numel():
            extra = (extra-self.extra_mean)/self.extra_scale
        return base + self.residual(phi, extra)
