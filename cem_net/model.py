"""Core PyTorch implementation of CEM-Net (Clinical Evidence Modulation Network)."""
import copy
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def fit_profile_normalizer(x_train: np.ndarray, phi_dim: int = None) -> dict:
    """Fit training-only median/IQR scaling and identify eligible numerical coordinates."""
    x = np.asarray(x_train, dtype=np.float64)
    n, d = x.shape
    if phi_dim is None:
        phi_dim = d
    center = np.zeros(d, dtype=np.float32)
    factor = np.ones(d, dtype=np.float32)
    binary = np.all((x == 0.0) | (x == 1.0), axis=0)
    nonconstant = x.min(axis=0) != x.max(axis=0)

    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75], axis=0)
    spread = q3 - q1

    if phi_dim > 0:
        half_range = 0.5 * (x[:, :phi_dim].max(axis=0) - x[:, :phi_dim].min(axis=0))
        sp_e = np.where(spread[:phi_dim] > 0.0, spread[:phi_dim], half_range)
        center[:phi_dim] = med[:phi_dim]
        factor[:phi_dim] = np.divide(1.0, sp_e, out=np.zeros_like(sp_e), where=sp_e != 0.0)

    if phi_dim < d:
        sp_l = np.where(spread[phi_dim:] > 1e-6, spread[phi_dim:], np.std(x[:, phi_dim:], axis=0))
        center[phi_dim:] = med[phi_dim:]
        factor[phi_dim:] = 1.0 / np.maximum(sp_l, 1.0)

    center[binary] = 0.0
    factor[binary] = 1.0
    numeric_indices = np.flatnonzero(~binary & nonconstant).astype(np.int64)
    return dict(
        center=center,
        factor=factor,
        binary=binary,
        passthrough=binary.copy(),
        numeric_indices=numeric_indices,
    )


class TypedPeriodic(nn.Module):
    """Coordinate-wise periodic numerical embeddings concatenated with the direct scaled profile."""

    def __init__(self, prep: dict, basis_dim: int = 16, out_dim: int = 3):
        super().__init__()
        self.register_buffer("center", torch.as_tensor(prep["center"], dtype=torch.float32))
        self.register_buffer("factor", torch.as_tensor(prep["factor"], dtype=torch.float32))
        self.register_buffer("binary", torch.as_tensor(prep["binary"], dtype=torch.bool))
        self.register_buffer("passthrough", torch.as_tensor(prep.get("passthrough", prep["binary"]), dtype=torch.bool))
        self.register_buffer("numeric_indices", torch.as_tensor(prep["numeric_indices"], dtype=torch.long))

        q = len(self.numeric_indices)
        self.basis_dim = int(basis_dim)
        self.out_dim = int(out_dim)
        self.frequency = nn.Parameter(0.1 * torch.randn(q, basis_dim))
        self.phase = nn.Parameter(math.pi * (2.0 * torch.rand(q, basis_dim) - 1.0))
        self.weight = nn.Parameter((2.0 * torch.rand(q, basis_dim, out_dim) - 1.0) / 4.0)
        self.bias = nn.Parameter((2.0 * torch.rand(q, out_dim) - 1.0) / 4.0)

    def scale_direct(self, raw: torch.Tensor) -> torch.Tensor:
        x = (raw - self.center) * self.factor
        return torch.where(self.passthrough, raw, x / torch.sqrt(1.0 + (x / 3.0).square()))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        x = self.scale_direct(raw)
        if len(self.numeric_indices) == 0:
            return x
        numeric = x.index_select(1, self.numeric_indices)
        z = torch.cos(2.0 * math.pi * numeric.unsqueeze(-1) * self.frequency + self.phase)
        z = torch.bmm(z.transpose(0, 1), self.weight).transpose(0, 1) + self.bias
        return torch.cat((z.flatten(1), x), dim=1)


class ParametricSELU(nn.Module):
    """Coordinate-wise parametric SELU activation: alpha * SELU(x) + (1 - alpha) * x."""

    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * F.selu(x) + (1.0 - self.weight) * x


class NTKLinear(nn.Module):
    """Linear layer with inverse-square-root fan-in scaling."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(1, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(1, 1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) * (1.0 / math.sqrt(self.in_features)) + self.bias


class CEMNet(nn.Module):
    """Clinical Evidence Modulation Network with K=8 modulated members and shared pSELU trunk."""

    def __init__(
        self,
        prep: dict,
        members: int = 8,
        hidden_width: int = 256,
        depth: int = 3,
        basis_dim: int = 16,
        out_dim: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.members = int(members)
        self.numerical = TypedPeriodic(prep, basis_dim=basis_dim, out_dim=out_dim)
        d = len(prep["center"])
        q = len(prep["numeric_indices"])
        expanded_dim = d + out_dim * q

        fmap = torch.cat((
            self.numerical.numeric_indices.repeat_interleave(out_dim),
            torch.arange(d, dtype=torch.long),
        ))
        self.register_buffer("feature_map", fmap)

        # Initialize member factors aligned per parent feature, then untie across all D coordinates
        parent_factors = torch.randn(self.members, d)
        self.input_members = nn.Parameter(parent_factors.index_select(1, fmap).clone())
        self.front_scale = nn.Parameter(torch.ones(1, 1, expanded_dim))

        layers = []
        in_dim = expanded_dim
        for _ in range(depth):
            layers.append(nn.ModuleList([NTKLinear(in_dim, hidden_width), ParametricSELU(hidden_width), nn.Dropout(dropout)]))
            in_dim = hidden_width
        self.blocks = nn.ModuleList(layers)
        self.head = NTKLinear(hidden_width, 2)

    def set_dropout(self, p: float):
        for _, _, drop in self.blocks:
            drop.p = float(p)

    def forward_trunk(self, encoded: torch.Tensor) -> torch.Tensor:
        h = encoded.unsqueeze(0) * self.input_members.unsqueeze(1)  # [K, B, D]
        h = h * self.front_scale
        for linear, act, drop in self.blocks:
            h = drop(act(linear(h)))
        return self.head(h).transpose(0, 1)  # [B, K, 2]

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.forward_trunk(self.numerical(raw))

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Memberwise two-class cross-entropy scaled by 0.5 (Eq. 9)."""
        targets = y.long().unsqueeze(1).expand(-1, self.members).reshape(-1)
        return 0.5 * F.cross_entropy(logits.reshape(-1, 2), targets)

    def probabilities(self, raw: torch.Tensor) -> torch.Tensor:
        """Mean member positive-class probability (Eq. 10)."""
        return self(raw).softmax(dim=-1)[..., 1].mean(dim=1)


class EMATracker:
    """Exponential moving average over trainable parameters (Eq. 11-12)."""

    def __init__(self, model: nn.Module, decay_cap: float = 0.999):
        self.decay_cap = float(decay_cap)
        self.step_count = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        beta = min(self.decay_cap, (self.step_count + 1.0) / (self.step_count + 10.0))
        for p_s, p in zip(self.shadow.parameters(), model.parameters()):
            p_s.lerp_(p.detach(), 1.0 - beta)
        self.step_count += 1
