"""One F2 backbone; optional numerical encoding and shared member modulation.

PBLD equations follow the installed PyTabKit implementation. Rank-one member
modulation is applied after the shared wide input projection, not to three
independently trained prediction networks. This is an architectural adaptation,
not a reproduction of either complete RealMLP or TabM training recipes.
"""
import hashlib
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def independent_init(seed, name):
    value = int.from_bytes(hashlib.sha256(f'{seed}/{name}'.encode()).digest()[:8], 'little') % (2**63-1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(value)
        yield


class MemberLinear(nn.Module):
    def __init__(self, d_in, d_out, members, seed, name):
        super().__init__()
        with independent_init(seed, name + '/shared'):
            self.shared = nn.Linear(d_in, d_out)
        if members > 1:
            with independent_init(seed, name + '/members'):
                self.fast_r = nn.Parameter(torch.randint(2, (members, d_in)).float().mul_(2).sub_(1))
                self.fast_s = nn.Parameter(torch.ones(members, d_out))
        else:
            self.register_parameter('fast_r', None)
            self.register_parameter('fast_s', None)

    def forward(self, x):
        if self.fast_r is None:
            return self.shared(x)
        # Keep the shared bias outside multiplicative modulation.
        return F.linear(x * self.fast_r, self.shared.weight) * self.fast_s + self.shared.bias


class ResidualBlock(nn.Module):
    def __init__(self, width, members, dropout, seed, name):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = MemberLinear(width, 2*width, members, seed, name + '/up')
        self.down = MemberLinear(2*width, width, members, seed, name + '/down')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.dropout(F.gelu(self.up(self.norm(x))))
        return x + self.dropout(self.down(z))


class NumericalEncoding(nn.Module):
    """Feature-wise scale and periodic-biased encoding; raw coordinates survive."""
    def __init__(self, n_features, width, seed, frequencies=16, output_dim=3):
        super().__init__()
        assert n_features > 0
        self.scale = nn.Parameter(torch.ones(n_features))
        with independent_init(seed, 'numerical/periodic'):
            self.frequency = nn.Parameter(.1 * torch.randn(n_features, frequencies))
            self.phase = nn.Parameter(math.pi * (2*torch.rand(n_features, frequencies)-1))
            self.weight = nn.Parameter((2*torch.rand(n_features, frequencies, output_dim)-1)/math.sqrt(frequencies))
            self.bias = nn.Parameter((2*torch.rand(n_features, output_dim)-1)/math.sqrt(frequencies))
        with independent_init(seed, 'numerical/projection'):
            self.projection = nn.Linear(n_features*output_dim, width, bias=False)

    def forward(self, scaled):
        z = torch.cos(2*math.pi*scaled.unsqueeze(-1)*self.frequency + self.phase)
        # Feature dimension is the batched matrix dimension, avoiding a B*F*H*D expansion.
        z = torch.bmm(z.transpose(0, 1), self.weight).transpose(0, 1) + self.bias
        return self.projection(z.flatten(1))


class UnifiedNetwork(nn.Module):
    def __init__(self, input_dim, prep, variant, seed=42, width=256, depth=3, dropout=.1):
        super().__init__()
        assert variant in ('N0', 'N1', 'N2', 'N3')
        self.variant = variant
        self.members = 8 if variant in ('N2', 'N3') else 1
        self.register_buffer('center', torch.as_tensor(prep['center'], dtype=torch.float32))
        self.register_buffer('factor', torch.as_tensor(prep['factor'], dtype=torch.float32))
        self.register_buffer('binary', torch.as_tensor(prep['binary'], dtype=torch.bool))
        self.register_buffer('numeric_indices', torch.as_tensor(prep['numeric_indices'], dtype=torch.long))
        assert self.center.numel() == self.factor.numel() == self.binary.numel() == input_dim
        with independent_init(seed, 'input_projection'):
            self.input_projection = nn.Linear(input_dim, width)
        self.numerical = NumericalEncoding(len(self.numeric_indices), width, seed) if variant in ('N1', 'N3') else None
        self.input_norm = nn.LayerNorm(width)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ResidualBlock(width, self.members, dropout, seed, f'blocks/{i}') for i in range(depth)])
        self.output_norm = nn.LayerNorm(width)
        self.heads = nn.ModuleList([MemberLinear(width, 1, self.members, seed, f'head/{i}') for i in range(2)])

    def transform(self, x):
        z = (x-self.center)*self.factor
        z = z / torch.sqrt(1+(z/3).square())
        return torch.where(self.binary, x, z)

    def forward(self, raw):
        x = self.transform(raw)
        if self.numerical is not None:
            numeric = x.index_select(1, self.numeric_indices) * self.numerical.scale
            x = x.index_copy(1, self.numeric_indices, numeric)
        h = self.input_projection(x)
        if self.numerical is not None:
            h = (h + self.numerical(numeric)) / math.sqrt(2)
        h = self.input_dropout(F.gelu(self.input_norm(h)))
        h = h.unsqueeze(1).expand(-1, self.members, -1)
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)
        return torch.cat([head(h) for head in self.heads], dim=-1)

    def probabilities(self, raw):
        return self(raw).sigmoid().mean(1)


def member_loss(logits, labels):
    assert logits.ndim == 3 and labels.shape == (len(logits), 2)
    return F.binary_cross_entropy_with_logits(logits, labels[:, None, :].expand_as(logits))


def parameter_hash(model, common_only=False):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if common_only and (name.startswith('numerical.') or '.fast_' in name):
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
