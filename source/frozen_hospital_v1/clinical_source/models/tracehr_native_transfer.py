"""Controlled, freshly initialized transfers around N12; historical code is unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.tracehr_unified import independent_init, MemberLinear
from models.tracehr_unified_next import EndpointResidual


class IndependentHead(nn.Module):
    def __init__(self, previous, members):
        super().__init__()
        assert torch.all(previous.fast_r == 1) and torch.all(previous.fast_s == 1)
        self.weight = nn.Parameter(previous.shared.weight.detach().T[None].repeat(members, 1, 1))
        self.bias = nn.Parameter(previous.shared.bias.detach()[None].repeat(members, 1))

    def forward(self, x):
        return torch.einsum('bki,kio->bko', x, self.weight) + self.bias


class TransferredLinear(nn.Module):
    """Member bias is outside output scaling, matching official TabM semantics."""
    def __init__(self, previous, members, independent_bias=False, input_only=False):
        super().__init__()
        self.shared = previous.shared
        if independent_bias:
            self.shared.bias = nn.Parameter(self.shared.bias.detach()[None].repeat(members, 1))
        self.fast_r = None if input_only else previous.fast_r
        self.fast_s = None if input_only else previous.fast_s

    def forward(self, x):
        z = F.linear(x if self.fast_r is None else x*self.fast_r, self.shared.weight)
        return (z if self.fast_s is None else z*self.fast_s) + self.shared.bias


class PiecewiseEncoding(nn.Module):
    """Three train-fitted intervals, matching PBLD's three learned coordinates."""
    def __init__(self, previous, edges):
        super().__init__()
        self.scale = previous.scale
        self.projection = previous.projection
        self.register_buffer('edges', torch.as_tensor(edges, dtype=torch.float32))
        assert self.edges.shape == (len(self.scale), 4)

    def coordinates(self, x):
        left, right = self.edges[:, :-1], self.edges[:, 1:]
        return ((x[..., None]-left)/(right-left).clamp_min(1e-6)).clamp(0., 1.)


class TransferNetwork(EndpointResidual):
    def __init__(self, input_dim, prep, variant, endpoint, seed=42, width=256, depth=3):
        assert variant in tuple('C'+str(i) for i in range(12))
        super().__init__(input_dim, prep, 'N12', endpoint, seed, width, depth)
        self.experiment_variant = variant
        full_head = variant in ('C1', 'C5', 'C6')
        member_bias = variant in ('C2', 'C5', 'C6')
        untie = variant in ('C3', 'C5', 'C6')
        input_only = variant in ('C4', 'C6')
        if full_head:
            self.heads[0] = IndependentHead(self.heads[0], self.members)
        if member_bias or input_only:
            for block in self.blocks:
                for name in ('up', 'down'):
                    block._modules[name] = TransferredLinear(getattr(block, name), self.members,
                                                             member_bias, input_only)
        if untie:
            init = self.input_members.detach().index_select(1, self.numeric_indices)
            self.numeric_members = nn.Parameter(init[..., None].repeat(1, 1, 3))
        if variant == 'C7':
            self.numeric_scale = self.numerical.scale
            self.numerical = None
        elif variant == 'C8':
            self.numerical = PiecewiseEncoding(self.numerical, prep['ple_edges'])
        assert all(p.requires_grad for p in self.parameters())

    def forward(self, raw):
        # Preserve the original operation order for the exact N12 control and
        # all variants that do not change numerical/input encoding.
        if not hasattr(self, 'numeric_members') and self.experiment_variant not in ('C7', 'C8'):
            return super().forward(raw)
        x = self.transform(raw)
        if self.numerical is None:
            numeric = x.index_select(1, self.numeric_indices) * self.numeric_scale
            x = x.index_copy(1, self.numeric_indices, numeric)
        if self.numerical is not None:
            numeric = x.index_select(1, self.numeric_indices) * self.numerical.scale
            x = x.index_copy(1, self.numeric_indices, numeric)
        h = self.input_projection(x[:, None] * self.input_members[None])
        if self.numerical is not None:
            if isinstance(self.numerical, PiecewiseEncoding):
                z = self.numerical.coordinates(numeric)
            else:
                z = torch.cos(2*math.pi*numeric.unsqueeze(-1)*self.numerical.frequency + self.numerical.phase)
                z = torch.bmm(z.transpose(0, 1), self.numerical.weight).transpose(0, 1) + self.numerical.bias
            if hasattr(self, 'numeric_members'):
                z = (z[:, None] * self.numeric_members[None]).flatten(2)
            else:
                scale = self.input_members.index_select(1, self.numeric_indices)
                z = (z[:, None] * scale[None, :, :, None]).flatten(2)
            h = (h + self.numerical.projection(z)) / math.sqrt(2)
        h = self.input_dropout(F.gelu(self.input_norm(h)))
        for block in self.blocks:
            h = block(h)
        return self.heads[0](self.output_norm(h))


def adapter_parameter(name):
    return '.fast_' in name or name in ('input_members', 'numeric_members')
