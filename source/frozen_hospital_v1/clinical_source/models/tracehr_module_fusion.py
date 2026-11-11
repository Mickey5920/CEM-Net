"""Direct representation fusion; native probability heads are training auxiliaries."""
import torch
from torch import nn


class ResidualReadout(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=.1):
        super().__init__()
        self.enter = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU())
        self.block = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                   nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.exit = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        x = self.enter(x)
        return self.exit(x + self.block(x)).squeeze(-1)


class ModuleFusion(nn.Module):
    def __init__(self, branches='AVRT', fusion='F1', r_dim=256, n_temporal=40,
                 width=128, head_width=256, dropout=.1, fixed_query=False,
                 compress_members=False):
        super().__init__()
        assert branches and len(set(branches)) == len(branches) and set(branches) <= set('AVRT')
        assert fusion in ('F0', 'F1')
        self.branches, self.fusion = branches, fusion
        self.fixed_query, self.compress_members = fixed_query, compress_members
        dimensions = dict(A=96, V=512 if compress_members else 256, R=r_dim, T=128)
        sizes = dict(A=2, V=2 if compress_members else 64, R=2, T=n_temporal)
        self.project = nn.ModuleDict({e: nn.Sequential(nn.LayerNorm(dimensions[e]),
                                  nn.Linear(dimensions[e], width), nn.GELU()) for e in branches})
        self.identity = nn.ParameterDict({e: nn.Parameter(torch.randn(sizes[e], width)*.02) for e in branches})
        self.query = nn.Parameter(torch.randn(2, width)*.02)
        self.pool = nn.ModuleDict({e: nn.MultiheadAttention(width, 4, dropout=dropout,
                                                          batch_first=True) for e in branches})
        self.head = ResidualReadout(len(branches)*width, head_width, dropout)
        if fusion == 'F1':
            self.context = nn.Sequential(nn.Linear(len(branches)*width, width), nn.GELU(), nn.Linear(width, width))
            self.read = nn.ModuleDict({e: nn.MultiheadAttention(width, 4, dropout=dropout,
                                                              batch_first=True) for e in branches})
            self.norm = nn.ModuleDict({e: nn.LayerNorm(width) for e in branches})

    def forward(self, hidden, valid):
        n = len(hidden[self.branches[0]])
        q = self.query[None].expand(n, -1, -1)
        tokens, masks, summary = {}, {}, {}
        for e in self.branches:
            x, mask = hidden[e], valid[e]
            if e == 'V' and self.compress_members:
                x = x.reshape(n, 2, 32, 256)
                x = torch.cat((x.mean(2), x.std(2, unbiased=False)), -1)
                mask = torch.ones((n, 2), dtype=torch.bool, device=x.device)
            tokens[e] = self.project[e](x) + self.identity[e][None]
            masks[e] = ~mask
            assert mask.any(1).all(), 'Each branch requires at least a numerical placeholder'
            summary[e], _ = self.pool[e](q, tokens[e], tokens[e], key_padding_mask=masks[e], need_weights=False)
        if self.fusion == 'F1':
            context = torch.cat([summary[e] for e in self.branches], -1)
            condition = q if self.fixed_query else q + self.context(context)
            for e in self.branches:
                value, _ = self.read[e](condition, tokens[e], tokens[e], key_padding_mask=masks[e], need_weights=False)
                summary[e] = self.norm[e](summary[e] + value)
        return self.head(torch.cat([summary[e] for e in self.branches], -1))
