"""G1--G3: variable-preserving trajectories inside one scratch E2 network.

The only variant switches are the source of the temporal query and the source
of summary refinement. Shapes, parameters, initialization and optimizer agree.
No independent probability expert, pretrained cache or stopped gradient exists.
"""
import math
import torch
from torch import nn
from models.tracehr_unified import independent_init


def masked_mean(x, mask, dim):
    return (x * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).clamp_min(1).unsqueeze(-1)


def masked_attention(query, keys, values, mask):
    score = (query.unsqueeze(-2) * keys).sum(-1) / math.sqrt(keys.shape[-1])
    score = score.masked_fill(~mask, -1e4)
    weights = score.softmax(-1) * mask
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return (weights.unsqueeze(-1) * values).sum(-2)


class VariableTrajectory(nn.Module):
    def __init__(self, width=64, bins=8):
        super().__init__()
        self.width, self.bins = width, bins
        self.variable = nn.Embedding(32, width)
        self.record = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        self.score = nn.Linear(width, 1, bias=False)
        self.block = nn.Linear(width + 2, width)
        self.temporal = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.GELU(),
                                      nn.Conv1d(width, width, 3, padding=1))
        self.norm = nn.LayerNorm(width)

    def forward(self, records, segments, counts):
        batch = counts.shape[0]
        n = batch * 32 * self.bins
        variable = torch.div(segments % (32*self.bins), self.bins, rounding_mode='floor')
        tokens = self.record(records) + self.variable(variable)
        scores = self.score(tokens).squeeze(-1)
        maximum = tokens.new_full((n,), -torch.inf)
        maximum.scatter_reduce_(0, segments, scores, reduce='amax', include_self=True)
        weights = (scores - maximum[segments]).exp()
        denominator = tokens.new_zeros(n).index_add(0, segments, weights)
        numerator = tokens.new_zeros((n, self.width)).index_add(0, segments, weights[:, None] * tokens)
        pooled = numerator / denominator.clamp_min(1e-12)[:, None]
        count = counts.reshape(-1).to(tokens.dtype)
        blocks = self.block(torch.cat((pooled, count.log1p()[:, None], (count > 0)[:, None]), -1))
        blocks = blocks.reshape(batch*32, self.bins, self.width)
        observed = counts.reshape(batch*32, self.bins) > 0
        # Never mix two variables while constructing their temporal trajectory.
        blocks = blocks * observed[..., None]
        blocks = self.norm(blocks + self.temporal(blocks.transpose(1, 2)).transpose(1, 2))
        blocks = blocks * observed[..., None]
        return blocks.reshape(batch, 32, self.bins, self.width), observed.reshape(batch, 32, self.bins)


class StructureFusion(nn.Module):
    def __init__(self, hidden=256, width=64, bins=8):
        super().__init__()
        self.trajectory = VariableTrajectory(width, bins)
        self.summary = nn.Sequential(nn.Linear(26, width), nn.GELU(), nn.Linear(width, width))
        self.context = nn.Linear(hidden, width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        # This block is used in ALL variants: self-refinement in G1/G2,
        # trajectory-conditioned refinement in G3. No dormant extra parameters.
        self.refine = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))
        self.gate = nn.Linear(width, width)
        self.combine = nn.Sequential(nn.Linear(2*width, width), nn.GELU())
        self.output = nn.Linear(32*width, hidden)
        self.residual_scale = nn.Parameter(torch.tensor(.1))

    def encode(self, batch):
        return self.trajectory(batch['records'], batch['segments'], batch['counts'])

    def fuse(self, hidden, summary, temporal, mask, variant):
        assert variant in ('G1', 'G2', 'G3')
        s = self.summary(summary) + self.context(hidden.mean(0))[:, None]
        s = s + self.trajectory.variable.weight[None]
        query_input = masked_mean(temporal, mask, 2) if variant == 'G1' else s
        t = masked_attention(self.query(query_input), self.key(temporal), self.value(temporal), mask)
        observed = mask.any(-1)
        refinement_input = t if variant == 'G3' else s
        feedback = .1 * self.gate(s).sigmoid() * self.refine(refinement_input)
        if variant == 'G3':
            feedback = feedback * observed[..., None]
        s = s + feedback
        joined = self.combine(torch.cat((s, t), -1)).flatten(1)
        delta = self.residual_scale * self.output(joined)
        return hidden + delta[None]


class StructureNetwork(nn.Module):
    def __init__(self, base, phi_dim, variant, seed=42, width=64, bins=8):
        super().__init__()
        assert variant in ('G1', 'G2', 'G3')
        self.base, self.variant, self.phi_dim = base, variant, phi_dim
        self.members = base.members
        layers = [m for m in base.net.modules() if type(m).__name__ == 'WeightLayer']
        assert len(layers) == 4 and layers[-1].weight.shape[-2:] == (256, 2)
        with independent_init(seed, 'G/shared_structure'):
            self.fusion = StructureFusion(256, width, bins)
        self.info = dict(base.info, structure=variant, fusion='pre-output latent residual; one original native classifier',
                         variables=32, temporal_bins=bins, fusion_width=width,
                         direct_base_path=True, independent_probability_experts=0,
                         new_parameter_initialization='identical across G1/G2/G3', freeze=False)
        self._context = None
        self.fusion_enabled = True
        self._hook = layers[-1].register_forward_pre_hook(self._before_head)

    def _before_head(self, module, args):
        if not self.fusion_enabled:
            return None
        assert self._context is not None
        summary, temporal, mask = self._context
        tensors = dict(args[0])
        tensors['x_cont'] = self.fusion.fuse(tensors['x_cont'], summary, temporal, mask, self.variant)
        return (tensors,)

    def forward(self, batch):
        raw = batch['raw']
        if self.fusion_enabled:
            scaled = (raw-self.base.numerical.center)*self.base.numerical.factor
            scaled = torch.where(self.base.numerical.passthrough, raw,
                                 scaled / torch.sqrt(1+(scaled/3).square()))
            simple = scaled[:, self.phi_dim:self.phi_dim+352].reshape(-1, 32, 11)
            state = scaled[:, self.phi_dim+352:].reshape(-1, 32, 15)
            temporal, mask = self.fusion.encode(batch)
            self._context = (torch.cat((simple, state), -1), temporal, mask)
        try:
            return self.base(raw)
        finally:
            self._context = None

    def loss(self, logits, y):
        return self.base.loss(logits, y)

    def probabilities(self, batch):
        return self(batch).softmax(-1)[..., 1].mean(1, keepdim=True)
