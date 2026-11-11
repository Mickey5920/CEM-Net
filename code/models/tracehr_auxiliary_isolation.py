"""C5: unchanged z0; a train-only fusion head has no dependency on the backbone."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from models.tracehr_direct_supervision import DirectSupervisionNetwork
from models.tracehr_structure import masked_attention
from models.tracehr_unified import independent_init
from models.tracehr_unified_next import parameter_hash


def retained_parameter_hash(model):
    value = hashlib.sha256()
    for name, p in model.named_parameters():
        if name.startswith('fusion.auxiliary_head.'):
            continue
        value.update(name.encode())
        value.update(p.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


class AuxiliaryIsolationNetwork(DirectSupervisionNetwork):
    def __init__(self, fresh_c1, seed=42):
        before = parameter_hash(fresh_c1)
        count = sum(p.numel() for p in fresh_c1.parameters())
        super().__init__(fresh_c1)
        # Same default Linear initialization as the existing fusion layers;
        # a separate RNG context leaves every retained random tensor unchanged.
        with independent_init(seed, 'C5/auxiliary_head'):
            self.fusion.auxiliary_head = nn.Linear(256, 2)
        self.variant = 'C5'
        self._aux_logits = None
        assert retained_parameter_hash(self) == before
        assert sum(p.numel() for p in self.parameters()) == count + 514
        self.info.update(structure='C5', auxiliary_view='zA', shared_classifier=False,
                         auxiliary_head='Linear(256,2), existing fusion optimizer',
                         auxiliary_initialization='independent_init(seed, C5/auxiliary_head); Linear default',
                         loss='0.5*member_CE(z0)+0.5*CE(zA)',
                         added_parameter_count=514, auxiliary_backbone_dependency=False,
                         random_C1_before_wrapping_sha256=before,
                         retained_random_parameters_sha256=retained_parameter_hash(self),
                         retained_parameter_count=count, freeze=False)

    def _before_head(self, module, args):
        if not self.fusion_enabled:
            return None
        summary, temporal, mask, dual = self._context
        f = self.fusion
        s = f.summary(summary) + f.trajectory.variable.weight[None]
        refined = s + .1 * f.gate(s).sigmoid() * f.refine(s)
        delta0 = f.residual_scale * f.output(
            f.combine(torch.cat((refined, torch.zeros_like(refined)), -1)).flatten(1))
        if dual:
            t = masked_attention(f.query(s), f.key(temporal), f.value(temporal), mask)
            delta1 = f.residual_scale * f.output(f.combine(torch.cat((refined, t), -1)).flatten(1))
            self._aux_logits = f.auxiliary_head(delta1)[:, None, :]
        tensors = dict(args[0])
        tensors['x_cont'] = tensors['x_cont'] + delta0[None]
        return (tensors,)

    def _forward(self, batch, dual):
        raw = batch['raw']
        if not self.fusion_enabled:
            assert not dual
            return self.base(raw)
        # Only fixed train-fitted buffers feed the summary, not learned base features.
        scaled = (raw-self.base.numerical.center)*self.base.numerical.factor
        scaled = torch.where(self.base.numerical.passthrough, raw,
                             scaled / torch.sqrt(1+(scaled/3).square()))
        simple = scaled[:, self.phi_dim:self.phi_dim+352].reshape(-1, 32, 11)
        state = scaled[:, self.phi_dim+352:].reshape(-1, 32, 15)
        temporal, mask = self.fusion.encode(batch) if dual else (None, None)
        self._context = (torch.cat((simple, state), -1), temporal, mask, dual)
        self._aux_logits = None
        try:
            logits = self.base(raw)
            auxiliary = self._aux_logits
        finally:
            self._context = None
            self._aux_logits = None
        assert logits.shape[1:] == (self.members, 2)
        if dual:
            assert auxiliary.shape == (len(raw), 1, 2)
            return logits, auxiliary
        return logits

    def path_losses(self, logits, y):
        main, auxiliary = logits
        return self.base.loss(main, y), F.cross_entropy(auxiliary[:, 0], y.long())

    def loss(self, logits, y):
        losses = self.path_losses(logits, y)
        self._last_path_losses = torch.stack(tuple(x.detach() for x in losses))
        return .5 * (losses[0] + losses[1])
