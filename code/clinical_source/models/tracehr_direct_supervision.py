"""C4: two supervised views of one C1 network; deployment is always z0."""
import torch
from torch import nn
from models.tracehr_structure import masked_attention


class DirectSupervisionNetwork(nn.Module):
    def __init__(self, fresh_c1):
        super().__init__()
        assert fresh_c1.variant == 'C1'
        # Move the freshly initialized modules, preserving native parameter metadata.
        # No checkpoint is read and no second classifier/backbone is constructed.
        fresh_c1._hook.remove()
        self.base, self.fusion = fresh_c1.base, fresh_c1.fusion
        self.phi_dim, self.members = fresh_c1.phi_dim, fresh_c1.members
        self.variant, self.fusion_enabled = 'C4', True
        self.info = dict(fresh_c1.info, structure='C4', main_view='z0', auxiliary_view='z1',
                         loss='0.5*member_CE(z0)+0.5*member_CE(z1)',
                         single_backbone_forward=True, shared_classifier=True,
                         learned_checkpoint_loaded=False, stop_gradient=False,
                         added_parameter_count=0, removed_from_C1=0)
        self._context = None
        layers = [m for m in self.base.net.modules() if type(m).__name__ == 'WeightLayer']
        assert len(layers) == 4 and layers[-1].weight.shape[-2:] == (256, 2)
        self._hook = layers[-1].register_forward_pre_hook(self._before_head)

    def _before_head(self, module, args):
        if not self.fusion_enabled:
            return None
        summary, temporal, mask, dual = self._context
        f = self.fusion
        s = f.summary(summary) + f.trajectory.variable.weight[None]
        t = masked_attention(f.query(s), f.key(temporal), f.value(temporal), mask) if dual else None
        refined = s + .1 * f.gate(s).sigmoid() * f.refine(s)
        delta0 = f.residual_scale * f.output(f.combine(torch.cat((refined, torch.zeros_like(refined)), -1)).flatten(1))
        tensors = dict(args[0])
        h = tensors['x_cont']
        if dual:
            delta1 = f.residual_scale * f.output(f.combine(torch.cat((refined, t), -1)).flatten(1))
            # The same hidden states and dropout draw feed both paths. Only the
            # final shared affine classifier sees the concatenated member axis.
            tensors['x_cont'] = torch.cat((h + delta0[None], h + delta1[None]), 0)
        else:
            tensors['x_cont'] = h + delta0[None]
        return (tensors,)

    def _forward(self, batch, dual):
        raw = batch['raw']
        if not self.fusion_enabled:
            assert not dual
            return self.base(raw)
        scaled = (raw-self.base.numerical.center)*self.base.numerical.factor
        scaled = torch.where(self.base.numerical.passthrough, raw,
                             scaled / torch.sqrt(1+(scaled/3).square()))
        simple = scaled[:, self.phi_dim:self.phi_dim+352].reshape(-1, 32, 11)
        state = scaled[:, self.phi_dim+352:].reshape(-1, 32, 15)
        temporal, mask = self.fusion.encode(batch) if dual else (None, None)
        self._context = (torch.cat((simple, state), -1), temporal, mask, dual)
        try:
            logits = self.base(raw)
        finally:
            self._context = None
        if dual:
            assert logits.shape[1:] == (2*self.members, 2)
            return logits[:, :self.members], logits[:, self.members:]
        assert logits.shape[1:] == (self.members, 2)
        return logits

    def forward(self, batch):
        return self._forward(batch, dual=self.training and self.fusion_enabled)

    def forward_views(self, batch):
        return self._forward(batch, dual=True)

    def loss(self, logits, y):
        assert isinstance(logits, tuple) and len(logits) == 2
        loss0, loss1 = (self.base.loss(z, y) for z in logits)
        self._last_path_losses = torch.stack((loss0.detach(), loss1.detach()))
        return .5*(loss0+loss1)

    def probabilities(self, batch):
        # Even if called while training, the public prediction is never z1.
        return self._forward(batch, dual=False).softmax(-1)[..., 1].mean(1, keepdim=True)
