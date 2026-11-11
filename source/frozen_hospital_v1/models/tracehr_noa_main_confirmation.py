"""Two functional removals inside one freshly initialized C5_noT network."""
import torch
from torch.nn import functional as F
from models.tracehr_not_confirmation import NoTrajectoryConfirmation
from models.tracehr_unified_next import parameter_hash

VARIANTS = ('C5_noT_noA', 'C5_noT_noR')


class NoTrajectoryFunctional(NoTrajectoryConfirmation):
    def __init__(self, fresh_c1, variant, seed=42):
        assert variant in VARIANTS and seed in (42,43,44,45,46,47,48,49,50)
        super().__init__(fresh_c1, 'C5_noT', seed)
        original = {n: p.detach().clone() for n, p in self.named_parameters()}
        buffers = {n: b.detach().clone() for n, b in self.named_buffers()}
        reference_hash = parameter_hash(self)
        reference_count = sum(p.numel() for p in self.parameters())
        self.functional_variant = variant
        self.auxiliary_enabled = variant != 'C5_noT_noA'
        self.main_residual_enabled = variant != 'C5_noT_noR'
        if not self.auxiliary_enabled:
            del self.fusion.auxiliary_head
        for n, p in self.named_parameters():
            assert torch.equal(p, original[n]), n
        for n, b in self.named_buffers():
            assert torch.equal(b, buffers[n]), n
        removed = sorted(set(original) - set(dict(self.named_parameters())))
        assert removed == (['fusion.auxiliary_head.bias', 'fusion.auxiliary_head.weight']
                           if not self.auxiliary_enabled else [])
        self.info.update(structure=variant, reference_noT_initial_weights_sha256=reference_hash,
            reference_noT_parameter_count=reference_count,
            removed_from_noT=reference_count-sum(p.numel() for p in self.parameters()),
            removed_from_noT_names=removed, retained_noT_tensors_exact=True,
            retained_noT_buffers_exact=True, auxiliary_enabled=self.auxiliary_enabled,
            main_residual_enabled=self.main_residual_enabled, main_loss_coefficient=.5,
            auxiliary_loss_coefficient=.5 if self.auxiliary_enabled else 0.,
            loss='0.5*member_CE(z0)+0.5*CE(zA)' if self.auxiliary_enabled else '0.5*member_CE(z0)',
            auxiliary_head='Linear(256,2)' if self.auxiliary_enabled else None,
            auxiliary_view='zA' if self.auxiliary_enabled else None,
            auxiliary_backbone_dependency=False, trajectory_loaded_by_network=False,
            all_retained_parameters_jointly_trainable=True, stop_gradient=False, freeze=False)
        assert all(p.requires_grad for p in self.parameters())

    def _before_head(self, module, args):
        if not self.fusion_enabled:
            return None
        summary, temporal, mask, dual = self._context
        assert temporal is None and mask is None
        if not self.main_residual_enabled and not dual:
            return None
        f = self.fusion
        s = f.summary(summary) + f.trajectory.variable.weight[None]
        refined = s + .1 * f.gate(s).sigmoid() * f.refine(s)
        delta = f.residual_scale * f.output(f.combine(refined).flatten(1))
        if dual and self.auxiliary_enabled:
            self._aux_logits = f.auxiliary_head(delta)[:, None, :]
        if not self.main_residual_enabled:
            return None
        tensors = dict(args[0])
        tensors['x_cont'] = tensors['x_cont'] + delta[None]
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
        self._context = (torch.cat((simple, state), -1), None, None, dual)
        self._aux_logits = None
        try:
            logits = self.base(raw)
            auxiliary = self._aux_logits
        finally:
            self._context = None
            self._aux_logits = None
        assert logits.shape[1:] == (self.members, 2)
        if dual:
            if self.auxiliary_enabled:
                assert auxiliary.shape == (len(raw), 1, 2)
            else:
                assert auxiliary is None
            return logits, auxiliary
        return logits

    def path_losses(self, logits, y):
        main, auxiliary = logits
        assert (auxiliary is not None) == self.auxiliary_enabled
        return self.base.loss(main, y), (F.cross_entropy(auxiliary[:, 0], y.long())
                                        if self.auxiliary_enabled else None)

    def loss(self, logits, y):
        main, auxiliary = self.path_losses(logits, y)
        self._last_path_losses = torch.stack((main.detach(), auxiliary.detach()
                                             if auxiliary is not None else main.detach().new_zeros(())))
        return .5*(main+auxiliary) if auxiliary is not None else .5*main
