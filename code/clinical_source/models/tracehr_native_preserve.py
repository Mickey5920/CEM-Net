"""Fresh native A/V/R paths with independently selected optimization and interaction."""
import math
import torch
from torch.nn import functional as F
from models.tracehr_native_joint import NativeJointModel, NativeFusion, mixture_logit


def configuration(variant, base=None):
    assert variant in ('Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5')
    if variant in ('Q4', 'Q5'):
        assert base in ('Q0', 'Q1', 'Q2', 'Q3')
    else:
        assert base is None or base == variant
        base = variant
    return dict(variant=variant, base=base, branch_clock=base in ('Q1', 'Q3'),
                gradient_mode='native_protected' if base in ('Q2', 'Q3') else 'pareto',
                rho=.5, interaction='bounded' if variant == 'Q5' else 'legacy' if variant == 'Q4' else 'none',
                rank=16, members=32, interaction_cap=.05 if variant == 'Q5' else 1.,
                initial_interaction_alpha=.01 if variant == 'Q5' else .1,
                beta_start=.1, beta_end=.5, beta_ramp_epochs=6., delta_penalty=1.)


class PreserveFusion(NativeFusion):
    def __init__(self, spec, pos_weight, seed):
        super().__init__('P0' if spec['interaction'] == 'none' else 'P2', pos_weight, 32, seed)
        self.spec, self.epoch_float = spec, 0.
        if spec['interaction'] == 'bounded':
            for head in self.tasks:
                # Positive initial amplitude keeps all interaction layers learning from step one.
                with torch.no_grad():
                    head.eta_logit.fill_(math.log(.2/.8))

    def amplitudes(self):
        return [] if self.spec['interaction'] == 'none' else [
            float(self.spec['interaction_cap'] * h.eta_logit.sigmoid()) for h in self.tasks]

    def forward(self, a, r, v, aux, observations=None):
        az = aux['A'] - self.log_pos_weight
        vz = aux['V']
        rz = aux['R'][..., 1] - aux['R'][..., 0]
        assert vz.shape[1:] == (self.k, 2)
        components = torch.stack([az[:, None].expand_as(vz), vz, rz[:, None].expand_as(vz)], -1)
        weights = torch.full_like(components[:, :1], -math.log(3.))
        base = mixture_logit(components, weights)
        if self.spec['interaction'] == 'none':
            delta = torch.zeros_like(base)
        else:
            delta = self.spec['interaction_cap'] * torch.stack([
                head(a, r[:, t], v[:, t]) for t, head in enumerate(self.tasks)], -1)
        # Live tensors are required by Q5's base supervision; diagnostics alone are detached.
        self.loss_context = dict(base=base, delta=delta)
        self.last = dict(base=base.detach(), delta=delta.detach(), weights=weights.detach().exp(),
                         a_calibrated=az.detach().sigmoid())
        return base + delta


class NativePreserveModel(NativeJointModel):
    def __init__(self, data, branches, variant, seed, r_config, v_config, base=None):
        spec = configuration(variant, base)
        super().__init__(data, branches, 'P0', seed, r_config, v_config)
        self.kind, self.spec = variant, spec
        self.fusion = PreserveFusion(spec, data.pos_weight, seed)
        assert all(p.requires_grad for p in self.parameters())


def joint_objectives(model, logits, y):
    target = y[:, None].expand_as(logits)
    full = F.binary_cross_entropy_with_logits(logits, target)
    if model.kind != 'Q5':
        return dict(J=full)
    context = model.fusion.loss_context
    base = F.binary_cross_entropy_with_logits(context['base'], target)
    beta = .1 + .4 * min(max(model.fusion.epoch_float, 0.) / 6., 1.)
    penalty = context['delta'].square().mean()
    return dict(J=(1-beta)*base + beta*full + penalty, base=base, full=full,
                delta_l2=penalty, beta=logits.new_tensor(beta))
