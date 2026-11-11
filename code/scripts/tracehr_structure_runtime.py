"""Keep the E2 optimizer intact, jointly update new structure parameters."""
import copy
import torch
from scripts.tracehr_backbone_refine_runtime import Optimizer as BaseOptimizer


class Optimizer:
    def __init__(self, model, spec, lr_multiplier=1.):
        assert lr_multiplier == 1. and not spec['tuning']
        self.model, self.spec = model, spec
        self.base = BaseOptimizer(model.base, spec, 1.)
        self.fusion = torch.optim.AdamW(model.fusion.parameters(), lr=spec['fusion_lr'],
                                       betas=(.9,spec['fusion_beta2']), weight_decay=spec['fusion_weight_decay'])
        actual = [id(p) for g in self.base.inner.param_groups for p in g['params']]
        if self.base.adapter:
            actual += [id(p) for g in self.base.adapter.param_groups for p in g['params']]
        actual += [id(p) for g in self.fusion.param_groups for p in g['params']]
        assert len(actual) == len(set(actual)) and set(actual) == {id(p) for p in model.parameters()}
        self.schedule(0.)

    def schedule(self, epoch_float):
        self.base.schedule(epoch_float)
        factor = self.base.wrapped.get_hyper_values('lr',self.base.clock_group)/self.base.clock_base
        self.fusion.param_groups[0]['lr'] = self.spec['fusion_lr']*float(factor)

    def step(self):
        self.base.step()
        self.fusion.step()

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def state_dict(self):
        return dict(base=self.base.state_dict(),fusion=self.fusion.state_dict(),spec=self.spec)

    def load_state_dict(self, value):
        assert value['spec'] == self.spec
        self.base.load_state_dict(value['base']); self.fusion.load_state_dict(value['fusion'])
        self.schedule(self.base.fraction)

    def values(self):
        return dict(base=self.base.values(),fusion={k:self.fusion.param_groups[0][k] for k in ('lr','betas','weight_decay')},
                    protocol_search=False)


def gradients(model):
    result={}
    for name,p in model.named_parameters():
        assert p.requires_grad and p.grad is not None, ('Frozen/disconnected',name)
        assert torch.isfinite(p.grad).all(),name
        key=('trajectory' if name.startswith('fusion.trajectory.') else 'fusion' if name.startswith('fusion.')
             else 'numerical' if name.startswith('base.numerical.') else 'input_members' if name=='base.input_members' else 'backbone')
        r=result.setdefault(key,dict(tensors=0,norm_squared=0.,parameter_norm_squared=0.))
        r['tensors']+=1;r['norm_squared']+=float(p.grad.square().sum());r['parameter_norm_squared']+=float(p.detach().square().sum())
    assert all(r['norm_squared']>0 for r in result.values()),result
    return result


@torch.no_grad()
def diagnostics(model, data, initial, seed=42):
    # No extra neural fitting or labels; these diagnostics cannot affect selection.
    return dict(fusion_residual_scale=float(model.fusion.residual_scale),
                parameter_relative_L2_displacement={n:float((p-initial[n]).norm()/(initial[n].norm()+1e-12))
                                                   for n,p in model.named_parameters()},
                variant=model.variant,all_parameters_jointly_trainable=all(p.requires_grad for p in model.parameters()))
