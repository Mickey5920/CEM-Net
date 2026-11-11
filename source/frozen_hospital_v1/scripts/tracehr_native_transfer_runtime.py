"""N12 optimizer control, with exactly one declared hyperparameter change per arm."""
import math
import torch
from models.tracehr_native_transfer import adapter_parameter
from scripts.tracehr_unified_next_runtime import Optimizer as PreviousOptimizer


class Optimizer(PreviousOptimizer):
    def __init__(self, model, spec, lr_multiplier=1.):
        self.model, self.spec, self.fraction = model, spec, 0.
        self.base, self.wrapped = spec['lr']*lr_multiplier, None
        if spec['adapter_lr_multiplier']==1. and spec['adapter_weight_decay']==spec['weight_decay']:
            groups = [dict(params=list(model.parameters()), lr_factor=1., weight_decay=spec['weight_decay'])]
        else:
            groups = [dict(params=[p for n,p in model.named_parameters() if adapter_parameter(n)==is_adapter],
                           lr_factor=spec['adapter_lr_multiplier'] if is_adapter else 1.,
                           weight_decay=spec['adapter_weight_decay'] if is_adapter else spec['weight_decay'])
                      for is_adapter in (False, True)]
        self.inner = torch.optim.AdamW(groups, lr=self.base, betas=(.9, spec['beta2']))
        actual = [id(p) for g in self.inner.param_groups for p in g['params']]
        assert len(actual)==len(set(actual)) and set(actual)=={id(p) for p in model.parameters()}
        self.schedule(0.)

    def schedule(self, epoch_float):
        self.fraction = float(epoch_float)
        f=self.fraction
        factor=(f/2) if f<2 else .05+.95*.5*(1+math.cos(math.pi*min(1.,(f-2)/14)))
        for g in self.inner.param_groups:
            g['lr']=self.base*factor*g['lr_factor']

    def values(self):
        return dict(epoch_float=self.fraction, groups=[dict(lr=g['lr'], weight_decay=g['weight_decay'],
                    betas=g['betas'], lr_factor=g['lr_factor']) for g in self.inner.param_groups])
