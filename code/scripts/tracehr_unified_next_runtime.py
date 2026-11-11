"""Preserve native optimizers, hyperparameter factors and the 256-epoch clock."""
import copy
import math
from types import SimpleNamespace
import torch


class Optimizer:
    def __init__(self, model, spec, lr_multiplier=1.):
        self.model, self.spec = model, spec
        self.fraction = 0.
        self.base = spec['lr'] * lr_multiplier
        if spec['native']:
            from pytabkit.models.optim.optimizers import get_opt_class
            from pytabkit.models.training.scheduling import LearnerProgress
            self.learner = SimpleNamespace(progress=LearnerProgress())
            self.learner.progress.max_epochs = 256
            self.wrapped = get_opt_class(model.info['config'].get('opt', 'adam'))(
                [dict(params=[p], lr=.01) for p in model.parameters()], model.hp)
            self.inner = self.wrapped.opt
        else:
            self.wrapped = None
            self.inner = torch.optim.AdamW(model.parameters(), lr=self.base, weight_decay=spec['weight_decay'])
        actual = [id(p) for g in self.inner.param_groups for p in g['params']]
        assert len(actual) == len(set(actual)) and set(actual) == {id(p) for p in model.parameters()}
        self.schedule(0.)

    def schedule(self, epoch_float):
        self.fraction = float(epoch_float)
        if self.wrapped:
            self.learner.progress.epoch_float = self.fraction
            self.model.hp.update_hypers(self.learner)
        else:
            f = self.fraction
            factor = (f/2) if f < 2 else .05+.95*.5*(1+math.cos(math.pi*min(1., (f-2)/14)))
            for g in self.inner.param_groups:
                g['lr'] = self.base * factor

    def step(self):
        if self.wrapped:
            self.wrapped.step()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2., error_if_nonfinite=True)
            self.inner.step()

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def state_dict(self):
        return dict(inner=self.inner.state_dict(), epoch_float=self.fraction, spec=self.spec, base=self.base)

    def load_state_dict(self, value):
        assert value['spec'] == self.spec and value['base'] == self.base
        self.inner.load_state_dict(value['inner'])
        if self.wrapped:
            self.wrapped.param_groups = self.inner.param_groups
        self.schedule(value['epoch_float'])

    def values(self):
        if not self.wrapped:
            return dict(epoch_float=self.fraction, lr=self.inner.param_groups[0]['lr'])
        opt = self.wrapped
        return dict(epoch_float=self.fraction, native=True,
                    schedules=copy.deepcopy(self.model.hp.get_hyper_sched_values()),
                    base=copy.deepcopy(self.model.hp.hyper_base_values),
                    groups=[dict(name=n, lr=opt.get_hyper_values('lr', i), wd=opt.get_hyper_values('wd', i),
                                 factors=copy.deepcopy(p.hyper_factors), scope=str(p.context.scope))
                            for i, (n, p) in enumerate(self.model.named_parameters())])
