"""Native optimizer for the shared backbone; small simultaneous input-adapter updates."""
import copy
from types import SimpleNamespace
import torch
from scripts.tracehr_native_transfer_runtime import Optimizer as ResidualOptimizer


class Optimizer:
    def __init__(self, model, spec, lr_multiplier=1.):
        self.model,self.spec,self.fraction = model,spec,0.
        self.base = spec['lr']*lr_multiplier
        self.residual = None
        if not spec['native']:
            residual_spec = dict(spec,adapter_lr_multiplier=spec['adapter_lr']/(spec['lr']))
            self.residual = ResidualOptimizer(model,residual_spec,lr_multiplier)
            return
        from pytabkit.models.optim.optimizers import get_opt_class
        from pytabkit.models.training.scheduling import LearnerProgress
        self.native_params = [(n,p) for n,p in model.named_parameters() if n!='input_members']
        assert all(hasattr(p,'context') for _,p in self.native_params)
        self.learner = SimpleNamespace(progress=LearnerProgress())
        self.learner.progress.max_epochs = 256
        self.wrapped = get_opt_class(model.info['config'].get('opt','adam'))(
            [dict(params=[p],lr=.01) for _,p in self.native_params],model.hp)
        self.inner = self.wrapped.opt
        self.clock_group = next(i for i,(n,p) in enumerate(self.native_params)
                                if n.startswith('net.') and p.ndim>=3 and p.shape[-1]==256)
        p = self.native_params[self.clock_group][1]
        self.clock_base = self.base*p.hyper_factors.get('lr',1.)
        self.adapter = (torch.optim.AdamW([model.input_members],lr=spec['adapter_lr']*lr_multiplier,
                         betas=(.9,spec['adapter_beta2']),weight_decay=spec['adapter_weight_decay'])
                        if model.input_members is not None else None)
        self.adapter_base = spec['adapter_lr']*lr_multiplier
        actual = [id(p) for g in self.inner.param_groups for p in g['params']]
        if self.adapter:
            actual += [id(p) for g in self.adapter.param_groups for p in g['params']]
        assert len(actual)==len(set(actual)) and set(actual)=={id(p) for p in model.parameters()}
        self.schedule(0.)

    def schedule(self, epoch_float):
        self.fraction = float(epoch_float)
        if self.residual:
            self.residual.schedule(epoch_float)
            return
        self.learner.progress.epoch_float = self.fraction
        self.model.hp.update_hypers(self.learner)
        if self.adapter:
            factor = self.wrapped.get_hyper_values('lr',self.clock_group)/self.clock_base
            self.adapter.param_groups[0]['lr'] = self.adapter_base*float(factor)

    def step(self):
        if self.residual:
            self.residual.step()
        else:
            self.wrapped.step()
            if self.adapter:
                self.adapter.step()

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def state_dict(self):
        if self.residual:
            return dict(residual=self.residual.state_dict(),spec=self.spec)
        return dict(inner=self.inner.state_dict(),adapter=self.adapter.state_dict() if self.adapter else None,
                    epoch_float=self.fraction,spec=self.spec,base=self.base)

    def load_state_dict(self,value):
        assert value['spec']==self.spec
        if self.residual:
            self.residual.load_state_dict(value['residual']);return
        assert value['base']==self.base
        self.inner.load_state_dict(value['inner']);self.wrapped.param_groups=self.inner.param_groups
        if self.adapter:
            self.adapter.load_state_dict(value['adapter'])
        else:
            assert value['adapter'] is None
        self.schedule(value['epoch_float'])

    def values(self):
        if self.residual:
            return self.residual.values()
        return dict(epoch_float=self.fraction,native=True,clock_reference_epochs=256,
                    schedules=copy.deepcopy(self.model.hp.get_hyper_sched_values()),
                    base=copy.deepcopy(self.model.hp.hyper_base_values),
                    groups=[dict(name=n,lr=self.wrapped.get_hyper_values('lr',i),wd=self.wrapped.get_hyper_values('wd',i),
                                 factors=copy.deepcopy(p.hyper_factors),scope=str(p.context.scope))
                            for i,(n,p) in enumerate(self.native_params)],
                    input_adapter=None if self.adapter is None else {k:self.adapter.param_groups[0][k] for k in ('lr','betas','weight_decay')})
