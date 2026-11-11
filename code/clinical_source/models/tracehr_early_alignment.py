"""T0-T9: native A/V/R, early R schedules and live residual interactions."""
from torch.nn import functional as F
from models.tracehr_native_joint import NativeJointModel
from models.tracehr_native_preserve import PreserveFusion

RECIPES = tuple('T'+str(i) for i in range(6))
INTERACTIONS = tuple('T'+str(i) for i in range(6,10))


def configuration(variant, base=None):
    assert variant in RECIPES+INTERACTIONS
    if variant in INTERACTIONS:
        assert base in RECIPES
    else:
        assert base is None or base == variant
        base = variant
    index = int(base[1:])
    return dict(variant=variant, base=base, r_cycle=(None,8.,12.)[index % 3],
        r_lr_floor=.05, r_regularization_horizon=256, slow_av=index>=3,
        a_initial_lr=2e-4 if index>=3 else 5e-4, v_initial_lr=3.5e-4 if index>=3 else 5e-4,
        branch_clock=False, gradient_mode='interaction_increment' if variant in ('T7','T8','T9') else 'pareto',
        interaction='legacy' if variant in INTERACTIONS else 'none',
        increment_ramp=variant in ('T7','T9'), increment_protection=variant in ('T8','T9'),
        increment_rho=.5, ramp_initial=.1, ramp_epochs=6.,
        rank=16, members=32, interaction_cap=1., initial_interaction_alpha=.1)


class EarlyAlignmentModel(NativeJointModel):
    def __init__(self,data,branches,variant,seed,r_config,v_config,base=None):
        spec=configuration(variant,base)
        super().__init__(data,branches,'P0',seed,r_config,v_config)
        self.kind,self.spec=variant,spec
        self.fusion=PreserveFusion(spec,data.pos_weight,seed)
        assert all(p.requires_grad for p in self.parameters())


def joint_objectives(model,logits,y):
    target=y[:,None].expand_as(logits)
    full=F.binary_cross_entropy_with_logits(logits,target)
    base=full if model.spec['interaction']=='none' else F.binary_cross_entropy_with_logits(model.fusion.loss_context['base'],target)
    return dict(J=full,B=base)
