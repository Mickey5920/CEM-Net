"""W0-W8: fresh native encoders, learned task weights and conditional mixing."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.tracehr_update_space import UpdateSpaceModel, configuration as u_configuration
from models.tracehr_native_joint import BoundedInteraction, mixture_logit

RECIPES = tuple('W'+str(i) for i in range(5))
INTERACTIONS = tuple('W'+str(i) for i in range(5,9))


def configuration(variant, base=None):
    assert variant in RECIPES+INTERACTIONS
    if variant in RECIPES:
        assert base is None or base == variant
        recipe = variant
    else:
        assert base in RECIPES+('U4',)
        recipe = base
    spec = u_configuration('U4')
    spec.update(variant=variant, base=recipe, weighted=recipe!='U4',
        weak_joint=recipe in ('W1','W2','W3','W4'), a_unweighted=recipe=='W4',
        r_clock='outer' if recipe=='W3' else 'exposure', r_split_clock=recipe=='W2',
        gradient_mode='weighted_update' if variant in INTERACTIONS else 'weighted_pareto',
        interaction={'W5':'legacy','W6':'linear','W7':'conditional','W8':'conditional'}.get(variant,'none'),
        persist_base_state=variant in INTERACTIONS, protect_displacement=variant=='W8',
        increment_rho=.1, joint_start=.05, joint_end=.25, ramp_epochs=8.,
        beta_start=.1, beta_end=.5, conditional_start=.1, conditional_end=.5,
        residual_penalty=.1 if variant=='W6' else 0., mix_lr=.003,
        head_lr=.0003, head_weight_decay=.001, small_readout_std=.01)
    return spec


def ramp(spec, start, end, epoch):
    return spec[start]+(spec[end]-spec[start])*min(max(float(epoch),0.)/spec['ramp_epochs'],1.)


class SmallPairReadout(nn.Module):
    def __init__(self, outputs, k=32, rank=16):
        super().__init__()
        self.a_low=nn.Sequential(nn.Linear(192,rank),nn.LayerNorm(rank))
        self.r_low=nn.Sequential(nn.Linear(256,rank),nn.LayerNorm(rank))
        self.v_low=nn.Sequential(nn.Linear(256,rank),nn.LayerNorm(rank))
        self.weight=nn.Parameter(torch.randn(k,3*rank,outputs)*(.01/math.sqrt(3*rank)))
        self.bias=nn.Parameter(torch.randn(k,outputs)*.001)

    def forward(self,a,r,v):
        aa=self.a_low(a.flatten(1))[:,None];rr=self.r_low(r)[:,None];vv=self.v_low(v)
        pair=torch.cat([aa*vv,rr*vv,(aa*rr).expand(-1,v.shape[1],-1)],-1)
        return torch.einsum('bkd,kdo->bko',pair,self.weight)+self.bias


class WeightedFusion(nn.Module):
    def __init__(self,spec,pos_weight,seed):
        super().__init__();self.spec=spec;self.k=32;self.epoch_float=0.
        offset=torch.zeros_like(torch.as_tensor(pos_weight)) if spec['a_unweighted'] else torch.as_tensor(pos_weight).log()
        self.register_buffer('log_pos_weight',offset.detach().cpu())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+6000)
            if spec['weighted']:self.theta=nn.Parameter(torch.randn(2,3)*.01)
            else:self.register_buffer('theta',torch.zeros(2,3))
        if spec['interaction']!='none':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+5000)
                self.tasks=nn.ModuleList([BoundedInteraction(32,192,16) if spec['interaction']=='legacy'
                    else SmallPairReadout(3 if spec['interaction']=='conditional' else 1) for _ in range(2)])

    def base_parameters(self):
        return [self.theta] if isinstance(self.theta,nn.Parameter) else []

    def interaction_parameters(self):
        return list(self.tasks.parameters()) if hasattr(self,'tasks') else []

    def amplitudes(self):
        return [float(h.eta_logit.sigmoid()) for h in self.tasks] if self.spec['interaction']=='legacy' else []

    def forward(self,a,r,v,aux,observations=None):
        az=aux['A']-self.log_pos_weight;vz=aux['V'];rz=aux['R'][...,1]-aux['R'][...,0]
        components=torch.stack([az[:,None].expand_as(vz),vz,rz[:,None].expand_as(vz)],-1)
        weights=self.theta.log_softmax(-1)[None,None].expand_as(components)
        base=mixture_logit(components,weights)
        kind=self.spec['interaction']
        if kind=='none':full=base
        elif kind=='conditional':
            adjustment=torch.stack([h(a,r[:,t],v[:,t]) for t,h in enumerate(self.tasks)],2)
            cap=ramp(self.spec,'conditional_start','conditional_end',self.epoch_float)
            weights=(self.theta[None,None]+cap*adjustment.tanh()).log_softmax(-1)
            full=mixture_logit(components,weights)
        else:
            delta=torch.stack([h(a,r[:,t],v[:,t]) for t,h in enumerate(self.tasks)],-1)
            if kind=='linear':delta=delta.squeeze(-2)
            full=base+delta
        delta=full-base
        self.loss_context=dict(base=base,delta=delta)
        self.last=dict(base=base.detach(),delta=delta.detach(),weights=weights.detach().exp(),a_calibrated=az.detach().sigmoid())
        return full


class WeightedFusionModel(UpdateSpaceModel):
    def __init__(self,data,branches,variant,seed,r_config,v_config,base=None):
        # The U4 constructor generates paired random encoders; no checkpoint is read.
        super().__init__(data,branches,'U4',seed,r_config,v_config)
        self.kind=variant;self.spec=configuration(variant,base)
        self.fusion=WeightedFusion(self.spec,data.pos_weight,seed)
        assert all(p.requires_grad for p in self.parameters())


def joint_objectives(model,logits,y):
    target=y[:,None].expand_as(logits)
    full=F.binary_cross_entropy_with_logits(logits,target)
    base=F.binary_cross_entropy_with_logits(model.fusion.loss_context['base'],target)
    beta=ramp(model.spec,'beta_start','beta_end',model.fusion.epoch_float) if model.spec['interaction']!='none' else 1.
    penalty=model.fusion.loss_context['delta'].square().mean()*model.spec['residual_penalty']
    return dict(J=(1-beta)*base+beta*full+penalty,B=base,F=full,penalty=penalty,beta=logits.new_tensor(beta))
