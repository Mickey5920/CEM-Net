"""Y experiments: native optimizer reference and matched, live multi-view fusion."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.tracehr_component_audit import ComponentModel, SubsetFusion
from models.tracehr_native_preserve import PreserveFusion
from models.tracehr_update_space import configuration as u_configuration
from models.tracehr_native_joint import mixture_logit

RECIPES = tuple('Y'+str(i) for i in range(6))
INTERACTIONS = tuple('Y'+str(i) for i in range(6,10))


def configuration(variant, base=None, no_interaction=False):
    assert variant in RECIPES+INTERACTIONS
    if variant in RECIPES:
        assert base is None or base == variant
        recipe = 'Y'+str(int(variant[1:])//2*2)
    else:
        assert base in RECIPES
        recipe = 'Y'+str(int(base[1:])//2*2)
    spec = u_configuration('U9', 'U4')
    interaction = 'none' if variant in ('Y0','Y2','Y4','Y9') or no_interaction else (
        'pooled' if variant in ('Y7','Y8') else 'legacy')
    spec.update(variant=variant, base=base, native_recipe=recipe,
        native_anchor=True, a_unweighted=recipe in ('Y2','Y4'),
        r_extra_steps=1 if recipe=='Y4' else 2, interaction=interaction,
        aggregate_fusion_loss=variant in INTERACTIONS, no_interaction=bool(no_interaction),
        sam_radius=.02 if variant=='Y8' and interaction!='none' else 0.,
        optimizer_reference='native_only', persist_base_state=True,
        encoder_interaction_weight=1., gradient_mode='update_space')
    return spec


def aggregate_logit(logits):
    """Logit of the mean probability, evaluated without probability clipping."""
    return torch.logsumexp(F.logsigmoid(logits),1)-torch.logsumexp(F.logsigmoid(-logits),1)


class PooledPairHead(nn.Module):
    def __init__(self, rank=16):
        super().__init__()
        self.branches='AVR';self.rank=rank
        self.a_low=nn.Sequential(nn.Linear(192,rank),nn.LayerNorm(rank))
        self.r_low=nn.Sequential(nn.Linear(256,rank),nn.LayerNorm(rank))
        self.v_low=nn.Sequential(nn.Linear(256,rank),nn.LayerNorm(rank))
        self.v_pool=nn.Sequential(nn.Linear(2*rank,rank),nn.LayerNorm(rank))
        self.readout=nn.Linear(3*rank,1)
        self.eta_logit=nn.Parameter(torch.tensor(math.log(.1/.9)))

    def restrict(self, branches):
        self.branches=branches
        for e in 'AVR':
            if e not in branches:delattr(self,e.lower()+'_low')
        if 'V' not in branches:del self.v_pool
        keep=[i for i,pair in enumerate(('AV','VR','AR')) if all(e in branches for e in pair)]
        indices=[i*self.rank+j for i in keep for j in range(self.rank)]
        linear=nn.Linear.__new__(nn.Linear);nn.Module.__init__(linear)
        linear.in_features=len(indices);linear.out_features=1
        linear.weight=nn.Parameter(self.readout.weight[:,indices].detach().clone())
        linear.bias=nn.Parameter(self.readout.bias.detach().clone())
        self.readout=linear

    def forward(self,a,r,v):
        h={}
        if 'A' in self.branches:h['A']=self.a_low(a.flatten(1))
        if 'R' in self.branches:h['R']=self.r_low(r)
        if 'V' in self.branches:
            vv=self.v_low(v)
            h['V']=self.v_pool(torch.cat([vv.mean(1),vv.var(1,unbiased=False)],-1))
        pairs=[h[x]*h[y] for x,y in ('AV','VR','AR') if x in h and y in h]
        return self.eta_logit.sigmoid()*self.readout(torch.cat(pairs,-1)).tanh()


class PooledFusion(nn.Module):
    def __init__(self,spec,pos_weight,seed,branches):
        super().__init__();self.spec=spec;self.branches=branches;self.k=1;self.epoch_float=0.
        offset=torch.zeros_like(torch.as_tensor(pos_weight)) if spec['a_unweighted'] else torch.as_tensor(pos_weight).log()
        self.register_buffer('log_pos_weight',offset.detach().cpu())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+5000)
            self.tasks=nn.ModuleList([PooledPairHead(spec['rank']) for _ in range(2)])
        if branches!='AVR':
            for head in self.tasks:head.restrict(branches)

    def amplitudes(self):
        return [float(self.spec['interaction_cap']*h.eta_logit.sigmoid()) for h in self.tasks]

    def forward(self,a,r,v,aux,observations=None):
        scores={};az=None
        if 'A' in self.branches:az=aux['A']-self.log_pos_weight;scores['A']=az
        if 'V' in self.branches:scores['V']=aggregate_logit(aux['V'])
        if 'R' in self.branches:scores['R']=aux['R'][...,1]-aux['R'][...,0]
        components=torch.stack([scores[e] for e in self.branches],-1)[:,None]
        weights=torch.full_like(components,-math.log(len(self.branches)))
        base=mixture_logit(components,weights)
        delta=self.spec['interaction_cap']*torch.stack([
            h(a,None if r is None else r[:,t],None if v is None else v[:,t]) for t,h in enumerate(self.tasks)],-1)
        self.loss_context=dict(base=base,delta=delta)
        self.last=dict(base=base.detach(),delta=delta.detach(),weights=weights.detach().exp(),
                       a_calibrated=None if az is None else az.detach().sigmoid())
        return base+delta


class NativeAnchorModel(ComponentModel):
    def __init__(self,data,branches,variant,seed,r_config,v_config,base=None,no_interaction=False):
        # This constructor only generates random weights, with the audited branch seeds.
        super().__init__(data,branches,'U4',seed,r_config,v_config)
        self.kind=variant;self.spec=configuration(variant,base,no_interaction)
        if self.spec['interaction']=='pooled':
            self.fusion=PooledFusion(self.spec,data.pos_weight,seed,branches)
        else:
            cls=PreserveFusion if branches=='AVR' else SubsetFusion
            self.fusion=cls(self.spec,data.pos_weight,seed,*(() if branches=='AVR' else (branches,)))
            if self.spec['a_unweighted']:self.fusion.log_pos_weight.zero_()
        assert all(p.requires_grad for p in self.parameters())


def joint_objectives(model,logits,y):
    base=model.fusion.loss_context['base']
    if model.spec['aggregate_fusion_loss']:
        return dict(J=F.binary_cross_entropy_with_logits(aggregate_logit(logits),y),
                    B=F.binary_cross_entropy_with_logits(aggregate_logit(base),y))
    return dict(J=F.binary_cross_entropy_with_logits(logits,y[:,None].expand_as(logits)),
                B=F.binary_cross_entropy_with_logits(base,y[:,None].expand_as(base)))
