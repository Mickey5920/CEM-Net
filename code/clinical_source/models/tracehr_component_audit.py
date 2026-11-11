"""Bounded X repairs and structural deletion of freshly initialized A/V/R modules."""
import math
import torch
from torch import nn
from models.tracehr_update_space import UpdateSpaceModel,configuration as original_configuration,initialization_context
from models.tracehr_native_joint import NativeJointModel,mixture_logit
from models.tracehr_native_preserve import PreserveFusion
from models.tracehr_joint_recovery import MemberLinear
from models.tracehr_scratch_fusion import FreshA
from models.tracehr_joint_redesign import make_realmlp
from models.tracehr_early_alignment import joint_objectives


def configuration(variant,base=None):
    assert variant in ('U4','U9','X0','X1')
    spec=original_configuration('U9' if variant.startswith('X') else variant,base)
    if variant.startswith('X'):
        spec.update(variant=variant,increment_rho=.1 if variant=='X0' else .5,
                    encoder_interaction_weight=.25 if variant=='X1' else 1.)
    return spec


class SubsetInteraction(nn.Module):
    """Keep the surviving pair of the original RANDOM head, including its scale."""
    def __init__(self,original,branches,rank=16):
        super().__init__();self.branches=branches;self.k=original.k
        for e in branches:setattr(self,e.lower()+'_low',getattr(original,e.lower()+'_low'))
        offset={'AV':0,'VR':rank,'AR':2*rank}[branches]
        # The original random readout slice avoids changing the surviving pair's
        # initialization merely because other feature pairs were removed.
        self.interaction=MemberLinear.__new__(MemberLinear);nn.Module.__init__(self.interaction)
        self.interaction.weight=nn.Parameter(original.interaction.weight[:,offset:offset+rank].detach().clone())
        self.interaction.bias=nn.Parameter(original.interaction.bias.detach().clone())
        self.eta_logit=original.eta_logit

    def forward(self,a,r,v):
        features={}
        if 'A' in self.branches:features['A']=self.a_low(a.flatten(1))[:,None]
        if 'R' in self.branches:features['R']=self.r_low(r)[:,None]
        if 'V' in self.branches:features['V']=self.v_low(v)
        # Retain the original RV multiplication order and the original 32 member
        # readouts even without V; no replacement encoder is introduced.
        left,right=('R','V') if self.branches=='VR' else tuple(self.branches)
        pair=(features[left]*features[right]).expand(-1,self.k,-1)
        return self.eta_logit.sigmoid()*self.interaction(pair).tanh()


class SubsetFusion(PreserveFusion):
    def __init__(self,spec,pos_weight,seed,branches):
        super().__init__(spec,pos_weight,seed);assert branches in ('AV','AR','VR')
        self.branches=branches
        if spec['interaction']!='none':
            self.tasks=nn.ModuleList([SubsetInteraction(h,branches,spec['rank']) for h in self.tasks])

    def forward(self,a,r,v,aux,observations=None):
        scores={};az=None
        if 'A' in self.branches:
            az=aux['A']-self.log_pos_weight;scores['A']=az[:,None].expand(-1,self.k,-1)
        if 'V' in self.branches:scores['V']=aux['V']
        if 'R' in self.branches:scores['R']=(aux['R'][...,1]-aux['R'][...,0])[:,None].expand(-1,self.k,-1)
        components=torch.stack([scores[e] for e in self.branches],-1)
        weights=torch.full_like(components[:,:1],-math.log(len(self.branches)))
        base=mixture_logit(components,weights)
        delta=torch.zeros_like(base) if self.spec['interaction']=='none' else self.spec['interaction_cap']*torch.stack([
            h(a,None if r is None else r[:,t],None if v is None else v[:,t]) for t,h in enumerate(self.tasks)],-1)
        self.loss_context=dict(base=base,delta=delta)
        self.last=dict(base=base.detach(),delta=delta.detach(),weights=weights.detach().exp(),
                       a_calibrated=None if az is None else az.detach().sigmoid())
        return base+delta


class ComponentModel(UpdateSpaceModel):
    def __init__(self,data,branches,variant,seed,r_config,v_config,base=None):
        assert branches in ('AVR','AV','AR','VR')
        if branches=='AVR':
            super().__init__(data,branches,'U9' if variant.startswith('X') else variant,seed,r_config,v_config,base)
            self.spec.update(configuration(variant,base));self.kind=variant
            return
        nn.Module.__init__(self)
        self.branches,self.kind,self.recipe,self.phi_dim=branches,variant,'native',data.phi_dim
        self.spec=configuration(variant,base);self.initialization,self.hp_managers={},[]
        for e in branches:
            data.initialization_progress(e)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+1000*('AVR'.index(e)+1))
                if e=='A':self.a=FreshA(data.phi_dim,58 if data.database=='mimic' else 53)
                elif e=='V':
                    from tabm import TabM
                    self.v=nn.ModuleList([TabM.make(n_num_features=data.input_dim,d_out=1,k=v_config['k'],
                        d_block=v_config['width'],n_blocks=v_config['depth'],dropout=v_config['dropout']) for _ in range(2)])
                else:
                    self.r,self.r_static=nn.ModuleList(),nn.ModuleList();raw=data.initialization_raw()
                    with initialization_context():
                        for t in range(2):
                            net,static,hp,info=make_realmlp(raw,data.targets['train']['y'][:,t],seed+3000+t,r_config,True)
                            self.r.append(net);self.r_static.append(static);self.hp_managers.append(hp);self.initialization[f'R{t}']=info
                    del raw
        self.fusion=SubsetFusion(self.spec,data.pos_weight,seed,branches)
        assert all(p.requires_grad for p in self.parameters())

    def forward(self,batch):
        if self.branches=='AVR':return NativeJointModel.forward(self,batch)
        raw,n=batch['raw'],len(batch['raw']);a=r=v=None;aux={}
        if 'A' in self.branches:a,aux['A']=self.a(raw[:,:self.phi_dim])
        if 'V' in self.branches:
            hh,zz=[],[]
            for net in self.v:
                found=[];hook=net.output.register_forward_pre_hook(lambda module,args:found.append(args[0]))
                try:pred=net(batch['v'])
                finally:hook.remove()
                hh.append(found[0]);zz.append(pred.squeeze(-1))
            v=torch.stack(hh,1);aux['V']=torch.stack(zz,-1)
        if 'R' in self.branches:
            hh,zz=[],[]
            for t,(net,static) in enumerate(zip(self.r,self.r_static)):
                x=static(dict(x_cont=raw[None],x_cat=torch.empty((1,n,0),device=raw.device,dtype=torch.long)))
                found=[];hook=net.get_submodule(self.initialization[f'R{t}']['prehead_module']).register_forward_pre_hook(lambda module,args:found.append(args[0]['x_cont']))
                try:pred=net(x)['x_cont']
                finally:hook.remove()
                hh.append(found[0][0]);zz.append(pred[0])
            r=torch.stack(hh,1);aux['R']=torch.stack(zz,1)
        return self.fusion(a,r,v,aux),aux
