"""U experiments: native paths, R exposure clocks and optimizer-displacement fusion."""
import os
from contextlib import contextmanager
import torch
from torch import nn
from models.tracehr_native_joint import NativeJointModel
from models.tracehr_scratch_fusion import FreshA
from models.tracehr_joint_redesign import make_realmlp
from models.tracehr_native_preserve import PreserveFusion
from models.tracehr_early_alignment import joint_objectives

RECIPES=tuple('U'+str(i) for i in range(6))
INTERACTIONS=tuple('U'+str(i) for i in range(6,10))
INITIALIZATION_PROTOCOL='cpu_single_thread_mkl_compatible_v1'


@contextmanager
def initialization_context():
    # Set before importing torch in the launcher. No saved or trained weights.
    assert os.environ.get('MKL_CBWR')=='COMPATIBLE'
    previous=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_num_threads(previous)


def configuration(variant,base=None):
    assert variant in RECIPES+INTERACTIONS
    if variant in INTERACTIONS:
        assert base in (*RECIPES[:5],'T2')
    else:
        assert base is None or base==variant
        base=variant
    recipe='T2' if base=='U5' else base
    return dict(variant=variant,base=base,r_cycle=12. if recipe=='T2' else None,
        r_lr_floor=.05,r_regularization_horizon=256,slow_av=False,
        a_initial_lr=5e-4,v_initial_lr=5e-4,branch_clock=recipe=='U4',
        r_clock='accelerated' if recipe=='U1' else 'exposure' if recipe in ('U3','U4') else 'outer',
        r_extra_steps=2 if recipe in ('U2','U3','U4') else 0,r_extra_epochs=6.,
        gradient_mode='update_space' if variant in INTERACTIONS else 'pareto',
        interaction='legacy' if variant in (*INTERACTIONS,'U5') else 'none',
        protect_displacement=variant in ('U8','U9'),persist_base_state=variant in ('U7','U9'),
        increment_protection=False,increment_ramp=False,increment_rho=.5,
        rank=16,members=32,interaction_cap=1.,initial_interaction_alpha=.1,
        initialization_protocol=INITIALIZATION_PROTOCOL)


class UpdateSpaceModel(NativeJointModel):
    def __init__(self,data,branches,variant,seed,r_config,v_config,base=None):
        nn.Module.__init__(self)
        assert branches=='AVR'
        self.branches,self.kind,self.recipe,self.phi_dim=branches,variant,'native',data.phi_dim
        self.spec=configuration(variant,base)
        self.initialization,self.hp_managers={},[]
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
                    self.r,self.r_static=nn.ModuleList(),nn.ModuleList()
                    raw=data.initialization_raw()
                    with initialization_context():
                        for t in range(2):
                            net,static,hp,info=make_realmlp(raw,data.targets['train']['y'][:,t],seed+3000+t,r_config,True)
                            self.r.append(net);self.r_static.append(static);self.hp_managers.append(hp)
                            self.initialization[f'R{t}']=info
                    del raw
        self.fusion=PreserveFusion(self.spec,data.pos_weight,seed)
        assert all(p.requires_grad for p in self.parameters())

    def r_forward(self,batch):
        raw=batch['raw'];n=len(raw)
        return torch.stack([net(static(dict(x_cont=raw[None],
            x_cat=torch.empty((1,n,0),device=raw.device,dtype=torch.long))))['x_cont'][0]
            for net,static in zip(self.r,self.r_static)],1)
