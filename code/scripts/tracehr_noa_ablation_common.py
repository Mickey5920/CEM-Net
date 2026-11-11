"""Locked first batch: three deletions, three seeds, two MIMIC endpoints."""
from pathlib import Path
import torch
from scripts.tracehr_not_stability_common import model_for as parent_model, protocol as parent_protocol
from scripts.tracehr_structure_common import Data, stopping
from scripts.tracehr_centered_native_common import model_for as e_model
from scripts.tracehr_structure_runtime import Optimizer as PreviousOptimizer
from scripts.tracehr_backbone_refine_runtime import Optimizer as BaseOptimizer
from models.tracehr_structure import StructureNetwork
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_noa_ablation import NoAAblation, VARIANTS
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_recovery_queue import read

RUN='tracehr_noa_ablation_20261019'
MATRIX=tuple((v,e,s) for s in (43,44,45) for v in VARIANTS for e in (0,1))
PROTOCOL_FIELDS={'variant','structure','description','ablated_component','ablation_implementation','seed'}

def expected():return read(Path(__file__).resolve().parents[1]/'work'/RUN/'expected.json')

def protocol(variant, database='mimic', seed=43):
    assert database=='mimic' and variant in VARIANTS and seed in (43,44,45)
    spec=parent_protocol('C5_noT_noA', database, seed)
    spec.update(variant=variant,structure=variant,ablated_component=variant[-1],
        description='single deletion inside noA; fixed z0; 0.5 main CE; no auxiliary loss',
        ablation_implementation={'P':'zero slots preserve native width and retained initial tensors',
          'M':'identity input modulation; eight internal members and normalization preserved',
          'R':'delete summary-to-main residual; prune the now disconnected summary branch'}[variant[-1]])
    return spec

def model_for(data, preparation, variant, endpoint, seed):
    base=e_model(data,preparation,'E2',endpoint,seed)
    base_hash=parameter_hash(base)
    c1=remove_hidden_context(StructureNetwork(base,data.record['phi_dim'],'G2',seed),seed)
    model=NoAAblation(c1,variant,seed)
    model.info['seed_route']=dict(seed=seed,native_initialization_seed=seed,
        periodic_namespace_seed=seed,input_members_namespace_seed=seed,fusion_namespace_seed=seed,
        auxiliary_head_namespace_seed=seed,training_order_seed=seed,training_torch_seed=seed+10000,
        base_initial_weights_sha256=base_hash)
    return model

class Optimizer(PreviousOptimizer):
    def __init__(self, model, spec, lr_multiplier=1.):
        if model.main_residual_enabled:
            super().__init__(model,spec,lr_multiplier)
            return
        assert lr_multiplier==1. and not spec['tuning']
        self.model,self.spec=model,spec
        self.base=BaseOptimizer(model.base,spec,1.)
        self.fusion=None
        actual=[id(p) for g in self.base.inner.param_groups for p in g['params']]
        if self.base.adapter: actual += [id(p) for g in self.base.adapter.param_groups for p in g['params']]
        assert len(actual)==len(set(actual)) and set(actual)=={id(p) for p in model.parameters()}
        self.schedule(0.)

    def schedule(self, epoch_float):
        if self.fusion is not None:return super().schedule(epoch_float)
        self.base.schedule(epoch_float)
    def step(self):
        if self.fusion is not None:return super().step()
        self.base.step()
    def state_dict(self):
        return dict(base=self.base.state_dict(),fusion=self.fusion.state_dict() if self.fusion else None,spec=self.spec)
    def load_state_dict(self,value):
        assert value['spec']==self.spec and (value['fusion'] is None)==(self.fusion is None)
        self.base.load_state_dict(value['base'])
        if self.fusion:self.fusion.load_state_dict(value['fusion'])
        self.schedule(self.base.fraction)
    def values(self):
        return dict(base=self.base.values(),fusion={k:self.fusion.param_groups[0][k] for k in ('lr','betas','weight_decay')} if self.fusion else None,protocol_search=False)

@torch.no_grad()
def diagnostics(model,data,initial,seed=43):
    return dict(fusion_residual_scale=float(model.fusion.residual_scale) if model.main_residual_enabled else None,
        parameter_relative_L2_displacement={n:float((p-initial[n]).norm()/(initial[n].norm()+1e-12)) for n,p in model.named_parameters()},
        variant=model.variant,all_parameters_jointly_trainable=all(p.requires_grad for p in model.parameters()))
