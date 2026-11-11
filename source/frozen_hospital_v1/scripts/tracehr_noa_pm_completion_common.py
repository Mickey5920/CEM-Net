"""Original noA P/M deletion completion: eight fits, retaining R and the fixed recipe."""
from pathlib import Path
import torch
from scripts.tracehr_not_stability_common import model_for as parent_model, protocol as parent_protocol
from scripts.tracehr_structure_common import Data, stopping
from scripts.tracehr_centered_native_common import model_for as e_model
from scripts.tracehr_structure_runtime import Optimizer as PreviousOptimizer
from scripts.tracehr_backbone_refine_runtime import Optimizer as BaseOptimizer
from models.tracehr_structure import StructureNetwork
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_noa_pm_completion import NoAAblation, VARIANTS
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_recovery_queue import read

RUN='tracehr_noa_pm_completion_20261019'
MATRIX=tuple((v,e,s) for s in (46,47) for v in VARIANTS for e in (0,1))
PROTOCOL_FIELDS={'variant','structure','description','ablated_component','ablation_implementation','seed'}

def expected():return read(Path(__file__).resolve().parents[1]/'work'/RUN/'expected.json')

def protocol(variant, database='mimic', seed=43):
    assert database=='mimic' and variant in VARIANTS and seed in (43,46,47)
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

def authorized(variant, endpoint, seed, action):
    return (variant,endpoint,seed) in MATRIX if action=='train' else (action=='probe' and variant in VARIANTS and endpoint in (0,1) and seed==43)
