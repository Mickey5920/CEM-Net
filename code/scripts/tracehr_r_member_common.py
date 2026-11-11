"""Exactly six conditional RM comparisons; original scientific recipe intact."""
from pathlib import Path
from scripts.tracehr_not_stability_common import protocol as old_protocol
from scripts.tracehr_centered_native_common import model_for as e_model
from scripts.tracehr_structure_common import Data,stopping
from scripts.tracehr_noa_ablation_common import Optimizer,diagnostics
from scripts.tracehr_recovery_queue import read
from models.tracehr_structure import StructureNetwork
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_r_member import MemberResidual,VARIANT
from models.tracehr_unified_next import parameter_hash

RUN='tracehr_r_member_20261019'
VARIANTS=(VARIANT,)
MATRIX=tuple((VARIANT,e,s) for s in (43,44,45) for e in (0,1))
PROTOCOL_FIELDS={'variant','structure','description','ablated_component','ablation_implementation'}
def expected():return read(Path(__file__).resolve().parents[1]/'work'/RUN/'expected.json')
def protocol(variant,database='mimic',seed=43):
    assert variant==VARIANT and database=='mimic' and seed in (43,44,45)
    p=old_protocol('C5_noT_noA',database,seed)
    p.update(variant=variant,structure=variant,description='existing M conditions shared summary R; no other model or recipe change',
        ablated_component=None,ablation_implementation='match raw summary coordinates; independent R output for each internal member')
    return p
def model_for(data,preparation,variant,endpoint,seed):
    assert (variant,endpoint,seed) in MATRIX
    base=e_model(data,preparation,'E2',endpoint,seed);h=parameter_hash(base)
    c1=remove_hidden_context(StructureNetwork(base,data.record['phi_dim'],'G2',seed),seed)
    model=MemberResidual(c1,seed)
    model.info['seed_route']=dict(seed=seed,native_initialization_seed=seed,periodic_namespace_seed=seed,
        input_members_namespace_seed=seed,fusion_namespace_seed=seed,auxiliary_head_namespace_seed=seed,
        training_order_seed=seed,training_torch_seed=seed+10000,base_initial_weights_sha256=h)
    return model
