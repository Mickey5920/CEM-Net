"""Authorized 18 MIMIC deletions and 30 eICU endpoints; no later ablations."""
from pathlib import Path
from scripts.tracehr_noa_main_confirmation_common import protocol as main_protocol, model_for as full_model
from scripts.tracehr_structure_common import Data, stopping
from scripts.tracehr_structure_common import protocol as structure_protocol
from scripts.tracehr_execution_eicu_common import protocol as eicu_native_protocol, stopping as native_stopping
from scripts.tracehr_noa_ablation_common import PROTOCOL_FIELDS, Optimizer, diagnostics
from scripts.tracehr_centered_native_common import model_for as e_model
from models.tracehr_noa_validation import NoAAblation, VARIANTS
from models.tracehr_structure import StructureNetwork
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_recovery_queue import read
RUN='tracehr_noa_validation_20261020'
SEEDS=(48,49,50)
FAMILIES=('tabm','lightgbm','xgboost')
ROOT=Path(__file__).resolve().parents[1]
def expected():return read(ROOT/'work'/RUN/'expected.json')
def matrix(db):
    assert db in ('mimic','eicu')
    variants=VARIANTS if db=='mimic' else ('C5_noT_noA','B0')
    return tuple((v,e,s) for s in SEEDS for v in variants for e in (0,1))
def protocol(variant,database,seed):
    assert (variant,0,seed) in matrix(database)
    if variant=='B0':
        spec=eicu_native_protocol('F0');spec['seed']=seed;return spec
    spec=main_protocol('C5_noT_noA','mimic',seed)
    if database=='eicu':
        old=structure_protocol('G2','eicu')
        for key in ('database','min_epochs','early_stop_boundaries','significant_gain_patience','early_stop_rule'):
            spec[key]=old[key]
    else:
        spec.update(variant=variant,structure=variant,ablated_component=variant[-1],
            description='single deletion inside noA; fixed z0; 0.5 main CE; no auxiliary loss',
            ablation_implementation={'P':'zero slots preserve native width and retained initial tensors',
            'M':'identity input modulation; eight internal members and normalization preserved',
            'R':'delete summary-to-main residual; prune the now disconnected summary branch'}[variant[-1]])
    return spec
def model_for(data,preparation,variant,endpoint,seed):
    if variant=='C5_noT_noA':return full_model(data,preparation,variant,endpoint,seed)
    base=e_model(data,preparation,'E2',endpoint,seed);base_hash=parameter_hash(base)
    c1=remove_hidden_context(StructureNetwork(base,data.record['phi_dim'],'G2',seed),seed)
    model=NoAAblation(c1,variant,seed)
    model.info['seed_route']=dict(seed=seed,native_initialization_seed=seed,periodic_namespace_seed=seed,
        input_members_namespace_seed=seed,fusion_namespace_seed=seed,auxiliary_head_namespace_seed=seed,
        training_order_seed=seed,training_torch_seed=seed+10000,base_initial_weights_sha256=base_hash)
    return model
def baseline_command(family,seed,output,python):
    assert family in FAMILIES and seed in SEEDS
    source=expected()['eicu_templates'][family]['command'];i=source.index('-B')
    cmd=[python,*source[i:]];cmd[cmd.index('--output')+1]=str(output)
    if '--seed' in cmd:cmd[cmd.index('--seed')+1]=str(seed)
    else:cmd+=['--seed',str(seed)]
    assert not any(a in cmd for a in ('timeout','--pilot','--probe','--probe-full'))
    return cmd
