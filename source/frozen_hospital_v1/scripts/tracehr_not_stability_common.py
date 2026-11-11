"""Only the authorized first 28 fits; scientific recipes are unchanged."""
from scripts.tracehr_not_confirmation_common import protocol as confirmation_protocol
from scripts.tracehr_not_functional_common import protocol as functional_protocol
from scripts.tracehr_centered_native_common import model_for as e_model
from scripts.tracehr_structure_common import Data, stopping
from scripts.tracehr_execution_refine_common import protocol as native_protocol, stopping as native_stopping
from models.tracehr_not_confirmation import NoTrajectoryConfirmation
from models.tracehr_not_stability_functional import NoTrajectoryFunctional
from models.tracehr_structure import StructureNetwork
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_unified_next import parameter_hash

RUN='tracehr_not_stability_20261018'
VARIANTS=('C5_noT','C5_noT_noP','C5_noT_noM','C5_noT_noA','C5_noT_noR')
MATRIX=tuple([(v,e,s) for s in (45,46,47) for e in (0,1) for v in ('C5_noT','B0')]
             +[(v,e,s) for s in (43,44) for v in VARIANTS[1:] for e in (0,1)])
PROTOCOL_FIELDS={'variant','structure','description','ablated_component','ablation_implementation',
                 'training_objective','auxiliary_topology','seed'}

def protocol(variant,database='mimic',seed=42):
    assert database=='mimic' and seed in (42,43,44,45,46,47)
    if variant=='B0':
        result=native_protocol('F0')
    elif variant in VARIANTS[-2:]:
        result=functional_protocol(variant,'mimic',42)
    else:
        assert variant in VARIANTS
        result=confirmation_protocol(variant,'mimic',42)
    result['seed']=seed
    return result

def model_for(data,preparation,variant,endpoint,seed,**kwargs):
    assert variant in VARIANTS and seed in (42,43,44,45,46,47)
    assert kwargs.get('lr_multiplier',1.)==1.
    base=e_model(data,preparation,'E2',endpoint,seed)
    base_hash=parameter_hash(base)
    if seed==42:assert base_hash==data.record['base_initial_weights_sha256'][str(endpoint)]
    c1=remove_hidden_context(StructureNetwork(base,data.record['phi_dim'],'G2',seed),seed)
    cls=NoTrajectoryFunctional if variant in VARIANTS[-2:] else NoTrajectoryConfirmation
    model=cls(c1,variant,seed)
    model.info['seed_route']=dict(seed=seed,native_initialization_seed=seed,
        periodic_namespace_seed=seed,input_members_namespace_seed=seed,
        fusion_namespace_seed=seed,auxiliary_head_namespace_seed=seed,
        training_order_seed=seed,training_torch_seed=seed+10000,
        base_initial_weights_sha256=base_hash)
    return model

def stopping_for(variant,history,spec):
    return (native_stopping if variant=='B0' else stopping)(history,spec)
