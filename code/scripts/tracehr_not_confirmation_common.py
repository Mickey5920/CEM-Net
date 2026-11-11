"""First authorized confirmation batch: exactly twelve MIMIC fits."""
from scripts.tracehr_c5_ablation_common import protocol as previous_protocol
from scripts.tracehr_centered_native_common import model_for as e_model
from scripts.tracehr_structure_common import Data, stopping
from scripts.tracehr_multiview_common import read
from models.tracehr_not_confirmation import VARIANTS, NoTrajectoryConfirmation
from models.tracehr_structure import StructureNetwork
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_unified_next import parameter_hash

RUN = 'tracehr_not_confirmation_20261017'
MATRIX = tuple([(v, e, s) for s in (43,44) for e in (0,1) for v in ('C5_noT','B0')]
               + [(v,e,42) for v in ('C5_noT_noP','C5_noT_noM') for e in (0,1)])
PROTOCOL_FIELDS = {'variant','structure','description','ablated_component','ablation_implementation','seed'}


def protocol(variant, database='mimic', seed=42):
    assert variant in VARIANTS and database=='mimic' and seed in (42,43,44)
    result = previous_protocol('C5_noT', database)
    result['seed'] = seed
    if variant!='C5_noT':
        component = 'learned_numeric_periodic_encoding' if variant.endswith('noP') else 'input_member_modulation'
        description = ('remove periodic encoder; keep zero slots and native width' if variant.endswith('noP') else
                       'remove input modulation; retain eight internal members and mean member loss')
        result.update(variant=variant, structure=variant, description='Within C5_noT: '+description,
                      ablated_component=component, ablation_implementation=description)
    original=previous_protocol('C5_noT', database)
    assert {k for k in set(original)|set(result) if original.get(k)!=result.get(k)} <= PROTOCOL_FIELDS
    return result


def model_for(data, preparation, variant, endpoint, seed, **kwargs):
    assert variant in VARIANTS and seed in (42,43,44)
    assert kwargs.get('lr_multiplier',1.)==1.
    base=e_model(data,preparation,'E2',endpoint,seed)
    base_hash=parameter_hash(base)
    if seed==42:
        assert base_hash==data.record['base_initial_weights_sha256'][str(endpoint)]
    c1=remove_hidden_context(StructureNetwork(base,data.record['phi_dim'],'G2',seed),seed)
    model=NoTrajectoryConfirmation(c1,variant,seed)
    model.info['seed_route']=dict(seed=seed, native_initialization_seed=seed,
        periodic_namespace_seed=seed, input_members_namespace_seed=seed,
        fusion_namespace_seed=seed, auxiliary_head_namespace_seed=seed,
        training_order_seed=seed, training_torch_seed=seed+10000,
        base_initial_weights_sha256=base_hash)
    if seed==42:
        project=data.project if hasattr(data,'project') else preparation.parents[4]
        # Caller audits against the immutable run manifest; factory never reads weights.
        model.info['seed42_base_initialization_exact']=True
    return model
