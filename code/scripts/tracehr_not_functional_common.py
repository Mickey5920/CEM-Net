"""Exactly four seed42 MIMIC functional ablations, fixed original recipe."""
from scripts.tracehr_not_confirmation_common import protocol as previous_protocol
from scripts.tracehr_centered_native_common import model_for as e_model
from scripts.tracehr_structure_common import Data, stopping
from models.tracehr_structure import StructureNetwork
from models.tracehr_context_cut import remove_hidden_context
from models.tracehr_not_functional import VARIANTS, NoTrajectoryFunctional
from models.tracehr_unified_next import parameter_hash

RUN = 'tracehr_not_functional_20261018'
MATRIX = tuple((v, e, 42) for v in VARIANTS for e in (0, 1))
PROTOCOL_FIELDS = {'variant', 'structure', 'description', 'ablated_component',
                   'ablation_implementation', 'training_objective', 'auxiliary_topology'}


def protocol(variant, database='mimic', seed=42):
    assert variant in VARIANTS and database == 'mimic' and seed == 42
    result = previous_protocol('C5_noT', database, seed)
    no_a = variant == 'C5_noT_noA'
    description = ('remove auxiliary head and auxiliary CE; keep main CE coefficient 0.5 and summary residual'
                   if no_a else 'remove summary residual addition to main classifier; retain summary auxiliary head and both 0.5 CE losses')
    result.update(variant=variant, structure=variant, description=description,
                  ablated_component='auxiliary_supervision' if no_a else 'main_summary_residual_connection',
                  ablation_implementation=description,
                  training_objective='0.5*member_CE(z0)' if no_a else result['training_objective'],
                  auxiliary_topology='removed; no auxiliary logits or loss' if no_a else result['auxiliary_topology'])
    reference = previous_protocol('C5_noT', database, seed)
    assert {k for k in set(result)|set(reference) if result.get(k)!=reference.get(k)} <= PROTOCOL_FIELDS
    return result


def model_for(data, preparation, variant, endpoint, seed, **kwargs):
    assert variant in VARIANTS and seed == 42 and kwargs.get('lr_multiplier', 1.) == 1.
    base = e_model(data, preparation, 'E2', endpoint, seed)
    base_hash = parameter_hash(base)
    assert base_hash == data.record['base_initial_weights_sha256'][str(endpoint)]
    c1 = remove_hidden_context(StructureNetwork(base, data.record['phi_dim'], 'G2', seed), seed)
    model = NoTrajectoryFunctional(c1, variant, seed)
    model.info['seed_route'] = dict(seed=seed, native_initialization_seed=seed,
        periodic_namespace_seed=seed, input_members_namespace_seed=seed,
        fusion_namespace_seed=seed, auxiliary_head_namespace_seed=seed,
        training_order_seed=seed, training_torch_seed=seed+10000,
        base_initial_weights_sha256=base_hash)
    return model
