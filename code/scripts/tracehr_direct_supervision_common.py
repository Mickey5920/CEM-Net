"""Registered C4 objective change; all C1 optimization settings remain fixed."""
from scripts.tracehr_context_cut_common import model_for as c1_model, protocol as c1_protocol
from scripts.tracehr_structure_common import Data, stopping

RUN = 'tracehr_direct_supervision_20261016'
VARIANTS = ('C4',)
PROTOCOL_CHANGES = {'variant', 'structure', 'description', 'training_objective',
                    'main_view', 'checkpoint_selection_view', 'stopping_view'}


def protocol(variant, database):
    assert variant == 'C4' and database == 'mimic'
    original = c1_protocol('C1', database)
    spec = dict(original, variant='C4', structure='C4',
                description='shared C1 network, directly supervised fixed z0 and auxiliary z1',
                training_objective='0.5*member_CE(z0)+0.5*member_CE(z1)',
                main_view='z0', checkpoint_selection_view='z0', stopping_view='z0')
    assert {k for k in set(original)|set(spec) if original.get(k)!=spec.get(k)} == PROTOCOL_CHANGES
    return spec


def model_for(data, preparation, variant, endpoint, seed, lr_multiplier=1., selected=None, tmp_folder=None):
    assert variant == 'C4' and seed == 42 and lr_multiplier == 1.
    from models.tracehr_unified_next import parameter_hash
    from models.tracehr_direct_supervision import DirectSupervisionNetwork
    fresh = c1_model(data, preparation, 'C1', endpoint, seed)
    before = parameter_hash(fresh)
    model = DirectSupervisionNetwork(fresh)
    assert parameter_hash(model) == before
    model.info['random_C1_before_wrapping_sha256'] = before
    return model
