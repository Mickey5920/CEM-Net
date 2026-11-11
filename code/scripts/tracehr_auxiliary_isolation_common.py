"""C5 changes only auxiliary topology; C4 numeric optimization settings are fixed."""
from scripts.tracehr_direct_supervision_common import protocol as c4_protocol
from scripts.tracehr_context_cut_common import model_for as c1_model
from scripts.tracehr_structure_common import Data, stopping

RUN = 'tracehr_auxiliary_isolation_20261017'
VARIANTS = ('C5',)
PROTOCOL_CHANGES = {'variant', 'structure', 'description', 'training_objective', 'auxiliary_topology'}


def protocol(variant, database):
    assert variant == 'C5' and database == 'mimic'
    original = c4_protocol('C4', database)
    spec = dict(original, variant='C5', structure='C5',
                description='fixed z0; fusion-only auxiliary head without base hidden state or main head',
                training_objective='0.5*member_CE(z0)+0.5*CE(zA)',
                auxiliary_topology='Linear256to2(shared fusion residual); no direct base dependency')
    assert {k for k in set(original)|set(spec) if original.get(k)!=spec.get(k)} == PROTOCOL_CHANGES
    return spec


def model_for(data, preparation, variant, endpoint, seed, lr_multiplier=1., selected=None, tmp_folder=None):
    assert variant == 'C5' and seed == 42 and lr_multiplier == 1.
    from models.tracehr_auxiliary_isolation import AuxiliaryIsolationNetwork
    return AuxiliaryIsolationNetwork(c1_model(data, preparation, 'C1', endpoint, seed), seed)
