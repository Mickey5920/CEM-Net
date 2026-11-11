"""One registered structural contrast, two MIMIC endpoints, unchanged G2 recipe."""
from scripts.tracehr_structure_common import Data, protocol as g_protocol, stopping
from scripts.tracehr_structure_common import model_for as g_model_for
from scripts.tracehr_multiview_common import read

RUN = 'tracehr_context_cut_20261016'
VARIANTS = ('C1',)


def protocol(variant, database):
    assert variant == 'C1' and database == 'mimic'
    original = g_protocol('G2', database)
    spec = dict(original, variant='C1', structure='C1',
                description='G2 without hidden-mean context projection', hidden_mean_context=False)
    changed = {k for k in set(original)|set(spec) if original.get(k)!=spec.get(k)}
    assert changed == {'variant','structure','description','hidden_mean_context'}
    return spec


def model_for(data, preparation, variant, endpoint, seed, lr_multiplier=1., selected=None, tmp_folder=None):
    assert variant == 'C1' and seed == 42 and lr_multiplier == 1.
    from models.tracehr_context_cut import remove_hidden_context
    from models.tracehr_unified_next import parameter_hash
    reference = g_model_for(data, preparation, 'G2', endpoint, seed)
    reference_hash = parameter_hash(reference)
    model = remove_hidden_context(reference, seed)
    model.info['random_G2_before_removal_sha256'] = reference_hash
    model.info['retained_base_random_initialization_sha256'] = parameter_hash(model.base)
    assert model.info['retained_base_random_initialization_sha256'] == data.record['base_initial_weights_sha256'][str(endpoint)]
    return model
