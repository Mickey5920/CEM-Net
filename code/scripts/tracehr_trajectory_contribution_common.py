"""Fixed C2 contribution experiment, using the verified C1 initialization route."""
from scripts.tracehr_context_cut_common import model_for as c1_model, protocol as c1_protocol
from scripts.tracehr_structure_common import Data as StructureData, stopping
from scripts.tracehr_unified_common import Data as RawData

RUN = 'tracehr_trajectory_contribution_20261016'
VARIANTS = ('C2',)


class Data(StructureData):
    def batch(self, split, indices, device='cuda'):
        raw, y = RawData.batch(self, split, indices, device)
        return dict(raw=raw), y


def protocol(variant, database):
    assert variant == 'C2' and database == 'mimic'
    original = c1_protocol('C1', database)
    spec = dict(original, variant=variant, structure=variant, trajectory=False,
                description='C1 summary residual only; added trajectory encoder and reading removed')
    changed = {k for k in set(original) | set(spec) if original.get(k) != spec.get(k)}
    assert changed == {'variant', 'structure', 'description', 'trajectory'}
    return spec


def model_for(data, preparation, variant, endpoint, seed, lr_multiplier=1., selected=None, tmp_folder=None):
    assert variant == 'C2' and seed == 42 and lr_multiplier == 1.
    from models.tracehr_unified_next import parameter_hash
    from models.tracehr_trajectory_contribution import remove_trajectory
    model = c1_model(data, preparation, 'C1', endpoint, seed)
    reference_hash = parameter_hash(model)
    model = remove_trajectory(model)
    model.info['random_C1_before_removal_sha256'] = reference_hash
    return model
