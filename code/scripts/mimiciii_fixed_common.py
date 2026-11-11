"""MIMIC-III routing for the frozen grouped-rate V1 implementation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts import group_v1_validation_common as fixed
from scripts.cem_structure_common import Data as ArchivedData
from scripts.tracehr_centered_native_common import stopping
from scripts.tracehr_multiview_common import atomic, read, sha

RUN = 'mimiciii_training_20261024'
ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'V1_group_rate'
SEEDS = (48, 49, 50)


def expected():
    return read(ROOT / 'work' / RUN / 'expected.json')


def protocol(variant, database, seed):
    assert variant == VARIANT and database == 'mimiciii' and seed in SEEDS
    value = dict(expected()['recipe'], seed=seed)
    assert value['database'] == database and value['variant'] == variant
    return value


# Only the database/seed recipe lookup is extended. The optimizer class,
# schedule, update, serialization and restoration operations remain unchanged.
fixed.protocol = protocol
Optimizer = fixed.Optimizer
diagnostics = fixed.diagnostics
parameter_hash = fixed.parameter_hash


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Data(ArchivedData):
    """Use the audited III matrices and the unchanged batch construction."""
    def __init__(self, preparation):
        preparation = Path(preparation)
        contract = expected()
        for name, entry in contract['inputs'].items():
            path = preparation / name
            assert not name.startswith('test')
            assert path.stat().st_size == entry['bytes']
            assert file_sha(path) == entry['sha256'], name
        self.record = read(preparation / 'completion.json')
        assert self.record['database'] == 'mimiciii' and self.record['complete']
        assert not self.record['test_used_for_fit']
        self.raw = {s: np.load(preparation / (s + '_raw.npy'), mmap_mode='r') for s in ('train', 'dev')}
        self.targets = {s: dict(np.load(preparation / (s + '_targets.npz'))) for s in self.raw}
        self.lengths = {s: np.load(preparation / (s + '_event_lengths.npy')) for s in self.raw}
        self.num_types = self.record['event_num_types']
        self.input_dim = self.record['input_dim']
        self.temporal = {}
        self.prep = dict(np.load(preparation / 'preprocessing.npz'))
        scaling = dict(np.load(preparation / 'native_summary_scaling.npz'))
        self.prep.update(center=scaling['center'], factor=scaling['factor'])
        assert self.input_dim == 13672 and self.num_types == 3207
        assert self.raw['train'].shape == (31451, 13672) and self.raw['dev'].shape == (4521, 13672)
        assert not set(self.targets['train']['ids'][:, 0]) & set(self.targets['dev']['ids'][:, 0])


def model_for(data, preparation, endpoint, seed):
    from models.tracehr_centered_native import CenteredNative
    from models.tracehr_structure import StructureNetwork
    from models.tracehr_context_cut import remove_hidden_context
    from models.tracehr_noa_validation import NoAAblation
    from models.tracehr_fte_binding import untie_fresh_pm
    from models.group_v1_paths import PathStructure
    spec = protocol(VARIANT, 'mimiciii', seed)
    config = dict(read(Path(preparation) / 'native_configs.json')[endpoint],
                  lr=spec['lr'], wd=spec['weight_decay'], p_drop=spec['dropout'])
    base = CenteredNative(data.raw['train'], data.targets['train']['y'][:, endpoint],
                          data.prep, config, seed, centered=False, init_rows=1024)
    phi = data.record['phi_dim']
    fresh = NoAAblation(remove_hidden_context(StructureNetwork(base, phi, 'G2', seed), seed),
                        'C5_noT_noA_noR', seed)
    fresh = untie_fresh_pm(fresh)
    fresh_hash = parameter_hash(fresh)
    model = PathStructure(fresh, phi, data.num_types, VARIANT, seed)
    assert model.info['reference_CEM_initial_weights_sha256'] == fresh_hash
    assert model.info['retained_CEM_tensors_exact'] and model.info['retained_CEM_buffers_exact']
    assert sum(p.numel() for p in model.parameters()) == 13533083
    return model
