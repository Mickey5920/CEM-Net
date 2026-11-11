"""Approved targeted controls; original CEM and prior experiment files are read-only."""
from pathlib import Path
import numpy as np
import torch
from scripts import cem_matched_common as original
from scripts.cem_matched_windows import OverlayMatrix
from scripts.tracehr_multiview_common import read, sha
from scripts.cem_icde72_common import mask_columns
from models.cem_icde72_mask import MaskedPeriodic

ROOT = Path(__file__).resolve().parents[1]
RUN = 'cem_targeted_evidence_20261106'
SEEDS = (48, 49, 50, 51, 52)
VARIANTS = ('CEM_basic', 'matched_basic', 'observation_weighted')
Optimizer, budget, protocol = original.Optimizer, original.budget, original.protocol


def configuration():
    return read(ROOT/'work'/RUN/'config.json')


class Data(original.Data):
    def __init__(self, project, variant='CEM_basic', test=False):
        super().__init__(project, window=False, test=test)
        self.variant = variant
        if variant == 'observation_weighted':
            folder = Path(project)/configuration()['overlay']
            receipt = read(folder/'completion.json')
            assert receipt['complete'] and receipt['changed_coordinates'] == 480
            fitted = dict(np.load(folder/'scaling.npz', allow_pickle=False))
            np.testing.assert_array_equal(fitted['numeric_indices'], self.prep['numeric_indices'])
            for name in ('center', 'factor'):
                np.testing.assert_array_equal(fitted[name][:-480], self.prep[name][:-480])
                self.prep[name] = fitted[name]
            for split in self.raw:
                self.raw[split] = OverlayMatrix(self.raw[split], np.load(folder/f'{split}_observations.npy', mmap_mode='r'))


def removed_columns(data):
    c = configuration()
    vocab = Path(data.project)/c['native_source_cache']/'token_to_id.json'
    return mask_columns(data.input_dim, data.record['phi_dim'], len(read(vocab)))


def construct(data, seed, endpoint, variant):
    model = original.construct(data, seed, endpoint, 'matched_tabm' if variant == 'matched_basic' else 'reference')
    if variant in ('CEM_basic', 'matched_basic'):
        model.base.numerical = MaskedPeriodic(model.base.numerical, removed_columns(data))
    else:
        assert variant == 'observation_weighted'
    model.base.info.update(targeted_variant=variant, original_model_not_modified=True)
    return model
