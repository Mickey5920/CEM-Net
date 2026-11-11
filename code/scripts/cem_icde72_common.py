"""Inputs and frozen recipes for the explicitly authorized 72-fit batch."""
import os
from pathlib import Path
import numpy as np
from scripts.tracehr_multiview_common import read, sha, atomic

ROOT = Path(__file__).resolve().parents[1]
RUN = 'cem_icde_evidence_20261028'

def expected():
    return read(ROOT/'work'/RUN/'config.json')

def mask_columns(dimension, phi_dim, n_types):
    assert dimension == phi_dim + 832
    n_sources = (phi_dim - 4*n_types - 4)//2
    assert phi_dim == 4*n_types + 2*n_sources + 4 and n_sources > 0
    # log-count / presence survive. Latest time, decayed activity, the last
    # three global temporal statistics, and duration summaries are removed.
    return np.concatenate((np.arange(2*n_types, 4*n_types),
                           np.arange(phi_dim-3, phi_dim),
                           np.arange(phi_dim+352, dimension))).astype(np.int64)

class RawData:
    def __init__(self, project):
        self.project = Path(project)
        self.c = expected()
        folder = self.project/self.c['raw_inputs']
        self.raw = {s:np.load(folder/(s+'_raw.npy'), mmap_mode='r') for s in ('train','dev')}
        self.targets = {s:dict(np.load(folder/(s+'_targets.npz'))) for s in ('train','dev')}
        self.input_dim = self.raw['train'].shape[1]
        for s in ('train','dev'):
            assert self.raw[s].dtype == np.float32 and self.raw[s].shape == (len(self.targets[s]['y']),self.input_dim)
            assert self.targets[s]['y'].shape[1] == 2
        assert not np.intersect1d(self.targets['train']['ids'][:,0],self.targets['dev']['ids'][:,0]).size
    def test(self):
        folder = self.project/self.c['test_inputs']
        for n,h in self.c['test_manifest'].items():
            assert sha(folder/n) == (h['sha256'] if isinstance(h,dict) else h), n
        self.raw['test'] = np.load(folder/'test_raw.npy',mmap_mode='r')
        self.targets['test'] = dict(np.load(folder/'test_targets.npz'))
        for s in ('train','dev'):
            assert not np.intersect1d(self.targets['test']['ids'][:,0],self.targets[s]['ids'][:,0]).size

def lightgbm_recipe(seed):
    return dict(objective='binary', metric='None', learning_rate=.04, num_leaves=50,
                bagging_fraction=.75, bagging_freq=1, feature_fraction=1.,
                min_data_in_leaf=40, min_sum_hessian_in_leaf=1e-7, max_bin=255,
                num_threads=4, deterministic=True, force_col_wise=True,
                seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed,
                lambda_l1=0., lambda_l2=0., max_depth=-1, min_gain_to_split=0.,
                is_unbalance=False, scale_pos_weight=1., use_missing=True, zero_as_missing=False,
                verbosity=-1)
