"""Fixed F2 data, train-only transforms, and identity for N0--N3."""
import os
from pathlib import Path
import time

import numpy as np
from scripts.tracehr_multiview_common import atomic, read, sha, checked_output, inputs

RUN = 'tracehr_unified_20261013'
VARIANTS = ('N0', 'N1', 'N2', 'N3')
LEARNING_RATES = (3e-4, 1e-3)
PROTOCOL = dict(width=256, depth=3, dropout=.1, weight_decay=1e-4,
                effective_batch=512, max_epochs=64, min_epochs=12, patience=8,
                min_delta=1e-4, warmup_epochs=2, cosine_end_epoch=16, min_lr_ratio=.05,
                precision='float32', loss='mean binary cross entropy across members and two endpoints',
                selection='strict maximum dev mean AP', seed=42, from_scratch=True,
                test_loaded=False, numerical_encoding='PBLD: 16 frequencies, 3 learned coordinates + raw coordinate',
                member_modulation='8 rank-one members in shared hidden blocks and heads; one shared input projection')


def original_pointer(project, database):
    pointer = read(project/'work/tracehr_module_scratch_20261010'/f'inputs_{database}.json')
    folder = checked_output(project, pointer['path'])
    assert sha(folder/'completion.json') == pointer['manifest_sha256']
    manifest = read(folder/'completion.json')
    assert manifest['complete'] and not manifest['test_loaded']
    return pointer, folder, manifest


def prepare(project, database, out):
    import joblib
    started = time.time()
    pointer, folder, original = original_pointer(project, database)
    provenance = original['identity']['inputs']
    for name, path in inputs(project, database).items():
        assert sha(path/'completion.json') == provenance[name], (name, 'input completion changed')
    atomic(out/'progress.json', dict(status='validating_original_inputs', started_at=started))
    for name in ('train_raw.npy', 'dev_raw.npy', 'train_targets.npz', 'dev_targets.npz'):
        record = original['artifacts'][name]
        assert (folder/name).stat().st_size == record['bytes']
        assert sha(folder/name) == record['sha256'], name
    raw = {s: np.load(folder/f'{s}_raw.npy', mmap_mode='r') for s in ('train', 'dev')}
    targets = {s: dict(np.load(folder/f'{s}_targets.npz')) for s in ('train', 'dev')}
    assert not np.intersect1d(targets['train']['ids'][:, 0], targets['dev']['ids'][:, 0]).size
    for s in raw:
        assert len(raw[s]) == len(targets[s]['y']) == len(targets[s]['ids'])
        assert raw[s].dtype == np.float32 and targets[s]['y'].shape[1] == 2
        assert np.isin(targets[s]['y'], [0, 1]).all()
        for start in range(0, len(raw[s]), 1024):
            assert np.isfinite(raw[s][start:start+1024]).all()
    base = project/'runs/tracehr_tabular_20261004'/database/('round1_r2' if database == 'mimic' else 'round1_r3')/'runs'
    prep_path = base/f'{database}24_tabm_cfg1_full_s42/preprocessing.joblib'
    assert sha(prep_path) == original['identity']['recipe']['v_preprocessing']
    old = joblib.load(prep_path)
    assert old['fit_split'] == 'train'
    dim = raw['train'].shape[1]
    phi_dim = old['phi_dim']
    assert dim == phi_dim+832 == original['identity']['input_dim']
    center = np.zeros(dim, np.float32)
    factor = np.ones(dim, np.float32)
    binary = np.zeros(dim, bool)
    center[:phi_dim] = old['phi_center']
    factor[:phi_dim] = old['phi_factor']
    binary[:phi_dim] = old['binary']
    binary[phi_dim + np.arange(32)*11] = True
    atomic(out/'progress.json', dict(status='fitting_train_only_transform', started_at=started))
    # The 832 laboratory summaries already use train-fitted physiological scales.
    # Fit their outer robust scale on train only; do not fit dev/test quantiles.
    for start in range(phi_dim, dim, 32):
        x = np.asarray(raw['train'][:, start:start+32], dtype=np.float64)
        low, med, high = np.quantile(x, [.25, .5, .75], axis=0)
        spread = high-low
        spread = np.where(spread > 1e-6, spread, np.std(x, axis=0))
        center[start:start+32] = med
        factor[start:start+32] = 1/np.maximum(spread, 1.)
    center[binary] = 0
    factor[binary] = 1
    nonconstant = np.zeros(dim, bool)
    first = raw['train'][0]
    for start in range(0, len(raw['train']), 1024):
        block = raw['train'][start:start+1024]
        nonconstant |= np.any(block != first, axis=0)
        assert np.isin(block[:, binary], [0, 1]).all()
    numeric_indices = np.flatnonzero(~binary & nonconstant)
    assert len(numeric_indices) > 0 and np.isfinite(center).all() and np.isfinite(factor).all()
    np.savez_compressed(out/'preprocessing.npz', center=center, factor=factor,
                        binary=binary, numeric_indices=numeric_indices)
    tasks = read(inputs(project, database)['phi']/'completion.json').get('tasks')
    result = dict(complete=True, database=database, from_scratch=True, test_loaded=False,
                  fit_split='train', input_pointer=pointer, input_dim=dim, phi_dim=phi_dim,
                  train_rows=len(raw['train']), dev_rows=len(raw['dev']), tasks=tasks,
                  numeric_features=len(numeric_indices), binary_features=int(binary.sum()),
                  constant_features=int((~nonconstant).sum()), raw_cache_duplicated=False,
                  preprocessing_sha256=sha(out/'preprocessing.npz'), source_inputs=provenance,
                  input_artifacts={n:original['artifacts'][n] for n in ('train_raw.npy','dev_raw.npy','train_targets.npz','dev_targets.npz')},
                  seconds=time.time()-started, protocol=PROTOCOL)
    atomic(out/'completion.json', result)
    atomic(out/'progress.json', dict(status='complete', **result))
    return result


class Data:
    def __init__(self, project, database, preparation):
        self.record = read(preparation/'completion.json')
        assert self.record['complete'] and self.record['database'] == database and not self.record['test_loaded']
        assert sha(preparation/'preprocessing.npz') == self.record['preprocessing_sha256']
        pointer, folder, original = original_pointer(project, database)
        assert pointer == self.record['input_pointer']
        self.prep = dict(np.load(preparation/'preprocessing.npz'))
        self.raw = {}
        self.targets = {}
        for split in ('train', 'dev'):
            path = folder/f'{split}_raw.npy'
            assert path.stat().st_size == original['artifacts'][path.name]['bytes']
            self.raw[split] = np.load(path, mmap_mode='r')
            path = folder/f'{split}_targets.npz'
            assert sha(path) == self.record['input_artifacts'][path.name]['sha256']
            self.targets[split] = dict(np.load(path))
        self.input_dim = self.record['input_dim']

    def batch(self, split, indices, device='cuda'):
        import torch
        x = torch.from_numpy(np.array(self.raw[split][indices], copy=True)).to(device)
        y = torch.from_numpy(self.targets[split]['y'][indices].astype(np.float32)).to(device)
        return x, y


def stopping(history):
    significant_best = -1.
    significant_epoch = 0
    for row in history:
        if row['dev_ap'] > significant_best+PROTOCOL['min_delta']:
            significant_best = row['dev_ap']
            significant_epoch = row['epoch']
    return dict(stop=len(history) >= PROTOCOL['min_epochs'] and len(history)-significant_epoch >= PROTOCOL['patience'],
                last_significant_epoch=significant_epoch, significant_best=significant_best,
                epochs_without_significant_gain=len(history)-significant_epoch)
