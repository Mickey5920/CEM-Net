"""Frozen inputs and fresh constructors for the two approved matched controls."""
import copy
import os
from pathlib import Path
import numpy as np
import torch

from scripts.tracehr_multiview_common import read, sha, atomic
from scripts.cem_matched_windows import OverlayMatrix
from scripts.tracehr_backbone_refine_runtime import Optimizer as OriginalOptimizer

ROOT = Path(__file__).resolve().parents[1]
RUN = 'cem_matched_controls_20261105'
VARIANTS = ('matched_tabm', 'window_summary')
SEEDS = (48, 49, 50, 51, 52)


def configuration():
    return read(ROOT/'work'/RUN/'config.json')


def budget(database, seed, endpoint):
    assert seed in SEEDS and endpoint in (0, 1)
    return 256 if database == 'eicu' else 52 if database == 'mimic' else 120 if endpoint == 0 or seed in (49, 51) else 52


def protocol(seed, endpoint):
    c = configuration()
    spec = copy.deepcopy(c['parent']['full_ema'][f'48:{endpoint}']['protocol'])
    spec.update(seed=seed)
    return spec


class Data:
    def __init__(self, project, window=False, test=False):
        self.project = Path(project); c = configuration(); self.database = c['database']
        self.preparation = self.project/c['parent']['input_preparation']
        self.record = read(self.preparation/'completion.json')
        self.input_dim = self.record['input_dim']
        self.num_types = self.record.get('event_num_types', (self.record['phi_dim']-12)//4)
        self.prep = dict(np.load(self.preparation/'preprocessing.npz'))
        scale = dict(np.load(self.preparation/'native_summary_scaling.npz'))
        self.prep.update(center=scale['center'], factor=scale['factor'])
        self.raw, self.targets = {}, {}
        for split in ('train', 'dev') + (('test',) if test else ()):
            folder = self.project/c['parent']['test_inputs'] if split == 'test' else Path(c['parent']['raw_inputs'])
            self.raw[split] = np.load(folder/f'{split}_raw.npy', mmap_mode='r')
            self.targets[split] = dict(np.load(folder/f'{split}_targets.npz'))
            assert self.raw[split].shape == (len(self.targets[split]['ids']), self.input_dim)
        if window:
            overlay = self.project/c['window_preparation']
            done = read(overlay/'completion.json'); assert done['complete']
            fitted = dict(np.load(overlay/'scaling.npz'))
            np.testing.assert_array_equal(fitted['numeric_indices'], self.prep['numeric_indices'])
            for name in ('center', 'factor'):
                np.testing.assert_array_equal(fitted[name][:-480], self.prep[name][:-480])
                self.prep[name] = fitted[name]
            for split in self.raw:
                self.raw[split] = OverlayMatrix(self.raw[split], np.load(overlay/f'{split}_windows.npy', mmap_mode='r'))
        for left, right in [('train', 'dev')] + ([('train', 'test'), ('dev', 'test')] if test else []):
            assert not np.intersect1d(self.targets[left]['ids'][:, 0], self.targets[right]['ids'][:, 0]).size

    def batch(self, split, indices, device='cuda'):
        x = torch.from_numpy(np.array(self.raw[split][indices], dtype='float32', copy=True)).to(device)
        y = torch.from_numpy(self.targets[split]['y'][indices].astype('float32')).to(device)
        return {'raw': x}, y


def construct(data, seed, endpoint, variant):
    from models.tracehr_centered_native import CenteredNative
    from models.tracehr_fte_binding import UntiedNative
    from models.cem_matched_tabm import MatchedTabM, Predictor
    spec = protocol(seed, endpoint)
    cfg = dict(read(data.preparation/'native_configs.json')[endpoint],
               lr=spec['lr'], wd=spec['weight_decay'], p_drop=spec['dropout'])
    base = UntiedNative(CenteredNative(data.raw['train'], data.targets['train']['y'][:, endpoint],
                                      data.prep, cfg, seed, centered=False, init_rows=1024))
    if variant == 'matched_tabm':
        base = MatchedTabM(base, seed, spec['dropout'])
    else:
        assert variant in ('window_summary', 'reference')
    return Predictor(base)


class Optimizer(OriginalOptimizer):
    def __init__(self, model, spec):
        super().__init__(model.base, spec, 1.)

    def schedule(self, epoch_float):
        super().schedule(epoch_float)
        if self.model.info.get('matched_control'):
            # Retain the same dropout probability schedule, including its horizon.
            hp = self.model.hp
            prob = hp.hyper_base_values['p_drop']['']*hp.get_hyper_sched_values()['p_drop']['']
            for module in self.model.net.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = float(prob)


def verify_original_constructor(project):
    """Prove the shortened constructor retains the full archived CEM state."""
    from scripts.cem_five_seed_common import construct as original
    from scripts.train_tracehr_unified import seed_all
    from models.tracehr_unified_next import parameter_hash
    data = Data(project)
    checks = {}
    for endpoint in (0, 1):
        seed_all(48)
        old = original(data, data.preparation, 48, endpoint)
        seed_all(48)
        direct = construct(data, 48, endpoint, 'reference')
        assert set(old.state_dict()) == set(direct.state_dict())
        assert all(torch.equal(v, direct.state_dict()[n]) for n, v in old.state_dict().items())
        assert parameter_hash(old) == parameter_hash(direct) == configuration()['parent']['full_ema'][f'48:{endpoint}']['initial_weights_sha256']
        x, labels = data.batch('train', np.arange(9), device='cpu')
        old.eval(); direct.eval()
        torch.testing.assert_close(old(x), direct(x), rtol=0, atol=0)
        old_logits = old(x)
        old_loss = old.loss(old_logits if data.database == 'mimiciii' else (old_logits, None), labels[:, endpoint])
        torch.testing.assert_close(old_loss, direct.loss(direct(x), labels[:, endpoint]), rtol=0, atol=0)
        checks[str(endpoint)] = dict(all_parameters_and_buffers_exact=True,
                                     initial_weights_sha256=parameter_hash(direct), loss_exact=True)
        del old, direct
    return checks
