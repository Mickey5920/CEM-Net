"""Fixed inputs and recipes for the authorized two-comparator extension."""
import os
from pathlib import Path
import numpy as np
from scripts.tracehr_multiview_common import read, sha, atomic

ROOT = Path(__file__).resolve().parents[1]
RUN = 'cem_baseline_replacement_20261030'

def expected():
    return read(ROOT/'work'/RUN/'config.json')

def protocol(variant):
    assert variant == 'STraTS_supervised'
    return dict(optimizer='author_AdamW', lr=5e-4, weight_decay=0., eps=1e-6,
                betas=[.9,.999], batch=16, micro=1, epochs=50, patience=10,
                clip_grad_norm=.3, hidden=64, layers=2, heads=16, dropout=.2,
                attention_dropout=.2, pos_weight='training negatives / positives',
                selection='validation average precision', pretraining=False,
                max_observations=None, recipe='official STraTS ss- author reimplementation')

def xgb_recipe(seed):
    return dict(objective='binary:logistic', learning_rate=.08, min_child_weight=5e-6,
                reg_lambda=0., reg_alpha=0., max_depth=6, colsample_bylevel=.9,
                subsample=.65, tree_method='hist', max_bin=256, nthread=4,
                seed=seed, device='cpu', disable_default_eval_metric=1)

class SequenceData:
    def __init__(self, project):
        from scripts.cem_sequence_inputs import SequenceData as Existing
        self.project=Path(project);self.folder=self.project/expected()['sequence_inputs']
        self.record=read(self.folder/'completion.json')
        assert self.record['complete'] and sha(self.folder/'completion.json')==expected()['sequence_completion_sha256']
        self.input_dim=self.record['input_dim'];self.num_types=self.record['num_types']
        self.raw={};self.targets={};self.variant='STraTS_supervised'
        self.load=Existing.load.__get__(self);self.batch=Existing.batch.__get__(self)
        for s in ('train','dev'):self.load(s)
        assert not np.intersect1d(self.targets['train']['ids'][:,0],self.targets['dev']['ids'][:,0]).size
    def test(self):
        # Existing immutable preparation only; this reader never rebuilds inputs.
        r=read(self.folder/'test-completion.json');assert r['complete']
        for n,h in r['artifacts'].items():assert sha(self.folder/n)==h,n
        self.load('test')
        for s in ('train','dev'):
            assert not np.intersect1d(self.targets['test']['ids'][:,0],self.targets[s]['ids'][:,0]).size

def make_data(project):return SequenceData(project)

def raw_data(project):
    from scripts import cem_icde72_common as previous
    previous.expected=expected
    return previous.RawData(project)
