"""Frozen Medformer recipe and read-only reuse of the existing hourly inputs."""
from pathlib import Path
import numpy as np
from scripts.cem_sequence_comparator_common import Data as HourlyData, read, save, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = 'cem_medformer_20261103'
PARENT = 'cem_sequence_comparators_20261102'
VARIANTS = ('Medformer',)
SEEDS = (48, 49, 50, 51, 52)


def expected():
    return read(ROOT / 'work' / RUN / 'config.json')


def recipe():
    return dict(batch=128, micro=128, eval_batch=128, epochs=100, patience=10,
        min_delta=1e-5, optimizer='Adam', lr=1e-4, epsilon=1e-8, weight_decay=0.,
        clip=4., loss='unweighted two-class cross entropy', precision='float32',
        swa='arithmetic parameter average after every epoch, starting at epoch 1',
        selection='development AP of SWA model', pretraining=False, hpo=False,
        augmentations='none', seeds=list(SEEDS),
        model=dict(task_name='classification', pred_len=0, output_attention=False,
            enc_in=224, single_channel=False, seq_len=24, patch_len_list='2,4,8',
            augmentations='none', d_model=128, d_ff=256, e_layers=6, n_heads=8,
            dropout=.1, activation='gelu', no_inter_attn=False, num_class=2))


class Data(HourlyData):
    def __init__(self, project, variant='Medformer'):
        assert variant == 'Medformer'
        super().__init__(project, 'TCN')
        c = expected()
        assert self.c['hourly_cache'] == c['hourly_cache']
        for split in ('train', 'dev'):
            assert sha(self.folder / (split + '_completion.json')) == c['input_receipts'][split]

    def test(self):
        # The CNN pipeline already materialized this cache. Never call its writer.
        self.load('test')
        cfg=expected();path=self.folder/'test_completion.json';receipt=read(path)
        frozen=cfg['input_receipts'].get('test')
        if frozen is not None: assert sha(path)==frozen
        for name,h in receipt['files'].items():
            if name!='test_ft.npy': assert sha(self.folder/name)==h,name
        original=dict(np.load(self.project/cfg['test_inputs']/'test_targets.npz',allow_pickle=False))
        for key in ('ids','y'): np.testing.assert_array_equal(self.targets['test'][key],original[key])
        for split in ('train', 'dev'):
            assert not np.intersect1d(self.targets[split]['ids'][:, 0], self.targets['test']['ids'][:, 0]).size
