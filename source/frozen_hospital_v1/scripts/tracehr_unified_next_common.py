"""Fixed identities and endpoint training protocols for N8--N15."""
from pathlib import Path
import shutil
import numpy as np
from scripts.tracehr_multiview_common import atomic, read, sha, checked_output
from scripts.tracehr_unified_common import Data, original_pointer

RUN = 'tracehr_unified_next_20261013'
VARIANTS = tuple('N' + str(i) for i in range(8, 16))


def protocol(variant):
    native = int(variant[1:]) >= 13
    return dict(native=native, effective_batch=256 if native else 512,
                min_epochs=48 if native else 12, max_epochs=96 if native else 64,
                extension_epochs=256 if native else 128, patience=24 if native else 8,
                min_delta=1e-4, lr=.04 if native else .0003,
                weight_decay=.02 if native else (.0001 if variant in ('N8', 'N9') else .001),
                members={'N8':1,'N13':1,'N15':32}.get(variant, 8),
                schedule_reference_epochs=256 if native else 16,
                selection='strict maximum endpoint dev AP', precision='float32',
                independent_endpoints=True, from_scratch=True, test_loaded=False,
                loss='mean memberwise cross entropy' if native else 'mean memberwise binary cross entropy')


def stopping(history, spec):
    best, at = -1., 0
    for row in history:
        if row['dev_ap'] > best + spec['min_delta']:
            best, at = row['dev_ap'], row['epoch']
    return dict(stop=len(history) >= spec['min_epochs'] and len(history)-at >= spec['patience'],
                last_significant_epoch=at, significant_best=best,
                epochs_without_significant_gain=len(history)-at)


def prepare(project, database, out):
    previous = project/'runs/tracehr_unified_20261013'/database/'v1/preparation'
    record = read(previous/'completion.json')
    assert record['complete'] and not record['test_loaded'] and record['fit_split'] == 'train'
    assert sha(previous/'preprocessing.npz') == record['preprocessing_sha256']
    pointer, folder, original = original_pointer(project, database)
    assert pointer == record['input_pointer']
    for name, artifact in record['input_artifacts'].items():
        atomic(out/'progress.json', dict(status='checking_input', file=name))
        assert sha(folder/name) == artifact['sha256']
    shutil.copyfile(previous/'preprocessing.npz', out/'preprocessing.npz')
    record = dict(record, reused_preparation=str(previous), previous_manifest_sha256=sha(previous/'completion.json'))
    base = project/'runs/tracehr_tabular_20261004'/database/('round1_r2' if database=='mimic' else 'round1_r3')/'runs'
    configs, source = [], []
    for endpoint in (0, 1):
        f=base/f'{database}24_realmlp_cfg0_full_s42'/f'effective_config_task{endpoint}.json'
        configs.append(read(f)); source.append(dict(path=str(f),sha256=sha(f)))
    atomic(out/'native_configs.json', configs)
    record.update(native_configs_sha256=sha(out/'native_configs.json'), native_config_sources=source,
                  protocols={v:protocol(v) for v in VARIANTS}, raw_cache_duplicated=False)
    atomic(out/'completion.json', record)
    atomic(out/'progress.json', dict(status='complete', train_rows=record['train_rows'], dev_rows=record['dev_rows']))


def model_for(data, preparation, variant, endpoint, seed, lr_multiplier=1.):
    from models.tracehr_unified_next import EndpointResidual, NativeModular
    if not protocol(variant)['native']:
        return EndpointResidual(data.input_dim, data.prep, variant, endpoint, seed)
    assert sha(preparation/'native_configs.json') == data.record['native_configs_sha256']
    cfg = read(preparation/'native_configs.json')[endpoint]
    cfg['lr'] *= lr_multiplier
    return NativeModular(data.raw['train'], data.targets['train']['y'][:, endpoint], variant, cfg, seed)
