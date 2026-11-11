"""Fixed hypotheses, provenance, and protocols for the native-transfer campaign."""
import shutil
from pathlib import Path
import numpy as np
from scripts.tracehr_multiview_common import atomic, read, sha
from scripts.tracehr_unified_common import Data, original_pointer
from scripts.tracehr_unified_next_common import prepare as previous_prepare, protocol as previous_protocol, stopping

RUN = 'tracehr_native_transfer_20261014'
CONTROLS = ('B0', 'B1', 'C0')
VARIANTS = tuple('C'+str(i) for i in range(12))
DESCRIPTIONS = {
    'B0': 'Official PyTabKit RealMLP cfg0, original full trainer and seed mapping',
    'B1': 'Official TabM cfg1, original model, preprocessing, sampling and optimizer',
    'C0': 'Exact N12 control with historical training microbatch',
    'C1': 'C0: independent complete member output heads only',
    'C2': 'C0: independent hidden member biases only',
    'C3': 'C0: untie learned numerical coordinates after coherent initialization only',
    'C4': 'C0: remove hidden-layer member adapters only; keep original head',
    'C5': 'C1+C2+C3: combined complete member transfer',
    'C6': 'C5: input-side-only modulation, independent hidden biases and full heads',
    'C7': 'C0: remove periodic numerical encoding; retain original raw features',
    'C8': 'C0: train-quantile piecewise-linear encoding, three coordinates per feature',
    'C9': 'C0: member-parameter learning-rate multiplier 0.25 only',
    'C10': 'C0: member-parameter weight decay zero only',
    'C11': 'C0: AdamW beta2=0.95 only',
}


def protocol(variant):
    assert variant in DESCRIPTIONS
    if variant in ('B0', 'B1'):
        return dict(native_package=True, max_epochs=256, effective_batch=256,
                    patience=None if variant=='B0' else 16, from_scratch=True,
                    test_loaded=False, selection='exact endpoint dev AP', description=DESCRIPTIONS[variant])
    spec = previous_protocol('N12')
    spec.update(description=DESCRIPTIONS[variant], beta2=.95 if variant=='C11' else .999,
                adapter_lr_multiplier=.25 if variant=='C9' else 1.,
                adapter_weight_decay=0. if variant=='C10' else .001,
                microbatch_policy='fixed to historical N12 for this database and endpoint')
    return spec


def old_baseline(project, database, variant):
    family, cfg = ('realmlp', 0) if variant=='B0' else ('tabm', 1)
    return project/'runs/tracehr_tabular_20261004'/database/('round1_r2' if database=='mimic' else 'round1_r3')/'runs'/f'{database}24_{family}_cfg{cfg}_full_s42'


def prepare(project, database, out):
    previous_prepare(project, database, out)
    record = read(out/'completion.json')
    _, folder, _ = original_pointer(project, database)
    prep = dict(np.load(out/'preprocessing.npz'))
    raw = np.load(folder/'train_raw.npy', mmap_mode='r')
    numeric = prep['numeric_indices']
    edges = np.empty((len(numeric), 4), np.float32)
    for start in range(0, len(numeric), 32):
        ix = numeric[start:start+32]
        x = (np.array(raw[:, ix], dtype=np.float64)-prep['center'][ix])*prep['factor'][ix]
        x /= np.sqrt(1+(x/3)**2)
        edges[start:start+len(ix)] = np.quantile(x, [0., 1/3, 2/3, 1.], axis=0).T
        atomic(out/'progress.json', dict(status='fitting_train_only_PLE', done=start+len(ix), total=len(numeric)))
    assert np.isfinite(edges).all() and np.all(np.diff(edges, axis=1)>=0)
    np.savez_compressed(out/'preprocessing.npz', **prep, ple_edges=edges)
    old = project/'runs/tracehr_unified_next_20261013'/database/'v1/screen'
    micros, evals, n12 = [], [], []
    for endpoint in (0, 1):
        pilot = read(old/f'pilots/probe_N12_t{endpoint}/completion.json')
        result = read(old/f'jobs/N12_t{endpoint}_s42/completion.json')
        assert pilot['complete'] and result['complete'] and result['from_scratch'] and not result['test_loaded']
        micros.append(pilot['selected_microbatch']); evals.append(pilot['evaluation_microbatch'])
        n12.append(dict(path=str(old/f'jobs/N12_t{endpoint}_s42'), dev_ap=result['dev_ap'],
                        initial_weights_sha256=result['initial_weights_sha256'],
                        predictions_sha256=result['predictions_sha256']))
    baselines = {}
    for variant in ('B0', 'B1'):
        path = old_baseline(project, database, variant)
        baselines[variant] = dict(path=str(path), manifest_sha256=sha(path/'run_manifest.json'),
                                 predictions_sha256=sha(path/'predictions_dev.npz'))
    record.update(preprocessing_sha256=sha(out/'preprocessing.npz'), n12_micro=micros, n12_eval=evals,
                  n12_reference=n12, baseline_references=baselines, protocols={v:protocol(v) for v in DESCRIPTIONS},
                  fit_split='train', PLE_quantiles=[0., 1/3, 2/3, 1.],
                  reproduction_gate=dict(max_abs_AP_difference=1e-5, max_abs_probability_difference=1e-4),
                  preparation_complete=True)
    atomic(out/'completion.json', record)
    atomic(out/'progress.json', dict(status='complete', train_rows=record['train_rows'],dev_rows=record['dev_rows']))


def model_for(data, preparation, variant, endpoint, seed, lr_multiplier=1.):
    from models.tracehr_native_transfer import TransferNetwork
    return TransferNetwork(data.input_dim, data.prep, variant, endpoint, seed)
