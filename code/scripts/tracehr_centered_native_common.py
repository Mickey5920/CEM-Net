"""Locked MIMIC-only eight-job E1--E4 protocol; E5 is intentionally absent."""
import copy
import shutil
from pathlib import Path
import numpy as np

from scripts.tracehr_multiview_common import atomic, read, sha
from scripts.tracehr_unified_common import Data, original_pointer
from scripts.tracehr_backbone_refine_common import protocol as d_protocol, old_baseline

RUN = 'tracehr_centered_native_20261014'
VARIANTS = STAGE1 = ('E1', 'E2', 'E3', 'E4')
STAGE2 = ()


def protocol(variant, selected=None):
    assert variant in VARIANTS
    spec = d_protocol('D3')
    spec.update(variant=variant, structure=variant, members=1 if variant=='E4' else 8,
                centered=variant in ('E1','E3'), native_summary_scaling=variant in ('E2','E3'),
                full_native_frontend=variant=='E4', training_microbatch=128, evaluation_microbatch=256,
                microbatch_policy='fixed 128, effective batch 256; FP32',
                early_stop_rule='only 52/120 boundaries; stop when no significant best-AP gain in last 17 epochs; cap 256',
                significant_gain_patience=17, selection='exact endpoint development AP',
                description={'E1':'D3 with fixed train-centered member modulation only',
                             'E2':'D3 with native scaling of existing numeric summary coordinates only',
                             'E3':'E1 plus E2',
                             'E4':'complete native frontend/trunk/random initialization, K=1, common extensible loop'}[variant])
    return spec


def stopping(history, spec):
    n = len(history)
    assert all(r['epoch']==i+1 for i,r in enumerate(history))
    significant_best, significant_epoch = -1., 0
    for row in history:
        if row['dev_ap']>=significant_best+spec['min_delta']:
            significant_best, significant_epoch = row['dev_ap'], row['epoch']
    return dict(stop=n==256 or (n in (52,120) and n-significant_epoch>=17),
                boundary=n in (52,120,256), next_boundary=next((b for b in (52,120,256) if b>n),None),
                schedule_reference_epochs=256, last_significant_epoch=significant_epoch,
                significant_best=significant_best, epochs_without_significant_gain=n-significant_epoch)


def native_summary_statistics(raw, prep, phi):
    """Match native median-center -> robust-scale float32 arithmetic, train only."""
    import torch
    from pytabkit.models.torch_utils import torch_np_quantile
    center, factor = prep['center'].copy(), prep['factor'].copy()
    indices = np.asarray(prep['numeric_indices'])
    indices = indices[indices>=phi]
    for columns in np.array_split(indices, max(1, int(np.ceil(len(indices)/32)))):
        x = torch.as_tensor(np.array(raw[:, columns], dtype=np.float32))
        median = torch_np_quantile(x,.5,dim=0)
        x = x-median
        spread = torch_np_quantile(x,.75,dim=0)-torch_np_quantile(x,.25,dim=0)
        fallback = .5*(x.max(0).values-x.min(0).values)
        spread = torch.where(spread==0, fallback, spread)
        scale = torch.where(spread==0, torch.zeros_like(spread), 1./(spread+1e-30))
        center[columns], factor[columns] = median.numpy(), scale.numpy()
    assert np.isfinite(center).all() and np.isfinite(factor).all()
    return center, factor, indices


def prepare(project, database, out):
    assert database=='mimic'
    previous = project/'runs/tracehr_backbone_refine_20261014/mimic/v1/preparation'
    record = read(previous/'completion.json')
    assert record['complete'] and record['preparation_complete'] and not record['test_loaded'] and record['fit_split']=='train'
    pointer, folder, _ = original_pointer(project,database)
    assert pointer==record['input_pointer']
    for name, artifact in record['input_artifacts'].items():
        assert name.startswith(('train_', 'dev_')), name
        atomic(out/'progress.json',dict(status='checking_input',file=name))
        assert sha(folder/name)==artifact['sha256']
    for filename,key in [('preprocessing.npz','preprocessing_sha256'),('native_configs.json','native_configs_sha256')]:
        assert sha(previous/filename)==record[key]
        shutil.copyfile(previous/filename,out/filename)
    atomic(out/'progress.json',dict(status='fitting_native_summary_statistics',fit_split='train'))
    prep = dict(np.load(out/'preprocessing.npz'))
    raw = np.load(folder/'train_raw.npy',mmap_mode='r')
    phi = raw.shape[1]-832
    assert phi==11676
    center,factor,indices = native_summary_statistics(raw,prep,phi)
    unchanged = np.ones(raw.shape[1],dtype=bool);unchanged[indices]=False
    assert np.array_equal(center[unchanged],prep['center'][unchanged]) and np.array_equal(factor[unchanged],prep['factor'][unchanged])
    np.savez(out/'native_summary_scaling.npz',center=center,factor=factor,indices=indices)
    replay=[]
    baseline = old_baseline(project,database,'B0')
    original = read(baseline/'history.json')
    histories = [(f'B0_t{e}',[dict(epoch=r['validation_event'],dev_ap=r['dev_ap']) for r in original if r['endpoint']==e],str(baseline/'history.json')) for e in (0,1)]
    droot = project/'runs/tracehr_backbone_refine_20261014/mimic/v1/screen/jobs'
    histories += [(f'D{v}_t{e}',read(droot/f'D{v}_t{e}_s42/history.json'),str(droot/f'D{v}_t{e}_s42/history.json')) for v in range(1,4) for e in (0,1)]
    for name,rows,path in histories:
        stop_at = next((n for n in range(1,len(rows)+1) if stopping(rows[:n],protocol('E1'))['stop']),len(rows))
        retained,max_ap = max(r['dev_ap'] for r in rows[:stop_at]),max(r['dev_ap'] for r in rows)
        assert retained==max_ap,('Proposed screening drops observed best',name,stop_at)
        replay.append(dict(name=name,stop_epoch=stop_at,observed_epochs=len(rows),retained_AP=retained,
                           observed_best_AP=max_ap,path=path,sha256=sha(path),complete_256_observation=len(rows)==256))
    # Remove unused D-only quantile pointers; no input data are copied.
    for key in list(record):
        if key.startswith('quantile_') or key in ('stage2_requires_common_selection','cycle_rule_replay','cycle_replay_source'):
            del record[key]
    record.update(reused_preparation=str(previous),previous_manifest_sha256=sha(previous/'completion.json'),
                  native_summary_scaling_sha256=sha(out/'native_summary_scaling.npz'),summary_numeric_features=len(indices),
                  protocols={v:protocol(v) for v in VARIANTS},screening_rule_replay=replay,
                  preparation_complete=True,raw_cache_duplicated=False,wait_for_eicu=False,E5_authorized=False)
    atomic(out/'completion.json',record)
    atomic(out/'progress.json',dict(status='complete',train_rows=record['train_rows'],dev_rows=record['dev_rows'],
                                   summary_numeric_features=len(indices),screening_rule_replay=replay))


def model_for(data, preparation, variant, endpoint, seed, lr_multiplier=1., selected=None, tmp_folder=None):
    from models.tracehr_centered_native import CenteredNative, FullNative
    spec = protocol(variant)
    if variant=='E4':
        assert tmp_folder is not None
        model = FullNative(data.raw,{s:data.targets[s]['y'] for s in ('train','dev')},endpoint,tmp_folder,seed)
        reference = Path(__import__('os').environ['TRACEHR_PROJECT'])/'runs/tracehr_native_transfer_20261014/mimic/v2/screen/jobs'/f'B0_t{endpoint}_s42/initialization.json'
        expected = read(reference)['initial_trainable_sha256']
        assert model.info['native_initial_trainable_sha256']==expected, ('Original B0 initialization mismatch',endpoint)
        model.info['B0_initialization_match'] = dict(complete=True,path=str(reference),sha256=sha(reference),initial_hash=expected)
        return model
    assert sha(preparation/'native_configs.json')==data.record['native_configs_sha256']
    cfg = copy.deepcopy(read(preparation/'native_configs.json')[endpoint])
    cfg.update(lr=spec['lr']*lr_multiplier,wd=spec['weight_decay'],p_drop=spec['dropout'])
    prep = copy.deepcopy(data.prep)
    if spec['native_summary_scaling']:
        assert sha(preparation/'native_summary_scaling.npz')==data.record['native_summary_scaling_sha256']
        scale = dict(np.load(preparation/'native_summary_scaling.npz'))
        prep.update(center=scale['center'],factor=scale['factor'])
    model = CenteredNative(data.raw['train'],data.targets['train']['y'][:,endpoint],prep,cfg,seed,
                           centered=spec['centered'],init_rows=spec['initialization_rows'])
    if variant=='E1':
        from models.tracehr_unified_next import parameter_hash
        reference = Path(__import__('os').environ['TRACEHR_PROJECT'])/'runs/tracehr_backbone_refine_20261014/mimic/v1/screen/jobs'/f'D3_t{endpoint}_s42/run_manifest.json'
        expected = read(reference)['initial_weights_sha256']
        assert parameter_hash(model)==expected,('E1 must share every initial learnable weight with D3',endpoint)
        model.info['D3_initialization_match'] = dict(complete=True,path=str(reference),sha256=sha(reference),initial_hash=expected)
    return model
