"""Locked D-round protocols, train-only preparation and cross-database selection."""
import copy
import shutil
from pathlib import Path

import numpy as np
from scripts.tracehr_multiview_common import atomic, read, sha
from scripts.tracehr_unified_common import Data as OriginalData, original_pointer
from scripts.tracehr_unified_next_common import stopping as residual_stopping
from scripts.tracehr_native_transfer_common import protocol as c_protocol, old_baseline

RUN = 'tracehr_backbone_refine_20261014'
STAGE1 = tuple('D'+str(i) for i in range(4))
STAGE2 = tuple('D'+str(i) for i in range(4, 8))
VARIANTS = STAGE1+STAGE2
BASELINES = {'mimic': [.540301595100345, .5489574506806235],
             'eicu': [.4243671396547053, .39008993154294835]}
DATABASE_BASELINES = {'mimic': .5446295228904843, 'eicu': .4037166646805196}


def protocol(variant, selected=None):
    assert variant in VARIANTS
    base = variant if variant in STAGE1 else selected
    assert base in STAGE1, 'D4--D7 require one jointly selected structure'
    spec = c_protocol('C4')
    native = base!='D0'
    spec.update(variant=variant, structure=base, native=native, members={'D0':8, 'D1':1, 'D2':4, 'D3':8}[base],
                dropout=.15 if native else .2, beta2=.95 if native else .999,
                lr=.04 if native else .0003, weight_decay=.02 if native else .001,
                effective_batch=256 if native else 512, adapter_lr=.0003, adapter_weight_decay=.001,
                adapter_beta2=.999, adapter_lr_multiplier=1., initialization_rows=1024,
                min_epochs=52 if native else 12, max_epochs=256 if native else 64,
                extension_epochs=256 if native else 128, patience=None if native else 8,
                schedule_reference_epochs=256 if native else 16, min_delta=1e-4,
                early_stop_boundaries=[52, 120, 256] if native else [],
                early_stop_rule='completed native cycles, never intra-cycle' if native else 'C4 patience',
                loss='mean memberwise cross entropy' if native else 'mean memberwise binary cross entropy',
                microbatch_policy='D0 historical C4; native bounded full-shape pilot',
                description='post-encoding scaling; typed periodic input; one shared jointly trained network',
                input_transform='existing robust typed transform')
    if variant=='D4':
        spec['lr'] *= .5
        spec['adapter_lr'] *= .5
    elif variant=='D5':
        spec['dropout'] += .10
    elif variant=='D6':
        spec['weight_decay'] *= 2
    elif variant=='D7':
        spec['input_transform'] = 'train-fitted quantile on existing nonbinary 832-summary coordinates only'
    return spec


def stopping(history, spec):
    if not spec['native']:
        return residual_stopping(history, spec)
    epoch = len(history)
    assert all(r['epoch']==i+1 for i,r in enumerate(history))
    previous_boundary = {52:17, 120:52, 256:120}
    result = dict(stop=False, boundary=epoch in previous_boundary, schedule_reference_epochs=256,
                  next_boundary=next((b for b in (52, 120, 256) if b>epoch), None))
    if epoch in previous_boundary:
        split = previous_boundary[epoch]
        before = max(r['dev_ap'] for r in history[:split])
        current = max(r['dev_ap'] for r in history[split:])
        result.update(previous_cycle_best=before, current_cycle_best=current, cycle_gain=current-before,
                      stop=epoch==256 or current<before+spec['min_delta'])
    return result


def select_structure(records):
    assert set(records)=={'mimic', 'eicu'}
    rows = []
    for v in STAGE1:
        aps, costs = {}, []
        for db in ('mimic', 'eicu'):
            aps[db] = []
            for endpoint in (0, 1):
                r = records[db][f'{v}_t{endpoint}_s42']
                assert r['complete'] and not r['pilot'] and r['from_scratch'] and not r['test_loaded']
                assert (r['database'], r['variant'], r['endpoint'], r['seed'])==(db, v, endpoint, 42)
                assert r['frozen_learnable_parameters']==0 and np.isfinite(r['dev_ap'])
                aps[db].append(r['dev_ap']); costs.append(r['seconds'])
        deltas = {db:float(np.mean(ap))-DATABASE_BASELINES[db] for db,ap in aps.items()}
        focus = min(aps['mimic'][0]-BASELINES['mimic'][0], aps['eicu'][1]-BASELINES['eicu'][1])
        rows.append(dict(structure=v, endpoint_ap=aps, database_delta=deltas, worst_database_delta=min(deltas.values()),
                         worst_focus_endpoint_delta=focus, mean_delta=float(np.mean(list(deltas.values()))),
                         members=protocol(v)['members'], training_seconds=sum(costs)))
    # No post-hoc per-database architectures; primary objective protects the weaker database.
    rows.sort(key=lambda r:(-r['worst_database_delta'], -r['worst_focus_endpoint_delta'],
                            -r['mean_delta'], r['members'], r['training_seconds'], r['structure']))
    return dict(selected=rows[0]['structure'], ranking=rows,
                rule='maximize worst database mean-AP delta; then worst focus endpoint delta; mean delta; lower K; lower cost',
                evaluation_usage='development selection only; no test claim')


def prepare(project, database, out):
    previous = project/'runs/tracehr_native_transfer_20261014'/database/'v2/preparation'
    record = read(previous/'completion.json')
    assert record['complete'] and record['preparation_complete'] and not record['test_loaded']
    assert record['fit_split']=='train'
    pointer, folder, _ = original_pointer(project, database)
    assert pointer==record['input_pointer']
    for name, artifact in record['input_artifacts'].items():
        atomic(out/'progress.json', dict(status='checking_input', file=name))
        assert sha(folder/name)==artifact['sha256']
    for filename,key in [('preprocessing.npz','preprocessing_sha256'),('native_configs.json','native_configs_sha256')]:
        assert sha(previous/filename)==record[key]
        shutil.copyfile(previous/filename, out/filename)
    # Replay the proposed boundary rule on all four completed original B0 histories.
    old = old_baseline(project, database, 'B0')
    history = read(old/'history.json')
    replay = []
    for endpoint in (0,1):
        rows = [dict(epoch=r['validation_event'],dev_ap=r['dev_ap']) for r in history if r['endpoint']==endpoint]
        assert len(rows)==256 and [r['epoch'] for r in rows]==list(range(1,257))
        stop_at = next(n for n in range(1,257) if stopping(rows[:n],protocol('D1'))['stop'])
        best = max(r['dev_ap'] for r in rows)
        retained = max(r['dev_ap'] for r in rows[:stop_at])
        assert best==retained, ('Cycle rule drops existing B0 best', database, endpoint, stop_at)
        replay.append(dict(endpoint=endpoint, stop_epoch=stop_at, retained_AP=retained, full_256_AP=best))
    # Reuse the existing train-fitted TabM transform, with raw cache hashes checked above.
    import joblib
    qpath = old_baseline(project, database, 'B1')/'preprocessing.joblib'
    q = joblib.load(qpath)
    assert q['fit_split']=='train'
    phi = int(q['phi_dim'])
    prep = dict(np.load(out/'preprocessing.npz'))
    assert len(prep['center'])-phi==832
    numeric_summary = np.flatnonzero(~prep['binary'][phi:])
    joblib.dump(q['numeric_quantile'], out/'summary_quantile.joblib')
    for split in ('train', 'dev'):
        raw = np.load(folder/f'{split}_raw.npy',mmap_mode='r')
        target = np.lib.format.open_memmap(out/f'{split}_quantile_summary.npy',mode='w+',dtype=np.float32,shape=(len(raw),len(numeric_summary)))
        for start in range(0,len(raw),1024):
            values = q['numeric_quantile'].transform(np.array(raw[start:start+1024,phi:],dtype=np.float32))
            target[start:start+len(values)] = values[:,numeric_summary]
        target.flush();del target
    record.update(reused_preparation=str(previous), previous_manifest_sha256=sha(previous/'completion.json'),
                  protocols={v:protocol(v) for v in STAGE1}, stage2_requires_common_selection=True,
                  cycle_rule_replay=replay, cycle_replay_source=dict(path=str(old/'history.json'),sha256=sha(old/'history.json')),
                  quantile_source=dict(path=str(qpath),sha256=sha(qpath),fit_split='train'),
                  quantile_summary_indices=(phi+numeric_summary).tolist(),
                  quantile_artifacts={s:dict(path=str(out/f'{s}_quantile_summary.npy'),sha256=sha(out/f'{s}_quantile_summary.npy')) for s in ('train','dev')},
                  preparation_complete=True, raw_cache_duplicated=False)
    atomic(out/'completion.json',record)
    atomic(out/'progress.json',dict(status='complete',train_rows=record['train_rows'],dev_rows=record['dev_rows'],cycle_rule_replay=replay))


class Data(OriginalData):
    def use_quantile(self, preparation):
        self.quantile = {}
        self.quantile_indices = np.array(self.record['quantile_summary_indices'])
        for split in ('train', 'dev'):
            artifact = self.record['quantile_artifacts'][split]
            assert sha(artifact['path'])==artifact['sha256']
            self.quantile[split] = np.load(artifact['path'], mmap_mode='r')
        self.prep = dict(self.prep)
        self.prep['passthrough'] = self.prep['binary'].copy()
        self.prep['passthrough'][self.quantile_indices] = True

    def raw_rows(self, split, indices):
        value = np.array(self.raw[split][indices],dtype=np.float32)
        if hasattr(self,'quantile'):
            value[:,self.quantile_indices] = self.quantile[split][indices]
        return value

    def batch(self, split, indices, device='cuda'):
        import torch
        return (torch.as_tensor(self.raw_rows(split,indices),device=device),
                torch.as_tensor(self.targets[split]['y'][indices],device=device,dtype=torch.float32))


def model_for(data, preparation, variant, endpoint, seed, lr_multiplier=1., selected=None):
    from models.tracehr_backbone_refine import PostScaleResidual, TypedNative
    spec = protocol(variant,selected)
    if variant=='D7':
        data.use_quantile(preparation)
    if not spec['native']:
        return PostScaleResidual(data.input_dim,data.prep,endpoint,seed,dropout=spec['dropout'])
    assert sha(preparation/'native_configs.json')==data.record['native_configs_sha256']
    cfg = copy.deepcopy(read(preparation/'native_configs.json')[endpoint])
    cfg.update(lr=spec['lr']*lr_multiplier,wd=spec['weight_decay'],p_drop=spec['dropout'])
    # Pass only the bounded initialization subset; the network never loads a checkpoint.
    ix = np.random.RandomState(seed).permutation(len(data.raw['train']))[:spec['initialization_rows']]
    # Keep original row selection/provenance inside TypedNative (proxy avoids a full copy for D7).
    class TrainingRows:
        shape = data.raw['train'].shape
        def __len__(self): return len(data.raw['train'])
        def __getitem__(self,index): return data.raw_rows('train',index)
    return TypedNative(TrainingRows(),data.targets['train']['y'][:,endpoint],data.prep,cfg,spec['members'],seed,spec['initialization_rows'])
