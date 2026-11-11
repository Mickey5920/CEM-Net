"""Fixed G0--G3 structure study. G0 is reused E2, never a new training job."""
import copy
from pathlib import Path
import shutil
import numpy as np
from scripts.tracehr_multiview_common import atomic, read, sha, inputs
from scripts.tracehr_unified_common import Data as BaseData, original_pointer
from scripts.tracehr_centered_native_common import protocol as e_protocol, model_for as e_model, stopping as e_stop

RUN = 'tracehr_structure_20261014'
VARIANTS = ('G1', 'G2', 'G3')


def protocol(variant, database):
    assert variant in ('G0',) + VARIANTS and database in ('mimic', 'eicu')
    spec = e_protocol('E2')
    spec.update(variant=variant, structure=variant, database=database, width=64, temporal_bins=8,
                fusion_lr=spec['adapter_lr'], fusion_weight_decay=spec['adapter_weight_decay'],
                fusion_beta2=spec['adapter_beta2'], fusion_schedule='same native clock as input adapter',
                fusion_residual_initial=.1, tuning=False, seed=42,
                description={'G0':'reused unchanged E2', 'G1':'variable trajectory + summary concatenation',
                             'G2':'summary-conditioned trajectory reading',
                             'G3':'G2 plus trajectory-conditioned summary refinement'}[variant])
    if database == 'eicu':
        spec.update(min_epochs=256, early_stop_boundaries=[], significant_gain_patience=None,
                    early_stop_rule='full 256 epochs; unchanged eICU E2 schedule')
    return spec


def stopping(history, spec):
    if spec['database'] == 'mimic':
        return e_stop(history, spec)
    return dict(stop=len(history)>=256, boundary=len(history)==256,
                next_boundary=256 if len(history)<256 else None,
                schedule_reference_epochs=256, early_stopping_enabled=False)


def reference_paths(project, database):
    root = project / ('runs/tracehr_centered_native_20261014/mimic/v1' if database == 'mimic'
                      else 'runs/tracehr_execution_refine_20261014/eicu/eicu_v1')
    return root/'preparation', {str(e):root/f'screen/jobs/E2_t{e}_s42' for e in (0,1)}


def patient_records(raw, observed, times, center, scale, bins=8):
    """All previously retained observations, stably sorted by variable/time.

    This is deterministic preprocessing, not a fitted or frozen neural encoder.
    Existing train-fitted clipping/units/timestamps remain unchanged.
    """
    assert raw.shape == observed.shape and raw.shape[1] == 32
    assert np.isfinite(times).all() and np.all((times >= 0) & (times <= 24))
    pieces, groups = [], []
    for variable in range(32):
        ix = np.flatnonzero(observed[:, variable])
        if not len(ix):
            continue
        ix = ix[np.argsort(times[ix], kind='stable')]
        t = (times[ix]/24).astype(np.float32)
        v = np.clip((raw[ix, variable]-center[variable])/scale[variable], -5, 5).astype(np.float32)
        assert np.isfinite(v).all()
        b = np.minimum((t*bins).astype(np.int64), bins-1)
        dt = np.r_[np.float32(0), np.diff(t)]
        dv = np.r_[np.float32(0), np.diff(v)]
        previous = np.ones(len(ix), np.float32); previous[0] = 0
        features = np.stack((v, t, dt, dv, previous, t*bins-b,
                             np.sin(2*np.pi*t), np.cos(2*np.pi*t)), -1).astype(np.float32)
        pieces.append(features); groups.append((variable*bins+b).astype(np.int16))
    records = np.concatenate(pieces) if pieces else np.empty((0,8), np.float32)
    segments = np.concatenate(groups) if groups else np.empty(0, np.int16)
    counts = np.bincount(segments, minlength=32*bins).astype(np.int32).reshape(32,bins)
    assert len(records) == int(observed.sum()) == int(counts.sum())
    return records, segments, counts


def prepare(project, database, out):
    previous, references = reference_paths(project, database)
    record = read(previous/'completion.json')
    assert record['complete'] and record['preparation_complete'] and not record['test_loaded']
    pointer, folder, original = original_pointer(project, database)
    assert pointer == record['input_pointer']
    for name, desc in record['input_artifacts'].items():
        atomic(out/'progress.json', dict(status='checking_existing_input', file=name))
        assert name.startswith(('train_', 'dev_')) and sha(folder/name) == desc['sha256']
    for name,key in [('preprocessing.npz','preprocessing_sha256'),('native_configs.json','native_configs_sha256'),
                     ('native_summary_scaling.npz','native_summary_scaling_sha256')]:
        assert sha(previous/name) == record[key]
        shutil.copyfile(previous/name, out/name)
    paths = inputs(project, database)
    nm, vm = read(paths['numeric']/'completion.json'), read(paths['values']/'completion.json')
    assert nm['complete'] and not nm['test_loaded'] and not vm['test_loaded']
    keys = read(paths['numeric']/'schema.json')['keys']
    assert keys == vm['keys'] and len(keys) == 32
    assert record['source_inputs']['numeric'] == sha(paths['numeric']/'completion.json')
    assert record['source_inputs']['values'] == sha(paths['values']/'completion.json')
    norm = read(paths['values']/'normalization.json')
    assert sha(paths['values']/'normalization.json') == vm['normalization_sha256']
    center = np.asarray([norm[k]['train_median'] for k in keys], np.float32)
    scale = np.asarray([norm[k]['train_iqr_or_std'] for k in keys], np.float32)
    assert np.isfinite(center).all() and np.all(scale>0)
    cache = out/'trajectory_cache'; cache.mkdir()
    artifacts, splits = {}, {}
    for split in ('train','dev'):
        arrays = {}
        for name in ('raw','observed','times','offsets'):
            f = paths['numeric']/f'{split}_{name}.npy'
            assert sha(f) == nm['artifacts'][f.name]['sha256']
            arrays[name] = np.load(f, mmap_mode='r')
        targets = dict(np.load(folder/f'{split}_targets.npz'))
        f = paths['numeric']/f'{split}_targets.npz'
        assert sha(f) == nm['artifacts'][f.name]['sha256']
        nt = dict(np.load(f))
        for key in ('ids','y'): assert np.array_equal(nt[key], targets[key]), (split,key)
        n = len(targets['ids']); total = int(arrays['observed'].sum())
        assert len(arrays['offsets']) == n+1
        rec = np.lib.format.open_memmap(cache/f'{split}_records.npy', 'w+', np.float32, (total,8))
        seg = np.lib.format.open_memmap(cache/f'{split}_segments.npy', 'w+', np.int16, (total,))
        counts = np.lib.format.open_memmap(cache/f'{split}_counts.npy', 'w+', np.int32, (n,32,8))
        offsets = np.zeros(n+1,np.int64); cursor=0
        for i in range(n):
            lo,hi = arrays['offsets'][i:i+2]
            r,g,c = patient_records(arrays['raw'][lo:hi], arrays['observed'][lo:hi],
                                    arrays['times'][lo:hi], center, scale)
            rec[cursor:cursor+len(r)] = r; seg[cursor:cursor+len(g)] = g; counts[i] = c
            cursor += len(r); offsets[i+1] = cursor
            if i%5000 == 0: atomic(out/'progress.json',dict(status='building_variable_trajectories',split=split,rows=i,total=n))
        assert cursor == total
        rec.flush(); seg.flush(); counts.flush(); del rec,seg,counts
        np.save(cache/f'{split}_offsets.npy', offsets)
        lengths = np.diff(offsets)
        splits[split] = dict(rows=n,observations=total,max_observations=int(lengths.max()),
                            empty_patients=int((lengths==0).sum()))
        for name in ('records','segments','counts','offsets'):
            f=cache/f'{split}_{name}.npy'; artifacts[f.name]=dict(bytes=f.stat().st_size,sha256=sha(f))
        # Check the existing summary layout against its source arrays, no guessed
        # clinical concept mapping or raw-data re-extraction is introduced.
        raw=np.load(folder/f'{split}_raw.npy',mmap_mode='r'); phi=record['phi_dim']
        for name,lo,hi in [('simple',phi,phi+352),('state',phi+352,phi+832)]:
            src=np.load(paths['values']/f'{split}_{name}.npz')['x']
            assert np.array_equal(raw[:,lo:hi],src), (split,name)
    record.update(reused_preparation=str(previous),previous_manifest_sha256=sha(previous/'completion.json'),
                  preparation_complete=True,protocols={v:protocol(v,database) for v in ('G0',)+VARIANTS},
                  trajectory_cache=str(cache),trajectory_artifacts=artifacts,trajectory_splits=splits,
                  trajectory_keys=keys,trajectory_normalization_sha256=sha(paths['values']/'normalization.json'),
                  trajectory_preprocessing='all retained observations; 32 variables x 8 blocks; no truncation',
                  g0_references={k:str(v) for k,v in references.items()},raw_cache_duplicated=False,
                  O_channel_started=False,hyperparameter_search=False)
    atomic(out/'completion.json',record)
    data=Data(project,database,out)
    from models.tracehr_unified_next import parameter_hash
    initial={}
    for endpoint in (0,1):
        base=e_model(data,out,'E2',endpoint,42)
        digest=parameter_hash(base)
        f=references[str(endpoint)]/'run_manifest.json'
        if f.exists(): assert digest==read(f)['initial_weights_sha256']
        else:
            assert database=='eicu'
            expected=read(previous/f'e2_initialization_task{endpoint}.json')['initial_weights_sha256']
            assert digest==expected
        initial[str(endpoint)]=digest
        del base
    record['base_initial_weights_sha256']=initial
    atomic(out/'completion.json',record)
    atomic(out/'progress.json',dict(status='complete',splits=splits,cache_gib=sum(x['bytes'] for x in artifacts.values())/2**30))


class Data(BaseData):
    def __init__(self, project, database, preparation):
        super().__init__(project,database,preparation)
        self.trajectory={}
        cache=Path(self.record['trajectory_cache'])
        for split in ('train','dev'):
            self.trajectory[split]={}
            for name in ('records','segments','counts','offsets'):
                f=cache/f'{split}_{name}.npy'
                assert sha(f)==self.record['trajectory_artifacts'][f.name]['sha256']
                self.trajectory[split][name]=np.load(f,mmap_mode='r')

    def batch(self, split, indices, device='cuda'):
        import torch
        raw,y=super().batch(split,indices,device)
        a=self.trajectory[split]; lengths=a['offsets'][indices+1]-a['offsets'][indices]
        offsets=np.r_[0,np.cumsum(lengths)]
        # Vectorized ragged gather retains original sample order and boundaries.
        pos=np.repeat(a['offsets'][indices]-offsets[:-1],lengths)+np.arange(offsets[-1])
        rec=np.asarray(a['records'][pos],dtype=np.float32)
        seg=np.asarray(a['segments'][pos],dtype=np.int64)+np.repeat(np.arange(len(indices))*256,lengths)
        return dict(raw=raw,records=torch.from_numpy(rec).to(device),segments=torch.from_numpy(seg).to(device),
                    counts=torch.from_numpy(np.array(a['counts'][indices])).to(device)),y


def model_for(data, preparation, variant, endpoint, seed, lr_multiplier=1., selected=None, tmp_folder=None):
    assert variant in VARIANTS and seed==42 and lr_multiplier==1.
    from models.tracehr_unified_next import parameter_hash
    from models.tracehr_structure import StructureNetwork
    base=e_model(data,preparation,'E2',endpoint,seed)
    initial=parameter_hash(base)
    assert initial==data.record['base_initial_weights_sha256'][str(endpoint)]
    model=StructureNetwork(base,data.record['phi_dim'],variant,seed)
    model.info['G0_initialization_match']=dict(complete=True,initial_weights_sha256=initial)
    return model
