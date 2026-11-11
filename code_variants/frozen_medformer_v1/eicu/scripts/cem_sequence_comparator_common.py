"""Immutable scientific recipes and compact shared inputs for the CNN/FT extension."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = 'cem_sequence_comparators_20261102'
VARIANTS = ('FTTransformer', 'TCN', 'ModernTCN', 'LITEMV')
SEEDS = (48, 49, 50, 51, 52)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.writing')
    q.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(q, p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def expected():
    return read(ROOT / 'work' / RUN / 'config.json')


def recipe(name):
    common = dict(selection='development AP', min_delta=1e-9, precision='float32',
                  loss='unweighted binary cross entropy / two-class cross entropy',
                  pretraining=False, hpo=False, seeds=list(SEEDS),
                  test_selection=False, train_only_preprocessing=True)
    settings = {
        'FTTransformer': dict(optimizer='AdamW', lr=1e-4, eps=1e-8, weight_decay=1e-5,
            epochs=256, patience=16, batch=256, micro=8, eval_batch=8,
            blocks=3, width=192, heads=8, attention_dropout=.2, ffn_dropout=.1,
            ffn_multiplier=4/3, residual_dropout=0., linformer=True, kv_positions=128,kv_sharing='key-value',
            configuration_basis='user-approved fixed author KV128 after full-attention compute preflight; no configuration search',
            source='author rtdl_revisiting_models recommended package',
            budget_source='predeclared project comparator cap, not a paper-wide optimum'),
        'TCN': dict(optimizer='AdamW', lr=5e-4, eps=1e-6, weight_decay=0.,
            epochs=50, patience=10, batch=16, micro=16, eval_batch=64,
            channels=[128]*4, kernel=4, dropout=.2, clip=.3,
            source='locuslab TemporalConvNet; final temporal output binary head',
            budget_source='predeclared clinical sequence cap'),
        'ModernTCN': dict(optimizer='Adam', lr=5e-4, eps=1e-8, weight_decay=0.,
            epochs=100, patience=10, batch=64, micro=64, eval_batch=64,
            blocks=[1,1], dims=[32,64], patch_size=8, patch_stride=4,
            large_size=[13,13], small_size=[5,5], ffn_ratio=1, dropout=.1,
            source='author classification EthanolConcentration example, fixed before clinical fits',
            budget_source='author classification example: 100 epochs, patience 10'),
        'LITEMV': dict(optimizer='KerasAdam', lr=1e-3, eps=1e-7, weight_decay=0.,
            epochs=1500, patience=None, batch=64, micro=64, eval_batch=64,
            filters=32, kernels=[40,20,10], dilation=[1,2,4], handcrafted_frozen=True,
            batchnorm_eps=.001, batchnorm_momentum=.99,
            source='author LITEMV, single network; audited PyTorch translation',
            reduce_loss=dict(factor=.5, patience=50, min_lr=1e-4, min_delta=1e-4),
            budget_source='published LITEMV 1500-epoch recipe; dev AP checkpoint adaptation'),
    }
    return dict(common, **settings[name])


def hourly_row(types, times, values, num_types):
    """Pure deterministic conversion: all existing records retained before binning."""
    types = np.asarray(types)
    times = np.asarray(times)
    values = np.asarray(values)
    assert len(types) == len(times) == len(values)
    assert np.isfinite(values).all() and np.all((times >= 0) & (times <= 24))
    hour = np.minimum(times.astype(np.int64), 23)
    event = types < num_types
    keys, counts = np.unique(hour[event] * num_types + types[event], return_counts=True)
    event_types = (keys % num_types).astype('int32')
    event_offsets = np.r_[0, np.cumsum(np.bincount(keys // num_types, minlength=24))].astype('int64')
    lab = np.zeros((24,32), np.float32)
    denom = np.zeros((24,32), np.int32)
    j = types[~event] - num_types
    assert np.all((j >= 0) & (j < 32))
    np.add.at(lab, (hour[~event], j), values[~event])
    np.add.at(denom, (hour[~event], j), 1)
    mask = denom > 0
    np.divide(lab, denom, out=lab, where=mask)
    last = np.zeros(32, dtype=np.int32)
    gap = np.empty((24,32), dtype=np.uint8)
    for t in range(24):
        gap[t] = t + 1 - last
        last = np.where(mask[t], t + 1, last)
    return event_types, counts.astype('float32'), event_offsets, lab, mask.astype('uint8'), gap


def prepare(project, split):
    """One cache writer per database; no learned encoder outputs are cached."""
    import fcntl
    c = expected()
    folder = Path(project) / c['hourly_cache']
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'builder.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = folder / (split + '_completion.json')
        if receipt.exists():
            result = read(receipt)
            assert result['complete']
            for n,h in result['files'].items():
                assert sha(folder / n) == h, n
            return result
        assert not list(folder.glob(split + '_*.npy')), 'Partial cache requires explicit recovery'
        tick = time.perf_counter()
        seq = Path(project) / c['sequence_inputs']
        origin = read(seq / ('test-completion.json' if split == 'test' else 'completion.json'))
        assert origin['complete']
        raw = {}
        provenance = {}
        for k in ('types','times','values','offsets'):
            name = split + '_' + k + '.npy'
            h = sha(seq / name)
            assert h == origin['artifacts'][name]
            raw[k] = np.load(seq / name, mmap_mode='r')
            provenance[str(seq / name)] = h
        targets = dict(np.load(seq / (split + '_targets.npz'), allow_pickle=False))
        assert sha(seq / (split + '_targets.npz')) == origin['artifacts'][split + '_targets.npz']
        provenance[str(seq/(split+'_targets.npz'))] = origin['artifacts'][split+'_targets.npz']
        if split != 'test':
            tab = Path(project)/c['tabular_inputs']
            tabrec = read(tab/'completion.json')
            assert tabrec['complete'] and tabrec['fit_split']=='train' and not tabrec['test_loaded']
            assert sha(tab/(split+'.npy')) == tabrec['artifacts'][split+'.npy']
            raw_targets = dict(np.load(Path(project)/c['raw_inputs']/(split+'_targets.npz'),allow_pickle=False))
            for key in ('ids','y'):np.testing.assert_array_equal(targets[key],raw_targets[key])
            provenance[str(tab/(split+'.npy'))] = tabrec['artifacts'][split+'.npy']
        n = len(targets['ids'])
        nt = read(seq / 'completion.json')['num_types']
        lab = np.lib.format.open_memmap(folder / (split + '_lab.npy'), mode='w+', dtype='float32', shape=(n,24,32))
        mask = np.lib.format.open_memmap(folder / (split + '_mask.npy'), mode='w+', dtype='uint8', shape=(n,24,32))
        gap = np.lib.format.open_memmap(folder / (split + '_gap.npy'), mode='w+', dtype='uint8', shape=(n,24,32))
        offsets = np.lib.format.open_memmap(folder / (split + '_offsets.npy'), mode='w+', dtype='int64', shape=(n*24+1,))
        offsets[0] = 0
        # Append compact codes/counts; avoids a maximum-size dense allocation and a second scan.
        typebin = folder / (split + '_types.bin')
        countbin = folder / (split + '_counts.bin')
        assert not typebin.exists() and not countbin.exists()
        with typebin.open('xb') as ft, countbin.open('xb') as fc:
            for i in range(n):
                lo,hi = raw['offsets'][i:i+2]
                ty,ct,off,lv,ma,de = hourly_row(raw['types'][lo:hi],raw['times'][lo:hi],raw['values'][lo:hi],nt)
                ft.write(ty.tobytes()); fc.write(ct.tobytes())
                offsets[i*24+1:(i+1)*24+1] = off[1:] + offsets[i*24]
                lab[i],mask[i],gap[i] = lv,ma,de
                if i % 2048 == 0:
                    save(folder / 'progress.json', dict(stage='hourly_cache',split=split,rows=i,total=n))
        for array in (lab,mask,gap,offsets):
            array.flush()
        np.savez_compressed(folder / (split + '_targets.npz'), **targets)
        if split == 'test':
            import joblib
            rawfolder=Path(project)/c['test_inputs']
            for name,record in c['test_manifest'].items():
                assert sha(rawfolder/name)==(record['sha256'] if isinstance(record,dict) else record)
            ft_targets=dict(np.load(rawfolder/'test_targets.npz',allow_pickle=False))
            for key in ('ids','y'):np.testing.assert_array_equal(targets[key],ft_targets[key])
            original = joblib.load(c['tabm_preprocessing'])
            assert sha(c['tabm_preprocessing']) == c['tabm_preprocessing_sha256']
            raw_ft = np.load(Path(project)/c['test_inputs']/'test_raw.npy',mmap_mode='r')
            z = np.lib.format.open_memmap(folder / 'test_ft.npy', mode='w+', dtype='float32', shape=raw_ft.shape)
            from scripts.cem_additional_common import transform
            for start in range(0,n,512):
                z[start:start+512] = transform(raw_ft[start:start+512],original)
            z.flush()
        files = {p.name:sha(p) for p in folder.glob(split + '_*') if p.suffix in ('.npy','.bin','.npz')}
        result = dict(complete=True,split=split,rows=n,num_types=nt,event_hour_pairs=int(offsets[-1]),
            elapsed_seconds=time.perf_counter()-tick,files=files,source_hashes=provenance,
            deterministic_only=True,learned_embeddings_cached=False,train_normalization_unchanged=True,
            bytes=sum((folder / p).stat().st_size for p in files))
        save(receipt,result)
        return result


class Data:
    def __init__(self, project, variant):
        self.project = Path(project)
        self.c = expected()
        self.variant = variant
        self.folder = self.project / self.c['hourly_cache']
        self.raw = {}; self.targets = {}; self.loaded = set()
        self.num_types = read(self.folder/'train_completion.json')['num_types']
        self.input_dim = self.c['input_dim']
        for split in ('train','dev'):
            self.load(split)

    def load(self, split):
        r = read(self.folder/(split+'_completion.json'))
        assert r['complete'] and r['num_types'] == self.num_types
        self.targets[split] = dict(np.load(self.folder/(split+'_targets.npz'),allow_pickle=False))
        if self.variant == 'FTTransformer':
            path = self.folder/'test_ft.npy' if split == 'test' else self.project/self.c['tabular_inputs']/(split+'.npy')
            self.raw[split] = np.load(path,mmap_mode='r')
            assert self.raw[split].shape == (len(self.targets[split]['ids']),self.input_dim)
        else:
            self.raw[split] = {k:np.load(self.folder/(split+'_'+k+'.npy'),mmap_mode='r') for k in ('lab','mask','gap','offsets')}
            for key,dtype in (('types','int32'),('counts','float32')):
                self.raw[split][key] = np.memmap(self.folder/(split+'_'+key+'.bin'),mode='r',dtype=dtype) if r['event_hour_pairs'] else np.empty(0,dtype=dtype)
        self.loaded.add(split)

    def cpu_batch(self, split, indices):
        import torch
        a = self.raw[split]
        if self.variant == 'FTTransformer':
            return (torch.from_numpy(np.ascontiguousarray(a[indices])),)
        codes=[];counts=[];offset=[0]
        for i in indices:
            off = a['offsets'][int(i)*24:int(i)*24+25]
            lo,hi = int(off[0]),int(off[-1])
            codes.append(a['types'][lo:hi]);counts.append(a['counts'][lo:hi])
            offset.extend((off[1:] - lo + offset[-1]).tolist())
        features = np.concatenate((a['lab'][indices],a['mask'][indices].astype('float32'),a['gap'][indices].astype('float32')/24.),axis=2)
        return tuple(torch.from_numpy(x) for x in (features,np.concatenate(codes).astype('int64'),np.concatenate(counts),np.asarray(offset,dtype='int64')))

    def batch(self, split, indices, device):
        return tuple(x.to(device) for x in self.cpu_batch(split,indices))

    def test(self):
        prepare(self.project, 'test')
        self.load('test')
        for split in ('train','dev'):
            assert not np.intersect1d(self.targets[split]['ids'][:,0],self.targets['test']['ids'][:,0]).size


def input_checks():
    ty,ct,off,lab,mask,gap = hourly_row(np.array([0,0,0,4,4]),np.array([.1,.2,24.,1.1,1.8]),np.array([1,1,1,2,4],np.float32),4)
    assert ty.tolist()==[0,0] and ct.tolist()==[2.,1.] and off[-1]==2
    assert lab[1,0]==3 and mask[1,0]==1 and mask[0,0]==0 and gap[2,0]==1
    empty = hourly_row(np.empty(0,int),np.empty(0),np.empty(0),4)
    assert not empty[3].any() and not empty[4].any() and empty[5][23,0]==24
    return ['within_hour_mean', 'event_multiplicity', '24h_boundary', 'missing_vs_zero', 'empty_histories', 'elapsed_time']
