"""Native observation inputs, using only existing timestamp-limited records."""
import os
from pathlib import Path
import numpy as np
import torch
from scripts.cem_additional_common import expected,preparation,Data,read,sha,atomic

def native_sources(project):
    c=expected();folder=project/c['native_source_cache'];m=read(folder/'source_files.json')
    for n,h in m['files'].items():assert sha(folder/n)==h,n
    return folder,m

def get_records(project,targets,split,normalization=None):
    """Return per-patient measurements and categorical occurrences; no labels used."""
    c=expected();db=c['database'];n=len(targets['ids']);lookup={int(s):i for i,s in enumerate(targets['ids'][:,2])}
    if db!='mimiciii' and split!='test':
        from scripts.tracehr_multiview_common import inputs
        inp=inputs(project,db);em=read(inp['events']/'completion.json');nm=read(inp['numeric']/'completion.json')
        for folder,meta,names in [(inp['events'],em,['cat_types.npy','cat_times.npy','cat_offsets.npy',split+'_targets.npz']),
            (inp['numeric'],nm,[split+'_'+k+'.npy' for k in ('raw','observed','times','offsets')]+[split+'_targets.npz'])]:
            for name in names:assert sha(folder/name)==meta['artifacts'][name]['sha256'],name
            tt=dict(np.load(folder/(split+'_targets.npz')))
            for k in ('ids','y'):np.testing.assert_array_equal(tt[k],targets[k])
        cat={k:np.load(inp['events']/('cat_'+k+'.npy'),mmap_mode='r') for k in ('types','times','offsets')}
        numeric={k:np.load(inp['numeric']/(split+'_'+k+'.npy'),mmap_mode='r') for k in ('raw','observed','times','offsets')}
        shift=0 if split=='train' else em['train_rows'];nt=em['num_types']
        if normalization is None:
            assert split=='train';mean=[];std=[]
            for j in range(32):
                v=np.asarray(numeric['raw'][:,j])[numeric['observed'][:,j].astype(bool)].astype('float64');mean.append(float(v.mean()));std.append(max(float(v.std()),1e-6))
            normalization=dict(mean=mean,std=std)
        def rows():
            for i in range(n):
                lo,hi=cat['offsets'][i+shift:i+shift+2];nl,nh=numeric['offsets'][i:i+2]
                r,j=np.nonzero(numeric['observed'][nl:nh]);v=numeric['raw'][nl+r,j]
                v=(v-np.asarray(normalization['mean'])[j])/np.asarray(normalization['std'])[j]
                yield np.r_[cat['types'][lo:hi],nt+j],np.r_[cat['times'][lo:hi],numeric['times'][nl+r]],np.r_[np.ones(hi-lo),v]
        counts=np.diff(cat['offsets'])[shift:shift+n]+np.asarray([numeric['observed'][numeric['offsets'][i]:numeric['offsets'][i+1]].sum() for i in range(n)])
        return rows,counts,nt,normalization
    import pandas as pd
    folder,source=native_sources(project)
    vocabulary=read(folder/'token_to_id.json');keys=source['lab_keys'];nt=len(vocabulary)
    # Predicate filters keep unrequested patients outside materialized frames.
    allowed=list(lookup);cats={};numbers={}
    for f in sorted((folder/'events').glob('*.parquet')):
        frame=pd.read_parquet(f,filters=[('stay_id','in',allowed),('time','<=',24.)]).drop_duplicates(['stay_id','time','token'])
        for sid,g in frame.groupby('stay_id',sort=False):
            types=[vocabulary.get(t,vocabulary[s+'::UNK']) for t,s in zip(g.token,g.source)]
            if int(sid) in cats:
                old=cats[int(sid)];cats[int(sid)]=(np.r_[old[0],types],np.r_[old[1],g.time.to_numpy()])
            else:cats[int(sid)]=(np.asarray(types),g.time.to_numpy())
    f=pd.read_parquet(folder/'numeric.parquet',filters=[('stay_id','in',allowed),('available_h','<=',24.)])
    f['key']=f.lab.astype(str)+'|'+f.unit;f=f[f.key.isin(keys)]
    f=f.groupby(['stay_id','available_h','key'],as_index=False,observed=True).agg(value=('value','median'))
    f['variable']=f.key.map({k:j for j,k in enumerate(keys)}).astype('int64')
    if normalization is None:
        assert split=='train';mean=[];std=[]
        for j in range(32):
            v=f.loc[f.variable==j,'value'].to_numpy(dtype='float64');assert len(v);mean.append(float(v.mean()));std.append(max(float(v.std()),1e-6))
        normalization=dict(mean=mean,std=std)
    j=f.variable.to_numpy();f['z']=(f.value-np.asarray(normalization['mean'])[j])/np.asarray(normalization['std'])[j]
    for sid,g in f.groupby('stay_id',sort=False):numbers[int(sid)]=(g.variable.to_numpy()+nt,g.available_h.to_numpy(),g.z.to_numpy())
    def rows():
        for sid in targets['ids'][:,2]:
            ct,tm=cats.get(int(sid),(np.array([],int),np.array([])));nn=numbers.get(int(sid),(np.array([],int),np.array([]),np.array([])))
            yield np.r_[ct,nn[0]],np.r_[tm,nn[1]],np.r_[np.ones(len(ct)),nn[2]]
    cat_counts=np.asarray([len(cats.get(int(s),((),()))[0]) for s in targets['ids'][:,2]])
    if 'lengths' in targets:np.testing.assert_array_equal(cat_counts,targets['lengths'])
    counts=cat_counts+np.asarray([len(numbers.get(int(s),((),(),()))[0]) for s in targets['ids'][:,2]])
    return rows,counts,nt,normalization

def write_split(project,target,split,norm=None):
    folder=preparation();rows,counts,nt,norm=get_records(project,target,split,norm)
    assert np.all(counts>=0);offsets=np.r_[0,np.cumsum(counts)].astype('int64');total=int(offsets[-1])
    arrays={k:np.lib.format.open_memmap(folder/(split+'_'+k+'.npy'),mode='w+',dtype=dt,shape=(total,)) for k,dt in [('types','int32'),('times','float32'),('values','float32')]}
    for i,(types,times,values) in enumerate(rows()):
        assert len(types)==counts[i] and len(types)==len(values)==len(times)
        assert np.isfinite(values).all() and np.all((times>=0)&(times<=24)) and np.all((types>=0)&(types<nt+32))
        order=np.lexsort((types,times));lo,hi=offsets[i:i+2]
        for k,v in [('types',types),('times',times),('values',values)]:arrays[k][lo:hi]=v[order]
    for v in arrays.values():v.flush()
    np.save(folder/(split+'_offsets.npy'),offsets);np.savez_compressed(folder/(split+'_targets.npz'),**target)
    return dict(rows=len(counts),observations=total,max_observations=int(counts.max()),empty_rows=int((counts==0).sum()),num_types=nt,normalization=norm)

def prepare_sequence(project):
    c=expected();folder=preparation();folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():
        r=read(folder/'completion.json')
        for n,h in r['artifacts'].items():assert sha(folder/n)==h,n
        return r
    assert not list(folder.glob('*.npy'));data=Data(project,project/c['input_preparation'])
    train=write_split(project,data.targets['train'],'train');dev=write_split(project,data.targets['dev'],'dev',train['normalization'])
    assert train['num_types']==dev['num_types'];atomic(folder/'normalization.json',train['normalization'])
    r=dict(complete=True,fit_split='train',test_loaded=False,optimizer_updates=0,num_types=train['num_types'],input_dim=train['num_types']+32,
        train={k:v for k,v in train.items() if k!='normalization'},dev={k:v for k,v in dev.items() if k!='normalization'},
        information='All categorical occurrences and all 32 numeric channels before the 24h boundary; no event truncation',
        artifacts={f.name:sha(f) for f in folder.iterdir() if f.suffix in ('.npy','.npz') or f.name=='normalization.json'})
    atomic(folder/'completion.json',r);return r

class SequenceData:
    def __init__(self,project):
        self.project=project;self.folder=preparation();self.record=read(self.folder/'completion.json');self.input_dim=self.record['input_dim'];self.num_types=self.record['num_types']
        self.raw={};self.targets={}
        for s in ('train','dev'):self.load(s)
        self.variant=None
    def load(self,split):
        self.targets[split]=dict(np.load(self.folder/(split+'_targets.npz')))
        self.raw[split]={k:np.load(self.folder/(split+'_'+k+'.npy'),mmap_mode='r') for k in ('types','times','values','offsets')}
    def batch(self,split,ix,device):
        a=self.raw[split]
        if self.variant=='GRU_D':
            # Native 24 hourly bins: within-hour mean of measurements, occurrence
            # indicators for events, separate observation masks and elapsed time.
            x=np.zeros((len(ix),24,self.input_dim),np.float32);count=np.zeros_like(x)
            for b,i in enumerate(ix):
                lo,hi=a['offsets'][i:i+2];j=a['types'][lo:hi];t=np.minimum(a['times'][lo:hi].astype(int),23);v=a['values'][lo:hi]
                np.add.at(x[b],(t,j),v);np.add.at(count[b],(t,j),1)
            mask=count>0;x=np.divide(x,count,out=np.zeros_like(x),where=mask)
            delta=np.zeros_like(x);last=np.zeros((len(ix),self.input_dim),np.float32)
            for t in range(24):delta[:,t]=t+1-last;last=np.where(mask[:,t],t+1,last)
            return tuple(torch.from_numpy(v).to(device) for v in (x,mask.astype('float32'),delta))
        lengths=[int(a['offsets'][i+1]-a['offsets'][i]) for i in ix];length=max(1,max(lengths))
        types=np.zeros((len(ix),length),np.int64);times=np.zeros_like(types,dtype=np.float32);values=times.copy()
        for b,i in enumerate(ix):
            lo,hi=a['offsets'][i:i+2];n=hi-lo;types[b,:n]=a['types'][lo:hi]+1;times[b,:n]=a['times'][lo:hi]/24;values[b,:n]=a['values'][lo:hi]
        return tuple(torch.from_numpy(v).to(device) for v in (types,times,values))
    def test(self):
        import fcntl
        c=expected();source=self.project/c['test_inputs']
        with (self.folder/'test-input.lock').open('a+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            if not (self.folder/'test-completion.json').exists():
                for n,h in c['test_manifest'].items():assert sha(source/n)==(h['sha256'] if isinstance(h,dict) else h)
                target=dict(np.load(source/'test_targets.npz'))
                for s in ('train','dev'):assert not np.intersect1d(target['ids'][:,0],self.targets[s]['ids'][:,0]).size
                report=write_split(self.project,target,'test',read(self.folder/'normalization.json'))
                atomic(self.folder/'test-completion.json',dict(complete=True,optimizer_updates=0,fit_split='train',summary={k:v for k,v in report.items() if k!='normalization'},artifacts={f.name:sha(f) for f in self.folder.glob('test_*') if f.suffix in ('.npy','.npz')}))
        self.load('test')
