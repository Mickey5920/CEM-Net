"""Fixed input contracts for the authorized additional clinical comparators."""
import importlib.util
import os
from pathlib import Path
import numpy as np
import torch
from scripts.tracehr_multiview_common import read, sha, atomic
from scripts.cem_evidence_common import Data

ROOT=Path(__file__).resolve().parents[1]
RUN='cem_evidence_completion_20261025'

def expected():
    return read(ROOT/'work'/RUN/(os.environ['TRACEHR_DATABASE']+'_'+os.environ['CEM_COMPLETION_STAGE']+'_expected.json'))

def preparation():return Path(os.environ['TRACEHR_PROJECT'])/expected()['baseline_preparation']

def transform(raw, old):
    z=np.array(raw,copy=True);d=old['phi_dim']
    v=(z[:,:d]-old['phi_center'])*old['phi_factor'];v/=np.sqrt(1+(v/3)**2)
    v[:,old['binary']]=z[:,:d][:,old['binary']];z[:,:d]=v
    z[:,d:]=old['numeric_quantile'].transform(z[:,d:])
    assert np.isfinite(z).all();return z

def prepare_tabular(project):
    import joblib, tabm, rtdl_num_embeddings as rtdl
    c=expected();folder=preparation();folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():
        r=read(folder/'completion.json')
        for n,h in r['artifacts'].items():assert sha(folder/n)==h,n
        return r
    assert not list(folder.glob('*.npy'))
    data=Data(project,project/c['input_preparation']);old=joblib.load(c['tabm_preprocessing'])
    assert old['fit_split']=='train' and sha(c['tabm_preprocessing'])==c['tabm_preprocessing_sha256']
    for s in ('train','dev'):
        z=np.lib.format.open_memmap(folder/(s+'.npy'),mode='w+',dtype='float32',shape=data.raw[s].shape)
        for start in range(0,len(z),1024):z[start:start+1024]=transform(data.raw[s][start:start+1024],old)
        z.flush();del z
    raw=np.load(folder/'train.npy',mmap_mode='r');bins=[];constant=[]
    # Every original feature is retained. Constant columns get a deterministic
    # unit-width interval; all nonconstant edges are train-only quantiles.
    for start in range(0,raw.shape[1],64):
        x=torch.from_numpy(np.array(raw[:,start:start+64],copy=True))
        edges=torch.quantile(x,torch.linspace(0,1,49),dim=0).T
        for j,q in enumerate(edges):
            q=q.unique()
            if len(q)==1:constant.append(start+j);q=torch.stack([q[0]-.5,q[0]+.5])
            bins.append(q)
    torch.save(bins,folder/'bins.pt')
    result=dict(complete=True,fit_split='train',test_loaded=False,optimizer_updates=0,
        input_dim=data.input_dim,train_rows=len(raw),constant_columns=constant,
        bin_rule='48 equal-mass quantiles, remove duplicate edges; unit interval for constant columns',
        preprocessing_sha256=sha(c['tabm_preprocessing']),
        package_files={getattr(m.__loader__,'archive',m.__file__):sha(getattr(m.__loader__,'archive',m.__file__)) for m in (tabm,rtdl)},
        artifacts={n:sha(folder/n) for n in ('train.npy','dev.npy','bins.pt')})
    atomic(folder/'completion.json',result);return result

class TabularData:
    def __init__(self,project):
        self.original=Data(project,project/expected()['input_preparation']);self.targets=self.original.targets
        self.raw={s:np.load(preparation()/(s+'.npy'),mmap_mode='r') for s in ('train','dev')}
        self.input_dim=self.original.input_dim
    def batch(self,split,ix,device):
        return (torch.from_numpy(np.array(self.raw[split][ix],copy=True)).to(device),)
    def test(self):
        import joblib
        c=expected();folder=Path(os.environ['TRACEHR_PROJECT'])/c['test_inputs']
        for n,h in c['test_manifest'].items():assert sha(folder/n)==(h['sha256'] if isinstance(h,dict) else h),n
        target=dict(np.load(folder/'test_targets.npz'));raw=np.load(folder/'test_raw.npy',mmap_mode='r')
        for s in ('train','dev'):assert not np.intersect1d(target['ids'][:,0],self.targets[s]['ids'][:,0]).size
        self.targets['test']=target;self.raw['test']=raw
        old=joblib.load(c['tabm_preprocessing']);original_batch=self.batch
        def test_batch(split,ix,device):
            if split!='test':return original_batch(split,ix,device)
            return (torch.from_numpy(transform(raw[ix],old)).to(device),)
        self.batch=test_batch

def protocol(variant):
    return expected()['recipes'][variant]

def make_data(project):
    if expected()['stage']=='sequence':
        from scripts.cem_sequence_inputs import SequenceData
        return SequenceData(project)
    return TabularData(project)

def cpu_checks():
    if expected()['stage']=='sequence':
        from models.cem_sequence_baselines import cpu_checks as checks
        return checks()
    from models.cem_additional_baselines import tabm_model
    import rtdl_num_embeddings as rtdl
    x=torch.tensor([[0.,1.],[1.,2.],[2.,3.],[3.,4.]])
    bins=rtdl.compute_bins(x,n_bins=3)
    for variant in ('TabM_PLE','TabM_mini_PLE'):
        m=tabm_model(variant,bins,small=True);z=m(x)
        assert z.shape==(4,32,1) and torch.isfinite(z).all()
        z.square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    return ['official_TabM_and_TabM_mini_with_PLE_B','train_only_quantile_edges','all_input_coordinates_retained','finite_forward_and_connected_gradients']
