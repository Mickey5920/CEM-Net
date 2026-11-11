"""Locked inference and matched-device timing; never constructs an optimizer."""
import argparse, gc, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

RUN='cem_icde_evidence_20261028'
OLD='cem_evidence_completion_20261025'
CONDITIONS=('lab_drop25','lab_drop50','lab_delay2h','lab_delay6h')

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.writing');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def state_hash(model):
    h=hashlib.sha256()
    for n,t in sorted(model.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def selection(project,db,cfg):
    """Freeze the comparator using only already-completed Validation results."""
    BASELINES=('RealMLP','TabM','TabM-mini','TabM_mini_PLE','TabM_PLE','SeFT','GRU_D','mTAND_Enc','LightGBM_full')
    oldfolder=project/'runs'/OLD/db/'comparison_v1';old=read(oldfolder/'verified-results.json');assert old['complete']
    devrows=[r for r in old['rows'] if r['split']=='dev' and r['method'] in BASELINES]
    host=project/'runs'/RUN/('host_eicu' if db=='eicu' else 'host_mimic');s=read(host/'state.json')
    group=next(g for g in s['groups'].values() if g['database']==db and g['stage']=='lgbm')
    for j in group['jobs'].values():
        if j['action']!='train':continue
        assert j['status']=='completed' and read(j['exit_receipt'])['returncode']==0
        r=read(Path(j['output'])/'completion.json');assert r['complete'] and not r['test_loaded']
        devrows.append(dict(method='LightGBM_full',seed=j['seed'],endpoint=j['endpoint'],AP=r['dev_ap']))
    selected={}
    for e in (0,1):
        means={m:float(np.mean([next(r['AP'] for r in devrows if (r['method'],r['seed'],r['endpoint'])==(m,s,e)) for s in (48,49,50)])) for m in BASELINES}
        selected[str(e)]=dict(method=max(BASELINES,key=lambda m:means[m]),means=means)
    # The supported families are asserted, never silently substituted.
    assert all(v['method'] in ('TabM_PLE','TabM_mini_PLE','TabM-mini','LightGBM_full') for v in selected.values()),selected
    jobs=[]
    for e in (0,1):
        for method in ('CEM-Net',selected[str(e)]['method']):
            if method!='LightGBM_full':meta=read(oldfolder/'bootstrap_cache'/('dev_'+method.replace('-','_')+'.json'))['identity']
            for seed in (48,49,50):
                fit=Path(cfg['full_ema'][f'{seed}:{e}']['output']) if method=='CEM-Net' else Path(next(j['output'] for j in group['jobs'].values() if j['action']=='train' and j['seed']==seed and j['endpoint']==e)) if method=='LightGBM_full' else Path(meta['predictions'][f'{seed}:{e}']['path']).parent
                completion=read(fit/'completion.json');checkpoint=fit/completion['checkpoint_file'];assert sha(checkpoint)==completion['checkpoint_sha256']
                jobs.append(dict(method=method,seed=seed,endpoint=e,fit=str(fit),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),fit_completion_sha256=sha(fit/'completion.json')))
    return dict(complete=True,database=db,selection='maximum unrounded three-seed Validation AP mean across nine fixed baselines',test_used_for_selection=False,
                pool=list(BASELINES),selected=selected,jobs=jobs,formal_training_launched=False)

def make_model(project,db,cfg,job,out,data):
    method=job['method'];seed=job['seed'];e=job['endpoint'];old=None
    if method=='CEM-Net':
        if db=='mimiciii':from scripts.mimiciii_ema_common import fresh_model
        else:from scripts.ema_cross_common import fresh_model
        model=fresh_model(data,project/cfg['input_preparation'],seed,e,out)
    elif method=='LightGBM_full':
        import lightgbm as lgb
        model=lgb.Booster(model_file=job['checkpoint']);return model,None
    else:
        import joblib
        from scripts.train_tracehr_unified import seed_all
        seed_all(seed);old=joblib.load(cfg['tabm_preprocessing'])
        if method=='TabM-mini':
            from tabm import TabM
            model=TabM.make(n_num_features=data.input_dim,d_out=1,k=32,d_block=256,n_blocks=3,dropout=.2,arch_type='tabm-mini')
        else:
            from models.cem_additional_baselines import tabm_model
            bins=torch.load(project/cfg['baseline_preparation']/'bins.pt',map_location='cpu',weights_only=False)
            model=tabm_model(method,bins)
    checkpoint=torch.load(job['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint['model_state_dict'],strict=True)
    del checkpoint
    return model.eval(),old

def apply_transform(raw,old):
    if old is None:return np.array(raw,copy=True)
    from scripts.cem_additional_common import transform
    return transform(raw,old)

@torch.inference_mode()
def predict(model,raw,method,old,device,micro):
    if method=='LightGBM_full':return np.asarray(model.predict(raw,num_threads=4),float)
    parts=[]
    for i in range(0,len(raw),micro):
        x=torch.from_numpy(apply_transform(raw[i:i+micro],old)).to(device)
        z=model({'raw':x}) if method=='CEM-Net' else model(x)
        q=z.softmax(-1)[...,1].mean(1) if method=='CEM-Net' else z.sigmoid().squeeze(-1).mean(1)
        parts.append(q.cpu().numpy())
    return np.concatenate(parts)

def benchmark(model,raw,method,old,device):
    """Resident dense features vs transform+transfer+model; same frozen rows/device."""
    assert method!='LightGBM_full'
    model.eval();rows=[]
    with torch.inference_mode():
        for n in (1,8,32):
            block=np.array(raw[:n],copy=True);x=torch.from_numpy(apply_transform(block,old)).to(device)
            def forward():
                z=model({'raw':x}) if method=='CEM-Net' else model(x)
                return z.softmax(-1)[...,1].mean(1) if method=='CEM-Net' else z.sigmoid().squeeze(-1).mean(1)
            for _ in range(10):forward()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();dense=[];end=[]
            for _ in range(30):
                t=time.perf_counter();forward();torch.cuda.synchronize();dense.append(time.perf_counter()-t)
                t=time.perf_counter();predict(model,block,method,old,device,n);torch.cuda.synchronize();end.append(time.perf_counter()-t)
            def report(a):return dict(median_ms=float(np.median(a)*1000),p95_ms=float(np.quantile(a,.95)*1000),records_per_second=float(n/np.median(a)),repeats=len(a))
            rows.append(dict(batch=n,resident_model=report(dense),dense_transform_transfer_model=report(end),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    return dict(device=torch.cuda.get_device_name(),precision='FP32; TF32 disabled',threads=4,warmup=10,rows=rows,
                parameters=sum(p.numel() for p in model.parameters()),scope='warm dense-feature inference; source-table scan and evidence extraction measured separately; no training updates')

def run_job(project,db,job,probe=False):
    cfg=read(Path(__file__).with_name('config.json'));out=project/'runs'/RUN/db/'frozen_inference_v2'/('probes' if probe else 'jobs')/(job['method']+f"_t{job['endpoint']}_s{job['seed']}")
    if (out/'completion.json').exists():
        r=read(out/'completion.json');assert r['complete'] and r['job']==job
        for n,h in r['artifacts'].items():assert sha(out/n)==h
        return r
    assert not out.exists(),'Partial inference must be inspected before any retry';out.mkdir(parents=True)
    assert sha(job['checkpoint'])==job['checkpoint_sha256'] and sha(Path(job['fit'])/'completion.json')==job['fit_completion_sha256']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from scripts.cem_evidence_common import Data
    data=Data(project,project/cfg['input_preparation']);e=job['endpoint'];method=job['method'];device='cpu' if probe or method=='LightGBM_full' else 'cuda'
    save(out/'progress.json',dict(stage='loading_locked_checkpoint',optimizer_updates=0))
    model,old=make_model(project,db,cfg,job,out,data)
    if method!='LightGBM_full':model=model.to(device);before=state_hash(model)
    micro=256 if method=='CEM-Net' else 32
    expected=dict(np.load(Path(job['fit'])/'predictions_dev.npz'));np.testing.assert_array_equal(expected['ids'],data.targets['dev']['ids'])
    ix=np.arange(min(32,len(expected['ids']))) if probe else np.arange(len(expected['ids']))
    q=predict(model,data.raw['dev'][ix],method,old,device,micro);ref=expected['probs'][ix];ref=ref[:,e if ref.shape[1]==2 else 0] if ref.ndim==2 else ref
    drift=float(np.max(np.abs(q-ref)));assert drift<=1e-5,(method,drift)
    artifacts={};results=[];clean_drift=None;timing=None
    if not probe:
        folder=project/cfg['test_inputs'];target=dict(np.load(folder/'test_targets.npz'));raw=np.load(folder/'test_raw.npy',mmap_mode='r')
        for split in ('train','dev'):assert not np.intersect1d(target['ids'][:,0],data.targets[split]['ids'][:,0]).size
        meta=read(project/'runs'/OLD/db/'comparison_v1/bootstrap_cache'/('test_'+method.replace('-','_')+'.json'))['identity'] if method!='LightGBM_full' else None
        if meta:
            r=meta['predictions'][f"{job['seed']}:{e}"];assert sha(r['path'])==r['sha256'];z=dict(np.load(r['path']));np.testing.assert_array_equal(z['ids'],target['ids'])
            saved=z['probs'];saved=saved[:,e if saved.shape[1]==2 else 0] if saved.ndim==2 else saved
        else:
            folderfit=Path(job['fit']);z=dict(np.load(folderfit.parent.parent/'test'/('test_'+folderfit.name)/'predictions_test.npz'));saved=z['probs'][:,0];np.testing.assert_array_equal(z['ids'],target['ids'])
        clean=predict(model,raw,method,old,device,micro);clean_drift=float(np.max(np.abs(clean-saved)));assert clean_drift<=1e-5
        stress=project/'runs'/RUN/db/'stress_inputs_v2';receipt=read(stress/'completion.json');assert receipt['complete'] and sha(folder/'test_targets.npz')==receipt['targets']['sha256']
        for condition in CONDITIONS:
            source=stress/(condition+'.npy');assert sha(source)==receipt['artifacts'][condition]['sha256'];changed=np.load(source,mmap_mode='r');assert changed.shape==raw.shape
            save(out/'progress.json',dict(stage='fixed_stress_inference',condition=condition,optimizer_updates=0));t=time.perf_counter();pred=predict(model,changed,method,old,device,micro)
            assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();name=condition+'.npz';np.savez_compressed(out/name,ids=target['ids'],labels=target['y'][:,e],probs=pred)
            artifacts[name]=sha(out/name);results.append(dict(condition=condition,AP=float(average_precision_score(target['y'][:,e],pred)),AUROC=float(roc_auc_score(target['y'][:,e],pred)),seconds=time.perf_counter()-t,rows=len(pred)))
        if method!='LightGBM_full':timing=benchmark(model,raw,method,old,device)
    if method!='LightGBM_full':assert state_hash(model)==before
    assert sha(job['checkpoint'])==job['checkpoint_sha256']
    result=dict(complete=True,job=job,probe=probe,optimizer_updates=0,parameters_updated=False,dev_replay_max_drift=drift,clean_test_replay_max_drift=clean_drift,
                results=results,timing=timing,artifacts=artifacts,source_sha256=sha(__file__),checkpoint_hash_unchanged=True,test_loaded=not probe)
    save(out/'completion.json',result);save(out/'progress.json',dict(stage='complete'));del model,data;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--database',required=True);p.add_argument('--mode',choices=('select','probe','job'),required=True);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    project=a.project.resolve(strict=True);cfg=read(Path(__file__).with_name('config.json'));folder=project/'runs'/RUN/a.database/'frozen_inference_v2';folder.mkdir(parents=True,exist_ok=True)
    if a.mode=='select':
        value=selection(project,a.database,cfg);f=folder/'validation-selection.json'
        assert value==read(folder.parent/'frozen_inference_v1/validation-selection.json'),'Execution-only repair must not change the frozen comparator'
        if f.exists():assert read(f)==value
        else:save(f,value)
        print(json.dumps(value))
    else:
        contract=read(folder/'validation-selection.json');assert contract['complete'] and not contract['test_used_for_selection'];print(json.dumps(run_job(project,a.database,contract['jobs'][a.index],a.mode=='probe')))
