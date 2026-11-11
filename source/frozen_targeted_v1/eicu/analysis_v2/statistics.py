"""Paired patient inference for the approved targeted contrasts and fixed strata."""
import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

RUN = 'cem_targeted_evidence_20261106'
OLD = 'cem_icde_evidence_20261028'
MATCH = 'cem_matched_controls_20261105'
FIVE = 'cem_five_seed_20261102'
SEEDS = (48,49,50,51,52)
B = 20000


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()
def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp,p)


def api():
    here = Path(__file__).parent
    ns = dict(np=np, average_precision_score=average_precision_score, roc_auc_score=roc_auc_score,
              brier_score_loss=brier_score_loss, log_loss=log_loss)
    for file,names in [('prior_statistics.py',('prepared_ap','weighted_ap','scores')),('prior_significance.py',('centered_p','holm'))]:
        nodes = [n for n in ast.parse((here/file).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(here/file),'exec'),ns)
    return ns


def patient_weights(cluster):
    unique, inverse = np.unique(cluster, return_inverse=True)
    rng = np.random.default_rng(20261027)
    for start in range(0,B,16):
        values = np.stack([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inverse]
                           for _ in range(min(16,B-start))]).astype(float)
        yield start,values


def summarize(values):
    x = np.asarray(values,float)
    assert x.shape == (5,) and np.isfinite(x).all()
    return dict(values=x.tolist(),mean=float(x.mean()),sample_SD=float(x.std(ddof=1)),positive_seeds=int((x>0).sum()))


def prediction(path,targets,endpoint):
    values = dict(np.load(path,allow_pickle=False))
    np.testing.assert_array_equal(values['ids'],targets['ids'])
    y,q = values['labels'],values['probs']
    y = y[:,endpoint if y.shape[1]==2 else 0] if y.ndim==2 else y
    q = q[:,endpoint if q.shape[1]==2 else 0] if q.ndim==2 else q
    np.testing.assert_array_equal(y,targets['y'][:,endpoint])
    assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    return q


def bootstrap_job(task):
    functions = api()
    target = Path(task['output'])
    identity = {k:v for k,v in task.items() if k!='output'}
    if target.with_suffix('.json').exists():
        meta = read(target.with_suffix('.json'))
        assert meta['identity']==identity and sha(target)==meta['sha256']
        return dict(key=task['key'],reused=True)
    targets = dict(np.load(task['targets'],allow_pickle=False))
    q = prediction(task['prediction'],targets,task['endpoint'])
    y = targets['y'][:,task['endpoint']]
    groups = dict(np.load(task['groups'],allow_pickle=False))
    np.testing.assert_array_equal(groups['ids'],targets['ids'])
    masks = {'all':np.ones(len(y),bool)}
    if task['subgroups']:
        masks.update(low=groups['group']==0,high=groups['group']==1)
    for mask in masks.values():
        assert y[mask].sum()>=20 and (1-y[mask]).sum()>=20
    values = {}
    cached = task.get('whole_cache')
    if cached:
        assert sha(cached['path'])==cached['sha256']
        values['all'] = np.load(cached['path'],allow_pickle=False)[cached['key']].copy()
        assert values['all'].shape==(B,)
    for key in masks:
        if key not in values: values[key]=np.empty(B)
    prepared = {key:functions['prepared_ap'](y[mask],q[mask]) for key,mask in masks.items()}
    todo = [key for key in masks if key!='all' or not cached]
    for start,weights in patient_weights(targets['ids'][:,0]):
        if start==0 and cached:
            np.testing.assert_allclose(functions['weighted_ap'](prepared['all'],weights),values['all'][:len(weights)],rtol=0,atol=1e-12)
        for key in todo:
            values[key][start:start+len(weights)] = functions['weighted_ap'](prepared[key],weights[:,masks[key]])
        if not todo: break
    assert all(np.isfinite(v).all() for v in values.values())
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists()
    with target.open('xb') as f: np.savez_compressed(f,**values)
    save(target.with_suffix('.json'),dict(identity=identity,sha256=sha(target),whole_cohort_cache_reused=bool(cached)))
    return dict(key=task['key'],reused=False)


def audit(project,db,source,out,allow_pending=False):
    functions = api()
    c = read(source/'work'/RUN/'config.json')
    parent = c['parent']
    state_path = project/'runs'/RUN/('host_eicu' if db=='eicu' else 'host_mimic')/'state.json'
    state = read(state_path)
    group = next(g for g in state['groups'].values() if g['database']==db)
    assert allow_pending or group['status']=='complete'
    assert group['source']==str(source) and sha(source/'source_manifest.json')==group['source_manifest_sha256']
    protected = {}
    def keep(path,digest=None):
        path=Path(path).resolve(strict=True)
        assert path.is_relative_to(project)
        value=sha(path)
        assert digest is None or value==digest,str(path)
        protected[str(path)]=value
        return path
    keep(source/'source_manifest.json',group['source_manifest_sha256'])
    for name,digest in read(source/'source_manifest.json').items():keep(source/name,digest)
    verification = read(keep(Path(group['output']).parent/'verification.json'))
    for path,digest in verification['original_protected'].items():keep(path,digest)
    lock = None
    if not allow_pending:
        lock = read(keep(Path(group['output'])/'fixed-checkpoint-lock.json'))
        assert lock['complete'] and len(lock['selections'])==17 and not lock['test_used_for_selection']
    target_path = keep(project/parent['test_inputs']/'test_targets.npz')
    targets = dict(np.load(target_path,allow_pickle=False))
    overlay = project/c['overlay']
    receipt = read(keep(overlay/'completion.json',verification['overlay_sha256']))
    groups_path = keep(overlay/'test_groups.npz',receipt['artifacts']['test_groups.npz']['sha256'])
    assert receipt['threshold_fit_split']=='train' and not receipt['labels_used']
    old = project/'runs'/OLD/(db+'_statistics_v1')/'bootstrap'
    five = project/'runs'/FIVE/(db+'_statistics_v1')/'bootstrap'
    matched = project/'runs'/MATCH/(db+'_statistics_v1')/'bootstrap'
    refs = {}
    def add(method,s,e,ref,cache=None):
        path = keep(ref['path'],ref['sha256'])
        key = f'{method}:{e}:{s}'
        if cache:keep(cache['path'],cache['sha256'])
        refs[key] = dict(key=key,method=method,endpoint=e,seed=s,prediction=str(path),prediction_sha256=sha(path),
                         targets=str(target_path),targets_sha256=sha(target_path),groups=str(groups_path),groups_sha256=sha(groups_path),
                         whole_cache=cache,subgroups=method in ('CEM-Net','window','observation'),B=B,rng_seed=20261027,
                         output=str(out/'bootstrap'/(key.replace(':','_')+'.npz')))
    meta = read(keep(old/'random_test_CEM_Net.json'))
    assert meta['identity']['replicates']==B and meta['identity']['rng_seed']==20261027
    for e in (0,1):
        for s in SEEDS:
            if s<=50:
                add('CEM-Net',s,e,meta['identity']['predictions'][f'{s}:{e}'],dict(path=str(old/'random_test_CEM_Net.npz'),sha256=meta['draws_sha256'],key=f'{s}:{e}'))
            else:
                m = read(keep(five/f'test_CEM-Net_s{s}_t{e}.json'))
                add('CEM-Net',s,e,m['identity']['prediction'],dict(path=str(five/f'test_CEM-Net_s{s}_t{e}.npz'),sha256=m['sha256'],key='AP'))
            for variant,method in [('window_summary','window'),('matched_tabm','matched_full')]:
                if method=='matched_full' and e!=c['evidence_endpoint']:continue
                f = matched/f'{variant}_s{s}_t{e}.json'
                m = read(keep(f))
                add(method,s,e,m['identity']['prediction'],dict(path=str(f.with_suffix('.npz')),sha256=m['sha256'],key='AP'))
    basicmeta = read(keep(old/'random_test_CEM_basic.json'))
    e = c['evidence_endpoint']
    for s in (48,49,50):
        add('CEM_basic',s,e,basicmeta['identity']['predictions'][f'{s}:{e}'],dict(path=str(old/'random_test_CEM_basic.npz'),sha256=basicmeta['draws_sha256'],key=f'{s}:{e}'))
    fits = []
    for job in group['jobs'].values():
        if allow_pending:break
        if job['action']!='train':continue
        testjob = group['jobs']['test_'+job['id']]
        assert job['status']==testjob['status']=='completed'
        for j in (job,testjob):assert read(keep(j['exit_receipt']))['returncode']==0
        fit,test = Path(job['output']),Path(testjob['output'])
        r,tc = read(keep(fit/'completion.json')),read(keep(test/'completion.json'))
        manifest = read(keep(fit/'run_manifest.json'))
        history = read(keep(fit/'history.json'))
        assert not r['test_loaded'] and r['from_scratch'] and not r['pilot']
        assert len(history)==r['epochs']==manifest['epoch_budget']
        assert r['best_epoch']==1+int(np.argmax([x['dev_ap'] for x in history]))
        assert r['optimizer_updates']==r['ema_updates']==r['epochs']*math.ceil(len(np.load(Path(parent['raw_inputs'])/'train_targets.npz')['ids'])/256)
        assert manifest['source_manifest_sha256']==group['source_manifest_sha256']
        assert r['checkpoint_sha256']==tc['model_sha256']==lock['selections'][job['id']]['checkpoint_sha256']
        keep(fit/'best.pt',r['checkpoint_sha256'])
        assert tc['optimizer_updates']==0 and not tc['parameters_updated'] and tc['dev_replay_max_drift']<=1e-5
        method={'CEM_basic':'CEM_basic','matched_basic':'matched_basic','observation_weighted':'observation'}[job['variant']]
        add(method,job['seed'],job['endpoint'],dict(path=str(test/'predictions_test.npz'),sha256=tc['predictions_sha256']))
        fits.append(dict(id=job['id'],epochs=r['epochs'],best_epoch=r['best_epoch'],seconds=r['seconds'],parameters=r['parameters']))
    assert (len(fits),len(refs))==((0,28) if allow_pending else (17,45))
    return dict(config=c,protected=protected,refs=refs,targets=targets,grouping=receipt,groups=np.load(groups_path)['group'],fits=fits)


def analyze(project,db,source,out,precompute=False):
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    owner=(out/'writer.lock').open('a+')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'completion.json').exists():
        for n,h in read(out/'completion.json')['files'].items():assert sha(out/n)==h
        return dict(complete=True,reused=True)
    save(out/'progress.json',dict(stage='audit'))
    data=audit(project,db,source,out,allow_pending=precompute)
    tasks=list(data['refs'].values())
    save(out/('precompute_plan.json' if precompute else 'bootstrap_plan.json'),dict(tasks=tasks,protocol=data['config']['protocol']))
    completed=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(bootstrap_job,task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();completed+=1
            save(out/'progress.json',dict(stage='paired_bootstrap',completed=completed,total=len(tasks),last=result,time=time.time()))
    if precompute:
        for path,digest in data['protected'].items():assert sha(path)==digest,path
        save(out/'precompute_completion.json',dict(complete=True,tasks=len(tasks),training_runs=0,original_results_unchanged=True))
        return dict(complete=True,precompute=True,tasks=len(tasks))
    functions=api();rows=[];draws={};metrics={};group_rows=[]
    for key,ref in data['refs'].items():
        q=prediction(ref['prediction'],data['targets'],ref['endpoint'])
        y=data['targets']['y'][:,ref['endpoint']]
        m=functions['scores'](y,q);metrics[key,'all']=m['AP']
        rows.append(dict(database=db,method=ref['method'],endpoint=ref['endpoint'],seed=ref['seed'],**m))
        with np.load(ref['output'],allow_pickle=False) as values:
            for name in values.files:draws[key,name]=values[name]
        if ref['subgroups']:
            for group,label in [(0,'low'),(1,'high'),(2,'ineligible')]:
                mask=data['groups']==group
                if not mask.any():continue
                ap=float(average_precision_score(y[mask],q[mask])) if y[mask].sum()>0 else None
                if ap is not None:metrics[key,label]=ap
                group_rows.append(dict(database=db,method=ref['method'],endpoint=ref['endpoint'],seed=ref['seed'],group=label,
                                       stays=int(mask.sum()),patients=len(np.unique(data['targets']['ids'][mask,0])),positives=int(y[mask].sum()),
                                       prevalence=float(y[mask].mean()),AP=ap))
    paired=[];paired_draws={}
    def contrast(name,endpoint,terms,family,count):
        samples=np.mean([sum(weight*draws[f'{method}:{endpoint}:{seed}',group] for method,group,weight in terms) for seed in SEEDS],axis=0)
        values=[sum(weight*metrics[f'{method}:{endpoint}:{seed}',group] for method,group,weight in terms) for seed in SEEDS]
        observed=float(np.mean(values))
        key=f'{name}:{endpoint}'
        paired_draws[key]=samples
        row=dict(database=db,name=name,endpoint=endpoint,family=family,family_size=count,terms=terms,**summarize(values),
                 ci95=np.quantile(samples,[.025,.975]).tolist(),adjusted_ci=np.quantile(samples,[.05/(2*count),1-.05/(2*count)]).tolist(),
                 p_bootstrap_centered=functions['centered_p'](samples,observed))
        paired.append(row)
    e=data['config']['evidence_endpoint']
    contrast('CEM_full_minus_basic',e,[('CEM-Net','all',1),('CEM_basic','all',-1)],'evidence',9)
    contrast('matched_full_minus_basic',e,[('matched_full','all',1),('matched_basic','all',-1)],'evidence',9)
    contrast('evidence_interaction',e,[('CEM-Net','all',1),('CEM_basic','all',-1),('matched_full','all',-1),('matched_basic','all',1)],'evidence',9)
    for e in (0,1):
        contrast('duration_minus_observation',e,[('CEM-Net','all',1),('observation','all',-1)],'aggregation',6)
        for other in ('observation','window'):
            for group in ('low','high'):
                contrast(f'duration_minus_{other}_{group}',e,[('CEM-Net',group,1),(other,group,-1)],'subgroup_simple',24)
            contrast(f'duration_minus_{other}_high_minus_low',e,[('CEM-Net','high',1),(other,'high',-1),('CEM-Net','low',-1),(other,'low',1)],'subgroup_interaction',12)
    summaries={}
    for method,e in {(r['method'],r['endpoint']) for r in rows}:
        select=sorted([r for r in rows if r['method']==method and r['endpoint']==e],key=lambda r:r['seed'])
        summaries[f'{method}:{e}']={metric:summarize([r[metric] for r in select]) for metric in ('AP','AUROC','Brier','log_loss')}
    for path,digest in data['protected'].items():assert sha(path)==digest,path
    with (out/'paired_draws.npz').open('xb') as f:np.savez_compressed(f,**paired_draws)
    result=dict(complete=True,database=db,new_fits=17,reused_CEM_basic_fits=3,rows=rows,summaries=summaries,paired=paired,
                subgroups=group_rows,threshold=data['grouping']['threshold'],grouping_definition=data['grouping']['discrepancy'],
                grouping_counts=data['grouping']['grouping'],fit_audits=data['fits'],protected=data['protected'],
                protocol=data['config']['protocol'],original_results_unchanged=True,original_manuscript_unchanged=True)
    save(out/'verified-results.json',result)
    save(out/'completion.json',dict(complete=True,files={n:sha(out/n) for n in ('verified-results.json','paired_draws.npz')}))
    return dict(complete=True,database=db,new_fits=17,contrasts=len(paired))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--database',choices=('mimiciii','mimic','eicu'),required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precompute',action='store_true')
    a=p.parse_args()
    project,source,out=a.project.resolve(strict=True),a.source.resolve(strict=True),a.output.resolve()
    assert source.is_relative_to(project) and out.is_relative_to(project)
    print(json.dumps(analyze(project,a.database,source,out,precompute=a.precompute)))
