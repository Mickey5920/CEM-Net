"""CPU-only paired AUROC/Brier analysis of immutable five-seed predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from cem_probability_intervals import loss_vectors

RUN = 'cem_los_secondary_metrics_20261110'
CORE = 'cem_core_five_seed_20261107'
OLD = 'cem_icde_evidence_20261028'
MATCHED = 'cem_matched_controls_20261105'
SEEDS = (48, 49, 50, 51, 52)
DBS = ('mimiciii', 'mimic', 'eicu')
B, FAMILY, RNG = 20000, 24, 20261110


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2), encoding='utf8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summary(values):
    x = np.asarray(values, float)
    return dict(values=x.tolist(), mean=float(x.mean()), sample_SD=float(x.std(ddof=1)))


def prepared_auc(y, q):
    order = np.argsort(q, kind='stable')
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(q[order]))]
    return order, starts, np.asarray(y[order], float)


def weighted_auc(prepared, weights):
    order, starts, y = prepared
    w = weights[:, order]
    pos = np.add.reduceat(w * y, starts, axis=1)
    neg = np.add.reduceat(w * (1-y), starts, axis=1)
    denom = pos.sum(axis=1) * neg.sum(axis=1)
    assert np.all(denom > 0), 'Undefined AUC: retain failed draw for investigation'
    return (pos * (np.cumsum(neg, axis=1) - .5*neg)).sum(axis=1) / denom


def weights(cluster, replicates, batch=32):
    unique, inverse = np.unique(cluster, return_inverse=True)
    n = len(unique)
    rng = np.random.default_rng(RNG)
    for start in range(0, replicates, batch):
        yield start, np.stack([np.bincount(rng.integers(0,n,n), minlength=n)[inverse]
                              for _ in range(min(batch,replicates-start))]).astype(float)


def inventory(project, db):
    protected = {}
    def keep(path, digest=None):
        p = Path(path).resolve(strict=True)
        assert p.is_relative_to(project)
        value = sha(p)
        assert digest is None or value == digest, str(p)
        protected[str(p)] = value
        return p
    def record(path, digest=None):
        return read(keep(path, digest))
    core_dir = project/'runs'/CORE/db/'statistics_v1'
    core = record(core_dir/'verified-results.json')
    assert core['complete'] and core['seeds'] == list(SEEDS)
    old_dir = project/'runs'/OLD/(db+'_statistics_v1')
    old = record(old_dir/'verified-results.json')
    aliases = {'TabM-mini':'TabM_mini'}
    selected = {e:aliases.get(old['strongest']['dev'][str(e)]['validation_selected'],
                              old['strongest']['dev'][str(e)]['validation_selected']) for e in (0,1)}
    refs = {}
    folders = [old_dir/'bootstrap', project/'runs/cem_five_seed_20261102'/(db+'_statistics_v1')/'bootstrap', core_dir/'bootstrap']
    for folder in folders:
        for path in sorted(folder.glob('*.json')):
            meta = read(path)
            identity = meta.get('identity', {})
            method = aliases.get(identity.get('method'), identity.get('method'))
            if identity.get('split') != 'test' or identity.get('arm','random') != 'random':
                continue
            if method not in {'CEM-Net',*selected.values()}:
                continue
            record(path)
            cells = identity.get('predictions', {})
            if 'prediction' in identity:
                cells = {f"{identity['seed']}:{identity['endpoint']}":identity['prediction']}
            for cell, ref in cells.items():
                seed, endpoint = map(int,cell.split(':'))
                if seed not in SEEDS or method not in ('CEM-Net', selected[endpoint]):
                    continue
                key = method,seed,endpoint
                if key in refs:
                    assert refs[key]['sha256'] == ref['sha256'], key
                refs[key] = dict(path=ref['path'],sha256=ref['sha256'])
    state = record(project/'runs'/MATCHED/('host_eicu' if db=='eicu' else 'host_mimic')/'state.json')
    group = next(g for g in state['groups'].values() if g['database']==db)
    cfg = record(Path(group['source'])/'work'/MATCHED/'config.json')
    target_path = (project/cfg['parent']['test_inputs']/'test_targets.npz').resolve(strict=True)
    target = dict(np.load(keep(target_path,core['protected'][str(target_path)]),allow_pickle=False))
    matched = record(project/'runs'/MATCHED/(db+'_statistics_v1')/'verified-results.json')
    assert matched['complete'] and matched['seeds']==list(SEEDS)
    for job in group['jobs'].values():
        if job.get('variant')!='matched_tabm' or job.get('action')!='train':
            continue
        test = group['jobs']['test_'+job['id']]
        assert test['status']=='completed'
        path = Path(test['output'])
        receipt = record(path/'completion.json')
        assert receipt['complete'] and receipt['optimizer_updates']==0
        refs['matched_tabm',job['seed'],job['endpoint']] = dict(path=str(path/'predictions_test.npz'),sha256=receipt['predictions_sha256'])
    expected = {(method,s,e) for e in (0,1) for method in ('CEM-Net',selected[e],'matched_tabm') for s in SEEDS}
    assert set(refs)==expected
    lookup = {(r['method'],r['seed'],r['endpoint']):r for r in core['rows'] if r['arm']=='random' and r['split']=='test'}
    lookup.update({(r['method'],r['seed'],r['endpoint']):r for r in matched['rows'] if r['split']=='test' and r['method']=='matched_tabm'})
    predictions, losses, rows = {}, {}, []
    for key, ref in sorted(refs.items()):
        method,seed,e = key
        z = dict(np.load(keep(ref['path'],ref['sha256']),allow_pickle=False))
        np.testing.assert_array_equal(z['ids'],target['ids'])
        y,q = z['labels'],z['probs']
        y = y[:,e if y.shape[1]==2 else 0] if y.ndim==2 else y
        q = q[:,e if q.shape[1]==2 else 0] if q.ndim==2 else q
        np.testing.assert_array_equal(y,target['y'][:,e])
        assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
        metrics = dict(AP=float(average_precision_score(y,q)),AUROC=float(roc_auc_score(y,q)),
                       Brier=float(brier_score_loss(y,q)),log_loss=float(log_loss(y,q,labels=[0,1])))
        for name,value in metrics.items():
            assert abs(value-lookup[key][name])<1e-12,(key,name,value,lookup[key][name])
        rows.append(dict(database=db,method=method,seed=seed,endpoint=e,split='test',**metrics))
        predictions[key] = q
        losses[key] = loss_vectors(y,q)['Brier']
    return dict(protected=protected,refs=refs,target=target,predictions=predictions,losses=losses,rows=rows,selected=selected)


def self_test():
    y = np.array([0,1,1,0,1,0])
    q = np.array([.2,.2,.8,.8,.9,.1])
    w = np.array([[1,1,1,1,1,1],[0,2,1,3,1,0],[3,0,2,0,2,1]],float)
    got = weighted_auc(prepared_auc(y,q),w)
    expected = np.array([roc_auc_score(y,q,sample_weight=row) for row in w])
    np.testing.assert_allclose(got,expected,atol=1e-14,rtol=0)
    for row,value in zip(w,got):
        ids=np.repeat(np.arange(len(y)),row.astype(int))
        assert abs(roc_auc_score(y[ids],q[ids])-value)<1e-14
    cluster=np.array([11,11,12,13,14,14])
    a=np.concatenate([v for _,v in weights(cluster,17,batch=5)])
    b=np.concatenate([v for _,v in weights(cluster,17,batch=8)])
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[:,0],a[:,1])
    return dict(complete=True,weighted_AUC_ties_zero_weights=True,integer_replication=True,patient_cluster_batch_invariance=True)


def analyze(project,db,out,mode):
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    owner=(out/'writer.lock').open('a+')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    destination=out/('pilot.json' if mode=='pilot' else 'verified-results.json')
    assert not destination.exists()
    started=time.time(); data=inventory(project,db)
    y=data['target']['y']; cluster=data['target']['ids'][:,0]
    prepared={key:prepared_auc(y[:,key[2]],q) for key,q in data['predictions'].items()}
    keys=[(e,ref,metric) for e in (0,1) for ref in (data['selected'][e],'matched_tabm') for metric in ('AUROC','Brier')]
    lookup={(r['method'],r['seed'],r['endpoint']):r for r in data['rows']}
    loss={}
    for e,ref,metric in keys:
        if metric=='Brier':
            loss[e,ref]=np.mean([data['losses']['CEM-Net',s,e]-data['losses'][ref,s,e] for s in SEEDS],axis=0)
            expected=np.mean([lookup['CEM-Net',s,e]['Brier']-lookup[ref,s,e]['Brier'] for s in SEEDS])
            assert abs(float(loss[e,ref].mean())-expected)<1e-12
    reps=200 if mode=='pilot' else B
    draws={key:np.zeros(reps) for key in keys}
    for start,w in weights(cluster,reps):
        end=start+len(w)
        auc={key:weighted_auc(value,w) for key,value in prepared.items()}
        for e,ref,metric in keys:
            values=(np.mean([auc['CEM-Net',s,e]-auc[ref,s,e] for s in SEEDS],axis=0) if metric=='AUROC'
                    else (w@loss[e,ref])/w.sum(axis=1))
            draws[e,ref,metric][start:end]=values
        if end%1024==0 or end==reps:
            save(out/'progress.json',dict(database=db,mode=mode,draws=end,total=reps,elapsed_seconds=time.time()-started))
    summaries={f'{m}:{e}':{metric:summary([lookup[m,s,e][metric] for s in SEEDS]) for metric in ('AP','AUROC','Brier','log_loss')}
               for e in (0,1) for m in ('CEM-Net',data['selected'][e],'matched_tabm')}
    paired=[]
    for e,ref,metric in keys:
        desc=summary([lookup['CEM-Net',s,e][metric]-lookup[ref,s,e][metric] for s in SEEDS])
        values=draws[e,ref,metric];mean=desc['mean']
        row=dict(database=db,endpoint=e,baseline=ref,metric=metric,**desc,
                 ci95=np.quantile(values,[.025,.975]).tolist(),
                 adjusted_ci=np.quantile(values,[.05/(2*FAMILY),1-.05/(2*FAMILY)]).tolist(),
                 p_bootstrap_centered=float((1+np.count_nonzero(np.abs(values-mean)>=abs(mean)))/(reps+1)))
        row['p_bonferroni']=min(1.,FAMILY*row['p_bootstrap_centered'])
        paired.append(row)
    for path,digest in data['protected'].items():assert sha(path)==digest,path
    result=dict(complete=True,pilot=mode=='pilot',database=db,seeds=list(SEEDS),replicates=reps,family_count=FAMILY,
                rng_seed=RNG,selected_baselines=data['selected'],rows=data['rows'],summaries=summaries,paired=paired,
                protected=data['protected'],prediction_refs={':'.join(map(str,k)):v for k,v in data['refs'].items()},
                samples=len(y),patients=len(np.unique(cluster)),seconds=time.time()-started,self_test=self_test(),
                original_results_unchanged=True,model_fits=0,model_inference=0,calibrator_fits=0,patient_records_exported=False,
                source_sha256=sha(__file__),inference='Test-only, within-seed contrasts averaged across five fits; patient-cluster percentile intervals; one 24-contrast family across 3 databases, 2 endpoints, 2 fixed references and 2 raw metrics.')
    if mode!='pilot':
        np.savez_compressed(out/'paired_draws.npz',**{f't{e}_{ref}_{metric}':v for (e,ref,metric),v in draws.items()})
        result['paired_draws_sha256']=sha(out/'paired_draws.npz')
    save(destination,result)
    print(json.dumps(dict(complete=True,database=db,mode=mode,seconds=result['seconds'],replicates=reps,samples=len(y))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path)
    parser.add_argument('--database',choices=DBS)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--mode',choices=('self-test','pilot','analyze'),required=True)
    a=parser.parse_args()
    if a.mode=='self-test':print(json.dumps(self_test()))
    else:
        p=a.project.resolve(strict=True)
        assert a.output.resolve().is_relative_to(p)
        analyze(p,a.database,a.output,a.mode)
