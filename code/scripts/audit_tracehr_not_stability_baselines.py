"""Read-only audit of archived train/dev controls before the first 28 fits."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.writing');t.write_text(json.dumps(v,indent=2,allow_nan=False));os.replace(t,p)

def audit(project,output):
    p=Path(project).resolve(strict=True);out=Path(output).resolve()
    assert out.is_relative_to(p) and out!=p
    expected=read(ROOT/'work/tracehr_not_stability_20261018/baseline_expected.json')
    protected={}
    def protect(path,digest=None):
        path=Path(path).resolve(strict=True);assert path.is_relative_to(p)
        actual=sha(path)
        if digest is not None:assert actual==digest,(str(path),'hash differs')
        protected[str(path)]=actual;return actual
    prep_path=p/'runs/tracehr_structure_20261014/mimic/g_v1/preparation/completion.json'
    prep=read(prep_path);protect(prep_path)
    assert prep['preparation_complete'] and not prep['test_loaded']
    folder=Path(prep['input_pointer']['path'])
    for name,rec in prep['input_artifacts'].items():
        assert name.startswith(('train_','dev_'));protect(folder/name,rec['sha256'])
    target=dict(np.load(folder/'dev_targets.npz'));training=dict(np.load(folder/'train_targets.npz'))
    ids,y=target['ids'],target['y'];assert not np.intersect1d(ids[:,0],training['ids'][:,0]).size
    result=dict(complete=False,source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        started_at=time.time(),test_loaded=False,patient_data_exported=False,optimizer_updates=0,
        new_training_started=False,dev_rows=len(ids),dev_patients=len(np.unique(ids[:,0])),
        baseline_archive=expected['provenance'],baselines={},controls={},native_archive_agreement={})
    # Each previous aggregate was checked on its own immutable training snapshot.
    summaries=[]
    for rel in ('tracehr_not_confirmation_20261017/results_v1/aggregates.json',
                'tracehr_not_functional_20261018/results_v1/aggregates.json'):
        path=p/'runs'/rel;summary=read(path);protect(path)
        assert summary['complete'] and summary['protected_files_unchanged'] and not summary['test_loaded']
        summaries.append(summary)
    entries=[(v,e,s) for s in (42,43,44) for v in ('C5_noT','B0') for e in (0,1)]
    entries += [(v,e,42) for v in ('C5_noT_noP','C5_noT_noM','C5_noT_noA','C5_noT_noR') for e in (0,1)]
    control_predictions={}
    for v,e,s in entries:
        key=f'{v}_t{e}_s{s}';row=next(a['records'][key] for a in reversed(summaries) if key in a['records'])
        path=Path(row['source']);c=read(path/'completion.json');m=read(path/'run_manifest.json')
        receipt=path.parent.parent/(path.name+'_exit.json');exitrow=read(receipt)
        assert exitrow['returncode']==0 and exitrow['source_manifest_sha256']==c['source_manifest_sha256']
        assert c['complete'] and c['from_scratch'] and not c['pilot'] and not c['test_loaded']
        assert c['seed']==s and c['endpoint']==e and c['frozen_learnable_parameters']==0
        protect(receipt);protect(path/'completion.json');protect(path/'run_manifest.json')
        protect(path/c.get('checkpoint_file','best.pt'),c['checkpoint_sha256'])
        protect(path/'predictions_dev.npz',c['predictions_sha256'])
        assert row['checkpoint_sha256']==c['checkpoint_sha256'] and row['predictions_sha256']==c['predictions_sha256']
        pr=dict(np.load(path/'predictions_dev.npz'))
        assert np.array_equal(pr['ids'],ids) and np.array_equal(pr['labels'],y[:,e:e+1])
        assert pr['probs'].shape==(len(ids),1) and np.isfinite(pr['probs']).all()
        ap=float(average_precision_score(y[:,e],pr['probs'][:,0]))
        assert abs(ap-c['dev_ap'])<1e-12 and abs(ap-row['AP'])<1e-12
        result['controls'][key]=dict(path=str(path),variant=v,endpoint=e,seed=s,dev_ap=ap,
            initial_weights_sha256=m['initial_weights_sha256'],checkpoint_sha256=c['checkpoint_sha256'],
            predictions_sha256=c['predictions_sha256'],source_manifest_sha256=c['source_manifest_sha256'],
            best_epoch=c['best_epoch'],exit_returncode=0)
        if v=='B0':control_predictions[(s,e)]=pr['probs'][:,0]
    recipes={};cached_queues={}
    for row in expected['rows']:
        family,s,cfg=row['family'],int(row['seed']),int(row['config'])
        if s==42:
            tree=family in ('lightgbm','xgboost')
            rel='tracehr_expansion_cpu_20261004/mimic/round1' if tree else 'tracehr_tabular_20261004/mimic/round1_r2'
            run=f'mimic24_{family}'+('_F2' if tree else '')+f'_cfg{cfg}_full_s{s}'
        else:
            rel='tracehr_expansion_confirm_20261004/mimic/round1'
            run=f'mimic24_{family}_F2_cfg{cfg}_full_s{s}'
        queue=p/'runs'/rel;path=queue/'runs'/run
        if rel not in cached_queues:
            cached_queues[rel]=read(queue/'state.json');protect(queue/'state.json')
        state=cached_queues[rel];assert state['status']=='complete'
        jobs=state['jobs'];jobs=jobs.values() if isinstance(jobs,dict) else jobs
        job=next(j for j in jobs if Path(j['output'])==path)
        assert job['status']=='done' and job['returncode']==0
        c=read(path/'completion.json');m=read(path/'run_manifest.json')
        assert c['complete'] and not c['pilot'] and not c['test_loaded']
        assert c['setting']=='mimic24' and c['seed']==s and c['config_index']==cfg and c['role']==family
        assert m['seed']==s and m['family']==family and m['feature_view']=='F2'
        assert not m['test_loaded'] and m['input_shape']=={'train':[52106,12508],'dev':[7460,12508]}
        protect(path/'completion.json');protect(path/'run_manifest.json')
        command=job['command'];source=next(x for x in command if x.endswith('.py'))
        protect(source,m['source_sha256'])
        for flag,key in (('--cache','cache_sha256'),('--values','values_sha256')):
            pointer=Path(command[command.index(flag)+1])/'completion.json';protect(pointer,m[key])
        contract={k:m.get(k) for k in ('recipe','feature_view','cache_sha256','values_sha256')}
        if family in recipes:assert recipes[family]==contract,(family,'seed recipe differs')
        recipes[family]=contract
        protect(path/'predictions_dev.npz',row['predictions_sha256'])
        pr=dict(np.load(path/'predictions_dev.npz'))
        assert np.array_equal(pr['ids'],ids) and np.array_equal(pr['labels'],y)
        assert pr['probs'].shape==y.shape and np.isfinite(pr['probs']).all()
        aps=[float(average_precision_score(y[:,e],pr['probs'][:,e])) for e in (0,1)]
        assert max(abs(a-b) for a,b in zip(aps,row['endpoint_ap']))<1e-12
        assert abs(float(np.mean(aps))-c['dev_ap'])<1e-12
        artifacts={}
        for e in (0,1):
            matches=[f for f in path.glob(f'best_task{e}.*') if f.is_file()]
            assert len(matches)==1,(str(path),e,matches)
            artifacts[matches[0].name]=protect(matches[0])
        result['baselines'][f'{family}_s{s}']=dict(path=str(path),family=family,seed=s,config=cfg,
            AP=aps,AUROC=[float(roc_auc_score(y[:,e],pr['probs'][:,e])) for e in (0,1)],
            predictions_sha256=row['predictions_sha256'],source_sha256=m['source_sha256'],
            checkpoint_current_hashes=artifacts,checkpoint_hash_provenance='computed during current read-only audit',
            historical_prediction_hash_verified=True,exit_evidence='archived completed queue entry with returncode zero',
            recipe=contract,seconds=c.get('elapsed_wall_sec',c.get('seconds')))
        if family=='realmlp':
            drift=[float(np.max(np.abs(pr['probs'][:,e]-control_predictions[(s,e)]))) for e in (0,1)]
            assert max(drift)<=1e-6,(s,'native B0 archive differs',drift)
            result['native_archive_agreement'][str(s)]=dict(max_probability_drift=drift,AP=aps)
    assert len(result['baselines'])==12 and len(result['controls'])==20
    result.update(complete=True,completed_at=time.time(),protected=protected,
        protected_files_unchanged=all(sha(path)==digest for path,digest in protected.items()),
        no_checkpoint_weights_loaded=True,matching_dev_ids_labels=True,unchanged_recipe_across_seeds=True)
    assert result['protected_files_unchanged']
    atomic(out,result)
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--project',required=True);a.add_argument('--output',required=True)
    args=a.parse_args();r=audit(args.project,args.output)
    print(json.dumps({k:r[k] for k in ('complete','dev_rows','dev_patients','native_archive_agreement')}))
