"""Recheck archived identities, exit evidence and developer-only predictions."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_recovery_queue import read,sha,atomic,same_process

def audit(project,output):
    import numpy as np
    from sklearn.metrics import average_precision_score
    p=project.resolve(strict=True)
    expect=read(ROOT/'work/tracehr_noa_ablation_20261019/expected.json')
    prior=p/'runs/tracehr_not_stability_stage2_20261018/results_v1/aggregates.json'
    aggregate=read(prior);assert aggregate['complete'] and aggregate['protected_files_unchanged'] and not aggregate['test_loaded']
    assert aggregate['records']==expect['records'] and aggregate['baseline_records']==expect['baseline_records']
    base=p/'runs/tracehr_not_stability_stage2_20261018/mimic/stage2_v1'
    state=read(base/'screen/state.json');receipt=read(base/'start_exit.json')
    assert state['status']=='complete' and state['endpoint_counts']==dict(pending=0,running=0,completed=24,failed=0)
    assert receipt['returncode']==0 and not same_process(receipt['child_pid'],receipt['child_identity'])
    protected={str(prior):sha(prior),str(base/'start_exit.json'):sha(base/'start_exit.json')}
    previous=read(base/'baseline-audit.json')
    for f,h in previous['protected'].items():assert sha(f)==h;protected[f]=h
    for j in state['jobs'].values():
        ex=read(j['exit_receipt']);assert ex['returncode']==0 and not same_process(ex['child_pid'],ex['child_identity'])
        protected[j['exit_receipt']]=sha(j['exit_receipt'])
        folder=Path(j['output'])
        for name in ('completion.json','run_manifest.json'):
            protected[str(folder/name)]=sha(folder/name)
    prep=p/'runs/tracehr_structure_20261014/mimic/g_v1/preparation/completion.json'
    pr=read(prep);folder=Path(pr['input_pointer']['path'])
    target=dict(np.load(folder/'dev_targets.npz'));controls={}
    for k,r in aggregate['records'].items():
        path=Path(r['source']);pred=path/'predictions_dev.npz';q=dict(np.load(pred));e=r['endpoint']
        assert np.array_equal(q['ids'],target['ids'])
        if q['labels'].shape[1]==1:
            assert np.array_equal(q['labels'],target['y'][:,e:e+1]);scores=q['probs'][:,0]
        else:
            assert np.array_equal(q['labels'],target['y']);scores=q['probs'][:,e]
        assert abs(average_precision_score(target['y'][:,e],scores)-r['AP'])<1e-12
        if r.get('predictions_sha256'):assert sha(pred)==r['predictions_sha256']
        protected[str(pred)]=sha(pred)
        if r.get('checkpoint_sha256'):
            c=read(path/'completion.json');f=path/c.get('checkpoint_file','best.pt')
            assert sha(f)==r['checkpoint_sha256'];protected[str(f)]=r['checkpoint_sha256']
        for name in ('completion.json','run_manifest.json'):
            protected[str(path/name)]=sha(path/name)
        controls[k]=dict(path=str(path),AP=r['AP'],endpoint=e,seed=r['seed'],variant=r['variant'])
    for row in aggregate['baseline_records'].values():
        for f,h in row['checkpoint_current_hashes'].items():
            path=Path(f) if Path(f).is_absolute() else Path(row['path'])/f
            assert sha(path)==h;protected[str(path)]=h
    for n,h in read(ROOT/'source_manifest.json').items():assert sha(ROOT/n)==h
    result=dict(complete=True,source_manifest_sha256=sha(ROOT/'source_manifest.json'),prior_aggregate_sha256=sha(prior),
        local_aggregate_sha256=expect['prior_aggregate_sha256'],aggregate_records_semantically_equal=True,
        protected_files_unchanged=True,protected=protected,controls=controls,test_loaded=False,patient_data_exported=False,
        rows=len(target['ids']),patients=len(np.unique(target['ids'][:,0])))
    atomic(output,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.project,a.output);print(json.dumps(dict(complete=True,controls=len(r['controls']),protected_files=len(r['protected']))))
