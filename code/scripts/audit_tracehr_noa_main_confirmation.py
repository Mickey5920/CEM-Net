"""Read-only original-entry, source, archived result and patient-identity audit."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_recovery_queue import read,sha,atomic,same_process
from scripts.audit_tracehr_stage2 import audit as predecessor_audit
from scripts.tracehr_noa_main_confirmation_common import expected
from scripts.tracehr_stage2_queue import Queue as PriorQueue
def audit(project,output):
    import numpy as np
    from sklearn.metrics import average_precision_score
    p=project.resolve(strict=True);expect=expected()
    result=predecessor_audit(p,output.with_name('predecessor-baseline-audit.json'))
    protected=dict(result['protected'])
    prior=p/'runs/tracehr_noa_pm_completion_20261019/results_v1/aggregates.json'
    old=read(prior);assert old['complete'] and old['protected_files_unchanged'] and not old['test_loaded']
    records={k:r for k,r in old['records'].items() if r['variant'] in ('C5_noT_noA','B0','tabm','lightgbm','xgboost') and r['seed'] in (43,44,45,46,47)}
    assert records==expect['records'] and len(records)==50
    protected[str(prior)]=sha(prior)
    previous=p/'runs/tracehr_noa_pm_completion_20261019/mimic/completion_v1'
    assert read(previous/'screen/state.json')['status']=='complete'
    exit=read(previous/'start_exit.json');assert exit['returncode']==0 and not same_process(exit['child_pid'],exit['child_identity'])
    for f in (previous/'screen/state.json',previous/'start_exit.json'):protected[str(f)]=sha(f)
    stage2=p/'runs/tracehr_not_stability_stage2_20261018/mimic/stage2_v1'
    state=read(stage2/'screen/state.json');assert state['status']=='complete'
    exit=read(stage2/'start_exit.json');assert exit['returncode']==0 and not same_process(exit['child_pid'],exit['child_identity'])
    validator=PriorQueue.__new__(PriorQueue);validator.state=state
    for j in state['jobs'].values():
        if j['kind']!='baseline':continue
        assert j['status']=='completed';r=validator.validate(j);assert r==j['result']
        receipt=read(j['exit_receipt']);assert receipt['returncode']==0 and not same_process(receipt['child_pid'],receipt['child_identity'])
        path=Path(j['output'])
        for n,h in r['checkpoint_hashes'].items():assert sha(path/n)==h;protected[str(path/n)]=h
        protected[j['exit_receipt']]=sha(j['exit_receipt'])
    prep=read(p/'runs/tracehr_structure_20261014/mimic/g_v1/preparation/completion.json')
    dev=dict(np.load(Path(prep['input_pointer']['path'])/'dev_targets.npz'));controls={}
    for k,r in records.items():
        path=Path(r['source']);c=read(path/'completion.json');m=read(path/'run_manifest.json');e=r['endpoint']
        assert c['complete'] and not c['test_loaded'] and c['seed']==r['seed']
        prediction_hash=r.get('predictions_sha256',result['baselines'].get(f"{r['variant']}_s{r['seed']}",{}).get('predictions_sha256'))
        assert prediction_hash and sha(path/'predictions_dev.npz')==prediction_hash
        pr=dict(np.load(path/'predictions_dev.npz'));assert np.array_equal(pr['ids'],dev['ids'])
        col=0 if pr['labels'].shape[1]==1 else e
        assert np.array_equal(pr['labels'][:,col],dev['y'][:,e])
        assert abs(average_precision_score(pr['labels'][:,col],pr['probs'][:,col])-r['AP'])<1e-12
        if r['variant'] in ('C5_noT_noA','B0'):
            checkpoint=path/c.get('checkpoint_file','best.pt');assert sha(checkpoint)==r['checkpoint_sha256']
            protected[str(checkpoint)]=r['checkpoint_sha256']
            receipt=read(path.parent.parent/(path.name+'_exit.json'));assert receipt['returncode']==0
            assert m['initial_weights_sha256']==r['initial_weights_sha256']
        for n in ('predictions_dev.npz','completion.json','run_manifest.json','history.json'):protected[str(path/n)]=sha(path/n)
        controls[k]=dict(r,path=str(path),dev_ap=r['AP'])
    for f,h in protected.items():assert sha(f)==h
    result.update(controls=controls,protected=protected,source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        prior_aggregate_sha256=sha(prior),reference_records=50,formal_endpoint_fits=30,formal_workers=21,
        new_seeds=[48,49,50],scientific_recipes_unchanged=True,fixed_main_method='C5_noT_noA')
    atomic(output,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.project,a.output);print(json.dumps(dict(complete=True,controls=len(r['controls']),protected_files=len(r['protected']))))
