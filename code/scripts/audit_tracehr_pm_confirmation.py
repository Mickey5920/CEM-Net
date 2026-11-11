"""Revalidate five-seed controls and strictly equivalent old PM results, read only."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_tracehr_noa_ablation import audit as old_audit
from scripts.tracehr_recovery_queue import read,sha,atomic,same_process
from scripts.tracehr_pm_confirmation_common import expected,MATRIX

def audit(project,output):
    import numpy as np
    from sklearn.metrics import average_precision_score
    previous=old_audit(project,output.with_name('old-controls-audit.json'))
    protected=dict(previous['protected']);refs=expected()
    prior=project/'runs/tracehr_noa_ablation_20261019/results_v2/aggregates.json'
    r=read(prior);assert r['complete'] and r['protected_files_unchanged'] and not r['test_loaded']
    protected[str(prior)]=sha(prior)
    oldbase=project/'runs/tracehr_noa_ablation_20261019/mimic/ablation_recovery_v4'
    state=read(oldbase/'screen/state.json');receipt=read(oldbase/'start_exit.json')
    assert state['status']=='complete' and state['counts']==dict(pending=0,running=0,completed=18,failed=0)
    assert receipt['returncode']==0 and not same_process(receipt['child_pid'],receipt['child_identity'])
    protected[str(oldbase/'start_exit.json')]=sha(oldbase/'start_exit.json')
    recovery=read(oldbase/'recovery-audit.json');assert recovery['complete'] and not recovery['scientific_recipe_changed']
    for f,h in recovery['protected'].items():assert sha(f)==h;protected[f]=h
    decision=read(oldbase/'reuse-decision.json')
    for s in (43,44,45):
        for e in (0,1):
            k=f'C5_noT_noA_noR_t{e}_s{s}';j=state['jobs'][k]
            assert j['status']=='completed' and r['records'][k]==refs['records'][k]
            if j['reused']:
                proof=read(Path(state['jobs'][j['equivalence_id']]['output'])/'completion.json')
                assert proof==decision['decisions'][k]==r['records'][k]['equivalence_proof']
                assert proof['equivalent'] and proof['updates']==20 and proof['continuous_exact']
                assert proof['early_snapshots_verified']==8 and proof['best_probability_drift']==0.
            else:
                ex=read(j['exit_receipt']);assert ex['returncode']==0 and not same_process(ex['child_pid'],ex['child_identity'])
                protected[j['exit_receipt']]=sha(j['exit_receipt'])
    for n in ('recovery-audit.json','reuse-decision.json','runtime_envelope.json'):protected[str(oldbase/n)]=sha(oldbase/n)
    prep=read(project/'runs/tracehr_structure_20261014/mimic/g_v1/preparation/completion.json')
    target=dict(np.load(Path(prep['input_pointer']['path'])/'dev_targets.npz'));controls={}
    for k,row in refs['records'].items():
        archived=(r['records'] if row['variant']=='C5_noT_noA_noR' else r['reference_five_seed_records'])[k]
        assert archived==row
        p=Path(row['source']);q=dict(np.load(p/'predictions_dev.npz'));e=row['endpoint']
        assert np.array_equal(q['ids'],target['ids'])
        if q['labels'].shape[1]==1:
            assert np.array_equal(q['labels'],target['y'][:,e:e+1]);scores=q['probs'][:,0]
        else:
            assert np.array_equal(q['labels'],target['y']);scores=q['probs'][:,e]
        assert abs(float(average_precision_score(target['y'][:,e],scores))-row['AP'])<1e-12
        if row.get('predictions_sha256'):assert sha(p/'predictions_dev.npz')==row['predictions_sha256']
        if row.get('checkpoint_sha256'):
            ck=p/read(p/'completion.json').get('checkpoint_file','best.pt');assert sha(ck)==row['checkpoint_sha256'];protected[str(ck)]=sha(ck)
        for n in ('completion.json','run_manifest.json','predictions_dev.npz'):protected[str(p/n)]=sha(p/n)
        controls[k]=dict(path=str(p),AP=row['AP'],endpoint=e,seed=row['seed'],variant=row['variant'])
    assert len(controls)==56 and len(MATRIX)==4
    result=dict(previous,protected=protected,controls=controls,old_PM_equivalence_verified=True,
        new_fits_authorized=4,prior_PM_aggregate_sha256=sha(prior),test_loaded=False,
        local_PM_aggregate_sha256=refs['local_reference_aggregate_sha256'],other_fits_authorized=False)
    atomic(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.project,a.output)
