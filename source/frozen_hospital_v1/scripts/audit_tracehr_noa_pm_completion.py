"""Read-only verification of 72 reusable endpoint records before eight new fits."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_tracehr_pm_confirmation import audit as parent_audit
from scripts.tracehr_recovery_queue import read,sha,atomic,same_process
from scripts.tracehr_noa_pm_completion_common import expected,MATRIX

def audit(project,output):
    import numpy as np
    from sklearn.metrics import average_precision_score
    from scripts.tracehr_pm_confirmation_queue import Queue as PMQueue
    from scripts.tracehr_noa_ablation_queue import Queue as OldQueue
    previous=parent_audit(project,output.with_name('parent-PM-controls-audit.json'))
    protected=dict(previous['protected']);refs=expected()
    pm_path=project/'runs/tracehr_pm_confirmation_20261019/results_v1/aggregates.json'
    ab_path=project/'runs/tracehr_noa_ablation_20261019/results_v2/aggregates.json'
    pm,ab=read(pm_path),read(ab_path)
    for f,r in ((pm_path,pm),(ab_path,ab)):
        assert r['complete'] and r['protected_files_unchanged'] and not r['test_loaded']
        protected[str(f)]=sha(f)
    pm_base=project/'runs/tracehr_pm_confirmation_20261019/mimic/pm_v1'
    pm_state=read(pm_base/'screen/state.json');ex=read(pm_base/'start_exit.json')
    assert pm_state['status']=='complete' and pm_state['counts']==dict(pending=0,running=0,completed=4,failed=0)
    assert ex['returncode']==0 and not same_process(ex['child_pid'],ex['child_identity'])
    validator=PMQueue.__new__(PMQueue);validator.state=pm_state
    for j in pm_state['jobs'].values():
        assert j['status']=='completed'
        receipt=read(j['exit_receipt']);assert receipt['returncode']==0 and not same_process(receipt['child_pid'],receipt['child_identity'])
        assert validator.validate(j)==j['result']
        protected[j['exit_receipt']]=sha(j['exit_receipt'])
    for n in ('start_exit.json','runtime_envelope.json','controls-audit.json','deployment.json'):
        protected[str(pm_base/n)]=sha(pm_base/n)
    source=project/'work/tracehr_pm_confirmation_20261019/pm_v1'
    assert sha(source/'source_manifest.json')==refs['parent_source_sha256']
    for n,h in read(source/'source_manifest.json').items():assert sha(source/n)==h
    protected[str(source/'source_manifest.json')]=sha(source/'source_manifest.json')
    old_state=read(project/'runs/tracehr_noa_ablation_20261019/mimic/ablation_recovery_v4/screen/state.json')
    old_validator=OldQueue.__new__(OldQueue);old_validator.state=old_state
    for s in (43,44,45):
        for v in ('C5_noT_noA_noP','C5_noT_noA_noM'):
            for e in (0,1):
                j=old_state['jobs'][f'{v}_t{e}_s{s}'];assert j['status']=='completed' and not j['reused']
                receipt=read(j['exit_receipt']);assert receipt['returncode']==0 and not same_process(receipt['child_pid'],receipt['child_identity'])
                assert old_validator.validate(j)==j['result']
                protected[j['exit_receipt']]=sha(j['exit_receipt'])
    prep=read(project/'runs/tracehr_structure_20261014/mimic/g_v1/preparation/completion.json')
    target=dict(np.load(Path(prep['input_pointer']['path'])/'dev_targets.npz'));controls={}
    for k,row in refs['records'].items():
        archived=(ab['records'] if row['variant'] in ('C5_noT_noA_noP','C5_noT_noA_noM') else pm['records'])[k]
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
    assert len(controls)==72 and len(MATRIX)==8
    result=dict(previous,protected=protected,controls=controls,new_fits_authorized=8,retained_R=True,
        previous_PM_four_complete=True,prior_PM_aggregate_sha256=sha(pm_path),prior_PM_deletion_aggregate_sha256=sha(ab_path),
        local_reference_aggregate_sha256=refs['local_reference_aggregate_sha256'],other_fits_authorized=False,test_loaded=False)
    atomic(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.project,a.output)
