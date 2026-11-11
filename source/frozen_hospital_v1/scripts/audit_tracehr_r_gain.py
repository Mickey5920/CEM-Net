"""Bind the completed RM results and two readonly reviews before new RG fitting."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_tracehr_r_member import audit as audit_previous
from scripts.tracehr_recovery_queue import read,sha,atomic,same_process
from scripts.tracehr_r_gain_common import expected,MATRIX


def audit(project,output):
    import numpy as np
    from sklearn.metrics import average_precision_score
    previous=audit_previous(project,output.with_name('old-controls-audit.json'))
    assert previous['complete']
    gate=read(ROOT/'work/tracehr_r_gain_20261019/review_gate.json')
    assert gate['supports_six_fit_contrast']
    assert {(r['variant'],r['endpoint'],r['seed']) for r in gate['authorized_matrix']}==set(MATRIX)
    protected=dict(previous['protected']);review_protected=dict(previous['review_protected'])
    for name,digest in gate['reviews'].items():
        folder=project/'runs/tracehr_rm_mechanism_20261019/mimic'/name
        assert sha(folder/'completion.json')==digest
        r=read(folder/'completion.json');ex=read(folder/'exit.json')
        assert ex['returncode']==0 and not same_process(ex['child_pid'],ex['child_identity'])
        assert r['complete'] and r['read_only'] and r['optimizer_updates']==r['new_fits']==0
        assert r['source_and_checkpoint_files_unchanged'] and not r['test_loaded'] and not r['patient_records_exported']
        assert r['models_examined']==12
        for f,h in r['protected'].items():assert sha(f)==h;review_protected[f]=h
        for n in ('completion.json','exit.json','source_binding.json'):review_protected[str(folder/n)]=sha(folder/n)
    refs=expected();old=project/'runs/tracehr_r_member_20261019/mimic/rm_v1'
    state=read(old/'screen/state.json');ex=read(old/'start_exit.json')
    assert state['status']=='complete' and state['counts']==dict(pending=0,running=0,completed=6,failed=0)
    assert ex['returncode']==0 and not same_process(ex['child_pid'],ex['child_identity'])
    for j in state['jobs'].values():
        exj=read(j['exit_receipt']);assert exj['returncode']==0 and not same_process(exj['child_pid'],exj['child_identity'])
        protected[j['exit_receipt']]=sha(j['exit_receipt'])
    prior=project/'runs/tracehr_r_member_20261019/results_v1/aggregates.json'
    aggregate=read(prior)
    assert aggregate['complete'] and aggregate['records']==refs['records']
    assert aggregate['source_manifest_sha256']==refs['parent_source_sha256']
    protected[str(prior)]=sha(prior);protected[str(old/'start_exit.json')]=sha(old/'start_exit.json')
    prep=read(project/'runs/tracehr_structure_20261014/mimic/g_v1/preparation/completion.json')
    target=dict(np.load(Path(prep['input_pointer']['path'])/'dev_targets.npz'));controls={}
    for k,r in refs['records'].items():
        folder=Path(r['source']);q=dict(np.load(folder/'predictions_dev.npz'));e=r['endpoint']
        assert np.array_equal(q['ids'],target['ids'])
        if q['labels'].shape[1]==1:
            assert np.array_equal(q['labels'],target['y'][:,e:e+1]);scores=q['probs'][:,0]
        else:
            assert np.array_equal(q['labels'],target['y']);scores=q['probs'][:,e]
        assert abs(float(average_precision_score(target['y'][:,e],scores))-r['AP'])<1e-12
        if r.get('predictions_sha256'):assert sha(folder/'predictions_dev.npz')==r['predictions_sha256']
        if r.get('checkpoint_sha256'):
            checkpoint=folder/read(folder/'completion.json').get('checkpoint_file','best.pt')
            assert sha(checkpoint)==r['checkpoint_sha256'];protected[str(checkpoint)]=sha(checkpoint)
        for n in ('completion.json','run_manifest.json','predictions_dev.npz'):protected[str(folder/n)]=sha(folder/n)
        controls[k]=dict(path=str(folder),AP=r['AP'],endpoint=e,seed=r['seed'],variant=r['variant'])
    assert len(controls)==42
    result=dict(previous,protected=protected,controls=controls,review_protected=review_protected,
        readonly_review_passed=True,review_completion_sha256=gate['reviews']['review_v1'],
        coherence_completion_sha256=gate['reviews']['coherence_v2'],
        gate_sha256=sha(ROOT/'work/tracehr_r_gain_20261019/review_gate.json'),
        hypothesis_not_causal_proof=True,new_fits_authorized=6)
    atomic(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.project,a.output)
