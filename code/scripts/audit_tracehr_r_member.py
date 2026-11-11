"""The readonly diagnosis gate and unchanged archived controls precede any fitting."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_tracehr_noa_ablation import audit as audit_previous
from scripts.tracehr_recovery_queue import read,sha,atomic,same_process

def audit(project,output):
    previous=audit_previous(project,output.with_name('old-controls-audit.json'))
    assert previous['complete']
    gate=read(ROOT/'work/tracehr_r_member_20261019/review_gate.json')
    folder=project/'runs/tracehr_r_member_20261019/mimic/review_v1'
    review=read(folder/'completion.json');ex=read(folder/'exit.json')
    assert ex['returncode']==0 and not same_process(ex['child_pid'],ex['child_identity'])
    assert gate['supports_six_fit_contrast'] and gate['review_completion_sha256']==sha(folder/'completion.json')
    assert review['complete'] and review['optimizer_updates']==review['new_fits']==0 and review['snapshots_examined']==108
    assert review['source_and_checkpoint_files_unchanged'] and not review['test_loaded']
    for f,h in review['protected'].items():assert sha(f)==h
    death=[r for r in review['comparisons'] if r['endpoint']==0]
    assert sum(r['risk_vs_noR']['0.05']['difference']<0 for r in death)>=2
    assert review['shared_highrisk_errors']['0_0.05']['introduced_FP_2plus_seeds']>0
    assert all(r['member_gradient_noncommon_energy_fraction']>0 for r in death)
    result=dict(previous,readonly_review_passed=True,review_completion_sha256=sha(folder/'completion.json'),
        gate_sha256=sha(ROOT/'work/tracehr_r_member_20261019/review_gate.json'),review_protected=review['protected'],
        hypothesis_not_causal_proof=True,new_fits_authorized=6)
    atomic(output,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.project,a.output)
