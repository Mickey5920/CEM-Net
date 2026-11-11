"""Validate archived evidence and exact native commands before any new fitting."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_recovery_queue import read,sha,atomic,same_process
from scripts.tracehr_stage2_common import RUN,FAMILIES,expected,baseline_command

def audit(project,output):
    import numpy as np
    from sklearn.metrics import average_precision_score
    p=project.resolve(strict=True);expect=expected();prior=p/'runs/tracehr_not_stability_20261018/results_v1/aggregates.json'
    aggregate=read(prior)
    assert aggregate['complete'] and aggregate['protected_files_unchanged'] and not aggregate['test_loaded']
    assert aggregate['records']==expect['records'] and aggregate['baseline_records']==expect['baseline_records']
    base=p/'runs/tracehr_not_stability_20261018/mimic/stage1_v1'
    state=read(base/'screen/state.json');exit=read(base/'start_exit.json')
    assert state['status']=='complete' and state['counts']==dict(pending=0,running=0,completed=28,failed=0)
    assert exit['returncode']==0 and not same_process(exit['child_pid'],exit['child_identity'])
    protected={str(prior):sha(prior)}
    old=read(base/'baseline-audit.json')
    for f,h in old['protected'].items():assert sha(f)==h;protected[f]=h
    prep=read(p/'runs/tracehr_structure_20261014/mimic/g_v1/preparation/completion.json')
    folder=Path(prep['input_pointer']['path'])
    for n,r in prep['input_artifacts'].items():assert sha(folder/n)==r['sha256'];protected[str(folder/n)]=r['sha256']
    dev=dict(np.load(folder/'dev_targets.npz'));controls={}
    for k,r in aggregate['records'].items():
        path=Path(r['source']);c=read(path/'completion.json');m=read(path/'run_manifest.json')
        pr=dict(np.load(path/'predictions_dev.npz'));e=r['endpoint']
        assert c['complete'] and not c['test_loaded'] and c['seed']==r['seed']
        assert np.array_equal(pr['ids'],dev['ids']) and np.array_equal(pr['labels'],dev['y'][:,e:e+1])
        assert abs(average_precision_score(pr['labels'][:,0],pr['probs'][:,0])-r['AP'])<1e-12
        for f,h in ((path/c.get('checkpoint_file','best.pt'),r['checkpoint_sha256']),(path/'predictions_dev.npz',r['predictions_sha256'])):
            assert sha(f)==h;protected[str(f)]=h
        receipt=read(path.parent.parent/(path.name+'_exit.json'));assert receipt['returncode']==0
        for f in (path/'completion.json',path/'run_manifest.json',path/'history.json'):protected[str(f)]=sha(f)
        controls[k]=dict(path=str(path),variant=r['variant'],endpoint=e,seed=r['seed'],dev_ap=r['AP'],initial_weights_sha256=m['initial_weights_sha256'],checkpoint_sha256=r['checkpoint_sha256'],predictions_sha256=r['predictions_sha256'])
    contracts={}
    for family in FAMILIES:
        t=expect['templates'][family];manifest=p/t['source_manifest_rel']
        assert sha(manifest)==t['source_manifest_sha256']
        for f,h in read(manifest).items():assert sha(manifest.parent/f)==h;protected[str(manifest.parent/f)]=h
        protected[str(manifest)]=sha(manifest)
        cmd=baseline_command(family,45,p/'runs'/RUN/'not_created',sys.executable)
        entry=Path(cmd[2]);assert sha(entry)==t['entry_sha256']
        oldrow=aggregate['baseline_records'][family+'_s44'];m=read(Path(oldrow['path'])/'run_manifest.json')
        assert sha(entry)==m['source_sha256']
        for key,arg in (('cache_sha256','--cache'),('values_sha256','--values'),('compact_sha256','--compact')):
            if arg in cmd:
                f=Path(cmd[cmd.index(arg)+1])/'completion.json';assert sha(f)==m[key];protected[str(f)]=sha(f)
        contracts[family]=dict(manifest=m,entry=str(entry),entry_sha256=sha(entry),source_manifest=str(manifest),source_manifest_sha256=sha(manifest),reference=oldrow)
        for s in (42,43,44):
            r=aggregate['baseline_records'][f'{family}_s{s}'];path=Path(r['path']);pr=dict(np.load(path/'predictions_dev.npz'))
            assert np.array_equal(pr['ids'],dev['ids']) and np.array_equal(pr['labels'],dev['y'])
            assert sha(path/'predictions_dev.npz')==r['predictions_sha256']
            for e in (0,1):assert abs(average_precision_score(pr['labels'][:,e],pr['probs'][:,e])-r['AP'][e])<1e-12
            protected[str(path/'predictions_dev.npz')]=r['predictions_sha256']
            for f,h in r['checkpoint_current_hashes'].items():
                fp=Path(f) if Path(f).is_absolute() else path/f
                assert sha(fp)==h;protected[str(fp)]=h
    for f,h in protected.items():assert sha(f)==h
    result=dict(complete=True,protected_files_unchanged=True,test_loaded=False,patient_data_exported=False,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),prior_aggregate_sha256=sha(prior),controls=controls,
        baselines=aggregate['baseline_records'],contracts=contracts,protected=protected,
        baseline_artifact_bound_gib=expect['baseline_artifact_bound_gib'],noA_previous_checkpoint_bytes=expect['noA_previous_checkpoint_bytes'])
    atomic(output,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.project,a.output);print(json.dumps(dict(complete=True,controls=len(r['controls']),baselines=len(r['baselines']),protected_files=len(r['protected']))))
