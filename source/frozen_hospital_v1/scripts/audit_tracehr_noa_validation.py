"""Read-only source, input, reused-result and native-command audit."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_noa_validation_common import *
from scripts.tracehr_recovery_queue import atomic,sha,same_process
def audit(project,database,output):
    import numpy as np
    from sklearn.metrics import average_precision_score
    project=project.resolve(strict=True);exp=expected();protected={}
    def check(f,wanted=None):
        f=Path(f);h=sha(f)
        if wanted:assert h==wanted,(str(f),h,wanted)
        protected[str(f)]=h;return h
    for n,h in exp['parent_manifest'].items():check(ROOT/n,h)
    for n,h in read(ROOT/'source_manifest.json').items():check(ROOT/n,h)
    prep=project/'runs/tracehr_structure_20261014'/database/'g_v1/preparation'
    check(prep/'completion.json',exp['preparation_sha256'][database]);pr=read(prep/'completion.json')
    assert pr['complete'] and pr['preparation_complete'] and not pr['test_loaded'] and pr['fit_split']=='train'
    folder=Path(pr['input_pointer']['path'])
    for n,r in pr['input_artifacts'].items():assert n.startswith(('train_','dev_'));check(folder/n,r['sha256'])
    for n,k in [('preprocessing.npz','preprocessing_sha256'),('native_configs.json','native_configs_sha256'),('native_summary_scaling.npz','native_summary_scaling_sha256')]:check(prep/n,pr[k])
    dev=dict(np.load(folder/'dev_targets.npz'));controls={};contracts={}
    if database=='mimic':
        previous=project/'runs/tracehr_noa_main_confirmation_20261019/mimic/confirm_v2'
        assert read(previous/'screen/state.json')['status']=='complete'
        receipt=read(previous/'start_exit.json');assert receipt['returncode']==0 and not same_process(receipt['child_pid'],receipt['child_identity'])
        check(previous/'start_exit.json');check(previous/'screen/state.json')
        for key,r in exp['mimic_records'].items():
            path=Path(r['source']);c=read(path/'completion.json');m=read(path/'run_manifest.json');e=r['endpoint']
            assert c['complete'] and not c['test_loaded'] and c['seed']==r['seed']
            prediction=dict(np.load(path/'predictions_dev.npz'));col=0 if prediction['labels'].shape[1]==1 else e
            assert np.array_equal(prediction['ids'],dev['ids']) and np.array_equal(prediction['labels'][:,col],dev['y'][:,e])
            assert abs(average_precision_score(dev['y'][:,e],prediction['probs'][:,col])-r['AP'])<1e-12
            check(path/'predictions_dev.npz',r.get('predictions_sha256'))
            if r['variant'].startswith('C5') or r['variant']=='B0':
                check(path/c.get('checkpoint_file','best.pt'),r['checkpoint_sha256'])
                assert m['initial_weights_sha256']==c['initial_weights_sha256']
                if 'initial_weights_sha256' in r:assert m['initial_weights_sha256']==r['initial_weights_sha256']
                exitpath=path.parent.parent/(path.name+'_exit.json');assert read(exitpath)['returncode']==0;check(exitpath)
            else:
                for f in path.glob('best_task*'):
                    if f.is_file():check(f)
            for n in ('completion.json','run_manifest.json','history.json'):check(path/n)
            controls[key]=dict(r,initial_weights_sha256=m.get('initial_weights_sha256'),run_manifest_sha256=sha(path/'run_manifest.json'))
    else:
        assert pr['input_dim']==18250 and pr['train_rows']==92565 and pr['dev_rows']==13140
        for family,t in exp['eicu_templates'].items():
            path=Path(t['reference']['path']);m=read(path/'run_manifest.json');c=read(path/'completion.json')
            assert m==t['reference']['manifest'] and c==t['reference']['completion']
            assert c['complete'] and not c['pilot'] and not c['test_loaded'] and m.get('view',m.get('feature_view'))=='F2'
            cmd=baseline_command(family,48,project/'runs'/RUN/'uncreated',sys.executable);entry=Path(cmd[2]);check(entry,m['source_sha256'])
            # Lock original entry's colocated dependencies as well.
            for f in entry.parent.glob('*.py'):check(f)
            for field,arg in [('cache_sha256','--cache'),('values_sha256','--values'),('compact_sha256','--compact')]:
                if arg in cmd:check(Path(cmd[cmd.index(arg)+1])/'completion.json',m[field])
            pred=dict(np.load(path/'predictions_dev.npz'))
            assert np.array_equal(pred['ids'],dev['ids']) and np.array_equal(pred['labels'],dev['y'])
            aps=[float(average_precision_score(dev['y'][:,e],pred['probs'][:,e])) for e in (0,1)]
            assert abs(sum(aps)/2-c['dev_ap'])<1e-12
            for n in ('completion.json','run_manifest.json','history.json','predictions_dev.npz'):check(path/n)
            for f in path.glob('best_task*'):
                if f.is_file():check(f)
            contracts[family]=dict(manifest=m,entry=str(entry),entry_sha256=sha(entry),reference=dict(path=str(path),AP=aps))
        # Scope inventory: previously fitted seeds were 42-44; require no same-name target writers.
        existing=[p for p in (project/'runs'/RUN/database).glob('*/screen/state.json') if p.parent.parent!=output.parent]
        assert not existing,('Inspect existing campaign before creating another queue',list(map(str,existing)))
    result=dict(complete=True,protected_files_unchanged=True,source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        database=database,controls=controls,contracts=contracts,protected=protected,test_loaded=False,patient_data_exported=False,
        input_manifest_sha256=sha(prep/'completion.json'),train_rows=pr['train_rows'],dev_rows=pr['dev_rows'],input_dim=pr['input_dim'])
    atomic(output,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--database',choices=['mimic','eicu'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.project,a.database,a.output);print(json.dumps(dict(complete=True,controls=len(r['controls']),contracts=len(r['contracts']),protected_files=len(r['protected']))))
