"""Bind the new screen to unchanged development controls and original inputs."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.cem_structure_common import *
from scripts.tracehr_recovery_queue import same_process
from sklearn.metrics import average_precision_score


def audit(project,preparation,output):
    project=project.resolve(strict=True);exp=expected();protected={}
    def check(path,wanted=None):
        path=Path(path);assert path.resolve().is_relative_to(project)
        if str(path) in protected:digest=protected[str(path)]
        else:digest=sha(path);protected[str(path)]=digest
        if wanted is not None:assert digest==wanted,(str(path),digest,wanted)
        return digest
    for n,h in exp['parent_manifest'].items():check(ROOT/n,h)
    for n,h in read(ROOT/'source_manifest.json').items():check(ROOT/n,h)
    pr=read(preparation/'completion.json')
    assert pr['complete'] and pr['preparation_complete'] and not pr['test_loaded'] and pr['fit_split']=='train'
    assert pr['categorical_count_reconstruction_rows']==pr['train_rows']+pr['dev_rows']
    check(pr['original_preparation']+'/completion.json',pr['original_preparation_sha256'])
    for name,field in [('preprocessing.npz','preprocessing_sha256'),('native_configs.json','native_configs_sha256'),('native_summary_scaling.npz','native_summary_scaling_sha256')]:check(preparation/name,pr[field])
    for name,desc in pr['input_artifacts'].items():
        assert name.startswith(('train_','dev_'));check(Path(pr['input_pointer']['path'])/name,desc['sha256'])
    for name,desc in pr['structure_artifacts'].items():check(preparation/name,desc['sha256'])
    data=Data(project,'mimic',preparation)
    controls={}
    for key,r in exp['controls'].items():
        folder=Path(r['source']);c=read(folder/'completion.json');manifest=read(folder/'run_manifest.json')
        assert c['complete'] and not c['test_loaded'] and r['seed']==48
        check(folder/'predictions_dev.npz',r['predictions_sha256'])
        if r['variant']=='tabm':
            for n,h in r['checkpoint_hashes'].items():check(folder/n,h)
        else:
            check(folder/c.get('checkpoint_file','best.pt'),r['checkpoint_sha256'])
            assert manifest['initial_weights_sha256']==r['initial_weights_sha256']
        check(folder/'run_manifest.json',r.get('run_manifest_sha256'));check(folder/'completion.json')
        values=dict(np.load(folder/'predictions_dev.npz'))
        assert np.array_equal(values['ids'],data.targets['dev']['ids'])
        e=r['endpoint'];col=e if values['probs'].shape[1]==2 else 0
        assert np.array_equal(values['labels'][:,col],data.targets['dev']['y'][:,e])
        ap=average_precision_score(values['labels'][:,col],values['probs'][:,col]);assert abs(ap-r['AP'])<1e-12
        if r['variant']==CEM:
            assert c['main_view']=='z0' and not c['auxiliary_enabled'] and not c['main_residual_enabled']
            queue=folder.parent.parent;state=read(queue/'state.json');job=state['jobs'][folder.name];receipt=read(job['exit_receipt'])
            assert job['status']=='completed' and receipt['returncode']==0 and not same_process(receipt['child_pid'],receipt['child_identity'])
            check(job['exit_receipt'])
        controls[key]=dict(r,run_manifest_sha256=sha(folder/'run_manifest.json'))
    assert len(controls)==6
    result=dict(complete=True,database='mimic',source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        input_manifest_sha256=check(preparation/'completion.json'),controls=controls,contracts={},protected=protected,
        test_loaded=False,patient_data_exported=False,dev_AP_recalculated=True,archive_records_verified=6)
    atomic(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('project','preparation','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=audit(a.project,a.preparation,a.output)
    print(json.dumps(dict(complete=True,controls=len(r['controls']),protected_files=len(r['protected']))))
