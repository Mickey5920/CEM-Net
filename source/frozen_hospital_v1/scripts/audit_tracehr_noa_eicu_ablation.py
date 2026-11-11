"""Verify archived controls and inputs without fitting or exporting patient records."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tracehr_noa_eicu_ablation_common import *
from scripts.tracehr_recovery_queue import atomic, sha, same_process

def audit(project, database, output):
    import numpy as np
    from sklearn.metrics import average_precision_score
    assert database == 'eicu'
    project = project.resolve(strict=True)
    exp, protected = expected(), {}
    def check(path, wanted=None):
        path = Path(path)
        digest = sha(path)
        if wanted is not None: assert digest == wanted, (str(path), digest, wanted)
        protected[str(path)] = digest
        return digest
    for name, digest in exp['parent_manifest'].items(): check(ROOT/name, digest)
    for name, digest in read(ROOT/'source_manifest.json').items(): check(ROOT/name, digest)
    old_source = project/'work/tracehr_noa_validation_20261020/validation_v2'
    check(old_source/'source_manifest.json', exp['parent_sha256'])
    previous = project/'runs/tracehr_noa_validation_20261020/eicu/validation_recovery_v3'
    previous_state = read(previous/'screen/state.json')
    previous_exit = read(previous/'start_exit.json')
    assert previous_state['status'] == 'complete' and previous_state['counts']['completed'] == 21
    assert previous_exit['returncode'] == 0
    assert not same_process(previous_exit['child_pid'], previous_exit['child_identity'])
    check(previous/'screen/state.json'); check(previous/'start_exit.json')
    archived = read(project/'runs/tracehr_noa_validation_20261020/eicu_results_v2/aggregates.json')
    assert archived['complete'] and archived['new_endpoint_fits'] == 30
    # Local and remote aggregate serialization can differ; verify records exactly.
    assert archived['records'] == exp['archive_records']
    check(project/'runs/tracehr_noa_validation_20261020/eicu_results_v2/aggregates.json')
    prep = project/'runs/tracehr_structure_20261014/eicu/g_v1/preparation'
    check(prep/'completion.json', exp['preparation_sha256'])
    pr = read(prep/'completion.json')
    assert pr['complete'] and pr['preparation_complete'] and not pr['test_loaded'] and pr['fit_split'] == 'train'
    assert (pr['input_dim'],pr['train_rows'],pr['dev_rows']) == (18250,92565,13140)
    folder = Path(pr['input_pointer']['path'])
    assert folder.resolve().is_relative_to(project)
    for name, record in pr['input_artifacts'].items():
        assert name.startswith(('train_','dev_')); check(folder/name, record['sha256'])
    for name, field in [('preprocessing.npz','preprocessing_sha256'),('native_configs.json','native_configs_sha256'),('native_summary_scaling.npz','native_summary_scaling_sha256')]:
        check(prep/name, pr[field])
    dev = dict(np.load(folder/'dev_targets.npz'))
    controls = {}
    # Preserve and verify the entire archived campaign, while the current comparison
    # scope is the main method plus B0/TabM/XGBoost as specified by the user.
    for key, record in exp['archive_records'].items():
        path = Path(record['source']); e = record['endpoint']
        c, manifest = read(path/'completion.json'), read(path/'run_manifest.json')
        assert c['complete'] and not c['test_loaded']
        prediction = dict(np.load(path/'predictions_dev.npz'))
        col = 0 if prediction['labels'].shape[1] == 1 else e
        assert np.array_equal(prediction['ids'],dev['ids'])
        assert np.array_equal(prediction['labels'][:,col],dev['y'][:,e])
        assert abs(average_precision_score(dev['y'][:,e],prediction['probs'][:,col])-record['AP']) < 1e-12
        check(path/'predictions_dev.npz',record['predictions_sha256'])
        if record['variant'] in ('C5_noT_noA','B0'):
            check(path/c.get('checkpoint_file','best.pt'),record['checkpoint_sha256'])
            assert manifest['initial_weights_sha256'] == record['initial_weights_sha256']
        else:
            for name,digest in record['checkpoint_hashes'].items(): check(path/name,digest)
        for name in ('run_manifest.json','completion.json','history.json'): check(path/name)
        job_id = path.name
        job = previous_state['jobs'][job_id]
        exit_record = read(job['exit_receipt'])
        assert job['status'] == 'completed' and exit_record['returncode'] == 0
        assert not same_process(exit_record['child_pid'],exit_record['child_identity'])
        check(job['exit_receipt'])
        if record['variant'] in ('C5_noT_noA',*BASELINES):
            controls[key] = dict(record,run_manifest_sha256=sha(path/'run_manifest.json'))
    assert len(controls) == 24
    result = dict(complete=True, database=database, source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        controls=controls,contracts={},protected=protected,test_loaded=False,patient_data_exported=False,
        archive_records_verified=30,comparison_records=24,baseline_scope=list(BASELINES),
        input_manifest_sha256=sha(prep/'completion.json'),train_rows=92565,dev_rows=13140,input_dim=18250)
    atomic(output,result)
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--database',choices=['eicu'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.project,a.database,a.output)
    print(json.dumps(dict(complete=True,controls=len(r['controls']),protected_files=len(r['protected']))))
