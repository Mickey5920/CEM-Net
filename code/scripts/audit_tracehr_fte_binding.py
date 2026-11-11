"""Reuse verified archives, checking identity and unchanged files for this batch."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_fte_binding_common import *
from scripts.tracehr_recovery_queue import atomic, sha, same_process

def audit(project,database,output):
    project=project.resolve(strict=True);exp=expected();db=exp['databases'][database];protected={}
    def check(path,wanted=None):
        path=Path(path);digest=sha(path)
        assert path.resolve().is_relative_to(project)
        if wanted is not None:assert digest==wanted,(str(path),digest,wanted)
        protected[str(path)]=digest
        return digest
    for name,digest in exp['parent_manifest'].items():check(ROOT/name,digest)
    for name,digest in read(ROOT/'source_manifest.json').items():check(ROOT/name,digest)
    archive_path=project/db['archive']
    archived=read(archive_path)
    assert archived['complete'] and not archived['test_loaded'] and archived['protected_files_unchanged']
    for key,record in db['controls'].items():assert archived['records'][key]==record
    check(archive_path)
    previous=project/db['previous_queue']
    state=read(previous/'screen/state.json');receipt=read(previous/'start_exit.json')
    assert state['status']=='complete' and state['counts']['running']==state['counts']['pending']==state['counts']['failed']==0
    assert receipt['returncode']==0 and not same_process(receipt['child_pid'],receipt['child_identity'])
    check(previous/'screen/state.json');check(previous/'start_exit.json')
    prep=project/'runs/tracehr_structure_20261014'/database/'g_v1/preparation'
    check(prep/'completion.json',db['preparation_sha256'])
    pr=read(prep/'completion.json')
    assert pr['complete'] and pr['preparation_complete'] and not pr['test_loaded'] and pr['fit_split']=='train'
    folder=Path(pr['input_pointer']['path']);assert folder.resolve().is_relative_to(project)
    for name,record in pr['input_artifacts'].items():
        assert name.startswith(('train_','dev_'));check(folder/name,record['sha256'])
    for name,field in [('preprocessing.npz','preprocessing_sha256'),('native_configs.json','native_configs_sha256'),('native_summary_scaling.npz','native_summary_scaling_sha256')]:check(prep/name,pr[field])
    controls={}
    for key,record in db['controls'].items():
        path=Path(record['source']);c=read(path/'completion.json');manifest=read(path/'run_manifest.json')
        assert c['complete'] and not c['test_loaded']
        check(path/'predictions_dev.npz',record['predictions_sha256'])
        if record['variant'] in (PM,'C5_noT_noA','B0'):
            check(path/c.get('checkpoint_file','best.pt'),record['checkpoint_sha256'])
            assert manifest['initial_weights_sha256']==record['initial_weights_sha256']
        else:
            for name,digest in record['checkpoint_hashes'].items():check(path/name,digest)
        for name in ('run_manifest.json','completion.json','history.json'):check(path/name)
        if record['variant']==PM:
            assert manifest['members']==8 and not c['auxiliary_enabled'] and not c['main_residual_enabled']
            j=state['jobs'][path.name];er=read(j['exit_receipt'])
            assert j['status']=='completed' and er['returncode']==0 and not same_process(er['child_pid'],er['child_identity'])
            check(j['exit_receipt'])
        controls[key]=dict(record,run_manifest_sha256=sha(path/'run_manifest.json'))
    assert len(controls)==30
    result=dict(complete=True,database=database,source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        controls=controls,contracts={},protected=protected,test_loaded=False,patient_data_exported=False,
        archived_AP_recalculation_reused=True,archive_records_verified=len(controls),
        input_manifest_sha256=sha(prep/'completion.json'),input_dim=pr['input_dim'],train_rows=pr['train_rows'],dev_rows=pr['dev_rows'])
    atomic(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--database',choices=['mimic','eicu'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.project,a.database,a.output)
    print(json.dumps(dict(complete=True,controls=len(r['controls']),protected_files=len(r['protected']))))
