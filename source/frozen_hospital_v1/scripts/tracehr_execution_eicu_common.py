"""eICU counterpart of the MIMIC F recipes, including its missing E2 control."""
import copy
from pathlib import Path
import shutil
import numpy as np

from scripts.tracehr_execution_refine_common import protocol as f_protocol, configure_e2
from scripts.tracehr_centered_native_common import native_summary_statistics, model_for as e_model
from scripts.tracehr_unified_common import Data, original_pointer
from scripts.tracehr_multiview_common import atomic, read, sha

RUN='tracehr_execution_refine_20261014'
VARIANTS=STAGE1=('F0','E2','F1','F2','F3','F4','F5')
STAGE2=()


def protocol(variant, selected=None):
    assert variant in VARIANTS
    spec=f_protocol('F4' if variant=='E2' else variant)
    if variant=='E2':
        spec.update(variant='E2',structure='E2',adapter_lr=.0003,
                    description='Unmodified E2 reference: native summary scaling and joint K8 shared trunk')
    spec.update(min_epochs=256,max_epochs=256,extension_epochs=256,
                early_stop_boundaries=[],early_stop_rule='full 256 epochs; no early stopping on eICU',
                significant_gain_patience=None,schedule_reference_epochs=256,
                duration_reason='Previous eICU D3 best epochs 90/103; preserve the complete native schedule')
    return spec


def stopping(history, spec):
    assert all(r['epoch']==i+1 for i,r in enumerate(history))
    return dict(stop=len(history)>=256,boundary=len(history)==256,
                next_boundary=256 if len(history)<256 else None,
                schedule_reference_epochs=256,early_stopping_enabled=False)


def prepare(project,database,out):
    assert database=='eicu'
    previous=project/'runs/tracehr_backbone_refine_20261014/eicu/v1/preparation'
    record=read(previous/'completion.json')
    assert record['complete'] and record['preparation_complete'] and not record['test_loaded']
    assert record['fit_split']=='train'
    pointer,folder,original=original_pointer(project,database)
    assert pointer==record['input_pointer']
    for name,artifact in record['input_artifacts'].items():
        assert name.startswith(('train_','dev_'))
        atomic(out/'progress.json',dict(status='checking_input',file=name))
        assert sha(folder/name)==artifact['sha256']
    for name,key in [('preprocessing.npz','preprocessing_sha256'),('native_configs.json','native_configs_sha256')]:
        assert sha(previous/name)==record[key]
        shutil.copyfile(previous/name,out/name)
    prep=dict(np.load(out/'preprocessing.npz'))
    raw=np.load(folder/'train_raw.npy',mmap_mode='r');phi=record['phi_dim']
    assert raw.shape[1]==phi+832==18250 and phi==17418
    atomic(out/'progress.json',dict(status='fitting_native_summary_statistics',fit_split='train'))
    center,factor,indices=native_summary_statistics(raw,prep,phi)
    unchanged=np.ones(raw.shape[1],bool);unchanged[indices]=False
    assert np.array_equal(center[unchanged],prep['center'][unchanged])
    assert np.array_equal(factor[unchanged],prep['factor'][unchanged])
    np.savez(out/'native_summary_scaling.npz',center=center,factor=factor,indices=indices)
    for key in list(record):
        if key.startswith('quantile_') or key in ('stage2_requires_common_selection','cycle_rule_replay','cycle_replay_source'):
            del record[key]
    record.update(reused_preparation=str(previous),previous_manifest_sha256=sha(previous/'completion.json'),
                  native_summary_scaling_sha256=sha(out/'native_summary_scaling.npz'),
                  summary_numeric_features=len(indices),protocols={v:protocol(v) for v in VARIANTS},
                  preparation_complete=False,raw_cache_duplicated=False,F6_F7_started=False)
    atomic(out/'completion.json',record)
    # Capture fresh random-initialization identities, never trained model weights.
    from models.tracehr_unified_next import parameter_hash
    data=Data(project,database,out)
    for endpoint in (0,1):
        model=e_model(data,out,'E2',endpoint,42)
        atomic(out/f'e2_initialization_task{endpoint}.json',dict(complete=True,seed=42,endpoint=endpoint,
               initial_weights_sha256=parameter_hash(model),source='fresh E2 random initialization; zero optimizer steps'))
        del model
    record.update(preparation_complete=True,
                  initial_reference_sha256={str(e):sha(out/f'e2_initialization_task{e}.json') for e in (0,1)})
    atomic(out/'completion.json',record)
    atomic(out/'progress.json',dict(status='complete',train_rows=record['train_rows'],dev_rows=record['dev_rows'],
                                   input_dim=record['input_dim'],summary_numeric_features=len(indices)))


def model_for(data,preparation,variant,endpoint,seed,lr_multiplier=1.,selected=None,tmp_folder=None):
    assert variant!='F0' and seed==42 and lr_multiplier==1.
    from models.tracehr_unified_next import parameter_hash
    model=e_model(data,preparation,'E2',endpoint,seed,tmp_folder=tmp_folder)
    initial=parameter_hash(model);reference=preparation/f'e2_initialization_task{endpoint}.json'
    assert sha(reference)==data.record['initial_reference_sha256'][str(endpoint)]
    assert initial==read(reference)['initial_weights_sha256']
    configure_e2(model,protocol(variant))
    assert parameter_hash(model)==initial
    model.info['E2_initialization_match']=dict(complete=True,initial_hash=initial,path=str(reference),sha256=sha(reference))
    return model
