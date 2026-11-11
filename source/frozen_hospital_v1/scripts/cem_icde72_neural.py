"""Reuse original model factories, preprocessing, optimization, and EMA runtime."""
import gc, os
from pathlib import Path
import numpy as np
import torch
from scripts.cem_icde72_common import expected, mask_columns, read, sha, atomic
from scripts.cem_evidence_common import Data, tensor_hashes, parameter_hash
from models.cem_icde72_mask import MaskedPeriodic, mask_tabm, invariance

def removed_columns(data):
    original=getattr(data,'original',data)
    vocab=Path(os.environ['TRACEHR_PROJECT'])/expected()['native_source_cache']/'token_to_id.json'
    n_types=len(read(vocab))
    return mask_columns(original.input_dim,original.record['phi_dim'],n_types)

def fresh_cem(data,preparation,seed,endpoint,out,variant):
    assert variant=='CEM_basic'
    if os.environ['TRACEHR_DATABASE']=='mimiciii':
        from scripts.mimiciii_ema_common import fresh_model as parent
    else:
        from scripts.ema_cross_common import fresh_model as parent
    model=parent(data,preparation,seed,endpoint,out)
    before=tensor_hashes(model);initial=parameter_hash(model);ref=expected()['full_ema'][f'{seed}:{endpoint}']
    assert initial==ref['initial_weights_sha256'] and sum(p.numel() for p in model.parameters())==ref['parameters']
    original=read(out/'connection_audit.json');removed=removed_columns(data)
    old=model.base.numerical
    model.base.numerical=MaskedPeriodic(old,removed)
    assert tensor_hashes(model)==before and parameter_hash(model)==initial
    x=torch.as_tensor(np.array(data.raw['train'][:16],copy=True))
    with torch.no_grad():
        full=old(x);basic=model.base.numerical(x);keep=torch.ones(full.shape[1],dtype=torch.bool)
        keep[model.base.numerical.removed_encoded_columns]=False
        torch.testing.assert_close(full[:,keep],basic[:,keep],rtol=0,atol=0)
        assert not torch.count_nonzero(basic[:,~keep])
    proof=invariance(model.base,x,removed);del old
    mapping=dict(retained_initials_exact=True,parameter_slots_unchanged=True,raw_and_encoded_masked=True,**proof)
    observation=dict(complete=True,optimizer_updates=0,reference_seed=seed,full_initial_weights_sha256=initial,
                     initial_weights_sha256=initial,retained_tensor_hashes=before,deletion=mapping,
                     full_initial_tensors_exact=True,original_static_buffers_exact=True)
    atomic(out/'initialization_observation.json',observation)
    original.update(observation,P_retained=True,M_retained=True,evidence_branch_present=False,auxiliary_present=False,
                    learned_checkpoint_loaded=False,all_retained_parameters_trainable=True)
    atomic(out/'connection_audit.json',original)
    model.train();return model

def tabular_protocol(variant):
    base=variant.removesuffix('_basic').removesuffix('_hospital')
    if base=='TabM_mini':
        return dict(epochs=256,batch=256,micro=256,k=32,d_block=256,n_blocks=3,dropout=.2,
                    optimizer='AdamW',lr=.0005,weight_decay=.001,patience=16,selection='validation AP',precision='float32',
                    loss='mean member BCE',tail='retained',architecture='tabm-mini',embedding=None)
    return expected()['recipes'][base]

def tabular_model(variant,seed,data):
    from scripts.train_tracehr_unified import seed_all
    from models.cem_additional_baselines import tabm_model
    seed_all(seed);base=variant.removesuffix('_basic').removesuffix('_hospital');c=expected()
    if base=='TabM_mini':
        from tabm import TabM
        model=TabM.make(n_num_features=data.input_dim,d_out=1,k=32,d_block=256,n_blocks=3,dropout=.2,arch_type='tabm-mini')
        state=read(Path(os.environ['TRACEHR_PROJECT'])/'runs/mimiciii_pm_mini_20261024/v1/screen/state.json')
        path=Path(state['jobs'][f'TabM-mini_t1_s{seed}']['output'])/'run_manifest.json'
    else:
        bins=torch.load(Path(os.environ['TRACEHR_PROJECT'])/c['baseline_preparation']/'bins.pt',map_location='cpu',weights_only=False)
        model=tabm_model(base,bins)
        path=Path(os.environ['TRACEHR_PROJECT'])/f'runs/cem_evidence_completion_20261025/{c["database"]}/tabular_v1/screen/jobs/{base}_t0_s{seed}/run_manifest.json'
    if c['stage']=='hospital':
        assert base=='TabM_mini_PLE'
        model._evidence_audit=dict(complete=True,from_scratch=True,all_input_coordinates_retained=True,optimizer_updates=0,
            training_hospitals_only_preprocessing=True,initial_weights_sha256=parameter_hash(model),old_random_split_weights_loaded=False)
        return model
    initial=parameter_hash(model);reference=read(path)
    assert initial==reference['initial_weights_sha256'] and sum(p.numel() for p in model.parameters())==reference['parameters']
    model=mask_tabm(model,removed_columns(data))
    assert parameter_hash(model)==initial
    model._evidence_audit=dict(complete=True,full_reference=str(path),reference_sha256=sha(path),retained_initials_exact=True,
                               parameter_slots_unchanged=True,raw_and_encoded_masked=True,optimizer_updates=0)
    return model

def configure_tabular():
    from scripts import cem_additional_common as base
    base.expected=expected
    base.protocol=tabular_protocol
    if expected()['stage']=='hospital':
        from scripts.cem_icde72_hospital import HospitalData
        base.Data=HospitalData
    return base

def cpu_checks(project):
    if expected()['stage']=='hospital':
        from scripts.cem_icde72_hospital import HospitalData
        c=expected();data=HospitalData(project,project/c['input_preparation'])
        assert data.record['patient_overlap']==data.record['hospital_overlap']==0
        return ['patient_and_hospital_disjoint_train_validation','training_hospitals_only_vocabulary_lab_selection_scaling',
                'original_full_CEM_and_published_TabM_mini_PLE_recipes','all_features_retained','random_initialization_only']
    c=expected();data=Data(project,project/c['input_preparation']);removed=removed_columns(data)
    assert len(np.unique(removed))==len(removed) and not np.intersect1d(removed,np.arange(data.record['phi_dim'],data.record['phi_dim']+352)).size
    # All seed/endpoint recipes stay byte-equivalent to their paired full cells.
    from scripts.cem_evidence_common import budget,protocol,EMA_RULE
    for cell in c['matrix']:
        if cell['variant']!='CEM_basic':continue
        ref=c['full_ema'][f"{cell['seed']}:{cell['endpoint']}"]
        assert protocol('EMA_NRE_MLP',cell['seed'])==ref['protocol'] and budget(cell['seed'],cell['endpoint'])==ref['epochs']
    # The real initialization and invariance proofs are repeated by each worker,
    # including seeds 49/50; no learned or probe weights are ever reused.
    return ['six_archived_CEM_recipes_and_budgets_identical','full_parameter_layout_preserved',
            'basic_evidence_mask_exact_feature_blocks','all_simple_laboratory_summaries_retained']
