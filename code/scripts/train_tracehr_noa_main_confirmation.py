"""C5 single-component ablations, unchanged training and fixed z0 selection."""
import argparse
import gc
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# These globals are scoped to this new worker process. The immutable running F
# snapshot is never edited, imported into its process, or otherwise mutated.
from scripts import tracehr_not_functional_runtime as runner
from scripts.tracehr_noa_main_confirmation_common import VARIANTS,Data,model_for,protocol,stopping,PROTOCOL_FIELDS,MATRIX
from scripts.tracehr_structure_runtime import Optimizer,gradients,diagnostics
from scripts.tracehr_multiview_common import atomic,read,sha,checked_output
from models.tracehr_unified_next import parameter_hash
import numpy as np
import torch

runner.Optimizer=Optimizer
runner.gradients=gradients
runner.diagnostics=diagnostics
runner.stopping=stopping


def initial_equivalence(model,data):
    model.eval()
    batch,_=data.batch('train',np.arange(16))
    with torch.no_grad():
        model.fusion_enabled=False
        ours=model(batch)
        reference=model.base(batch['raw'])
        model.fusion_enabled=True
        active=model(batch)
        z0,z1=model.forward_views(batch)
        view_drift=float((active-z0).abs().max())
        probability_drift=float((active.softmax(-1)-z0.softmax(-1)).abs().max())
        # Batched CUDA GEMM uses a different accumulation layout for 8 vs 16
        # members. Check relative+absolute FP32 agreement, and probabilities,
        # rather than an absolute logit threshold independent of logit scale.
        torch.testing.assert_close(active,z0,atol=1e-6,rtol=1e-6)
        assert probability_drift<=1e-6
        if model.auxiliary_enabled:assert z1 is not None and torch.isfinite(z1).all()
        else:assert z1 is None
        if not model.main_residual_enabled:assert torch.equal(active,reference)
    drift=float((ours-reference).abs().max())
    assert drift==0 and torch.isfinite(active).all()
    return dict(complete=True,disabled_fusion_max_logit_drift=drift,z0_single_dual_view_drift=view_drift,
                z0_single_dual_probability_drift=probability_drift,
                active_fusion_logit_difference=float((active-reference).abs().max()),
                base_initial_weights_sha256=parameter_hash(model.base),fit_split='train',rows=16)


def long_sequence_stress(model,data,spec,endpoint):
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=Optimizer(model,spec);opt.schedule(0.)
    lengths=np.diff(data.trajectory['train']['offsets'])
    ix=np.argsort(lengths,kind='stable')[-256:]
    torch.cuda.reset_peak_memory_stats();tick=time.time()
    runner.update(model,opt,data,ix,128,endpoint,check=True)
    dev_lengths=np.diff(data.trajectory['dev']['offsets'])
    runner.predict(model,data,256,np.argsort(dev_lengths,kind='stable')[-256:])
    torch.cuda.synchronize()
    result=dict(complete=True,train_patients=len(ix),train_observations=int(lengths[ix].sum()),
                max_patient_observations=int(lengths.max()),seconds=time.time()-tick,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    model.load_state_dict(initial);del initial,opt
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('project','output','preparation','pilot-record','start-barrier'):
        p.add_argument('--'+name,type=Path,required=name in ('project','output'))
    p.add_argument('--database',choices=['mimic'],required=True)
    p.add_argument('--action',choices=['probe','train'],required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--endpoint',type=int,choices=[0,1],default=0)
    p.add_argument('--seed',type=int,choices=[48,49,50],required=True)
    p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.project=a.project.resolve(strict=True);a.lr_multiplier=1.
    assert not a.resume and a.variant=='C5_noT_noA'
    assert (a.variant,a.endpoint,a.seed) in MATRIX
    out=checked_output(a.project,a.output)
    assert out.is_dir() if a.resume else not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30
    out.mkdir(parents=True,exist_ok=a.resume)
    for key in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        checked_output(a.project,os.environ[key]).mkdir(parents=True,exist_ok=True)
    atomic(out/'storage.json',dict(project=str(a.project),code=str(ROOT.resolve()),output=str(out),
                                 cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    torch.set_num_threads(4)
    assert a.preparation and torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=Data(a.project,a.database,a.preparation);spec=protocol(a.variant,a.database,a.seed)
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    model=model_for(data,a.preparation,a.variant,a.endpoint,a.seed)
    identity=dict(database=a.database,variant=a.variant,endpoint=a.endpoint,seed=a.seed,lr_multiplier=1.,
                  protocol=spec,source_manifest_sha256=sha(ROOT/'source_manifest.json'),
                  input_manifest_sha256=sha(a.preparation/'completion.json'),selected_structure=spec['structure'],
                  initial_weights_sha256=parameter_hash(model),common_weights_sha256=parameter_hash(model,True),
                  parameters=sum(p.numel() for p in model.parameters()),input_dim=data.input_dim,members=model.members)
    assert all(p.requires_grad for p in model.parameters())
    if a.resume:assert read(out/'run_manifest.json')==identity
    else:atomic(out/'run_manifest.json',identity)
    atomic(out/'native_initialization.json',model.info)
    reference_path=a.project/f'runs/tracehr_c5_ablation_20261017/mimic/ablation_v1/screen/jobs/C5_noT_t{a.endpoint}_s42/run_manifest.json'
    reference=read(reference_path);info=model.info
    if a.seed==42:
        assert info['reference_noT_initial_weights_sha256']==reference['initial_weights_sha256']
    else:
        assert info['reference_noT_initial_weights_sha256']!=reference['initial_weights_sha256']
    assert info['reference_noT_parameter_count']==reference['parameters']
    assert identity['parameters']==reference['parameters']-info['removed_from_noT']
    assert info['retained_noT_tensors_exact'] and info['retained_noT_buffers_exact']
    differences={k:dict(reference=reference['protocol'].get(k),current=spec.get(k))
                 for k in set(reference['protocol'])|set(spec) if reference['protocol'].get(k)!=spec.get(k)}
    assert set(differences)<=PROTOCOL_FIELDS
    assert spec['seed']==a.seed and info['seed_route']['training_order_seed']==a.seed
    atomic(out/'connection_audit.json',dict(complete=True,reference_variant='C5_noT',new_variant=a.variant,
          parameter_count_removed=info['removed_from_noT'],retained_random_tensors_match_noT=True,
          retained_buffers_exact=True,removed_parameter_names=info['removed_from_noT_names'],
          reference_random_initial_hash=info['reference_noT_initial_weights_sha256'],
          reference_seed=a.seed,seed42_reference_manifest_sha256=sha(reference_path),
          seed42_reference_initial_hash=reference['initial_weights_sha256'],seed_route=info['seed_route'],
          protocol_differences=differences,learned_checkpoint_loaded=False,stop_gradient=False,
          original_summary_features_preserved=True,all_parameters_trainable=True,members=8,
          main_view='z0',checkpoint_selection_view='z0',stopping_view='z0',
          single_backbone_forward=True,shared_classifier=False,auxiliary_base_dependency=False,
          auxiliary_enabled=model.auxiliary_enabled,main_residual_enabled=model.main_residual_enabled,
          main_loss_coefficient=.5,auxiliary_loss_coefficient=.5 if model.auxiliary_enabled else 0.))
    model=model.cuda();gc.collect()
    atomic(out/'G0_equivalence.json',initial_equivalence(model,data))
    atomic(out/'execution_audit.json',dict(effective_batch=256,microbatch=128,precision='FP32',
          training_sampler='unchanged E2 NumPy default_rng permutation; retain tail batch',
          batches_per_epoch=math.ceil(len(data.raw['train'])/256),retained_parameter_initialization_matches_noT=True,
          all_learnable_parameters_trainable=True,optimizer_clock_reference_epochs=256,
          fusion_training='same first joint update, no freezing/pretraining/distillation',
          hyperparameter_search=False))
    if a.action=='probe':
        atomic(out/'progress.json',dict(status='long_sequence_stress'))
        stress=long_sequence_stress(model,data,spec,a.endpoint)
        assert parameter_hash(model)==identity['initial_weights_sha256']
        runner.probe(a,out,data,model,identity,spec)
        result=read(out/'completion.json')
        result.update(long_sequence_stress=stress,evaluation_microbatch=256,
                      peak_reserved_mib=max(result['peak_reserved_mib'],stress['peak_reserved_mib']))
        atomic(out/'completion.json',result)
    else:
        assert a.pilot_record
        runner.train(a,out,data,model,identity,spec)


if __name__=='__main__':main()
