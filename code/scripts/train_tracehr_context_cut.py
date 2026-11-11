"""C1 entry point: identical G2 training loop; only the hidden-context edge is removed."""
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
from scripts import train_tracehr_execution_eicu_runtime as runner
from scripts.tracehr_context_cut_common import VARIANTS,Data,model_for,protocol,stopping
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
    drift=float((ours-reference).abs().max())
    assert drift==0 and torch.isfinite(active).all()
    return dict(complete=True,disabled_fusion_max_logit_drift=drift,
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
    p.add_argument('--variant',choices=VARIANTS,default='C1')
    p.add_argument('--endpoint',type=int,choices=[0,1],default=0)
    p.add_argument('--seed',type=int,choices=[42],default=42)
    p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.project=a.project.resolve(strict=True);a.lr_multiplier=1.
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
    data=Data(a.project,a.database,a.preparation);spec=protocol(a.variant,a.database)
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
    reference=read(a.project/f'runs/tracehr_structure_20261014/mimic/g_v1/screen/jobs/G2_t{a.endpoint}_s42/run_manifest.json')
    assert model.info['random_G2_before_removal_sha256']==reference['initial_weights_sha256']
    assert identity['parameters']==reference['parameters']-16448
    assert not any(n.startswith('fusion.context.') for n,p in model.named_parameters())
    differences={k:dict(reference=reference['protocol'].get(k),current=spec.get(k))
                 for k in set(reference['protocol'])|set(spec) if reference['protocol'].get(k)!=spec.get(k)}
    assert set(differences)=={'variant','structure','description','hidden_mean_context'}
    atomic(out/'connection_audit.json',dict(complete=True,reference_variant='G2',new_variant='C1',
          only_deleted_parameters=['fusion.context.weight','fusion.context.bias'],removed_parameters=16448,
          retained_random_tensors_match_G2=True,reference_random_initial_hash=reference['initial_weights_sha256'],
          protocol_differences=differences,learned_checkpoint_loaded=False,stop_gradient=False,
          all_remaining_parameters_trainable=True,extra_hidden_feedback_jacobian='identically zero by graph; unit test verified'))
    model=model.cuda();gc.collect()
    atomic(out/'G0_equivalence.json',initial_equivalence(model,data))
    atomic(out/'execution_audit.json',dict(effective_batch=256,microbatch=128,precision='FP32',
          training_sampler='unchanged E2 NumPy default_rng permutation; retain tail batch',
          batches_per_epoch=math.ceil(len(data.raw['train'])/256),initial_weights_match_E2=True,
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
