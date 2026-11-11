"""F1--F5 reuse the verified E2 loop; only declared runtime hyperparameters vary."""
import argparse
import gc
import math
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_tracehr_execution_eicu_runtime as runner
from scripts.tracehr_execution_eicu_common import VARIANTS,Data,prepare,model_for,protocol
from scripts.tracehr_multiview_common import atomic,read,sha,checked_output
from models.tracehr_unified_next import parameter_hash
import torch


def main():
    p=argparse.ArgumentParser()
    for name in ('project','output','preparation','pilot-record','start-barrier'):
        p.add_argument('--'+name,type=Path,required=name in ('project','output'))
    p.add_argument('--database',choices=['eicu'],required=True)
    p.add_argument('--action',choices=['prepare','probe','smoke','train'],required=True)
    p.add_argument('--variant',choices=VARIANTS,default='F1')
    p.add_argument('--endpoint',type=int,choices=[0,1],default=0)
    p.add_argument('--seed',type=int,default=42)
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
    if a.action=='prepare':prepare(a.project,a.database,out);return
    assert a.variant!='F0' and a.preparation and torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=Data(a.project,a.database,a.preparation);spec=protocol(a.variant)
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    model=model_for(data,a.preparation,a.variant,a.endpoint,a.seed,tmp_folder=out/'native_init_tmp')
    identity=dict(database=a.database,variant=a.variant,endpoint=a.endpoint,seed=a.seed,lr_multiplier=1.,
                  protocol=spec,source_manifest_sha256=sha(ROOT/'source_manifest.json'),
                  input_manifest_sha256=sha(a.preparation/'completion.json'),selected_structure=spec['structure'],
                  initial_weights_sha256=parameter_hash(model),common_weights_sha256=parameter_hash(model,True),
                  parameters=sum(p.numel() for p in model.parameters()),input_dim=data.input_dim,members=model.members)
    assert all(p.requires_grad for p in model.parameters())
    if a.resume:assert read(out/'run_manifest.json')==identity
    else:atomic(out/'run_manifest.json',identity)
    atomic(out/'native_initialization.json',model.info)
    model=model.cuda();gc.collect()
    atomic(out/'execution_audit.json',dict(effective_batch=256,microbatch=128,precision='FP32',
          training_sampler='unchanged E2 NumPy default_rng permutation; retain tail batch',
          batches_per_epoch=math.ceil(len(data.raw['train'])/256),initial_weights_match_E2=True,
          all_learnable_parameters_trainable=True,optimizer_clock_reference_epochs=256))
    if a.action=='probe':runner.probe(a,out,data,model,identity,spec)
    else:
        assert a.pilot_record
        runner.train(a,out,data,model,identity,spec)


if __name__=='__main__':main()
