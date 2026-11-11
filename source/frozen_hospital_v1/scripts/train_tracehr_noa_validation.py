"""Existing scratch loops; only database/seed/deletion routing and identity checks are new."""
import argparse,gc,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_noa_validation_common import *
from scripts.tracehr_multiview_common import atomic,sha,checked_output
from scripts.train_tracehr_noa_main_confirmation import initial_equivalence,long_sequence_stress
import numpy as np
import torch
def main():
    p=argparse.ArgumentParser()
    for name in ('project','output','preparation','pilot-record','start-barrier'):
        p.add_argument('--'+name,type=Path,required=name in ('project','output','preparation'))
    p.add_argument('--database',choices=['mimic','eicu'],required=True)
    p.add_argument('--action',choices=['probe','train'],required=True)
    p.add_argument('--variant',choices=[*VARIANTS,'C5_noT_noA'],required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--endpoint',type=int,choices=[0,1],required=True)
    p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();a.project=a.project.resolve(strict=True);a.resume=False;a.lr_multiplier=1.
    assert (a.variant,a.endpoint,a.seed) in matrix(a.database)
    if a.database=='mimic' and a.action=='probe':assert a.seed==48
    out=checked_output(a.project,a.output);assert not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30
    out.mkdir(parents=True)
    for key in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):checked_output(a.project,os.environ[key]).mkdir(parents=True,exist_ok=True)
    atomic(out/'storage.json',dict(project=str(a.project),code=str(ROOT),output=str(out),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    torch.set_num_threads(4);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=Data(a.project,a.database,a.preparation);spec=protocol(a.variant,a.database,a.seed)
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    model=model_for(data,a.preparation,a.variant,a.endpoint,a.seed);info=model.info
    identity=dict(database=a.database,variant=a.variant,endpoint=a.endpoint,seed=a.seed,lr_multiplier=1.,protocol=spec,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),input_manifest_sha256=sha(a.preparation/'completion.json'),
        selected_structure=spec['structure'],initial_weights_sha256=parameter_hash(model),common_weights_sha256=parameter_hash(model,True),
        parameters=sum(p.numel() for p in model.parameters()),input_dim=data.input_dim,members=model.members)
    assert model.members==8 and not model.auxiliary_enabled and all(p.requires_grad for p in model.parameters())
    if a.database=='mimic':
        ref=expected()['mimic_records'][f'C5_noT_noA_t{a.endpoint}_s{a.seed}']
        reference=read(Path(ref['source'])/'run_manifest.json')
        atomic(out/'initialization_observation.json',dict(optimizer_updates=0,RNG_or_parameters_modified=False,original_assertions_preserved=True,
            actual_reference_noA_hash=info['reference_noA_initial_weights_sha256'],expected_reference_noA_hash=reference['initial_weights_sha256'],
            retained_tensor_hashes={n:__import__('hashlib').sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}))
        assert info['reference_noA_initial_weights_sha256']==reference['initial_weights_sha256']
        assert identity['parameters']==reference['parameters']-info['removed_from_noA']
        differences={k for k in set(spec)|set(reference['protocol']) if spec.get(k)!=reference['protocol'].get(k)}
        assert differences<=PROTOCOL_FIELDS,differences
        edge=dict(reference_random_initial_hash=info['reference_noA_initial_weights_sha256'],retained_random_tensors_match_noA=True,
            parameter_count_removed=info['removed_from_noA'],removed_parameter_names=info['removed_from_noA_names'],protocol_differences=sorted(differences))
        from scripts import tracehr_noa_validation_runtime as runner
    else:
        assert data.input_dim==18250 and data.record['phi_dim']==17418
        assert info['removed_from_noT']==514 and info['retained_noT_tensors_exact'] and info['retained_noT_buffers_exact']
        edge=dict(reference_random_initial_hash=info['reference_noT_initial_weights_sha256'],retained_random_tensors_match_noT=True,
            parameter_count_removed=514,removed_parameter_names=info['removed_from_noT_names'],reference_kind='fresh same-seed noT construction; no learned state')
        from scripts import tracehr_not_functional_runtime as runner
    runner.Optimizer=Optimizer;runner.diagnostics=diagnostics;runner.stopping=stopping
    edge.update(complete=True,reference_seed=a.seed,learned_checkpoint_loaded=False,stop_gradient=False,all_parameters_trainable=True,
        members=8,main_view='z0',checkpoint_selection_view='z0',stopping_view='z0',auxiliary_enabled=False,auxiliary_base_dependency=False,
        main_loss_coefficient=.5,auxiliary_loss_coefficient=0.,main_residual_enabled=model.main_residual_enabled)
    atomic(out/'connection_audit.json',edge);atomic(out/'run_manifest.json',identity);atomic(out/'native_initialization.json',info)
    model=model.cuda();gc.collect();atomic(out/'G0_equivalence.json',initial_equivalence(model,data))
    atomic(out/'execution_audit.json',dict(from_scratch=True,training_sampler='unchanged E2 NumPy default_rng permutation; retain tail',
        effective_batch=256,microbatch=128,precision='FP32',hyperparameter_search=False,optimizer_clock_reference_epochs=256))
    if a.action=='probe':
        stress=long_sequence_stress(model,data,spec,a.endpoint) if a.database=='eicu' else None
        assert parameter_hash(model)==identity['initial_weights_sha256']
        runner.probe(a,out,data,model,identity,spec)
        if stress:
            r=read(out/'completion.json');r.update(long_sequence_stress=stress,peak_reserved_mib=max(r['peak_reserved_mib'],stress['peak_reserved_mib']));atomic(out/'completion.json',r)
    else:
        assert a.pilot_record;runner.train(a,out,data,model,identity,spec)
if __name__=='__main__':main()
