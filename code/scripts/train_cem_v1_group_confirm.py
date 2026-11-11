"""Fresh CEM plus one prespecified evidence structure; unchanged optimization loop."""
import argparse, gc, hashlib, os, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.cem_v1_group_confirm_common import *
from scripts.tracehr_multiview_common import checked_output
from scripts.train_tracehr_noa_main_confirmation import initial_equivalence
from scripts import cem_v1_group_confirm_runtime as runner
from scripts.cem_v1_group_confirm_checks import bounded_rate_check

runner.Optimizer=Optimizer;runner.diagnostics=diagnostics;runner.stopping=stopping


def longest_evidence_stress(model,data,spec,endpoint):
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=Optimizer(model,spec);opt.schedule(0.)
    ix=np.argsort(data.lengths['train'],kind='stable')[-256:]
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();tick=time.time()
    _,grad=runner.update(model,opt,data,ix,128,endpoint,check=True)
    runner.predict(model,data,256,np.argsort(data.lengths['dev'],kind='stable')[-256:])
    torch.cuda.synchronize()
    result=dict(complete=True,train_rows=len(ix),max_event_types=int(data.lengths['train'].max()),
        seconds=time.time()-tick,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        all_retained_tensors_have_main_gradients=all(g['main_connected']==g['tensors'] for g in grad['loss_paths']['groups'].values()))
    model.load_state_dict(initial);del initial,opt
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('project','output','preparation','pilot-record','start-barrier'):
        p.add_argument('--'+name,type=Path,required=name in ('project','output','preparation'))
    p.add_argument('--database',choices=['mimic'],required=True)
    p.add_argument('--action',choices=['probe','train'],required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--endpoint',type=int,choices=[0,1],required=True)
    p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();a.project=a.project.resolve(strict=True);a.resume=False;a.lr_multiplier=1.
    assert (a.variant,a.endpoint,a.seed) in matrix(a.database)
    out=checked_output(a.project,a.output);assert not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30
    out.mkdir(parents=True)
    for key in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        checked_output(a.project,os.environ[key]).mkdir(parents=True,exist_ok=True)
    atomic(out/'storage.json',dict(project=str(a.project),code=str(ROOT),output=str(out),
        preparation=str(a.preparation),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    torch.set_num_threads(4);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=Data(a.project,a.database,a.preparation,need_time=False);spec=protocol(a.variant,a.database,a.seed)
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    os.environ['CEM_INITIALIZATION_OBSERVER']=str(out/'S3_initialization_observation.json')
    model=model_for(data,a.preparation,a.variant,a.endpoint,a.seed);info=model.info
    identity=dict(database=a.database,variant=a.variant,endpoint=a.endpoint,seed=a.seed,lr_multiplier=1.,protocol=spec,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),input_manifest_sha256=sha(a.preparation/'completion.json'),
        selected_structure=spec['structure'],initial_weights_sha256=parameter_hash(model),common_weights_sha256=parameter_hash(model,True),
        parameters=sum(p.numel() for p in model.parameters()),input_dim=data.input_dim,members=model.members)
    ref=expected()['controls'][f'{CEM}_t{a.endpoint}_s{a.seed}']
    manifest=read(Path(ref['source'])/'run_manifest.json')
    atomic(out/'initialization_observation.json',dict(optimizer_updates=0,reference_seed=a.seed,
        actual_reference_CEM_hash=info['reference_CEM_initial_weights_sha256'],expected_reference_CEM_hash=manifest['initial_weights_sha256'],
        retained_tensor_hashes={n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}))
    assert info['reference_CEM_initial_weights_sha256']==manifest['initial_weights_sha256']
    assert info['reference_CEM_parameter_count']==manifest['parameters']
    assert info['retained_CEM_tensors_exact'] and info['retained_CEM_buffers_exact']
    assert identity['parameters']==manifest['parameters']+info['added_parameters']
    assert model.members==8 and model.main_residual_enabled and not model.auxiliary_enabled
    assert all(p.requires_grad for p in model.parameters())
    differences={k for k in set(spec)|set(manifest['protocol']) if spec.get(k)!=manifest['protocol'].get(k)}
    assert differences<=PROTOCOL_FIELDS,differences
    edge=dict(complete=True,reference_seed=a.seed,reference_CEM_random_initial_hash=info['reference_CEM_initial_weights_sha256'],
        retained_CEM_tensors_exact=True,added_parameters=info['added_parameters'],protocol_differences=sorted(differences),
        learned_checkpoint_loaded=False,stop_gradient=False,all_parameters_trainable=True,members=8,
        main_view='z0',checkpoint_selection_view='z0',stopping_view='z0',auxiliary_enabled=False,auxiliary_base_dependency=False,
        main_loss_coefficient=.5,auxiliary_loss_coefficient=0.,main_residual_enabled=True,
        fusion_position=info['fusion_position'],same_original_inputs=True,time_bin_reorganization=False,
        event_direct_enabled=info['event_direct_enabled'],interaction_event_skip_enabled=info['interaction_event_skip_enabled'],
        retained_S3_tensors_exact=info['retained_S3_tensors_exact'],fresh_S3_retained_initialization_verified=info['retained_S3_tensors_exact'],
        reference_S3_initial_weights_sha256=info['reference_S3_initial_weights_sha256'],
        removed_S3_parameters=info['removed_S3_parameters'],removed_S3_parameter_names=info['removed_S3_parameter_names'],
        new_S3_parameters=info['new_S3_parameters'],new_S3_parameter_names=info['new_S3_parameter_names'])
    atomic(out/'connection_audit.json',edge);atomic(out/'run_manifest.json',identity);atomic(out/'native_initialization.json',info)
    model=model.cuda();gc.collect();atomic(out/'G0_equivalence.json',initial_equivalence(model,data))
    atomic(out/'execution_audit.json',dict(from_scratch=True,training_sampler='unchanged NumPy default_rng permutation; retain tail',
        effective_batch=256,microbatch=128,precision='FP32',hyperparameter_search=False, prespecified_group_rate_only=True,optimizer_clock_reference_epochs=256))
    if a.action=='probe':
        rate_check=bounded_rate_check(model,data,spec,a.endpoint,a.seed)
        atomic(out/'rate_check.json',rate_check)
        stress=longest_evidence_stress(model,data,spec,a.endpoint)
        assert parameter_hash(model)==identity['initial_weights_sha256']
        runner.probe(a,out,data,model,identity,spec)
        r=read(out/'completion.json');r.update(longest_evidence_stress=stress,rate_check=rate_check,
            peak_reserved_mib=max(r['peak_reserved_mib'],stress['peak_reserved_mib']))
        atomic(out/'completion.json',r)
    else:
        assert a.pilot_record;runner.train(a,out,data,model,identity,spec)


if __name__=='__main__':main()
