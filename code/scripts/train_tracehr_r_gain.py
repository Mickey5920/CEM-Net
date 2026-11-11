"""Bounded critical-path probe or scratch training; never a learned initialization."""
import argparse,gc,os,sys,shutil,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import tracehr_noa_ablation_runtime as runner
from scripts.tracehr_r_gain_common import Data,model_for,protocol,VARIANTS,MATRIX,expected,PROTOCOL_FIELDS
from scripts.tracehr_multiview_common import atomic,read,sha,checked_output
from models.tracehr_unified_next import parameter_hash
import numpy as np
import torch

def initialize(a,out,data):
    spec=protocol(a.variant,a.database,a.seed)
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    model=model_for(data,a.preparation,a.variant,a.endpoint,a.seed)
    identity=dict(database=a.database,variant=a.variant,endpoint=a.endpoint,seed=a.seed,lr_multiplier=1.,
        protocol=spec,source_manifest_sha256=sha(ROOT/'source_manifest.json'),input_manifest_sha256=sha(a.preparation/'completion.json'),
        selected_structure=spec['structure'],initial_weights_sha256=parameter_hash(model),common_weights_sha256=parameter_hash(model,True),
        parameters=sum(p.numel() for p in model.parameters()),input_dim=data.input_dim,members=model.members)
    atomic(out/'initialization_observation.json',dict(own_initial_hash=identity['initial_weights_sha256'],reference_noA_initial_hash=model.info['reference_noA_initial_weights_sha256'],parameters=identity['parameters'],optimizer_updates=0))
    ref=expected()['records'][f'C5_noT_noA_t{a.endpoint}_s{a.seed}']
    reference=read(Path(ref['source'])/'run_manifest.json');info=model.info
    assert info['reference_noA_initial_weights_sha256']==reference['initial_weights_sha256']
    assert info['reference_noA_parameter_count']==reference['parameters']
    assert identity['parameters']==reference['parameters']-info['removed_from_noA']
    differences={k:dict(reference=reference['protocol'].get(k),current=spec.get(k)) for k in set(reference['protocol'])|set(spec) if reference['protocol'].get(k)!=spec.get(k)}
    assert set(differences)<=PROTOCOL_FIELDS,differences
    assert info['retained_noA_tensors_exact'] and info['retained_noA_buffers_exact']
    assert all(p.requires_grad for p in model.parameters())
    atomic(out/'run_manifest.json',identity);atomic(out/'native_initialization.json',info)
    atomic(out/'connection_audit.json',dict(complete=True,reference_variant='C5_noT_noA',reference_seed=a.seed,
        reference_random_initial_hash=info['reference_noA_initial_weights_sha256'],reference_manifest_sha256=sha(Path(ref['source'])/'run_manifest.json'),
        retained_random_tensors_match_noA=True,retained_buffers_exact=True,parameter_count_removed=info['removed_from_noA'],
        removed_parameter_names=info['removed_from_noA_names'],protocol_differences=differences,learned_checkpoint_loaded=False,
        stop_gradient=False,all_parameters_trainable=True,members=8,main_view='z0',checkpoint_selection_view='z0',stopping_view='z0',
        auxiliary_enabled=False,main_loss_coefficient=.5,auxiliary_loss_coefficient=0.,main_residual_enabled=model.main_residual_enabled))
    model=model.cuda();gc.collect();model.eval()
    x,_=data.batch('train',np.arange(16))
    with torch.no_grad():
        active=model(x);z0,zA=model.forward_views(x)
        assert zA is None and torch.equal(active,z0)
        model.fusion_enabled=False;disabled=model(x);native=model.base(x['raw']);model.fusion_enabled=True
        assert torch.equal(disabled,native)
        if a.variant.endswith('noR'):assert torch.equal(z0,native)
    atomic(out/'G0_equivalence.json',dict(complete=True,disabled_fusion_max_logit_drift=0.,z0_single_dual_view_drift=0.,
        auxiliary_absent=True,noR_equals_own_backbone=not model.main_residual_enabled,fit_split='train',rows=16))
    # No weight update: verify the actual hook and original unmodulated R input.
    before=parameter_hash(model)
    x,y=data.batch('train',np.arange(16))
    raw=x['raw'];scaled=(raw-model.base.numerical.center)*model.base.numerical.factor
    scaled=torch.where(model.base.numerical.passthrough,raw,scaled/torch.sqrt(1+(scaled/3).square()))
    summary=torch.cat((scaled[:,model.phi_dim:model.phi_dim+352].reshape(-1,32,11),scaled[:,model.phi_dim+352:].reshape(-1,32,15)),-1)
    delta=model.summary_delta(summary)
    gm=torch.autograd.grad(delta.square().mean(),model.base.input_members,allow_unused=True,retain_graph=True)[0]
    assert gm is None, 'R must not receive M directly'
    from models.tracehr_r_gain import apply_gain
    hidden=torch.arange(8*16*256,device='cuda',dtype=torch.float32).reshape(8,16,256)/1000
    assert torch.equal(apply_gain(hidden,torch.zeros_like(delta)),hidden)
    assert torch.count_nonzero(apply_gain(torch.zeros_like(hidden),delta))==0
    model._context=(summary,None,None,False)
    hooked=model._before_head(None,({'x_cont':hidden},))[0]['x_cont'];model._context=None
    assert torch.equal(hooked,apply_gain(hidden,delta))
    seen=[]
    hook=model.fusion.summary.register_forward_pre_hook(lambda module,args:seen.append(args[0]))
    try:
        z0,zA=model.forward_views(x)
    finally:
        hook.remove()
    assert zA is None and len(seen)==1 and torch.equal(seen[0],summary)
    params=list(model.named_parameters())
    gs=torch.autograd.grad(model.loss((z0,zA),y[:,a.endpoint]),[p for _,p in params],allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in gs)
    group_norms={}
    for prefix in ('base.','fusion.'):
        group_norms[prefix]=float(sum(g.square().sum() for (n,p),g in zip(params,gs) if n.startswith(prefix)).sqrt())
        assert group_norms[prefix]>0
    # Check r=0 through the complete model without changing a parameter or buffer.
    original_delta=model.summary_delta
    model.summary_delta=lambda s: original_delta(s)*0
    try:
        with torch.no_grad():zero_gain=model(x)
        model.fusion_enabled=False
        with torch.no_grad():native=model(x)
    finally:
        model.fusion_enabled=True;del model.summary_delta
    assert torch.equal(zero_gain,native)
    gamma=(1+delta).detach().flatten()
    assert torch.isfinite(gamma).all() and parameter_hash(model)==before
    assert all(p.grad is None for p in model.parameters())
    atomic(out/'RG_connection_audit.json',dict(complete=True,formula='hidden * (1 + summary_delta)',
        unmodulated_summary_exact=True,summary_calls_per_forward=1,R_direct_M_gradient_absent=True,
        zero_gain_exact_native=True,zero_hidden_no_summary_output=True,actual_hook_formula_exact=True,
        weights_unchanged=True,optimizer_updates=0,all_parameters_main_connected=True,main_gradient_norms=group_norms,
        gamma=dict(min=float(gamma.min()),max=float(gamma.max()),mean=float(gamma.mean()),std=float(gamma.std()),
                   negative_fraction=float((gamma<0).float().mean())),fit_split='train',rows=16))
    atomic(out/'execution_audit.json',dict(effective_batch=256,microbatch=128,precision='FP32',from_scratch=True,
        training_sampler='unchanged E2 NumPy default_rng permutation; retain tail batch',all_learnable_parameters_trainable=True,
        optimizer_clock_reference_epochs=256,hyperparameter_search=False,main_loss_coefficient=.5,seed=a.seed))
    return model,identity,spec

def main():
    p=argparse.ArgumentParser()
    for name in ('project','output','preparation','pilot-record','start-barrier'):p.add_argument('--'+name,type=Path,required=name in ('project','output','preparation'))
    p.add_argument('--database',choices=['mimic'],required=True);p.add_argument('--action',choices=['probe','train'],required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--seed',type=int,choices=[43,44,45],required=True)
    p.add_argument('--endpoint',type=int,choices=[0,1],required=True);p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();a.project=a.project.resolve(strict=True);a.resume=False;a.lr_multiplier=1.
    assert (a.variant,a.endpoint,a.seed) in MATRIX
    if a.action=='probe':assert a.seed==43
    out=checked_output(a.project,a.output);assert not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30
    out.mkdir(parents=True)
    for key in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):checked_output(a.project,os.environ[key]).mkdir(parents=True,exist_ok=True)
    atomic(out/'storage.json',dict(project=str(a.project),code=str(ROOT),output=str(out),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    torch.set_num_threads(4);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=Data(a.project,'mimic',a.preparation);model,identity,spec=initialize(a,out,data)
    if a.action=='probe':runner.probe(a,out,data,model,identity,spec)
    else:
        assert a.pilot_record
        runner.train(a,out,data,model,identity,spec)

if __name__=='__main__':main()
