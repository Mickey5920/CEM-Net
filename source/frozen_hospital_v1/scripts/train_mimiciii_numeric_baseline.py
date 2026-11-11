"""III input routing around unchanged native and branch-deletion training loops."""
import argparse,gc,hashlib,os,random,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.mimiciii_baselines_common import *
from scripts.tracehr_multiview_common import atomic
import numpy as np
import torch
from torch import nn
from scripts.mimiciii_fixed_common import Data,model_for,parameter_hash,stopping
from scripts.tracehr_oof_common import all_scores
from scripts.train_tracehr_unified import seed_all,rng_state,restore_rng

class NoEvidence(nn.Module):
    def __init__(self,full):
        super().__init__();full._hook.remove()
        self.base=full.base;self.fusion=nn.Module();self.members=8
        self.auxiliary_enabled=self.main_residual_enabled=False;self.fusion_enabled=True
        self.variant='noEvidence';self.info=dict(full.info,main_residual_enabled=False,additional_channels=0)
        assert parameter_hash(self)==full.info['reference_CEM_initial_weights_sha256']
    def forward(self,x):return self.base(x['raw'])
    def forward_views(self,x):return self(x),None
    def path_losses(self,z,y):return self.base.loss(z,y),None
    def loss(self,z,y):
        loss=self.base.loss(z,y);self._last_path_losses=torch.stack((loss.detach(),loss.detach()*0));return .5*loss

def branch(a,out,data,identity):
    from scripts import cem_v1_group_confirm_runtime as runner
    from scripts.tracehr_noa_ablation_common import Optimizer,diagnostics
    from scripts.train_tracehr_noa_main_confirmation import initial_equivalence
    runner.Optimizer=Optimizer;runner.diagnostics=diagnostics;runner.stopping=stopping
    full=model_for(data,a.preparation,a.endpoint,a.seed);model=NoEvidence(full);del full
    identity.update(initial_weights_sha256=parameter_hash(model),common_weights_sha256=parameter_hash(model,True),
        parameters=sum(p.numel() for p in model.parameters()),members=8,lr_multiplier=1.,selected_structure='noEvidence')
    old=a.project/expected()['predecessor']/'screen/jobs'/f'V1_group_rate_t{a.endpoint}_s{a.seed}'
    reference=read(old/'initialization_observation.json')
    hashes={n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
    assert set(hashes)=={n for n in reference['retained_tensor_hashes'] if n.startswith('base.')}
    assert all(reference['retained_tensor_hashes'][n]==h for n,h in hashes.items())
    assert identity['initial_weights_sha256']==reference['actual_reference_CEM_hash']
    atomic(out/'initialization_observation.json',dict(complete=True,optimizer_updates=0,retained_tensor_hashes=hashes,
        full_model_initialization_sha256=sha(old/'initialization_observation.json'),same_seed_full_base_exact=True))
    atomic(out/'connection_audit.json',dict(complete=True,reference_seed=a.seed,retained_CEM_tensors_exact=True,
        learned_checkpoint_loaded=False,evidence_branch_removed=True,auxiliary_enabled=False,main_loss_coefficient=.5))
    atomic(out/'run_manifest.json',identity);model.cuda();gc.collect()
    atomic(out/'G0_equivalence.json',initial_equivalence(model,data))
    if a.action=='probe':runner.probe(a,out,data,model,identity,identity['protocol'])
    else:runner.train(a,out,data,model,identity,identity['protocol'])

def b0(a,out,data,identity):
    import joblib
    from scripts import train_tracehr_noa_validation_native as rt
    from models.tracehr_centered_native import FullNative
    rt.protocol=lambda variant:numeric_protocol('B0',a.seed)
    rt.stopping=stopping
    capture=np.random.default_rng(a.seed);saved=rng_state(capture)
    model=FullNative(data.raw,{s:data.targets[s]['y'] for s in ('train','dev')},a.endpoint,out/'fresh_native_init_tmp',a.seed)
    a.fresh_initial_hash=model.info['native_initial_trainable_sha256'];a.fresh_internal_seeds=model.info['internal_seeds']
    atomic(out/'fresh_initialization.json',model.info);del model;gc.collect();torch.cuda.empty_cache();restore_rng(saved,capture)
    if a.action=='probe':a.action='audit'
    trace=rt.NativeTrace(a,out,data,identity);trace.install();model=None;started=time.time()
    try:model,pred,_,best_epoch,_=rt.native_fit(data.raw,{s:data.targets[s]['y'] for s in ('train','dev')},out,a.endpoint,a.seed,256,'cuda:0',None)
    except rt.ShortRunComplete:assert a.action=='audit' and trace.steps==20
    finally:trace.uninstall()
    if a.action=='audit':
        result=dict(complete=True,pilot=True,identity=identity,updates=trace.steps,
            continuous_native_entry_parity=True,max_gradient_drift=trace.max_gradient_drift,
            max_parameter_drift=trace.max_parameter_drift,max_logit_drift=trace.max_logit_drift,
            checkpoint_bytes=identity['parameters']*12+4*2**20)
    else:
        assert model is not None
        model.to('cpu');joblib.dump(model,out/'best.joblib');del model;gc.collect();torch.cuda.empty_cache()
        loaded=joblib.load(out/'best.joblib');loaded.to('cuda:0');replay=loaded.predict_proba(data.raw['dev'])[:,1]
        drift=float(np.max(np.abs(pred-replay)));assert drift<=1e-5
        labels=data.targets['dev']['y'][:,a.endpoint:a.endpoint+1];metrics=all_scores(labels,pred[:,None])
        assert abs(metrics['mean_ap']-max(h['dev_ap'] for h in trace.history))<1e-7
        np.savez_compressed(out/'predictions_dev.npz',probs=pred[:,None],labels=labels,ids=data.targets['dev']['ids'])
        atomic(out/'metrics.json',metrics)
        result=dict(complete=True,pilot=False,**identity,epochs=len(trace.history),best_epoch=best_epoch,
            dev_ap=metrics['mean_ap'],metrics=metrics,stopping_status=stopping(trace.history,identity['protocol']),
            checkpoint_file='best.joblib',checkpoint_sha256=sha(out/'best.joblib'),predictions_sha256=sha(out/'predictions_dev.npz'),
            checkpoint_replay=dict(complete=True,max_probability_drift=drift))
    result.update(from_scratch=True,test_loaded=False,frozen_learnable_parameters=0,seconds=time.time()-started,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,finished_at=time.time())
    atomic(out/'completion.json',result);atomic(out/'progress.json',dict(status='complete',updates=trace.steps,dev_ap=result.get('dev_ap')))

def tabm(a,out,data,identity):
    import joblib
    from tabm import TabM
    from scripts.train_tracehr_native_baseline import tabm_fit,tabm_predict
    transform=joblib.load(a.project/expected()['preparation']/'tabm_preprocessing.joblib');raw={};d=transform['phi_dim']
    for s in ('train','dev'):
        z=np.array(data.raw[s],copy=True)
        for start in range(0,len(z),1024):
            block=z[start:start+1024,:d];v=(block-transform['phi_center'])*transform['phi_factor'];v/=np.sqrt(1+(v/3)**2)
            v[:,transform['binary']]=block[:,transform['binary']];z[start:start+1024,:d]=v
        z[:,d:]=transform['numeric_quantile'].transform(z[:,d:]);assert np.isfinite(z).all();raw[s]=z
    original=TabM.make
    def make(*args,**kwargs):
        model=original(*args,**kwargs)
        identity.update(initial_weights_sha256=parameter_hash(model),parameters=sum(p.numel() for p in model.parameters()),members=32)
        if a.pilot_record:assert read(a.pilot_record/'completion.json')['identity']['initial_weights_sha256']==identity['initial_weights_sha256']
        atomic(out/'run_manifest.json',identity);atomic(out/'progress.json',dict(status='training',updates=0));return model
    TabM.make=make
    torch.manual_seed(a.seed);np.random.seed(a.seed);started=time.time()
    try:model,pred,history,best_epoch=tabm_fit(raw,{s:data.targets[s]['y'] for s in raw},out,a.endpoint,a.seed,2 if a.action=='probe' else 256)
    finally:TabM.make=original
    drift=float(np.max(np.abs(pred-tabm_predict(model,raw['dev']))));assert drift<=1e-5
    labels=data.targets['dev']['y'][:,a.endpoint:a.endpoint+1];metrics=all_scores(labels,pred[:,None])
    np.savez_compressed(out/'predictions_dev.npz',probs=pred[:,None],labels=labels,ids=data.targets['dev']['ids'])
    atomic(out/'metrics.json',metrics)
    result=dict(complete=True,pilot=a.action=='probe',identity=identity,**identity,epochs=len(history),best_epoch=best_epoch,
        dev_ap=metrics['mean_ap'],metrics=metrics,from_scratch=True,test_loaded=False,frozen_learnable_parameters=0,
        seconds=time.time()-started,finished_at=time.time(),checkpoint_bytes=(out/'best.pt').stat().st_size,
        checkpoint_file='best.pt',checkpoint_sha256=sha(out/'best.pt'),predictions_sha256=sha(out/'predictions_dev.npz'),
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,checkpoint_replay=dict(complete=True,max_probability_drift=drift))
    atomic(out/'completion.json',result);atomic(out/'progress.json',dict(status='complete',epochs=len(history),dev_ap=result['dev_ap']))

def main():
    p=argparse.ArgumentParser()
    for n in ('project','output','pilot-record'):p.add_argument('--'+n,type=Path,required=n!='pilot-record')
    p.add_argument('--variant',choices=NUMERIC,required=True);p.add_argument('--endpoint',type=int,choices=[0,1],required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--action',choices=['probe','train'],required=True)
    p.add_argument('--expected-gib',type=float,required=True);a=p.parse_args();a.project=a.project.resolve(strict=True)
    a.preparation=a.project/expected()['numeric_preparation'];a.resume=False;a.lr_multiplier=1.;a.start_barrier=None
    out=a.output.resolve();assert out.is_relative_to(a.project) and not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30;out.mkdir(parents=True)
    for k in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        v=Path(os.environ[k]).resolve();assert v.is_relative_to(a.project);v.mkdir(parents=True,exist_ok=True)
    atomic(out/'storage.json',dict(project=str(a.project),output=str(out),preparation=str(a.preparation),expected_gib=a.expected_gib,
        code=str(ROOT),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR']))
    torch.set_num_threads(4);assert torch.cuda.is_available();seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats();atomic(out/'progress.json',dict(status='loading_verified_inputs'));data=Data(a.preparation)
    identity=dict(database='mimiciii',variant=a.variant,endpoint=a.endpoint,seed=a.seed,
        input_dim=data.input_dim,source_manifest_sha256=sha(ROOT/'source_manifest.json'),input_manifest_sha256=sha(a.preparation/'completion.json'),
        protocol=numeric_protocol(a.variant,a.seed))
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    {'noEvidence':branch,'B0':b0,'TabM':tabm}[a.variant](a,out,data,identity)
if __name__=='__main__':main()
