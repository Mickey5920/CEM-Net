"""Bounded main-path equivalence; diagnostic weights never initialize formal fits."""
import argparse,gc,math,os,sys,time,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tracehr_noa_ablation_common import Data,model_for,protocol,expected,Optimizer,stopping
from scripts.tracehr_not_stability_common import model_for as old_model,protocol as old_protocol
from scripts.tracehr_structure_runtime import Optimizer as OldOptimizer
from scripts.tracehr_noa_ablation_runtime import update,predict
from scripts.tracehr_recovery_queue import read,sha,atomic
from models.tracehr_unified_next import parameter_hash
from scripts.train_tracehr_unified import seed_all
import numpy as np
import torch

class NotEquivalent(Exception):pass
def exact(a,b,path='root'):
    if torch.is_tensor(a):
        if not torch.equal(a,b):raise NotEquivalent(path)
    elif isinstance(a,dict):
        if set(a)!=set(b):raise NotEquivalent(path+' keys')
        for k in a:
            if k!='spec':exact(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise NotEquivalent(path+' length')
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'.'+str(i))
    elif a!=b:raise NotEquivalent(path)

def audit(a):
    started=time.time();p=a.project.resolve(strict=True);out=a.output.resolve()
    assert out.is_relative_to(p) and not out.exists() and a.seed in (43,44)
    assert shutil.disk_usage(p).free>(a.expected_gib+10)*2**30
    out.mkdir(parents=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);assert torch.cuda.is_available()
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=Data(p,'mimic',a.preparation)
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    old=old_model(data,a.preparation,'C5_noT_noR',a.endpoint,a.seed)
    oldrow=expected()['records'][f'C5_noT_noR_t{a.endpoint}_s{a.seed}'];oldpath=Path(oldrow['source'])
    manifest=read(oldpath/'run_manifest.json');completion=read(oldpath/'completion.json')
    assert parameter_hash(old)==manifest['initial_weights_sha256']
    assert sha(oldpath/'best.pt')==oldrow['checkpoint_sha256']
    assert sha(oldpath/'predictions_dev.npz')==oldrow['predictions_sha256']
    new=model_for(data,a.preparation,'C5_noT_noA_noR',a.endpoint,a.seed)
    assert new.info['reference_noA_initial_weights_sha256']==read(Path(expected()['records'][f'C5_noT_noA_t{a.endpoint}_s{a.seed}']['source'])/'run_manifest.json')['initial_weights_sha256']
    spec=protocol('C5_noT_noA_noR','mimic',a.seed);ospec=old_protocol('C5_noT_noR','mimic',a.seed)
    allowed={'variant','structure','description','ablated_component','ablation_implementation','training_objective','auxiliary_topology'}
    assert {k for k in set(spec)|set(ospec) if spec.get(k)!=ospec.get(k)}<=allowed
    assert not any(isinstance(m,torch.nn.Dropout) for m in old.fusion.modules())
    history=read(oldpath/'history.json');assert stopping(history,spec)==completion['stopping_status']
    assert not any(stopping(history[:i],spec)['stop'] for i in range(1,len(history)))
    assert all(r['main_view']==r['checkpoint_selection_view']==r['stopping_view']=='z0' for r in history)
    source_manifest=p/'work/tracehr_not_stability_20261018/stage1_v1/source_manifest.json'
    assert sha(source_manifest)==manifest['source_manifest_sha256']
    for n,h in read(source_manifest).items():assert sha(source_manifest.parent/n)==h
    old=old.cuda();new=new.cuda();gc.collect()
    opt0=OldOptimizer(old,ospec);opt1=Optimizer(new,spec)
    n=len(data.targets['train']['y']);steps=math.ceil(n/256)
    order=np.random.default_rng(a.seed).permutation(n);seed_all(a.seed+10000)
    equivalent=True;reason=None;updates=0;early=0;replay_drift=None
    torch.cuda.reset_peak_memory_stats()
    try:
        exact(old.base.state_dict(),new.base.state_dict(),'initial_base')
        for i in range(20):
            atomic(out/'progress.json',dict(status='continuous_equivalence_audit',update=i))
            ix=order[i*256:(i+1)*256]
            cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state()
            opt0.schedule(i/steps);update(old,opt0,data,ix,128,a.endpoint,check=i==0)
            cpu_after=torch.get_rng_state();cuda_after=torch.cuda.get_rng_state()
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
            opt1.schedule(i/steps);update(new,opt1,data,ix,128,a.endpoint,check=i==0)
            exact(cpu_after,torch.get_rng_state(),'cpu_rng');exact(cuda_after,torch.cuda.get_rng_state(),'cuda_rng')
            exact(old.base.state_dict(),new.base.state_dict(),'base_parameters')
            exact({k:v.grad for k,v in old.base.named_parameters()},{k:v.grad for k,v in new.base.named_parameters()},'base_gradients')
            exact(opt0.base.state_dict(),opt1.base.state_dict(),'base_optimizer')
            exact(opt0.base.values(),opt1.base.values(),'optimizer_hyper_values');updates=i+1
        del opt0,opt1
        atomic(out/'progress.json',dict(status='archived_checkpoint_replay'))
        for epoch in (1,5,8,10,12,15,18,20):
            f=oldpath/f'epoch_{epoch:03d}_model.pt';diag=read(oldpath/f'epoch_{epoch:03d}_diagnostics.json')
            assert sha(f)==diag['checkpoint_sha256']
            gradient=diag['gradient']['groups']['loss_paths']
            assert gradient['auxiliary_base_gradient_absent'] and gradient['main_fusion_gradient_absent']
            ck=torch.load(f,map_location='cpu',weights_only=False)
            old.load_state_dict(ck['model']);new.base.load_state_dict({k[5:]:v for k,v in ck['model'].items() if k.startswith('base.')});del ck
            old.eval();new.eval();x,_=data.batch('dev',np.arange(128))
            with torch.no_grad():exact(old.forward_views(x)[0],new.forward_views(x)[0],'early_z0')
            early+=1
        ck=torch.load(oldpath/'best.pt',map_location='cpu',weights_only=False)
        assert ck['identity']==manifest and ck['state']['epoch']==completion['best_epoch']
        new.base.load_state_dict({k[5:]:v for k,v in ck['model'].items() if k.startswith('base.')});del ck
        selected=dict(np.load(oldpath/'predictions_dev.npz'))
        assert np.array_equal(selected['ids'],data.targets['dev']['ids'])
        assert np.array_equal(selected['labels'],data.targets['dev']['y'][:,a.endpoint:a.endpoint+1])
        predicted=predict(new,data,256)
        replay_drift=float(np.max(np.abs(predicted-selected['probs'])))
        if not np.array_equal(predicted,selected['probs']):raise NotEquivalent('best prediction replay is not bitwise identical')
    except NotEquivalent as exc:
        equivalent=False;reason=str(exc)
    result=dict(complete=True,equivalent=equivalent,continuous_exact=updates==20,updates=updates,
        early_snapshots_verified=early,best_probability_drift=replay_drift,reason=reason,pilot=True,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),seed=a.seed,endpoint=a.endpoint,
        old_source=str(oldpath),old_initial_weights_sha256=manifest['initial_weights_sha256'],
        old_checkpoint_sha256=oldrow['checkpoint_sha256'],old_predictions_sha256=oldrow['predictions_sha256'],
        new_initialization_is_fresh=True,diagnostic_weights_discarded=True,formal_training_started=False,
        test_loaded=False,patient_data_exported=False,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        seconds=time.time()-started,finished_at=time.time())
    atomic(out/'completion.json',result);atomic(out/'progress.json',dict(status='complete',equivalent=equivalent))
    print(json.dumps(result))

if __name__=='__main__':
    q=argparse.ArgumentParser()
    for n in ('project','output','preparation'):q.add_argument('--'+n,type=Path,required=True)
    q.add_argument('--seed',type=int,required=True);q.add_argument('--endpoint',type=int,required=True)
    q.add_argument('--expected-gib',type=float,required=True)
    audit(q.parse_args())
