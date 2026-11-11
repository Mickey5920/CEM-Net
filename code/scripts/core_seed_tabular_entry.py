"""Fresh comparator fits, bounded restoration checks, and locked test inference."""
import argparse,gc,math,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from scripts.cem_additional_common import expected,protocol,make_data,read,sha,atomic
from models.cem_additional_baselines import make_model
from models.tracehr_unified_next import parameter_hash
from scripts.train_tracehr_unified import seed_all,rng_state,restore_rng
from scripts.mimiciii_ema_runtime import digest
from scripts.tracehr_oof_common import all_scores

def logits(model,batch):return model(*batch).squeeze(-1)

@torch.no_grad()
def predict(model,data,split,micro):
    model.eval();parts=[]
    for start in range(0,len(data.targets[split]['y']),micro):
        z=logits(model,data.batch(split,np.arange(start,min(start+micro,len(data.targets[split]['y']))),'cuda'))
        p=z.sigmoid().mean(1) if z.ndim==2 else z.sigmoid()
        parts.append(p.cpu().numpy())
    return np.concatenate(parts)

def step(model,opt,data,ix,endpoint,micro):
    model.train();opt.zero_grad(set_to_none=True);total=0.
    for start in range(0,len(ix),micro):
        sub=ix[start:start+micro];z=logits(model,data.batch('train',sub,'cuda'))
        y=torch.as_tensor(data.targets['train']['y'][sub,endpoint],device='cuda',dtype=torch.float32)
        if z.ndim==2:y=y[:,None].expand_as(z)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)*len(sub)/len(ix)
        assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step();return total

def optimizer(model,spec):
    if spec['optimizer']=='Adam':return torch.optim.Adam(model.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay'])
    return torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay'])

def initialize(a,out,data):
    m=make_model(a.variant,a.seed,data)
    identity=dict(database=a.database,variant=a.variant,seed=a.seed,endpoint=a.endpoint,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),protocol=protocol(a.variant),
        initial_weights_sha256=parameter_hash(m),parameters=sum(p.numel() for p in m.parameters()),
        input_preparation_sha256=sha(Path(os.environ['TRACEHR_PROJECT'])/expected()['baseline_preparation']/'completion.json'))
    atomic(out/'initialization_observation.json',dict(**identity,optimizer_updates=0,learned_checkpoint_loaded=False))
    atomic(out/'run_manifest.json',identity)
    m=m.cuda();seed_all(a.seed);return m,identity

def probe(a,out,data):
    m,identity=initialize(a,out,data);spec=identity['protocol'];opt=optimizer(m,spec);rng=np.random.default_rng(a.seed)
    initial={n:p.detach().cpu().clone() for n,p in m.named_parameters()};tick=time.time();times=[]
    for i in range(20):
        ix=rng.permutation(len(data.targets['train']['y']))[:spec['batch']];begin=time.perf_counter()
        loss=step(m,opt,data,ix,a.endpoint,spec['micro']);torch.cuda.synchronize();times.append(time.perf_counter()-begin)
        atomic(out/'progress.json',dict(status='training_probe',updates=i+1,loss=loss))
    moved={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in m.named_parameters()};del initial
    unchanged=[n for n,v in moved.items() if v==0]
    allowed=('psi.','context.','context_key.','source_encoder.0.','source_encoder.2.weight') if a.variant=='SeFT' else ('att.linears.1.bias',) if a.variant=='mTAND_Enc' else ()
    assert all(n.startswith(allowed) for n in unchanged),unchanged
    atomic(out/'gradient_check.json',dict(all_connected=True,relative_update=moved,analytically_cancelled_parameters=unchanged))
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng)),out/'representative_checkpoint.pt')
    ix=rng.permutation(len(data.targets['train']['y']))[:spec['batch']];step(m,opt,data,ix,a.endpoint,spec['micro'])
    reference=digest(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng)))
    ck=torch.load(out/'representative_checkpoint.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);del ck
    ix2=rng.permutation(len(data.targets['train']['y']))[:spec['batch']];assert np.array_equal(ix,ix2)
    step(m,opt,data,ix2,a.endpoint,spec['micro'])
    assert digest(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng)))==reference
    before=digest(m.state_dict());m.eval()
    with torch.no_grad():
        ix=np.arange(16);z=logits(m,data.batch('dev',ix,'cuda'))
        zz=torch.cat([logits(m,data.batch('dev',ix[i:i+8],'cuda')) for i in (0,8)])
        torch.testing.assert_close(z,zz,atol=2e-5,rtol=2e-5)
    assert before==digest(m.state_dict())
    atomic(out/'path_audit.json',dict(complete=True,continuous_updates=20,resume_exact=True,
        all_parameters_updated=not unchanged,analytically_cancelled_parameters=unchanged,evaluation_preserves_state=True,median_step_seconds=float(np.median(times[1:])),test_loaded=False))
    atomic(out/'completion.json',dict(complete=True,pilot=True,identity=identity,from_scratch=True,test_loaded=False,
        checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.time()-tick))
    atomic(out/'progress.json',dict(status='complete'))

def train(a,out,data):
    m,identity=initialize(a,out,data);spec=identity['protocol'];ref=read(a.pilot_record/'completion.json')['identity']
    assert ref['protocol']==identity['protocol'] and ref['source_manifest_sha256']==identity['source_manifest_sha256']
    assert ref['variant']==identity['variant'] and ref['endpoint']==identity['endpoint']
    if a.seed==51:assert ref['initial_weights_sha256']==identity['initial_weights_sha256']
    atomic(out/'pilot_identity_check.json',dict(match=True,fresh_initialization=True,pilot_weights_loaded=False))
    opt=optimizer(m,spec);rng=np.random.default_rng(a.seed);tick=time.time();best=-1.;stale=0;history=[];updates=0
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=15,cooldown=15,min_lr=1e-5) if spec.get('reduce_on_plateau') else None
    atomic(out/'training_started.json',dict(started_at=tick,actual_optimizer_updates=0))
    for epoch in range(1,spec['epochs']+1):
        begin=time.time();order=rng.permutation(len(data.targets['train']['y']));total=0.
        for start in range(0,len(order),spec['batch']):
            ix=order[start:start+spec['batch']];loss=step(m,opt,data,ix,a.endpoint,spec['micro']);total+=loss*len(ix);updates+=1
            if start==0 or updates%30==0:atomic(out/'progress.json',dict(status='training',epoch=epoch,step=start//spec['batch']+1,optimizer_updates=updates,best_ap=best))
        pred=predict(m,data,'dev',spec['micro']);ap=float(average_precision_score(data.targets['dev']['y'][:,a.endpoint],pred))
        history.append(dict(epoch=epoch,dev_ap=ap,loss=total/len(order),lr=opt.param_groups[0]['lr'],seconds=time.time()-begin));atomic(out/'history.json',history)
        if scheduler:scheduler.step(ap)
        if ap>best+1e-9:
            best=ap;best_epoch=epoch;selected=pred.copy();stale=0
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,rng=rng_state(rng),epoch=epoch,identity=identity),out/'best.pt')
        else:stale+=1
        atomic(out/'progress.json',dict(status='epoch_complete',epoch=epoch,best_ap=best,best_epoch=best_epoch,optimizer_updates=updates))
        if stale>=spec['patience']:break
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);del ck
    replay=predict(m,data,'dev',spec['micro']);drift=float(np.max(np.abs(replay-selected)));assert drift<=1e-5
    labels=data.targets['dev']['y'][:,a.endpoint:a.endpoint+1];metrics=all_scores(labels,replay[:,None])
    np.savez_compressed(out/'predictions_dev.npz',ids=data.targets['dev']['ids'],labels=labels,probs=replay[:,None])
    atomic(out/'completion.json',dict(complete=True,pilot=False,identity=identity,**identity,from_scratch=True,test_loaded=False,
        frozen_learnable_parameters=0,epochs=len(history),best_epoch=best_epoch,dev_ap=metrics['mean_ap'],metrics=metrics,
        checkpoint_file='best.pt',checkpoint_sha256=sha(out/'best.pt'),predictions_sha256=sha(out/'predictions_dev.npz'),
        checkpoint_replay=dict(complete=True,max_probability_drift=drift),seconds=time.time()-tick,optimizer_updates=updates,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    atomic(out/'progress.json',dict(status='complete',epochs=len(history),best_ap=best))

def evaluate(a,out,data):
    fit=a.fit_output;c=read(fit/'completion.json');identity=read(fit/'run_manifest.json');lock=read(fit.parent.parent/'fixed-checkpoint-lock.json')
    assert lock['complete'] and len(lock['selections'])==expected()['formal_fits']
    assert sha(fit/'best.pt')==c['checkpoint_sha256']==lock['selections'][fit.name]['checkpoint_sha256']
    torch.set_grad_enabled(False);m=make_model(a.variant,a.seed,data).cuda();ck=torch.load(fit/'best.pt',map_location='cpu',weights_only=False)
    assert ck['identity']==identity;m.load_state_dict(ck['model']);del ck
    saved=dict(np.load(fit/'predictions_dev.npz'));pred=predict(m,data,'dev',identity['protocol']['micro'])
    np.testing.assert_array_equal(saved['ids'],data.targets['dev']['ids']);np.testing.assert_array_equal(saved['labels'][:,0],data.targets['dev']['y'][:,a.endpoint])
    drift=float(np.max(np.abs(pred-saved['probs'][:,0])));assert drift<=1e-5
    data.test();before=digest(m.state_dict());tick=time.time();pred=predict(m,data,'test',identity['protocol']['micro']);elapsed=time.time()-tick
    assert digest(m.state_dict())==before and sha(fit/'best.pt')==c['checkpoint_sha256']
    labels=data.targets['test']['y'][:,a.endpoint:a.endpoint+1];metrics=all_scores(labels,pred[:,None])
    np.savez_compressed(out/'predictions_test.npz',ids=data.targets['test']['ids'],labels=labels,probs=pred[:,None]);atomic(out/'run_manifest.json',identity)
    atomic(out/'completion.json',dict(complete=True,**identity,split='test',optimizer_updates=0,parameters_updated=False,
        fixed_development_checkpoint=True,model_sha256=c['checkpoint_sha256'],predictions_sha256=sha(out/'predictions_test.npz'),
        metrics=metrics,dev_replay_max_drift=drift,inference_seconds=elapsed,rows=len(pred),test_loaded=True))
    atomic(out/'progress.json',dict(status='complete',split='test'))

def main():
    p=argparse.ArgumentParser()
    for n in ('project','output','pilot-record','fit-output'):p.add_argument('--'+n,type=Path,required=n in ('project','output'))
    p.add_argument('--database',required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--endpoint',type=int,required=True);p.add_argument('--action',choices=('probe','train','evaluate'),required=True);p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();assert a.database==os.environ['TRACEHR_DATABASE'];assert a.seed in (51,52) and a.endpoint in (0,1)
    out=a.output.resolve();a.project=a.project.resolve(strict=True);assert out.is_relative_to(a.project) and not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30;out.mkdir(parents=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    atomic(out/'storage.json',dict(project=str(a.project),code=str(ROOT),output=str(out),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    atomic(out/'progress.json',dict(status='loading_verified_inputs'));data=make_data(a.project)
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    globals()[a.action](a,out,data)

if __name__=='__main__':main()
