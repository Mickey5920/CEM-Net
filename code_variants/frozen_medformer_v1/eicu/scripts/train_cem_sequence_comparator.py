"""Five-seed fixed recipes, bounded probes and checkpoint-locked evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import resource
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from scripts.cem_sequence_comparator_common import Data, expected, recipe, read, save, sha, SEEDS, VARIANTS
from models.cem_sequence_comparators import make_model, optimizer
from scripts.train_tracehr_unified import seed_all, rng_state, restore_rng
from scripts.mimiciii_ema_runtime import digest
from scripts.tracehr_oof_common import all_scores


def batches(data,split,order,size):
    # One bounded producer, at most two CPU batches, shared read-only memmaps.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=None
        for start in range(0,len(order),size):
            ids=order[start:start+size]
            if pending is None:pending=pool.submit(data.cpu_batch,split,ids)
            batch=pending.result()
            nxt=order[start+size:start+2*size]
            pending=pool.submit(data.cpu_batch,split,nxt) if len(nxt) else None
            yield ids,batch


def step(m,opt,data,ix,endpoint,spec,cpu=None):
    m.train();opt.zero_grad(set_to_none=True)
    # FT input copied from mmap once per effective batch, rather than per sample.
    if cpu is None:cpu=data.cpu_batch('train',ix)
    total=torch.zeros((),device='cuda')
    for start in range(0,len(ix),spec['micro']):
        subset=ix[start:start+spec['micro']]
        if m.variant=='FTTransformer':batch=(cpu[0][start:start+len(subset)].to('cuda'),)
        else:
            assert len(ix)<=spec['micro'],'BatchNorm physical batch must not be silently split'
            batch=tuple(x.to('cuda') for x in cpu)
        z=m(*batch);y=torch.as_tensor(data.targets['train']['y'][subset,endpoint],device='cuda',dtype=torch.float32)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)*len(subset)/len(ix)
        loss.backward();total+=loss.detach()
    if spec.get('clip'):torch.nn.utils.clip_grad_norm_(m.parameters(),spec['clip'],error_if_nonfinite=True)
    if not bool(torch.isfinite(total)):raise FloatingPointError('Non-finite training loss')
    opt.step();return float(total)


@torch.no_grad()
def predict(m,data,split,size):
    m.eval();values=[]
    for _,cpu in batches(data,split,np.arange(len(data.targets[split]['y'])),size):
        z=m(*(x.to('cuda') for x in cpu));assert torch.isfinite(z).all()
        values.append(z.sigmoid().cpu().numpy())
    return np.concatenate(values)


def initialize(a,out,data):
    m=make_model(a.variant,a.seed,data)
    initial=digest(m.state_dict())
    # Independent same-seed reconstruction, before any training data update.
    rebuilt=make_model(a.variant,a.seed,data);assert digest(rebuilt.state_dict())==initial;del rebuilt
    identity=dict(database=a.database,variant=a.variant,seed=a.seed,endpoint=a.endpoint,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),protocol=recipe(a.variant),
        initial_weights_sha256=initial,parameters=sum(p.numel() for p in m.parameters()),
        shared_hourly_cache_sha256={s:sha(data.folder/(s+'_completion.json')) for s in ('train','dev')},
        source_normalization_unchanged=True,clinical_adapter='trainable 128D event sum + 32 lab values/masks/gaps' if a.variant!='FTTransformer' else None)
    save(out/'initialization_observation.json',dict(**identity,optimizer_updates=0,learned_checkpoint_loaded=False,same_seed_rebuild_exact=True))
    save(out/'run_manifest.json',identity);m=m.cuda();seed_all(a.seed)
    return m,identity


def resource_summary():
    return dict(peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)


def probe(a,out,data):
    m,identity=initialize(a,out,data);spec=identity['protocol'];opt=optimizer(m,spec);rng=np.random.default_rng(a.seed)
    initial={n:p.detach().cpu().clone() for n,p in m.named_parameters()}
    fixed={n:v.detach().cpu().clone() for n,v in m.named_buffers() if 'handcrafted' in n}
    times=[];tick=time.time();n=len(data.targets['train']['y'])
    for i in range(20):
        ix=rng.permutation(n)[:spec['batch']];begin=time.perf_counter()
        loss=step(m,opt,data,ix,a.endpoint,spec);torch.cuda.synchronize();times.append(time.perf_counter()-begin)
        for name,p in m.named_parameters():
            assert p.grad is not None or name.startswith(m.allowed_unconnected()),name
            if p.grad is not None:assert torch.isfinite(p.grad).all(),name
        save(out/'progress.json',dict(status='training_probe',updates=i+1,loss=loss,last_step_seconds=times[-1]))
    moved={name:float((p.detach().cpu()-initial[name]).abs().max()) for name,p in m.named_parameters()};del initial
    unchanged=[n for n,v in moved.items() if v==0]
    assert all(n.startswith(m.allowed_unconnected()) for n in unchanged),unchanged
    for name,value in fixed.items():assert torch.equal(dict(m.named_buffers())[name].cpu(),value)
    save(out/'gradient_check.json',dict(complete=True,relative_update=moved,disconnected_author_classification_parameters=unchanged,
        fixed_filter_buffers_unchanged=True))
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng)),out/'representative_checkpoint.pt')
    ix=rng.permutation(n)[:spec['batch']];step(m,opt,data,ix,a.endpoint,spec)
    reference=digest(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng)))
    ck=torch.load(out/'representative_checkpoint.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);del ck
    ix2=rng.permutation(n)[:spec['batch']];assert np.array_equal(ix,ix2);step(m,opt,data,ix2,a.endpoint,spec)
    assert digest(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng)))==reference
    before=digest(m.state_dict());m.eval();ev=time.perf_counter()
    with torch.no_grad():
        size=min(spec['eval_batch'],16);ix=np.arange(size)
        p=m(*data.batch('dev',ix,'cuda'));q=torch.cat([m(*data.batch('dev',[i],'cuda')) for i in ix])
        torch.testing.assert_close(p,q,atol=3e-5,rtol=3e-5)
    evalseconds=time.perf_counter()-ev
    assert before==digest(m.state_dict())
    save(out/'path_audit.json',dict(complete=True,continuous_updates=20,resume_exact=True,
        effective_batch=spec['batch'],physical_batch=spec['micro'],evaluation_preserves_state=True,
        fixed_filter_buffers_unchanged=True,median_step_seconds=float(np.median(times[1:])),
        estimated_epoch_training_seconds=float(np.median(times[1:])*np.ceil(n/spec['batch'])),
        small_evaluation_seconds=evalseconds,checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size,
        test_loaded=False,disconnected_author_classification_parameters=unchanged))
    save(out/'completion.json',dict(complete=True,pilot=True,identity=identity,from_scratch=True,test_loaded=False,
        checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size,seconds=time.time()-tick,**resource_summary()))
    save(out/'progress.json',dict(status='complete'))


def train(a,out,data):
    m,identity=initialize(a,out,data);spec=identity['protocol'];ref=read(a.pilot_record/'completion.json')['identity']
    for key in ('protocol','variant','endpoint'):assert ref[key]==identity[key]
    if ref['source_manifest_sha256']!=identity['source_manifest_sha256']:
        assert a.variant!='FTTransformer' and ref['source_manifest_sha256']==expected()['preflight_parent_manifest']
        assert read(out.parent.parent.parent/'verification.json')['unchanged_CNN_preflights_reused']
    if a.seed==48:assert ref['initial_weights_sha256']==identity['initial_weights_sha256']
    save(out/'pilot_identity_check.json',dict(match=True,fresh_initialization=True,pilot_weights_loaded=False))
    opt=optimizer(m,spec);rng=np.random.default_rng(a.seed);tick=time.time();best=-1.;stale=0;history=[];updates=0
    # Keras ReduceLROnPlateau uses wait >= patience and absolute min_delta.
    lrbest=float('inf');lrwait=0
    save(out/'training_started.json',dict(started_at=tick,actual_optimizer_updates=0))
    for epoch in range(1,spec['epochs']+1):
        begin=time.time();order=rng.permutation(len(data.targets['train']['y']));total=0.
        for ix,cpu in batches(data,'train',order,spec['batch']):
            loss=step(m,opt,data,ix,a.endpoint,spec,cpu);total+=loss*len(ix);updates+=1
            if updates%30==1:save(out/'progress.json',dict(status='training',epoch=epoch,optimizer_updates=updates,best_ap=best))
        trainloss=total/len(order);pred=predict(m,data,'dev',spec['eval_batch'])
        ap=float(average_precision_score(data.targets['dev']['y'][:,a.endpoint],pred))
        history.append(dict(epoch=epoch,dev_ap=ap,loss=trainloss,lr=opt.param_groups[0]['lr'],seconds=time.time()-begin));save(out/'history.json',history)
        if 'reduce_loss' in spec:
            lr=spec['reduce_loss']
            if trainloss<lrbest-lr['min_delta']:lrbest=trainloss;lrwait=0
            else:
                lrwait+=1
                if lrwait>=lr['patience']:
                    for group in opt.param_groups:group['lr']=max(group['lr']*lr['factor'],lr['min_lr'])
                    lrwait=0
        checkpoint=dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),epoch=epoch,
            identity=identity,lrbest=lrbest if np.isfinite(lrbest) else None,lrwait=lrwait,optimizer_updates=updates)
        if ap>best+spec['min_delta']:
            best=ap;best_epoch=epoch;selected=pred.copy();stale=0;torch.save(checkpoint,out/'best.pt')
        else:stale+=1
        checkpoint.update(best=best,best_epoch=best_epoch,stale=stale)
        torch.save(checkpoint,out/'last.pt')
        save(out/'progress.json',dict(status='epoch_complete',epoch=epoch,best_ap=best,best_epoch=best_epoch,optimizer_updates=updates))
        if spec['patience'] is not None and stale>=spec['patience']:break
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);del ck
    replay=predict(m,data,'dev',spec['eval_batch']);drift=float(np.max(np.abs(replay-selected)));assert drift<=1e-5
    labels=data.targets['dev']['y'][:,a.endpoint:a.endpoint+1];metrics=all_scores(labels,replay[:,None])
    np.savez_compressed(out/'predictions_dev.npz',ids=data.targets['dev']['ids'],labels=labels,probs=replay[:,None])
    save(out/'completion.json',dict(complete=True,pilot=False,identity=identity,**identity,from_scratch=True,test_loaded=False,
        epochs=len(history),best_epoch=best_epoch,dev_ap=metrics['mean_ap'],metrics=metrics,
        checkpoint_file='best.pt',checkpoint_sha256=sha(out/'best.pt'),predictions_sha256=sha(out/'predictions_dev.npz'),
        checkpoint_replay=dict(complete=True,max_probability_drift=drift),seconds=time.time()-tick,optimizer_updates=updates,**resource_summary()))
    save(out/'progress.json',dict(status='complete',epochs=len(history),best_ap=best))


def evaluate(a,out,data):
    fit=a.fit_output;c=read(fit/'completion.json');identity=read(fit/'run_manifest.json');lock=read(fit.parent.parent/'fixed-checkpoint-lock.json')
    assert lock['complete'] and len(lock['selections'])==expected()['formal_fits']
    assert sha(fit/'best.pt')==c['checkpoint_sha256']==lock['selections'][fit.name]['checkpoint_sha256']
    torch.set_grad_enabled(False);m=make_model(a.variant,a.seed,data).cuda();ck=torch.load(fit/'best.pt',map_location='cpu',weights_only=False)
    assert ck['identity']==identity;m.load_state_dict(ck['model']);del ck
    saved=dict(np.load(fit/'predictions_dev.npz'));pred=predict(m,data,'dev',identity['protocol']['eval_batch'])
    np.testing.assert_array_equal(saved['ids'],data.targets['dev']['ids']);np.testing.assert_array_equal(saved['labels'][:,0],data.targets['dev']['y'][:,a.endpoint])
    drift=float(np.max(np.abs(pred-saved['probs'][:,0])));assert drift<=1e-5
    data.test();before=digest(m.state_dict());tick=time.time();pred=predict(m,data,'test',identity['protocol']['eval_batch']);elapsed=time.time()-tick
    assert digest(m.state_dict())==before and sha(fit/'best.pt')==c['checkpoint_sha256']
    labels=data.targets['test']['y'][:,a.endpoint:a.endpoint+1];metrics=all_scores(labels,pred[:,None])
    np.savez_compressed(out/'predictions_test.npz',ids=data.targets['test']['ids'],labels=labels,probs=pred[:,None]);save(out/'run_manifest.json',identity)
    save(out/'completion.json',dict(complete=True,**identity,split='test',optimizer_updates=0,parameters_updated=False,
        fixed_development_checkpoint=True,model_sha256=c['checkpoint_sha256'],predictions_sha256=sha(out/'predictions_test.npz'),
        metrics=metrics,dev_replay_max_drift=drift,inference_seconds=elapsed,rows=len(pred),test_loaded=True,**resource_summary()))
    save(out/'progress.json',dict(status='complete',split='test'))


def main():
    p=argparse.ArgumentParser()
    for n in ('project','output','pilot-record','fit-output'):p.add_argument('--'+n,type=Path,required=n in ('project','output'))
    p.add_argument('--database',required=True);p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--endpoint',type=int,required=True);p.add_argument('--action',choices=('probe','train','evaluate'),required=True);p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();assert a.database==os.environ['TRACEHR_DATABASE'];assert a.seed in SEEDS and a.endpoint in (0,1)
    out=a.output.resolve();a.project=a.project.resolve(strict=True);assert out.is_relative_to(a.project) and not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30;out.mkdir(parents=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(out/'storage.json',dict(project=str(a.project),code=str(ROOT),output=str(out),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    save(out/'progress.json',dict(status='loading_verified_inputs'));data=Data(a.project,a.variant)
    save(out/'progress.json',dict(status='randomly_initializing_native'));globals()[a.action](a,out,data)


if __name__=='__main__':main()
