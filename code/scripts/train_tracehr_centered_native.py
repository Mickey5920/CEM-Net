"""Bounded pilots and independent endpoint training, always fresh initialization."""
import argparse
import gc
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
deps = Path(os.environ['TRACEHR_TABULAR_DEPS'])
sys.path.insert(0, str(deps))
for wheel in deps.glob('*.whl'):
    sys.path.insert(0, str(wheel))
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from models.tracehr_unified_next import parameter_hash
from scripts.tracehr_centered_native_common import VARIANTS, Data, prepare, model_for, protocol, stopping
from scripts.tracehr_backbone_refine_runtime import Optimizer
from scripts.tracehr_centered_diagnostics import diagnostics
from scripts.tracehr_multiview_common import atomic, read, sha, checked_output
from scripts.tracehr_oof_common import all_scores
from scripts.train_tracehr_unified import seed_all, rng_state, restore_rng


def gradients(model):
    result = {}
    for name, p in model.named_parameters():
        assert p.requires_grad and p.grad is not None, ('Frozen/disconnected', name)
        assert torch.isfinite(p.grad).all(), name
        key = ('member' if '.fast_' in name or name in ('input_members', 'numeric_members') else
               'numerical' if 'numerical' in name or '/plr' in str(getattr(getattr(p, 'context', None), 'scope', '')) else 'backbone')
        r = result.setdefault(key, dict(tensors=0, norm_squared=0., parameter_norm_squared=0.))
        r['tensors'] += 1
        r['norm_squared'] += float(p.grad.detach().square().sum())
        r['parameter_norm_squared'] += float(p.detach().square().sum())
    assert all(r['norm_squared'] > 0 for r in result.values()), result
    return result


def update(model, opt, data, ix, micro, endpoint, check=False):
    model.train()
    opt.zero_grad()
    total = torch.zeros((), device='cuda')
    for start in range(0, len(ix), micro):
        part = ix[start:start+micro]
        x, labels = data.batch('train', part)
        loss = model.loss(model(x), labels[:, endpoint])
        assert torch.isfinite(loss)
        ratio = len(part)/len(ix)
        (loss*ratio).backward()
        total += loss.detach()*ratio
    report = gradients(model) if check else None
    before = {n:p.detach().clone() for n,p in model.named_parameters()} if check else None
    opt.step()
    if check:
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), name
        report['relative_update'] = {
            name: float((p.detach()-before[name]).norm()/(before[name].norm()+1e-12))
            for name,p in model.named_parameters()}
    return total, report


@torch.no_grad()
def predict(model, data, micro, indices=None):
    model.eval()
    indices = np.arange(len(data.targets['dev']['y'])) if indices is None else indices
    values, members = [], []
    for start in range(0, len(indices), micro):
        x, _ = data.batch('dev', indices[start:start+micro])
        logits = model(x)
        member_prob = logits.softmax(-1)[..., 1:2] if logits.shape[-1]==2 else logits.sigmoid()
        values.append(member_prob.mean(1).cpu().numpy())
        members.append(member_prob[..., 0].cpu().numpy())
    p = np.concatenate(values)
    assert p.shape == (len(indices), 1) and np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1))
    model._diagnostic_member_probs = np.concatenate(members)
    return p


def save(path, model, opt, rng, state, identity):
    tmp = path.with_name(path.name+'.writing')
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(rng),
                    state=state, identity=identity), tmp)
    os.replace(tmp, path)


def probe(a, out, data, model, identity, spec):
    started = time.time()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    n, batch = len(data.targets['train']['y']), spec['effective_batch']
    rng = np.random.default_rng(a.seed)
    orders = [rng.permutation(n)[:batch] for _ in range(4)]
    rows, checks, opt = [], {}, None
    for micro in (spec['training_microbatch'],):
        model.load_state_dict(initial)
        seed_all(a.seed+10000)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            opt = Optimizer(model, spec, a.lr_multiplier)
            opt.schedule(1./math.ceil(n/batch))
            times = []
            for i, ix in enumerate(orders):
                atomic(out/'progress.json', dict(status='training_probe', microbatch=micro, update=i))
                torch.cuda.synchronize(); tick=time.perf_counter()
                loss, report = update(model,opt,data,ix,micro,a.endpoint,check=i==0)
                torch.cuda.synchronize(); times.append(time.perf_counter()-tick)
                if report is not None:checks[str(micro)] = report
            rows.append(dict(microbatch=micro, complete=True, median_seconds=float(np.median(times[1:])),
                             peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, step_seconds=times))
        except torch.OutOfMemoryError as exc:
            rows.append(dict(microbatch=micro, complete=False, error='CUDA OOM in bounded pilot', detail=str(exc)[:500]))
        finally:
            del opt
            opt = None
            model.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
    good = [r for r in rows if r['complete']]
    assert good, ('No train microbatch fits', rows)
    fastest = min(r['median_seconds'] for r in good)
    chosen = next(r for r in good if r['median_seconds'] <= fastest*1.1)
    micro = chosen['microbatch']
    model.load_state_dict(initial)
    ix = np.arange(min(512, len(data.targets['dev']['y'])))
    evaluation, reference = [], None
    for ev in ((128, 256) if spec['native'] else (data.record['n12_eval'][a.endpoint],)):
        tick = time.perf_counter()
        p = predict(model, data, ev, ix)
        drift = 0. if reference is None else float(np.max(np.abs(reference-p)))
        assert drift <= 1e-5
        reference = p if reference is None else reference
        evaluation.append(dict(microbatch=ev, seconds=time.perf_counter()-tick, drift=drift))
    ev = spec['evaluation_microbatch']
    seed_all(a.seed+10000)
    opt = Optimizer(model, spec, a.lr_multiplier)
    opt.schedule(1./math.ceil(n/batch))
    update(model,opt,data,orders[0],micro,a.endpoint)
    save(out/'representative_checkpoint.pt',model,opt,rng,dict(updates=1),identity)
    ck = torch.load(out/'representative_checkpoint.pt',map_location='cpu',weights_only=False)
    next_ix = rng.permutation(n)[:batch]
    opt.schedule(2./math.ceil(n/batch))
    update(model,opt,data,next_ix,micro,a.endpoint)
    expected = {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); restore_rng(ck['rng'],rng)
    replay_ix = rng.permutation(n)[:batch]
    assert np.array_equal(next_ix,replay_ix)
    opt.schedule(2./math.ceil(n/batch))
    update(model,opt,data,replay_ix,micro,a.endpoint)
    drift = max(float((p.detach().cpu()-expected[n]).abs().max()) for n,p in model.named_parameters())
    assert drift <= 1e-6, ('Optimizer/RNG restore mismatch', drift)
    result = dict(complete=True,pilot=True,from_scratch=True,test_loaded=False,frozen_learnable_parameters=0,
                  identity=identity,selected_microbatch=micro,evaluation_microbatch=ev,effective_batch=batch,
                  benchmarks=rows,evaluation_benchmarks=evaluation,gradient_checks=checks,
                  optimizer_resume_max_parameter_drift=drift,
                  peak_reserved_mib=max(torch.cuda.max_memory_reserved()/2**20,max(r['peak_reserved_mib'] for r in good)),
                  checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size,
                  seconds=time.time()-started,finished_at=time.time())
    atomic(out/'completion.json',result)
    atomic(out/'progress.json',dict(status='complete',seconds=result['seconds']))


def train(a,out,data,model,identity,spec):
    pilot=read(a.pilot_record/'completion.json')
    assert pilot['complete'] and pilot['pilot']
    for name in ('database','variant','endpoint','seed','input_manifest_sha256','source_manifest_sha256',
                 'initial_weights_sha256','common_weights_sha256'):
        assert pilot['identity'][name] == identity[name], ('Pilot identity mismatch',name)
    micro,ev=pilot['selected_microbatch'],pilot['evaluation_microbatch']
    atomic(out/'pilot_identity_check.json',dict(match=True,pilot=str(a.pilot_record),microbatch=micro,evaluation_microbatch=ev))
    seed_all(a.seed+10000)
    rng=np.random.default_rng(a.seed)
    opt=Optimizer(model,spec,a.lr_multiplier)
    atomic(out/'optimizer_initial.json',opt.values())
    n,batch=len(data.targets['train']['y']),spec['effective_batch']
    steps=math.ceil(n/batch)
    history=[]; best_ap=-1.; best_epoch=0; updates=0; first_epoch=1
    initial_parameters={n:p.detach().clone() for n,p in model.named_parameters()}
    if a.resume:
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert ck['identity']==identity
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng)
        state=ck['state'];history=state['history'];best_ap=state['best_ap'];best_epoch=state['best_epoch'];updates=state['updates']
        first_epoch=state['epoch']+1
        del ck
    if a.start_barrier:
        atomic(out/'progress.json',dict(status='waiting_coexistence_barrier'))
        deadline=time.time()+1200
        while not a.start_barrier.exists():
            assert time.time()<deadline,'Coexistence timeout'
            time.sleep(.5)
    atomic(out/'mechanism_initial.json',diagnostics(model,data,initial_parameters,a.seed))
    started=time.time()
    atomic(out/'training_started.json',dict(started_at=started,pid=os.getpid()))
    torch.cuda.reset_peak_memory_stats()
    limit=1 if a.action=='smoke' else spec['max_epochs']
    epoch=first_epoch
    while epoch<=limit:
        tick=time.time();order=rng.permutation(n)
        if a.action=='smoke':order=order[:12*batch]
        total=torch.zeros((),device='cuda')
        epoch_steps=math.ceil(len(order)/batch)
        for step,start in enumerate(range(0,len(order),batch),1):
            ix=order[start:start+batch]
            # Preserve the old residual warmup's (update+1)/steps convention;
            # native HP callbacks use progress before the current update.
            opt.schedule((updates+(0 if spec['native'] else 1))/steps)
            loss,report=update(model,opt,data,ix,micro,a.endpoint,check=step==1)
            total+=loss*len(ix);updates+=1
            if report is not None:atomic(out/'gradient_latest.json',dict(epoch=epoch,groups=report,optimizer=opt.values()))
            if step==1 or step%30==0 or step==epoch_steps:
                atomic(out/'progress.json',dict(status='training',epoch=epoch,step=step,steps=epoch_steps,
                      best_ap=best_ap,best_epoch=best_epoch,elapsed_seconds=time.time()-started))
        train_seconds=time.time()-tick
        ix=np.arange(min(1024,len(data.targets['dev']['y']))) if a.action=='smoke' else None
        atomic(out/'progress.json',dict(status='evaluating',epoch=epoch,best_ap=best_ap,best_epoch=best_epoch))
        val_tick=time.time();pred=predict(model,data,ev,ix)
        labels=data.targets['dev']['y'][:len(pred),a.endpoint:a.endpoint+1]
        metrics=all_scores(labels,pred)
        member_probs=model._diagnostic_member_probs
        member_ap=[float(average_precision_score(labels[:,0],member_probs[:,i])) for i in range(model.members)]
        corr=np.corrcoef((member_probs-labels).T) if model.members>1 else None
        diagnostic=dict(member_AP=member_ap,mean_member_AP=float(np.mean(member_ap)),
                        mean_probability_AP=metrics['mean_ap'],
                        average_probability_gain_vs_mean_member_AP=metrics['mean_ap']-float(np.mean(member_ap)),
                        residual_correlation_mean=float(corr[~np.eye(model.members,dtype=bool)].mean()) if corr is not None else None,
                        mean_probability_std_across_members=float(member_probs.std(1).mean()))
        row=dict(epoch=epoch,dev_ap=metrics['mean_ap'],metrics=metrics,train_loss=float(total)/len(order),
                 train_seconds=train_seconds,evaluation_seconds=time.time()-val_tick,optimizer=opt.values(),
                 member_diagnostics=diagnostic,mechanism_diagnostics=diagnostics(model,data,initial_parameters,a.seed))
        assert math.isfinite(row['dev_ap']) and math.isfinite(row['train_loss'])
        history.append(row)
        improved=row['dev_ap']>best_ap
        if improved:best_ap,best_epoch=row['dev_ap'],epoch
        state=dict(epoch=epoch,updates=updates,best_ap=best_ap,best_epoch=best_epoch,history=history)
        if improved:
            save(out/'best.pt',model,opt,rng,state,identity)
            np.savez_compressed(out/'predictions_dev.npz',ids=data.targets['dev']['ids'][:len(pred)],
                                labels=labels,probs=pred,epoch=epoch,endpoint=a.endpoint)
            atomic(out/'metrics.json',metrics)
        save(out/'latest.pt',model,opt,rng,state,identity)
        row['epoch_seconds_with_checkpoint']=time.time()-tick
        atomic(out/'history.json',history)
        stop=stopping(history,spec)
        atomic(out/'progress.json',dict(status='epoch_complete',epoch=epoch,dev_ap=row['dev_ap'],best_ap=best_ap,
              best_epoch=best_epoch,epoch_seconds=row['epoch_seconds_with_checkpoint'],stopping=stop,
              elapsed_seconds=time.time()-started,active_epoch_limit=limit))
        if a.action!='smoke' and stop['stop']:break
        if epoch==limit and a.action!='smoke' and best_epoch>limit-12 and limit<spec['extension_epochs']:
            limit=spec['extension_epochs']
            atomic(out/'extension.json',dict(reason='new maximum within final 12 epochs',best_epoch=best_epoch,
                   new_limit=limit,same_training_run=True,schedule_unchanged=True))
        epoch+=1
    del opt
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    assert ck['identity']==identity and ck['state']['epoch']==best_epoch
    model.load_state_dict(ck['model']);del ck
    selected=dict(np.load(out/'predictions_dev.npz'))
    assert int(selected['epoch'])==best_epoch and int(selected['endpoint'])==a.endpoint
    assert np.array_equal(selected['ids'],data.targets['dev']['ids'][:len(selected['probs'])])
    assert np.array_equal(selected['labels'],data.targets['dev']['y'][:len(selected['probs']),a.endpoint:a.endpoint+1])
    scores=all_scores(selected['labels'],selected['probs'])
    assert scores==read(out/'metrics.json') and abs(scores['mean_ap']-best_ap)<1e-12
    replay=predict(model,data,ev,np.arange(len(selected['probs'])))
    drift=float(np.max(np.abs(replay-selected['probs'])))
    assert drift<=1e-5,('Checkpoint replay mismatch',drift)
    result=dict(complete=True,pilot=a.action=='smoke',from_scratch=True,test_loaded=False,
                frozen_learnable_parameters=0,**identity,epochs=len(history),best_epoch=best_epoch,dev_ap=best_ap,
                metrics=scores,seconds=time.time()-started,finished_at=time.time(),stopping_status=stopping(history,spec),
                termination='smoke' if a.action=='smoke' else ('early_stopping' if stopping(history,spec)['stop'] else 'max_epochs'),
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,checkpoint_sha256=sha(out/'best.pt'),
                predictions_sha256=sha(out/'predictions_dev.npz'),
                checkpoint_replay=dict(complete=True,max_probability_drift=drift,
                    replay_ap=all_scores(selected['labels'],replay)['mean_ap']))
    atomic(out/'completion.json',result)
    atomic(out/'progress.json',dict(status='complete',epochs=len(history),best_epoch=best_epoch,best_ap=best_ap))


def main():
    p=argparse.ArgumentParser()
    for name in ('project','output','preparation','pilot-record','start-barrier'):
        p.add_argument('--'+name,type=Path,required=name in ('project','output'))
    p.add_argument('--database',choices=['mimic','eicu'],required=True)
    p.add_argument('--action',choices=['prepare','probe','smoke','train'],required=True)
    p.add_argument('--variant',choices=VARIANTS,default='E1')
    p.add_argument('--endpoint',type=int,choices=[0,1],default=0)
    p.add_argument('--selected-structure',choices=['D0','D1','D2','D3'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr-multiplier',type=float,default=1.)
    p.add_argument('--expected-gib',type=float,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.project=a.project.resolve(strict=True)
    out=checked_output(a.project,a.output)
    assert out.is_dir() if a.resume else not out.exists(),'Ambiguous output reuse'
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30
    for key in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        checked_output(a.project,os.environ[key]).mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=a.resume)
    atomic(out/'storage.json',dict(project=str(a.project),code=str(ROOT.resolve()),output=str(out),
          cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    if a.action=='prepare':prepare(a.project,a.database,out);return
    assert torch.cuda.is_available() and a.preparation
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=Data(a.project,a.database,a.preparation)
    spec=protocol(a.variant,a.selected_structure)
    atomic(out/'progress.json',dict(status='randomly_initializing_native' if spec['native'] else 'initializing'))
    model=model_for(data,a.preparation,a.variant,a.endpoint,a.seed,a.lr_multiplier,a.selected_structure,tmp_folder=out/'native_init_tmp')
    identity=dict(database=a.database,variant=a.variant,endpoint=a.endpoint,seed=a.seed,
                  lr_multiplier=a.lr_multiplier,protocol=spec,source_manifest_sha256=sha(ROOT/'source_manifest.json'),
                  input_manifest_sha256=sha(a.preparation/'completion.json'),selected_structure=spec['structure'],initial_weights_sha256=parameter_hash(model),
                  common_weights_sha256=parameter_hash(model,True),parameters=sum(p.numel() for p in model.parameters()),
                  input_dim=data.input_dim,members=model.members)
    assert all(p.requires_grad for p in model.parameters())
    if not a.resume:atomic(out/'run_manifest.json',identity)
    else:assert read(out/'run_manifest.json')==identity
    if spec['native']:atomic(out/'native_initialization.json',model.info)
    model=model.cuda();gc.collect()
    atomic(out/'execution_audit.json',dict(effective_batch=256,microbatch=128,precision='FP32',
          training_sampler='numpy.default_rng(seed).permutation; retain final incomplete batch',
          original_native_sampler='torch batched randperm; drop_last=True; 203 full batches for 52106 rows',
          current_batches_per_epoch=math.ceil(len(data.raw['train'])/256),
          optimizer_clock='native epoch_float before update; coslog4 reference 256',
          all_learnable_parameters_trainable=all(p.requires_grad for p in model.parameters()),
          initial_hash=identity['initial_weights_sha256']))
    if a.action=='probe':probe(a,out,data,model,identity,spec)
    else:
        assert a.pilot_record
        train(a,out,data,model,identity,spec)


if __name__=='__main__':main()
