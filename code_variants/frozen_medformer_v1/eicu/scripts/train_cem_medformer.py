"""Medformer-only fresh fitting and locked SWA checkpoint testing."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from scripts.cem_medformer_common import Data, expected, recipe, read, save, sha, SEEDS
from models.cem_medformer import make_model, optimizer
from scripts.train_cem_sequence_comparator import batches, resource_summary
from scripts.train_tracehr_unified import seed_all, rng_state, restore_rng
from scripts.mimiciii_ema_runtime import digest
from scripts.tracehr_oof_common import all_scores


def step(model, opt, data, indices, endpoint, cpu=None):
    model.train(); opt.zero_grad(set_to_none=True)
    if cpu is None: cpu = data.cpu_batch('train', indices)
    logits = model(*(x.to('cuda') for x in cpu))
    labels = torch.as_tensor(data.targets['train']['y'][indices, endpoint], device='cuda', dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), recipe()['clip'], error_if_nonfinite=True)
    opt.step()
    return float(loss.detach())


@torch.no_grad()
def predict(model, data, split):
    model.eval(); values = []
    for _, cpu in batches(data, split, np.arange(len(data.targets[split]['y'])), recipe()['eval_batch']):
        logits = model(*(x.to('cuda') for x in cpu)); assert torch.isfinite(logits).all()
        values.append(logits.softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(values)


def initialize(a, out, data):
    model = make_model(a.seed, data); initial = digest(model.state_dict())
    rebuilt = make_model(a.seed, data)
    assert digest(rebuilt.state_dict()) == initial
    del rebuilt
    identity = dict(database=a.database, variant='Medformer', seed=a.seed, endpoint=a.endpoint,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'), protocol=recipe(),
        initial_weights_sha256=initial, parameters=sum(p.numel() for p in model.parameters()),
        shared_hourly_cache_sha256={s:sha(data.folder/(s+'_completion.json')) for s in ('train','dev')},
        source_normalization_unchanged=True, clinical_adapter='trainable 128D event sum + 32 lab values/masks/gaps',
        selected_weights='epoch-end SWA; development AP only')
    save(out/'initialization_observation.json', dict(**identity, optimizer_updates=0,
        learned_checkpoint_loaded=False, same_seed_rebuild_exact=True))
    save(out/'run_manifest.json', identity)
    model = model.cuda(); seed_all(a.seed)
    return model, identity


def state(model, swa, opt, rng):
    return dict(raw_model=model.state_dict(), swa=swa.state_dict(), optimizer=opt.state_dict(), rng=rng_state(rng))


def probe(a, out, data):
    model, identity = initialize(a, out, data)
    opt = optimizer(model); swa = torch.optim.swa_utils.AveragedModel(model)
    rng = np.random.default_rng(a.seed); n = len(data.targets['train']['y'])
    initial = {k:p.detach().cpu().clone() for k,p in model.named_parameters()}
    times = []; tick = time.time(); average = None
    for i in range(20):
        indices = rng.permutation(n)[:128]; begin = time.perf_counter()
        loss = step(model, opt, data, indices, a.endpoint); torch.cuda.synchronize()
        times.append(time.perf_counter()-begin)
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if i in (9, 19):
            values = {k:p.detach().clone() for k,p in model.named_parameters()}
            average = values if average is None else {k:(average[k]+v)/2 for k,v in values.items()}
            swa.update_parameters(model)
        save(out/'progress.json', dict(status='training_probe', updates=i+1, loss=loss))
    for name,p in swa.module.named_parameters():
        torch.testing.assert_close(p, average[name], atol=2e-7, rtol=2e-6)
    assert int(swa.n_averaged) == 2
    moved = {k:float((p.detach().cpu()-initial[k]).abs().max()) for k,p in model.named_parameters()}
    assert all(v>0 for v in moved.values()), [k for k,v in moved.items() if v==0]
    del initial, average
    save(out/'gradient_check.json', dict(complete=True, relative_update=moved,
        all_trainable_parameters_updated=True, swa_arithmetic_average_verified=True))
    torch.save(state(model,swa,opt,rng), out/'representative_checkpoint.pt')
    indices = rng.permutation(n)[:128]; step(model,opt,data,indices,a.endpoint); swa.update_parameters(model)
    reference = digest(state(model,swa,opt,rng))
    checkpoint = torch.load(out/'representative_checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['raw_model']); swa.load_state_dict(checkpoint['swa'])
    opt.load_state_dict(checkpoint['optimizer']); restore_rng(checkpoint['rng'],rng); del checkpoint
    indices2 = rng.permutation(n)[:128]; assert np.array_equal(indices,indices2)
    step(model,opt,data,indices2,a.endpoint); swa.update_parameters(model)
    assert digest(state(model,swa,opt,rng)) == reference
    before = digest(swa.state_dict()); swa.eval()
    with torch.no_grad():
        ids = np.arange(8); p = swa(*data.batch('dev',ids,'cuda'))
        q = torch.cat([swa(*data.batch('dev',[i],'cuda')) for i in ids])
        torch.testing.assert_close(p,q,atol=3e-5,rtol=3e-5)
    assert before == digest(swa.state_dict())
    save(out/'path_audit.json', dict(complete=True, continuous_updates=20, resume_exact=True,
        resume_raw_optimizer_swa_rng_exact=True, swa_arithmetic_average_verified=True,
        effective_batch=128, physical_batch=128, evaluation_preserves_state=True, test_loaded=False,
        median_step_seconds=float(np.median(times[1:])),
        estimated_epoch_training_seconds=float(np.median(times[1:])*np.ceil(n/128)),
        checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size))
    save(out/'completion.json', dict(complete=True,pilot=True,identity=identity,from_scratch=True,test_loaded=False,
        checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size,seconds=time.time()-tick,**resource_summary()))
    save(out/'progress.json', dict(status='complete'))


def train(a, out, data):
    model, identity = initialize(a,out,data); spec = recipe()
    ref = read(a.pilot_record/'completion.json')['identity']
    for key in ('protocol','variant','endpoint','source_manifest_sha256'): assert ref[key]==identity[key]
    if a.seed==48: assert ref['initial_weights_sha256']==identity['initial_weights_sha256']
    save(out/'pilot_identity_check.json', dict(match=True,fresh_initialization=True,pilot_weights_loaded=False))
    opt=optimizer(model); swa=torch.optim.swa_utils.AveragedModel(model); rng=np.random.default_rng(a.seed)
    tick=time.time(); best=-1.; stale=0; history=[]; updates=0
    save(out/'training_started.json',dict(started_at=tick,actual_optimizer_updates=0))
    for epoch in range(1,spec['epochs']+1):
        begin=time.time(); order=rng.permutation(len(data.targets['train']['y'])); total=0.
        for indices,cpu in batches(data,'train',order,128):
            loss=step(model,opt,data,indices,a.endpoint,cpu);total+=loss*len(indices);updates+=1
            if updates%30==1: save(out/'progress.json',dict(status='training',epoch=epoch,optimizer_updates=updates,swa_updates=int(swa.n_averaged)))
        swa.update_parameters(model); assert int(swa.n_averaged)==epoch
        pred=predict(swa,data,'dev'); ap=float(average_precision_score(data.targets['dev']['y'][:,a.endpoint],pred))
        history.append(dict(epoch=epoch,dev_ap=ap,loss=total/len(order),lr=opt.param_groups[0]['lr'],
            swa_updates=int(swa.n_averaged),seconds=time.time()-begin));save(out/'history.json',history)
        checkpoint=dict(**state(model,swa,opt,rng),model=swa.module.state_dict(),identity=identity,
            epoch=epoch,optimizer_updates=updates,swa_updates=int(swa.n_averaged))
        if ap>best+spec['min_delta']:
            best=ap;best_epoch=epoch;selected=pred.copy();stale=0;torch.save(checkpoint,out/'best.pt')
        else: stale+=1
        checkpoint.update(best=best,best_epoch=best_epoch,stale=stale)
        torch.save(checkpoint,out/'last.pt')
        save(out/'progress.json',dict(status='epoch_complete',epoch=epoch,best_ap=best,best_epoch=best_epoch,optimizer_updates=updates,swa_updates=epoch))
        if stale>=spec['patience']: break
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model']);del checkpoint
    replay=predict(model,data,'dev');drift=float(np.max(np.abs(replay-selected)));assert drift<=1e-5
    labels=data.targets['dev']['y'][:,a.endpoint:a.endpoint+1];metrics=all_scores(labels,replay[:,None])
    np.savez_compressed(out/'predictions_dev.npz',ids=data.targets['dev']['ids'],labels=labels,probs=replay[:,None])
    save(out/'completion.json',dict(complete=True,pilot=False,identity=identity,**identity,from_scratch=True,test_loaded=False,
        epochs=len(history),best_epoch=best_epoch,dev_ap=metrics['mean_ap'],metrics=metrics,
        checkpoint_file='best.pt',checkpoint_sha256=sha(out/'best.pt'),predictions_sha256=sha(out/'predictions_dev.npz'),
        checkpoint_replay=dict(complete=True,max_probability_drift=drift),seconds=time.time()-tick,
        optimizer_updates=updates,swa_updates=len(history),**resource_summary()))
    save(out/'progress.json',dict(status='complete',epochs=len(history),best_ap=best))


def evaluate(a,out,data):
    fit=a.fit_output;c=read(fit/'completion.json');identity=read(fit/'run_manifest.json')
    lock=read(fit.parent.parent/'fixed-checkpoint-lock.json')
    assert lock['complete'] and len(lock['selections'])==expected()['formal_fits']==10
    assert sha(fit/'best.pt')==c['checkpoint_sha256']==lock['selections'][fit.name]['checkpoint_sha256']
    torch.set_grad_enabled(False);model=make_model(a.seed,data).cuda()
    checkpoint=torch.load(fit/'best.pt',map_location='cpu',weights_only=False)
    assert checkpoint['identity']==identity and checkpoint['swa_updates']==c['best_epoch']
    model.load_state_dict(checkpoint['model']);del checkpoint
    saved=dict(np.load(fit/'predictions_dev.npz'));pred=predict(model,data,'dev')
    np.testing.assert_array_equal(saved['ids'],data.targets['dev']['ids'])
    np.testing.assert_array_equal(saved['labels'][:,0],data.targets['dev']['y'][:,a.endpoint])
    drift=float(np.max(np.abs(pred-saved['probs'][:,0])));assert drift<=1e-5
    data.test();before=digest(model.state_dict());tick=time.time();pred=predict(model,data,'test');elapsed=time.time()-tick
    assert digest(model.state_dict())==before and sha(fit/'best.pt')==c['checkpoint_sha256']
    labels=data.targets['test']['y'][:,a.endpoint:a.endpoint+1];metrics=all_scores(labels,pred[:,None])
    np.savez_compressed(out/'predictions_test.npz',ids=data.targets['test']['ids'],labels=labels,probs=pred[:,None])
    save(out/'run_manifest.json',identity)
    save(out/'completion.json',dict(complete=True,**identity,split='test',optimizer_updates=0,parameters_updated=False,
        fixed_development_checkpoint=True,model_sha256=c['checkpoint_sha256'],predictions_sha256=sha(out/'predictions_test.npz'),
        metrics=metrics,dev_replay_max_drift=drift,inference_seconds=elapsed,rows=len(pred),test_loaded=True,**resource_summary()))
    save(out/'progress.json',dict(status='complete',split='test'))


def main():
    p=argparse.ArgumentParser()
    for n in ('project','output','pilot-record','fit-output'): p.add_argument('--'+n,type=Path,required=n in ('project','output'))
    p.add_argument('--database',required=True);p.add_argument('--variant',choices=('Medformer',),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--endpoint',type=int,required=True)
    p.add_argument('--action',choices=('probe','train','evaluate'),required=True);p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();assert a.database==os.environ['TRACEHR_DATABASE'];assert a.seed in SEEDS and a.endpoint in (0,1)
    out=a.output.resolve();a.project=a.project.resolve(strict=True);assert out.is_relative_to(a.project) and not out.exists()
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30;out.mkdir(parents=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(out/'storage.json',dict(project=str(a.project),code=str(ROOT),output=str(out),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    save(out/'progress.json',dict(status='loading_verified_inputs'));data=Data(a.project)
    save(out/'progress.json',dict(status='randomly_initializing_native'));globals()[a.action](a,out,data)


if __name__=='__main__': main()
