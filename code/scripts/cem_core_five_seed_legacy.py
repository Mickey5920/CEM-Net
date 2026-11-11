"""Original RealMLP entry and raw-CEM checkpoint selection for new seeds."""
import argparse, copy, gc, os, shutil, time
from pathlib import Path
import numpy as np
import torch
from scripts import cem_core_five_seed_common as common
from scripts.tracehr_multiview_common import read, sha, atomic
from scripts.train_tracehr_unified import seed_all, rng_state, restore_rng
from scripts.tracehr_oof_common import all_scores
from scripts.mimiciii_ema_runtime import digest

def native_probe_resume(trace, data, seed, endpoint, out):
    model, opt = trace.shadow, trace.shadow_opt
    rng=np.random.default_rng(seed)
    saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state(rng))
    torch.save(saved,out/'representative_checkpoint.pt')
    def step():
        ix=rng.permutation(len(data.raw['train']))[:256]
        raw=torch.as_tensor(np.array(data.raw['train'][ix],copy=True),device='cuda')
        y=torch.as_tensor(data.targets['train']['y'][ix,endpoint],device='cuda')
        model.train();opt.zero_grad();opt.schedule(20/max(1,len(data.raw['train'])//256))
        loss=model.loss(model(raw),y);loss.backward();opt.step()
        return digest(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng)))
    first=step();reloaded=torch.load(out/'representative_checkpoint.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(reloaded['model']);opt.load_state_dict(reloaded['optimizer']);restore_rng(reloaded['rng'],rng)
    assert step()==first
    atomic(out/'path_audit.json',dict(complete=True,continuous_updates=20,resume_exact=True,
        native_continuous_update_parity=True,max_gradient_drift=trace.max_gradient_drift,
        max_parameter_drift=trace.max_parameter_drift,max_logit_drift=trace.max_logit_drift,
        test_loaded=False,optimizer_rng_reload_exact=True))

def realmlp(a,out,data,identity):
    import joblib
    from scripts import train_mimiciii_numeric_baseline as original
    from scripts import train_tracehr_noa_validation_native as native
    from scripts.tracehr_centered_native_common import stopping
    from models.tracehr_centered_native import FullNative
    spec=identity['protocol']
    original.numeric_protocol=lambda variant,seed:dict(spec,seed=seed)
    original.stopping=stopping
    if a.action=='evaluate':
        fit=a.fit_output;c=read(fit/'completion.json');identity=read(fit/'run_manifest.json')
        lock=read(fit.parent.parent/'fixed-checkpoint-lock.json')
        assert lock['complete'] and not lock['test_used_for_selection']
        ck=fit/'best.joblib';assert sha(ck)==c['checkpoint_sha256']==lock['selections'][fit.name]['checkpoint_sha256']
        model=joblib.load(ck);model.to('cuda:0');torch.set_grad_enabled(False)
        saved=dict(np.load(fit/'predictions_dev.npz'));q=model.predict_proba(data.raw['dev'])[:,1]
        drift=float(np.max(np.abs(q-saved['probs'][:,0])));assert drift<=1e-5
        cdata=common.expected();folder=a.project/cdata['test_inputs']
        for n,h in cdata['test_manifest'].items():assert sha(folder/n)==(h['sha256'] if isinstance(h,dict) else h)
        raw=np.load(folder/'test_raw.npy',mmap_mode='r');target=dict(np.load(folder/'test_targets.npz'))
        for split in ('train','dev'):assert not np.intersect1d(target['ids'][:,0],data.targets[split]['ids'][:,0]).size
        q=model.predict_proba(raw)[:,1:2];labels=target['y'][:,a.endpoint:a.endpoint+1]
        assert sha(ck)==c['checkpoint_sha256']
        np.savez_compressed(out/'predictions_test.npz',probs=q,ids=target['ids'],labels=labels)
        atomic(out/'run_manifest.json',identity)
        atomic(out/'completion.json',dict(complete=True,**identity,split='test',optimizer_updates=0,parameters_updated=False,
            model_sha256=c['checkpoint_sha256'],predictions_sha256=sha(out/'predictions_test.npz'),metrics=all_scores(labels,q),dev_replay_max_drift=drift))
        atomic(out/'progress.json',dict(status='complete'));return
    if a.action=='probe':
        seed_all(48)
        check=FullNative(data.raw,{s:data.targets[s]['y'] for s in ('train','dev')},a.endpoint,out/'reference48_init_tmp',48)
        archived=read(Path(common.expected()['archived_controls'][f'RealMLP:{a.endpoint}'])/'run_manifest.json')
        assert check.info['native_initial_trainable_sha256']==archived['initial_weights_sha256']
        del check;gc.collect();torch.cuda.empty_cache();seed_all(a.seed)
    else:
        pi=read(a.pilot_record/'completion.json')['identity']
        assert pi['source_manifest_sha256']==identity['source_manifest_sha256']
        assert {k:v for k,v in pi['protocol'].items() if k!='seed'}=={k:v for k,v in spec.items() if k!='seed'}
        atomic(out/'pilot_identity_check.json',dict(complete=True,fresh_initialization=True,pilot_weights_loaded=False,
            original_native_entry=True,same_seed_native_initialization_checked_in_seed_audit=True))
    saved_action,saved_pilot=a.action,a.pilot_record
    # NativeTrace independently captures and verifies each formal seed; the
    # seed-51 resource pilot is not a seed-52 weight-initialization reference.
    a.pilot_record=None
    base=native.NativeTrace;holder={}
    class Capture(base):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);holder['trace']=self
    native.NativeTrace=Capture
    try:original.b0(a,out,data,identity)
    finally:native.NativeTrace=base;a.action=saved_action;a.pilot_record=saved_pilot
    if saved_action=='probe':
        trace=holder['trace'];assert trace.steps==20 and max(trace.max_gradient_drift,trace.max_parameter_drift,trace.max_logit_drift)<=1e-6
        native_probe_resume(trace,data,a.seed,a.endpoint,out)
        r=read(out/'completion.json');r['checkpoint_bytes']=(out/'representative_checkpoint.pt').stat().st_size
        atomic(out/'completion.json',r)

def raw_reuse(a,out,data,model,identity):
    """Reuse a saved raw checkpoint only if it is the exact validation winner."""
    original=Path(common.expected()['full_ema'][f'{a.seed}:{a.endpoint}']['output'])
    history=read(original/'history.json');best=max(history,key=lambda r:r['raw_metrics']['mean_ap'])
    path=original/f"epoch_{best['epoch']:03d}_model.pt"
    if not path.exists():return False
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['epoch']==best['epoch'] and ck['identity']['seed']==a.seed and ck['identity']['endpoint']==a.endpoint
    model.load_state_dict(ck['model']);del ck
    from scripts import core_seed_raw_runtime as runtime
    q=runtime.predict(model,data,256);labels=data.targets['dev']['y'][:,a.endpoint:a.endpoint+1]
    scores=all_scores(labels,q);assert abs(scores['mean_ap']-best['raw_metrics']['mean_ap'])<1e-12
    torch.save(dict(model=model.state_dict(),identity=identity,state=dict(epoch=best['epoch']),role='raw_selected_inference'),out/'best.pt')
    np.savez_compressed(out/'predictions_dev.npz',probs=q,labels=labels,ids=data.targets['dev']['ids'])
    atomic(out/'history.json',[dict(r,dev_ap=r['raw_metrics']['mean_ap']) for r in history])
    atomic(out/'reuse.json',dict(complete=True,source=str(path),source_sha256=sha(path),original_history_sha256=sha(original/'history.json'),raw_validation_winner=True,new_training_updates=0))
    atomic(out/'completion.json',dict(complete=True,**identity,pilot=False,from_scratch=True,test_loaded=False,reused_fit=True,
        optimizer_updates=0,epochs=len(history),best_epoch=best['epoch'],dev_ap=scores['mean_ap'],metrics=scores,
        checkpoint_file='best.pt',checkpoint_sha256=sha(out/'best.pt'),predictions_sha256=sha(out/'predictions_dev.npz'),
        checkpoint_replay=dict(complete=True,max_probability_drift=0.),parameter_source='raw',seconds=0.))
    atomic(out/'progress.json',dict(status='complete',reused=True));return True

def main():
    p=argparse.ArgumentParser()
    for n in ('project','output','pilot-record','fit-output'):p.add_argument('--'+n,type=Path,required=n in ('project','output'))
    p.add_argument('--database',required=True);p.add_argument('--variant',required=True)
    p.add_argument('--seed',type=int,choices=(51,52),required=True);p.add_argument('--endpoint',type=int,choices=(0,1),required=True)
    p.add_argument('--action',choices=('probe','train','evaluate'),required=True);p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();a.project=a.project.resolve(strict=True);out=a.output.resolve();c=common.expected()
    assert dict(variant=a.variant,seed=a.seed,endpoint=a.endpoint) in c['matrix']
    assert out.is_relative_to(a.project) and not out.exists() and shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30
    out.mkdir(parents=True);a.preparation=a.project/c['input_preparation']
    torch.set_num_threads(4);seed_all(a.seed);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    atomic(out/'progress.json',dict(status='loading_verified_inputs'));data=common.Data(a.project,a.preparation)
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    spec=common.protocol(a.variant,a.seed)
    identity=dict(database=a.database,variant=a.variant,endpoint=a.endpoint,seed=a.seed,input_dim=data.input_dim,
        protocol=spec,source_manifest_sha256=sha(common.ROOT/'source_manifest.json'),input_manifest_sha256=sha(a.preparation/'completion.json'))
    if a.variant=='RealMLP':realmlp(a,out,data,identity);return
    assert a.variant=='CEM_noEMA'
    model=common.fresh_model(data,a.preparation,a.seed,a.endpoint,out,a.variant)
    identity.update(initial_weights_sha256=common.parameter_hash(model),parameters=sum(p.numel() for p in model.parameters()),
        members=8,lr_multiplier=1.,ablated_component=a.variant,ema_rule=common.EMA_RULE,epoch_budget=common.budget(a.seed,a.endpoint))
    atomic(out/'run_manifest.json',identity);model.cuda();gc.collect()
    base=common.bind_runtime()
    from scripts import core_seed_raw_runtime as runtime
    runtime.update, runtime.predict = base.update, base.predict
    if a.action=='probe':base.probe(a,out,data,model,identity,spec)
    elif a.action=='train':
        if not raw_reuse(a,out,data,model,identity):
            runtime.train(a,out,data,model,identity,spec)
            original=read(Path(c['full_ema'][f'{a.seed}:{a.endpoint}']['output'])/'history.json')
            now=read(out/'history.json');assert len(now)==len(original)
            delta=max(abs(x['raw_metrics']['mean_ap']-y['raw_metrics']['mean_ap']) for x,y in zip(now,original))
            assert delta<1e-10,delta
            atomic(out/'raw_trajectory_replay.json',dict(complete=True,all_epochs_AP_match=True,max_AP_difference=delta))
    else:
        from scripts import core_seed_raw_entry as entry
        entry.runtime=runtime;entry.evaluate(a,out,data)

if __name__=='__main__':main()
