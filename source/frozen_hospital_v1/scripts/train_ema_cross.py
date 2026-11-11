"""Fresh fits and locked, forward-only tests for the authorized two methods."""
import argparse,gc,math,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.ema_cross_common import *
import numpy as np
from scripts.train_tracehr_unified import seed_all,rng_state,restore_rng
from scripts.tracehr_oof_common import all_scores
from scripts.train_tracehr_native_baseline import tabm_data,tabm_predict,tabm_fit

def mini_make(data,seed):
    from tabm import TabM
    seed_all(seed)
    m=TabM.make(n_num_features=data.input_dim,d_out=1,k=32,d_block=256,n_blocks=3,dropout=.2,arch_type='tabm-mini')
    assert type(m.backbone).__name__!='MLPBackboneBatchEnsemble'
    assert all(p.requires_grad for p in m.parameters())
    return m

def mini_identity(a,out,data,identity):
    m=mini_make(data,a.seed)
    identity.update(initial_weights_sha256=parameter_hash(m),parameters=sum(p.numel() for p in m.parameters()),members=32)
    atomic(out/'initialization_observation.json',dict(complete=True,optimizer_updates=0,**identity))
    atomic(out/'architecture_audit.json',dict(complete=True,official_arch_type='tabm-mini',backbone_class=type(m.backbone).__name__,
          recipe=protocol(a.variant,a.seed),learned_checkpoint_loaded=False))
    atomic(out/'run_manifest.json',identity)
    return m

def mini_probe(a,out,data,identity):
    from scripts.mimiciii_ema_runtime import digest
    m=mini_identity(a,out,data,identity).cuda();seed_all(a.seed)
    raw=tabm_data(data,Path(expected()['tabm_preprocessing']).parent,out)
    opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=.001);rng=np.random.default_rng(a.seed)
    order=rng.permutation(len(raw['train']));times=[];initial={n:p.detach().clone() for n,p in m.named_parameters()}
    def step(ix):
        m.train();opt.zero_grad(set_to_none=True)
        x=torch.as_tensor(raw['train'][ix],device='cuda');y=torch.as_tensor(data.targets['train']['y'][ix,a.endpoint],device='cuda',dtype=torch.float32)
        z=m(x).squeeze(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y[:,None].expand_as(z))
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step();return float(loss)
    started=time.time()
    for i in range(20):
        tick=time.perf_counter();step(order[i*256:(i+1)*256]);torch.cuda.synchronize();times.append(time.perf_counter()-tick)
        atomic(out/'progress.json',dict(status='training_probe',updates=i+1))
    moved={n:float((p.detach()-initial[n]).abs().max()) for n,p in m.named_parameters()}
    assert all(x>0 for x in moved.values());atomic(out/'gradient_check.json',dict(all_connected=True,relative_update=moved))
    del initial
    ck=dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng))
    torch.save(ck,out/'representative_checkpoint.pt');del ck
    ix=rng.permutation(len(raw['train']))[:256];step(ix)
    reference=digest(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng)))
    ck=torch.load(out/'representative_checkpoint.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng)
    ix2=rng.permutation(len(raw['train']))[:256];assert np.array_equal(ix,ix2);step(ix2)
    assert digest(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng)))==reference
    pred=tabm_predict(m,raw['dev'][:512]);assert np.isfinite(pred).all()
    audit=dict(complete=True,continuous_updates=20,all_parameters_updated=True,resume_exact=True,test_loaded=False,
               official_arch_type='tabm-mini',median_step_seconds=float(np.median(times[1:])))
    atomic(out/'path_audit.json',audit)
    atomic(out/'completion.json',dict(complete=True,pilot=True,identity=identity,from_scratch=True,test_loaded=False,
          frozen_learnable_parameters=0,checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size,
          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.time()-started,path_audit_sha256=sha(out/'path_audit.json')))
    atomic(out/'progress.json',dict(status='complete'))

def mini_train(a,out,data,identity):
    from tabm import TabM
    m=mini_identity(a,out,data,identity);del m
    pilot=read(a.pilot_record/'completion.json');assert pilot['complete'] and pilot['identity']==identity
    atomic(out/'pilot_identity_check.json',dict(match=True,fresh_initialization=True,pilot_weights_loaded=False))
    raw=tabm_data(data,Path(expected()['tabm_preprocessing']).parent,out)
    original=TabM.make
    def make(*args,**kwargs):
        m=original(*args,**dict(kwargs,arch_type='tabm-mini'))
        assert parameter_hash(m)==identity['initial_weights_sha256'];return m
    TabM.make=make;seed_all(a.seed);started=time.time()
    atomic(out/'training_started.json',dict(started_at=started,actual_optimizer_updates=0))
    try: m,pred,history,best_epoch=tabm_fit(raw,{s:data.targets[s]['y'] for s in raw},out,a.endpoint,a.seed,256)
    finally: TabM.make=original
    drift=float(np.max(np.abs(pred-tabm_predict(m,raw['dev']))));assert drift<=1e-5
    labels=data.targets['dev']['y'][:,a.endpoint:a.endpoint+1];scores=all_scores(labels,pred[:,None])
    np.savez_compressed(out/'predictions_dev.npz',ids=data.targets['dev']['ids'],labels=labels,probs=pred[:,None])
    atomic(out/'completion.json',dict(complete=True,pilot=False,identity=identity,**identity,epochs=len(history),best_epoch=best_epoch,
          dev_ap=scores['mean_ap'],metrics=scores,from_scratch=True,test_loaded=False,frozen_learnable_parameters=0,
          checkpoint_file='best.pt',checkpoint_sha256=sha(out/'best.pt'),predictions_sha256=sha(out/'predictions_dev.npz'),
          checkpoint_replay=dict(complete=True,max_probability_drift=drift),seconds=time.time()-started,
          optimizer_updates=len(history)*math.ceil(len(raw['train'])/256),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    atomic(out/'progress.json',dict(status='complete',epochs=len(history),best_ap=scores['mean_ap']))

def evaluate(a,out,data):
    job=a.fit_output;c=read(job/'completion.json');identity=read(job/'run_manifest.json')
    lock=read(job.parent.parent/'fixed-checkpoint-lock.json')
    assert lock['complete'] and len(lock['selections'])==12
    assert sha(job/'best.pt')==c['checkpoint_sha256']==lock['selections'][job.name]['checkpoint_sha256']
    spec=protocol(a.variant,a.seed);torch.set_grad_enabled(False)
    if a.variant==VARIANT:
        m=fresh_model(data,a.project/expected()['input_preparation'],a.seed,a.endpoint,out).cuda()
        ck=torch.load(job/'best.pt',map_location='cpu',weights_only=False)
        assert ck['role']=='EMA_selected_inference' and ck['identity']==identity
        m.load_state_dict(ck['model']);del ck
        def predict_raw(raw):
            m.eval();parts=[]
            for start in range(0,len(raw),256):
                x=torch.from_numpy(np.array(raw[start:start+256],copy=True)).cuda()
                parts.append(m(x).softmax(-1)[...,1].mean(1).cpu().numpy())
            return np.concatenate(parts)
    else:
        import joblib
        m=mini_make(data,a.seed).cuda();ck=torch.load(job/'best.pt',map_location='cpu',weights_only=False)
        m.load_state_dict(ck['model_state_dict']);del ck
        transform=joblib.load(expected()['tabm_preprocessing']);d=transform['phi_dim']
        def predict_raw(raw):
            z=np.array(raw,copy=True)
            for start in range(0,len(z),1024):
                block=z[start:start+1024,:d];v=(block-transform['phi_center'])*transform['phi_factor'];v/=np.sqrt(1+(v/3)**2)
                v[:,transform['binary']]=block[:,transform['binary']];z[start:start+1024,:d]=v
            z[:,d:]=transform['numeric_quantile'].transform(z[:,d:]);assert np.isfinite(z).all()
            return tabm_predict(m,z)
    atomic(out/'run_manifest.json',identity)
    saved=dict(np.load(job/'predictions_dev.npz'));replay=predict_raw(data.raw['dev'])
    np.testing.assert_array_equal(saved['ids'],data.targets['dev']['ids'])
    np.testing.assert_array_equal(saved['labels'][:,0],data.targets['dev']['y'][:,a.endpoint])
    drift=float(np.max(np.abs(replay-saved['probs'][:,0])));assert drift<=1e-5
    # No test arrays are opened before the fixed-checkpoint lock and dev replay pass.
    folder=a.project/expected()['test_inputs']
    for name,h in expected()['test_artifacts'].items():assert sha(folder/name)==h,name
    target=dict(np.load(folder/'test_targets.npz'));raw=np.load(folder/'test_raw.npy',mmap_mode='r')
    assert not np.intersect1d(target['ids'][:,0],data.targets['train']['ids'][:,0]).size
    assert not np.intersect1d(target['ids'][:,0],data.targets['dev']['ids'][:,0]).size
    from scripts.mimiciii_ema_runtime import digest
    before=digest(m.state_dict());tick=time.time();prob=predict_raw(raw);elapsed=time.time()-tick
    assert digest(m.state_dict())==before and sha(job/'best.pt')==c['checkpoint_sha256']
    labels=target['y'][:,a.endpoint:a.endpoint+1];scores=all_scores(labels,prob[:,None])
    np.savez_compressed(out/'predictions_test.npz',ids=target['ids'],labels=labels,probs=prob[:,None])
    atomic(out/'completion.json',dict(complete=True,**identity,split='test',optimizer_updates=0,parameters_updated=False,
          fixed_development_checkpoint=True,model_sha256=c['checkpoint_sha256'],predictions_sha256=sha(out/'predictions_test.npz'),
          metrics=scores,dev_replay_max_drift=drift,inference_seconds=elapsed,rows=len(prob),test_loaded=True))
    atomic(out/'progress.json',dict(status='complete',split='test'))

def main():
    p=argparse.ArgumentParser()
    for n in ('project','output','pilot-record','fit-output'):p.add_argument('--'+n,type=Path,required=n in ('project','output'))
    p.add_argument('--database',choices=('mimic','eicu'),required=True);p.add_argument('--variant',choices=(VARIANT,'TabM-mini'),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--endpoint',type=int,choices=(0,1),required=True)
    p.add_argument('--action',choices=('probe','train','evaluate'),required=True);p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();assert a.database==database();a.project=a.project.resolve(strict=True);out=a.output.resolve()
    assert out.is_relative_to(a.project) and not out.exists();assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30
    out.mkdir(parents=True)
    for k in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        q=Path(os.environ[k]).resolve();assert q.is_relative_to(a.project);q.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    atomic(out/'storage.json',dict(project=str(a.project),code=str(ROOT),output=str(out),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    prep=a.project/expected()['input_preparation'];data=Data(a.project,a.database,prep)
    if a.action=='evaluate':evaluate(a,out,data);return
    atomic(out/'progress.json',dict(status='randomly_initializing_native'))
    identity=dict(database=a.database,variant=a.variant,endpoint=a.endpoint,seed=a.seed,protocol=protocol(a.variant,a.seed),
                  source_manifest_sha256=sha(ROOT/'source_manifest.json'),input_manifest_sha256=sha(prep/'completion.json'),input_dim=data.input_dim)
    if a.variant=='TabM-mini':
        (mini_probe if a.action=='probe' else mini_train)(a,out,data,identity);return
    model=fresh_model(data,prep,a.seed,a.endpoint,out)
    identity.update(initial_weights_sha256=parameter_hash(model),parameters=sum(p.numel() for p in model.parameters()),members=8,
                    ema_rule=EMA_RULE,epoch_budget=budget(a.seed,a.endpoint),parameter_source='EMA')
    atomic(out/'run_manifest.json',identity);model=model.cuda();gc.collect()
    runner=bind_ema_runtime()
    (runner.probe if a.action=='probe' else runner.train)(a,out,data,model,identity,identity['protocol'])
    if a.action=='probe':
        atomic(out/'path_audit.json',read(out/'ema_path_audit.json'))

if __name__=='__main__':main()
