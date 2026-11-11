"""Keep the historical clinical trainer; adapt only III inputs and execution evidence."""
import argparse,copy,gc,hashlib,json,os,random,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.mimiciii_baselines_common import *
from scripts.tracehr_multiview_common import atomic
# Import this historical model family in a fresh process, isolated from the
# pinned native-model package with overlapping module names.
sys.path.insert(0,str(ROOT/'clinical_source'))
import numpy as np
import torch
import train_exp2_clinical_deep as rt
import preprocess.ClinicalDataset as cd

original_collate=cd.clinical_collate_fn
def collate(items):
    b=original_collate(items)
    if b['event_type'].shape[1]==0:
        for n in ('event_type','time_since_start','time_since_last_event'):
            b[n]=torch.zeros((len(items),1),dtype=b[n].dtype)
    return b
cd.clinical_collate_fn=collate

native_build=rt.build_model
def empty_attention_guard(module,args,kwargs):
    mask=kwargs.get('src_key_padding_mask')
    if mask is not None:
        assert mask.dtype==torch.bool
        empty=mask.all(1)
        if empty.any():
            # Avoid undefined all-masked softmax. The original observation mask
            # still zeros every resulting hidden state and the pooled evidence.
            mask=mask.clone();mask[empty,0]=False
            kwargs=dict(kwargs,src_key_padding_mask=mask)
            return args,kwargs
    return None
def adapted_build(args,*rest,**kw):
    model=native_build(args,*rest,**kw)
    if args.model in ('hymate','xtsformer'):
        for module in model.modules():
            if isinstance(module,torch.nn.TransformerEncoder):
                module.register_forward_pre_hook(empty_attention_guard,with_kwargs=True)
    return model
rt.build_model=adapted_build

def weights(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def config(a,variant=None):
    variant=variant or a.variant;cfg=expected()['clinical_configs'][variant]
    argv=sys.argv
    try:
        sys.argv=['historical-clinical-entry','--model',variant,'--data_dir',str(a.project/expected()['preparation']),'--output_dir',str(a.output)]
        c=rt.parse_args()
    finally:sys.argv=argv
    for k,v in cfg.items():setattr(c,k,v)
    c.model=variant;c.seed=a.seed;c.device='cuda:0';c.data_dir=str(a.project/expected()['preparation']);c.output_dir=str(a.output)
    c.primary_tasks=list(TASKS);c.evaluation_scope='dev_only';c.time_scale_hours=24.
    c.max_train_samples=c.max_eval_samples=None;c.pretrained_checkpoint=None;c.save_diagnostic_stats=False
    c.save_predictions=True;c.progress='log';c.log_interval=20
    assert c.max_epochs==50 and c.early_stopping_patience==999 and c.monitor=='mean_primary_auprc'
    assert c.ranking_loss_weight==0 and c.grad_clip==1. and c.n_layers==2 and c.d_model==64
    return c
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def initial(model,identity,out,pilot):
    assert all(p.requires_grad for p in model.parameters())
    identity.update(initial_weights_sha256=weights(model),parameters=sum(p.numel() for p in model.parameters()))
    if pilot:
        ref=read(pilot/'completion.json')['identity']
        for k in ('variant','database','source_manifest_sha256','input_manifest_sha256','parameters'):assert ref[k]==identity[k],k
        if identity['seed']==48:assert ref['initial_weights_sha256']==identity['initial_weights_sha256']
    atomic(out/'initialization.json',dict(**identity,optimizer_updates=0,pretrained_loaded=False))
    atomic(out/'identity.json',identity)
def infer(model,b,device):
    return model(b['event_type'].to(device),b['time_since_start'].to(device),
        time_gap=b['time_since_last_event'].to(device),source_features=b['source_features'].to(device))[0]
def empty_test(model,device):
    model.eval()
    with torch.no_grad():
        z=torch.zeros((2,1),dtype=torch.long,device=device);t=torch.zeros((2,1),device=device)
        logits=model(z,t,time_gap=t,source_features=torch.zeros((2,0),device=device))[0]
        assert logits.shape==(2,2) and torch.isfinite(logits).all(),type(model.encoder).__name__
    return dict(complete=True,empty_stays_retained=True,padding_only=True,finite_logits=True)
def cpu_checks(a):
    torch.set_num_threads(4);rows={};nt=len(read(a.project/expected()['numeric_preparation']/'token_to_id.json'))
    for v in CLINICAL:
        c=config(a,v);rt.seed_everything(48);m=rt.build_model(c,nt,2)
        m.eval();z=torch.tensor([[1,2,0],[2,1,3]]);t=torch.tensor([[0.,1.,0.],[0.,1.,2.]])
        with torch.no_grad():adapted=m(z,t,time_gap=t)[0]
        rt.seed_everything(48);original=native_build(c,nt,2);original.eval()
        with torch.no_grad():unchanged=original(z,t,time_gap=t)[0]
        assert torch.equal(adapted,unchanged) and weights(original)==weights(m)
        del original
        rows[v]=dict(parameters=sum(p.numel() for p in m.parameters()),batch_size=c.batch_size,lr=c.lr,
            trainable=all(p.requires_grad for p in m.parameters()),empty=empty_test(m,'cpu'),nonempty_logit_drift=0.)
        # Mixed empty/nonempty records must also have finite training gradients.
        z[1]=0;m.train();m.zero_grad(set_to_none=True);logits=m(z,t,time_gap=t)[0]
        rt.masked_bce_with_logits(logits,torch.zeros_like(logits)).backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        rows[v]['mixed_gradients_finite']=True
        del m;gc.collect()
    atomic(a.output/'completion.json',dict(complete=True,optimizer_updates=0,test_loaded=False,methods=rows))
def probe(a,c,out,identity):
    rt.seed_everything(a.seed);samples,tasks,nt=rt.load_splits(c.data_dir,'dev_only')
    assert tasks==list(TASKS)
    model=rt.build_model(c,nt,2).cuda();initial(model,identity,out,None);empty=empty_test(model,'cuda')
    loader=cd.get_clinical_dataloader(samples['train'],tasks,batch_size=c.batch_size,shuffle=True,
        num_workers=c.num_workers,pin_memory=c.pin_memory,seed=a.seed,source_names=[])
    batches=[]
    for b in loader:
        batches.append(b)
        if len(batches)==3:break
    # A full native batch of longest histories bounds attention memory without
    # changing any formal batch size, loss normalization or data selection.
    longest=sorted(samples['train'],key=lambda x:x['length'],reverse=True)[:c.batch_size]
    stress=next(iter(cd.get_clinical_dataloader(longest,tasks,batch_size=c.batch_size,shuffle=False,source_names=[])))
    opt=torch.optim.Adam(model.parameters(),lr=c.lr,weight_decay=c.weight_decay)
    pos=rt.compute_pos_weight(samples['train'],tasks,'cuda');started=time.time();torch.cuda.reset_peak_memory_stats()
    atomic(out/'progress.json',dict(status='training_probe',updates=0))
    init={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    rt.train_epoch(model,[stress],opt,'cuda',pos,grad_clip=c.grad_clip,progress='none',log_interval=0)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.load_state_dict(init);del init,opt;gc.collect();torch.cuda.empty_cache()
    opt=torch.optim.Adam(model.parameters(),lr=c.lr,weight_decay=c.weight_decay)
    rt.train_epoch(model,[batches[0]],opt,'cuda',pos,grad_clip=c.grad_clip,progress='none',log_interval=0)
    ck=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()),rng=rng())
    torch.save(ck,out/'representative_checkpoint.pt')
    rt.train_epoch(model,[batches[1]],opt,'cuda',pos,grad_clip=c.grad_clip,progress='none',log_interval=0)
    after={k:p.detach().cpu().clone() for k,p in model.named_parameters()}
    model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore(ck['rng'])
    rt.train_epoch(model,[batches[1]],opt,'cuda',pos,grad_clip=c.grad_clip,progress='none',log_interval=0)
    drift=max(float((p.detach().cpu()-after[n]).abs().max()) for n,p in model.named_parameters());assert drift<=1e-6
    model.eval()
    with torch.no_grad():assert torch.isfinite(infer(model,batches[2],'cuda')).all()
    r=dict(complete=True,pilot=True,identity=identity,from_scratch=True,test_loaded=False,frozen_learnable_parameters=0,
        updates=4,empty_input=empty,full_batch_longest_stress=True,optimizer_resume_max_parameter_drift=drift,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,checkpoint_bytes=(out/'representative_checkpoint.pt').stat().st_size,
        seconds=time.time()-started,finished_at=time.time())
    atomic(out/'completion.json',r);atomic(out/'progress.json',dict(status='complete',updates=4))
def train(a,c,out,identity):
    from sklearn.metrics import average_precision_score
    assert a.pilot_record
    state=dict(epoch=0,updates=0,best_ap=None);models=[];started=time.time()
    build=rt.build_model;epoch=rt.train_epoch;predict=rt.predict;step=torch.optim.Adam.step
    def observed_build(*args,**kw):
        m=build(*args,**kw);initial(m,identity,out,a.pilot_record);models.append(m);return m
    def observed_step(opt,*args,**kw):
        if state['updates']==0:
            grads={n:p.grad for n,p in models[0].named_parameters()}
            assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)
            before={n:p.detach().clone() for n,p in models[0].named_parameters()}
        result=step(opt,*args,**kw);state['updates']+=1
        if state['updates']==1:
            changed=[n for n,p in models[0].named_parameters() if not torch.equal(before[n],p)]
            assert changed
            atomic(out/'first_update.json',dict(updates=1,updated_tensors=len(changed),parameter_tensors=len(before),
                disconnected_parameters=[n for n,g in grads.items() if g is None],all_retained_parameters_trainable=True))
        if state['updates']==1 or state['updates']%20==0:atomic(out/'progress.json',dict(status='training',**state,elapsed_seconds=time.time()-started))
        return result
    def observed_epoch(model,loader,opt,*args,**kw):
        state['epoch']=kw['epoch'];r=epoch(model,loader,opt,*args,**kw)
        assert np.isfinite(r['loss'])
        temp=out/'latest.pt.writing';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng(),epoch=state['epoch'],identity=identity),temp)
        os.replace(temp,out/'latest.pt');return r
    def observed_predict(*args,**kw):
        result=predict(*args,**kw);probs,labels,_,_=result;assert np.isfinite(probs).all()
        ap=[float(average_precision_score(labels[:,e],probs[:,e])) for e in (0,1)]
        state['best_ap']=max(state['best_ap'] or -1.,float(np.mean(ap)))
        atomic(out/'progress.json',dict(status='evaluating',**state,dev_ap=ap,elapsed_seconds=time.time()-started));return result
    rt.parse_args=lambda:c;rt.build_model=observed_build;rt.train_epoch=observed_epoch;rt.predict=observed_predict;torch.optim.Adam.step=observed_step
    try:rt.main()
    finally:rt.build_model=build;rt.train_epoch=epoch;rt.predict=predict;torch.optim.Adam.step=step
    assert not (out/'predictions_test.npz').exists()
    pred=dict(np.load(out/'predictions_dev.npz',allow_pickle=True));target=dict(np.load(a.project/expected()['numeric_preparation']/'dev_targets.npz'))
    np.testing.assert_array_equal(pred['ids'],target['ids']);np.testing.assert_array_equal(pred['labels'],target['y'])
    ap=[float(average_precision_score(pred['labels'][:,e],pred['probs'][:,e])) for e in (0,1)]
    metrics=read(out/'metrics.json');assert metrics['evaluation_scope']=='dev_only'
    checkpoint=out/'checkpoints/best.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    m=build(c,ck['num_event_types_raw'],2).cuda();m.load_state_dict(ck['model_state_dict'])
    samples,tasks,_=rt.load_splits(c.data_dir,'dev_only')
    loader=cd.get_clinical_dataloader(samples['dev'],tasks,batch_size=c.batch_size,shuffle=False,num_workers=c.num_workers,
        pin_memory=c.pin_memory,seed=c.seed+100000,source_names=[])
    replay=predict(m,loader,'cuda',tasks)[0];drift=float(np.max(np.abs(replay-pred['probs'])));assert drift<=1e-5
    import csv
    with (out/'train_log.csv').open() as f:hist=list(csv.DictReader(f))
    assert len(hist)==50
    best=max(hist,key=lambda x:float(x['monitor_value']));assert ck['epoch']==int(best['epoch'])
    assert abs(np.mean(ap)-float(best['monitor_value']))<1e-7
    r=dict(complete=True,pilot=False,identity=identity,from_scratch=True,test_loaded=False,frozen_learnable_parameters=0,
        dev_ap=ap,epochs=len(hist),best_epoch=ck['epoch'],selection='one checkpoint by mean dev AP of both tasks',
        seconds=time.time()-started,finished_at=time.time(),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        checkpoint_file='checkpoints/best.pt',checkpoint_sha256=sha(checkpoint),predictions_sha256=sha(out/'predictions_dev.npz'),
        checkpoint_replay=dict(complete=True,max_probability_drift=drift),actual_updates=state['updates'])
    atomic(out/'completion.json',r);atomic(out/'progress.json',dict(status='complete',dev_ap=ap,epochs=50,updates=state['updates']))
def main():
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true')
    for n in ('project','output','pilot-record'):p.add_argument('--'+n,type=Path,required=n!='pilot-record')
    p.add_argument('--variant',choices=CLINICAL);p.add_argument('--seed',type=int,choices=SEEDS,default=48)
    p.add_argument('--action',choices=['probe','train']);p.add_argument('--expected-gib',type=float,default=40)
    a=p.parse_args();a.project=a.project.resolve(strict=True);a.output=a.output.resolve();assert a.output.is_relative_to(a.project)
    assert not a.output.exists();assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30;a.output.mkdir(parents=True)
    for k in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        v=Path(os.environ[k]).resolve();assert v.is_relative_to(a.project);v.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if a.verify_only:return cpu_checks(a)
    assert a.variant and a.action and torch.cuda.is_available();c=config(a)
    identity=dict(database='mimiciii',variant=a.variant,seed=a.seed,endpoints=[0,1],
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),input_manifest_sha256=sha(a.project/expected()['preparation']/'completion.json'),
        protocol={k:getattr(c,k) for k in ('model','batch_size','lr','weight_decay','max_epochs','early_stopping_patience','monitor','pooling','dropout','d_model','d_inner','n_layers','n_head','d_k','d_v','grad_clip','num_workers','pin_memory','time_scale_hours')})
    atomic(a.output/'storage.json',dict(project=str(a.project),code=str(ROOT),output=str(a.output),cache=os.environ['XDG_CACHE_HOME'],temp=os.environ['TMPDIR'],expected_gib=a.expected_gib))
    atomic(a.output/'progress.json',dict(status='loading_verified_inputs'))
    (probe if a.action=='probe' else train)(a,c,a.output,identity)
if __name__=='__main__':main()
