"""Fresh official baselines using the historical complete training recipes."""
import argparse
import gc
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
deps=Path(os.environ['TRACEHR_TABULAR_DEPS'])
sys.path.insert(0,str(deps))
for w in deps.glob('*.whl'):sys.path.insert(0,str(w))
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from scripts.tracehr_native_transfer_common import Data, old_baseline, protocol
from scripts.tracehr_multiview_common import atomic, read, sha, checked_output
from scripts.tracehr_oof_common import all_scores


def verify_reference(data, reference, endpoint, pred=None):
    archive=dict(np.load(reference/'predictions_dev.npz'))
    assert np.array_equal(archive['ids'],data.targets['dev']['ids'])
    assert np.array_equal(archive['labels'],data.targets['dev']['y'])
    target=data.targets['dev']['y'][:,endpoint]
    ref=archive['probs'][:,endpoint]
    result=dict(reference=str(reference),reference_predictions_sha256=sha(reference/'predictions_dev.npz'),
                reference_AP=float(average_precision_score(target,ref)),ids_and_labels_match=True)
    if pred is not None:
        delta=float(average_precision_score(target,pred))-result['reference_AP']
        drift=float(np.max(np.abs(pred-ref)))
        result.update(AP_difference=delta,max_probability_difference=drift,
                      passed=abs(delta)<=1e-5 and drift<=1e-4,
                      tolerances=dict(AP=1e-5,probability=1e-4))
    return result


def native_fit(raw, target, out, endpoint, seed, epochs, device, progress):
    sys.modules['lightning']=None
    from pytabkit import RealMLP_TD_Classifier
    from pytabkit.models.training.metrics import Metrics
    from pytabkit.models.training.nn_creator import NNCreator
    original_metric,original_create=Metrics.apply,NNCreator.create_model
    history=[];capture={}
    started=time.time()
    def capture_create(creator,ds,idxs_list):
        net=original_create(creator,ds,idxs_list)
        import hashlib
        h=hashlib.sha256()
        for name,p in net.named_parameters():
            if p.requires_grad:
                h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        capture.update(internal_seeds=[list(map(int,x.sub_split_seeds)) for x in idxs_list],
                       initial_trainable_sha256=h.hexdigest(),
                       trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
                       fixed_statistics=sum(p.numel() for p in net.parameters() if not p.requires_grad))
        assert capture['internal_seeds']==[[int(np.random.RandomState(seed).randint(0,2**31-1))]]
        for mod in net.modules():
            if type(mod).__name__=='WeightLayer':assert mod.weight.requires_grad
        atomic(out/'initialization.json',capture)
        return net
    def exact_ap(logits,labels,name):
        if name!='tracehr_1_ap':return original_metric(logits,labels,name)
        assert torch.isfinite(logits).all()
        ap=float(average_precision_score(labels.detach().cpu().numpy().reshape(-1),
                        logits.softmax(-1)[:,1].detach().cpu().numpy()))
        row=dict(validation_event=len(history)+1,dev_ap=ap,elapsed_seconds=time.time()-started)
        history.append(row);atomic(out/'history.json',history)
        atomic(out/'progress.json',dict(status='native_validation',**row))
        return logits.new_tensor(1-ap)
    Metrics.apply=staticmethod(exact_ap);NNCreator.create_model=capture_create
    try:
        model=RealMLP_TD_Classifier(device=device,random_state=seed,n_cv=1,n_refit=0,n_threads=4,
            tmp_folder=str(out/'native_tmp'),verbosity=2,n_epochs=epochs,batch_size=256,predict_batch_size=256,
            val_metric_name='tracehr_1_ap',use_ls=False,use_early_stopping=False,num_emb_type='pbld')
        atomic(out/'effective_config.json',model.get_config())
        model.fit(raw['train'],target['train'][:,endpoint].astype(np.int64),
                  X_val=raw['dev'],y_val=target['dev'][:,endpoint].astype(np.int64))
        pred=model.predict_proba(raw['dev'])[:,1]
        fit=model.cv_alg_interface_.fit_params
        atomic(out/'fit_params.json',fit)
        best_epoch=int(fit[0]['stop_epoch']['tracehr_1_ap'])
    finally:
        Metrics.apply=staticmethod(original_metric);NNCreator.create_model=original_create
    return model,pred,history,best_epoch,capture


def tabm_data(data, reference, out):
    old=joblib.load(reference/'preprocessing.joblib')
    assert old['fit_split']=='train'
    from scripts.tracehr_unified_common import original_pointer
    _,_,manifest=original_pointer(Path(os.environ['TRACEHR_PROJECT']),os.environ['TRACEHR_DATABASE'])
    assert sha(reference/'preprocessing.joblib')==manifest['identity']['recipe']['v_preprocessing']
    atomic(out/'preprocessing.json',dict(path=str(reference/'preprocessing.joblib'),
             sha256=sha(reference/'preprocessing.joblib'),fit_split='train',learned_weights_loaded=False))
    result={}
    d=old['phi_dim']
    for split in ('train','dev'):
        z=np.array(data.raw[split],copy=True)
        for start in range(0,len(z),1024):
            block=z[start:start+1024,:d]
            v=(block-old['phi_center'])*old['phi_factor']
            v/=np.sqrt(1+(v/3)**2)
            v[:,old['binary']]=block[:,old['binary']]
            z[start:start+1024,:d]=v
        z[:,d:]=old['numeric_quantile'].transform(z[:,d:])
        assert np.isfinite(z).all()
        result[split]=z
    return result


@torch.no_grad()
def tabm_predict(model,raw):
    model.eval();parts=[]
    for start in range(0,len(raw),256):
        x=torch.as_tensor(raw[start:start+256],device='cuda:0')
        parts.append(model(x).sigmoid().mean(1).squeeze(-1).cpu().numpy())
    return np.concatenate(parts)


def tabm_fit(raw,target,out,endpoint,seed,limit):
    from tabm import TabM
    model=TabM.make(n_num_features=raw['train'].shape[1],d_out=1,k=32,d_block=256,n_blocks=3,dropout=.2).cuda()
    assert all(p.requires_grad for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
    rng=np.random.default_rng(seed);best=-1.;stale=0;history=[]
    for epoch in range(1,limit+1):
        tick=time.time();model.train();order=rng.permutation(len(raw['train']));loss_sum=0.
        for step,start in enumerate(range(0,len(order),256),1):
            ix=order[start:start+256]
            bx=torch.as_tensor(raw['train'][ix],device='cuda:0')
            by=torch.as_tensor(target['train'][ix,endpoint],device='cuda:0',dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits=model(bx).squeeze(-1)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,by[:,None].expand_as(logits))
            assert torch.isfinite(loss);loss.backward()
            if epoch==1 and step==1:
                assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                atomic(out/'gradient_check.json',dict(all_learnable_parameters_connected=True))
            optimizer.step();loss_sum+=float(loss.detach())*len(ix)
            if step==1 or step%30==0:
                atomic(out/'progress.json',dict(status='training',epoch=epoch,step=step,best_ap=best))
        pred=tabm_predict(model,raw['dev'])
        ap=float(average_precision_score(target['dev'][:,endpoint],pred))
        history.append(dict(epoch=epoch,dev_ap=ap,train_loss=loss_sum/len(order),seconds=time.time()-tick))
        atomic(out/'history.json',history)
        if ap>best+1e-9:
            best=ap;best_epoch=epoch;stale=0;selected=pred.copy()
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch,numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.random.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all()),out/'best.pt')
        else:stale+=1
        atomic(out/'progress.json',dict(status='epoch_complete',epoch=epoch,best_ap=best,best_epoch=best_epoch))
        if stale>=16:break
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model_state_dict'])
    replay=tabm_predict(model,raw['dev']);assert np.max(np.abs(replay-selected))<=1e-6
    return model,selected,history,best_epoch


def main():
    p=argparse.ArgumentParser()
    for name in ('project','output','preparation'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--database',choices=['mimic','eicu'],required=True)
    p.add_argument('--variant',choices=['B0','B1'],required=True)
    p.add_argument('--endpoint',type=int,choices=[0,1],required=True)
    p.add_argument('--action',choices=['probe','smoke','train'],required=True)
    p.add_argument('--start-barrier',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--expected-gib',type=float,required=True)
    a=p.parse_args();a.project=a.project.resolve(strict=True)
    out=checked_output(a.project,a.output);assert not out.exists();out.mkdir(parents=True)
    import shutil
    assert shutil.disk_usage(a.project).free>(a.expected_gib+10)*2**30
    for k in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','MPLCONFIGDIR'):
        checked_output(a.project,os.environ[k]).mkdir(parents=True,exist_ok=True)
    os.environ['TRACEHR_PROJECT']=str(a.project);os.environ['TRACEHR_DATABASE']=a.database
    atomic(out/'progress.json',dict(status='loading_verified_inputs'))
    data=Data(a.project,a.database,a.preparation);reference=old_baseline(a.project,a.database,a.variant)
    assert sha(reference/'run_manifest.json')==data.record['baseline_references'][a.variant]['manifest_sha256']
    verify_reference(data,reference,a.endpoint)
    identity=dict(database=a.database,variant=a.variant,endpoint=a.endpoint,seed=a.seed,
                  input_manifest_sha256=sha(a.preparation/'completion.json'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),
                  protocol=protocol(a.variant),precision='float32',from_scratch=True,test_loaded=False,
                  frozen_learnable_parameters=0,baseline_code='original package training recipe, instrumented only')
    atomic(out/'run_manifest.json',identity)
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.cuda.set_device(0);torch.cuda.manual_seed_all(a.seed);torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_per_process_memory_fraction(30*2**30/torch.cuda.get_device_properties(0).total_memory)
    raw=tabm_data(data,reference,out) if a.variant=='B1' else data.raw
    target={s:data.targets[s]['y'] for s in ('train','dev')}
    # Historical trainer resets these immediately before each endpoint.
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    limit=2 if a.action!='train' else 256
    if a.start_barrier:
        atomic(out/'progress.json',dict(status='waiting_coexistence_barrier'))
        deadline=time.time()+1200
        while not a.start_barrier.exists():
            assert time.time()<deadline
            time.sleep(.5)
    started=time.time();atomic(out/'training_started.json',dict(started_at=started,pid=os.getpid()))
    atomic(out/'progress.json',dict(status='initializing_official_package'))
    if a.variant=='B0':
        model,pred,history,best_epoch,initial=native_fit(raw,target,out,a.endpoint,a.seed,limit,'cuda:0',None)
        model.to('cpu');joblib.dump(model,out/'best.joblib')
        del model;gc.collect();torch.cuda.empty_cache()
        loaded=joblib.load(out/'best.joblib');loaded.to('cuda:0')
        replay=loaded.predict_proba(raw['dev'])[:,1]
        del loaded
        checkpoint=out/'best.joblib';epochs=limit
    else:
        model,pred,history,best_epoch=tabm_fit(raw,target,out,a.endpoint,a.seed,limit)
        replay=tabm_predict(model,raw['dev']);checkpoint=out/'best.pt';epochs=len(history)
    drift=float(np.max(np.abs(pred-replay)));assert drift<=1e-5
    labels=target['dev'][:,a.endpoint:a.endpoint+1];scores=all_scores(labels,pred[:,None])
    np.savez_compressed(out/'predictions_dev.npz',probs=pred[:,None],labels=labels,ids=data.targets['dev']['ids'],endpoint=a.endpoint)
    atomic(out/'metrics.json',scores)
    reproduction=verify_reference(data,reference,a.endpoint,pred) if a.action=='train' else None
    if reproduction:atomic(out/'reproduction.json',reproduction)
    result=dict(complete=True,pilot=a.action!='train',**identity,dev_ap=scores['mean_ap'],metrics=scores,
                best_epoch=best_epoch,epochs=epochs,seconds=time.time()-started,finished_at=time.time(),
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,checkpoint_bytes=checkpoint.stat().st_size,
                checkpoint_file=checkpoint.name,checkpoint_sha256=sha(checkpoint),predictions_sha256=sha(out/'predictions_dev.npz'),
                checkpoint_replay=dict(complete=True,max_probability_drift=drift),reproduction=reproduction,
                effective_batch=256,selected_microbatch=256,evaluation_microbatch=256,
                probe_scope='two full epochs with short pilot schedule; formal initialization starts again' if a.action=='probe' else None)
    atomic(out/'completion.json',result)
    atomic(out/'progress.json',dict(status='complete',epochs=epochs,best_epoch=best_epoch,best_ap=scores['mean_ap']))


if __name__=='__main__':main()
