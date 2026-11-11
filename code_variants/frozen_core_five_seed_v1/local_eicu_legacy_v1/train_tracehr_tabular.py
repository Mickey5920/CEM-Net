"""Official strong tabular baselines on the frozen F2 train/dev inputs.

PyTabKit supplies RealMLP's entire preprocessing/model/optimizer/training loop.
The only extension is exact AP for validation checkpoint selection. TabM uses
the official package, independent member losses and mean-probability inference.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
for folder in [ROOT/'vendor', ROOT/'work/www2027_rep_screen_deps_20260929']:
    for wheel in folder.glob('*.whl'):
        sys.path.insert(0, str(wheel))
deps = os.environ.get('TRACEHR_TABULAR_DEPS')
if deps:
    sys.path.insert(0, deps)
    for wheel in Path(deps).glob('*.whl'):
        sys.path.insert(0, str(wheel))

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import QuantileTransformer
import torch
from torch import nn
from scripts.build_tracehr_retention import atomic, sha
from train_tracehr_tree_gap import inputs
from train_tracehr_same_features import scores


RECIPES = {
    'tabm': [dict(width=512, depth=3, k=32, dropout=.1, lr=.002, wd=.0003),
             dict(width=256, depth=3, k=32, dropout=.2, lr=.0005, wd=.001)],
    'realmlp': [dict(num_emb_type='pbld'), dict(num_emb_type='none')],
    'xgboost': [dict(max_depth=6, learning_rate=.08, min_child_weight=5e-6,
                     reg_lambda=0., subsample=.65, colsample_bylevel=.9, colsample_bytree=1.),
                dict(max_depth=4, learning_rate=.04, min_child_weight=10,
                     reg_lambda=10., subsample=.8, colsample_bylevel=1., colsample_bytree=.8)],
}


def dense(x, extra):
    result = {}
    for split in ['train', 'dev']:
        z = np.empty((x[split].shape[0], x[split].shape[1]+extra[split].shape[1]), dtype=np.float32)
        z[:, :x[split].shape[1]] = x[split].toarray()
        z[:, x[split].shape[1]:] = extra[split]
        result[split] = z
    return result


def tabm_preprocess(data, phi_dim, out):
    # Preserve sparse-block zero semantics and binary indicators; no vocabulary
    # selection. Robust clipping is fitted blockwise on TRAIN ONLY. Continuous
    # numeric summaries receive a separate train-fitted quantile transform.
    train = data['train']
    center = np.zeros(phi_dim, dtype=np.float32)
    factor = np.ones(phi_dim, dtype=np.float32)
    binary = np.zeros(phi_dim, dtype=bool)
    for start in range(0, phi_dim, 256):
        stop = min(start+256, phi_dim)
        block = train[:, start:stop]
        q1, med, q3 = np.quantile(block, [.25, .5, .75], axis=0)
        lo, hi = block.min(0), block.max(0)
        spread = q3-q1
        spread[spread == 0] = .5*(hi-lo)[spread == 0]
        scale = np.divide(1., spread, out=np.zeros_like(spread), where=spread != 0)
        center[start:stop], factor[start:stop] = med, scale
        binary[start:stop] = np.all((block == 0) | (block == 1), axis=0)
    qt = QuantileTransformer(n_quantiles=min(1000, len(train)), output_distribution='normal',
                             subsample=min(10000, len(train)), random_state=42, copy=True)
    qt.fit(train[:, phi_dim:])
    joblib.dump(dict(phi_center=center, phi_factor=factor, binary=binary, numeric_quantile=qt,
                     phi_dim=phi_dim, fit_split='train'), out/'preprocessing.joblib')
    for split, z in data.items():
        for start in range(0, len(z), 1024):
            block = z[start:start+1024, :phi_dim]
            v = (block-center)*factor
            v /= np.sqrt(1+(v/3)**2)
            v[:, binary] = block[:, binary]
            z[start:start+1024, :phi_dim] = v
        z[:, phi_dim:] = qt.transform(z[:, phi_dim:])
        assert np.isfinite(z).all()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--values',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=list(RECIPES),required=True)
    p.add_argument('--config-index',type=int,choices=[0,1],required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--probe-full',action='store_true')
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--gpu-limit-gib',type=float,default=30)
    a=p.parse_args();assert not(a.pilot and a.probe_full)
    out=a.output;assert not out.exists();out.mkdir(parents=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mpl_cache'))
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.device.startswith('cuda'):
        torch.cuda.set_device(a.device)
        torch.cuda.set_per_process_memory_fraction(a.gpu_limit_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory)
        torch.cuda.manual_seed_all(a.seed);torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    meta,x,extra,target=inputs(a.cache,a.values,'F2',a.pilot)
    phi_dim=x['train'].shape[1]
    data=dense(x,extra);del x,extra;gc.collect()
    y,dy,tasks=target['train']['y'],target['dev']['y'],meta['tasks']
    cfg=RECIPES[a.family][a.config_index]
    short=a.pilot or a.probe_full
    limit=2 if short else 256
    manifest=dict(family=a.family,role=a.family,config_index=a.config_index,recipe=cfg,
                  setting=meta['setting'],seed=a.seed,pilot=short,input_pilot=a.pilot,
                  probe_full=a.probe_full,test_loaded=False,feature_view='F2',
                  cache_sha256=sha(a.cache/'completion.json'),values_sha256=sha(a.values/'completion.json'),
                  source_sha256=sha(Path(__file__)),input_shape={s:list(v.shape) for s,v in data.items()},
                  independent_endpoints=True,precision='FP32',class_weighting='none',
                  checkpoint_selection='exact endpoint dev AP',epochs=limit,
                  gpu_limit_gib=a.gpu_limit_gib,scope='development screen only; not independent confirmation')
    if a.family=='tabm':
        manifest.update(implementation='tabm 0.0.3 official model and recommended AdamW; project blockwise preprocessing; no numerical embeddings',
                        batch=256,patience=16,member_loss='mean of independent binary cross-entropies',
                        prediction='mean sigmoid probability')
    elif a.family=='realmlp':
        manifest.update(implementation='PyTabKit 1.7.3 RealMLP_TD_Classifier; cfg1 explicitly disables numerical embeddings',
                        probability_task_option='use_ls=False as recommended by authors',batch=256,
                        validation_adapter='exact sklearn AP replaces only validation metric; training code unmodified',
                        early_stopping=False)
    else:
        manifest.update(implementation='XGBoost 3.2.0 CPU histogram',threads=4,
                        input_storage='dense; zero is observed zero, missing=NaN',trees=20 if short else 1000,patience=50)
    atomic(out/'run_manifest.json',manifest)
    if a.family=='tabm':
        tabm_preprocess(data,phi_dim,out)
    history=[];details=[];prob=np.zeros_like(dy,dtype=np.float64)
    endpoints=[0] if a.probe_full else list(range(len(tasks)))
    for j in endpoints:
        endpoint_start=time.monotonic();torch.manual_seed(a.seed);np.random.seed(a.seed)
        if a.family=='xgboost':
            import xgboost
            cb=xgboost.callback.EarlyStopping(rounds=50,metric_name='average_precision_score',
                                               data_name='validation_0',maximize=True,save_best=True)
            model=xgboost.XGBClassifier(n_estimators=20 if short else 1000,tree_method='hist',
                    max_bin=256,n_jobs=4,random_state=a.seed,eval_metric=average_precision_score,
                    callbacks=[cb],**cfg)
            model.fit(data['train'],y[:,j],eval_set=[(data['dev'],dy[:,j])],verbose=False)
            prob[:,j]=model.predict_proba(data['dev'])[:,1]
            model.save_model(out/f'best_task{j}.ubj')
            h=model.evals_result()['validation_0']['average_precision_score']
            history.extend([dict(endpoint=j,epoch=i+1,dev_ap=v) for i,v in enumerate(h)])
            best_epoch,epochs=int(model.best_iteration)+1,len(h)
        elif a.family=='realmlp':
            # Pin the tested PyTorch Lightning backend even on machines that
            # also contain the unrelated legacy lightning.app distribution.
            sys.modules['lightning']=None
            from pytabkit import RealMLP_TD_Classifier
            from pytabkit.models.training.metrics import Metrics
            original=Metrics.apply
            def exact_ap(logits,labels,name):
                if name!='tracehr_1_ap':
                    return original(logits,labels,name)
                assert logits.ndim==2 and logits.shape[-1]==2
                assert torch.isfinite(logits).all(), 'Nonfinite RealMLP logits'
                score=float(average_precision_score(labels.detach().cpu().numpy().reshape(-1),
                                    torch.softmax(logits,-1)[:,1].detach().cpu().numpy()))
                history.append(dict(endpoint=j,validation_event=sum(v['endpoint']==j for v in history)+1,
                                    dev_ap=score,seconds=time.monotonic()-endpoint_start))
                atomic(out/'history.json',history)
                return logits.new_tensor(1-score)
            Metrics.apply=staticmethod(exact_ap)
            try:
                model=RealMLP_TD_Classifier(device=a.device,random_state=a.seed,n_cv=1,n_refit=0,
                    n_threads=4,tmp_folder=str(out/'tmp'),verbosity=2,n_epochs=limit,batch_size=256,
                    predict_batch_size=256,val_metric_name='tracehr_1_ap',use_ls=False,
                    use_early_stopping=False,**cfg)
                atomic(out/f'effective_config_task{j}.json',model.get_config())
                model.fit(data['train'],y[:,j].astype(np.int64),X_val=data['dev'],y_val=dy[:,j].astype(np.int64))
                prob[:,j]=model.predict_proba(data['dev'])[:,1]
                fit=model.cv_alg_interface_.fit_params
                atomic(out/f'fit_params_task{j}.json',fit)
                best_epoch=fit[0]['stop_epoch']['tracehr_1_ap'];epochs=limit
                model.to('cpu')
                joblib.dump(model,out/f'best_task{j}.joblib')
            finally:
                Metrics.apply=staticmethod(original)
        else:
            from tabm import TabM
            model=TabM.make(n_num_features=data['train'].shape[1],d_out=1,k=cfg['k'],
                           d_block=cfg['width'],n_blocks=cfg['depth'],dropout=cfg['dropout']).to(a.device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
            rng=np.random.default_rng(a.seed);best=-1.;stale=0
            for epoch in range(1,limit+1):
                tick=time.monotonic();model.train();order=rng.permutation(len(y));loss_sum=0.
                for start in range(0,len(order),256):
                    idx=order[start:start+256]
                    bx=torch.as_tensor(data['train'][idx],device=a.device)
                    by=torch.as_tensor(y[idx,j],device=a.device,dtype=torch.float32)
                    optimizer.zero_grad(set_to_none=True)
                    logits=model(bx).squeeze(-1)
                    loss=nn.functional.binary_cross_entropy_with_logits(logits,by[:,None].expand_as(logits))
                    assert torch.isfinite(loss)
                    loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(idx)
                model.eval();parts=[]
                with torch.no_grad():
                    for start in range(0,len(dy),256):
                        logits=model(torch.as_tensor(data['dev'][start:start+256],device=a.device))
                        parts.append(torch.sigmoid(logits).mean(1).squeeze(-1).cpu().numpy())
                pred=np.concatenate(parts);score=float(average_precision_score(dy[:,j],pred))
                rec=dict(endpoint=j,epoch=epoch,dev_ap=score,train_loss=loss_sum/len(y),seconds=time.monotonic()-tick)
                history.append(rec);atomic(out/'history.json',history);print(json.dumps(rec),flush=True)
                if score>best+1e-9:
                    best=score;stale=0;best_epoch=epoch;prob[:,j]=pred
                    torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                            manifest=manifest,endpoint=j,epoch=epoch,numpy_rng_state=rng.bit_generator.state,
                            torch_rng_state=torch.random.get_rng_state(),
                            cuda_rng_state=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else []),out/f'best_task{j}.pt')
                else:
                    stale+=1
                if stale>=16:break
            epochs=epoch
            del optimizer
        details.append(dict(endpoint=j,best_epoch=int(best_epoch),epochs=int(epochs),seconds=time.monotonic()-endpoint_start))
        atomic(out/'history.json',history);del model;gc.collect()
        if a.device.startswith('cuda'):torch.cuda.empty_cache()
    assert np.isfinite(prob).all()
    peak=torch.cuda.max_memory_reserved()/2**20 if a.device.startswith('cuda') else 0.
    elapsed=time.monotonic()-started
    # A full-geometry probe uses one endpoint; placeholder columns are never
    # scored or admitted to scientific result tables.
    metrics=scores(dy,prob,tasks) if not a.probe_full else {'aggregate':{'mean_primary_auprc':None}}
    np.savez_compressed(out/'predictions_dev.npz',probs=prob,labels=dy,ids=target['dev']['ids'],tasks=np.asarray(tasks))
    atomic(out/'metrics.json',dict(splits={'dev':metrics},models=details,elapsed_wall_sec=elapsed,peak_reserved_mib=peak))
    if a.probe_full:
        epoch_times=[v.get('seconds',0) for v in history if v.get('endpoint')==0]
        per_epoch=((epoch_times[-1]-epoch_times[-2]) if a.family=='realmlp' and len(epoch_times)>1 else
                   max(epoch_times[-2:],default=elapsed))
        projected=elapsed+max(0,256-2)*per_epoch
        atomic(out/'performance.json',dict(full_shape=manifest['input_shape'],peak_reserved_mib=peak,
             endpoint_epoch_seconds=per_epoch,projected_two_endpoint_seconds=2*projected,
             estimate_scope='two full epochs; upper-bound max epochs, excluding shared-load variability'))
    atomic(out/'completion.json',dict(complete=True,role=a.family,setting=meta['setting'],config_index=a.config_index,
          pilot=short,input_pilot=a.pilot,probe_full=a.probe_full,test_loaded=False,seed=a.seed,
          dev_ap=metrics['aggregate']['mean_primary_auprc'],best_epoch=[v['best_epoch'] for v in details],
          epochs=[v['epochs'] for v in details],elapsed_wall_sec=elapsed,peak_reserved_mib=peak,
          artifact_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())))


if __name__=='__main__':
    main()
