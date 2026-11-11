"""Audited, paired summaries of the fixed 72-fit batch; no model fitting here."""
import argparse,fcntl,gc,json,math,os,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
from sklearn.linear_model import LogisticRegression

RUN='cem_icde_evidence_20261028';OLD_RUN='cem_evidence_completion_20261025'
SEEDS=(48,49,50);B=20000
BASELINES=('RealMLP','TabM','TabM-mini','TabM_mini_PLE','TabM_PLE','SeFT','GRU_D','mTAND_Enc','LightGBM_full')

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def json_scalar(x):
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.floating):return float(x)
    raise TypeError(f'Unsupported summary value: {type(x).__name__}')
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.writing');q.write_text(json.dumps(x,indent=2,default=json_scalar));os.replace(q,p)
def summary(v):
    a=np.asarray(v,float);return dict(values=a.tolist(),mean=float(a.mean()),sample_SD=float(a.std(ddof=1)),positive_seeds=int((a>0).sum()))
def prepared_ap(y,q):
    order=np.argsort(-q,kind='stable');ends=np.r_[np.flatnonzero(np.diff(q[order])),len(q)-1]
    return order,ends,y[order].astype(float)
def weighted_ap(prepared,weights):
    order,ends,y=prepared;w=weights[:,order];tp=np.cumsum(w*y,axis=1)[:,ends];total=np.cumsum(w,axis=1)[:,ends]
    assert (tp[:,-1]>0).all();inc=np.diff(np.c_[np.zeros(len(w)),tp],axis=1)
    return (np.divide(tp,total,out=np.zeros_like(tp),where=total>0)*inc).sum(1)/tp[:,-1]
def scores(y,q):return dict(AP=float(average_precision_score(y,q)),AUROC=float(roc_auc_score(y,q)),Brier=float(brier_score_loss(y,q)),log_loss=float(log_loss(y,q,labels=[0,1])))
def probability_metrics(y,q):
    edges=np.linspace(0,1,16);groups=np.minimum(np.searchsorted(edges,q,side='right')-1,14);rows=[];ece=0.
    for j in range(15):
        ix=groups==j
        if ix.any():
            observed=float(y[ix].mean());predicted=float(q[ix].mean());n=int(ix.sum());ece+=n/len(y)*abs(observed-predicted)
            rows.append(dict(bin=j,count=n,observed=observed,predicted=predicted))
    return dict(**scores(y,q),ECE15=ece,reliability=rows,prevalence=float(y.mean()),mean_probability=float(q.mean()))
def utility(y,q,ids):
    order=np.lexsort((ids[:,2],-q));result={}
    for fraction in (.05,.10):
        k=int(math.ceil(len(y)*fraction));tp=int(y[order[:k]].sum())
        result[str(fraction)]=dict(reviewed=k,true_positives=tp,precision=tp/k,recall=tp/y.sum(),true_positives_per_1000=tp/len(y)*1000)
    return result
def self_test():
    y=np.array([1,0,1,0,1,0]);q=np.array([.8,.8,.5,.2,.2,.1]);w=np.array([[1,1,1,1,1,1],[0,3,2,1,2,0],[2,0,0,4,1,3]],float)
    np.testing.assert_allclose(weighted_ap(prepared_ap(y,q),w),[average_precision_score(y,q,sample_weight=z) for z in w],rtol=0,atol=1e-14)
    payload={'nested':[np.float32(.25),np.int64(3),np.bool_(True)]}
    assert json.loads(json.dumps(payload,default=json_scalar))=={'nested':[.25,3,True]}
    try:json.dumps({'unexpected_array':np.array([1.])},default=json_scalar)
    except TypeError:pass
    else:raise AssertionError('Unexpected arrays must not be silently exported')
    return ['weighted_AP_ties_and_zero_weights_match_sklearn','fixed_three_seed_average_without_probability_pooling','nested_numpy_summary_scalars_serialize_without_array_export']

def analyze(project,db,require_bootstrap_cache=False):
    out=project/'runs'/RUN/(db+'_statistics_v1');out.mkdir(parents=True,exist_ok=True)
    owner=(out/'writer.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'completion.json').exists():
        done=read(out/'completion.json');assert done['complete']
        for n,h in done['files'].items():assert sha(out/n)==h
        return dict(complete=True,reused=True,receipt=done)
    host=project/'runs'/RUN/('host_eicu' if db=='eicu' else 'host_mimic');state=read(host/'state.json')
    exitfile=Path(state.get('active_scheduler_receipt',host/'start_exit.json'));exitrecord=read(exitfile);assert exitrecord['returncode']==0
    groups=[g for g in state['groups'].values() if g['database']==db]
    expected_fits=36 if db=='eicu' else 18;assert sum(len(g['matrix']) for g in groups)==expected_fits
    assert all(g['status']=='complete' for g in groups)
    oldpath=project/'runs'/OLD_RUN/db/'comparison_v1/verified-results.json'
    old=read(oldpath);assert old['complete'] and old['verified_new_fits']==old['verified_fixed_tests']==48
    artifacts={str(oldpath):sha(oldpath),str(exitfile):sha(exitfile)};targets={};refs={};preds={};rows=[];audits=[]
    for split in ('dev','test'):
        for method in ('CEM-Net','EMA_NRE_MLP',*BASELINES[:-1]):
            meta=read(oldpath.parent/'bootstrap_cache'/(split+'_'+method.replace('-','_')+'.json'))['identity']
            tsha=meta['target_sha256'];target=next(Path(n) for n,h in old['artifacts'].items() if h==tsha and n.endswith(split+'_targets.npz'))
            assert sha(target)==tsha;targets['random',split]=dict(np.load(target));artifacts[str(target)]=tsha
            for s in SEEDS:
                for e in (0,1):refs['random',split,method,s,e]=meta['predictions'][f'{s}:{e}']
    for g in groups:
        source=Path(g['source']);assert sha(source/'source_manifest.json')==g['source_manifest_sha256']
        for n,h in read(source/'source_manifest.json').items():assert sha(source/n)==h,n
        cfg=read(source/'work'/RUN/'config.json');screen=Path(g['output']);lock=read(screen/'fixed-checkpoint-lock.json')
        assert lock['complete'] and not lock['test_used_for_selection'] and len(lock['selections'])==len(g['matrix'])
        arm='hospital' if g['stage']=='hospital' else 'random'
        if arm=='hospital':
            for split in ('dev','test'):targets[arm,split]=dict(np.load(project/cfg['raw_inputs']/(split+'_targets.npz')))
        for j in g['jobs'].values():
            if j['action']!='train':continue
            fit=Path(j['output']);testjob=g['jobs']['test_'+j['id']];test=Path(testjob['output'])
            for job in (j,testjob):assert job['status']=='completed' and read(job['exit_receipt'])['returncode']==0
            r=read(fit/'completion.json');tc=read(test/'completion.json');identity=read(fit/'run_manifest.json');history=read(fit/'history.json')
            assert r['complete'] and r['from_scratch'] and not r['pilot'] and not r['test_loaded']
            assert identity['source_manifest_sha256']==g['source_manifest_sha256']
            h=sha(fit/r['checkpoint_file']);assert h==r['checkpoint_sha256']==lock['selections'][j['id']]['checkpoint_sha256']==tc['model_sha256']
            assert tc['complete'] and tc['optimizer_updates']==0 and not tc['parameters_updated'] and tc['dev_replay_max_drift']<=1e-5
            assert r['checkpoint_replay']['max_probability_drift']<=1e-5
            if j['variant'].startswith('CEM'):
                assert len(history)==r['epochs']==identity['epoch_budget'] and r['parameter_source']=='EMA'
                assert r['best_epoch']==1+int(np.argmax([z['dev_ap'] for z in history]))
                assert r['optimizer_updates']==r['ema_updates']==r['epochs']*math.ceil(cfg['train_rows']/256)
                obs=read(fit/'initialization_observation.json');assert obs['full_initial_tensors_exact'] and obs['deletion']['retained_initials_exact']
                assert obs['initial_weights_sha256']==identity['initial_weights_sha256']
                import torch
                for ep in (1,5,8,10,12,15,18,20):
                    d=read(fit/f'epoch_{ep:03d}_diagnostics.json');assert sha(fit/f'epoch_{ep:03d}_model.pt')==d['checkpoint_sha256']
                    assert all(z['main_connected']==z['tensors'] and z['aux_connected']==0 for z in d['gradient']['groups']['loss_paths']['groups'].values())
                first=torch.load(fit/'epoch_001_model.pt',map_location='cpu',weights_only=False)['model'];last=torch.load(fit/'epoch_020_model.pt',map_location='cpu',weights_only=False)['model']
                assert all(not torch.equal(first[n],last[n]) for n in obs['retained_tensor_hashes']);del first,last;gc.collect()
            elif j['variant'].startswith('TabM'):
                assert 1<=len(history)<=identity['protocol']['epochs']
                best=-1.;selected=0;stale=0
                for k,z in enumerate(history,1):
                    if z['dev_ap']>best+1e-9:best=z['dev_ap'];selected=k;stale=0
                    else:stale+=1
                    assert stale<identity['protocol']['patience'] or k==len(history)
                assert selected==r['best_epoch'] and (len(history)==identity['protocol']['epochs'] or stale==identity['protocol']['patience'])
                audit=read(fit/('full_input_audit.json' if arm=='hospital' else 'evidence_mask_audit.json'));assert audit['complete']
            else:
                assert identity['protocol']['max_rounds']==1000 and identity['protocol']['early_stopping_rounds']==300
                assert r['best_epoch']==1+int(np.argmax([z['dev_ap'] for z in history]))
                assert len(history)==1000 or len(history)-r['best_epoch']==300
            audits.append(dict(arm=arm,id=j['id'],fit_exit=0,test_exit=0,checkpoint_sha256=h,fit_seconds=r['seconds'],epochs=r['epochs'],
                test_updates=0,source_manifest_sha256=g['source_manifest_sha256'],parameters=identity.get('parameters'),peak_reserved_mib=r.get('peak_reserved_mib'),
                completion_sha256=sha(fit/'completion.json'),test_completion_sha256=sha(test/'completion.json')))
            for split,folder,receipt in (('dev',fit,r),('test',test,tc)):
                refs[arm,split,j['variant'],j['seed'],j['endpoint']]=dict(path=str(folder/f'predictions_{split}.npz'),sha256=receipt['predictions_sha256'])
    oldlookup={(r['split'],r['method'],r['seed'],r['endpoint']):r for r in old['rows']}
    for key,ref in refs.items():
        arm,split,method,s,e=key;assert sha(ref['path'])==ref['sha256'];artifacts[ref['path']]=ref['sha256']
        z=dict(np.load(ref['path']));t=targets[arm,split];np.testing.assert_array_equal(z['ids'],t['ids'])
        col=e if z['labels'].ndim==2 and z['labels'].shape[1]==2 else 0
        y=z['labels'] if z['labels'].ndim==1 else z['labels'][:,col];q=z['probs'] if z['probs'].ndim==1 else z['probs'][:,col]
        np.testing.assert_array_equal(y,t['y'][:,e]);assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
        value=scores(y,q);preds[key]=q;rows.append(dict(arm=arm,split=split,method=method,seed=s,endpoint=e,**value))
        if (split,method,s,e) in oldlookup:assert abs(value['AP']-oldlookup[split,method,s,e]['AP'])<1e-12
    lookup={(r['arm'],r['split'],r['method'],r['seed'],r['endpoint']):r for r in rows}
    summaries={};selection={}
    for arm,split,method,_,e in refs:
        key=(arm,split,method,e)
        if key not in summaries:summaries[key]={metric:summary([lookup[arm,split,method,s,e][metric] for s in SEEDS]) for metric in ('AP','AUROC','Brier','log_loss')}
    for e in (0,1):selection[e]=max(BASELINES,key=lambda m:summaries['random','dev',m,e]['AP']['mean'])
    save(out/'validation-selection.json',dict(complete=True,baseline_pool=list(BASELINES),by_endpoint=selection,criterion='three-seed Validation AP mean only',test_used_for_selection=False))
    cache=out/'bootstrap';cache.mkdir(exist_ok=True);draws={};clusterinfo={}
    for (arm,split),t in targets.items():
        if arm=='random':cluster=t['ids'][:,0]
        else:
            # A patient visiting multiple hospitals connects their hospitals.
            parent={int(h):int(h) for h in np.unique(t['hospital_ids'])}
            def find(x):
                while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
                return x
            for patient in np.unique(t['ids'][:,0]):
                hs=np.unique(t['hospital_ids'][t['ids'][:,0]==patient]);a=find(int(hs[0]))
                for h in hs[1:]:parent[find(int(h))]=a
            cluster=np.asarray([find(int(h)) for h in t['hospital_ids']])
        unique,inverse=np.unique(cluster,return_inverse=True);n=len(unique);clusterinfo[arm+':'+split]=dict(clusters=n,unit='patient' if arm=='random' else 'patient-connected hospital component')
        assert n>=2
        for method in sorted({k[2] for k in refs if k[:2]==(arm,split)}):
            cells=[(s,e) for s in SEEDS for e in (0,1) if (arm,split,method,s,e) in refs]
            identity=dict(arm=arm,split=split,method=method,database=db,replicates=B,rng_seed=20261026+(split=='test'),
                cluster_ids_sha256=__import__('hashlib').sha256(cluster.tobytes()).hexdigest(),predictions={f'{s}:{e}':refs[arm,split,method,s,e] for s,e in cells})
            stem=arm+'_'+split+'_'+method.replace('-','_');mp=cache/(stem+'.json');dp=cache/(stem+'.npz')
            if mp.exists():
                meta=read(mp);assert meta['identity']==identity and sha(dp)==meta['draws_sha256'];samples=dict(np.load(dp))
            else:
                assert not require_bootstrap_cache,'Recovery requires every existing bootstrap cache: '+str(mp)
                samples={f'{s}:{e}':np.empty(B) for s,e in cells};prepared={(s,e):prepared_ap(t['y'][:,e],preds[arm,split,method,s,e]) for s,e in cells}
                oldmp=oldpath.parent/'bootstrap_cache'/(split+'_'+method.replace('-','_')+'.json');start_at=0
                if arm=='random' and oldmp.exists():
                    meta=read(oldmp);od=oldmp.with_suffix('.npz');assert sha(od)==meta['draws_sha256']
                    olddraw=dict(np.load(od));assert meta['identity']['replicates']==4000
                    for s,e in cells:
                        assert meta['identity']['predictions'][f'{s}:{e}']==refs[arm,split,method,s,e]
                        samples[f'{s}:{e}'][:4000]=olddraw[f'{s}:{e}']
                    start_at=4000
                rng=np.random.default_rng(identity['rng_seed'])
                for start in range(0,B,16):
                    weights=np.stack([np.bincount(rng.integers(0,n,n),minlength=n)[inverse] for _ in range(min(16,B-start))]).astype(float)
                    if start<start_at:
                        if start==0:
                            for cell,pair in prepared.items():np.testing.assert_allclose(weighted_ap(pair,weights),samples[f'{cell[0]}:{cell[1]}'][:len(weights)],rtol=0,atol=1e-12)
                        continue
                    for (s,e),pair in prepared.items():samples[f'{s}:{e}'][start:start+len(weights)]=weighted_ap(pair,weights)
                    if start%320==0:save(out/'progress.json',dict(stage='paired_bootstrap',arm=arm,split=split,method=method,completed=start+len(weights),total=B,time=time.time()))
                tmp=dp.with_suffix('.writing')
                with tmp.open('wb') as stream:np.savez_compressed(stream,**samples)
                os.replace(tmp,dp);save(mp,dict(identity=identity,draws_sha256=sha(dp),old_4000_reused=start_at==4000))
            for s,e in cells:draws[arm,split,method,s,e]=samples[f'{s}:{e}']
    def contrast(arm,split,e,coefficients,family):
        values=[sum(c*lookup[arm,split,m,s,e]['AP'] for m,c in coefficients.items()) for s in SEEDS]
        sample=np.mean([sum(c*draws[arm,split,m,s,e] for m,c in coefficients.items()) for s in SEEDS],axis=0);a=.05/family
        return dict(summary(values),coefficients=coefficients,family_count=family,ci95=np.quantile(sample,[.025,.975]).tolist(),adjusted_ci=np.quantile(sample,[a/2,1-a/2]).tolist())
    paired={}
    strongest={}
    for split in ('dev','test'):
        paired[split]=dict(candidates={m:{b:{str(e):contrast('random',split,e,{m:1,b:-1},54) for e in (0,1)} for b in BASELINES} for m in ('CEM-Net','EMA_NRE_MLP')},
            MLP_minus_CEM={str(e):contrast('random',split,e,{'EMA_NRE_MLP':1,'CEM-Net':-1},6) for e in (0,1)},evidence={})
        strongest[split]={}
        for e in (0,1):
            basic=next(m for a,sp,m,s,ee in refs if a=='random' and sp==split and ee==e and m.startswith('TabM') and m.endswith('_basic'))
            full=basic.removesuffix('_basic');full='TabM-mini' if full=='TabM_mini' else full
            paired[split]['evidence'][str(e)]=dict(CEM_full_minus_basic=contrast('random',split,e,{'CEM-Net':1,'CEM_basic':-1},18),
                baseline_full_minus_basic=contrast('random',split,e,{full:1,basic:-1},18),
                interaction=contrast('random',split,e,{'CEM-Net':1,'CEM_basic':-1,full:-1,basic:1},18),preselected_baseline=full)
            best=max(BASELINES,key=lambda m:summaries['random',split,m,e]['AP']['mean'])
            strongest[split][str(e)]=dict(mean_strongest=best,validation_selected=selection[e],by_seed=[max(BASELINES,key=lambda m:lookup['random',split,m,s,e]['AP']) for s in SEEDS])
        if db=='eicu':paired[split]['hospital']={b:{str(e):contrast('hospital',split,e,{'CEM_hospital':1,b:-1},4) for e in (0,1)} for b in ('TabM_mini_PLE_hospital','LightGBM_hospital')}
    calibration=[];utilityrows=[];models={};counts=0
    for e in (0,1):
        for method in ('CEM-Net',selection[e]):
            for s in SEEDS:
                q=preds['random','dev',method,s,e];y=targets['random','dev']['y'][:,e]
                logit=lambda p:np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))[:,None]
                calibrator=LogisticRegression(penalty=None,solver='lbfgs',max_iter=1000,tol=1e-10).fit(logit(q),y);assert calibrator.n_iter_[0]<1000;counts+=1
                models[f'{method}:{s}:{e}']=dict(coef=float(calibrator.coef_[0,0]),intercept=float(calibrator.intercept_[0]),fit_split='Validation',regularization='none')
                for split in ('dev','test'):
                    t=targets['random',split];q=preds['random',split,method,s,e];y=t['y'][:,e];cal=calibrator.predict_proba(logit(q))[:,1]
                    calibration.append(dict(split=split,method=method,seed=s,endpoint=e,raw=probability_metrics(y,q),Platt=probability_metrics(y,cal),validation_fit_resubstitution=(split=='dev')))
                    utilityrows.append(dict(split=split,method=method,seed=s,endpoint=e,**utility(y,q,t['ids'])))
    assert counts==12
    result=dict(complete=True,database=db,verified_new_fits=expected_fits,verified_fixed_tests=expected_fits,host_exit=exitrecord,
        source_sha256=sha(__file__),rows=rows,summaries={':'.join(map(str,k)):v for k,v in summaries.items()},paired=paired,strongest=strongest,
        baseline_pool=list(BASELINES),original_method_retained=True,main_method_switched=False,validation_test_separate=True,
        bootstrap=dict(replicates=B,clusters=clusterinfo,common_draws=True,seed_probability_pooling=False,prior_4000_reused=True,
            candidate_baseline_family=54,evidence_family=18,hospital_family=4,self_test=self_test(),
            limitation='Conditional on the three fixed fits and observed cohorts; does not include training-population or adaptive-selection uncertainty. Extreme family-adjusted quantiles retain Monte Carlo error.'),
        calibration_fits=counts,calibrators=models,calibration=calibration,review_budget_utility=utilityrows,worker_audits=audits,
        old_results_retained=dict(path=str(oldpath),sha256=sha(oldpath)),artifacts=artifacts,patient_records_exported=False,
        outstanding=['fixed corruption inference','coverage and demographic/LOS subgroup analyses','matched-hardware pipeline and scaling measurements','manuscript integration'])
    save(out/'verified-results.json',result)
    lines=['# '+db+'：新增固定比较与配对统计','','主方法保持原完整EMA CEM-Net；以下统计包含全部预定比较。','']
    for arm in ('random','hospital'):
        if arm=='hospital' and db!='eicu':continue
        for split in ('dev','test'):
            lines+=['## '+arm+' / '+('Validation' if split=='dev' else 'Test'),'','|方法|终点|seed48|seed49|seed50|AP均值±样本SD|','|---|---|---:|---:|---:|---:|']
            for (a,sp,m,e),v in summaries.items():
                if (a,sp)!=(arm,split):continue
                r=v['AP'];lines.append('|'+m+'|'+('死亡' if e==0 else 'LOS')+'|'+'|'.join(f'{x:.6f}' for x in r['values'])+f"|{r['mean']:.6f}±{r['sample_SD']:.6f}|")
    lines+=['','配对差、95%及族校正条件区间、原始与Platt校准、复核预算指标见同目录verified-results.json。Validation校准为拟合集上的描述性结果，Test校准未再拟合。',
        '本文件完成的是固定比较与统计；扰动、分层、统一硬件成本及稿件整合需单独完成，不将本收集完成标记解释为整项论文任务完成。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    done=dict(complete=True,stage='fixed-comparison-and-statistics',overall_paper_task_complete=False,files={n:sha(out/n) for n in ('verified-results.json','RESULTS.md','validation-selection.json')})
    save(out/'completion.json',done);save(out/'progress.json',dict(stage='complete',time=time.time()));return dict(complete=True,reused=False,receipt=done)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--database',choices=['mimiciii','mimic','eicu'],required=True);p.add_argument('--self-test',action='store_true');p.add_argument('--require-bootstrap-cache',action='store_true');a=p.parse_args()
    print(json.dumps(self_test() if a.self_test else analyze(a.project.resolve(strict=True),a.database,a.require_bootstrap_cache)))
