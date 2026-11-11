"""Paired, zero-update statistics and a pre-test decision for the frozen eICU batch."""
import argparse, fcntl, json, math, os, time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from scripts.cem_observation_common import RUN, ROOT, SEEDS, VARIANTS, EARLY, EMA_RULE, expected, protocol, read, sha, atomic

BASELINES=('RealMLP','TabM','TabM-mini','TabM_mini_PLE','TabM_PLE','SeFT','GRU_D','mTAND_Enc')
OLD=('CEM-Net','EMA_NRE_MLP',*BASELINES)
B=4000

def prepared_ap(y,q):
    order=np.argsort(-q,kind='stable');ends=np.r_[np.flatnonzero(np.diff(q[order])),len(q)-1]
    return order,ends,y[order].astype(float)
def weighted_ap(prepared,weights):
    order,ends,y=prepared;w=weights[:,order]
    tp=np.cumsum(w*y,axis=1)[:,ends];total=np.cumsum(w,axis=1)[:,ends]
    assert (tp[:,-1]>0).all()
    inc=np.diff(np.c_[np.zeros(len(w)),tp],axis=1)
    return (np.divide(tp,total,out=np.zeros_like(tp),where=total>0)*inc).sum(1)/tp[:,-1]
def summary(v):
    a=np.asarray(v,float)
    return dict(values=a.tolist(),mean=float(a.mean()),sample_SD=float(a.std(ddof=1)),positive_seeds=int((a>0).sum()))

def advancement(primary,los,mean,threshold):
    mechanism=all(x['mean']>0 and x['positive_seeds']>=2 and x['adjusted_ci'][0]>0 for x in primary.values())
    protection=all(x['one_sided_975_lower']>-.002 for x in los.values())
    performance=mean>=threshold
    return dict(mechanism_pass=mechanism,mortality_performance_pass=performance,LOS_protection_pass=protection,
                advancement_supported=mechanism and protection and performance,
                next_training_authorized=False,main_method_changed=False)

def cpu_checks():
    y=np.array([1,0,1,0,1,0]);q=np.array([.8,.8,.5,.2,.2,.1])
    w=np.array([[1,1,1,1,1,1],[0,3,2,1,2,0],[2,0,0,4,1,3]],float)
    np.testing.assert_allclose(weighted_ap(prepared_ap(y,q),w),
        [average_precision_score(y,q,sample_weight=x) for x in w],atol=1e-14,rtol=0)
    p={str(k):dict(mean=.01,positive_seeds=2,adjusted_ci=[.001,.02]) for k in range(3)}
    l={str(k):dict(one_sided_975_lower=-.001) for k in range(2)}
    assert advancement(p,l,.5,.4)['advancement_supported']
    p['1']['adjusted_ci'][0]=0
    assert not advancement(p,l,.5,.4)['advancement_supported']
    assert not advancement({},l,.3,.4)['advancement_supported']
    l['1']['one_sided_975_lower']=-.002
    assert not advancement({},l,.5,.4)['advancement_supported']
    return ['weighted_tied_AP_matches_sklearn','strict_CI_and_LOS_boundaries','performance_threshold_gate']

def audit_fits(project,screen,state):
    import torch
    c=expected();records=[]
    fits=[j for j in state['jobs'].values() if j['action']=='train']
    assert len(fits)==18 and all(j['status']=='completed' for j in fits)
    lock=read(screen/'fixed-checkpoint-lock.json');assert len(lock['selections'])==18 and not lock['test_used_for_selection']
    for j in fits:
        out=Path(j['output']);r=read(out/'completion.json');i=read(out/'run_manifest.json')
        obs=read(out/'initialization_observation.json');ref=c['full_ema'][f"{j['seed']}:{j['endpoint']}"]
        assert read(j['exit_receipt'])['returncode']==0 and r['complete'] and not r['pilot']
        assert r['from_scratch'] and not r['test_loaded'] and r['frozen_learnable_parameters']==0
        assert i['protocol']==protocol(j['variant'],j['seed']) and i['ema_rule']==EMA_RULE
        assert i['source_manifest_sha256']==sha(ROOT/'source_manifest.json')
        assert obs['full_initial_weights_sha256']==ref['initial_weights_sha256'] and obs['full_initial_tensors_exact']
        assert obs['original_static_buffers_exact'] and obs['initial_predictions_exact'] and obs['added_parameters']==368
        assert obs['initial_weights_sha256']==i['initial_weights_sha256'] and i['parameters']==ref['parameters']+368
        assert obs['context_normalization_sha256']==sha(project/'cache'/RUN/'context_normalization.json')
        h=read(out/'history.json');assert len(h)==r['epochs']==256
        assert r['best_epoch']==1+int(np.argmax([x['dev_ap'] for x in h]))
        assert r['optimizer_updates']==r['ema_updates']==256*math.ceil(92565/256)
        assert r['parameter_source']=='EMA' and r['main_view']==r['checkpoint_selection_view']=='z0'
        assert sha(out/'best.pt')==r['checkpoint_sha256']==lock['selections'][j['id']]['checkpoint_sha256']
        assert r['checkpoint_replay']['max_probability_drift']<=1e-5
        for ep in EARLY:
            d=read(out/f'epoch_{ep:03d}_diagnostics.json')
            assert sha(out/f'epoch_{ep:03d}_model.pt')==d['checkpoint_sha256']
            assert all(g['main_connected']==g['tensors'] and g['aux_connected']==0 for g in d['gradient']['groups']['loss_paths']['groups'].values())
            assert all(np.isfinite(v) and v>=0 for v in d['gradient']['groups']['relative_update'].values())
        left=torch.load(out/'epoch_001_model.pt',map_location='cpu',weights_only=False)['model']
        right=torch.load(out/'epoch_020_model.pt',map_location='cpu',weights_only=False)['model']
        assert all(not torch.equal(left[n],right[n]) for n in obs['retained_tensor_hashes']), 'Parameter did not change in audit window'
        del left,right
        records.append(dict(id=j['id'],completion_sha256=sha(out/'completion.json'),checkpoint_sha256=r['checkpoint_sha256'],
            initialization_sha256=sha(out/'initialization_observation.json'),history_sha256=sha(out/'history.json'),
            parameters=i['parameters'],seconds=r['seconds'],peak_reserved_mib=r['peak_reserved_mib']))
    return dict(complete=True,fit_count=18,records=records,all_initials_and_static_buffers_exact=True,
        all_retained_parameters_updated=True,early_main_only_gradient_snapshots=18*8,EMA_selection_replay_exact=True)

def analyze(project,screen,split):
    assert split in ('dev','test');c=expected();source_sha=sha(ROOT/'source_manifest.json')
    for n,h in read(ROOT/'source_manifest.json').items(): assert sha(ROOT/n)==h,n
    for n,h in c['protected'].items(): assert sha(n)==h,n
    state=read(screen/'state.json');lock=read(screen/'fixed-checkpoint-lock.json')
    assert lock['complete'] and len(lock['selections'])==18
    oldpath=project/c['comparison_path'];assert sha(oldpath)==c['comparison_sha256']
    old=read(oldpath);assert old['complete'] and old['verified_new_fits']==old['verified_fixed_tests']==48
    out=screen.parent/'comparison';cache=project/'cache'/RUN/'bootstrap';out.mkdir(exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    receiptpath=screen/('validation-statistics.json' if split=='dev' else 'test-statistics.json')
    if receiptpath.exists():
        r=read(receiptpath);assert r['complete'] and r['source_manifest_sha256']==source_sha
        assert r['checkpoint_lock_sha256']==sha(screen/'fixed-checkpoint-lock.json')
        return r
    if split=='dev':
        assert all(j['status']=='pending' for j in state['jobs'].values() if j['action']=='evaluate')
        audit=audit_fits(project,screen,state);atomic(screen/'training-audit.json',audit)
    else:
        decision=read(screen/'validation-decision.json');assert decision['complete'] and not decision['test_loaded']
        assert decision['statistics_sha256']==sha(screen/'validation-statistics.json')
        audit=read(screen/'training-audit.json');assert audit['complete']
        for record in audit['records']:
            fit=Path(state['jobs'][record['id']]['output'])
            assert sha(fit/'completion.json')==record['completion_sha256'] and sha(fit/'best.pt')==record['checkpoint_sha256']
    cpu_checks();rows=[];aps={};draws={};preds={};refs={};artifacts={};meta={}
    oldlookup={(r['split'],r['method'],r['seed'],r['endpoint']):r for r in old['rows']}
    for method in OLD:
        mp=oldpath.parent/'bootstrap_cache'/(split+'_'+method.replace('-','_')+'.json');m=read(mp)
        dp=mp.with_suffix('.npz');assert sha(dp)==m['draws_sha256']
        assert m['identity']['replicates']==B and m['identity']['rng_seed']==20261026+(split=='test')
        assert m['identity']['database']=='eicu' and m['identity']['split']==split and m['identity']['method']==method
        meta[method]=m;draws[method]=dict(np.load(dp,allow_pickle=False));refs[method]=m['identity']['predictions']
    tsha=meta['CEM-Net']['identity']['target_sha256']
    assert all(m['identity']['target_sha256']==tsha for m in meta.values())
    targetpath=next(Path(n) for n,h in old['artifacts'].items() if h==tsha and n.endswith(split+'_targets.npz'))
    assert sha(targetpath)==tsha;target=dict(np.load(targetpath,allow_pickle=False));artifacts[str(targetpath)]=tsha
    if split=='test':
        dev=np.load(Path(c['raw_inputs'])/'dev_targets.npz');train=np.load(Path(c['raw_inputs'])/'train_targets.npz')
        assert not np.intersect1d(target['ids'][:,0],dev['ids'][:,0]).size
        assert not np.intersect1d(target['ids'][:,0],train['ids'][:,0]).size
    for method in VARIANTS:
        refs[method]={}
        for seed in SEEDS:
            for e in (0,1):
                jid=f'{method}_t{e}_s{seed}';jid=('test_'+jid) if split=='test' else jid
                j=state['jobs'][jid];assert j['status']=='completed' and read(j['exit_receipt'])['returncode']==0
                folder=Path(j['output']);completion=read(folder/'completion.json')
                assert completion['complete'] and completion['parameter_source']=='EMA'
                if split=='test':
                    assert completion['optimizer_updates']==0 and not completion['parameters_updated']
                    assert completion['dev_replay_max_drift']<=1e-5
                    assert completion['model_sha256']==lock['selections'][j['fit_id']]['checkpoint_sha256']
                refs[method][f'{seed}:{e}']=dict(path=str(folder/f'predictions_{split}.npz'),sha256=completion['predictions_sha256'])
                artifacts[str(folder/'completion.json')]=sha(folder/'completion.json')
    for method in (*OLD,*VARIANTS):
        for seed in SEEDS:
            for e in (0,1):
                ref=refs[method][f'{seed}:{e}'];assert sha(ref['path'])==ref['sha256'];artifacts[ref['path']]=ref['sha256']
                z=dict(np.load(ref['path'],allow_pickle=False));np.testing.assert_array_equal(z['ids'],target['ids'])
                y,q=z['labels'],z['probs'];col=e if y.ndim==2 and y.shape[1]==2 else 0
                y=y if y.ndim==1 else y[:,col];q=q if q.ndim==1 else q[:,col]
                np.testing.assert_array_equal(y,target['y'][:,e]);assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
                ap=float(average_precision_score(y,q));aps[method,seed,e]=ap;preds[method,seed,e]=q
                if method in OLD: assert abs(ap-oldlookup[split,method,seed,e]['AP'])<1e-12
                else:
                    completion=read(Path(ref['path']).parent/'completion.json')
                    assert abs(ap-(completion['dev_ap'] if split=='dev' else completion['metrics']['mean_ap']))<1e-12
                rows.append(dict(split=split,method=method,seed=seed,endpoint=e,AP=ap,
                    AUROC=float(roc_auc_score(y,q)),Brier=float(brier_score_loss(y,q)),
                    log_loss=float(log_loss(y,q)),reused=method in OLD))
    patients,inverse=np.unique(target['ids'][:,0],return_inverse=True);n=len(patients)
    prep={(m,s,e):prepared_ap(target['y'][:,e],preds[m,s,e]) for m in (*OLD,*VARIANTS) for s in SEEDS for e in (0,1)}
    rng=np.random.default_rng(20261026+(split=='test'))
    w=np.stack([np.bincount(rng.integers(0,n,n),minlength=n)[inverse] for _ in range(16)]).astype(float)
    for m in OLD:
        for s in SEEDS:
            for e in (0,1):
                np.testing.assert_allclose(weighted_ap(prep[m,s,e],w),draws[m][f'{s}:{e}'][:16],atol=1e-12,rtol=0)
    for method in VARIANTS:
        identity=dict(source_manifest_sha256=source_sha,split=split,method=method,target_sha256=tsha,
            predictions=refs[method],replicates=B,rng_seed=20261026+(split=='test'))
        mp=cache/(split+'_'+method+'.json');dp=mp.with_suffix('.npz')
        if mp.exists():
            m=read(mp);assert m['identity']==identity and sha(dp)==m['draws_sha256'];draws[method]=dict(np.load(dp))
        else:
            values={f'{s}:{e}':np.empty(B) for s in SEEDS for e in (0,1)};rng=np.random.default_rng(identity['rng_seed'])
            for start in range(0,B,16):
                w=np.stack([np.bincount(rng.integers(0,n,n),minlength=n)[inverse] for _ in range(min(16,B-start))]).astype(float)
                for s in SEEDS:
                    for e in (0,1): values[f'{s}:{e}'][start:start+len(w)]=weighted_ap(prep[method,s,e],w)
                if start%160==0: atomic(out/'progress.json',dict(split=split,method=method,completed=start+len(w),total=B,time=time.time()))
            with dp.with_suffix('.writing').open('wb') as stream: np.savez_compressed(stream,**values)
            os.replace(dp.with_suffix('.writing'),dp);atomic(mp,dict(identity=identity,draws_sha256=sha(dp)));draws[method]=values
    stats={str(e):{m:summary([aps[m,s,e] for s in SEEDS]) for m in (*OLD,*VARIANTS)} for e in (0,1)}
    def contrast(other,e,family):
        r=summary([aps[VARIANTS[0],s,e]-aps[other,s,e] for s in SEEDS])
        d=np.mean([draws[VARIANTS[0]][f'{s}:{e}']-draws[other][f'{s}:{e}'] for s in SEEDS],axis=0)
        r.update(reference=other,endpoint=e,family=family,coverage=1-.05/family,
            adjusted_ci=np.quantile(d,[.025/family,1-.025/family]).tolist(),one_sided_975_lower=float(np.quantile(d,.025)))
        return r
    primary={m:contrast(m,0,3) for m in ('CEM-Net',VARIANTS[1],VARIANTS[2])}
    los={m:contrast(m,1,2) for m in ('CEM-Net','EMA_NRE_MLP')}
    baseline_pairs={str(e):{m:contrast(m,e,16) for m in BASELINES} for e in (0,1)}
    validation_best={str(e):max(BASELINES,key=lambda m:np.mean([oldlookup['dev',m,s,e]['AP'] for s in SEEDS])) for e in (0,1)}
    strongest={str(e):max(BASELINES,key=lambda m:stats[str(e)][m]['mean']) for e in (0,1)}
    per_seed={str(e):{str(s):max(BASELINES,key=lambda m:aps[m,s,e]) for s in SEEDS} for e in (0,1)}
    result=dict(complete=True,split=split,source_manifest_sha256=source_sha,checkpoint_lock_sha256=sha(screen/'fixed-checkpoint-lock.json'),
        rows=rows,summary=stats,primary_mortality=primary,LOS_protection=los,local_vs_baselines=baseline_pairs,
        split_best_baseline=strongest,validation_selected_baseline=validation_best,per_seed_best_baseline=per_seed,
        artifacts=artifacts,replicates=B,seeds=list(SEEDS),optimizer_updates=0,test_loaded=split=='test',
        baseline_comparison_family=16,separate_validation_test=True,conditional_on_fitted_models=True,
        training_population_and_repeated_selection_uncertainty_included=False,patient_records_exported=False,
        historical_test_exposure=True,main_method_changed=False,created_at=time.time())
    atomic(receiptpath,result)
    if split=='dev':
        b=validation_best['0'];threshold=max(stats['0'][b]['mean'],stats['0']['EMA_NRE_MLP']['mean'])
        decision=dict(complete=True,test_loaded=False,statistics_sha256=sha(receiptpath),
            checkpoint_lock_sha256=sha(screen/'fixed-checkpoint-lock.json'),created_at=time.time(),
            mortality_validation_threshold=threshold,validation_baseline=b,LOS_margin=-.002,
            **advancement(primary,los,stats['0'][VARIANTS[0]]['mean'],threshold))
        atomic(screen/'validation-decision.json',decision)
    else:
        validation=read(screen/'validation-statistics.json')
        final=dict(complete=True,validation=validation,test=result,decision=read(screen/'validation-decision.json'),
            training_audit=audit,formal_fits=18,fixed_tests=18,source_manifest_sha256=source_sha,
            fixed_tests_zero_updates=True,main_method_changed=False)
        atomic(out/'verified-results.json',final);report(final,out)
        atomic(out/'completion.json',dict(complete=True,results_sha256=sha(out/'verified-results.json'),report_sha256=sha(out/'RESULTS.md'),
            formal_fits=18,fixed_tests=18,main_method_changed=False))
    atomic(out/'progress.json',dict(status='validation_complete' if split=='dev' else 'complete',time=time.time()))
    return result

def report(final,out):
    labels={'CEM-Net':'原CEM-Net','EMA_NRE_MLP':'等参数MLP','EMA_OC_local':'正确局部上下文',
            'EMA_OC_global':'全局上下文','EMA_OC_mismatch':'错配局部上下文'}
    lines=['# eICU观测上下文：固定18项最终结果','',
        '18项从随机初始化联合训练及18项固定EMA z0测试均通过核验。原CEM-Net继续作为主方法。','',
        '推进判断在新Test开始前依据Validation锁定：'+json.dumps(final['decision'],ensure_ascii=False),'']
    for key,title in [('validation','Validation'),('test','Test')]:
        r=final[key];lines += ['## '+title,'','|终点|方法|seed48|seed49|seed50|均值 ± 样本SD|','|---|---|---:|---:|---:|---:|']
        for e,endpoint in [('0','死亡'),('1','LOS')]:
            for m,s in r['summary'][e].items():
                lines.append('|'+endpoint+'|'+labels.get(m,m)+'|'+'|'.join(f'{x:.6f}' for x in s['values'])+f"|{s['mean']:.6f} ± {s['sample_SD']:.6f}|")
        lines+=['','|正确局部−对照|终点|seed48 / 49 / 50差值|均值 ± 样本SD|正向数|区间|','|---|---|---|---:|---:|---|']
        for m,x in r['primary_mortality'].items():
            lines.append(f"|{labels.get(m,m)}|死亡|"+', '.join(f'{v:+.6f}' for v in x['values'])+f"|{x['mean']:+.6f} ± {x['sample_SD']:.6f}|{x['positive_seeds']}/3|98.3333% [{x['adjusted_ci'][0]:+.6f}, {x['adjusted_ci'][1]:+.6f}]|")
        for m,x in r['LOS_protection'].items():
            lines.append(f"|{labels.get(m,m)}|LOS|"+', '.join(f'{v:+.6f}' for v in x['values'])+f"|{x['mean']:+.6f} ± {x['sample_SD']:.6f}|{x['positive_seeds']}/3|单侧97.5%下界 {x['one_sided_975_lower']:+.6f}|")
        for e in ('0','1'):
            for tag,m in [('本划分均值最强',r['split_best_baseline'][e]),('Validation所选',r['validation_selected_baseline'][e])]:
                x=r['local_vs_baselines'][e][m];lines+=['',f"{tag}基线（{'死亡' if e=='0' else 'LOS'}）：{m}；正确局部−基线 {x['mean']:+.6f}，{x['positive_seeds']}/3；16项族99.6875%条件区间 [{x['adjusted_ci'][0]:+.6f}, {x['adjusted_ci'][1]:+.6f}]。"]
        lines+=['']
    lines+=['4000次共同患者簇重采样，每个种子单独计算AP后平均差值，不混合种子预测。Validation/Test分族。区间条件于既有划分和已拟合模型，不包含训练总体和重复选择波动；既有Test暴露仍然存在。全部逐种子指标、16项基线比较、逐种子最强基线、训练成本和核验明细保留在verified-results.json。',
        '−0.002是预先固定的LOS研究推进容忍量，不是临床非劣效界值。未通过门槛表示此固定候选证据不足，不自动扩展训练、调参或切换主方法。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--split',choices=['dev','test'],required=True);a=p.parse_args()
    assert a.screen.resolve().is_relative_to(a.project.resolve())
    with (a.screen/'statistics.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);analyze(a.project,a.screen,a.split)
