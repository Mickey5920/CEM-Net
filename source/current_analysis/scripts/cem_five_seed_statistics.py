"""Fixed five-seed Table VI comparisons; old predictions and AP caches are read-only."""
import argparse, gc, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
from prior_utilities import read, sha, save, functions, shared, patient_weights

RUN='cem_five_seed_20261102'
PRIOR='cem_icde_evidence_20261028'
DATABASES=('mimiciii','mimic','eicu')
SEEDS=(48,49,50,51,52)
B=20000
FAMILY=54
NAMES={'CEM':'CEM-Net','TabM_mini':'TabM-mini'}

def summary(values):
    a=np.asarray(values,float);assert a.shape==(5,) and np.isfinite(a).all()
    return dict(values=a.tolist(),mean=float(a.mean()),sample_SD=float(a.std(ddof=1)))

def audit(project,db,folder,api):
    protected={}
    def keep(path,digest=None):
        path=Path(path).resolve(strict=True);assert path.is_relative_to(project)
        h=sha(path);assert digest is None or h==digest,str(path)
        protected[str(path)]=h;return path
    def record(path,digest=None):return read(keep(path,digest))
    host=project/'runs'/RUN/('host_eicu' if db=='eicu' else 'host_mimic')
    state=record(host/'state.json');assert record(host/'start_exit.json')['returncode']==0
    g=next(g for g in state['groups'].values() if g['database']==db);assert g['status']=='complete'
    source=Path(g['source']);manifest=record(source/'source_manifest.json',read(folder/'training_manifests.json')[db])
    assert g['source_manifest_sha256']==sha(source/'source_manifest.json')
    for n,h in manifest.items():keep(source/n,h)
    cfg=record(source/'work'/RUN/'config.json');screen=Path(g['output'])
    assert cfg['seeds']==[51,52] and cfg['inference']['adjusted_alpha_denominator']==FAMILY
    assert cfg['budget_inheritance']['extra_predecessor_fits']==0
    release=record(screen/'formal_release.json');lock=record(screen/'fixed-checkpoint-lock.json')
    assert release['complete'] and lock['complete'] and not lock['test_used_for_selection']
    assert len(lock['selections'])==len(cfg['matrix'])
    verification=record(screen.parent/'verification.json')
    prior=project/'runs'/PRIOR/(db+'_statistics_v1')
    active=record(prior/'active-execution.json') if (prior/'active-execution.json').exists() else {}
    assert record(prior/active.get('receipt','exit.json'))['returncode']==0
    old=record(prior/'verified-results.json',record(prior/'completion.json')['files']['verified-results.json'])
    prior_source=Path(active.get('script',project/'work'/PRIOR/('analysis_'+db+'_v1')/'statistics.py'))
    keep(prior_source,old['source_sha256'])
    assert functions(prior_source,('prepared_ap','weighted_ap','scores'),{})==api['_ap_ast']
    methods={e:sorted({NAMES.get(j['variant'],j['variant']) for j in cfg['matrix'] if j['endpoint']==e}) for e in (0,1)}
    refs={};audits=[];targets={};oldmeta={}
    for split in ('dev','test'):
        path=project/cfg['raw_inputs']/f'{split}_targets.npz' if split=='dev' else project/cfg['test_inputs']/'test_targets.npz'
        targets[split]=dict(np.load(keep(path),allow_pickle=False))
        cluster=targets[split]['ids'][:,0]
        for method in sorted(set(sum(methods.values(),[]))):
            mp=prior/'bootstrap'/('random_'+split+'_'+method.replace('-','_')+'.json')
            meta=record(mp);identity=meta['identity'];dp=keep(mp.with_suffix('.npz'),meta['draws_sha256'])
            assert all(identity[k]==v for k,v in dict(arm='random',split=split,method=method,database=db,replicates=B,rng_seed=20261026+(split=='test')).items())
            assert identity['cluster_ids_sha256']==hashlib.sha256(cluster.tobytes()).hexdigest()
            samples=dict(np.load(dp,allow_pickle=False));oldmeta[split,method]=(meta,samples)
            for e in (0,1):
                if method not in methods[e]:continue
                for s in (48,49,50):
                    ref=identity['predictions'][f'{s}:{e}'];keep(ref['path'],ref['sha256'])
                    refs[split,method,s,e]=ref
                    assert samples[f'{s}:{e}'].shape==(B,)
    assert not np.intersect1d(targets['dev']['ids'][:,0],targets['test']['ids'][:,0]).size
    fits=[j for j in g['jobs'].values() if j['action']=='train']
    assert {(j['variant'],j['seed'],j['endpoint']) for j in fits}=={(j['variant'],j['seed'],j['endpoint']) for j in cfg['matrix']}
    stopns={};functions(source/'scripts/tracehr_centered_native_common.py',('stopping',),stopns)
    for j in fits:
        tj=g['jobs']['test_'+j['id']];pj=g['jobs'][j['pilot_id']]
        for job in (j,tj,pj):assert job['status']=='completed' and record(job['exit_receipt'])['returncode']==0
        fit,test,probe=map(Path,(j['output'],tj['output'],pj['output']))
        r=record(fit/'completion.json');tc=record(test/'completion.json');identity=record(fit/'run_manifest.json')
        init=record(fit/'initialization_observation.json');history=record(fit/'history.json')
        pathcheck=record(probe/'path_audit.json',release['audit_sha256'][pj['id']])
        assert pathcheck['complete'] and not record(fit/'pilot_identity_check.json')['pilot_weights_loaded']
        assert r['complete'] and r['from_scratch'] and not r['pilot'] and not r['test_loaded']
        assert identity['source_manifest_sha256']==g['source_manifest_sha256']
        for key in ('variant','seed','endpoint'):assert identity[key]==r[key]==tc[key]==j[key]
        checkpoint=keep(fit/r['checkpoint_file'],r['checkpoint_sha256'])
        assert sha(checkpoint)==lock['selections'][j['id']]['checkpoint_sha256']==tc['model_sha256']
        assert tc['optimizer_updates']==0 and not tc['parameters_updated'] and tc['dev_replay_max_drift']<=1e-5
        assert r['checkpoint_replay']['max_probability_drift']<=1e-5
        if j['variant']=='CEM':
            assert not init['learned_checkpoint_loaded'] and init['full_initial_tensors_exact']
            assert init['initial_weights_sha256']==verification['initials'][f"CEM:{j['seed']}:{j['endpoint']}"]['initial_weights_sha256']
            assert identity['protocol']==dict(cfg['full_ema']['48:0']['protocol'],seed=j['seed'])
            assert r['parameter_source']=='EMA' and len(history)==r['epochs']<=256
            assert r['optimizer_updates']==r['ema_updates']==len(history)*math.ceil(cfg['train_rows']/256)
            assert r['best_epoch']==1+int(np.argmax([z['dev_ap'] for z in history]))
            raw=[dict(epoch=z['epoch'],dev_ap=z['raw_metrics']['mean_ap']) for z in history]
            for n in range(1,len(raw)+1):
                stop=(n==256) if db=='eicu' else stopns['stopping'](raw[:n],identity['protocol'])['stop']
                assert stop==(n==len(raw)), 'Original stopping rule not followed'
            import torch
            for ep in (1,5,8,10,12,15,18,20):
                diag=record(fit/f'epoch_{ep:03d}_diagnostics.json');keep(fit/f'epoch_{ep:03d}_model.pt',diag['checkpoint_sha256'])
                assert all(z['main_connected']==z['tensors'] and z['aux_connected']==0 for z in diag['gradient']['groups']['loss_paths']['groups'].values())
            first=torch.load(fit/'epoch_001_model.pt',map_location='cpu',weights_only=False)['model']
            last=torch.load(fit/'epoch_020_model.pt',map_location='cpu',weights_only=False)['model']
            assert all(not torch.equal(first[n],last[n]) for n in init['retained_tensor_hashes'])
            del first,last;gc.collect()
        else:
            tree=j['variant']=='LightGBM_full';protocol=identity['protocol']
            if not tree:assert protocol==cfg['recipes'][j['variant']]
            best=-1.;selected=0;stale=0
            for n,z in enumerate(history,1):
                assert z['iteration' if tree else 'epoch']==n
                if z['dev_ap']>best+(0 if tree else 1e-9):best=z['dev_ap'];selected=n;stale=0
                else:stale+=1
                assert stale<(300 if tree else 16) or n==len(history)
            assert r['best_epoch']==selected
            assert len(history)==(1000 if tree else 256) or stale==(300 if tree else 16)
        for split,path,receipt in (('dev',fit,r),('test',test,tc)):
            pred=keep(path/f'predictions_{split}.npz',receipt['predictions_sha256'])
            refs[split,NAMES.get(j['variant'],j['variant']),j['seed'],j['endpoint']]=dict(path=str(pred),sha256=sha(pred))
        audits.append(dict(id=j['id'],epochs=r['epochs'],best_epoch=r['best_epoch'],fit_seconds=r['seconds'],fit_exit=0,test_exit=0,test_updates=0,checkpoint_sha256=r['checkpoint_sha256']))
    rows=[];preds={};oldlookup={(r['split'],r['method'],r['seed'],r['endpoint']):r for r in old['rows'] if r['arm']=='random'}
    for key,ref in refs.items():
        split,method,s,e=key;z=dict(np.load(ref['path'],allow_pickle=False));t=targets[split]
        np.testing.assert_array_equal(z['ids'],t['ids'])
        y=z['labels'];q=z['probs'];y=y[:,e if y.shape[1]==2 else 0] if y.ndim==2 else y;q=q[:,e if q.shape[1]==2 else 0] if q.ndim==2 else q
        np.testing.assert_array_equal(y,t['y'][:,e]);assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
        metrics=api['scores'](y,q)
        if s<=50:assert abs(metrics['AP']-oldlookup[key]['AP'])<1e-12
        else:
            receipt=read(Path(ref['path']).parent/'completion.json')
            assert abs(metrics['AP']-receipt['metrics']['mean_ap'])<1e-12
        rows.append(dict(database=db,split=split,method=method,seed=s,endpoint=e,**metrics));preds[key]=q
    return dict(protected=protected,targets=targets,methods=methods,refs=refs,oldmeta=oldmeta,rows=rows,preds=preds,audits=audits)

def analyze(project,db,folder):
    import fcntl
    out=project/'runs'/RUN/(db+'_statistics_v1');out.mkdir(parents=True,exist_ok=True)
    owner=(out/'writer.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'completion.json').exists()
    api=shared(folder);data=audit(project,db,folder,api);cache=out/'bootstrap';cache.mkdir(exist_ok=True);draws={}
    for split in ('dev','test'):
        t=data['targets'][split];cluster=t['ids'][:,0];_,first_weights=next(patient_weights(cluster,split))
        for e,methods in data['methods'].items():
            for method in methods:
                _,old=data['oldmeta'][split,method]
                for seed in SEEDS:
                    key=(split,method,seed,e);prepared=api['prepared_ap'](t['y'][:,e],data['preds'][key])
                    if seed<=50:
                        samples=old[f'{seed}:{e}']
                        np.testing.assert_allclose(api['weighted_ap'](prepared,first_weights),samples[:16],rtol=0,atol=1e-12)
                    else:
                        identity=dict(split=split,method=method,seed=seed,endpoint=e,replicates=B,
                            cluster_sha256=hashlib.sha256(cluster.tobytes()).hexdigest(),prediction=data['refs'][key],rng_seed=20261026+(split=='test'))
                        mp=cache/f'{split}_{method}_s{seed}_t{e}.json';dp=mp.with_suffix('.npz')
                        if mp.exists():
                            meta=read(mp);assert meta['identity']==identity and sha(dp)==meta['sha256'];samples=np.load(dp)['AP']
                        else:
                            assert not dp.exists();samples=np.empty(B)
                            for start,w in patient_weights(cluster,split):
                                samples[start:start+len(w)]=api['weighted_ap'](prepared,w)
                                if start%320==0:save(out/'progress.json',dict(stage='new_seed_AP',split=split,method=method,seed=seed,endpoint=e,completed=start+len(w),total=B,time=time.time()))
                            with dp.open('xb') as stream:np.savez_compressed(stream,AP=samples)
                            save(mp,dict(identity=identity,sha256=sha(dp)))
                    assert samples.shape==(B,) and np.isfinite(samples).all();draws[key]=samples
    lookup={(r['split'],r['method'],r['seed'],r['endpoint']):r for r in data['rows']};summaries={};paired=[]
    for split in ('dev','test'):
        for e,methods in data['methods'].items():
            for m in methods:summaries[f'{split}:{m}:{e}']={metric:summary([lookup[split,m,s,e][metric] for s in SEEDS]) for metric in ('AP','AUROC','Brier','log_loss')}
            for m in methods:
                if m=='CEM-Net':continue
                values=[lookup[split,'CEM-Net',s,e]['AP']-lookup[split,m,s,e]['AP'] for s in SEEDS]
                samples=np.mean([draws[split,'CEM-Net',s,e]-draws[split,m,s,e] for s in SEEDS],axis=0)
                p=api['centered_p'](samples,float(np.mean(values)))
                paired.append(dict(database=db,split=split,candidate='CEM-Net',baseline=m,endpoint=e,**summary(values),
                    family_count=FAMILY,ci95=np.quantile(samples,[.025,.975]).tolist(),
                    adjusted_ci=np.quantile(samples,[.05/FAMILY/2,1-.05/FAMILY/2]).tolist(),p_bootstrap_centered=p,p_bonferroni=min(1.,FAMILY*p),wilcoxon=api['exact_signed_rank'](values)))
    for path,h in data['protected'].items():assert sha(path)==h,path
    result=dict(complete=True,database=db,verified_new_fits=len(data['audits']),verified_fixed_tests=len(data['audits']),
        seeds=list(SEEDS),replicates=B,old_three_seed_results_unchanged=True,existing_AP_caches_recomputed=False,
        calibration_fits=0,original_method_retained=True,source_manifest_sha256=sha(folder/'source_manifest.json'),
        artifacts=data['protected'],fit_audits=data['audits'],rows=data['rows'],summaries=summaries,paired=paired,
        inference='Five paired fixed fits; common patient-cluster resamples. Original 54-comparison Bonferroni allowance retained for nine fixed Table VI contrasts, separately per split. Null-centered Bootstrap p is approximate and never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Historical Test exposure and training-population/selection uncertainty remain outside these conditional intervals.')
    save(out/'verified-results.json',result);(out/'RESULTS.md').write_text(report(result),encoding='utf-8')
    save(out/'completion.json',dict(complete=True,files={n:sha(out/n) for n in ('verified-results.json','RESULTS.md')}))
    return dict(complete=True,database=db,verified_new_fits=len(data['audits']),verified_fixed_tests=len(data['audits']))

def report(result):
    lines=['# Five-seed Table VI comparison','',result['inference'],'']
    for split in ('dev','test'):
        lines += ['## '+split,'','|Database|Endpoint|Method|48|49|50|51|52|Mean ± sample SD|','|---|---|---|---:|---:|---:|---:|---:|---:|']
        for db in DATABASES:
            for key,value in sorted(result['summaries'].items()):
                parts=key.split(':');rowdb=result.get('database',parts[0]);parts=parts if 'database' in result else parts[1:]
                if rowdb!=db or parts[0]!=split:continue
                m,e=parts[1:];r=value['AP'];lines.append(f'|{db}|{e}|{m}|'+ '|'.join(f'{v:.6f}' for v in r['values'])+f"|{r['mean']:.6f} ± {r['sample_SD']:.6f}|")
    lines+=['','All paired differences, nominal/corrected conditional intervals and p values are in verified-results.json. Original three-seed analyses remain unchanged.']
    return '\n'.join(lines)+'\n'

def aggregate(results,api=None):
    assert {r['database'] for r in results}==set(DATABASES)
    assert sum(r['verified_new_fits'] for r in results)==sum(r['verified_fixed_tests'] for r in results)==30
    paired=[x for r in results for x in r['paired']];assert len(paired)==18
    assert all(sum(x['split']==split for x in paired)==9 for split in ('dev','test'))
    return dict(complete=True,verified_new_fits=30,verified_fixed_tests=30,seeds=list(SEEDS),replicates=B,
        inference=results[0]['inference'],paired=paired,rows=[x for r in results for x in r['rows']],
        summaries={r['database']+':'+k:v for r in results for k,v in r['summaries'].items()},old_three_seed_results_unchanged=True,
        existing_AP_caches_recomputed=False,calibration_fits=0,overall_paper_task_complete=False)

def self_test(folder):
    api=shared(folder);assert summary([1,2,3,4,5])['mean']==3
    p=api['exact_signed_rank']([1,2,3,4,5]);assert p.get('p_two_sided',p.get('pvalue',p.get('p'))) == .0625,p
    assert api['centered_p'](np.zeros(B),0.)==1.
    assert api['centered_p'](np.ones(B),1.)>0
    return dict(complete=True,five_seed_summary=True,seed_test_minimum_p=.0625,zero_and_extreme_bootstrap=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--database',choices=DATABASES,required=True)
    p.add_argument('--self-test',action='store_true');p.add_argument('--audit-only',action='store_true');a=p.parse_args();folder=Path(__file__).parent
    if a.self_test:r=self_test(folder)
    elif a.audit_only:
        d=audit(a.project,a.database,folder,shared(folder));r=dict(complete=True,new_fits=len(d['audits']),rows=len(d['rows']),sampling_started=False)
    else:r=analyze(a.project,a.database,folder)
    print(json.dumps(r))
if __name__=='__main__':main()
