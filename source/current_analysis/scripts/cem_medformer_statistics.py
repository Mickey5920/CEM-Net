"""Fixed five-seed Medformer inference; reuse all original CEM AP caches."""
import argparse
import hashlib
import math
import time
from pathlib import Path
import numpy as np
from prior_utilities import read,sha,save,functions,shared,patient_weights

RUN='cem_medformer_20261103'
OLD='cem_icde_evidence_20261028'
FIVE='cem_five_seed_20261102'
DATABASES=('mimiciii','mimic','eicu')
VARIANTS=('Medformer',)
SEEDS=(48,49,50,51,52)
B=20000
FAMILY=6


def summary(values):
    a=np.asarray(values,float);assert a.shape==(5,) and np.isfinite(a).all()
    return dict(values=a.tolist(),mean=float(a.mean()),sample_SD=float(a.std(ddof=1)))


def audit(project,db,folder,api):
    protected={}
    def keep(path,expected=None):
        p=Path(path).resolve(strict=True);assert p.is_relative_to(project);h=sha(p)
        assert expected is None or h==expected,str(p)
        protected[str(p)]=h;return p
    def record(path,expected=None):return read(keep(path,expected))
    host=project/'runs'/RUN/('host_eicu' if db=='eicu' else 'host_mimic')
    state=record(host/'state.json');active_exit=Path(state.get('active_scheduler_receipt',host/'start_exit.json'))
    assert active_exit.resolve().is_relative_to(host.resolve())
    assert active_exit==host/'start_exit.json'
    owner=record(host/'start_exit_owner.json');assert owner['child_pid']==state['pid']
    rc=record(active_exit);assert rc['returncode']==0
    assert not Path('/proc/'+str(state['pid'])).exists(),'Original queue has not exited'
    g=next(g for g in state['groups'].values() if g['database']==db);assert g['status']=='complete'
    source=Path(g['source']);manifest=record(source/'source_manifest.json',read(folder/'training_manifests.json')[db])
    for name,h in manifest.items():keep(source/name,h)
    cfg=record(source/'work'/RUN/'config.json');screen=Path(g['output'])
    assert cfg['snapshot']=='v1' and cfg['recipes']['Medformer']['batch']==128
    assert record(screen.parent/'verification.json')['existing_CNN_files_unchanged']
    assert cfg['seeds']==list(SEEDS) and cfg['variants']==list(VARIANTS) and cfg['formal_fits']==10
    assert cfg['deferred_variants']==['MedSpaformer'] and cfg['inference']['family_size']==FAMILY
    release=record(screen/'formal_release.json');lock=record(screen/'fixed-checkpoint-lock.json')
    assert release['complete'] and lock['complete'] and not lock['test_used_for_selection'] and len(lock['selections'])==10
    old=project/'runs'/OLD/(db+'_statistics_v1');five=project/'runs'/FIVE/(db+'_statistics_v1')
    five_result=record(five/'verified-results.json',record(five/'completion.json')['files']['verified-results.json'])
    assert record(five/'exit.json')['returncode']==0 and five_result['complete']
    assert five_result['seeds']==list(SEEDS) and five_result['replicates']==B
    targets={};refs={};draws={};audits=[]
    for split in ('dev','test'):
        path=project/cfg['raw_inputs']/f'{split}_targets.npz' if split=='dev' else project/cfg['test_inputs']/'test_targets.npz'
        targets[split]=dict(np.load(keep(path),allow_pickle=False));cluster=targets[split]['ids'][:,0]
        digest=hashlib.sha256(cluster.tobytes()).hexdigest()
        meta=record(old/'bootstrap'/f'random_{split}_CEM_Net.json');identity=meta['identity']
        assert identity['cluster_ids_sha256']==digest and identity['replicates']==B and identity['database']==db
        assert identity['rng_seed']==20261026+(split=='test') and identity['method']=='CEM-Net' and identity['split']==split
        arr=dict(np.load(keep(old/'bootstrap'/f'random_{split}_CEM_Net.npz',meta['draws_sha256']),allow_pickle=False))
        for e in (0,1):
            for s in SEEDS:
                if s<=50:
                    ref=identity['predictions'][f'{s}:{e}'];samples=arr[f'{s}:{e}']
                else:
                    mp=five/'bootstrap'/f'{split}_CEM-Net_s{s}_t{e}.json';m=record(mp);i=m['identity']
                    assert i['cluster_sha256']==digest and i['replicates']==B and i['rng_seed']==20261026+(split=='test')
                    assert (i['split'],i['method'],i['seed'],i['endpoint'])==(split,'CEM-Net',s,e)
                    ref=i['prediction'];samples=np.load(keep(mp.with_suffix('.npz'),m['sha256']),allow_pickle=False)['AP']
                keep(ref['path'],ref['sha256']);refs[split,'CEM-Net',s,e]=ref
                assert samples.shape==(B,) and np.isfinite(samples).all();draws[split,'CEM-Net',s,e]=samples
    for j in g['jobs'].values():
        if j['action']!='train':continue
        tj=g['jobs']['test_'+j['id']];pj=g['jobs'][j['pilot_id']]
        for job in (j,tj,pj):assert job['status']=='completed' and record(job['exit_receipt'])['returncode']==0
        fit,test,probe=map(Path,(j['output'],tj['output'],pj['output']))
        r=record(fit/'completion.json');tc=record(test/'completion.json');i=record(fit/'run_manifest.json')
        init=record(fit/'initialization_observation.json');h=record(fit/'history.json')
        pa=record(probe/'path_audit.json',release['audit_sha256'][pj['id']]);grad=record(probe/'gradient_check.json')
        assert pa['complete'] and pa['continuous_updates']==20 and pa['resume_exact'] and pa['resume_raw_optimizer_swa_rng_exact'] and pa['swa_arithmetic_average_verified'] and grad['complete'] and grad['all_trainable_parameters_updated']
        assert not record(fit/'pilot_identity_check.json')['pilot_weights_loaded']
        assert r['complete'] and r['from_scratch'] and not r['pilot'] and not r['test_loaded']
        assert init['same_seed_rebuild_exact'] and not init['learned_checkpoint_loaded'] and init['optimizer_updates']==0
        assert i['initial_weights_sha256']==init['initial_weights_sha256'] and i['protocol']==cfg['recipes'][j['variant']]
        assert i['source_manifest_sha256']==sha(source/'source_manifest.json')==g['source_manifest_sha256']
        for key in ('variant','seed','endpoint'):assert i[key]==r[key]==tc[key]==j[key]
        checkpoint=keep(fit/'best.pt',r['checkpoint_sha256'])
        assert sha(checkpoint)==lock['selections'][j['id']]['checkpoint_sha256']==tc['model_sha256']
        assert tc['optimizer_updates']==0 and not tc['parameters_updated'] and tc['dev_replay_max_drift']<=1e-5
        assert r['checkpoint_replay']['max_probability_drift']<=1e-5 and r['epochs']==len(h)
        spec=i['protocol'];best=-1.;selected=0;stale=0
        for epoch,row in enumerate(h,1):
            assert row['epoch']==row['swa_updates']==epoch and np.isfinite(row['loss'])
            if row['dev_ap']>best+spec['min_delta']:best=row['dev_ap'];selected=epoch;stale=0
            else:stale+=1
            if spec['patience'] is not None:assert stale<spec['patience'] or epoch==len(h)
        assert r['best_epoch']==selected and abs(r['dev_ap']-best)<1e-12
        assert len(h)==spec['epochs'] or (spec['patience'] is not None and stale==spec['patience'])
        train_rows=record(screen.parent/'verification.json')['train_rows']
        assert r['optimizer_updates']==len(h)*math.ceil(train_rows/spec['batch']) and r['swa_updates']==len(h)
        for split,path,receipt in (('dev',fit,r),('test',test,tc)):
            pred=keep(path/f'predictions_{split}.npz',receipt['predictions_sha256'])
            refs[split,j['variant'],j['seed'],j['endpoint']]=dict(path=str(pred),sha256=sha(pred))
        audits.append(dict(id=j['id'],epochs=r['epochs'],best_epoch=r['best_epoch'],fit_seconds=r['seconds'],parameters=i['parameters'],
            fit_exit=0,test_exit=0,test_updates=0,checkpoint_sha256=r['checkpoint_sha256'],peak_reserved_mib=r['peak_reserved_mib']))
    assert len(audits)==10 and len(refs)==40
    rows=[];preds={};five_lookup={(z['split'],z['method'],z['seed'],z['endpoint']):z for z in five_result['rows']}
    for key,ref in refs.items():
        split,method,s,e=key;z=dict(np.load(ref['path'],allow_pickle=False));t=targets[split]
        np.testing.assert_array_equal(z['ids'],t['ids']);y=z['labels'];q=z['probs']
        y=y[:,e if y.shape[1]==2 else 0] if y.ndim==2 else y;q=q[:,e if q.shape[1]==2 else 0] if q.ndim==2 else q
        np.testing.assert_array_equal(y,t['y'][:,e]);assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
        metrics=api['scores'](y,q)
        reference=five_lookup[key]['AP'] if method=='CEM-Net' else read(Path(ref['path']).parent/'completion.json')['metrics']['mean_ap']
        assert abs(metrics['AP']-reference)<1e-12
        rows.append(dict(database=db,split=split,method=method,seed=s,endpoint=e,**metrics));preds[key]=q
    return dict(protected=protected,targets=targets,refs=refs,draws=draws,rows=rows,preds=preds,audits=audits)


def analyze(project,db,folder):
    import fcntl
    out=project/'runs'/RUN/(db+'_statistics_v1');out.mkdir(parents=True,exist_ok=True)
    owner=(out/'writer.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (out/'completion.json').exists()
    api=shared(folder);d=audit(project,db,folder,api);cache=out/'bootstrap';cache.mkdir(exist_ok=True);draws=d['draws']
    for split in ('dev','test'):
        t=d['targets'][split];cluster=t['ids'][:,0];_,first=next(patient_weights(cluster,split));pending={}
        for key,ref in d['refs'].items():
            if key[0]!=split:continue
            _,method,s,e=key;prepared=api['prepared_ap'](t['y'][:,e],d['preds'][key])
            if method=='CEM-Net':np.testing.assert_allclose(api['weighted_ap'](prepared,first),draws[key][:16],rtol=0,atol=1e-12);continue
            identity=dict(split=split,method=method,seed=s,endpoint=e,replicates=B,
                cluster_sha256=hashlib.sha256(cluster.tobytes()).hexdigest(),prediction=ref,rng_seed=20261026+(split=='test'))
            mp=cache/f'{split}_{method}_s{s}_t{e}.json';dp=mp.with_suffix('.npz')
            if mp.exists():
                meta=read(mp);assert meta['identity']==identity and sha(dp)==meta['sha256'];draws[key]=np.load(dp,allow_pickle=False)['AP']
            else:
                assert not dp.exists();draws[key]=np.empty(B);pending[key]=(prepared,identity,mp,dp)
        # Generate each common weight batch once; all new predictions use it.
        if pending:
            for start,w in patient_weights(cluster,split):
                for key,(prepared,_,_,_) in pending.items():draws[key][start:start+len(w)]=api['weighted_ap'](prepared,w)
                if start%160==0:save(out/'progress.json',dict(stage='new_models_AP',split=split,models=len(pending),completed=start+len(w),total=B,time=time.time()))
            for key,(_,identity,mp,dp) in pending.items():
                assert np.isfinite(draws[key]).all()
                with dp.open('xb') as stream:np.savez_compressed(stream,AP=draws[key])
                save(mp,dict(identity=identity,sha256=sha(dp)))
    lookup={(r['split'],r['method'],r['seed'],r['endpoint']):r for r in d['rows']};summaries={};paired=[]
    for split in ('dev','test'):
        for e in (0,1):
            for method in ('CEM-Net',)+VARIANTS:
                summaries[f'{split}:{method}:{e}']={metric:summary([lookup[split,method,s,e][metric] for s in SEEDS]) for metric in ('AP','AUROC','Brier','log_loss')}
            for method in VARIANTS:
                values=[lookup[split,'CEM-Net',s,e]['AP']-lookup[split,method,s,e]['AP'] for s in SEEDS]
                samples=np.mean([draws[split,'CEM-Net',s,e]-draws[split,method,s,e] for s in SEEDS],axis=0)
                p=api['centered_p'](samples,float(np.mean(values)))
                paired.append(dict(database=db,split=split,candidate='CEM-Net',baseline=method,endpoint=e,**summary(values),
                    family_count=FAMILY,ci95=np.quantile(samples,[.025,.975]).tolist(),
                    adjusted_ci=np.quantile(samples,[.05/FAMILY/2,1-.05/FAMILY/2]).tolist(),
                    p_bootstrap_centered=p,p_bonferroni=min(1.,FAMILY*p),wilcoxon=api['exact_signed_rank'](values)))
    for path,h in d['protected'].items():assert sha(path)==h,path
    result=dict(complete=True,database=db,verified_new_fits=10,verified_fixed_tests=10,seeds=list(SEEDS),replicates=B,
        deferred_methods=['MedSpaformer'],family_count=FAMILY,old_results_unchanged=True,existing_AP_caches_recomputed=False,calibration_fits=0,
        source_manifest_sha256=sha(folder/'source_manifest.json'),artifacts=d['protected'],fit_audits=d['audits'],
        rows=d['rows'],summaries=summaries,paired=paired,
        inference='Five paired fixed fits; 20000 common patient-cluster draws. Six fixed CEM-Net versus Medformer comparisons per split form an independent family; all previous families remain unchanged. Centered Bootstrap p is approximate, never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Intervals condition on fitted models and do not cover training-population or historical selection uncertainty. No new calibration or reselection of prior secondary baselines.')
    save(out/'verified-results.json',result);(out/'RESULTS.md').write_text(report(result),encoding='utf-8')
    save(out/'completion.json',dict(complete=True,files={n:sha(out/n) for n in ('verified-results.json','RESULTS.md')}))
    return dict(complete=True,database=db,verified_new_fits=10,verified_fixed_tests=10)


def report(r):
    lines=['# Fixed five-seed comparator results','',r['inference'],'','|Database|Split|Endpoint|Method|AP mean|Sample SD|','|---|---|---|---|---:|---:|']
    for key,s in sorted(r['summaries'].items()):
        db=r.get('database');parts=key.split(':')
        if db is None:db=parts.pop(0)
        split,method,e=parts;v=s['AP'];lines.append(f'|{db}|{split}|{e}|{method}|{v["mean"]:.6f}|{v["sample_SD"]:.6f}|')
    return '\n'.join(lines)+'\n'


def aggregate(results,api):
    assert {r['database'] for r in results}==set(DATABASES)
    assert sum(r['verified_new_fits'] for r in results)==sum(r['verified_fixed_tests'] for r in results)==30
    paired=[dict(x) for r in results for x in r['paired']];assert len(paired)==12
    for split in ('dev','test'):
        rows=[x for x in paired if x['split']==split];assert len(rows)==6
        corrected=api['holm']([x['p_bootstrap_centered'] for x in rows])
        for x,p in zip(rows,corrected):x['p_holm']=float(p)
    return dict(complete=True,verified_new_fits=30,verified_fixed_tests=30,seeds=list(SEEDS),replicates=B,
        deferred_methods=['MedSpaformer'],family_count=6,active_comparisons_per_split=6,inference=results[0]['inference'],
        paired=paired,rows=[x for r in results for x in r['rows']],fit_audits=[dict(database=r['database'],**x) for r in results for x in r['fit_audits']],
        summaries={r['database']+':'+k:v for r in results for k,v in r['summaries'].items()},
        old_results_unchanged=True,existing_AP_caches_recomputed=False,calibration_fits=0,overall_paper_task_complete=False)


def self_test(folder):
    api=shared(folder);assert summary([1,2,3,4,5])['sample_SD']==np.std([1,2,3,4,5],ddof=1)
    assert api['centered_p'](np.zeros(B),0.)==1 and api['centered_p'](np.ones(B),1.)>0
    p=api['exact_signed_rank']([1,2,3,4,5]);assert p.get('p_two_sided',p.get('pvalue',p.get('p')))==.0625
    values=[.001,.02,.5]+[1.]*3;corrected=api['holm'](values);assert np.isclose(corrected[0],.006)
    for split in ('dev','test'):
        _,w=next(patient_weights(np.array([1,1,2,3]),split));assert w.shape==(16,4) and np.array_equal(w[:,0],w[:,1])
    return dict(complete=True,family_allowance=6,active_comparisons=6,deferred_comparisons=0,seed_test_minimum_p=.0625,patient_cluster_pairing=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--database',choices=DATABASES,required=True)
    p.add_argument('--self-test',action='store_true');p.add_argument('--audit-only',action='store_true');a=p.parse_args();folder=Path(__file__).parent
    if a.self_test:r=self_test(folder)
    elif a.audit_only:
        d=audit(a.project,a.database,folder,shared(folder));r=dict(complete=True,new_fits=len(d['audits']),rows=len(d['rows']),sampling_started=False)
    else:r=analyze(a.project,a.database,folder)
    import json
    print(json.dumps(r))

if __name__=='__main__':main()


