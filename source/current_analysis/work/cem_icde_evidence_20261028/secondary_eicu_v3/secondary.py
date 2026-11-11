"""Fixed stress, clinical strata and review-budget analyses from existing predictions."""
import argparse,fcntl,hashlib,importlib.util,json,math,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score
RUN='cem_icde_evidence_20261028';OLD='cem_evidence_completion_20261025';B=20000
CONDITIONS=('lab_drop25','lab_drop50','lab_delay2h','lab_delay6h')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
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
def summary(x):return dict(values=list(map(float,x)),mean=float(np.mean(x)),sample_SD=float(np.std(x,ddof=1)))
def interval(x,family=1):
    a=.05/family
    return dict(ci95=np.quantile(x,[.025,.975]).tolist(),adjusted_ci=np.quantile(x,[a/2,1-a/2]).tolist(),family=family)
def weighted_budget(y,q,ids,w,fraction,order=None):
    order=np.lexsort((ids[:,2],-q)) if order is None else order;weights=w[:,order];prior=np.cumsum(weights,axis=1)-weights;n=w.sum(1);k=np.ceil(n*fraction)
    selected=np.maximum(0,np.minimum(weights,k[:,None]-prior));tp=(selected*y[order]).sum(1)
    return dict(precision=tp/k,recall=tp/(w*y).sum(1),true_positives_per_1000=1000*tp/n)
def self_test():
    y=np.array([1,0,1,0]);q=np.array([.8,.7,.5,.2]);ids=np.arange(12).reshape(4,3);w=np.array([[1,1,1,1],[2,0,1,1]],float)
    r=weighted_budget(y,q,ids,w,.5)
    np.testing.assert_allclose(r['precision'],[.5,1.]);np.testing.assert_allclose(r['recall'],[.5,2/3]);np.testing.assert_allclose(r['true_positives_per_1000'],[250,500])
    assert json.loads(json.dumps({'float':np.float32(.25),'integer':np.int64(3),'boolean':np.bool_(True)},default=json_scalar))=={'float':.25,'integer':3,'boolean':True}
    return ['cluster-multiplicity review budgets match explicit duplicate-patient example','ties resolved by fixed stay ID; no label-dependent ties','numpy_summary_scalars_serialize']

def pipeline_cost(project,db,cfg,out):
    # The completed exact-equality measurements use the original feature producer.
    # This recovery never reruns inference, modifies raw inputs, or relaxes equality.
    assert db=='eicu','This version is only for the diagnosed eICU runtime mismatch'
    receipt=out/'pipeline_original_runtime_v1/completion.json'
    assert sha(receipt)=='9c9fb72d5376fa7cc27c9be92692abacb4c8bd331ea58f7bc9786ab22db40a3a'
    result=read(receipt);assert result['complete'] and result['database']==db
    source=project/'work'/RUN/'pipeline_eicu_original_runtime_v1/measure.py'
    assert sha(source)==result['source_sha256']
    folder=project/cfg['test_inputs'];ref=read(folder/'completion.json')
    assert sha(folder/'test_raw.npy')==result['reference_raw_sha256']==ref['artifacts']['test_raw.npy']['sha256']
    assert sha(folder/'test_targets.npz')==result['reference_targets_sha256']==ref['artifacts']['test_targets.npz']['sha256']
    assert result['train_normalization_sha256']==ref['train_value_scales_sha256']
    assert result['train_vocabulary_sha256']==ref['train_vocabulary_sha256']
    assert result['threads']==2 and result['optimizer_updates']==0 and result['fit_transforms'] is False
    assert result['patient_records_exported'] is False
    raw=np.load(folder/'test_raw.npy',mmap_mode='r')
    assert [r['stays'] for r in result['rows']]==[128,512,2048]
    for row in result['rows']:
        assert row['raw_reconstruction_exact'] is True and row['repeats']==3 and row['dense_dimension']==raw.shape[1]
        times=row['seconds_excluding_integrity_hash'];assert len(times)==3 and np.isfinite(times).all() and min(times)>0
        assert row['median_seconds']==float(np.median(times))
    return result

def run(project,db):
    out=project/'runs'/RUN/(db+'_secondary_v1');out.mkdir(parents=True,exist_ok=True);owner=(out/'writer.lock').open('a+');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'completion.json').exists():
        done=read(out/'completion.json');assert done['complete']
        for n,h in done['files'].items():assert sha(out/n)==h
        return dict(complete=True,reused=True)
    analysis=project/'runs'/RUN/(db+'_statistics_v1');primary=read(analysis/'verified-results.json');done=read(analysis/'completion.json')
    execution=read(analysis/'active-execution.json') if (analysis/'active-execution.json').exists() else {}
    exitfile=analysis/execution.get('receipt','exit.json');assert exitfile.resolve().is_relative_to(analysis.resolve())
    assert primary['complete'] and done['complete'] and read(exitfile)['returncode']==0
    for name,digest in execution.get('protected',{}).items():assert sha(name)==digest
    assert sha(analysis/'verified-results.json')==done['files']['verified-results.json']
    statistics_source=Path(execution.get('script',project/'work'/RUN/('analysis_'+db+'_v1')/'statistics.py'))
    assert statistics_source.resolve().is_relative_to((project/'work'/RUN).resolve()) and sha(statistics_source)==primary['source_sha256']
    util=importlib.util.spec_from_file_location('fixed_statistics',statistics_source);stat=importlib.util.module_from_spec(util);util.loader.exec_module(stat);assert stat.B==B
    stressdir=project/'runs'/RUN/db/'frozen_inference_v2';stress=read(stressdir/'verified-results.json');assert stress['complete'] and stress['fixed_stress_cases']==48 and read(stressdir/'exit.json')['returncode']==0
    assert sha(stressdir/'verified-results.json')==read(stressdir/'completion.json')['files']['verified-results.json']
    cfg=read(project/'work'/RUN/db/'basic_v2/work'/RUN/'config.json')
    selected={int(e):v['validation_selected'] for e,v in primary['strongest']['dev'].items()}
    assert {str(e):m for e,m in selected.items()}=={e:v['method'] for e,v in stress['selection']['selected'].items()}
    metadata=project/'cache'/RUN/'clinical_metadata'/(db+'.parquet');mr=read(metadata.with_suffix('.json'));assert sha(metadata)==mr['sha256'];frame=pd.read_parquet(metadata).set_index('stay_id',drop=False);assert not frame.index.duplicated().any()
    prep=read(project/cfg['input_preparation']/'completion.json');phi=prep['phi_dim'];rawfolder=project/cfg['raw_inputs']
    train=np.load(rawfolder/'train_raw.npy',mmap_mode='r');train_coverage=np.asarray(train[:,phi:phi+352:11]).sum(1);assert ((train_coverage>=0)&(train_coverage<=32)).all()
    cuts=np.quantile(train_coverage,[.25,.5,.75]);targets={};raws={};probs={};clinical=[];paired=[];utility=[];stressrows=[];refs={};cache=out/'bootstrap';cache.mkdir(exist_ok=True)
    for split in ('dev','test'):
        folder=rawfolder if split=='dev' else project/cfg['test_inputs'];target=dict(np.load(folder/(split+'_targets.npz')));targets[split]=target;raw=np.load(folder/(split+'_raw.npy'),mmap_mode='r');raws[split]=raw
        meta=frame.loc[target['ids'][:,2]];np.testing.assert_array_equal(meta[['subject_id','hadm_id','stay_id']].to_numpy(),target['ids']);np.testing.assert_array_equal(meta[['mortality','los']].to_numpy(),target['y'])
        coverage=np.asarray(raw[:,phi:phi+352:11]).sum(1);band=np.searchsorted(cuts,coverage,side='left')
        subsets={'all':np.ones(len(meta),bool),**{f'coverage_Q{i+1}':band==i for i in range(4)},**{'age_'+x:meta.age_band.eq(x).to_numpy() for x in ('18-64','65+','unknown')},**{'sex_'+x:meta.sex.eq(x).to_numpy() for x in ('F','M','unknown')},'LOS_excluding_early_death':~meta.death_within_7d_after_landmark.to_numpy(dtype=bool)}
        for e in (0,1):
            methods=('CEM-Net',selected[e]);y=target['y'][:,e]
            for method in methods:
                bmeta=read(project/'runs'/OLD/db/'comparison_v1/bootstrap_cache'/(split+'_'+method.replace('-','_')+'.json'))['identity']
                for s in (48,49,50):
                    ref=bmeta['predictions'][f'{s}:{e}'];assert sha(ref['path'])==ref['sha256'];refs[ref['path']]=ref['sha256'];z=dict(np.load(ref['path']));np.testing.assert_array_equal(z['ids'],target['ids']);q=z['probs'];q=q[:,e if q.shape[1]==2 else 0] if q.ndim==2 else q
                    probs[split,e,method,s]=q
                    oldrow=next(r for r in primary['rows'] if (r['arm'],r['split'],r['method'],r['seed'],r['endpoint'])==('random',split,method,s,e))
                    assert abs(average_precision_score(y,q)-oldrow['AP'])<1e-12
            for name,mask in subsets.items():
                if name=='LOS_excluding_early_death' and e!=1:continue
                yy=y[mask];ids=target['ids'][mask];n=len(yy);positive=int(yy.sum());negative=n-positive
                cell=dict(split=split,endpoint=e,subset=name,n=n,patients=len(np.unique(ids[:,0])) if n else 0,positive=positive,negative=negative,prevalence=positive/n if n else None,methods={})
                enough=positive>=20 and negative>=20
                for method in methods:
                    per=[]
                    for s in (48,49,50):
                        q=probs[split,e,method,s][mask];coef=primary['calibrators'][f'{method}:{s}:{e}'];clipped=np.clip(q,1e-7,1-1e-7);cal=expit(coef['coef']*np.log(clipped/(1-clipped))+coef['intercept'])
                        if positive and negative:per.append(dict(seed=s,raw=stat.probability_metrics(yy,q),Platt=stat.probability_metrics(yy,cal)))
                    cell['methods'][method]=dict(per_seed=per,metrics={form:{m:summary([v[form][m] for v in per]) for m in ('AP','AUROC','Brier','log_loss','ECE15')} for form in ('raw','Platt')} if per else {},subgroup_calibrator_fitted=False,validation_resubstitution=split=='dev')
                cell['conditional_AP_inference_available']=enough
                if not enough:cell['interval_unavailable_reason']='Fewer than 20 positives or negatives in this fixed stratum; keep counts and available point estimates.'
                clinical.append(cell)
                if not enough:continue
                key=f'{split}_t{e}_{name}';npz=cache/(key+'.npz');identity=dict(split=split,endpoint=e,subset=name,mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),metadata_sha256=sha(metadata),primary_sha256=sha(analysis/'verified-results.json'),replicates=B,seed=20261026+(split=='test'))
                if npz.exists():
                    saved=read(npz.with_suffix('.json'));assert saved['identity']==identity and sha(npz)==saved['sha256'];draws=dict(np.load(npz))
                else:
                    raise RuntimeError('Recovery requires every existing subgroup bootstrap cache; no resampling authorized')
                    prepared={(m,s):stat.prepared_ap(yy,probs[split,e,m,s][mask]) for m in methods for s in (48,49,50)};draws={f'{m}:{s}':np.empty(B) for m,s in prepared}
                    if name=='all':
                        for m,s in prepared:
                            d=dict(np.load(analysis/'bootstrap'/('random_'+split+'_'+m.replace('-','_')+'.npz')));draws[f'{m}:{s}'][:]=d[f'{s}:{e}']
                    else:
                        unique,inverse=np.unique(ids[:,0],return_inverse=True);rng=np.random.default_rng(identity['seed'])
                        for start in range(0,B,16):
                            w=np.stack([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inverse] for _ in range(min(16,B-start))]).astype(float)
                            assert ((w*yy).sum(1)>0).all(),'Undefined AP bootstrap: retain failure for explicit rare-stratum handling'
                            for pair,p in prepared.items():draws[f'{pair[0]}:{pair[1]}'][start:start+len(w)]=stat.weighted_ap(p,w)
                            if start%320==0:save(out/'progress.json',dict(stage='clinical_subgroup_bootstrap',cell=key,completed=start+len(w),total=B,time=time.time()))
                    with npz.open('xb') as f:np.savez_compressed(f,**draws)
                    save(npz.with_suffix('.json'),dict(identity=identity,sha256=sha(npz)))
                delta=np.mean([draws[f'{methods[0]}:{s}']-draws[f'{methods[1]}:{s}'] for s in (48,49,50)],axis=0)
                differences=[cell['methods'][methods[0]]['metrics']['raw']['AP']['values'][i]-cell['methods'][methods[1]]['metrics']['raw']['AP']['values'][i] for i in range(3)]
                paired.append(dict(split=split,endpoint=e,subset=name,CEM_minus=methods[1],**summary(differences),**interval(delta),interpretation='exploratory nominal patient-cluster interval; not family-corrected confirmatory subgroup evidence'))
            # Review budget: resample whole patients, recompute each weighted ranking.
            key=f'{split}_t{e}_review';file=cache/(key+'.npz');identity=dict(primary_sha256=sha(analysis/'verified-results.json'),replicates=B,seed=20261026+(split=='test'),budgets=[.05,.1])
            if file.exists():
                saved=read(file.with_suffix('.json'));assert saved['identity']==identity and sha(file)==saved['sha256'];draws=dict(np.load(file))
            else:
                raise RuntimeError('Recovery requires every existing review-budget bootstrap cache; no resampling authorized')
                unique,inverse=np.unique(target['ids'][:,0],return_inverse=True);rng=np.random.default_rng(identity['seed']);draws={f'{m}:{s}:{f}:{metric}':np.empty(B) for m in methods for s in (48,49,50) for f in (.05,.1) for metric in ('precision','recall','true_positives_per_1000')}
                budget_orders={(m,s):np.lexsort((target['ids'][:,2],-probs[split,e,m,s])) for m in methods for s in (48,49,50)}
                for start in range(0,B,16):
                    w=np.stack([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inverse] for _ in range(min(16,B-start))]).astype(float)
                    for m in methods:
                        for s in (48,49,50):
                            for f in (.05,.1):
                                for metric,v in weighted_budget(y,probs[split,e,m,s],target['ids'],w,f,budget_orders[m,s]).items():draws[f'{m}:{s}:{f}:{metric}'][start:start+len(w)]=v
                    if start%320==0:save(out/'progress.json',dict(stage='review_budget_bootstrap',cell=key,completed=start+len(w),total=B,time=time.time()))
                with file.open('xb') as f:np.savez_compressed(f,**draws)
                save(file.with_suffix('.json'),dict(identity=identity,sha256=sha(file)))
            for f in (.05,.1):
                for metric in ('precision','recall','true_positives_per_1000'):
                    point={m:[float(weighted_budget(y,probs[split,e,m,s],target['ids'],np.ones((1,len(y))),f)[metric][0]) for s in (48,49,50)] for m in methods}
                    difference=np.mean([draws[f'{methods[0]}:{s}:{f}:{metric}']-draws[f'{methods[1]}:{s}:{f}:{metric}'] for s in (48,49,50)],axis=0)
                    utility.append(dict(split=split,endpoint=e,budget=f,metric=metric,methods={m:summary(v) for m,v in point.items()},CEM_minus_baseline=dict(**summary(np.asarray(point[methods[0]])-point[methods[1]]),**interval(difference,12)),family_definition='six database-endpoint cells by two budgets, separately for each utility metric and split'))
    target=targets['test'];unique,inverse=np.unique(target['ids'][:,0],return_inverse=True)
    for e in (0,1):
        methods=('CEM-Net',selected[e]);y=target['y'][:,e]
        for condition in CONDITIONS:
            key=f'stress_t{e}_{condition}';file=cache/(key+'.npz');prepared={};points={}
            for m in methods:
                for s in (48,49,50):
                    rec=next(r for r in stress['records'] if (r['job']['method'],r['job']['seed'],r['job']['endpoint'])==(m,s,e));f=stressdir/'jobs'/(m+f'_t{e}_s{s}')/(condition+'.npz');assert sha(f)==rec['artifacts'][condition+'.npz'];z=dict(np.load(f));np.testing.assert_array_equal(z['ids'],target['ids']);np.testing.assert_array_equal(z['labels'],y)
                    prepared[m,s]=stat.prepared_ap(y,z['probs']);points[m,s]=average_precision_score(y,z['probs'])
            identity=dict(stress_sha256=sha(stressdir/'verified-results.json'),condition=condition,endpoint=e,replicates=B,seed=20261027)
            if file.exists():
                saved=read(file.with_suffix('.json'));assert saved['identity']==identity and sha(file)==saved['sha256'];draws=dict(np.load(file))
            else:
                raise RuntimeError('Recovery requires every existing stress bootstrap cache; no resampling authorized')
                rng=np.random.default_rng(20261027);draws={f'{m}:{s}':np.empty(B) for m,s in prepared}
                for start in range(0,B,16):
                    w=np.stack([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inverse] for _ in range(min(16,B-start))]).astype(float)
                    for (m,s),pair in prepared.items():draws[f'{m}:{s}'][start:start+len(w)]=stat.weighted_ap(pair,w)
                    if start%320==0:save(out/'progress.json',dict(stage='stress_bootstrap',cell=key,completed=start+len(w),total=B,time=time.time()))
                with file.open('xb') as f:np.savez_compressed(f,**draws)
                save(file.with_suffix('.json'),dict(identity=identity,sha256=sha(file)))
            clean={m:dict(np.load(analysis/'bootstrap'/('random_test_'+m.replace('-','_')+'.npz'))) for m in methods}
            gap=np.mean([draws[f'{methods[0]}:{s}']-draws[f'{methods[1]}:{s}'] for s in (48,49,50)],axis=0)
            clean_gap=np.mean([clean[methods[0]][f'{s}:{e}']-clean[methods[1]][f'{s}:{e}'] for s in (48,49,50)],axis=0)
            point_clean={m:[average_precision_score(y,probs['test',e,m,s]) for s in (48,49,50)] for m in methods}
            point_stress={m:[points[m,s] for s in (48,49,50)] for m in methods}
            stressrows.append(dict(endpoint=e,condition=condition,methods={m:dict(clean=summary(point_clean[m]),perturbed=summary(point_stress[m]),change=summary(np.asarray(point_stress[m])-point_clean[m]),change_ci=interval(np.mean([draws[f'{m}:{s}']-clean[m][f'{s}:{e}'] for s in (48,49,50)],axis=0),24)) for m in methods},
                CEM_minus_baseline=dict(**summary(np.asarray(point_stress[methods[0]])-point_stress[methods[1]]),**interval(gap,24)),relative_degradation=dict(**summary((np.asarray(point_stress[methods[0]])-point_clean[methods[0]])-(np.asarray(point_stress[methods[1]])-point_clean[methods[1]])),**interval(gap-clean_gap,24))))
    costs=pipeline_cost(project,db,cfg,out)
    result=dict(complete=True,database=db,primary_sha256=sha(analysis/'verified-results.json'),stress_sha256=sha(stressdir/'verified-results.json'),metadata_sha256=sha(metadata),source_sha256=sha(__file__),clinical=clinical,subgroup_AP_pairs=paired,review_budget_pairs=utility,stress=stressrows,pipeline_cost=costs,
        coverage=dict(train_only_quartile_cuts=cuts.tolist(),observed_of_32=True,ties_kept_together=True),bootstrap_replicates=B,original_main_method_retained=True,
        calibration_refits=0,patient_records_exported=False,validation_test_separate=True,self_test=self_test(),
        limitations=['Intervals condition on fixed models, observed cohorts and the chosen validation baseline; historical test exposure and training-population uncertainty are not included.',
            'Subgroups are exploratory, not causal effects or proof of subgroup equivalence; calibration uses the same overall Validation-fitted calibrator.',
            'Early-death exclusion is an outcome-conditioned retrospective LOS sensitivity analysis, not a replacement label, survival model or prospective deployment population.',
            'Stress inputs are synthetic; MIMIC-III timing is specimen-based and does not establish measured result-availability robustness.'])
    save(out/'verified-results.json',result);save(out/'completion.json',dict(complete=True,overall_paper_task_complete=False,files={'verified-results.json':sha(out/'verified-results.json')}));save(out/'progress.json',dict(stage='complete',time=time.time()));return dict(complete=True,reused=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--database',required=True);p.add_argument('--self-test',action='store_true');a=p.parse_args();print(json.dumps(self_test() if a.self_test else run(a.project.resolve(strict=True),a.database)))
