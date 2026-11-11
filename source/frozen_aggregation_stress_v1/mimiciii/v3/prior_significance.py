"""Reuse fixed bootstrap caches; no new predictions, model fits or selection.

The centered bootstrap tests are approximate, conditional on the saved fits.
Bonferroni families match the archived interval families. Holm is added only
when every database is present, without changing the original decisions.
Legacy collectors retained intervals but not draws. Only those missing arrays
are reconstructed from fixed predictions with the original RNG, and their
archived intervals must reproduce. Existing AP caches are never recomputed.
"""
import argparse
import base64
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np

RUN = 'cem_icde_evidence_20261028'
OLD = 'cem_evidence_completion_20261025'
OC = 'cem_observation_context_20261027'
SEEDS = (48, 49, 50)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def centered_p(draws, observed, null=0., alternative='two-sided'):
    x = np.asarray(draws, dtype=float)
    assert x.ndim == 1 and len(x) > 1 and np.isfinite(x).all()
    assert np.isfinite(observed) and np.isfinite(null)
    # Center the sampling error at the null; do not test the uncentered
    # bootstrap distribution or interpret its sign fraction as a p-value.
    error = x - observed
    delta = observed - null
    if alternative == 'two-sided':
        count = np.count_nonzero(np.abs(error) >= abs(delta))
    elif alternative == 'greater':
        count = np.count_nonzero(error >= delta)
    else:
        raise ValueError(alternative)
    return float((1 + count) / (len(x) + 1))


def exact_signed_rank(values):
    """Enumerate signs of average ranks, dropping exact zero differences.

    This conditional signed-rank calculation handles ties by enumeration;
    it is supplemental for three fitted seeds, not patient-level inference.
    """
    d = np.asarray(values, dtype=float)
    assert d.ndim == 1 and np.isfinite(d).all() and len(d) <= 16
    d = d[d != 0]
    if not len(d):
        return dict(p_two_sided=1., nonzero_pairs=0, zero_rule='wilcox')
    a = np.abs(d)
    ranks = np.array([1 + np.count_nonzero(a < v) + (np.count_nonzero(a == v)-1)/2 for v in a])
    obs = abs(float(np.dot(np.sign(d), ranks)))
    null = [abs(float(np.dot(signs, ranks))) for signs in itertools.product((-1., 1.), repeat=len(d))]
    return dict(p_two_sided=float(np.mean(np.asarray(null) >= obs)),
                nonzero_pairs=len(d), zero_rule='wilcox', ties='average ranks; all sign assignments')


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    assert np.isfinite(p).all() and ((p > 0) & (p <= 1)).all()
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def weighted_AP(prepared,weights):
    order,ends,y=prepared
    w=np.asarray(weights,dtype=float)[:,order];tp=np.cumsum(w*y,axis=1)[:,ends];total=np.cumsum(w,axis=1)[:,ends]
    assert (tp[:,-1]>0).all()
    increments=np.diff(np.c_[np.zeros(len(w)),tp],axis=1)
    return (np.divide(tp,total,out=np.zeros_like(tp),where=total>0)*increments).sum(1)/tp[:,-1]


def analyze(project, db, source_sha):
    p = Path(project).resolve(strict=True)
    out = p/'runs'/RUN/(db+'_significance_v1')
    out.mkdir(exist_ok=True)
    import fcntl
    owner = (out/'writer.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    destination = out/'verified-results.json'
    if destination.exists():
        done = read(destination)
        assert done['complete'] and done['source_sha256'] == source_sha
        for name, digest in done['inputs'].items():
            assert sha(name) == digest
        return done
    protected = {}
    caches = {}
    rows = []

    def load_json(path):
        path = Path(path)
        protected[str(path)] = sha(path)
        return read(path)

    def result(folder, filename='verified-results.json', receipt='completion.json', digest_key=None):
        folder = Path(folder)
        c = load_json(folder/receipt)
        assert c['complete']
        expected = c[digest_key] if digest_key else c['files'][filename]
        assert sha(folder/filename) == expected
        r = load_json(folder/filename)
        assert r['complete']
        return r

    def cache(path, split, method=None, replicates=20000, primary_sha=None):
        path = Path(path)
        if str(path) in caches:
            return caches[str(path)]
        meta = load_json(path.with_suffix('.json'))
        identity = meta['identity']
        assert sha(path) == meta.get('draws_sha256', meta.get('sha256'))
        assert identity['replicates'] == replicates
        if 'split' in identity:
            assert identity['split'] == split
        if method is not None:
            assert identity.get('method', identity.get('cell')) == method
        if 'database' in identity:
            assert identity['database'] == db
        if primary_sha is not None:
            assert identity['primary_sha256'] == primary_sha
        rng = identity.get('rng_seed', identity.get('seed', identity.get('rng_'+split)))
        assert rng == 20261026 + (split == 'test'), (path, rng)
        protected[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k:z[k].copy() for k in z.files}
        assert all(v.shape == (replicates,) and np.isfinite(v).all() for v in arrays.values())
        caches[str(path)] = arrays
        return arrays

    def add(split, endpoint, family_id, label, old, samples, family=None, metric='AP', **extra):
        values = np.asarray(old['values'], dtype=float)
        assert values.shape == (3,) and np.isfinite(values).all()
        point = float(values.mean())
        assert abs(point-old['mean']) < 1e-12
        m = int(family or old.get('family_count', old.get('family', 1)))
        ci = np.quantile(samples, [.025/m, 1-.025/m]).tolist()
        existing_ci = old.get('adjusted_ci', old.get('family_adjusted_conditional_ci'))
        if existing_ci is not None:
            np.testing.assert_allclose(ci, existing_ci, rtol=0, atol=1e-12)
        pv = centered_p(samples, point)
        rows.append(dict(database=db, split=split, endpoint=int(endpoint), family=family_id,
                         family_count=m, contrast=label, metric=metric, seed_differences=values.tolist(),
                         mean=point, sample_SD=float(values.std(ddof=1)), positive_seeds=int((values>0).sum()),
                         bootstrap_replicates=len(samples), ci95=np.quantile(samples, [.025,.975]).tolist(),
                         original_adjusted_ci=ci, p_bootstrap_centered=pv,
                         p_bonferroni=min(1., m*pv), wilcoxon=exact_signed_rank(values), **extra))

    fixed = p/'runs'/RUN/(db+'_statistics_v1')
    primary = result(fixed)
    assert primary['verified_new_fits'] == primary['verified_fixed_tests'] == (36 if db=='eicu' else 18)
    primary_sha = sha(fixed/'verified-results.json')

    def fixed_draw(arm, split, method, e, s):
        data = cache(fixed/'bootstrap'/f'{arm}_{split}_{method.replace("-", "_")}.npz', split, method)
        return data[f'{s}:{e}']

    for split, blocks in primary['paired'].items():
        def fixed_contrast(label, r, e, family_id, arm='random'):
            draws = np.mean([sum(c*fixed_draw(arm,split,m,e,s) for m,c in r['coefficients'].items()) for s in SEEDS], axis=0)
            add(split,e,family_id,label,r,draws,arm=arm)
        for candidate, baselines in blocks['candidates'].items():
            for baseline, endpoints in baselines.items():
                for e,r in endpoints.items():
                    fixed_contrast(candidate+' - '+baseline,r,e,'candidate:'+candidate)
        for e,r in blocks['MLP_minus_CEM'].items():
            fixed_contrast('EMA_NRE_MLP - CEM-Net',r,e,'MLP-CEM')
        for e,contrasts in blocks['evidence'].items():
            for label,r in contrasts.items():
                if isinstance(r,dict): fixed_contrast(label,r,e,'evidence')
        for baseline,endpoints in blocks.get('hospital',{}).items():
            for e,r in endpoints.items(): fixed_contrast('CEM_hospital - '+baseline,r,e,'hospital','hospital')

    secondary_dir = p/'runs'/RUN/(db+'_secondary_v1')
    secondary = result(secondary_dir)
    assert secondary['primary_sha256'] == primary_sha
    for r in secondary['subgroup_AP_pairs']:
        split,e,name = r['split'],r['endpoint'],r['subset']
        d = cache(secondary_dir/'bootstrap'/f'{split}_t{e}_{name}.npz',split,primary_sha=primary_sha)
        sample=np.mean([d[f'CEM-Net:{s}']-d[f'{r["CEM_minus"]}:{s}'] for s in SEEDS],axis=0)
        add(split,e,'exploratory_subgroups',name+': CEM-Net - '+r['CEM_minus'],r,sample,
            exploratory=True, subset=name, original_intervals_pointwise=True)
    for r in secondary['review_budget_pairs']:
        split,e,f,metric=r['split'],r['endpoint'],r['budget'],r['metric']
        d=cache(secondary_dir/'bootstrap'/f'{split}_t{e}_review.npz',split,primary_sha=primary_sha)
        baseline=next(m for m in r['methods'] if m!='CEM-Net')
        sample=np.mean([d[f'CEM-Net:{s}:{f}:{metric}']-d[f'{baseline}:{s}:{f}:{metric}'] for s in SEEDS],axis=0)
        add(split,e,'review:'+metric,str(f)+': CEM-Net - '+baseline,r['CEM_minus_baseline'],sample,metric=metric,budget=f)
    for r in secondary['stress']:
        e,condition=r['endpoint'],r['condition']
        d=cache(secondary_dir/'bootstrap'/f'stress_t{e}_{condition}.npz','test')
        methods=r['methods'];baseline=next(m for m in methods if m!='CEM-Net')
        changes={}
        for m,v in methods.items():
            changes[m]=np.mean([d[f'{m}:{s}']-fixed_draw('random','test',m,e,s) for s in SEEDS],axis=0)
            role='CEM' if m=='CEM-Net' else 'baseline'
            add('test',e,'stress_change:'+role,condition+': '+m+' perturbed - clean',dict(**v['change'],**v['change_ci']),changes[m],condition=condition)
        gap=np.mean([d[f'CEM-Net:{s}']-d[f'{baseline}:{s}'] for s in SEEDS],axis=0)
        add('test',e,'stress_gap',condition+': CEM-Net - '+baseline,r['CEM_minus_baseline'],gap,condition=condition)
        add('test',e,'stress_interaction',condition+': differential change',r['relative_degradation'],changes['CEM-Net']-changes[baseline],condition=condition)

    olddir=p/'runs'/OLD/db/'comparison_v1'
    old=load_json(olddir/'verified-results.json');assert old['complete']
    def old_draw(split,method,e,s):
        return cache(olddir/'bootstrap_cache'/f'{split}_{method.replace("-","_")}.npz',split,method,4000)[f'{s}:{e}']
    for split,blocks in old['paired'].items():
        for method,endpoints in blocks['CEM_minus_deletion'].items():
            for e,r in endpoints.items():
                samples=np.mean([old_draw(split,'CEM-Net',e,s)-old_draw(split,method,e,s) for s in SEEDS],axis=0)
                add(split,e,'duration_joint_deletion','CEM-Net - '+method,r,samples)
    idir=p/'runs'/OLD/db/'nre_cmm_interaction_v1'
    interaction=result(idir,'interaction.json','receipt.json','result_sha256')
    assert interaction['identity']['current_sha256']==sha(olddir/'verified-results.json')
    coefficients={'interaction':{'full':1,'NRE_only':-1,'CMM_only':-1,'neither':1},
                  'NRE_with_CMM':{'full':1,'CMM_only':-1},'NRE_without_CMM':{'NRE_only':1,'neither':-1},
                  'CMM_with_NRE':{'full':1,'NRE_only':-1},'CMM_without_NRE':{'CMM_only':1,'neither':-1}}
    def cell_draw(split,cell,e,s):
        if cell in ('full','neither'):
            return old_draw(split,'CEM-Net' if cell=='full' else 'EMA_no_NRE_CMM',e,s)
        return cache(p/'cache'/OLD/'nre_cmm_interaction_v1'/db/f'{split}_{cell}.npz',split,cell,4000)[f'{s}:{e}']
    for split,endpoints in interaction['results'].items():
        for e,entry in endpoints.items():
            for name,r in entry['contrasts'].items():
                samples=np.mean([sum(c*cell_draw(split,cell,e,s) for cell,c in coefficients[name].items()) for s in SEEDS],axis=0)
                add(split,e,'NRE-CMM-interaction' if name=='interaction' else 'NRE-CMM-simple',name,r,samples)

    def legacy_draws(folder, needed, rng_seed, tag):
        """Recover unsaved historical draws, never rewrite the legacy result."""
        legacy=result(folder,receipt='collection-receipt.json')
        original_sha=sha(Path(folder)/'verified-results.json')
        lrows={(r['split'],r['method'],r['seed'],r['endpoint']):r for r in legacy['rows']}
        artifacts=legacy.get('artifacts',legacy.get('checked_result_artifacts'))
        recovered={}
        for split in ('dev','test'):
            target_meta=load_json(olddir/'bootstrap_cache'/f'{split}_CEM_Net.json')['identity']
            target_path=next(Path(n) for n,h in old['artifacts'].items() if h==target_meta['target_sha256'] and n.endswith(split+'_targets.npz'))
            assert sha(target_path)==target_meta['target_sha256'];protected[str(target_path)]=sha(target_path)
            t=dict(np.load(target_path,allow_pickle=False));patients,inverse=np.unique(t['ids'][:,0],return_inverse=True);n=len(patients)
            identity=dict(legacy_result_sha256=original_sha,split=split,methods=needed,replicates=4000,rng_seed=rng_seed+(split=='test'))
            file=out/(tag+'_'+split+'.npz');metafile=file.with_suffix('.json')
            if file.exists():
                meta=load_json(metafile);assert meta['identity']==identity and sha(file)==meta['sha256']
                recovered[split]=dict(np.load(file,allow_pickle=False));continue
            prepared={}
            for m in needed:
                for s in SEEDS:
                    for e in (0,1):
                        row=lrows[split,m,s,e]
                        h=row.get('predictions_sha256')
                        if h is None:
                            alias={'noEvidence_EMA':'CEM-Net','fullEMA':'CEM-Net','B0':'RealMLP'}.get(m,m)
                            mp=olddir/'bootstrap_cache'/f'{split}_{alias.replace("-","_")}.json'
                            if mp.exists():
                                ref=load_json(mp)['identity']['predictions'][f'{s}:{e}'];h=ref['sha256']
                            else:
                                assert db=='mimiciii' and m=='noEvidence'
                                candidates=[n for n in artifacts if n.endswith('predictions_'+split+'.npz') and Path(n).parent.name in (f'{m}_t{e}_s{s}',f'{m}_s{s}_t{e}')]
                                assert len(candidates)==1;h=artifacts[candidates[0]]
                        pred=next(Path(n) for n,digest in artifacts.items() if digest==h and n.endswith('predictions_'+split+'.npz'))
                        assert sha(pred)==h;protected[str(pred)]=h
                        z=dict(np.load(pred,allow_pickle=False));np.testing.assert_array_equal(z['ids'],t['ids'])
                        y,q=z['labels'],z['probs'];col=e if y.ndim==2 and y.shape[1]==2 else 0
                        y=y if y.ndim==1 else y[:,col];q=q if q.ndim==1 else q[:,col]
                        np.testing.assert_array_equal(y,t['y'][:,e]);assert np.isfinite(q).all()
                        order=np.argsort(-q,kind='stable');ends=np.r_[np.flatnonzero(np.diff(q[order])),len(q)-1]
                        prepared[f'{m}:{s}:{e}']=(order,ends,y[order].astype(float))
                        point=weighted_AP(prepared[f'{m}:{s}:{e}'],np.ones((1,len(y))))[0]
                        assert abs(point-row.get('AP',row.get('ap')))<1e-12
            samples={key:np.empty(4000) for key in prepared};rng=np.random.default_rng(identity['rng_seed'])
            for start in range(0,4000,16):
                weights=np.stack([np.bincount(rng.integers(0,n,n),minlength=n)[inverse] for _ in range(min(16,4000-start))])
                for key,pair in prepared.items():samples[key][start:start+len(weights)]=weighted_AP(pair,weights)
            with file.open('xb') as stream:np.savez_compressed(stream,**samples)
            with metafile.open('x') as stream:json.dump(dict(identity=identity,sha256=sha(file)),stream,indent=2)
            recovered[split]=samples
        return legacy,recovered

    if db=='mimiciii':
        legacy_folder=p/'runs/mimiciii_ema_20261024/v2/results'
        full='noEvidence_EMA';comparators=['noEvidence','TabM-mini','TabM','B0']
    else:
        legacy_folder=p/'runs/tracehr_ema_cross_20261024'/db/'v1/results'
        full='fullEMA';comparators=['noEMA','RealMLP','TabM','TabM-mini']
    legacy,draw=legacy_draws(legacy_folder,[full,*comparators],20261024,'legacy_EMA')
    for split,blocks in legacy['paired'].items():
        for m in comparators:
            for e in (0,1):
                r=dict(blocks[f'{m}_{e}'])
                if db=='mimiciii':family=8;family_id='legacy_III_EMA_and_baselines'
                else:family=4 if m=='noEMA' else 12;family_id='legacy_IV_eICU_EMA' if m=='noEMA' else 'legacy_IV_eICU_baselines'
                r['adjusted_ci']=r.get('conditional_interval',r.get('family_adjusted_conditional_ci'))
                sample=np.mean([draw[split][f'{full}:{s}:{e}']-draw[split][f'{m}:{s}:{e}'] for s in SEEDS],axis=0)
                add(split,e,family_id,full+' - '+m,r,sample,family=family,legacy_unsaved_draws_reconstructed=True)
    if db in ('mimiciii','eicu'):
        folder=p/'runs/tracehr_ema_internal_20261025'/db/('v1' if db=='mimiciii' else 'v2')/'results'
        controls=['EMA_no_numeric_direct'] if db=='mimiciii' else ['EMA_M_raw','EMA_M_periodic']
        methods=['fullEMA',*controls]+([] if db=='mimiciii' else ['EMA_noM'])
        internal,draw=legacy_draws(folder,methods,20261025,'legacy_internal')
        for split,blocks in internal['full_minus_internal_control'].items():
            for m in controls:
                for e in (0,1):
                    r=dict(blocks[f'{m}_{e}']);r['adjusted_ci']=r['conditional_interval']
                    sample=np.mean([draw[split][f'fullEMA:{s}:{e}']-draw[split][f'{m}:{s}:{e}'] for s in SEEDS],axis=0)
                    add(split,e,'coordinate_paths','fullEMA - '+m,r,sample,family=6,legacy_unsaved_draws_reconstructed=True)
        for split,endpoints in internal['modulation_factorial_interaction'].items():
            for e,r in endpoints.items():
                sample=np.mean([draw[split][f'fullEMA:{s}:{e}']-draw[split][f'EMA_M_raw:{s}:{e}']-draw[split][f'EMA_M_periodic:{s}:{e}']+draw[split][f'EMA_noM:{s}:{e}'] for s in SEEDS],axis=0)
                add(split,e,'raw_encoded_interaction','raw-by-encoded modulation',dict(r,adjusted_ci=r['conditional_interval']),sample,family=2,legacy_unsaved_draws_reconstructed=True)
    if db=='eicu':
        oc=result(p/'runs'/OC/'eicu/v2/comparison',digest_key='results_sha256')
        for split,tag in [('dev','validation'),('test','test')]:
            source=oc[tag]
            def oc_draw(method,e,s):
                if not method.startswith('EMA_OC_'):return old_draw(split,method,e,s)
                return cache(p/'cache'/OC/'bootstrap'/f'{split}_{method}.npz',split,method,4000)[f'{s}:{e}']
            for block,family_id in [('primary_mortality','OC_mortality'),('LOS_protection','OC_LOS')]:
                for method,r in source[block].items():
                    e=r['endpoint'];samples=np.mean([oc_draw('EMA_OC_local',e,s)-oc_draw(method,e,s) for s in SEEDS],axis=0)
                    extra={}
                    if block=='LOS_protection':
                        pv=centered_p(samples,r['mean'],null=-.002,alternative='greater')
                        extra=dict(noninferiority_margin=-.002,p_noninferiority_one_sided=pv,p_noninferiority_bonferroni=min(1.,2*pv))
                    add(split,e,family_id,'EMA_OC_local - '+method,r,samples,frozen_advancement_unchanged=True,**extra)
            for e,baselines in source['local_vs_baselines'].items():
                for method,r in baselines.items():
                    samples=np.mean([oc_draw('EMA_OC_local',e,s)-oc_draw(method,e,s) for s in SEEDS],axis=0)
                    add(split,e,'OC_baselines','EMA_OC_local - '+method,r,samples,frozen_advancement_unchanged=True)
    assert len({(r['split'],r['endpoint'],r['family'],r['contrast']) for r in rows})==len(rows)
    for name,digest in protected.items():assert sha(name)==digest
    final=dict(complete=True,database=db,source_sha256=source_sha,rows=rows,inputs=protected,
               existing_AP_caches_recomputed=False,legacy_missing_draws_reconstructed=True,additional_predictions=False,training_launched=False,
               patient_records_exported=False,calibration_refits=0,original_decisions_unchanged=True,
               method='Two-sided null-centered paired bootstrap: (1 + count(|delta_star - delta_hat| >= |delta_hat|))/(B+1). Approximate, not an exact permutation test.',
               limitation='Conditional on the saved models and observed patient/hospital clusters; excludes training-population, calibration-fit and adaptive-selection uncertainty. Historical Test exposure remains. Three-seed signed-rank tests are supplemental; minimum two-sided p=.25 with three nonzero pairs. AUROC and probability metrics remain descriptive.',
               subgroup_inference='Original pointwise CIs retained; pooled eligible subgroup Holm correction is exploratory and defined after the experiments, not a new confirmatory selection rule.')
    with destination.open('x',encoding='utf-8',newline='\n') as stream:json.dump(final,stream,indent=2)
    return final


def aggregate(root,out):
    from collections import defaultdict
    inputs={};rows=[]
    for db in ('mimiciii','mimic','eicu'):
        f=root/'runs'/RUN/(db+'_significance_v1')/'verified-results.json'
        r=read(f);assert r['complete'] and not r['existing_AP_caches_recomputed']
        inputs[str(f)]=sha(f);rows.extend(r['rows'])
    groups=defaultdict(list)
    for row in rows:groups[row['split'],row['family']].append(row)
    for (_,name),family in groups.items():
        expected=family[0]['family_count']
        assert all(r['family_count']==expected for r in family)
        if name!='exploratory_subgroups':assert len(family)==expected,(name,len(family),expected)
        for r,pv in zip(family,holm([r['p_bootstrap_centered'] for r in family])):
            r.update(p_holm_full_family=pv,holm_family_count=len(family))
    result=dict(complete=True,databases=['mimiciii','mimic','eicu'],input_sha256=inputs,rows=rows,
                family_counts={s+':'+f:len(v) for (s,f),v in groups.items()},frozen_intervals_and_decisions_unchanged=True)
    dest=out/'verified-results.json';content=json.dumps(result,indent=2).encode()
    if dest.exists():assert dest.read_bytes()==content
    else:dest.write_bytes(content)
    lines=['# Fixed-model paired statistical tests','',
           'All original intervals and decisions are preserved. Centered bootstrap p-values are approximate, conditional on the fitted models. Bonferroni follows the archived interval families; Holm includes every member of the listed full family. Seed signed-rank p-values enumerate signs of average ranks and are supplemental. Validation and Test are separate. Probability metrics remain descriptive.','',
           '|Database|Split|Endpoint|Family|Contrast|Seed 48 / 49 / 50 differences|Mean ± sample SD|Conditional 95% CI|Original family CI|p raw|p Bonferroni|p Holm|Seed signed-rank p|',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        fmt=lambda x:f'{x:.6g}'
        interval=lambda x:'['+', '.join(fmt(v) for v in x)+']'
        lines.append('|'+ '|'.join([r['database'],r['split'],str(r['endpoint']),r['family'],r['contrast'],
            ' / '.join(fmt(x) for x in r['seed_differences']),fmt(r['mean'])+' ± '+fmt(r['sample_SD']),interval(r['ci95']),
            interval(r['original_adjusted_ci']),fmt(r['p_bootstrap_centered']),fmt(r['p_bonferroni']),fmt(r['p_holm_full_family']),fmt(r['wilcoxon']['p_two_sided'])])+'|')
    content=('\n'.join(lines)+'\n').encode();report=out/'RESULTS.md'
    if report.exists():assert report.read_bytes()==content
    else:report.write_bytes(content)
    return dict(complete=True,comparisons=len(rows),families=result['family_counts'],report=str(report))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--database',choices=['mimiciii','mimic','eicu']);parser.add_argument('--aggregate',action='store_true');a=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(os.environ['PROJECT_RUN_DIR']).resolve(strict=True)
    if a.aggregate:
        assert out==root/'runs'/RUN/'statistical_tests_v1';print(json.dumps(aggregate(root,out)));return
    assert a.database and out==root/'runs'/RUN/(a.database+'_significance_v1')
    sys.path.insert(0,str(root))
    from scripts.launch_ema_cross import HOSTS,ssh
    host='eicu' if a.database=='eicu' else 'mimic';source=Path(__file__).read_bytes();digest=hashlib.sha256(source).hexdigest()
    code="import os,json,base64,hashlib\nfrom pathlib import Path\np=Path("+repr(HOSTS[host][1])+")\n"
    code+="os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(p/'work'/"+repr(RUN)+"/'tmp_statistics'))\n"
    code+="source=base64.b64decode("+repr(base64.b64encode(source).decode())+")\n"
    code+="folder=p/'work'/"+repr(RUN)+"/'significance_v1';folder.mkdir(exist_ok=True)\nscript=folder/'significance.py'\n"
    code+="assert not script.exists() or script.read_bytes()==source\nif not script.exists():script.write_bytes(source)\n"
    code+="ns={'__name__':'cached_significance','__file__':str(script)};exec(compile(source,str(script),'exec'),ns)\n"
    code+="print(json.dumps(ns['analyze'](p,"+repr(a.database)+","+repr(digest)+")))\n"
    result=ssh(host,code,timeout=7200);content=json.dumps(result,indent=2).encode();dest=out/'verified-results.json'
    if dest.exists():assert dest.read_bytes()==content
    else:dest.write_bytes(content)
    print(json.dumps(dict(complete=True,database=a.database,comparisons=len(result['rows']),existing_AP_caches_recomputed=False,patient_records_exported=False)))


if __name__=='__main__':main()
