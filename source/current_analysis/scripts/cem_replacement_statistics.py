"""Fixed-prediction extension of the existing ICDE inference, without fitting.

Only the two new AP distributions are sampled. Existing candidate caches must
exist, match their identities and reproduce their first sixteen patient draws.
The four 12-comparison families are assembled only after all databases arrive.
"""
import argparse
import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np

RUN = 'cem_baseline_replacement_20261030'
PRIOR = 'cem_icde_evidence_20261028'
OLD = 'cem_evidence_completion_20261025'
DATABASES = ('mimiciii', 'mimic', 'eicu')
SEEDS = (48, 49, 50)
BASELINES = ('XGBoost_TD', 'STraTS_supervised')
CANDIDATES = ('CEM-Net', 'EMA_NRE_MLP')
B = 20000
FAMILY = 12
MANIFESTS = dict(
    mimiciii='2cfcf36c30b8e471b692869e30f11c06872cd3ce9e3c90d2eeef6fcfa90cfe91',
    mimic='d262f5b6ed2a24e25c55eb9511d0cf79803f06fc4ffb043afaefff700e665d36',
    eicu='71ea9072a2a27be3501b28f38470c5360e96347e97594b23a6ad25e3721510eb')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.writing')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(pending, path)


def functions(path, names, namespace):
    """Load only named pure functions, never an older analysis entry point."""
    parsed = ast.parse(Path(path).read_bytes())
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return ast.dump(module, include_attributes=False)


def shared(folder):
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
    ns = dict(np=np, itertools=itertools, average_precision_score=average_precision_score,
              roc_auc_score=roc_auc_score, brier_score_loss=brier_score_loss, log_loss=log_loss)
    ns['_ap_ast'] = functions(folder/'prior_statistics.py', ('prepared_ap', 'weighted_ap', 'scores'), ns)
    ns['_p_ast'] = functions(folder/'prior_significance.py', ('centered_p', 'exact_signed_rank', 'holm'), ns)
    return ns


def summary(values):
    a = np.asarray(values, dtype=float)
    assert a.shape == (3,) and np.isfinite(a).all()
    return dict(values=a.tolist(), mean=float(a.mean()), sample_SD=float(a.std(ddof=1)))


def patient_weights(cluster, split, replicates=B, batch=16):
    """Exactly the ICDE72 patient-cluster RNG and ordering, without seed pooling."""
    assert split in ('dev', 'test')
    unique, inverse = np.unique(cluster, return_inverse=True)
    n = len(unique)
    assert n >= 2
    rng = np.random.default_rng(20261026 + (split == 'test'))
    for start in range(0, replicates, batch):
        w = np.stack([np.bincount(rng.integers(0, n, n), minlength=n)[inverse]
                      for _ in range(min(batch, replicates-start))]).astype(float)
        yield start, w


def validate_cache(meta, identity, path):
    assert meta['identity'] == identity, 'Bootstrap identity changed'
    assert sha(path) == meta['draws_sha256'], 'Bootstrap content changed'
    with np.load(path, allow_pickle=False) as z:
        samples = dict(z)
    assert set(samples) == {f'{s}:{e}' for s in SEEDS for e in (0, 1)}
    assert all(x.shape == (B,) and np.isfinite(x).all() for x in samples.values())
    return samples


def stopping_audit(history, result, protocol, tree):
    assert 1 <= len(history) == result['epochs'] <= (1000 if tree else 50)
    assert np.isfinite([r['dev_ap'] for r in history]).all()
    best, selected, stale = -1., 0, 0
    patience = protocol['early_stopping_rounds'] if tree else protocol['patience']
    for k, row in enumerate(history, 1):
        assert row['iteration' if tree else 'epoch'] == k
        if row['dev_ap'] > best + (0 if tree else 1e-9):
            best, selected, stale = row['dev_ap'], k, 0
        else:
            stale += 1
        assert stale < patience or k == len(history), 'Continued after fixed stopping point'
    assert selected == result['best_epoch']
    assert len(history) == (1000 if tree else 50) or stale == patience
    # AP is recomputed from saved predictions below. Probability replay tolerance
    # is checked separately; it does not imply the same numerical AP tolerance.


def audit(project, db, api):
    """Read existing outputs; no inference, calibration, sampling or model updates."""
    protected = {}

    def keep(path, expected=None):
        path = Path(path).resolve(strict=True)
        assert path.is_relative_to(project)
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        protected[str(path)] = digest
        return path

    def record(path, expected=None):
        return read(keep(path, expected))

    host = project/'runs'/RUN/('host_eicu' if db == 'eicu' else 'host_mimic')
    state = record(host/'state.json')
    exitfile = Path(state.get('active_scheduler_receipt', host/'start_exit.json'))
    assert record(exitfile)['returncode'] == 0, 'Wait for the effective host receipt'
    groups = [g for g in state['groups'].values() if g['database'] == db]
    assert len(groups) == 1 and groups[0]['status'] == 'complete'
    g = groups[0]
    source = Path(g['source'])
    manifest = record(source/'source_manifest.json', MANIFESTS[db])
    assert g['source_manifest_sha256'] == MANIFESTS[db]
    for name, digest in manifest.items():
        keep(source/name, digest)
    cfg = record(source/'work'/RUN/'config.json')
    expected_matrix = {(v, s, e) for v in BASELINES for s in SEEDS for e in (0, 1)}
    assert {(r['variant'], r['seed'], r['endpoint']) for r in cfg['matrix']} == expected_matrix
    assert cfg['formal_fits'] == 12 and cfg['inference']['new_main_comparisons_per_candidate_family'] == FAMILY
    assert cfg['inference']['old_families_unchanged'] and cfg['inference']['secondary_selected_baselines_unchanged']
    screen = Path(g['output'])
    release = record(screen/'formal_release.json')
    assert release['complete'] and release['source_manifest_sha256'] == MANIFESTS[db]
    if db != 'eicu':
        handoff = record(host/'allocator-handoff.json', state['allocator_handoff_sha256'])
        assert handoff['complete'] and handoff['active_workers']==handoff['formal_fits_started']==0
        assert not handoff['scientific_changes']
        keep(host/'allocator-predecessor-state.json',handoff['state_sha256'])
        keep(project/'work'/RUN/'allocator_runtime_v1/continue.py',handoff['code_sha256'])
        for filename,digest in handoff['protected'].items():
            keep(filename,digest)
        parity = record(screen/'allocator_parity.json')
        assert parity['complete'] and len(parity['checks'])==2
        assert all(r['model_optimizer_rng_exact'] for r in parity['checks'].values())
    lock = record(screen/'fixed-checkpoint-lock.json')
    assert lock['complete'] and not lock['test_used_for_selection'] and len(lock['selections']) == 12
    recipes = {}
    functions(source/'scripts/cem_replacement_common.py', ('protocol', 'xgb_recipe'), recipes)
    fits = [j for j in g['jobs'].values() if j['action'] == 'train']
    assert len(fits) == 12 and {(j['variant'], j['seed'], j['endpoint']) for j in fits} == expected_matrix
    refs, audits = {}, []
    for j in fits:
        tj = g['jobs']['test_'+j['id']]
        pj = g['jobs'][j['pilot_id']]
        for job in (j, tj, pj):
            assert job['status'] == 'completed' and record(job['exit_receipt'])['returncode'] == 0
        fit, test, probe = map(Path, (j['output'], tj['output'], pj['output']))
        r, tc, identity = record(fit/'completion.json'), record(test/'completion.json'), record(fit/'run_manifest.json')
        history, init = record(fit/'history.json'), record(fit/'initialization_observation.json')
        pilot = record(probe/'completion.json')
        pathcheck = record(probe/'path_audit.json',release['audit_sha256'][pj['id']])
        fresh = record(fit/'pilot_identity_check.json')
        assert r['complete'] and r['from_scratch'] and not r['pilot'] and not r['test_loaded']
        assert not init['learned_checkpoint_loaded'] and not fresh['pilot_weights_loaded']
        assert pilot['complete'] and pilot['pilot'] and pathcheck['complete']
        assert identity == record(test/'run_manifest.json')
        for key in ('variant', 'seed', 'endpoint'):
            assert identity[key] == r[key] == tc[key] == j[key]
        assert identity['database'] == db and identity['source_manifest_sha256'] == MANIFESTS[db]
        tree = j['variant'] == 'XGBoost_TD'
        protocol = identity['protocol']
        if tree:
            assert protocol['params'] == recipes['xgb_recipe'](j['seed'])
            assert protocol['max_rounds'] == 1000 and protocol['early_stopping_rounds'] == 300
            assert identity['package_version'] == '3.2.0' and protocol['class_weight'] is None
            assert protocol['missing'] == 'NaN only; dense zero is an observed value'
            assert init['tree_updates'] == tc['tree_updates'] == 0
            assert pathcheck['continuous_tree_updates'] == 20 and pathcheck['reload_prediction_exact']
        else:
            assert protocol == recipes['protocol'](j['variant'])
            assert init['optimizer_updates'] == 0 and fresh['fresh_initialization']
            assert init['initial_weights_sha256'] == identity['initial_weights_sha256']
            assert r['optimizer_updates'] == len(history)*math.ceil(cfg['train_rows']/16)
            assert r['frozen_learnable_parameters'] == 0
            assert pathcheck['continuous_updates'] == 20 and pathcheck['resume_exact']
            grad = record(probe/'gradient_check.json')
            assert grad['all_connected'] and grad['analytically_cancelled_parameters'] == ['demo_emb.0.weight']
            assert [n for n, x in grad['relative_update'].items() if x == 0] == ['demo_emb.0.weight']
            if j['seed'] == 48:
                assert identity['initial_weights_sha256'] == pilot['identity']['initial_weights_sha256']
        stopping_audit(history, r, protocol, tree)
        checkpoint = keep(fit/r['checkpoint_file'], r['checkpoint_sha256'])
        assert sha(checkpoint) == lock['selections'][j['id']]['checkpoint_sha256'] == tc['model_sha256']
        assert tc['complete'] and tc['optimizer_updates'] == 0 and not tc['parameters_updated']
        assert tc['dev_replay_max_drift'] <= 1e-5 and r['checkpoint_replay']['max_probability_drift'] <= 1e-5
        for split, folder, receipt in (('dev', fit, r), ('test', test, tc)):
            pred = keep(folder/f'predictions_{split}.npz', receipt['predictions_sha256'])
            refs[split, j['variant'], j['seed'], j['endpoint']] = dict(path=str(pred), sha256=sha(pred))
        audits.append(dict(id=j['id'],fit_exit=0,test_exit=0,epochs=r['epochs'],best_epoch=r['best_epoch'],
                           checkpoint_sha256=sha(checkpoint),test_updates=0,fit_seconds=r['seconds'],
                           parameters=identity.get('parameters'),peak_reserved_mib=r.get('peak_reserved_mib'),
                           protocol=protocol,source_manifest_sha256=MANIFESTS[db]))
    prior = project/'runs'/PRIOR/(db+'_statistics_v1')
    active = record(prior/'active-execution.json') if (prior/'active-execution.json').exists() else {}
    prior_exit = (prior/active.get('receipt', 'exit.json')).resolve()
    assert prior_exit.is_relative_to(prior) and record(prior_exit)['returncode'] == 0
    done = record(prior/'completion.json')
    assert done['complete']
    prior_result = record(prior/'verified-results.json', done['files']['verified-results.json'])
    assert prior_result['complete']
    prior_source = Path(active.get('script', project/'work'/PRIOR/('analysis_'+db+'_v1')/'statistics.py'))
    keep(prior_source,prior_result['source_sha256'])
    assert functions(prior_source,('prepared_ap','weighted_ap','scores'),{})==api['_ap_ast']
    prior_significance = record(project/'runs'/PRIOR/(db+'_significance_v1')/'verified-results.json')
    significance_source = keep(project/'work'/PRIOR/'significance_v1/significance.py',prior_significance['source_sha256'])
    assert functions(significance_source,('centered_p','exact_signed_rank','holm'),{})==api['_p_ast']
    for filename, digest in active.get('protected', {}).items():
        keep(filename, digest)
    old = record(project/'runs'/OLD/db/'comparison_v1/verified-results.json')
    assert old['complete']
    targets, oldmeta = {}, {}
    for split in ('dev', 'test'):
        for candidate in CANDIDATES:
            original = record(project/'runs'/OLD/db/'comparison_v1/bootstrap_cache'/
                              (split+'_'+candidate.replace('-', '_')+'.json'))['identity']
            targetpath = next(Path(n) for n,h in old['artifacts'].items()
                              if h == original['target_sha256'] and n.endswith(split+'_targets.npz'))
            with np.load(keep(targetpath, original['target_sha256']), allow_pickle=False) as z:
                t = dict(z)
            if split in targets:
                for key in ('ids', 'y'):
                    np.testing.assert_array_equal(t[key], targets[split][key])
            targets[split] = t
            mp = prior/'bootstrap'/('random_'+split+'_'+candidate.replace('-', '_')+'.json')
            meta = record(mp)
            keep(mp.with_suffix('.npz'), meta['draws_sha256'])
            oldmeta[split,candidate] = (meta, mp.with_suffix('.npz'))
            for s in SEEDS:
                for e in (0, 1):
                    ref = original['predictions'][f'{s}:{e}']
                    keep(ref['path'], ref['sha256'])
                    refs[split,candidate,s,e] = ref
    assert not np.intersect1d(targets['dev']['ids'][:,0], targets['test']['ids'][:,0]).size
    prior_rows = {(r['split'],r['method'],r['seed'],r['endpoint']):r
                  for r in prior_result['rows'] if r['arm'] == 'random'}
    preds, rows = {}, []
    for key, ref in refs.items():
        split, method, s, e = key
        with np.load(ref['path'], allow_pickle=False) as z:
            ids, y, q = z['ids'], z['labels'], z['probs']
        np.testing.assert_array_equal(ids, targets[split]['ids'])
        if y.ndim == 2:
            y = y[:,e if y.shape[1] == 2 else 0]
        if q.ndim == 2:
            q = q[:,e if q.shape[1] == 2 else 0]
        np.testing.assert_array_equal(y, targets[split]['y'][:,e])
        assert q.shape == y.shape and np.isfinite(q).all() and ((q >= 0)&(q <= 1)).all()
        metrics = api['scores'](y, q)
        if method in CANDIDATES:
            assert abs(metrics['AP']-prior_rows[key]['AP']) < 1e-12
        else:
            j = next(j for j in fits if (j['variant'],j['seed'],j['endpoint']) == (method,s,e))
            receipt = read(Path(j['output'] if split == 'dev' else g['jobs']['test_'+j['id']]['output'])/'completion.json')
            assert abs(metrics['AP']-receipt['metrics']['mean_ap']) < 1e-12
        rows.append(dict(database=db,split=split,method=method,seed=s,endpoint=e,**metrics))
        preds[key] = q
    return dict(protected=protected,targets=targets,refs=refs,preds=preds,rows=rows,audits=audits,oldmeta=oldmeta,
                prior_result_sha256=sha(prior/'verified-results.json'))


def contrast(values, draws, api):
    result = summary(values)
    samples = np.asarray(draws, dtype=float)
    assert samples.shape == (B,) and np.isfinite(samples).all()
    p = api['centered_p'](samples, result['mean'])
    result.update(family_count=FAMILY,ci95=np.quantile(samples,[.025,.975]).tolist(),
                  adjusted_ci=np.quantile(samples,[.05/FAMILY/2,1-.05/FAMILY/2]).tolist(),
                  p_bootstrap_centered=p,p_bonferroni=min(1.,FAMILY*p),
                  wilcoxon=api['exact_signed_rank'](values))
    return result


def analyze(project, db, source_folder):
    import fcntl
    out = project/'runs'/RUN/(db+'_statistics_v1')
    out.mkdir(parents=True,exist_ok=True)
    owner = (out/'writer.lock').open('a+')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'completion.json').exists(), 'Completed analysis must be reused'
    api = shared(source_folder)
    data = audit(project,db,api)
    cache = out/'bootstrap'
    cache.mkdir(exist_ok=True)
    draws = {}
    for split in ('dev','test'):
        t = data['targets'][split]
        cluster = t['ids'][:,0]
        for method in (*CANDIDATES,*BASELINES):
            identity = dict(arm='random',split=split,method=method,database=db,replicates=B,
                            rng_seed=20261026+(split=='test'),
                            cluster_ids_sha256=hashlib.sha256(cluster.tobytes()).hexdigest(),
                            predictions={f'{s}:{e}':data['refs'][split,method,s,e] for s in SEEDS for e in (0,1)})
            prepared = {(s,e):api['prepared_ap'](t['y'][:,e],data['preds'][split,method,s,e]) for s in SEEDS for e in (0,1)}
            if method in CANDIDATES:
                meta, dp = data['oldmeta'][split,method]
                samples = validate_cache(meta,identity,dp)
                _, weights = next(patient_weights(cluster,split))
                for (s,e), pair in prepared.items():
                    np.testing.assert_allclose(api['weighted_ap'](pair,weights),samples[f'{s}:{e}'][:16],rtol=0,atol=1e-12)
            else:
                mp = cache/(split+'_'+method+'.json')
                dp = mp.with_suffix('.npz')
                if mp.exists():
                    samples = validate_cache(read(mp),identity,dp)
                else:
                    assert not dp.exists(), 'Incomplete cache needs inspection; never overwrite'
                    samples = {f'{s}:{e}':np.empty(B) for s in SEEDS for e in (0,1)}
                    for start, weights in patient_weights(cluster,split):
                        for (s,e), pair in prepared.items():
                            samples[f'{s}:{e}'][start:start+len(weights)] = api['weighted_ap'](pair,weights)
                        if start % 320 == 0:
                            save(out/'progress.json',dict(stage='new_baseline_AP',split=split,method=method,completed=start+len(weights),total=B,time=time.time()))
                    tmp = dp.with_suffix('.writing')
                    with tmp.open('xb') as stream:
                        np.savez_compressed(stream,**samples)
                    os.replace(tmp,dp)
                    save(mp,dict(identity=identity,draws_sha256=sha(dp),existing_AP_caches_recomputed=False))
            for s in SEEDS:
                for e in (0,1):
                    draws[split,method,s,e] = samples[f'{s}:{e}']
    lookup = {(r['split'],r['method'],r['seed'],r['endpoint']):r for r in data['rows']}
    summaries, paired = {}, []
    for split in ('dev','test'):
        for m in (*CANDIDATES,*BASELINES):
            for e in (0,1):
                summaries[f'{split}:{m}:{e}'] = {metric:summary([lookup[split,m,s,e][metric] for s in SEEDS])
                                                for metric in ('AP','AUROC','Brier','log_loss')}
        for candidate in CANDIDATES:
            for baseline in BASELINES:
                for e in (0,1):
                    values = [lookup[split,candidate,s,e]['AP']-lookup[split,baseline,s,e]['AP'] for s in SEEDS]
                    samples = np.mean([draws[split,candidate,s,e]-draws[split,baseline,s,e] for s in SEEDS],axis=0)
                    paired.append(dict(database=db,split=split,candidate=candidate,baseline=baseline,endpoint=e,
                                       **contrast(values,samples,api)))
    for filename,digest in data['protected'].items():
        assert sha(filename) == digest, filename
    result = dict(complete=True,database=db,verified_new_fits=12,verified_fixed_tests=12,replicates=B,
                  existing_AP_caches_recomputed=False,calibration_fits=0,old_families_unchanged=True,
                  secondary_selected_baselines_unchanged=True,full_family_holm_pending=True,
                  source_manifest_sha256=sha(source_folder/'source_manifest.json'),
                  prior_result_sha256=data['prior_result_sha256'],inputs=data['protected'],
                  fit_audits=data['audits'],rows=data['rows'],summaries=summaries,paired=paired,
                  inference='Fixed-model patient-cluster conditional inference. Two-sided null-centered bootstrap p=(1+count(|draw-observed|>=|observed|))/(B+1), approximate. Bonferroni family=12 for each candidate and split across all three databases. Exact seed signed ranks are supplemental. Historical test exposure and training/selection uncertainty are not covered.')
    save(out/'verified-results.json',result)
    (out/'RESULTS.md').write_text(report(result),encoding='utf-8')
    save(out/'completion.json',dict(complete=True,database=db,verified_new_fits=12,verified_fixed_tests=12,
                                    files={n:sha(out/n) for n in ('verified-results.json','RESULTS.md')}))
    return dict(complete=True,database=db,verified_new_fits=12,verified_fixed_tests=12,full_family_holm_pending=True)


def report(result):
    lines = ['# Fixed two-baseline comparison','',result['inference'],'']
    for split,title in (('dev','Validation'),('test','Test')):
        lines += ['## '+title,'','|Database|Endpoint|Method|48|49|50|Mean ± sample SD|',
                  '|---|---|---|---:|---:|---:|---:|']
        dbs = DATABASES if result.get('all_databases') else (result['database'],)
        for db in dbs:
            for e in (0,1):
                for m in (*BASELINES,'EMA_NRE_MLP','CEM-Net'):
                    values = [next(r['AP'] for r in result['rows'] if (r['database'],r['split'],r['method'],r['seed'],r['endpoint']) == (db,split,m,s,e)) for s in SEEDS]
                    z = summary(values)
                    lines.append('|'+ '|'.join([db,'Mortality' if e==0 else 'LOS',m,*[f'{v:.6f}' for v in values],f"{z['mean']:.6f} ± {z['sample_SD']:.6f}"])+'|')
        lines += ['','|Database|Endpoint|Candidate minus baseline|Difference|95% CI|Family CI|p|Bonferroni p|Holm p|Seed signed-rank p|',
                  '|---|---|---|---:|---|---|---:|---:|---:|---:|']
        for r in result['paired']:
            if r['split'] != split:
                continue
            ci = lambda v: f'[{v[0]:.6f}, {v[1]:.6f}]'
            hp = f"{r['p_holm_full_family']:.6g}" if 'p_holm_full_family' in r else 'pending three databases'
            lines.append('|'+ '|'.join([r['database'],str(r['endpoint']),r['candidate']+' − '+r['baseline'],f"{r['mean']:.6f}",
                                        ci(r['ci95']),ci(r['adjusted_ci']),f"{r['p_bootstrap_centered']:.6g}",f"{r['p_bonferroni']:.6g}",hp,f"{r['wilcoxon']['p_two_sided']:.6g}"])+'|')
        lines.append('')
    return '\n'.join(lines)+'\n'


def aggregate(results, api):
    assert len(results)==3 and {r['database'] for r in results}==set(DATABASES)
    assert all(r['complete'] and r['verified_new_fits']==r['verified_fixed_tests']==12 for r in results)
    paired = [dict(row) for r in results for row in r['paired']]
    assert len(paired)==48
    for split in ('dev','test'):
        for candidate in CANDIDATES:
            family = [r for r in paired if r['split']==split and r['candidate']==candidate]
            assert len(family)==FAMILY
            assert {(r['database'],r['baseline'],r['endpoint']) for r in family} == {
                (db,b,e) for db in DATABASES for b in BASELINES for e in (0,1)}
            for row,p in zip(family,api['holm']([r['p_bootstrap_centered'] for r in family])):
                row.update(p_holm_full_family=p,holm_family_count=FAMILY)
    return dict(complete=True,all_databases=True,verified_new_fits=36,verified_fixed_tests=36,
                rows=[row for r in results for row in r['rows']],paired=paired,replicates=B,
                calibration_fits=0,existing_AP_caches_recomputed=False,old_families_unchanged=True,
                secondary_selected_baselines_unchanged=True,inference=results[0]['inference'])


def self_test(folder):
    api = shared(folder)
    y = np.array([1,0,1,0,1,0])
    q = np.array([.8,.8,.5,.2,.2,.1])
    weights = np.array([[1,1,1,1,1,1],[0,3,2,1,2,0]],float)
    from sklearn.metrics import average_precision_score
    np.testing.assert_allclose(api['weighted_ap'](api['prepared_ap'](y,q),weights),
                               [average_precision_score(y,q,sample_weight=w) for w in weights],rtol=0,atol=1e-14)
    x = np.linspace(-.3,.7,20000)
    assert api['centered_p'](x,.2)==api['centered_p'](-x,-.2)
    assert api['centered_p'](x,0)==1
    assert api['centered_p'](np.ones(B),1)==1/(B+1)
    assert api['exact_signed_rank']([1,2,3])['p_two_sided']==.25
    assert api['exact_signed_rank']([0,0,0])['p_two_sided']==1
    np.testing.assert_allclose(api['holm']([.01,.04,.03]),[.03,.06,.06])
    return dict(complete=True,checks=['weighted_AP_ties_zero_weights','centered_p_symmetry_zero_extreme_tail',
                                     'three_seed_exact_signed_ranks','full_family_Holm'],training_launched=False,
                resampling_launched=False,calibration_fits=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--database',choices=DATABASES,required=True)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    a = p.parse_args()
    folder = Path(__file__).resolve().parent
    manifest = read(folder/'source_manifest.json')
    for name,digest in manifest.items():
        assert sha(folder/name)==digest
    if a.self_test:
        result = self_test(folder)
    elif a.audit_only:
        data = audit(a.project.resolve(strict=True),a.database,shared(folder))
        result = dict(complete=True,audit_only=True,verified_new_fits=len(data['audits']),verified_fixed_tests=len(data['audits']),
                      rows_checked=len(data['rows']),training_launched=False,resampling_launched=False)
    else:
        result = analyze(a.project.resolve(strict=True),a.database,folder)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
