"""One prespecified 12-comparison Test family for the two matched controls."""
import argparse
import hashlib
import math
import time
from pathlib import Path
import numpy as np
from prior_utilities import read, sha, save, functions, shared, patient_weights

RUN = 'cem_matched_controls_20261105'
OLD = 'cem_icde_evidence_20261028'
FIVE = 'cem_five_seed_20261102'
DATABASES = ('mimiciii', 'mimic', 'eicu')
VARIANTS = ('matched_tabm', 'window_summary')
SEEDS = (48, 49, 50, 51, 52)
B, FAMILY = 20000, 12


def summary(values):
    x = np.asarray(values, float); assert x.shape == (5,) and np.isfinite(x).all()
    return dict(values=x.tolist(), mean=float(x.mean()), sample_SD=float(x.std(ddof=1)))


def epoch_budget(db, seed, endpoint):
    return 256 if db == 'eicu' else 52 if db == 'mimic' else 120 if endpoint == 0 or seed in (49, 51) else 52


def audit(project, db, folder, api):
    protected = {}
    def keep(path, expected=None):
        path = Path(path).resolve(strict=True); assert path.is_relative_to(project)
        value = sha(path); assert expected is None or value == expected, str(path)
        protected[str(path)] = value
        return path
    def record(path, expected=None): return read(keep(path, expected))
    host = project/'runs'/RUN/('host_eicu' if db == 'eicu' else 'host_mimic')
    state = record(host/'state.json'); rc = record(host/'scheduler_v4_exit.json')
    assert state['status'] == 'complete' and rc['returncode'] == 0 and rc['child_pid'] == state['pid']
    group = next(g for g in state['groups'].values() if g['database'] == db)
    assert group['status'] == 'complete'
    source = Path(group['source']); screen = Path(group['output'])
    manifest = record(source/'source_manifest.json', read(folder/'training_manifests.json')[db])
    for name, h in manifest.items(): keep(source/name, h)
    cfg = record(source/'work'/RUN/'config.json'); parent = cfg['parent']
    assert cfg['seeds'] == list(SEEDS) and cfg['variants'] == list(VARIANTS)
    assert cfg['inference']['contrasts'] == FAMILY and cfg['inference']['patient_cluster_bootstrap_replicates'] == B
    release = record(screen/'formal_release.json'); lock = record(screen/'fixed-checkpoint-lock.json')
    assert release['complete'] and lock['complete'] and len(lock['selections']) == 20 and not lock['test_used_for_selection']
    verification = record(screen.parent/'verification.json')
    windows = project/cfg['window_preparation']; wm = record(windows/'completion.json', verification['windows_sha256'])
    for name, desc in wm['artifacts'].items(): keep(windows/name, desc['sha256'])
    old = project/'runs'/OLD/(db+'_statistics_v1'); five = project/'runs'/FIVE/(db+'_statistics_v1')
    prior = record(five/'verified-results.json', record(five/'completion.json')['files']['verified-results.json'])
    assert prior['complete'] and prior['seeds'] == list(SEEDS)
    targets, refs, old_draws, audits = {}, {}, {}, []
    for split in ('dev', 'test'):
        path = Path(parent['raw_inputs'])/f'{split}_targets.npz' if split == 'dev' else project/parent['test_inputs']/'test_targets.npz'
        targets[split] = dict(np.load(keep(path), allow_pickle=False))
        cluster_hash = hashlib.sha256(targets[split]['ids'][:, 0].tobytes()).hexdigest()
        meta = record(old/'bootstrap'/f'random_{split}_CEM_Net.json'); identity = meta['identity']
        assert identity['cluster_ids_sha256'] == cluster_hash and identity['replicates'] == B and identity['database'] == db
        assert identity['rng_seed'] == 20261026+(split == 'test') and identity['method'] == 'CEM-Net'
        arr = dict(np.load(keep(old/'bootstrap'/f'random_{split}_CEM_Net.npz', meta['draws_sha256']), allow_pickle=False))
        for endpoint in (0, 1):
            for seed in SEEDS:
                if seed <= 50:
                    ref = identity['predictions'][f'{seed}:{endpoint}']; samples = arr[f'{seed}:{endpoint}']
                else:
                    mp = five/'bootstrap'/f'{split}_CEM-Net_s{seed}_t{endpoint}.json'; m = record(mp); ident = m['identity']
                    assert ident['cluster_sha256'] == cluster_hash and ident['replicates'] == B
                    assert (ident['split'], ident['method'], ident['seed'], ident['endpoint']) == (split, 'CEM-Net', seed, endpoint)
                    ref = ident['prediction']; samples = np.load(keep(mp.with_suffix('.npz'), m['sha256']), allow_pickle=False)['AP']
                keep(ref['path'], ref['sha256']); refs[split, 'CEM-Net', seed, endpoint] = ref
                assert samples.shape == (B,) and np.isfinite(samples).all()
                if split == 'test': old_draws[seed, endpoint] = samples
    fits = [j for j in group['jobs'].values() if j['action'] == 'train']
    assert {(j['variant'], j['seed'], j['endpoint']) for j in fits} == {(v, s, e) for v in VARIANTS for s in SEEDS for e in (0, 1)}
    for j in fits:
        tj = group['jobs']['test_'+j['id']]; pj = group['jobs'][j['pilot_id']]
        for job in (j, tj, pj): assert job['status'] == 'completed' and record(job['exit_receipt'])['returncode'] == 0
        fit, test, probe = map(Path, (j['output'], tj['output'], pj['output']))
        r = record(fit/'completion.json'); tc = record(test/'completion.json'); identity = record(fit/'run_manifest.json')
        history = record(fit/'history.json'); pa = record(probe/'path_audit.json', release['audit_sha256'][pj['id']])
        assert pa['complete'] and pa['resume_raw_optimizer_ema_rng_exact'] and pa['all_parameters_updated']
        assert r['complete'] and r['from_scratch'] and not r['pilot'] and not r['test_loaded']
        assert identity['source_manifest_sha256'] == group['source_manifest_sha256']
        for key in ('variant', 'seed', 'endpoint'): assert identity[key] == r[key] == tc[key] == j[key]
        assert identity['protocol'] == dict(parent['full_ema'][f"48:{j['endpoint']}"]['protocol'], seed=j['seed'])
        assert len(history) == r['epochs'] == epoch_budget(db, j['seed'], j['endpoint'])
        assert r['optimizer_updates'] == r['ema_updates'] == r['epochs']*math.ceil(parent['train_rows']/256)
        assert r['best_epoch'] == 1+int(np.argmax([x['dev_ap'] for x in history]))
        assert all(row['epoch'] == i+1 and np.isfinite(row['dev_ap']) for i, row in enumerate(history))
        keep(fit/'best.pt', r['checkpoint_sha256'])
        assert r['checkpoint_sha256'] == lock['selections'][j['id']]['checkpoint_sha256'] == tc['model_sha256']
        assert tc['optimizer_updates'] == 0 and not tc['parameters_updated'] and tc['dev_replay_max_drift'] <= 1e-5
        assert r['checkpoint_replay']['max_probability_drift'] <= 1e-5
        for split, location, receipt in [('dev', fit, r), ('test', test, tc)]:
            path = keep(location/f'predictions_{split}.npz', receipt['predictions_sha256'])
            refs[split, j['variant'], j['seed'], j['endpoint']] = dict(path=str(path), sha256=sha(path))
        audits.append(dict(id=j['id'], epochs=r['epochs'], best_epoch=r['best_epoch'], parameters=r['parameters'],
                           fit_exit=0, test_exit=0, test_updates=0, fit_seconds=r['seconds'], checkpoint_sha256=r['checkpoint_sha256']))
    rows, predictions = [], {}
    old_rows = {(x['split'], x['seed'], x['endpoint']): x for x in prior['rows'] if x['method'] == 'CEM-Net'}
    for key, ref in refs.items():
        split, method, seed, endpoint = key
        z = dict(np.load(ref['path'], allow_pickle=False)); t = targets[split]
        np.testing.assert_array_equal(z['ids'], t['ids'])
        labels, probs = z['labels'], z['probs']
        labels = labels[:, endpoint if labels.shape[1] == 2 else 0] if labels.ndim == 2 else labels
        probs = probs[:, endpoint if probs.shape[1] == 2 else 0] if probs.ndim == 2 else probs
        np.testing.assert_array_equal(labels, t['y'][:, endpoint])
        metrics = api['scores'](labels, probs)
        if method == 'CEM-Net': assert abs(metrics['AP']-old_rows[split, seed, endpoint]['AP']) < 1e-12
        else: assert abs(metrics['AP']-read(Path(ref['path']).parent/'completion.json')['metrics']['mean_ap']) < 1e-12
        rows.append(dict(database=db, split=split, method=method, seed=seed, endpoint=endpoint, **metrics))
        predictions[key] = probs
    return dict(protected=protected, targets=targets, refs=refs, old_draws=old_draws, rows=rows, predictions=predictions, audits=audits)


def analyze(project, db, folder):
    import fcntl
    out = project/'runs'/RUN/(db+'_statistics_v1'); out.mkdir(parents=True, exist_ok=True)
    owner = (out/'writer.lock').open('a+'); fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out/'completion.json').exists()
    api = shared(folder); data = audit(project, db, folder, api)
    cache = out/'bootstrap'; cache.mkdir(exist_ok=True)
    cluster = data['targets']['test']['ids'][:, 0]; y = data['targets']['test']['y']
    _, first = next(patient_weights(cluster, 'test'))
    for endpoint in (0, 1):
        for seed in SEEDS:
            prepared = api['prepared_ap'](y[:, endpoint], data['predictions']['test', 'CEM-Net', seed, endpoint])
            np.testing.assert_allclose(api['weighted_ap'](prepared, first), data['old_draws'][seed, endpoint][:16], rtol=0, atol=1e-12)
    draws = {}
    for variant in VARIANTS:
        for endpoint in (0, 1):
            for seed in SEEDS:
                key = ('test', variant, seed, endpoint)
                identity = dict(database=db, variant=variant, seed=seed, endpoint=endpoint, replicates=B,
                                rng_seed=20261027, cluster_sha256=hashlib.sha256(cluster.tobytes()).hexdigest(), prediction=data['refs'][key])
                mp = cache/f'{variant}_s{seed}_t{endpoint}.json'; dp = mp.with_suffix('.npz')
                if mp.exists():
                    meta = read(mp); assert meta['identity'] == identity and sha(dp) == meta['sha256']
                    values = np.load(dp, allow_pickle=False)['AP']
                else:
                    assert not dp.exists(); values = np.empty(B)
                    prepared = api['prepared_ap'](y[:, endpoint], data['predictions'][key])
                    for start, weights in patient_weights(cluster, 'test'):
                        values[start:start+len(weights)] = api['weighted_ap'](prepared, weights)
                        if start % 320 == 0: save(out/'progress.json', dict(stage='control_AP', variant=variant, seed=seed, endpoint=endpoint, completed=start+len(weights), total=B, time=time.time()))
                    with dp.open('xb') as f: np.savez_compressed(f, AP=values)
                    save(mp, dict(identity=identity, sha256=sha(dp)))
                assert values.shape == (B,) and np.isfinite(values).all()
                draws[variant, seed, endpoint] = values
    lookup = {(r['split'], r['method'], r['seed'], r['endpoint']): r for r in data['rows']}
    summaries, paired, differences = {}, [], {}
    for split in ('dev', 'test'):
        for method in ('CEM-Net',)+VARIANTS:
            for endpoint in (0, 1):
                summaries[f'{split}:{method}:{endpoint}'] = {metric: summary([lookup[split, method, seed, endpoint][metric] for seed in SEEDS]) for metric in ('AP', 'AUROC', 'Brier', 'log_loss')}
    for variant in VARIANTS:
        for endpoint in (0, 1):
            observed = [lookup['test', 'CEM-Net', seed, endpoint]['AP']-lookup['test', variant, seed, endpoint]['AP'] for seed in SEEDS]
            samples = np.mean([data['old_draws'][seed, endpoint]-draws[variant, seed, endpoint] for seed in SEEDS], axis=0)
            differences[f'{variant}:{endpoint}'] = samples
            paired.append(dict(database=db, endpoint=endpoint, split='test', candidate='CEM-Net', baseline=variant,
                               **summary(observed), family_count=FAMILY, ci95=np.quantile(samples, [.025, .975]).tolist(),
                               adjusted_ci=np.quantile(samples, [.05/(2*FAMILY), 1-.05/(2*FAMILY)]).tolist(),
                               p_bootstrap_centered=api['centered_p'](samples, float(np.mean(observed)))))
    with (out/'paired_draws.npz').open('xb') as f: np.savez_compressed(f, **differences)
    for path, digest in data['protected'].items(): assert sha(path) == digest, path
    result = dict(complete=True, database=db, verified_new_fits=20, verified_fixed_tests=20, seeds=list(SEEDS), replicates=B,
                  family_count=FAMILY, rows=data['rows'], summaries=summaries, paired=paired, fit_audits=data['audits'],
                  artifacts=data['protected'], source_manifest_sha256=sha(folder/'source_manifest.json'),
                  paired_draws_sha256=sha(out/'paired_draws.npz'), original_results_unchanged=True,
                  original_CEM_bootstrap_reused=True, calibration_fits=0,
                  inference='One family of 12 Test AP comparisons. Paired patient-cluster resamples; metric differences averaged across five saved fits. Nominal 95% and Bonferroni simultaneous intervals; complete-family Holm adjustment after all three databases. Validation and other metrics are descriptive.')
    save(out/'verified-results.json', result)
    save(out/'completion.json', dict(complete=True, files={n: sha(out/n) for n in ('verified-results.json', 'paired_draws.npz')}))
    return dict(complete=True, database=db, verified_new_fits=20, verified_fixed_tests=20)


def aggregate(results, api):
    assert {r['database'] for r in results} == set(DATABASES)
    assert all(r['verified_new_fits'] == r['verified_fixed_tests'] == 20 for r in results)
    paired = [dict(row) for result in results for row in result['paired']]; assert len(paired) == FAMILY
    for row, p in zip(paired, api['holm']([r['p_bootstrap_centered'] for r in paired])): row['p_holm'] = p
    return dict(complete=True, verified_new_fits=60, verified_fixed_tests=60, seeds=list(SEEDS), replicates=B,
                family_count=FAMILY, inference=results[0]['inference'], paired=paired,
                rows=[row for result in results for row in result['rows']],
                summaries={result['database']+':'+k: v for result in results for k, v in result['summaries'].items()},
                original_results_unchanged=True, original_CEM_bootstrap_reused=True, calibration_fits=0)


def self_test(folder):
    api = shared(folder)
    labels = np.array([1, 0, 1, 0, 1, 0]); probabilities = np.array([.8, .8, .5, .2, .2, .1])
    weights = np.array([[1, 1, 1, 1, 1, 1], [0, 3, 2, 1, 2, 0]], float)
    from sklearn.metrics import average_precision_score
    np.testing.assert_allclose(api['weighted_ap'](api['prepared_ap'](labels, probabilities), weights),
                               [average_precision_score(labels, probabilities, sample_weight=w) for w in weights], rtol=0, atol=1e-14)
    assert api['centered_p'](np.zeros(B), 0.) == 1.
    assert len(api['holm']([.001]*12)) == 12
    return dict(complete=True, weighted_AP_ties_and_zero_weights=True, family_count=12, five_seed_averaging=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True)
    p.add_argument('--database', choices=DATABASES, required=True); p.add_argument('--self-test', action='store_true'); p.add_argument('--audit-only', action='store_true')
    a = p.parse_args(); folder = Path(__file__).parent
    if a.self_test: result = self_test(folder)
    elif a.audit_only:
        data = audit(a.project, a.database, folder, shared(folder)); result = dict(complete=True, fits=len(data['audits']), sampling_started=False)
    else: result = analyze(a.project, a.database, folder)
    import json
    print(json.dumps(result))
