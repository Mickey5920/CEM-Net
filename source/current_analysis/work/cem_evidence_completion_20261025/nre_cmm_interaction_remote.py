"""Post-hoc AP factorial analysis of existing predictions; no model is imported."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

RUN = 'cem_evidence_completion_20261025'
SEEDS = (48, 49, 50)
CELLS = ('full', 'NRE_only', 'CMM_only', 'neither')
B = 4000


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(temp, path)


def prepared_ap(y, q):
    import numpy as np
    order = np.argsort(-q, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(q[order])), len(q)-1]
    return order, ends, y[order].astype(float)


def weighted_ap(prepared, weights):
    import numpy as np
    order, ends, y = prepared
    w = weights[:, order]
    tp = np.cumsum(w*y, axis=1)[:, ends]
    total = np.cumsum(w, axis=1)[:, ends]
    assert (tp[:, -1] > 0).all()
    increments = np.diff(np.c_[np.zeros(len(w)), tp], axis=1)
    return (np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)*increments).sum(1)/tp[:, -1]


def stat(values):
    import numpy as np
    a = np.asarray(values, float)
    return dict(values=a.tolist(), mean=float(a.mean()), sample_SD=float(a.std(ddof=1)), positive_seeds=int((a > 0).sum()))


def analyze(d):
    import numpy as np
    from sklearn.metrics import average_precision_score
    p = Path(d['project']).resolve(strict=True)
    db = d['database']
    assert db in ('mimiciii', 'mimic', 'eicu')
    current_path = p/'runs'/RUN/db/'comparison_v1'/'verified-results.json'
    old_path = (p/'runs/mimiciii_ema_pm_20261024/v1/results/verified-results.json' if db == 'mimiciii'
                else p/'runs/tracehr_ema_pm_cross_20261025'/db/'v1/results/verified-results.json')
    assert sha(current_path) == d['current_sha256']
    assert sha(old_path) == d['old_sha256']
    current, old = read(current_path), read(old_path)
    assert current['complete'] and old['complete']
    assert current['fixed_checkpoint_tests_zero_updates'] and not current['training_launched']
    assert current['verified_new_fits'] == current['verified_fixed_tests'] == 48
    out = p/'runs'/RUN/db/'nre_cmm_interaction_v1'
    cache = p/'cache'/RUN/'nre_cmm_interaction_v1'/db
    original_cache = current_path.parent/'bootstrap_cache'
    temp = p/'work'/RUN/'tmp'
    for folder in (out, cache, temp):
        assert folder.resolve().is_relative_to(p)
    assert shutil.disk_usage(p).free > 10.05 * 2**30
    identity = {k: d[k] for k in ('database', 'current_sha256', 'old_sha256', 'analysis_sha256')}
    identity.update(replicates=B, interaction_family=6, simple_effect_family=24,
                    rng_dev=20261026, rng_test=20261027)
    old_artifacts = old.get('artifacts', old.get('checked_result_artifacts'))
    old_names = ({'full': 'noEvidence_EMA', 'NRE_only': 'noEvidence_EMA_noM', 'CMM_only': 'noEvidence_EMA_noP'}
                 if db == 'mimiciii' else {'full': 'fullEMA', 'NRE_only': 'EMA_noM', 'CMM_only': 'EMA_noP'})
    old_lookup = {(r['split'], r['method'], r['seed'], r['endpoint']): r for r in old['rows']}
    new_lookup = {(r['split'], r['method'], r['seed'], r['endpoint']): r for r in current['rows']}
    targets, cache_meta, cached, references = {}, {}, {}, {}
    for split in ('dev', 'test'):
        for cell, method in (('full', 'CEM-Net'), ('neither', 'EMA_no_NRE_CMM')):
            meta_path = original_cache/(split+'_'+method.replace('-', '_')+'.json')
            draws_path = meta_path.with_suffix('.npz')
            meta = read(meta_path)
            assert meta['identity']['method'] == method
            assert meta['identity']['database'] == db and meta['identity']['split'] == split
            assert meta['identity']['replicates'] == B and meta['identity']['rng_seed'] == 20261026+(split == 'test')
            assert sha(draws_path) == meta['draws_sha256']
            cache_meta[split, cell] = meta
            cached[split, cell] = dict(np.load(draws_path, allow_pickle=False))
            references[split, cell] = meta['identity']['predictions']
        target_sha = cache_meta[split, 'full']['identity']['target_sha256']
        assert cache_meta[split, 'neither']['identity']['target_sha256'] == target_sha
        paths = [Path(name) for name, h in current['artifacts'].items() if h == target_sha and name.endswith(split+'_targets.npz')]
        assert paths and sha(paths[0]) == target_sha
        targets[split] = dict(np.load(paths[0], allow_pickle=False))
        for cell in ('NRE_only', 'CMM_only'):
            refs = {}
            for s in SEEDS:
                for e in (0, 1):
                    row = old_lookup[split, old_names[cell], s, e]
                    paths = [name for name, h in old_artifacts.items() if h == row['predictions_sha256'] and name.endswith('predictions_'+split+'.npz')]
                    assert paths
                    refs[f'{s}:{e}'] = dict(path=paths[0], sha256=row['predictions_sha256'])
            references[split, cell] = refs
    assert not np.intersect1d(targets['dev']['ids'][:, 0], targets['test']['ids'][:, 0]).size
    if d.get('inspect'):
        return dict(complete=True, inspection_only=True, database=db,
                    rows={s: len(t['ids']) for s, t in targets.items()},
                    reusable_method_split_caches=4, new_method_split_caches=4,
                    expected_disk_gib=.05, outputs=str(out), cache=str(cache), temp=str(temp),
                    free_gib=shutil.disk_usage(p).free/2**30, no_training=True)
    for folder in (out, cache, temp):
        folder.mkdir(parents=True, exist_ok=True)
    with (out/'analysis.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = out/'interaction.json'
        receipt_path = out/'receipt.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            assert receipt['identity'] == identity and sha(result_path) == receipt['result_sha256']
            return dict(result=read(result_path), receipt=receipt, reused=True)
        y_test = np.array([1, 0, 1, 0, 1, 0])
        q_test = np.array([.8, .8, .5, .2, .2, .1])
        w_test = np.array([[1, 1, 1, 1, 1, 1], [0, 3, 2, 1, 2, 0], [2, 0, 0, 4, 1, 3]], float)
        np.testing.assert_allclose(weighted_ap(prepared_ap(y_test, q_test), w_test),
                                   [average_precision_score(y_test, q_test, sample_weight=w) for w in w_test], atol=1e-14, rtol=0)
        predictions, ap, verified = {}, {}, []
        max_ap_error, max_cache_error = 0., 0.
        for split in ('dev', 'test'):
            t = targets[split]
            for cell in CELLS:
                for s in SEEDS:
                    for e in (0, 1):
                        ref = references[split, cell][f'{s}:{e}']
                        assert sha(ref['path']) == ref['sha256']
                        z = dict(np.load(ref['path'], allow_pickle=False))
                        np.testing.assert_array_equal(z['ids'], t['ids'])
                        labels, q = z['labels'], z['probs']
                        if labels.ndim == 1:
                            labels, q = labels[:, None], q[:, None]
                        col = e if labels.shape[1] == 2 else 0
                        np.testing.assert_array_equal(labels[:, col], t['y'][:, e])
                        q = q[:, col]
                        assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
                        key = (split, cell, s, e)
                        predictions[key] = q
                        ap[key] = float(average_precision_score(t['y'][:, e], q))
                        if cell in ('full', 'neither'):
                            method = 'CEM-Net' if cell == 'full' else 'EMA_no_NRE_CMM'
                            row = new_lookup[split, method, s, e]
                        else:
                            row = old_lookup[split, old_names[cell], s, e]
                        error = abs(ap[key]-row.get('AP', row.get('ap')))
                        max_ap_error = max(max_ap_error, error)
                        assert error < 1e-12
                        if cell == 'full':
                            old_row = old_lookup[split, old_names[cell], s, e]
                            assert ref['sha256'] == old_row['predictions_sha256']
                        comp_path = Path(ref['path']).parent/'completion.json'
                        comp = read(comp_path)
                        assert comp['complete']
                        if split == 'test':
                            assert comp['optimizer_updates'] == 0
                            assert not comp.get('parameters_updated', False)
                        verified.append(dict(split=split, cell=cell, seed=s, endpoint=e,
                                             predictions_sha256=ref['sha256'], completion_sha256=sha(comp_path)))
            patients, inverse = np.unique(t['ids'][:, 0], return_inverse=True)
            n = len(patients)
            prepared = {(cell, s, e): prepared_ap(t['y'][:, e], predictions[split, cell, s, e])
                        for cell in CELLS for s in SEEDS for e in (0, 1)}
            rng = np.random.default_rng(20261026+(split == 'test'))
            first_weights = np.stack([np.bincount(rng.integers(0, n, n), minlength=n)[inverse] for _ in range(16)]).astype(float)
            for cell in ('full', 'neither'):
                for s in SEEDS:
                    for e in (0, 1):
                        actual = weighted_ap(prepared[cell, s, e], first_weights)
                        error = float(np.max(np.abs(actual-cached[split, cell][f'{s}:{e}'][:16])))
                        max_cache_error = max(max_cache_error, error)
                        assert error < 1e-12
            for cell in ('NRE_only', 'CMM_only'):
                cache_id = dict(identity, split=split, cell=cell, target_sha256=cache_meta[split, 'full']['identity']['target_sha256'], predictions=references[split, cell])
                meta_path, draws_path = cache/(split+'_'+cell+'.json'), cache/(split+'_'+cell+'.npz')
                if meta_path.exists():
                    meta = read(meta_path)
                    assert meta['identity'] == cache_id and sha(draws_path) == meta['draws_sha256']
                    draws = dict(np.load(draws_path, allow_pickle=False))
                else:
                    draws = {f'{s}:{e}': np.empty(B) for s in SEEDS for e in (0, 1)}
                    rng = np.random.default_rng(20261026+(split == 'test'))
                    for start in range(0, B, 16):
                        k = min(16, B-start)
                        weights = np.stack([np.bincount(rng.integers(0, n, n), minlength=n)[inverse] for _ in range(k)]).astype(float)
                        for s in SEEDS:
                            for e in (0, 1):
                                draws[f'{s}:{e}'][start:start+k] = weighted_ap(prepared[cell, s, e], weights)
                        if start % 160 == 0:
                            save(out/'progress.json', dict(database=db, split=split, cell=cell, completed=start+k, total=B, time=time.time()))
                    tmp_path = draws_path.with_name(draws_path.name+'.writing')
                    with tmp_path.open('wb') as stream:
                        np.savez_compressed(stream, **draws)
                    os.replace(tmp_path, draws_path)
                    save(meta_path, dict(identity=cache_id, draws_sha256=sha(draws_path)))
                cached[split, cell] = draws
        contrasts = dict(interaction={'full': 1, 'NRE_only': -1, 'CMM_only': -1, 'neither': 1},
                         NRE_with_CMM={'full': 1, 'CMM_only': -1}, NRE_without_CMM={'NRE_only': 1, 'neither': -1},
                         CMM_with_NRE={'full': 1, 'NRE_only': -1}, CMM_without_NRE={'CMM_only': 1, 'neither': -1})
        summary = {}
        for split in ('dev', 'test'):
            summary[split] = {}
            for e in (0, 1):
                entry = dict(cells={cell: stat([ap[split, cell, s, e] for s in SEEDS]) for cell in CELLS}, contrasts={})
                for name, coefficients in contrasts.items():
                    values = [sum(sign*ap[split, cell, s, e] for cell, sign in coefficients.items()) for s in SEEDS]
                    draws = np.mean([sum(sign*cached[split, cell][f'{s}:{e}'] for cell, sign in coefficients.items()) for s in SEEDS], axis=0)
                    family = 6 if name == 'interaction' else 24
                    alpha = .05/family
                    entry['contrasts'][name] = dict(stat(values), family=family, coverage=1-alpha,
                        ci95=np.quantile(draws, [.025, .975]).tolist(),
                        adjusted_ci=np.quantile(draws, [alpha/2, 1-alpha/2]).tolist())
                interaction = entry['contrasts']['interaction']['mean']
                assert abs(interaction-(entry['contrasts']['NRE_with_CMM']['mean']-entry['contrasts']['NRE_without_CMM']['mean'])) < 1e-12
                assert abs(interaction-(entry['contrasts']['CMM_with_NRE']['mean']-entry['contrasts']['CMM_without_NRE']['mean'])) < 1e-12
                summary[split][str(e)] = entry
        result = dict(complete=True, database=db, identity=identity, seeds=list(SEEDS), results=summary,
                      MLP_minus_CEM={s: current['paired'][s]['MLP_minus_CEM'] for s in ('dev', 'test')},
                      verification=dict(prediction_files=len(verified), AP_max_error=max_ap_error, reused_cache_max_error=max_cache_error,
                                        patient_ids_labels_exact=True, disjoint_validation_test=True, test_updates_zero=True,
                                        weighted_AP_self_test_passed=True, interaction_algebra_checked=True),
                      artifacts=verified, optimizer_updates=0, models_loaded=0, training_launched=False,
                      new_predictions_generated=False, patient_records_exported=False, main_method_changed=False,
                      bootstrap=dict(replicates=B, cluster='patient', common_draws_all_cells_seeds_endpoints=True,
                                     reused_method_split_caches=4, additional_method_split_caches=4,
                                     family_interaction=6, family_simple_effects=24, separate_validation_test_families=True,
                                     limitation='Post-hoc analysis conditional on fitted models and existing splits; excludes training-population and adaptive-selection uncertainty.'),
                      observed_at=time.time())
        save(result_path, result)
        receipt = dict(complete=True, identity=identity, result_sha256=sha(result_path), patient_records_exported=False)
        save(receipt_path, receipt)
        save(out/'progress.json', dict(status='complete', time=time.time()))
        return dict(result=result, receipt=receipt, reused=False)
