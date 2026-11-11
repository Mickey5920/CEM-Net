"""Verify the new aggregate evidence without patient data or model execution."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import numpy as np

RUN = 'cem_matched_controls_20261105'
DATABASES = ('mimiciii', 'mimic', 'eicu')
VARIANTS = ('matched_tabm', 'window_summary')
MANIFESTS = {
    'mimiciii': '7ecfd0f1cdfb59c71fb22f609e1d78a9b67e643e41450f0092b67555ef7e8d24',
    'mimic': '9be120d18fe4ede9d73da112ed1ab4169e7deb7e80e2aeabb11731cfca7b7c61',
    'eicu': '4930922fbbcadd50419ae1c61ef18696b641c3ca3e51463e65970379a5ae75a3'}


from source_loader import read
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_targeted(bundle):
    """Verify packaged targeted controls using saved aggregate draws only."""
    bundle = Path(bundle); run = 'cem_targeted_evidence_20261106'
    folder = bundle/'aggregate/runs'/run
    combined = read(folder/'paired_results_v1/verified-results.json')
    assert combined['complete'] and combined['new_fits'] == 51 and len(combined['rows']) == 135
    assert len(combined['paired']) == 51 and len(combined['subgroups']) == 270
    assert combined['protocol']['seeds'] == [48, 49, 50, 51, 52]
    for name, digest in read(folder/'paired_results_v1/completion.json')['files'].items():
        assert sha(folder/'paired_results_v1'/name) == digest
    metrics = {(r['database'], r['method'], r['endpoint'], r['seed'], 'all'): r['AP'] for r in combined['rows']}
    metrics.update({(r['database'], r['method'], r['endpoint'], r['seed'], r['group']): r['AP'] for r in combined['subgroups']})
    namespace = dict(np=np)
    for db in DATABASES:
        snapshot = bundle/'source/frozen_targeted_v1'/db
        mapping, encoded = read(snapshot/'source_manifest.json'), read(snapshot/'source_bundle.json')
        assert mapping.keys() == encoded.keys() and len(mapping) == 458
        assert sha(snapshot/'source_manifest.json') == read(snapshot/'build-receipt.json')['source_manifest_sha256']
        for name, digest in mapping.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert Path(name).suffix not in ('.npz', '.npy', '.pt', '.pth', '.parquet', '.pkl', '.joblib')
            assert hashlib.sha256(base64.b64decode(encoded[name])).hexdigest() == digest
        analysis = snapshot/'analysis_v2'
        for name, digest in read(analysis/'source_manifest.json').items():
            assert sha(analysis/name) == digest
        if not namespace.get('holm'):
            nodes = [n for n in ast.parse((analysis/'prior_significance.py').read_bytes()).body
                     if isinstance(n, ast.FunctionDef) and n.name in ('holm', 'centered_p')]
            assert len(nodes) == 2
            exec(compile(ast.Module(body=nodes, type_ignores=[]), 'targeted_significance', 'exec'), namespace)
        metadata = read(snapshot/'metadata.json')
        assert metadata['overlay']['threshold_fit_split'] == 'train' and not metadata['overlay']['labels_used']
        assert metadata['overlay']['changed_coordinates'] == 480
        assert len(metadata['fits']) == 17 and len(metadata['pilots']) == 4
        for job, records in metadata['fits'].items():
            fit, test, history = records['fit'], records['test'], records['history']
            assert fit['complete'] and fit['from_scratch'] and not fit['test_loaded'] and not fit['pilot']
            assert len(history) == fit['epochs'] == fit['epoch_budget']
            assert fit['best_epoch'] == 1+int(np.argmax([r['dev_ap'] for r in history]))
            assert fit['checkpoint_sha256'] == test['model_sha256'] == metadata['lock']['selections'][job]['checkpoint_sha256']
            assert test['optimizer_updates'] == 0 and not test['parameters_updated'] and test['dev_replay_max_drift'] <= 1e-5
            assert fit['source_manifest_sha256'] == sha(snapshot/'source_manifest.json')
        local = folder/(db+'_results_v1')
        for sub in ('analysis', 'cost'):
            for name, digest in read(local/sub/'completion.json')['files'].items():
                assert sha(local/sub/name) == digest
        result = read(local/'analysis/verified-results.json')
        assert result['new_fits'] == 17 and result['original_results_unchanged']
        assert result['rows'] == [r for r in combined['rows'] if r['database'] == db]
        draws = np.load(local/'analysis/paired_draws.npz', allow_pickle=False)
        selected = [r for r in combined['paired'] if r['database'] == db]
        assert len(draws.files) == len(selected) == 17
        for row in selected:
            values = [sum(w*metrics[db, m, row['endpoint'], s, g] for m, g, w in row['terms']) for s in range(48, 53)]
            np.testing.assert_allclose(values, row['values'], rtol=0, atol=1e-12)
            np.testing.assert_allclose([np.mean(values), np.std(values, ddof=1)], [row['mean'], row['sample_SD']], rtol=0, atol=1e-12)
            sample = draws[f"{row['name']}:{row['endpoint']}"]
            assert sample.shape == (20000,) and np.isfinite(sample).all()
            np.testing.assert_array_equal(np.quantile(sample, [.025, .975]), row['ci95'])
            n = row['family_size']
            np.testing.assert_array_equal(np.quantile(sample, [.05/(2*n), 1-.05/(2*n)]), row['adjusted_ci'])
            assert namespace['centered_p'](sample, row['mean']) == row['p_bootstrap_centered']
        cost = read(local/'cost/results.json')
        assert len(cost['rows']) == 27 and all(r['replay_passed'] and r['threads'] == 2 for r in cost['rows'])
        for cell in cost['summary']:
            rows = [r for r in cost['rows'] if (r['method'], r['stays']) == (cell['method'], cell['stays'])]
            assert len(rows) == 3 and len({r['records'] for r in rows}) == 1
            for key in cell.keys()-{'method', 'stays', 'records'}:
                assert np.median([r[key] for r in rows]) == cell[key]
        assert [dict(database=db, **r) for r in cost['summary']] == [r for r in combined['costs'] if r['database'] == db]
    for family, count in [('evidence', 9), ('aggregation', 6), ('subgroup_simple', 24), ('subgroup_interaction', 12)]:
        rows = [r for r in combined['paired'] if r['family'] == family]
        assert len(rows) == count and all(r['family_size'] == count for r in rows)
        np.testing.assert_array_equal(namespace['holm']([r['p_bootstrap_centered'] for r in rows]), [r['p_holm'] for r in rows])
    return dict(targeted_new_fits=51, targeted_contrasts=51, targeted_source_snapshots=3,
                targeted_cost_measurements=81, targeted_intervals_and_Holm_replayed=True)


def verify(bundle):
    bundle = Path(bundle); folder = bundle/'aggregate/runs'/RUN
    combined_dir = folder/'paired_results_v1'; combined = read(combined_dir/'verified-results.json')
    complete = read(combined_dir/'completion.json'); assert complete['complete']
    for name, digest in complete['files'].items(): assert sha(combined_dir/name) == digest
    assert combined['complete'] and combined['verified_new_fits'] == combined['verified_fixed_tests'] == 60
    assert combined['seeds'] == [48, 49, 50, 51, 52] and combined['replicates'] == 20000
    assert combined['family_count'] == 12 and len(combined['paired']) == 12 and len(combined['rows']) == 180
    assert combined['original_results_unchanged'] and combined['original_CEM_bootstrap_reused'] and combined['calibration_fits'] == 0
    statistics = bundle/'source/matched_statistics_v1'
    namespace = dict(np=np)
    tree = ast.parse((statistics/'prior_significance.py').read_text(encoding='utf8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('holm', 'centered_p')]
    assert len(nodes) == 2
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(statistics/'prior_significance.py'), 'exec'), namespace)
    analysis_manifest = read(statistics/'source_manifest.json')
    for name, digest in analysis_manifest.items(): assert sha(statistics/name) == digest
    assert read(statistics/'training_manifests.json') == MANIFESTS
    all_rows, all_paired, all_summaries = [], [], {}
    combined_draws = np.load(combined_dir/'paired_draws.npz', allow_pickle=False)
    expected_keys = {f'{db}:{v}:{e}' for db in DATABASES for v in VARIANTS for e in (0, 1)}
    assert set(combined_draws.files) == expected_keys
    for db in DATABASES:
        snapshot = bundle/'source/frozen_matched_v4'/db
        assert sha(snapshot/'source_manifest.json') == MANIFESTS[db]
        build = read(snapshot/'build-receipt.json'); assert build['source_manifest_sha256'] == MANIFESTS[db]
        encoded, mapping = read(snapshot/'source_bundle.json'), read(snapshot/'source_manifest.json')
        assert encoded.keys() == mapping.keys() and len(mapping) == 451
        inherited = read(bundle/'source/frozen_five_seed_v3'/db/'source_manifest.json')
        assert len(inherited) == 442 and all(mapping[name] == digest for name, digest in inherited.items())
        for name, digest in mapping.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert Path(name).suffix not in ('.npz', '.npy', '.pt', '.pth', '.parquet', '.pkl', '.joblib')
            assert hashlib.sha256(base64.b64decode(encoded[name])).hexdigest() == digest
        result_dir = folder/(db+'_results_v1'); receipt = read(result_dir/'collection-receipt.json')
        assert receipt['exit']['returncode'] == 0 and receipt['receipt']['complete']
        for name, digest in receipt['receipt']['files'].items(): assert sha(result_dir/name) == digest
        result = read(result_dir/'verified-results.json')
        assert result['source_manifest_sha256'] == sha(statistics/'source_manifest.json') == receipt['exit']['source_manifest_sha256']
        assert result['complete'] and result['verified_new_fits'] == result['verified_fixed_tests'] == 20
        assert result['paired_draws_sha256'] == sha(result_dir/'paired_draws.npz')
        assert result['original_results_unchanged'] and result['original_CEM_bootstrap_reused']
        assert {v for k, v in result['artifacts'].items() if k.endswith(f'/work/{RUN}/{db}/v4/source_manifest.json')} == {MANIFESTS[db]}
        fit_by_id = {r['id']: r for r in result['fit_audits']}
        assert set(fit_by_id) == {f'{v}_t{e}_s{s}' for v in VARIANTS for e in (0, 1) for s in range(48, 53)}
        for variant in VARIANTS:
            for endpoint in (0, 1):
                for seed in range(48, 53):
                    row = fit_by_id[f'{variant}_t{endpoint}_s{seed}']
                    budget = 256 if db == 'eicu' else 52 if db == 'mimic' else 120 if endpoint == 0 or seed in (49, 51) else 52
                    assert row['epochs'] == budget and 0 < row['best_epoch'] <= budget
                    assert row['fit_exit'] == row['test_exit'] == row['test_updates'] == 0
        old = read(bundle/'aggregate/runs/cem_five_seed_20261102'/(db+'_results_v1/verified-results.json'))
        original = {(r['split'], r['seed'], r['endpoint']): r for r in old['rows'] if r['method'] == 'CEM-Net'}
        for row in result['rows']:
            if row['method'] == 'CEM-Net':
                previous = original[row['split'], row['seed'], row['endpoint']]
                for metric in ('AP', 'AUROC', 'Brier', 'log_loss'):
                    assert abs(row[metric]-previous[metric]) < 1e-12
        draws = np.load(result_dir/'paired_draws.npz', allow_pickle=False)
        assert set(draws.files) == {f'{v}:{e}' for v in VARIANTS for e in (0, 1)}
        for row in result['paired']:
            key = f"{row['baseline']}:{row['endpoint']}"; sample = draws[key]
            assert sample.shape == (20000,) and np.isfinite(sample).all()
            np.testing.assert_array_equal(sample, combined_draws[db+':'+key])
            np.testing.assert_array_equal(np.quantile(sample, [.025, .975]), row['ci95'])
            np.testing.assert_array_equal(np.quantile(sample, [.05/24, 1-.05/24]), row['adjusted_ci'])
            assert namespace['centered_p'](sample, row['mean']) == row['p_bootstrap_centered']
            a = result['summaries'][f"test:CEM-Net:{row['endpoint']}"]['AP']['values']
            b = result['summaries'][f"test:{row['baseline']}:{row['endpoint']}"]['AP']['values']
            np.testing.assert_allclose(np.asarray(a)-b, row['values'], rtol=0, atol=1e-12)
            np.testing.assert_allclose([np.mean(row['values']), np.std(row['values'], ddof=1)], [row['mean'], row['sample_SD']], rtol=0, atol=1e-12)
        all_rows.extend(result['rows']); all_paired.extend(dict(r) for r in result['paired'])
        all_summaries.update({db+':'+key: value for key, value in result['summaries'].items()})
        resources = folder/(db+'_resources_v1'); cost = read(resources/'verified-results.json')
        collected = read(resources/'collection-receipt.json')
        assert collected['exit']['returncode'] == 0 and cost['complete']
        assert sha(resources/'verified-results.json') == collected['receipt']['files']['verified-results.json']
        assert cost['original_checkpoints_unchanged'] and cost['optimizer_updates'] == 0 and len(cost['rows']) == 20
        assert cost['source_manifest_sha256'] == MANIFESTS[db]
        assert cost['benchmark_source_sha256'] == sha(bundle/'source/matched_resources_v1/benchmark.py')
        assert cost['timing_source_sha256'] == sha(bundle/'source/matched_resources_v1/reference_inference.py')
        for row in cost['rows']:
            assert row['replay_max_drift'] <= 1e-5
            assert row['timing']['threads'] == 4 and row['timing']['warmup'] == 10
            assert {r['batch'] for r in row['timing']['rows']} == {1, 8, 32}
            if row['variant'] == 'matched_tabm':
                fit = fit_by_id[f"matched_tabm_t{row['endpoint']}_s{row['seed']}"]
                assert row['checkpoint_sha256'] == fit['checkpoint_sha256']
                assert row['timing']['parameters'] == fit['parameters']
            for batch in row['timing']['rows']:
                assert batch['resident_model']['repeats'] == batch['dense_transform_transfer_model']['repeats'] == 30
    for row, p in zip(all_paired, namespace['holm']([r['p_bootstrap_centered'] for r in all_paired])): row['p_holm'] = p
    assert combined['rows'] == all_rows and combined['paired'] == all_paired and combined['summaries'] == all_summaries
    return dict(matched_new_fits=60, matched_fixed_tests=60, matched_AP_contrasts=12,
                matched_family_size=12, matched_source_snapshots=3, matched_resource_checkpoints=60,
                matched_intervals_and_Holm_replayed=True, matched_original_scores_unchanged=True)


def verify_aggregation(bundle):
    """Replay aggregate arithmetic and inference provenance, without patient data."""
    import csv
    bundle = Path(bundle); run = 'cem_aggregation_stress_20261106'
    folder = bundle/'aggregate/runs'/run
    combined_dir = folder/'paired_results_v1'; combined = read(combined_dir/'verified-results.json')
    for name, digest in read(combined_dir/'completion.json')['files'].items():
        assert sha(combined_dir/name) == digest
    assert combined['training_runs'] == 0 and combined['new_inferences'] == 168 and combined['reused_inferences'] == 72
    assert combined['protocol']['seeds'] == [48, 49, 50, 51, 52] and combined['original_files_unchanged']
    conditions = ('lab_drop25', 'lab_drop50', 'lab_delay2h', 'lab_delay6h')
    combined_draws = np.load(combined_dir/'paired_draws.npz', allow_pickle=False)
    expected = {f'{db}:{estimand}:{e}:{c}' for db in DATABASES for estimand in ('degradation', 'absolute')
                for e in (0, 1) for c in conditions}
    assert set(combined_draws.files) == expected
    all_rows, all_pairs, namespace = [], [], dict(np=np)
    for db in DATABASES:
        result_dir = folder/(db+'_results_v1'); result = read(result_dir/'verified-results.json')
        for name, digest in read(result_dir/'completion.json')['files'].items():
            assert sha(result_dir/name) == digest
        assert read(result_dir/'exit.json')['returncode'] == 0
        assert result['complete'] and result['models'] == 20 and result['bootstrap_replicates'] == 20000
        assert result['training_runs'] == 0 and result['new_inferences'] == 56 and result['reused_inferences'] == 24
        snapshot = bundle/'source/frozen_aggregation_stress_v1'/db
        metadata = read(snapshot/'metadata.json'); current = snapshot/metadata['version']
        assert metadata['complete'] and sha(current/'worker.py') == result['source_sha256']
        assert read(current/'protocol.json') == result['run_protocol'] == combined['protocol']
        assert sha(current/'source_manifest.json') == read(result_dir/'exit.json')['source_manifest_sha256']
        workers = set()
        for version in snapshot.glob('v*'):
            mapping = read(version/'source_manifest.json')
            for name, digest in mapping.items(): assert sha(version/name) == digest
            workers.add(sha(version/'worker.py'))
        if not namespace.get('holm'):
            nodes = [n for n in ast.parse((current/'prior_significance.py').read_bytes()).body
                     if isinstance(n, ast.FunctionDef) and n.name in ('holm', 'centered_p')]
            assert len(nodes) == 2
            exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frozen_significance', 'exec'), namespace)
        matched = read(bundle/'aggregate/runs'/RUN/(db+'_results_v1/verified-results.json'))
        costs = read(bundle/'aggregate/runs'/RUN/(db+'_resources_v1/verified-results.json'))
        fits = {x['id']: x for x in matched['fit_audits']}
        receipts = metadata['receipts']
        assert set(receipts) == {f'{m}_t{e}_s{s}' for m in ('duration', 'window') for e in (0, 1) for s in range(48, 53)}
        for name, receipt in receipts.items():
            m, e, s = receipt['method'], receipt['endpoint'], receipt['seed']
            assert receipt['source_sha256'] in workers
            assert receipt['complete'] and receipt['weights_unchanged'] and receipt['optimizer_updates'] == 0
            assert not receipt['normalization_refit'] and set(receipt['predictions']) == set(conditions)
            if m == 'window': expected_hash = fits[f'window_summary_t{e}_s{s}']['checkpoint_sha256']
            else: expected_hash = next(x['checkpoint_sha256'] for x in costs['rows']
                                       if (x['variant'], x['endpoint'], x['seed']) == ('reference', e, s))
            assert receipt['checkpoint_sha256'] == expected_hash
            assert receipt['clean_probability_drift'] <= 1e-5 and abs(receipt['clean_AP_drift']) <= 1e-5
            if 'reused_completed_evaluation' in receipt:
                original = base64.b64decode(metadata['adopted_receipts'][name])
                assert hashlib.sha256(original).hexdigest() == receipt['reused_completed_evaluation']['sha256']
                previous = dict(receipt); previous.pop('reused_completed_evaluation')
                assert json.loads(original) == previous
        preparation = metadata['window_preparation']
        assert preparation['complete'] and not preparation['normalization_refit']
        assert len(preparation['checks']) == 5 and preparation['checks'][0]['clean_max_drift'] <= 2e-6
        draws = np.load(result_dir/'paired_draws.npz', allow_pickle=False)
        for cell in result['rows']:
            e, c = cell['endpoint'], cell['condition']
            for method, old_name in (('duration', 'CEM-Net'), ('window', 'window_summary')):
                metrics = cell['methods'][method]
                old = [next(x['AP'] for x in matched['rows'] if
                            (x['split'], x['method'], x['endpoint'], x['seed']) == ('test', old_name, e, s))
                       for s in range(48, 53)]
                np.testing.assert_allclose(old, metrics['clean']['values'], rtol=0, atol=1e-12)
                np.testing.assert_array_equal(np.asarray(metrics['perturbed']['values'])-metrics['clean']['values'],
                                              metrics['change']['values'])
                for value in metrics.values():
                    np.testing.assert_allclose([np.mean(value['values']), np.std(value['values'], ddof=1)],
                                               [value['mean'], value['sample_SD']], rtol=0, atol=1e-12)
        for pair in result['paired']:
            key = f"{pair['estimand']}:{pair['endpoint']}:{pair['condition']}"
            samples = draws[key]
            assert samples.shape == (20000,) and np.isfinite(samples).all()
            np.testing.assert_array_equal(samples, combined_draws[db+':'+key])
            np.testing.assert_array_equal(np.quantile(samples, [.025, .975]), pair['ci95'])
            np.testing.assert_array_equal(np.quantile(samples, [.05/48, 1-.05/48]), pair['adjusted_ci'])
            cell = next(x for x in result['rows'] if (x['endpoint'], x['condition']) == (pair['endpoint'], pair['condition']))
            a, b = cell['methods']['duration'], cell['methods']['window']
            expected_values = np.asarray(a['perturbed']['values'])-b['perturbed']['values']
            if pair['estimand'] == 'degradation':
                expected_values -= np.asarray(a['clean']['values'])-b['clean']['values']
            np.testing.assert_array_equal(expected_values, pair['values'])
            np.testing.assert_allclose([np.mean(expected_values), np.std(expected_values, ddof=1)],
                                       [pair['mean'], pair['sample_SD']], rtol=0, atol=1e-12)
            assert namespace['centered_p'](samples, pair['mean']) == pair['p_bootstrap_centered']
        all_rows.extend(result['rows']); all_pairs.extend(dict(x) for x in result['paired'])
    for estimand in ('degradation', 'absolute'):
        family = [x for x in all_pairs if x['estimand'] == estimand]
        assert len(family) == 24
        for row, p in zip(family, namespace['holm']([x['p_bootstrap_centered'] for x in family])): row['p_holm'] = p
    assert combined['rows'] == all_rows and combined['paired'] == all_pairs
    with (combined_dir/'comparisons.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 24
    for row in csv_rows:
        for estimand in ('degradation', 'absolute'):
            pair = next(x for x in all_pairs if (x['database'], x['endpoint'], x['condition'], x['estimand']) ==
                        (row['database'], int(row['endpoint']), row['condition'], estimand))
            for stat in ('mean', 'sample_SD', 'p_bootstrap_centered', 'p_holm'):
                assert float(row[estimand+'_'+stat]) == pair[stat]
    return dict(aggregation_fixed_models=60, aggregation_stress_cases=240, aggregation_new_inferences=168,
                aggregation_reused_inferences=72, aggregation_training_runs=0, aggregation_contrasts=48,
                aggregation_family_size=24, aggregation_intervals_and_Holm_replayed=True,
                aggregation_clean_scores_unchanged=True, aggregation_source_snapshots_verified=3)
