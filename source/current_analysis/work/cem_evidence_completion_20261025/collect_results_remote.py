"""Read-only collection of locked predictions; never imports a training entry point."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

RUN = 'cem_evidence_completion_20261025'
SEEDS = (48, 49, 50)
BASELINES = ('RealMLP', 'TabM', 'TabM-mini', 'TabM_mini_PLE', 'TabM_PLE', 'SeFT', 'GRU_D', 'mTAND_Enc')
CONTROL_NAMES = ('EMA_no_duration', 'EMA_NRE_MLP', 'EMA_no_NRE_CMM')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temp, path)


def same_process(pid, record):
    try:
        line = Path(f'/proc/{pid}/stat').read_text()
        tail = line[line.rfind(')')+2:].split()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return bool(record and tail[0] != 'Z' and tail[19] == record['start_ticks'] and boot == record['boot'])
    except (FileNotFoundError, ProcessLookupError):
        return False


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


def self_test():
    import numpy as np
    from sklearn.metrics import average_precision_score
    y = np.array([1, 0, 1, 0, 1, 0])
    q = np.array([.8, .8, .5, .2, .2, .1])
    weights = np.array([[1, 1, 1, 1, 1, 1], [0, 3, 2, 1, 2, 0], [2, 0, 0, 4, 1, 3]], float)
    actual = weighted_ap(prepared_ap(y, q), weights)
    expected = [average_precision_score(y, q, sample_weight=w) for w in weights]
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    return dict(weighted_AP_matches_sklearn_with_ties_and_zero_weights=True)


def scheduler_receipt(project, host, state):
    out = project/'runs'/RUN/('host_'+host)
    if not state.get('scheduler_recovery'):
        assert not state.get('active_scheduler_receipt')
        return out/'start_exit.json', None
    assert host == 'eicu'
    handoff = out/'test-dependency-handoff.json'
    assert state['scheduler_recovery'] == str(handoff)
    audit = read(handoff)
    assert audit['complete'] and audit['optimizer_updates'] == 0
    assert audit['training_jobs_preserved'] == 48 and audit['completed_tests_preserved'] == 30
    assert audit['retry_job'] == 'test_SeFT_t0_s48'
    predecessor = out/'test-dependency-predecessor-state.json'
    assert sha(predecessor) == audit['predecessor_state_sha256']
    previous = read(predecessor)
    assert previous['pid'] == audit['predecessor_pid']
    assert previous['groups'].keys() == state['groups'].keys()
    old_exit_path = out/'start_exit.json'
    assert sha(old_exit_path) == audit['predecessor_exit_sha256']
    old_exit = read(old_exit_path)
    assert old_exit['returncode'] == -15 and old_exit['child_pid'] == previous['pid']
    assert not same_process(old_exit['child_pid'], old_exit['child_identity'])
    failed = []
    for key, old_group in previous['groups'].items():
        current = state['groups'][key]
        assert current['source_manifest_sha256'] == audit['source_manifests'][key] == old_group['source_manifest_sha256']
        assert current['jobs'].keys() == old_group['jobs'].keys()
        assert all(j['status'] == 'completed' for j in old_group['jobs'].values() if j['action'] in ('train', 'probe'))
        for name, before in old_group['jobs'].items():
            after = current['jobs'][name]
            if before['status'] == 'completed':
                assert after == before, ('Completed job changed', name)
                continue
            assert before['action'] == after['action'] == 'evaluate'
            assert before['status'] in ('pending', 'failed')
            for field in ('id', 'variant', 'seed', 'endpoint', 'action', 'pilot_id', 'fit_id'):
                assert after[field] == before[field], (name, field)
            if before['status'] == 'failed':
                failed.append(name)
                assert before['output'] == audit['failed_output']
                assert before['log'] == audit['failed_log'] and before['exit_receipt'] == audit['failed_exit']
                assert after['output'] == audit['retry_output'] and after['execution_attempt'] == 2
            else:
                assert after['output'] == before['output']
    assert failed == [audit['retry_job']]
    assert sha(audit['failed_log']) == audit['failed_log_sha256']
    assert sha(audit['failed_exit']) == audit['failed_exit_sha256']
    failure = read(audit['failed_exit'])
    assert failure['returncode'] == audit['failed_returncode'] == 1
    assert not same_process(failure['child_pid'], failure['child_identity'])
    assert {f.name for f in Path(audit['failed_output']).iterdir()} == {'storage.json', 'progress.json'}
    assert sha(audit['fixed_checkpoint_lock']) == audit['fixed_checkpoint_lock_sha256']
    assert sha(project/'work'/RUN/'host_v1/cem_evidence_host_queue.py') == audit['original_queue_sha256']
    assert sha(audit['runtime']) == audit['runtime_sha256']
    assert sha(audit['dependency_install']) == audit['dependency_install_sha256']
    install = read(audit['dependency_install'])
    assert install['returncode'] == 0 and install['requested'] == 'pyarrow==23.0.1'
    assert install['target'] == audit['dependency_path']
    for name, h in install['files'].items():
        assert sha(Path(audit['dependency_path'])/name) == h
    check = read(Path(audit['runtime']).parent/'dependency-check.json')
    assert check == audit['dependency_check']
    assert check['complete'] and check['synthetic_filter_roundtrip'] and check['optimizer_updates'] == 0
    assert not check['patient_data_loaded'] and check['pyarrow'] == '23.0.1'
    active = out/'test_runtime_v1_exit.json'
    assert state['active_scheduler_receipt'] == audit['active_scheduler_receipt'] == str(active)
    if active.exists():
        assert read(active)['source_manifest_sha256'] == audit['runtime_sha256']
    return active, dict(verified=True, handoff=str(handoff), handoff_sha256=sha(handoff),
                        reason=audit['reason'], failed_test=audit['retry_job'], failed_attempt_preserved=True,
                        completed_tests_repeated=False, training_restarted=False,
                        dependency_version=check['pyarrow'], predecessor_exit=old_exit)


def collect(d):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
    p = Path(d['project']).resolve(strict=True)
    assert shutil.disk_usage(p).free > 10.5 * 2**30
    db = d['database']
    assert db in ('mimiciii', 'mimic', 'eicu')
    host = 'eicu' if db == 'eicu' else 'mimic'
    out = p/'runs'/RUN/db/'comparison_v1'
    cache = out/'bootstrap_cache'
    for folder in (out, cache):
        assert folder.resolve().is_relative_to(p)
        folder.mkdir(parents=True, exist_ok=True)
    with (out/'collector.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(out/'progress.json', dict(status='checking_completed_groups', database=db, time=time.time()))
        state = read(p/'runs'/RUN/('host_'+host)/'state.json')
        groups = {stage: state['groups'][db+'_'+stage] for stage in ('controls', 'tabular', 'sequence')}
        counts = {stage: dict(training=g['counts'], tests=g['evaluation_counts'], status=g['status']) for stage, g in groups.items()}
        assert all(not g['counts']['failed'] and not g['evaluation_counts']['failed'] and not g['probe_counts']['failed'] for g in groups.values()), 'A worker failed; investigate before collecting.'
        completed = [stage for stage, g in groups.items() if g['status'] == 'complete']
        assert 'controls' in completed
        exit_path, recovery = scheduler_receipt(p, host, state)
        host_exit = read(exit_path) if exit_path.exists() else None
        if host_exit is not None:
            assert host_exit['returncode'] == 0
            assert not same_process(host_exit['child_pid'], host_exit['child_identity'])
        signature = dict(analysis_sha256=d['analysis_sha256'], database=db,
                         completed_groups={stage: groups[stage]['source_manifest_sha256'] for stage in completed},
                         prior_controls_sha256=d['controls_sha256'], host_exit_sha256=sha(exit_path) if host_exit is not None else None,
                         execution_recovery_sha256=recovery['handoff_sha256'] if recovery else None)
        receipt_path = out/'collection-receipt.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt['signature'] == signature:
                for name, h in receipt['files'].items():
                    assert sha(out/name) == h
                return dict(receipt=receipt, reused=True, counts=counts,
                            files={name: (out/name).read_text() for name in receipt['files']})

        artifacts = {}
        def checked(path, expected=None):
            path = Path(path)
            value = artifacts.get(str(path)) or sha(path)
            if expected is not None:
                assert value == expected, str(path)
            artifacts[str(path)] = value
            return value

        csource = Path(groups['controls']['source'])
        c = read(csource/'work'/RUN/(db+'_expected.json'))
        prior = d['controls_verified']
        assert prior['complete'] and prior['new_fits'] == prior['fixed_tests'] == 18
        assert prior['source_manifest_sha256'] == groups['controls']['source_manifest_sha256']
        assert prior['AP_recompute_max_difference'] == 0 and prior['checkpoint_replay_max_difference'] <= 1e-5
        for path, h in c['protected'].items():
            checked(path, h)
        raw = Path(c['raw_inputs'])
        targets = {split: dict(np.load((raw if split != 'test' else p/c['test_inputs'])/(split+'_targets.npz'))) for split in ('train', 'dev', 'test')}
        for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
            assert not set(targets[a]['ids'][:, 0]) & set(targets[b]['ids'][:, 0])
        target_hashes = {split: checked((raw if split != 'test' else p/c['test_inputs'])/(split+'_targets.npz')) for split in targets}
        rows, preds, prediction_refs, audits = [], {}, {}, {}

        def prediction(path, split, method, seed, endpoint, expected_hash, meta=None):
            path = Path(path)
            h = checked(path, expected_hash)
            z = dict(np.load(path, allow_pickle=False))
            t = targets[split]
            np.testing.assert_array_equal(z['ids'], t['ids'])
            labels, q = z['labels'], z['probs']
            if labels.ndim == 1:
                labels, q = labels[:, None], q[:, None]
            column = endpoint if labels.shape[1] == 2 else 0
            np.testing.assert_array_equal(labels[:, column], t['y'][:, endpoint])
            q = q[:, column]
            assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
            key = (split, method, seed, endpoint)
            assert key not in preds, key
            preds[key] = q
            prediction_refs[key] = dict(path=str(path), sha256=h, column=column)
            y = t['y'][:, endpoint]
            scores = dict(AP=float(average_precision_score(y, q)), AUROC=float(roc_auc_score(y, q)),
                          Brier=float(brier_score_loss(y, q)), log_loss=float(log_loss(y, q, labels=[0, 1])))
            rows.append(dict(split=split, method=method, seed=seed, endpoint=endpoint, **scores, **(meta or {})))
            return scores

        # Original main method and the 54 controls reuse the previous complete
        # scientific audit. Their immutable output hashes are checked again.
        prior_lookup = {(r['split'], r['method'], r['seed'], r['endpoint']): r for r in prior['rows']}
        for seed in SEEDS:
            for endpoint in (0, 1):
                ref = c['full_ema'][f'{seed}:{endpoint}']
                fit = Path(ref['output'])
                comp = read(fit/'completion.json')
                checked(fit/comp.get('checkpoint_file', 'best.pt'), comp['checkpoint_sha256'])
                test = fit.parent.parent/'test'/('test_'+fit.name)
                testcomp = read(test/'completion.json')
                assert testcomp['optimizer_updates'] == 0
                for split, folder, rc in (('dev', fit, comp), ('test', test, testcomp)):
                    scores = prediction(folder/f'predictions_{split}.npz', split, 'CEM-Net', seed, endpoint, rc['predictions_sha256'], dict(reused=True))
                    assert abs(scores['AP']-prior_lookup[split, 'CEM-Net', seed, endpoint]['AP']) < 1e-12

        for stage in completed:
            g = groups[stage]
            source, queue = Path(g['source']), Path(g['output'])
            checked(source/'source_manifest.json', g['source_manifest_sha256'])
            for name, h in read(source/'source_manifest.json').items():
                checked(source/name, h)
            expected_count = len(g['variants'])*6
            assert g['counts'] == g['evaluation_counts'] == dict(pending=0, running=0, completed=expected_count, failed=0)
            locked = read(queue/'fixed-checkpoint-lock.json')
            assert locked['complete'] and not locked['test_used_for_selection'] and len(locked['selections']) == expected_count
            checked(queue/'fixed-checkpoint-lock.json')
            config = read(source/'work'/RUN/(db+('_expected.json' if stage == 'controls' else '_'+stage+'_expected.json')))
            for path, h in config['protected'].items():
                checked(path, h)
            if stage != 'controls':
                prep = p/config['baseline_preparation']
                prep_record = read(prep/'completion.json')
                assert prep_record['complete'] and not prep_record['test_loaded']
                checked(prep/'completion.json')
                for name, h in prep_record.get('artifacts', {}).items():
                    checked(prep/name, h)
                if stage == 'sequence':
                    test_prep = read(prep/'test-completion.json')
                    assert test_prep['complete'] and test_prep['optimizer_updates'] == 0 and test_prep['fit_split'] == 'train'
                    checked(prep/'test-completion.json')
                    for name, h in test_prep['artifacts'].items():
                        checked(prep/name, h)
            for job in g['jobs'].values():
                assert job['status'] == 'completed'
                rc = read(job['exit_receipt'])
                assert rc['returncode'] == 0 and rc['job_id'] == job['id']
                assert rc['source_manifest_sha256'] == g['source_manifest_sha256']
                assert not same_process(rc['child_pid'], rc['child_identity'])
                checked(job['exit_receipt'])
                admission = job['gpu_admission']
                assert admission['observed']['index'] == job['gpu']
                assert sum(v['project_worker'] for v in admission['observed']['processes']) < 2
                if not admission['preferred']:
                    assert admission['lease']['idle_seconds'] >= 900
                if job['action'] == 'probe':
                    audit = read(Path(job['output'])/'path_audit.json')
                    assert audit['complete'] and audit['continuous_updates'] == 20
                    continue
                if job['action'] != 'train':
                    continue
                fit = Path(job['output'])
                test = Path(g['jobs']['test_'+job['id']]['output'])
                comp, tc = read(fit/'completion.json'), read(test/'completion.json')
                manifest = read(fit/'run_manifest.json')
                identity = comp.get('identity', comp)
                for name in ('variant', 'seed', 'endpoint'):
                    assert identity[name] == job[name] and tc[name] == job[name]
                assert identity['source_manifest_sha256'] == tc['source_manifest_sha256'] == g['source_manifest_sha256']
                assert comp['complete'] and comp['from_scratch'] and not comp['pilot'] and not comp['test_loaded']
                assert comp['optimizer_updates'] > 0
                assert tc['complete'] and tc['optimizer_updates'] == 0 and not tc['parameters_updated']
                assert tc['dev_replay_max_drift'] <= 1e-5 and comp['checkpoint_replay']['max_probability_drift'] <= 1e-5
                assert locked['selections'][job['id']]['checkpoint_sha256'] == tc['model_sha256'] == comp['checkpoint_sha256']
                checked(fit/comp.get('checkpoint_file', 'best.pt'), comp['checkpoint_sha256'])
                checked(fit/'completion.json'); checked(test/'completion.json'); checked(fit/'run_manifest.json')
                history = read(fit/'history.json')
                assert len(history) == comp['epochs']
                if stage != 'controls':
                    spec = config['recipes'][job['variant']]
                    assert manifest['protocol'] == identity['protocol'] == spec
                    assert comp['frozen_learnable_parameters'] == 0
                    # Match the entry point's strict improvement and native patience.
                    best, stale, selected = -1., 0, None
                    for n, h in enumerate(history, 1):
                        assert h['epoch'] == n
                        if h['dev_ap'] > best+1e-9:
                            best, stale, selected = h['dev_ap'], 0, n
                        else:
                            stale += 1
                        assert stale < spec['patience'] or n == len(history)
                    assert selected == comp['best_epoch']
                    assert len(history) == spec['epochs'] or stale >= spec['patience']
                    assert comp['optimizer_updates'] == len(history)*int(np.ceil(len(targets['train']['y'])/spec['batch']))
                    pilot = read(Path(g['jobs'][job['pilot_id']]['output'])/'completion.json')['identity']
                    assert pilot['protocol'] == identity['protocol']
                    if job['seed'] == 48:
                        assert pilot['initial_weights_sha256'] == identity['initial_weights_sha256']
                    check = read(fit/'pilot_identity_check.json')
                    assert check['fresh_initialization'] and not check['pilot_weights_loaded']
                for split, folder, receipt in (('dev', fit, comp), ('test', test, tc)):
                    scores = prediction(folder/f'predictions_{split}.npz', split, job['variant'], job['seed'], job['endpoint'], receipt['predictions_sha256'],
                                        dict(reused=False, parameters=identity['parameters'], epochs=comp['epochs'], best_epoch=comp['best_epoch'],
                                             training_seconds=comp['seconds'], checkpoint_sha256=comp['checkpoint_sha256']))
                    recorded = comp['dev_ap'] if split == 'dev' else tc['metrics']['mean_ap']
                    assert abs(scores['AP']-recorded) < 1e-12
                    if stage == 'controls':
                        assert abs(scores['AP']-prior_lookup[split, job['variant'], job['seed'], job['endpoint']]['AP']) < 1e-12
                audits[job['id']] = dict(stage=stage, checkpoint_sha256=comp['checkpoint_sha256'], fixed_test_updates=0,
                                        validation_replay_max_drift=tc['dev_replay_max_drift'], all_worker_exits_zero=True)

        # The three retained strong baselines share seeds 48--50 and the target IDs.
        old_refs = {}
        if db == 'mimiciii':
            oldstate = read(p/'runs/mimiciii_pm_mini_20261024/v1/screen/state.json')
            oldtest = p/'runs/mimiciii_test_20261024/v3'
            oldconfig = read(oldtest/'evaluation_config.json')
            for model in ('RealMLP', 'TabM', 'TabM-mini'):
                for seed in SEEDS:
                    for endpoint in (0, 1):
                        if model == 'TabM-mini':
                            key = f'{model}_t{endpoint}_s{seed}'
                            fit = Path(oldstate['jobs'][key]['output'])
                            test = Path(oldstate['jobs']['test_'+key]['output'])
                        else:
                            key = f"{'B0' if model == 'RealMLP' else model}_s{seed}_t{endpoint}"
                            fit = Path(oldconfig['jobs'][key][0]['output'])
                            test = oldtest/'test'/key
                        for split, folder in (('dev', fit), ('test', test)):
                            rc = read(folder/'completion.json')
                            old_refs[split, model, seed, endpoint] = (folder/f'predictions_{split}.npz', rc['predictions_sha256'])
        else:
            for key, path in c['old_prediction_paths'].items():
                split, model, seed, endpoint = key.split(':')
                if model in ('RealMLP', 'TabM', 'TabM-mini'):
                    old_refs[split, model, int(seed), int(endpoint)] = (Path(path), c['protected'][path])
        for (split, model, seed, endpoint), (path, h) in old_refs.items():
            prediction(path, split, model, seed, endpoint, h, dict(reused=True))
        assert len(old_refs) == 36
        lookup = {(r['split'], r['method'], r['seed'], r['endpoint']): r for r in rows}
        assert len(lookup) == len(rows)
        methods = sorted({r['method'] for r in rows})
        for split in ('dev', 'test'):
            for method in methods:
                assert all((split, method, seed, endpoint) in lookup for seed in SEEDS for endpoint in (0, 1))
        stats = lambda values: dict(values=(a := np.asarray(list(values), float)).tolist(), mean=float(a.mean()), sample_SD=float(a.std(ddof=1)), above_zero=int((a > 0).sum()))
        summaries = {split: {m: {str(e): {metric: stats(lookup[split, m, s, e][metric] for s in SEEDS) for metric in ('AP', 'AUROC', 'Brier', 'log_loss')} for e in (0, 1)} for m in methods} for split in ('dev', 'test')}
        B = 4000
        self_test_result = self_test()
        samples = {}
        for split in ('dev', 'test'):
            patients, inverse = np.unique(targets[split]['ids'][:, 0], return_inverse=True)
            n = len(patients)
            for method in methods:
                safe = method.replace('-', '_')
                metadata = dict(database=db, split=split, method=method, replicates=B, rng_seed=20261026+(split == 'test'),
                                target_sha256=target_hashes[split], predictions={f'{s}:{e}': prediction_refs[split, method, s, e] for s in SEEDS for e in (0, 1)},
                                estimator='weighted sklearn-compatible AP v1; patient-cluster bootstrap; common draws')
                meta_path = cache/(split+'_'+safe+'.json')
                draws_path = cache/(split+'_'+safe+'.npz')
                if meta_path.exists():
                    old = read(meta_path)
                    assert old['identity'] == metadata and sha(draws_path) == old['draws_sha256']
                    draws = dict(np.load(draws_path))
                else:
                    save(out/'progress.json', dict(status='paired_patient_bootstrap', database=db, split=split, method=method, time=time.time()))
                    prepared = {(s, e): prepared_ap(targets[split]['y'][:, e], preds[split, method, s, e]) for s in SEEDS for e in (0, 1)}
                    for (s, e), item in prepared.items():
                        assert abs(weighted_ap(item, np.ones((1, len(inverse))))[0]-lookup[split, method, s, e]['AP']) < 1e-12
                    draws = {f'{s}:{e}': np.empty(B) for s in SEEDS for e in (0, 1)}
                    rng = np.random.default_rng(metadata['rng_seed'])
                    for start in range(0, B, 16):
                        k = min(16, B-start)
                        weights = np.stack([np.bincount(rng.integers(0, n, n), minlength=n)[inverse] for _ in range(k)]).astype(float)
                        for (s, e), item in prepared.items():
                            draws[f'{s}:{e}'][start:start+k] = weighted_ap(item, weights)
                    temp = draws_path.with_name(draws_path.name+'.writing')
                    with temp.open('wb') as stream:
                        np.savez_compressed(stream, **draws)
                    os.replace(temp, draws_path)
                    save(meta_path, dict(identity=metadata, draws_sha256=sha(draws_path)))
                for seed in SEEDS:
                    for endpoint in (0, 1):
                        samples[split, method, seed, endpoint] = draws[f'{seed}:{endpoint}']

        def pair(split, left, right, endpoint, family_count):
            r = stats(lookup[split, left, s, endpoint]['AP']-lookup[split, right, s, endpoint]['AP'] for s in SEEDS)
            draws = np.mean([samples[split, left, s, endpoint]-samples[split, right, s, endpoint] for s in SEEDS], axis=0)
            alpha = .05/family_count
            r.update(left=left, right=right, endpoint=endpoint, conditional_ci95=np.quantile(draws, [.025, .975]).tolist(),
                     family_count=family_count, family_coverage=1-alpha, family_adjusted_conditional_ci=np.quantile(draws, [alpha/2, 1-alpha/2]).tolist())
            return r
        paired, strongest = {}, {}
        available_baselines = [m for m in BASELINES if m in methods]
        for split in ('dev', 'test'):
            paired[split] = dict(MLP_minus_CEM={str(e): pair(split, 'EMA_NRE_MLP', 'CEM-Net', e, 6) for e in (0, 1)},
                                 candidate_minus_baseline={m: {b: {str(e): pair(split, m, b, e, 48) for e in (0, 1)} for b in available_baselines} for m in ('CEM-Net', 'EMA_NRE_MLP')},
                                 CEM_minus_deletion={m: {str(e): pair(split, 'CEM-Net', m, e, 12) for e in (0, 1)} for m in ('EMA_no_duration', 'EMA_no_NRE_CMM')})
            strongest[split] = {}
            for e in (0, 1):
                best = max(available_baselines, key=lambda m: summaries[split][m][str(e)]['AP']['mean'])
                valbest = max(available_baselines, key=lambda m: summaries['dev'][m][str(e)]['AP']['mean'])
                strongest[split][str(e)] = dict(empirical_mean_strongest=best, mean_AP=summaries[split][best][str(e)]['AP']['mean'], validation_mean_strongest=valbest,
                    comparisons={m: paired[split]['candidate_minus_baseline'][m][best][str(e)] for m in ('CEM-Net', 'EMA_NRE_MLP')},
                    against_validation_strongest={m: paired[split]['candidate_minus_baseline'][m][valbest][str(e)] for m in ('CEM-Net', 'EMA_NRE_MLP')},
                    by_seed=[dict(seed=s, method=(winner := max(available_baselines, key=lambda m: lookup[split, m, s, e]['AP'])), AP=lookup[split, winner, s, e]['AP']) for s in SEEDS],
                    pool_complete=len(available_baselines) == len(BASELINES))
        result = dict(complete=len(completed) == 3 and host_exit is not None, verified_available_results=True, database=db, counts=counts,
                      complete_groups=completed, expected_new_fits=48, verified_new_fits=sum(groups[s]['counts']['completed'] for s in completed),
                      verified_fixed_tests=sum(groups[s]['evaluation_counts']['completed'] for s in completed),
                      rows=rows, groups=summaries, paired=paired, strongest_baselines=strongest,
                      paired_baseline_pool=list(BASELINES), available_baselines=available_baselines, source_manifests=signature['completed_groups'],
                      prior_controls_audit_reused=d['controls_sha256'], artifacts=artifacts, worker_audits=audits,
                      host_queue_exit=host_exit, execution_recovery=recovery, fixed_checkpoint_tests_zero_updates=True, patient_records_exported=False,
                      weights_selected=False, training_launched=False, seed_probability_ensemble=False,
                      bootstrap=dict(replicates=B, patient_clusters=True, same_draws_for_all_models_and_seeds=True,
                                     validation_test_separate=True, MLP_CEM_family=6, each_candidate_baselines_family=48, duration_joint_deletion_family=12,
                                     baseline_family_prespecified_for_all_eight_including_pending=True, self_test=self_test_result,
                                     limitation='Conditional on fitted models and available splits; excludes training-population and adaptive-selection uncertainty. Extreme adjusted quantiles have limited Monte Carlo resolution.'),
                      observed_at=time.time())
        save(out/'verified-results.json', result)
        labels = {'CEM-Net': '原CEM-Net（完整EMA）', 'EMA_NRE_MLP': '等参数MLP候选', 'EMA_no_duration': '去持续时间统计', 'EMA_no_NRE_CMM': '同时去NRE/CMM', 'TabM_mini_PLE': 'TabM-mini＋PLE', 'TabM_PLE': 'TabM＋PLE', 'GRU_D': 'GRU-D', 'mTAND_Enc': 'mTAND监督编码器'}
        order = ['CEM-Net', 'EMA_NRE_MLP']+[m for m in BASELINES if m in methods]+['EMA_no_duration', 'EMA_no_NRE_CMM']
        lines = ['# '+{'mimiciii':'MIMIC-III','mimic':'MIMIC-IV','eicu':'eICU'}[db]+'：固定检查点核验与配对统计', '',
                 ('本库48项训练与48项固定测试已全部核验。' if result['complete'] else f"阶段结果：已核验{result['verified_new_fits']}/48项训练及对应固定测试，尚非本库最终结果。"),
                 '持续时间统计、CMM和EMA均保留在MLP候选中。原CEM-Net仍为当前主方法，未按本表切换。',
                 '主指标为AP；均值±样本标准差基于seed48、49、50。Validation和Test分别报告。', '']
        for split in ('dev', 'test'):
            lines += ['## '+('Validation' if split == 'dev' else 'Test'), '', '|方法|死亡 AP|LOS AP|', '|---|---:|---:|']
            for m in order:
                values = [summaries[split][m][str(e)]['AP'] for e in (0, 1)]
                lines.append('|'+labels.get(m, m)+'|'+'|'.join(f"{v['mean']:.6f} ± {v['sample_SD']:.6f}" for v in values)+'|')
            lines += ['', '|逐种子AP|seed|死亡|LOS|', '|---|---:|---:|---:|']
            for m in order:
                for s in SEEDS:
                    lines.append(f"|{labels.get(m,m)}|{s}|{lookup[split,m,s,0]['AP']:.6f}|{lookup[split,m,s,1]['AP']:.6f}|")
            lines += ['', '|配对比较|终点|48/49/50差值|均值±样本SD|提高种子数|校正后的患者条件区间|', '|---|---|---|---|---:|---|']
            for e in (0, 1):
                best = strongest[split][str(e)]['empirical_mean_strongest']
                pairs = [paired[split]['MLP_minus_CEM'][str(e)]]+[paired[split]['candidate_minus_baseline'][m][best][str(e)] for m in ('CEM-Net', 'EMA_NRE_MLP')]
                for r in pairs:
                    ci = r['family_adjusted_conditional_ci']
                    lines.append('|'+labels.get(r['left'],r['left'])+' − '+labels.get(r['right'],r['right'])+'|'+['死亡','LOS'][e]+'|'+', '.join(f'{v:+.6f}' for v in r['values'])+f"|{r['mean']:+.6f} ± {r['sample_SD']:.6f}|{r['above_zero']}/3|[{ci[0]:+.6f}, {ci[1]:+.6f}]|")
                lines.append(f"\n{['死亡','LOS'][e]}：当前均值最强为{labels.get(best,best)}；按Validation均值选定的基线为{labels.get(strongest[split][str(e)]['validation_mean_strongest'],strongest[split][str(e)]['validation_mean_strongest'])}。")
        lines += ['', '4000次共同患者簇重采样；各训练种子的AP差先分别计算再求均值，不合并预测。MLP−原模型六项比较为一族；每个候选对八种保留/新增baseline的48项比较为一族；两类删除的12项比较另成族，Validation和Test分别计算。',
                  '最强baseline表是当前完整三种子结果的描述性排序；未完成方法不计作失败或低分。最终仍需全部八种baseline完成。所有逐baseline配对、95%及族校正区间、AUROC/Brier/log loss和成本记录见JSON。',
                  '患者条件区间不涵盖训练总体波动及反复选择的不确定性。旧历史基线种子与配对池不一致时单列，不强行建立同种子配对。', '']
        (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        receipt = dict(complete=result['complete'], verified_available_results=True, signature=signature, patient_records_exported=False,
                       files={name: sha(out/name) for name in ('verified-results.json', 'RESULTS.md')})
        save(receipt_path, receipt)
        save(out/'progress.json', dict(status='complete' if result['complete'] else 'partial_complete', database=db, time=time.time()))
        return dict(receipt=receipt, reused=False, counts=counts, files={name: (out/name).read_text() for name in receipt['files']})
