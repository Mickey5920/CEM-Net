"""Extend the existing core comparisons to seeds 48--52 without changing fits.

Archived predictions, bootstrap draws, comparator choices, resampling seeds and
multiplicity families are read-only. New AP draws are computed only when the
matching prediction/draw cache does not already exist. Test inference and
descriptive validation scores are separate. No manuscript file is written.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from prior_utilities import read, sha, save, functions, shared

RUN = 'cem_core_five_seed_20261107'
OLD = 'cem_evidence_completion_20261025'
ICDE = 'cem_icde_evidence_20261028'
FIVE = 'cem_five_seed_20261102'
SEEDS = (48, 49, 50, 51, 52)
DATABASES = ('mimiciii', 'mimic', 'eicu')
MAIN = ('RealMLP', 'TabM', 'TabM_mini', 'TabM_PLE', 'TabM_mini_PLE',
        'LightGBM_full', 'XGBoost_TD', 'SeFT', 'mTAND_Enc', 'STraTS_supervised')
CONTROL = ('EMA_noP', 'EMA_noM', 'EMA_no_NRE_CMM', 'EMA_no_duration', 'CEM_noEMA')
ALIASES = {'CEM': 'CEM-Net', 'TabM-mini': 'TabM_mini', 'B0': 'RealMLP',
           'fullEMA': 'CEM-Net', 'noEvidence_EMA': 'CEM-Net',
           'noEvidence': 'CEM_noEMA', 'noEMA': 'CEM_noEMA'}


def summary(values):
    a = np.asarray(values, float)
    assert a.ndim == 1 and len(a) in (3, 5) and np.isfinite(a).all()
    return dict(values=a.tolist(), mean=float(a.mean()), sample_SD=float(a.std(ddof=1)))


def tabm_protocol(manifest):
    """Compare the two archived manifest schemas by their training settings."""
    if 'protocol' in manifest:
        p = dict(manifest['protocol'])
        for key in ('database', 'variant', 'seed'):
            if key in p:
                assert p.pop(key) == manifest[key]
        if 'arch_type' in p:
            arch = p.pop('arch_type')
            assert p.get('architecture', arch) == arch
            p['architecture'] = arch
        if 'modification' in p:
            assert p.pop('modification') == 'only official arch_type differs from archived TabM'
            assert p['architecture'] == 'tabm-mini'
    else:
        assert manifest['family'] == 'tabm' and manifest['independent_endpoints']
        assert manifest['precision'] == 'FP32' and manifest['class_weighting'] == 'none'
        assert manifest['checkpoint_selection'] == 'exact endpoint dev AP'
        assert manifest['member_loss'] == 'mean of independent binary cross-entropies'
        assert manifest['prediction'] == 'mean sigmoid probability'
        r = manifest['recipe']
        assert set(r) == {'width', 'depth', 'k', 'dropout', 'lr', 'wd'}
        # AdamW, retained final batches and no numerical embeddings are fixed
        # by the archived train_tracehr_tabular.py source checked by the caller.
        p = dict(epochs=manifest['epochs'], batch=manifest['batch'],
                 k=r['k'], d_block=r['width'], n_blocks=r['depth'], dropout=r['dropout'],
                 optimizer='AdamW', lr=r['lr'], weight_decay=r['wd'],
                 patience=manifest['patience'], selection='endpoint dev AP',
                 precision='float32', loss='mean member BCE', tail='retained')
    assert p['selection'] in ('endpoint dev AP', 'validation AP')
    p['selection'] = 'endpoint dev AP'
    p.setdefault('micro', p['batch'])
    p.setdefault('architecture', 'tabm')
    p.setdefault('embedding', None)
    return p


def seedless_protocol(manifest):
    p = dict(manifest['protocol'])
    if 'seed' in p:
        assert p.pop('seed') == manifest['seed']
    if 'params' in p:
        params = dict(p['params'])
        for key in ('seed', 'bagging_seed', 'feature_fraction_seed', 'data_random_seed', 'random_state'):
            if key in params:
                assert params.pop(key) == manifest['seed'], (key, manifest['seed'])
        p['params'] = params
    return p


def weights(cluster, rng_seed, replicates, batch=16):
    unique, inverse = np.unique(cluster, return_inverse=True)
    n = len(unique)
    assert n >= 2
    rng = np.random.default_rng(rng_seed)
    for start in range(0, replicates, batch):
        yield start, np.stack([
            np.bincount(rng.integers(0, n, n), minlength=n)[inverse]
            for _ in range(min(batch, replicates-start))]).astype(float)


def hospital_components(target):
    """Exactly the archived patient-connected-hospital clustering rule."""
    hs = target['hospital_ids']
    parent = {int(h): int(h) for h in np.unique(hs)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for patient in np.unique(target['ids'][:, 0]):
        hospitals = np.unique(hs[target['ids'][:, 0] == patient])
        root = find(int(hospitals[0]))
        for h in hospitals[1:]:
            parent[find(int(h))] = root
    return np.asarray([find(int(h)) for h in hs])


class Audit:
    def __init__(self, project, database, source, output):
        self.project = project.resolve(strict=True)
        self.db, self.source, self.out = database, source, output
        self.protected, self.records = {}, {}
        self.refs, self.old_AP, self.draw_sources = {}, {}, defaultdict(list)
        self.targets, self.target_paths, self.clusters = {}, {}, {}
        self.rows, self.predictions, self.audits = [], {}, []
        self.api = shared(source)
        self.configs, self.pending, self.groups = {}, [], []

    def keep(self, path, digest=None):
        p = Path(path).resolve(strict=True)
        assert p.is_relative_to(self.project), str(p)
        if str(p) not in self.protected:
            self.protected[str(p)] = sha(p)
        assert digest is None or self.protected[str(p)] == digest, str(p)
        return p

    def record(self, path, digest=None):
        p = self.keep(path, digest)
        if str(p) not in self.records:
            self.records[str(p)] = read(p)
        return self.records[str(p)]

    def ref(self, arm, split, method, seed, endpoint, reference, old_ap=None):
        method = ALIASES.get(method, method)
        key = arm, split, method, int(seed), int(endpoint)
        r = {k: reference[k] for k in ('path', 'sha256')}
        self.keep(r['path'], r['sha256'])
        if key in self.refs:
            assert self.refs[key]['sha256'] == r['sha256'], (key, self.refs[key], r)
        else:
            self.refs[key] = r
        if old_ap is not None:
            assert key not in self.old_AP or abs(self.old_AP[key]-old_ap) < 1e-12
            self.old_AP[key] = float(old_ap)
        return key

    def cache(self, metadata, method=None):
        """Load an archived multi-cell cache and its prediction identities."""
        meta = self.record(metadata)
        ident = meta['identity']
        method = method or ident['method']
        split, arm = ident['split'], ident.get('arm', 'random')
        rng = ident.get('rng_seed', ident.get('rng_'+split))
        assert rng is not None
        dp = self.keep(Path(metadata).with_suffix('.npz'), meta.get('draws_sha256', meta.get('sha256')))
        for cell, ref in ident['predictions'].items():
            s, e = map(int, cell.split(':'))
            key = self.ref(arm, split, method, s, e, ref)
            self.draw_sources[key, rng].append(dict(path=str(dp), key=cell, n=ident['replicates']))

    def prior_inventory(self):
        p, db = self.project, self.db
        state = read(p/'runs'/RUN/('host_eicu' if db == 'eicu' else 'host_mimic')/'state.json')
        self.groups = [g for g in state['groups'].values() if g['database'] == db]
        assert len(self.groups) == (7 if db == 'eicu' else 6)
        for group in self.groups:
            source = Path(group['source'])
            manifest = self.record(source/'source_manifest.json', group['source_manifest_sha256'])
            # The source is immutable. Check each file once, including inherited code.
            for name, h in manifest.items():
                self.keep(source/name, h)
            cfg = self.record(source/'work'/RUN/'config.json')
            assert cfg['seeds'] == [51, 52] and cfg['original_three_seed_results_immutable']
            assert not cfg['hyperparameter_search'] and cfg['matrix'] == group['matrix']
            self.configs[group['stage']] = cfg
        cfg = self.configs['controls']
        for arm, acfg in [('random', cfg)] + ([('hospital', self.configs['hospital'])] if db == 'eicu' else []):
            for split in ('dev', 'test'):
                path = p/acfg['raw_inputs']/f'{split}_targets.npz' if split == 'dev' else p/acfg['test_inputs']/'test_targets.npz'
                self.target_paths[arm, split] = self.keep(path)
                self.targets[arm, split] = dict(np.load(path, allow_pickle=False))
                t = self.targets[arm, split]
                self.clusters[arm, split] = t['ids'][:, 0] if arm == 'random' else hospital_components(t)
            assert not np.intersect1d(self.targets[arm, 'dev']['ids'][:, 0], self.targets[arm, 'test']['ids'][:, 0]).size
        oldfolder = p/'runs'/ICDE/(db+'_statistics_v1')
        old = self.record(oldfolder/'verified-results.json')
        for f in sorted((oldfolder/'bootstrap').glob('*.json')):
            method = read(f)['identity']['method']
            if method in ('EMA_NRE_MLP', 'GRU_D'):
                continue
            self.cache(f)
        for r in old['rows']:
            key = r['arm'], r['split'], ALIASES.get(r['method'], r['method']), r['seed'], r['endpoint']
            if key in self.refs:
                self.old_AP[key] = r['AP']
        completion = p/'runs'/OLD/db/'comparison_v1'
        oldcomp = self.record(completion/'verified-results.json')
        for m in ('EMA_no_duration', 'EMA_no_NRE_CMM'):
            for split in ('dev', 'test'):
                self.cache(completion/'bootstrap_cache'/f'{split}_{m}.json')
        for r in oldcomp['rows']:
            key = 'random', r['split'], ALIASES.get(r['method'], r['method']), r['seed'], r['endpoint']
            if key in self.refs:
                assert key not in self.old_AP or abs(self.old_AP[key]-r['AP']) < 1e-12
                self.old_AP[key] = r['AP']
        for m, cell in (('EMA_noP', 'CMM_only'), ('EMA_noM', 'NRE_only')):
            for split in ('dev', 'test'):
                self.cache(p/'cache'/OLD/'nre_cmm_interaction_v1'/db/f'{split}_{cell}.json', m)
        replacement = p/'runs/cem_baseline_replacement_20261030'/(db+'_statistics_v1')
        rep = self.record(replacement/'verified-results.json')
        for m in ('XGBoost_TD', 'STraTS_supervised'):
            for split in ('dev', 'test'):
                self.cache(replacement/'bootstrap'/f'{split}_{m}.json')
        for r in rep['rows']:
            key = 'random', r['split'], ALIASES.get(r['method'], r['method']), r['seed'], r['endpoint']
            if key in self.refs:
                self.old_AP[key] = r['AP']
        self.legacy_inventory()
        # Reuse the already completed extra seeds and their AP caches.
        existing = p/'runs'/FIVE/(db+'_statistics_v1')
        r = self.record(existing/'verified-results.json')
        assert r['complete'] and r['seeds'] == list(SEEDS)
        for f in sorted((existing/'bootstrap').glob('*.json')):
            meta = self.record(f)
            i = meta['identity']
            key = self.ref('random', i['split'], i['method'], i['seed'], i['endpoint'], i['prediction'])
            dp = self.keep(f.with_suffix('.npz'), meta['sha256'])
            self.draw_sources[key, i['rng_seed']].append(dict(path=str(dp), key='AP', n=i['replicates']))
        for row in r['rows']:
            key = 'random', row['split'], ALIASES.get(row['method'], row['method']), row['seed'], row['endpoint']
            assert key in self.refs
            self.old_AP[key] = row['AP']
        targeted = p/'runs/cem_targeted_evidence_20261106'/db/'v1'
        for f in sorted((targeted/'analysis_v1/bootstrap').glob('CEM_basic_*.json')):
            meta = self.record(f)
            i = meta['identity']
            if i['seed'] < 51:
                continue
            key = self.ref('random', 'test', 'CEM_basic', i['seed'], i['endpoint'], dict(path=i['prediction'], sha256=i['prediction_sha256']))
            dp = self.keep(f.with_suffix('.npz'), meta['sha256'])
            self.draw_sources[key, i['rng_seed']].append(dict(path=str(dp), key='all', n=i['B']))
            fit = targeted/'screen/jobs'/f"CEM_basic_t{i['endpoint']}_s{i['seed']}"
            receipt = self.record(fit/'completion.json')
            self.ref('random', 'dev', 'CEM_basic', i['seed'], i['endpoint'], dict(path=str(fit/'predictions_dev.npz'), sha256=receipt['predictions_sha256']))

    def legacy_inventory(self):
        p, db = self.project, self.db
        folder = (p/'runs/mimiciii_ema_20261024/v2/results' if db == 'mimiciii'
                  else p/'runs/tracehr_ema_cross_20261024'/db/'v1/results')
        self.legacy(folder, 'legacy_EMA', {'noEvidence' if db == 'mimiciii' else 'noEMA': 'CEM_noEMA'}, 20261024)
        if db in ('mimiciii', 'eicu'):
            folder = p/'runs/tracehr_ema_internal_20261025'/db/('v1' if db == 'mimiciii' else 'v2')/'results'
            names = ['EMA_no_numeric_direct'] if db == 'mimiciii' else ['EMA_M_raw', 'EMA_M_periodic']
            self.legacy(folder, 'legacy_internal', {v: v for v in names}, 20261025)

    def legacy(self, folder, tag, needed, rng):
        result = self.record(folder/'verified-results.json')
        assert result['complete']
        artifacts = result.get('artifacts', result.get('checked_result_artifacts'))
        for r in result['rows']:
            if r['method'] not in needed:
                continue
            split, s, e, m = r['split'], r['seed'], r['endpoint'], r['method']
            h = r.get('predictions_sha256')
            paths = [name for name, digest in artifacts.items()
                     if name.endswith(f'predictions_{split}.npz') and
                     (digest == h if h else Path(name).parent.name in (f'{m}_s{s}_t{e}', f'{m}_t{e}_s{s}'))]
            assert paths, (split, s, e, m)
            self.ref('random', split, needed[m], s, e, dict(path=paths[0], sha256=artifacts[paths[0]]), r.get('AP', r.get('ap')))
        for split in ('dev', 'test'):
            f = self.project/'runs'/ICDE/(self.db+'_significance_v1')/f'{tag}_{split}.json'
            meta = self.record(f)
            i = meta['identity']
            assert i['rng_seed'] == rng+(split == 'test') and i['replicates'] == 4000
            assert i['legacy_result_sha256'] == self.protected[str((folder/'verified-results.json').resolve())]
            dp = self.keep(f.with_suffix('.npz'), meta['sha256'])
            for m in i['methods']:
                mapped = needed.get(m, ALIASES.get(m, m))
                for s in (48, 49, 50):
                    for e in (0, 1):
                        key = 'random', split, mapped, s, e
                        if key in self.refs:
                            self.draw_sources[key, i['rng_seed']].append(dict(path=str(dp), key=f'{m}:{s}:{e}', n=4000))

    def new_inventory(self, allow_pending):
        for g in self.groups:
            source, screen = Path(g['source']), Path(g['output'])
            cfg = self.configs[g['stage']]
            verification = self.record(screen.parent/'verification.json')
            assert verification['complete'] and verification['source_manifest_sha256'] == g['source_manifest_sha256']
            arm = 'hospital' if g['stage'] == 'hospital' else 'random'
            fits = [j for j in g['jobs'].values() if j['action'] == 'train']
            assert {(j['variant'], j['seed'], j['endpoint']) for j in fits} == {(j['variant'], j['seed'], j['endpoint']) for j in cfg['matrix']}
            for j in fits:
                tj, pj = g['jobs']['test_'+j['id']], g['jobs'][j['pilot_id']]
                if any(job['status'] != 'completed' for job in (j, tj, pj)):
                    self.pending.append(dict(database=self.db, stage=g['stage'], id=j['id'], train=j['status'], test=tj['status']))
                    continue
                release = self.record(screen/'formal_release.json')
                lock = self.record(screen/'fixed-checkpoint-lock.json')
                assert release['complete'] and lock['complete'] and not lock['test_used_for_selection']
                for job in (j, tj, pj):
                    assert self.record(job['exit_receipt'])['returncode'] == 0
                fit, test, probe = map(Path, (j['output'], tj['output'], pj['output']))
                self.record(probe/'path_audit.json', release['audit_sha256'][pj['id']])
                r, tc, identity, history = [self.record(path) for path in
                    (fit/'completion.json', test/'completion.json', fit/'run_manifest.json', fit/'history.json')]
                assert r['complete'] and r['from_scratch'] and not r['pilot'] and not r['test_loaded']
                assert identity['source_manifest_sha256'] == g['source_manifest_sha256']
                for k in ('variant', 'seed', 'endpoint'):
                    assert identity[k] == r[k] == tc[k] == j[k]
                checkpoint = self.keep(fit/r['checkpoint_file'], r['checkpoint_sha256'])
                assert self.protected[str(checkpoint)] == lock['selections'][j['id']]['checkpoint_sha256'] == tc['model_sha256']
                assert tc['complete'] and tc['optimizer_updates'] == 0 and not tc['parameters_updated'] and tc['dev_replay_max_drift'] <= 1e-5
                assert r['checkpoint_replay']['max_probability_drift'] <= 1e-5
                self.training_rules(j, cfg, verification, identity, r, history, fit, source)
                for split, folder, receipt in (('dev', fit, r), ('test', test, tc)):
                    self.ref(arm, split, j['variant'], j['seed'], j['endpoint'], dict(path=str(folder/f'predictions_{split}.npz'), sha256=receipt['predictions_sha256']), receipt['metrics']['mean_ap'])
                self.audits.append(dict(id=j['id'], database=self.db, stage=g['stage'], epochs=r['epochs'], best_epoch=r['best_epoch'], fit_seconds=r['seconds'],
                    reused_fit=r.get('reused_fit', False), checkpoint_sha256=r['checkpoint_sha256'], source_manifest_sha256=g['source_manifest_sha256'],
                    fit_completion_sha256=self.protected[str((fit/'completion.json').resolve())], test_completion_sha256=self.protected[str((test/'completion.json').resolve())],
                    fit_exit=0, test_exit=0, test_updates=0))
            if not allow_pending:
                assert g['status'] == 'complete', (g['database'], g['stage'], g['status'])
        if not allow_pending:
            assert not self.pending
            assert len(self.audits) == (80 if self.db == 'eicu' else 62)

    def training_rules(self, j, cfg, verification, identity, r, history, fit, source):
        v, s, e = j['variant'], j['seed'], j['endpoint']
        assert len(history) == r['epochs']
        archived = cfg.get('archived_controls', {}).get(f'{v}:{e}')
        if archived:
            original = self.record(Path(archived)/'run_manifest.json')
            if v in ('TabM', 'TabM_mini'):
                if 'protocol' not in original:
                    self.keep(source/'train_tracehr_tabular.py', original['source_sha256'])
                assert tabm_protocol(identity) == tabm_protocol(original), (v, s, e)
            elif 'protocol' in original:
                assert seedless_protocol(identity) == seedless_protocol(original), (v, s, e)
        if v.startswith(('EMA_', 'CEM_')):
            assert len(history) == cfg['budgets'][f'{s}:{e}'] == identity['epoch_budget']
            assert identity['initial_weights_sha256'] == verification['initials'][f'{v}:{s}:{e}']['initial_weights_sha256']
            obs = self.record(fit/'initialization_observation.json')
            assert obs['initial_weights_sha256'] == identity['initial_weights_sha256'] and not obs.get('learned_checkpoint_loaded', False)
            assert r['best_epoch'] == 1+int(np.argmax([z['dev_ap'] for z in history]))
            if v == 'CEM_noEMA':
                assert r['parameter_source'] == 'raw'
                if r.get('reused_fit'):
                    reuse = self.record(fit/'reuse.json')
                    self.keep(reuse['source'], reuse['source_sha256'])
                    assert reuse['raw_validation_winner'] and reuse['new_training_updates'] == 0
                else:
                    proof = self.record(fit/'raw_trajectory_replay.json')
                    assert proof['complete'] and proof['all_epochs_AP_match'] and proof['max_AP_difference'] < 1e-10
            else:
                assert r['parameter_source'] == 'EMA'
                assert r['optimizer_updates'] == r['ema_updates'] == len(history)*math.ceil(cfg['train_rows']/256)
        elif v == 'RealMLP':
            ns = {}
            functions(source/'scripts/tracehr_centered_native_common.py', ('stopping',), ns)
            for n in range(1, len(history)+1):
                assert bool(ns['stopping'](history[:n], identity['protocol'])['stop']) == (n == len(history))
        else:
            tree = v.startswith(('LightGBM', 'XGBoost'))
            protocol = identity['protocol']
            maximum = protocol.get('max_rounds', protocol.get('epochs'))
            patience = protocol.get('early_stopping_rounds', protocol.get('patience'))
            assert isinstance(maximum, int) and isinstance(patience, int), (v, protocol)
            best, selected, stale = -1., 0, 0
            for n, h in enumerate(history, 1):
                assert h['iteration' if tree else 'epoch'] == n
                if h['dev_ap'] > best+(0 if tree else 1e-9):
                    best, selected, stale = h['dev_ap'], n, 0
                else:
                    stale += 1
                assert stale < patience or n == len(history)
            assert selected == r['best_epoch']
            assert len(history) == maximum or stale == patience

    def metrics(self):
        for key, reference in sorted(self.refs.items()):
            arm, split, method, seed, endpoint = key
            t = self.targets[arm, split]
            z = dict(np.load(reference['path'], allow_pickle=False))
            np.testing.assert_array_equal(z['ids'], t['ids'])
            y, q = z['labels'], z['probs']
            column = endpoint if y.ndim == 2 and y.shape[1] == 2 else 0
            y = y if y.ndim == 1 else y[:, column]
            q = q if q.ndim == 1 else q[:, column]
            np.testing.assert_array_equal(y, t['y'][:, endpoint])
            assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
            m = self.api['scores'](y, q)
            if key in self.old_AP:
                assert abs(m['AP']-self.old_AP[key]) < 1e-12, key
            self.rows.append(dict(database=self.db, arm=arm, split=split, method=method, seed=seed, endpoint=endpoint, **m))
            self.predictions[key] = q
        self.lookup = {(r['arm'], r['split'], r['method'], r['seed'], r['endpoint']): r for r in self.rows}


def contrast_specifications(db):
    specs = []

    def add(name, coefficients, family, count, e, n=4000, rng=20261027, arm='random'):
        specs.append(dict(name=name, coefficients=coefficients, family=family, family_count=count,
                          endpoint=e, replicates=n, rng_seed=rng, arm=arm, split='test'))

    for e in (0, 1):
        add('interaction', {'CEM-Net': 1, 'EMA_noM': -1, 'EMA_noP': -1, 'EMA_no_NRE_CMM': 1}, 'response_modulation_interaction', 6, e)
        for name, coeff in {
            'responses_with_modulation': {'CEM-Net': 1, 'EMA_noP': -1},
            'responses_without_modulation': {'EMA_noM': 1, 'EMA_no_NRE_CMM': -1},
            'modulation_with_responses': {'CEM-Net': 1, 'EMA_noM': -1},
            'modulation_without_responses': {'EMA_noP': 1, 'EMA_no_NRE_CMM': -1}}.items():
            add(name, coeff, 'response_modulation_simple', 24, e)
        for m in ('EMA_no_duration', 'EMA_no_NRE_CMM'):
            add('CEM-Net - '+m, {'CEM-Net': 1, m: -1}, 'duration_joint_deletion', 12, e)
        baseline = {'mimiciii': ('TabM_PLE', 'TabM_mini'), 'mimic': ('TabM_PLE',)*2, 'eicu': ('TabM_mini_PLE',)*2}[db][e]
        for name, coeff in {
            'CEM_full_minus_basic': {'CEM-Net': 1, 'CEM_basic': -1},
            'baseline_full_minus_basic': {baseline: 1, baseline+'_basic': -1},
            'interaction': {'CEM-Net': 1, 'CEM_basic': -1, baseline: -1, baseline+'_basic': 1}}.items():
            add(name, coeff, 'evidence_original_recipe', 18, e, n=20000)
        for m in ('XGBoost_TD', 'STraTS_supervised'):
            add('CEM-Net - '+m, {'CEM-Net': 1, m: -1}, 'replacement_baselines', 12, e, n=20000)
        if db in ('mimiciii', 'eicu'):
            for m in (('EMA_no_numeric_direct',) if db == 'mimiciii' else ('EMA_M_raw', 'EMA_M_periodic')):
                add('CEM-Net - '+m, {'CEM-Net': 1, m: -1}, 'coordinate_paths', 6, e, rng=20261026)
        if db == 'eicu':
            add('raw-by-encoded modulation', {'CEM-Net': 1, 'EMA_M_raw': -1, 'EMA_M_periodic': -1, 'EMA_noM': 1}, 'raw_encoded_interaction', 2, e, rng=20261026)
            for m in ('TabM_mini_PLE_hospital', 'LightGBM_hospital'):
                add('CEM_hospital - '+m, {'CEM_hospital': 1, m: -1}, 'hospital', 4, e, n=20000, arm='hospital')
        # Keep the original EMA comparison families and their original draws.
        for m in ('CEM_noEMA', 'RealMLP', 'TabM', 'TabM_mini'):
            family, count = (('legacy_III_EMA_and_baselines', 8) if db == 'mimiciii' else
                             ('legacy_IV_eICU_EMA', 4) if m == 'CEM_noEMA' else ('legacy_IV_eICU_baselines', 12))
            add('CEM-Net - '+m, {'CEM-Net': 1, m: -1}, family, count, e, rng=20261025)
    return specs


def cache_tasks(audit, specs):
    tasks = {}
    for spec in specs:
        for m in spec['coefficients']:
            for s in SEEDS:
                key = spec['arm'], spec['split'], m, s, spec['endpoint']
                if key not in audit.refs:
                    continue
                taskkey = (*key, spec['rng_seed'])
                tasks[taskkey] = max(tasks.get(taskkey, 0), spec['replicates'])
    return tasks


def materialize(audit, tasks):
    """Caches contain AP samples, never patient records or pooled predictions."""
    cache = audit.out/'bootstrap'
    cache.mkdir(exist_ok=True)
    values, loaded = {}, {}
    for index, (taskkey, count) in enumerate(sorted(tasks.items())):
        key, rng_seed = taskkey[:-1], taskkey[-1]
        arm, split, m, s, e = key
        cluster = audit.clusters[arm, split]
        prepared = audit.api['prepared_ap'](audit.targets[arm, split]['y'][:, e], audit.predictions[key])
        reference = audit.refs[key]
        ident = dict(database=audit.db, arm=arm, split=split, method=m, seed=s, endpoint=e,
                     rng_seed=rng_seed, replicates=count, prediction=reference,
                     cluster_sha256=hashlib.sha256(cluster.tobytes()).hexdigest())
        filename = f'{arm}_{split}_{m}_s{s}_t{e}_r{rng_seed}_b{count}'
        mp, dp = cache/(filename+'.json'), cache/(filename+'.npz')
        existing = sorted(audit.draw_sources[key, rng_seed], key=lambda x: x['n'], reverse=True)
        _, first = next(weights(cluster, rng_seed, 16))
        if mp.exists():
            meta = read(mp)
            assert meta['identity'] == ident and sha(dp) == meta['draws_sha256']
            result = np.load(dp, allow_pickle=False)['AP']
        elif existing and existing[0]['n'] >= count:
            src = existing[0]
            if src['path'] not in loaded:
                loaded[src['path']] = dict(np.load(src['path'], allow_pickle=False))
            result = loaded[src['path']][src['key']][:count]
        else:
            assert not dp.exists(), str(dp)
            result = np.empty(count)
            start_at = 0
            if existing:
                src = existing[0]
                if src['path'] not in loaded:
                    loaded[src['path']] = dict(np.load(src['path'], allow_pickle=False))
                start_at = src['n']
                result[:start_at] = loaded[src['path']][src['key']]
                np.testing.assert_allclose(audit.api['weighted_ap'](prepared, first), result[:16], rtol=0, atol=1e-12)
            for start, w in weights(cluster, rng_seed, count):
                if start < start_at:
                    continue
                result[start:start+len(w)] = audit.api['weighted_ap'](prepared, w)
                if start % 320 == 0:
                    save(audit.out/'progress.json', dict(stage='AP_resampling', task=index+1, tasks=len(tasks), method=m, seed=s,
                                                       endpoint=e, replicates_done=start+len(w), replicates=count, time=time.time()))
            tmp = dp.with_suffix('.npz.writing')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, AP=result)
            os.replace(tmp, dp)
            save(mp, dict(identity=ident, draws_sha256=sha(dp), reused_prefix=start_at))
        assert result.shape == (count,) and np.isfinite(result).all()
        np.testing.assert_allclose(audit.api['weighted_ap'](prepared, first), result[:16], rtol=0, atol=1e-12)
        values[taskkey] = result
    return values


def contrasts(audit, specs, samples):
    rows = []
    for spec in specs:
        arm, split, e, rng, n = [spec[k] for k in ('arm', 'split', 'endpoint', 'rng_seed', 'replicates')]
        coeff = spec['coefficients']
        if any((arm, split, m, s, e, rng) not in samples for m in coeff for s in SEEDS):
            continue
        row = dict(database=audit.db, **spec)
        for seeds, label in ((SEEDS[:3], 'three_seed'), (SEEDS, 'five_seed')):
            points = [sum(c*audit.lookup[arm, split, m, s, e]['AP'] for m, c in coeff.items()) for s in seeds]
            distribution = np.mean([sum(c*samples[arm, split, m, s, e, rng][:n] for m, c in coeff.items()) for s in seeds], axis=0)
            point = float(np.mean(points))
            a = .05/spec['family_count']
            pv = audit.api['centered_p'](distribution, point)
            interval = np.quantile(distribution, [a/2, 1-a/2]).tolist()
            row[label] = dict(**summary(points), seeds=list(seeds), ci95=np.quantile(distribution, [.025, .975]).tolist(),
                              adjusted_ci=interval, p_bootstrap_centered=pv, p_bonferroni=min(1., pv*spec['family_count']),
                              direction='positive' if interval[0] > 0 else 'negative' if interval[1] < 0 else 'crosses_zero',
                              wilcoxon=audit.api['exact_signed_rank'](points))
        row['mean_change'] = row['five_seed']['mean']-row['three_seed']['mean']
        row['adjusted_interval_direction_changed'] = row['five_seed']['direction'] != row['three_seed']['direction']
        rows.append(row)
    return rows


def replay_original_three(audit, specs, samples):
    """The extension must reproduce every original core interval before use."""
    previous = audit.record(audit.project/'runs'/ICDE/(audit.db+'_significance_v1')/'verified-results.json')
    replacement = audit.record(audit.project/'runs/cem_baseline_replacement_20261030'/(audit.db+'_statistics_v1')/'verified-results.json')
    families = {'response_modulation_interaction': 'NRE-CMM-interaction',
                'response_modulation_simple': 'NRE-CMM-simple',
                'evidence_original_recipe': 'evidence'}
    simple_names = {'responses_with_modulation': 'NRE_with_CMM',
                    'responses_without_modulation': 'NRE_without_CMM',
                    'modulation_with_responses': 'CMM_with_NRE',
                    'modulation_without_responses': 'CMM_without_NRE'}
    checked, maximum_error = 0, 0.
    for spec in specs:
        arm, split, e, rng, n = [spec[k] for k in ('arm', 'split', 'endpoint', 'rng_seed', 'replicates')]
        coeff = spec['coefficients']
        points = [sum(c*audit.lookup[arm, split, m, s, e]['AP'] for m, c in coeff.items()) for s in SEEDS[:3]]
        distribution = np.mean([sum(c*samples[arm, split, m, s, e, rng][:n] for m, c in coeff.items()) for s in SEEDS[:3]], axis=0)
        if spec['family'] == 'replacement_baselines':
            old = next(r for r in replacement['paired'] if r['split'] == split and r['endpoint'] == e
                       and r['candidate'] == 'CEM-Net' and r['baseline'] == spec['name'].split(' - ')[1])
            old_values, old_ci = old['values'], old['adjusted_ci']
        else:
            family = families.get(spec['family'], spec['family'])
            name = simple_names.get(spec['name'], spec['name'])
            if family == 'coordinate_paths':
                name = name.replace('CEM-Net - ', 'fullEMA - ')
            if family.startswith('legacy_'):
                baseline = spec['name'].split(' - ')[1]
                alias = {'CEM_noEMA': 'noEvidence' if audit.db == 'mimiciii' else 'noEMA',
                         'RealMLP': 'B0' if audit.db == 'mimiciii' else 'RealMLP', 'TabM_mini': 'TabM-mini'}
                name = ('noEvidence_EMA' if audit.db == 'mimiciii' else 'fullEMA')+' - '+alias.get(baseline, baseline)
            old = next(r for r in previous['rows'] if r['split'] == split and r['endpoint'] == e and r['family'] == family and r['contrast'] == name)
            old_values, old_ci = old['seed_differences'], old['original_adjusted_ci']
        assert old['family_count'] == spec['family_count']
        np.testing.assert_allclose(points, old_values, rtol=0, atol=1e-12)
        a = .05/spec['family_count']
        ci = np.quantile(distribution, [a/2, 1-a/2])
        error = float(np.max(np.abs(ci-old_ci)))
        maximum_error = max(maximum_error, error)
        assert error < 1e-12, (audit.db, spec, ci, old_ci)
        pv = audit.api['centered_p'](distribution, float(np.mean(points)))
        assert abs(pv-old['p_bootstrap_centered']) < 1e-12
        checked += 1
    return dict(complete=True, original_contrasts_reproduced=checked, maximum_adjusted_interval_error=maximum_error,
                original_point_differences_and_centered_p_reproduced=True)


def analyze(project, db, source, output, mode):
    import fcntl
    project = project.resolve(strict=True)
    assert output.resolve().is_relative_to(project)
    output.mkdir(parents=True, exist_ok=True)
    owner = (output/'writer.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output/'completion.json').exists():
        receipt = read(output/'completion.json')
        assert sha(output/'verified-results.json') == receipt['result_sha256']
        return dict(complete=True, reused=True, database=db)
    audit = Audit(project, db, source, output)
    audit.prior_inventory()
    audit.new_inventory(mode != 'analyze')
    audit.metrics()
    specs = contrast_specifications(db)
    tasks = cache_tasks(audit, specs)
    info = dict(database=db, verified_new_cells=len(audit.audits), pending_cells=len(audit.pending),
                metric_rows=len(audit.rows), AP_tasks=len(tasks), missing_cells=audit.pending,
                old_three_seed_predictions_unchanged=True)
    if mode == 'audit':
        save(output/'audit.json', info)
        return info
    samples = materialize(audit, tasks)
    prior_replay = replay_original_three(audit, specs, samples)
    paired = contrasts(audit, specs, samples)
    if mode == 'precompute':
        save(output/'precompute.json', dict(info, completed_AP_tasks=len(samples), original_replay=prior_replay, time=time.time()))
        return dict(info, completed_AP_tasks=len(samples), original_replay=prior_replay)
    assert len(paired) == len(specs)
    expected = {('random', m, e) for m in ('CEM-Net', *MAIN, *CONTROL, 'CEM_basic') for e in (0, 1)}
    for e in (0, 1):
        baseline = {'mimiciii': ('TabM_PLE', 'TabM_mini'), 'mimic': ('TabM_PLE',)*2,
                    'eicu': ('TabM_mini_PLE',)*2}[db][e]
        expected.add(('random', baseline+'_basic', e))
    for m in (('EMA_no_numeric_direct',) if db == 'mimiciii' else ('EMA_M_raw', 'EMA_M_periodic') if db == 'eicu' else ()):
        expected.update(('random', m, e) for e in (0, 1))
    if db == 'eicu':
        expected.update(('hospital', m, e) for m in ('CEM_hospital', 'TabM_mini_PLE_hospital', 'LightGBM_hospital') for e in (0, 1))
    assert set(audit.refs) == {(arm, split, m, s, e) for arm, m, e in expected for split in ('dev', 'test') for s in SEEDS}
    groups = defaultdict(list)
    for r in audit.rows:
        groups[r['arm'], r['split'], r['method'], r['endpoint']].append(r)
    summaries = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r['seed'])
        assert [r['seed'] for r in rows] == list(SEEDS), key
        record = dict(database=db, arm=key[0], split=key[1], method=key[2], endpoint=key[3])
        for seeds, label in ((SEEDS[:3], 'three_seed'), (SEEDS, 'five_seed')):
            record[label] = {m: summary([r[m] for r in rows if r['seed'] in seeds]) for m in ('AP', 'AUROC', 'Brier', 'log_loss')}
        summaries.append(record)
    for path, h in audit.protected.items():
        assert sha(path) == h, path
    result = dict(complete=True, **info, seeds=list(SEEDS), fit_audits=audit.audits, rows=audit.rows, summaries=summaries, paired=paired,
                  protected=audit.protected, original_replay=prior_replay, source_manifest_sha256=sha(source/'source_manifest.json'),
                  inference='Original patient/hospital clusters, RNGs, replicate counts and multiplicity families; within-seed AP contrasts averaged over five fitted seeds. Validation metrics are descriptive. Original comparator selection is retained. Intervals remain conditional on the fitted models and observed cohorts.',
                  manuscript_modified=False, model_modified=False, original_results_modified=False, patient_records_exported=False)
    save(output/'verified-results.json', result)
    save(output/'completion.json', dict(complete=True, result_sha256=sha(output/'verified-results.json')))
    return dict(complete=True, database=db, verified_cells=len(audit.audits), new_fits=sum(not r['reused_fit'] for r in audit.audits), comparisons=len(paired))


def aggregate(results, api):
    assert {r['database'] for r in results} == set(DATABASES)
    assert sum(r['verified_new_cells'] for r in results) == 204
    paired = [x for r in results for x in r['paired']]
    families = defaultdict(list)
    for row in paired:
        families[row['family']].append(row)
    for family, rows in families.items():
        assert len(rows) == rows[0]['family_count'], (family, len(rows), rows[0]['family_count'])
        for scope in ('three_seed', 'five_seed'):
            for row, pv in zip(rows, api['holm']([r[scope]['p_bootstrap_centered'] for r in rows])):
                row[scope]['p_holm_full_family'] = pv
    return dict(complete=True, seeds=list(SEEDS), verified_cells=204,
                new_fits=sum(not x['reused_fit'] for r in results for x in r['fit_audits']),
                reused_saved_fit=sum(x['reused_fit'] for r in results for x in r['fit_audits']),
                summaries=[x for r in results for x in r['summaries']], rows=[x for r in results for x in r['rows']], paired=paired,
                family_counts={k: len(v) for k, v in families.items()},
                inference=results[0]['inference'], manuscript_modified=False, original_results_modified=False)


def self_test(source):
    api = shared(source)
    from sklearn.metrics import average_precision_score
    y, q = np.array([1, 0, 1, 0, 1, 0]), np.array([.8, .8, .5, .2, .2, .1])
    w = np.array([[1, 1, 1, 1, 1, 1], [0, 3, 2, 1, 2, 0], [2, 0, 0, 4, 1, 3]])
    np.testing.assert_allclose(api['weighted_ap'](api['prepared_ap'](y, q), w),
                               [average_precision_score(y, q, sample_weight=z) for z in w], atol=1e-14, rtol=0)
    assert api['centered_p'](np.zeros(20), 0.) == 1.
    assert api['exact_signed_rank']([1, 2, 3, 4, 5])['p_two_sided'] == .0625
    fs = defaultdict(list)
    for db in DATABASES:
        for spec in contrast_specifications(db):
            fs[spec['family']].append(spec)
    assert all(len(v) == v[0]['family_count'] for v in fs.values())
    t = dict(ids=np.array([[1, 1], [1, 2], [2, 3], [3, 4]]), hospital_ids=np.array([4, 7, 8, 7]))
    c = hospital_components(t)
    assert c[0] == c[1] == c[3] and c[2] != c[0]
    assert summary([1, 2, 3, 4, 5])['mean'] == 3
    old_tabm = dict(database='mimiciii', variant='TabM', seed=48,
                    protocol=dict(database='mimiciii', variant='TabM', seed=48,
                                  batch=256, selection='endpoint dev AP', lr=.0005))
    new_tabm = dict(database='mimiciii', variant='TabM', seed=51,
                    protocol=dict(batch=256, micro=256, selection='validation AP',
                                  lr=.0005, architecture='tabm', embedding=None))
    assert tabm_protocol(old_tabm) == tabm_protocol(new_tabm)
    for key, value in (('micro', 128), ('lr', .001), ('architecture', 'tabm-mini')):
        changed = dict(new_tabm, protocol=dict(new_tabm['protocol'], **{key: value}))
        assert tabm_protocol(old_tabm) != tabm_protocol(changed)
    old_mini = dict(old_tabm, protocol=dict(old_tabm['protocol'], arch_type='tabm-mini',
                                          modification='only official arch_type differs from archived TabM'))
    new_mini = dict(new_tabm, protocol=dict(new_tabm['protocol'], architecture='tabm-mini'))
    assert tabm_protocol(old_mini) == tabm_protocol(new_mini)
    old_tree = dict(seed=48, protocol=dict(params=dict(seed=48, bagging_seed=48, learning_rate=.04)))
    new_tree = dict(seed=51, protocol=dict(params=dict(seed=51, bagging_seed=51, learning_rate=.04)))
    assert seedless_protocol(old_tree) == seedless_protocol(new_tree)
    changed_tree = dict(seed=51, protocol=dict(params=dict(seed=51, bagging_seed=51, learning_rate=.08)))
    assert seedless_protocol(old_tree) != seedless_protocol(changed_tree)
    try:
        seedless_protocol(dict(seed=51, protocol=dict(params=dict(seed=51, bagging_seed=52))))
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched tree RNG seeds must be rejected')
    return dict(complete=True, weighted_AP_ties_and_zero_weights=True, hospital_connectivity=True,
                unchanged_family_counts={k: len(v) for k, v in fs.items()}, five_seed_summary=True,
                tabm_manifest_schema_equivalence=True, tabm_setting_changes_detected=True,
                nested_tree_seed_change_only=True, inconsistent_tree_seeds_rejected=True)


def wait_for_fits(project, database, output):
    host = 'host_eicu' if database == 'eicu' else 'host_mimic'
    state = project/'runs'/RUN/host/'state.json'
    while True:
        s = read(state)
        groups = [g for g in s['groups'].values() if g['database'] == database]
        assert len(groups) == (7 if database == 'eicu' else 6)
        jobs = [j for g in groups for j in g['jobs'].values()]
        done = all(g['status'] == 'complete' for g in groups)
        save(output/'wait_progress.json', dict(database=database, ready_for_analysis=done,
            fitted_cells=sum(j['action'] == 'train' and j['status'] == 'completed' for j in jobs),
            fixed_tests=sum(j['action'] == 'evaluate' and j['status'] == 'completed' for j in jobs),
            failed_jobs=[j['id'] for j in jobs if j['status'] == 'failed'],
            groups={g['stage']: g['status'] for g in groups}, time=time.time()))
        if done:
            return
        time.sleep(30)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--database', choices=DATABASES, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('audit', 'precompute', 'analyze', 'self-test', 'wait'), required=True)
    a = p.parse_args()
    source = Path(__file__).resolve().parent
    if a.mode == 'wait':
        wait_for_fits(a.project, a.database, a.output)
        a.mode = 'analyze'
    result = self_test(source) if a.mode == 'self-test' else analyze(a.project, a.database, source, a.output, a.mode)
    print(json.dumps(result))
