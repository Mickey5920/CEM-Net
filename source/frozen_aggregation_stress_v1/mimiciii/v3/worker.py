"""Paired frozen CEM duration/window stress evaluation; never trains a model."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

RUN = 'cem_aggregation_stress_20261106'
MATCH = 'cem_matched_controls_20261105'
OLD = 'cem_icde_evidence_20261028'
FIVE = 'cem_five_seed_20261102'
CONDITIONS = ('lab_drop25', 'lab_drop50', 'lab_delay2h', 'lab_delay6h')
SEEDS = (48, 49, 50, 51, 52)
B = 20000


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, p)


def load_functions(path, names, ns):
    nodes = [n for n in ast.parse(Path(path).read_bytes()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)


def api(folder):
    import numpy as np
    ns = dict(np=np, itertools=itertools)
    load_functions(folder/'prior_statistics.py', ('prepared_ap', 'weighted_ap'), ns)
    load_functions(folder/'prior_significance.py', ('centered_p', 'holm'), ns)
    return ns


def weights(cluster, count=B):
    import numpy as np
    unique, inverse = np.unique(cluster, return_inverse=True)
    rng = np.random.default_rng(20261027)
    for start in range(0, count, 16):
        w = np.stack([np.bincount(rng.integers(0, len(unique), len(unique)),
                                  minlength=len(unique))[inverse]
                      for _ in range(min(16, count-start))]).astype(float)
        yield start, w


def summary(values):
    import numpy as np
    a = np.asarray(values, float)
    assert a.shape == (5,) and np.isfinite(a).all()
    return dict(values=a.tolist(), mean=float(a.mean()), sample_SD=float(a.std(ddof=1)))


class Evaluation:
    def __init__(self, args):
        import numpy as np
        self.a = args
        self.project = args.project.resolve(strict=True)
        self.source = args.source.resolve(strict=True)
        self.out = args.output.resolve()
        for p in (self.source, self.out, args.cache.resolve(), args.temp.resolve()):
            assert p.is_relative_to(self.project) and p != self.project
        assert shutil.disk_usage(self.project).free > (args.expected_gib + 10)*2**30
        self.out.mkdir(parents=True, exist_ok=True)
        import fcntl
        self.owner = (self.out/'writer.lock').open('a+')
        fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.protected = {}
        self.checked = set()
        self.db = args.database
        self.cfg = read(self.keep(self.source/'work'/MATCH/'config.json'))
        assert self.cfg['database'] == self.db and self.cfg['seeds'] == list(SEEDS)
        sys.path.insert(0, str(self.source))
        self.host = 'host_eicu' if self.db == 'eicu' else 'host_mimic'
        host = self.project/'runs'/MATCH/self.host
        state = read(self.keep(host/'state.json'))
        assert state['status'] == 'complete'
        assert read(self.keep(host/'scheduler_v4_exit.json'))['returncode'] == 0
        self.group = next(g for g in state['groups'].values() if g['database'] == self.db)
        assert Path(self.group['source']).resolve() == self.source
        assert self.group['source_manifest_sha256'] == sha(self.source/'source_manifest.json')
        self.keep(self.source/'source_manifest.json')
        manifest = read(self.source/'source_manifest.json')
        for name in ('scripts/cem_matched_common.py', 'scripts/cem_matched_windows.py',
                     'scripts/tracehr_value_features.py', 'models/tracehr_centered_native.py',
                     'models/tracehr_fte_binding.py', 'models/cem_matched_tabm.py'):
            self.keep(self.source/name, manifest[name])
        self.parent = self.cfg['parent']
        target = self.project/self.parent['test_inputs']/'test_targets.npz'
        target = self.keep(target)
        self.targets = dict(np.load(target, allow_pickle=False))
        self.raw = np.load(self.project/self.parent['test_inputs']/'test_raw.npy', mmap_mode='r')
        self.stress = self.project/'runs'/OLD/self.db/'stress_inputs_v2'
        self.stress_receipt = read(self.keep(self.stress/'completion.json'))
        assert self.stress_receipt['complete']
        assert self.protected[str(target)] == self.stress_receipt['targets']['sha256']
        self.changed = {}
        for cond in CONDITIONS:
            desc = self.stress_receipt['artifacts'][cond]
            p = self.keep(self.stress/(cond+'.npy'), desc['sha256'])
            self.changed[cond] = np.load(p, mmap_mode='r')
            assert self.changed[cond].shape == self.raw.shape
        self.window_dir = self.project/self.cfg['window_preparation']
        wm = read(self.keep(self.window_dir/'completion.json'))
        assert wm['complete'] and wm['changed_coordinates'] == 480
        self.keep(self.window_dir/'scaling.npz', wm['artifacts']['scaling.npz']['sha256'])
        self.clean_window = np.load(self.keep(self.window_dir/'test_windows.npy',
                                             wm['artifacts']['test_windows.npy']['sha256']), mmap_mode='r')
        old_group = read(self.keep(self.project/'runs'/FIVE/self.host/'state.json'))
        self.five_group = next(g for g in old_group['groups'].values() if g['database'] == self.db)
        self.bootstrap_old = self.project/'runs'/OLD/(self.db+'_statistics_v1')/'bootstrap'
        self.bootstrap_five = self.project/'runs'/FIVE/(self.db+'_statistics_v1')/'bootstrap'
        self.bootstrap_match = self.project/'runs'/MATCH/(self.db+'_statistics_v1')/'bootstrap'
        self.old_inference = self.project/'runs'/OLD/self.db/'frozen_inference_v2'
        self.old_secondary = self.project/'runs'/OLD/(self.db+'_secondary_v1')/'bootstrap'
        self.oldmeta = read(self.keep(self.bootstrap_old/'random_test_CEM_Net.json'))
        ident = self.oldmeta['identity']
        assert ident['replicates'] == B and ident['rng_seed'] == 20261027
        assert ident['cluster_ids_sha256'] == hashlib.sha256(self.targets['ids'][:, 0].tobytes()).hexdigest()
        self.saved_old_draws = dict(np.load(self.keep(self.bootstrap_old/'random_test_CEM_Net.npz',
                                                    self.oldmeta['draws_sha256']), allow_pickle=False))
        self.jobs = []
        for method in ('duration', 'window'):
            for e in (0, 1):
                for s in SEEDS:
                    if method == 'duration':
                        fit = Path(self.parent['full_ema'][f'{s}:{e}']['output']) if s <= 50 else Path(self.five_group['jobs'][f'CEM_t{e}_s{s}']['output'])
                        if s <= 50:
                            ref = ident['predictions'][f'{s}:{e}']
                            draws = self.saved_old_draws[f'{s}:{e}']
                        else:
                            mp = self.bootstrap_five/f'test_CEM-Net_s{s}_t{e}.json'
                            meta = read(self.keep(mp))
                            ref = meta['identity']['prediction']
                            draws = np.load(self.keep(mp.with_suffix('.npz'), meta['sha256']))['AP']
                    else:
                        fit = Path(self.group['jobs'][f'window_summary_t{e}_s{s}']['output'])
                        test = Path(self.group['jobs'][f'test_window_summary_t{e}_s{s}']['output'])
                        tc = read(self.keep(test/'completion.json'))
                        assert tc['optimizer_updates'] == 0 and not tc['parameters_updated']
                        ref = dict(path=str(test/'predictions_test.npz'), sha256=tc['predictions_sha256'])
                        mp = self.bootstrap_match/f'window_summary_s{s}_t{e}.json'
                        meta = read(self.keep(mp))
                        assert meta['identity']['prediction'] == ref
                        draws = np.load(self.keep(mp.with_suffix('.npz'), meta['sha256']))['AP']
                    receipt = read(self.keep(fit/'completion.json'))
                    checkpoint = self.keep(fit/receipt['checkpoint_file'], receipt['checkpoint_sha256'])
                    self.keep(ref['path'], ref['sha256'])
                    q = self.read_predictions(ref['path'], e)
                    assert draws.shape == (B,) and np.isfinite(draws).all()
                    self.jobs.append(dict(method=method, endpoint=e, seed=s, fit=str(fit),
                                          checkpoint=str(checkpoint), checkpoint_sha256=receipt['checkpoint_sha256'],
                                          clean_reference=ref, clean=q, clean_draws=draws))
        self.progress('inputs_verified', test_rows=len(self.raw), checkpoint_count=len(self.jobs))

    def keep(self, p, expected=None):
        p = Path(p).resolve(strict=True)
        assert p.is_relative_to(self.project)
        name = str(p)
        if name not in self.protected:
            self.protected[name] = sha(p)
        if expected is not None:
            assert self.protected[name] == expected, str(p)
        return p

    def progress(self, stage, **details):
        save(self.out/'progress.json', dict(stage=stage, database=self.db, time=time.time(), **details))

    def read_predictions(self, p, e):
        import numpy as np
        z = dict(np.load(p, allow_pickle=False))
        np.testing.assert_array_equal(z['ids'], self.targets['ids'])
        labels, q = z['labels'], z['probs']
        labels = labels[:, e if labels.shape[1] == 2 else 0] if labels.ndim == 2 else labels
        q = q[:, e if q.shape[1] == 2 else 0] if q.ndim == 2 else q
        np.testing.assert_array_equal(labels, self.targets['y'][:, e])
        assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
        return q

    def prepare_windows(self):
        import numpy as np
        import pandas as pd
        from scripts.tracehr_value_features import duration_summary
        dest = self.out/'windows'
        if (dest/'completion.json').exists():
            r = read(dest/'completion.json')
            assert r['complete'] and r['source_sha256'] == sha(Path(__file__))
            for name, h in r['files'].items(): assert sha(dest/name) == h
            return r
        dest.mkdir(exist_ok=True)
        source = self.project/'work'/OLD/f'stress_{self.db}_v2'
        script = self.keep(source/'stress.py')
        spec = importlib.util.spec_from_file_location('frozen_stress_builder', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frozen = read(self.keep(source/'frozen_inputs.json'))
        assert frozen['lab_keys'] == self.cfg['measurements']['keys']
        assert frozen['normalization_sha256'] == self.cfg['measurements']['normalization_sha256']
        folder = self.project/'cache/cem_evidence_completion_20261025/native_observations'/self.db
        manifest = read(self.keep(folder/'source_files.json'))
        numeric = self.keep(folder/'numeric.parquet', manifest['files']['numeric.parquet'])
        ids = self.targets['ids'][:, -1]
        keys, norm = frozen['lab_keys'], frozen['normalization']
        frame = pd.read_parquet(numeric, filters=[('stay_id', 'in', ids.tolist()), ('available_h', '<=', 24.)])
        frame['key'] = frame.lab.astype(str)+'|'+frame.unit.astype(str)
        frame = frame[frame.key.isin(keys)].copy()
        frame['lab_identity'] = frame.lab.map(module.clean_lab) if self.db == 'eicu' else frame.lab.astype(str)
        subset = np.unique(np.linspace(0, len(ids)-1, 48, dtype=int))
        lookup = {int(s): i for i, s in enumerate(ids)}
        check_ids = set(ids[subset])
        results, files = [], {}
        begin = time.time()
        for cond in ('clean',)+CONDITIONS:
            self.progress('construct_window_summaries', condition=cond)
            g = module.altered(frame, cond, 'available_h', 'lab_identity')
            g = g.groupby(['key', 'stay_id', 'available_h'], as_index=False, observed=True).agg(value=('value', 'median'))
            g['z'] = np.clip((g.value-g.key.map({k: norm[k]['train_median'] for k in keys}))/g.key.map({k: norm[k]['train_iqr_or_std'] for k in keys}), -5, 5)
            g = g.sort_values(['key', 'stay_id', 'available_h'], kind='stable')
            a = np.zeros((len(ids), 480), np.float32)
            for wi, w in enumerate((1., 6., 24.)):
                tab = g[g.available_h >= 24-w].groupby(['key', 'stay_id'], sort=False, observed=True).z.agg(['count', 'mean', 'min', 'max', 'last'])
                tab['count'] = np.log1p(tab['count'])
                present = set(tab.index.get_level_values(0))
                for j, key in enumerate(keys):
                    if key in present:
                        a[:, j*15+wi*5:j*15+wi*5+5] = tab.xs(key, level=0).reindex(ids, fill_value=0).to_numpy(dtype='float32')
            ref = self.raw if cond == 'clean' else self.changed[cond]
            expected_duration = np.zeros((len(subset), 480), np.float32)
            selected = {int(ids[row]): i for i, row in enumerate(subset)}
            key_index = {k: j for j, k in enumerate(keys)}
            for (key, sid), part in g[g.stay_id.isin(check_ids)].groupby(['key', 'stay_id'], sort=False):
                j = key_index[key]
                expected_duration[selected[int(sid)], j*15:j*15+15] = duration_summary(part.available_h.to_numpy(), part.z.to_numpy())
            np.testing.assert_allclose(expected_duration, ref[subset, -480:], rtol=2e-6, atol=2e-6)
            if cond == 'clean':
                np.testing.assert_allclose(a, self.clean_window, rtol=2e-6, atol=2e-6)
                drift = float(np.max(np.abs(a-self.clean_window)))
            else:
                name = cond+'.npy'
                with (dest/name).open('xb') as f: np.save(f, a)
                files[name] = sha(dest/name)
                drift = None
            results.append(dict(condition=cond, records=len(g), clean_max_drift=drift,
                                duration_replay_rows=len(subset)))
        result = dict(complete=True, files=files, checks=results, source_sha256=sha(Path(__file__)),
                      construction_seconds=time.time()-begin, normalization_refit=False,
                      total_bytes=sum((dest/name).stat().st_size for name in files))
        save(dest/'completion.json', result)
        del frame, g, a
        gc.collect()
        return result

    def predict(self, model, raw, indices=None):
        import numpy as np
        import torch
        import concurrent.futures
        if indices is None: indices = np.arange(len(raw))
        def batch(begin):
            with torch.inference_mode():
                x = torch.from_numpy(np.array(raw[indices[begin:begin+256]], dtype='float32', copy=True))
                q = model({'raw': x}).softmax(-1)[..., 1].mean(1)
                return q.cpu().numpy()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.a.workers) as pool:
            values = list(pool.map(batch, range(0, len(indices), 256)))
        return np.concatenate(values)

    def infer(self):
        import numpy as np
        import torch
        from sklearn.metrics import average_precision_score
        from scripts.cem_matched_common import Data, construct
        from scripts.cem_matched_windows import OverlayMatrix
        from scripts.train_tracehr_unified import seed_all
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        data = {m: Data(self.project, window=m == 'window') for m in ('duration', 'window')}
        windows = {c: np.load(self.stress_windows/(c+'.npy'), mmap_mode='r') for c in CONDITIONS}
        self.predictions, self.clean_draws, self.refs = {}, {}, {}
        count = 0
        for job in self.jobs:
            method, e, s = job['method'], job['endpoint'], job['seed']
            key = f'{method}_t{e}_s{s}'
            folder = self.out/'predictions'/key
            completed = folder/'completion.json'
            self.progress('fixed_inference', model=key, completed=count, total=20)
            raw = self.raw if method == 'duration' else OverlayMatrix(self.raw, self.clean_window)
            self.predictions[method, e, s, 'clean'] = job['clean']
            self.clean_draws[method, e, s] = job['clean_draws']
            self.refs[method, e, s, 'clean'] = job['clean_reference']
            if not completed.exists() and self.a.resume_from:
                previous = self.a.resume_from/'predictions'/key/'completion.json'
                if previous.exists():
                    adopted = read(previous)
                    assert adopted['complete'] and adopted['weights_unchanged'] and adopted['optimizer_updates'] == 0
                    assert adopted['checkpoint_sha256'] == job['checkpoint_sha256']
                    assert adopted['clean_probability_drift'] <= 1e-5 and abs(adopted['clean_AP_drift']) <= 1e-5
                    for ref in adopted['predictions'].values(): assert sha(ref['path']) == ref['sha256']
                    adopted['reused_completed_evaluation'] = dict(path=str(previous), sha256=sha(previous))
                    save(completed, adopted)
            if completed.exists():
                receipt = read(completed)
                assert receipt['complete'] and receipt['checkpoint_sha256'] == job['checkpoint_sha256']
                for cond, ref in receipt['predictions'].items():
                    assert sha(ref['path']) == ref['sha256']
                    self.predictions[method, e, s, cond] = self.read_predictions(ref['path'], e)
                    self.refs[method, e, s, cond] = ref
                count += 1
                continue
            folder.mkdir(parents=True, exist_ok=True)
            seed_all(s)
            model = construct(data[method], s, e, 'reference' if method == 'duration' else 'window_summary')
            state = torch.load(job['checkpoint'], map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            del state
            model.eval()
            before = {n: v.detach().clone() for n, v in model.state_dict().items()}
            if not (self.out/'parallel_pilot.json').exists():
                sample = np.arange(min(2048, len(raw)))
                workers = self.a.workers
                self.a.workers = 1
                t = time.time(); sequential = self.predict(model, raw, sample); sequential_seconds = time.time()-t
                self.a.workers = workers
                t = time.time(); parallel = self.predict(model, raw, sample); parallel_seconds = time.time()-t
                np.testing.assert_array_equal(parallel, sequential)
                save(self.out/'parallel_pilot.json', dict(complete=True, model=key, rows=len(sample),
                     sequential_seconds=sequential_seconds, parallel_seconds=parallel_seconds,
                     batches_in_parallel=workers, threads_per_batch=4, predictions_exact=True))
            if e == 0 and s == 48:
                sample = np.arange(min(2048, len(raw)))
                pilot_start = time.time()
                pilot_q = self.predict(model, raw, sample)
                elapsed = time.time()-pilot_start
                pilot_drift = float(np.max(np.abs(pilot_q-job['clean'][sample])))
                assert pilot_drift <= 1e-5
                save(self.out/(method+'_cpu_pilot.json'), dict(complete=True, rows=len(sample),
                     seconds=elapsed, max_probability_drift=pilot_drift,
                     projected_full_condition_seconds=elapsed*len(raw)/len(sample),
                     device='CPU', threads=4, batch=256, optimizer_updates=0))
                assert elapsed < 300, 'CPU pilot exceeds bounded execution budget; inspect before expanding'
            # CPU is a performance-only adaptation: require full clean replay before stress scoring.
            begin = time.time()
            q = self.predict(model, raw)
            drift = float(np.max(np.abs(q-job['clean'])))
            assert drift <= 1e-5, (key, drift)
            ap_drift = float(average_precision_score(self.targets['y'][:, e], q)-average_precision_score(self.targets['y'][:, e], job['clean']))
            # Meaningful score/order differences must be investigated rather than accepted by probability tolerance.
            assert abs(ap_drift) <= 1e-5, (key, 'AP replay', ap_drift)
            clean_seconds = time.time()-begin
            save(folder/'replay.json', dict(complete=True, max_probability_drift=drift, AP_drift=ap_drift,
                                           seconds=clean_seconds, device='CPU', threads=4, batch=256))
            references, times, reused = {}, {}, 0
            for cond in CONDITIONS:
                if method == 'duration' and s <= 50:
                    olddir = self.old_inference/'jobs'/f'CEM-Net_t{e}_s{s}'
                    r = read(self.keep(olddir/'completion.json'))
                    assert r['job']['checkpoint_sha256'] == job['checkpoint_sha256']
                    path = self.keep(olddir/(cond+'.npz'), r['artifacts'][cond+'.npz'])
                    pred = self.read_predictions(path, e)
                    references[cond] = dict(path=str(path), sha256=self.protected[str(path)], reused=True)
                    reused += 1
                    # One full perturbation replay checks CPU/GPU equivalence on changed inputs too.
                    if cond == CONDITIONS[0]:
                        replay = self.predict(model, self.changed[cond])
                        assert np.max(np.abs(replay-pred)) <= 1e-5
                        assert abs(average_precision_score(self.targets['y'][:, e], replay)-average_precision_score(self.targets['y'][:, e], pred)) <= 1e-5
                else:
                    altered = self.changed[cond] if method == 'duration' else OverlayMatrix(self.changed[cond], windows[cond])
                    start = time.time()
                    pred = self.predict(model, altered)
                    times[cond] = time.time()-start
                    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
                    path = folder/(cond+'.npz')
                    assert not path.exists()
                    np.savez_compressed(path, ids=self.targets['ids'], labels=self.targets['y'][:, e], probs=pred)
                    references[cond] = dict(path=str(path), sha256=sha(path), reused=False)
                self.predictions[method, e, s, cond] = pred
                self.refs[method, e, s, cond] = references[cond]
                self.progress('fixed_inference', model=key, condition=cond, completed=count, total=20)
            assert all(torch.equal(before[n], v) for n, v in model.state_dict().items())
            assert sha(job['checkpoint']) == job['checkpoint_sha256']
            receipt = dict(complete=True, method=method, endpoint=e, seed=s, checkpoint_sha256=job['checkpoint_sha256'],
                           source_sha256=sha(Path(__file__)), predictions=references, reused_cases=reused,
                           new_cases=4-reused, inference_seconds=times, clean_seconds=clean_seconds,
                           clean_probability_drift=drift, clean_AP_drift=ap_drift,
                           optimizer_updates=0, weights_unchanged=True, normalization_refit=False,
                           device='CPU', precision='FP32', threads=4, batch=256, batches_in_parallel=self.a.workers)
            save(completed, receipt)
            del model, before
            gc.collect()
            count += 1
        del data
        gc.collect()

    def analyze(self):
        import numpy as np
        from sklearn.metrics import average_precision_score
        ns = api(Path(__file__).parent)
        cluster = self.targets['ids'][:, 0]
        y = self.targets['y']
        first = next(weights(cluster, 16))[1]
        draws, refs = {}, {}
        for job in self.jobs:
            m, e, s = job['method'], job['endpoint'], job['seed']
            q = self.predictions[m, e, s, 'clean']
            np.testing.assert_allclose(ns['weighted_ap'](ns['prepared_ap'](y[:, e], q), first),
                                       self.clean_draws[m, e, s][:16], rtol=0, atol=1e-12)
            draws[m, e, s, 'clean'] = self.clean_draws[m, e, s]
        for e in (0, 1):
            for cond in CONDITIONS:
                p = self.old_secondary/f'stress_t{e}_{cond}.npz'
                meta = read(self.keep(p.with_suffix('.json')))
                assert meta['identity']['replicates'] == B and meta['identity']['seed'] == 20261027
                self.keep(p, meta['sha256'])
                saved = dict(np.load(p, allow_pickle=False))
                for s in (48, 49, 50):
                    k = ('duration', e, s, cond)
                    values = saved[f'CEM-Net:{s}']
                    np.testing.assert_allclose(ns['weighted_ap'](ns['prepared_ap'](y[:, e], self.predictions[k]), first), values[:16], rtol=0, atol=1e-12)
                    draws[k] = values
        cache = self.out/'bootstrap'
        cache.mkdir(exist_ok=True)
        pending, metadata = {}, {}
        for key, q in self.predictions.items():
            if key in draws: continue
            name = '_'.join(map(str, key))
            mp, dp = cache/(name+'.json'), cache/(name+'.npz')
            ident = dict(database=self.db, key=list(key), replicates=B, rng_seed=20261027,
                         cluster_sha256=hashlib.sha256(cluster.tobytes()).hexdigest(), prediction=self.refs[key])
            if mp.exists():
                saved = read(mp)
                assert saved['identity'] == ident and sha(dp) == saved['sha256']
                draws[key] = np.load(dp, allow_pickle=False)['AP']
            else:
                assert not dp.exists()
                pending[key] = ns['prepared_ap'](y[:, key[1]], q)
                metadata[key] = (mp, dp, ident)
                draws[key] = np.empty(B)
        start_time = time.time()
        # Generate common patient weights once per batch for all predictions.
        for start, w in weights(cluster) if pending else []:
            for key, prepared in pending.items():
                draws[key][start:start+len(w)] = ns['weighted_ap'](prepared, w)
            if start % 320 == 0:
                self.progress('patient_bootstrap', completed=start+len(w), total=B,
                              new_distributions=len(pending), elapsed_seconds=time.time()-start_time)
        for key, (mp, dp, ident) in metadata.items():
            assert np.isfinite(draws[key]).all()
            with dp.open('xb') as f: np.savez_compressed(f, AP=draws[key])
            save(mp, dict(identity=ident, sha256=sha(dp)))
        rows, paired, paired_draws = [], [], {}
        for e in (0, 1):
            for cond in CONDITIONS:
                cell = dict(database=self.db, endpoint=e, condition=cond, methods={})
                for m in ('duration', 'window'):
                    clean = [average_precision_score(y[:, e], self.predictions[m, e, s, 'clean']) for s in SEEDS]
                    changed = [average_precision_score(y[:, e], self.predictions[m, e, s, cond]) for s in SEEDS]
                    cell['methods'][m] = dict(clean=summary(clean), perturbed=summary(changed),
                                              change=summary(np.asarray(changed)-clean))
                absolute = np.asarray(cell['methods']['duration']['perturbed']['values'])-cell['methods']['window']['perturbed']['values']
                clean_gap = np.asarray(cell['methods']['duration']['clean']['values'])-cell['methods']['window']['clean']['values']
                g = np.mean([draws['duration', e, s, cond]-draws['window', e, s, cond] for s in SEEDS], axis=0)
                cg = np.mean([draws['duration', e, s, 'clean']-draws['window', e, s, 'clean'] for s in SEEDS], axis=0)
                for estimand, observed, samples in (('degradation', absolute-clean_gap, g-cg), ('absolute', absolute, g)):
                    row = dict(database=self.db, endpoint=e, condition=cond, estimand=estimand,
                               **summary(observed), family_count=24,
                               ci95=np.quantile(samples, [.025, .975]).tolist(),
                               adjusted_ci=np.quantile(samples, [.05/48, 1-.05/48]).tolist(),
                               p_bootstrap_centered=ns['centered_p'](samples, float(np.mean(observed))))
                    paired.append(row)
                    paired_draws[f'{estimand}:{e}:{cond}'] = samples
                rows.append(cell)
        with (self.out/'paired_draws.npz').open('xb') as f: np.savez_compressed(f, **paired_draws)
        for p, h in self.protected.items(): assert sha(p) == h, p
        replay = [read(self.out/'predictions'/f"{j['method']}_t{j['endpoint']}_s{j['seed']}"/'completion.json') for j in self.jobs]
        result = dict(complete=True, database=self.db, seeds=list(SEEDS), conditions=list(CONDITIONS),
                      test_rows=len(self.raw), patient_count=len(np.unique(cluster)),
                      models=20, new_inferences=sum(x['new_cases'] for x in replay),
                      reused_inferences=sum(x['reused_cases'] for x in replay), training_runs=0,
                      bootstrap_replicates=B, bootstrap_seconds=time.time()-start_time,
                      primary_family='24 paired AP degradation differences across six tasks and four conditions',
                      secondary_family='24 paired perturbed AP differences across six tasks and four conditions',
                      confidence_scope='patient sampling conditional on saved fits, cohort and fixed perturbation; five-seed SD reported separately',
                      cpu_replay_max_probability_drift=max(x['clean_probability_drift'] for x in replay),
                      cpu_replay_max_AP_drift=max(abs(x['clean_AP_drift']) for x in replay),
                      rows=rows, paired=paired, original_files_unchanged=True, protected=self.protected,
                      source_sha256=sha(Path(__file__)), run_protocol=read(Path(__file__).parent/'protocol.json'))
        save(self.out/'verified-results.json', result)
        save(self.out/'completion.json', dict(complete=True, files={n: sha(self.out/n) for n in ('verified-results.json', 'paired_draws.npz')}))
        self.progress('complete', new_inferences=result['new_inferences'], reused_inferences=result['reused_inferences'])
        return {k: result[k] for k in ('complete', 'database', 'new_inferences', 'reused_inferences', 'training_runs')}


def main():
    p = argparse.ArgumentParser()
    for name in ('project', 'source', 'output', 'cache', 'temp'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--database', choices=('mimiciii', 'mimic', 'eicu'), required=True)
    p.add_argument('--expected-gib', type=float, default=2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume-from', type=Path)
    a = p.parse_args()
    obj = Evaluation(a)
    if (obj.out/'completion.json').exists():
        print(json.dumps(dict(complete=True, already_complete=True)))
        return
    obj.stress_windows = obj.out/'windows'
    if a.resume_from:
        a.resume_from = a.resume_from.resolve(strict=True)
        assert a.resume_from.is_relative_to(obj.project) and a.resume_from != obj.out
        assert read(a.resume_from/'exit.json')['returncode'] != 0
        old_source = Path(read(a.resume_from/'storage.json')['source'])/'worker.py'
        old_ast = ast.parse(old_source.read_bytes()); new_ast = ast.parse(Path(__file__).read_bytes())
        def extraction(tree):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Evaluation')
            return ast.dump(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_windows'), include_attributes=False)
        assert extraction(old_ast) == extraction(new_ast), 'Reuse requires identical feature-building implementation'
        preparation = read(a.resume_from/'windows/completion.json')
        assert preparation['complete'] and preparation['source_sha256'] == sha(old_source)
        for n, h in preparation['files'].items(): assert sha(a.resume_from/'windows'/n) == h
        obj.stress_windows = a.resume_from/'windows'
        save(obj.out/'reused_preparation.json', dict(path=str(obj.stress_windows),
             completion_sha256=sha(obj.stress_windows/'completion.json'), feature_function_unchanged=True,
             reason='execution-only parallel batch processing; previous completed predictions retained'))
    else:
        obj.prepare_windows()
    obj.infer()
    print(json.dumps(obj.analyze()))


if __name__ == '__main__':
    main()
