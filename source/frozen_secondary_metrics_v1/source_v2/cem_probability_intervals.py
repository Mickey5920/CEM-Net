"""Fixed-prediction loss intervals; no model, calibrator or AP fitting.

The local launcher sends only this source and downloads aggregate JSON. Patient
records stay on their existing host. Each split uses the original PCG64 patient
draw sequence; losses, not probabilities, are averaged across the three fits.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time

RUN = 'cem_icde_evidence_20261028'
SEEDS = (48, 49, 50)
B = 20000
FAMILY = 24  # 3 databases x 2 endpoints x 2 losses x raw/fixed Platt
VERSION = 'probability_intervals_v2'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p)
    q = p.with_suffix(p.suffix + '.writing')
    q.write_text(json.dumps(value, indent=2), encoding='utf8')
    os.replace(q, p)


def loss_vectors(y, q):
    import numpy as np
    from scipy.special import xlogy
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.metrics import _classification as classification
    q = np.asarray(q)
    assert q.ndim == 1 and np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
    # Preserve the original prediction dtype and sklearn's clipping convention.
    clipped = np.clip(q, np.finfo(q.dtype).eps, 1 - np.finfo(q.dtype).eps)
    two = np.column_stack((1 - clipped, clipped))
    two = two / two.sum(axis=1, keepdims=True)
    if hasattr(classification, '_validate_binary_probabilistic_prediction'):
        # sklearn >=1.7 uses the two-class Brier implementation, and expands
        # binary probabilities before log-loss clipping. Match the installed
        # reference exactly instead of relaxing replay tolerances.
        labels, proba = classification._validate_binary_probabilistic_prediction(y, q, None, None)
        brier = np.sum((labels-proba)**2, axis=1)*.5
        labels, proba = classification._validate_multiclass_probabilistic_prediction(y, q, None, [0,1])
        eps = np.finfo(proba.dtype).eps
        log = -xlogy(labels, np.clip(proba, eps, 1-eps)).sum(axis=1)
    else:
        brier = (q-y)**2
        log = -xlogy(np.column_stack((1-y,y)), two).sum(axis=1)
    losses = {'Brier': brier, 'log_loss': log}
    np.testing.assert_allclose(losses['Brier'].mean(), brier_score_loss(y, q), rtol=0, atol=1e-12)
    np.testing.assert_allclose(losses['log_loss'].mean(), log_loss(y, q, labels=[0, 1]), rtol=0, atol=1e-12)
    return losses


def cluster_totals(values, inverse, n):
    import numpy as np
    return np.column_stack([np.bincount(inverse, weights=v, minlength=n) for v in values])


def self_test():
    import numpy as np
    from sklearn.metrics import log_loss, brier_score_loss
    y = np.array([1, 0, 1, 0, 0, 1])
    inv = np.array([0, 0, 1, 2, 2, 2])
    for dtype in (np.float32, np.float64):
        a = np.array([0, 1, .3, .8, .1, .7], dtype=dtype)
        b = np.array([.2, .9, .4, .7, .2, .8], dtype=dtype)
        aa, bb = loss_vectors(y, a), loss_vectors(y, b)
        for metric in aa:
            diff = aa[metric]-bb[metric]
            totals = cluster_totals([diff, -diff, np.zeros(len(y))], inv, 3)
            multiplicity = np.array([2, 0, 1])
            weights = multiplicity[inv]
            actual = multiplicity @ totals / weights.sum()
            ref = (brier_score_loss(y, a, sample_weight=weights)-brier_score_loss(y, b, sample_weight=weights)
                   if metric == 'Brier' else log_loss(y, a, sample_weight=weights, labels=[0,1])-log_loss(y, b, sample_weight=weights, labels=[0,1]))
            np.testing.assert_allclose(actual, [ref, -ref, 0], rtol=0, atol=1e-12)
    # Shared draw stream is independent of batch grouping; keep all stays.
    r1, r2 = np.random.default_rng(20261027), np.random.default_rng(20261027)
    x = [r1.integers(0, 7, 7) for _ in range(32)]
    z = [r2.integers(0, 7, 7) for _ in range(16)] + [r2.integers(0, 7, 7) for _ in range(16)]
    np.testing.assert_array_equal(x, z)
    return ['float32_and_float64_loss_parity_including_extreme_probabilities',
            'unequal_stay_count_patient_multiplicity_matches_sklearn',
            'sign_swap_and_identical_predictions', 'unchanged_draw_stream']


def worker(project, db):
    import fcntl
    import numpy as np
    from scipy.special import expit
    import sklearn
    project = Path(project).resolve(strict=True)
    out = project/'runs'/RUN/VERSION/db
    out.mkdir(parents=True, exist_ok=True)
    owner = (out/'writer.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'completion.json').exists():
        done = read(out/'completion.json')
        assert done['complete'] and done['source_sha256'] == sha(__file__)
        for name, digest in done['files'].items():
            assert sha(out/name) == digest
        return {'reused': True, 'complete': True}
    assert not (out/'attempt.json').exists(), 'Existing attempt: inspect; do not repeat'
    save(out/'attempt.json', {'started': time.time(), 'source_sha256': sha(__file__)})
    try:
        tests = self_test()
        base = project/'runs'/RUN
        fixed = base/(db+'_statistics_v1')
        secondary = base/(db+'_secondary_v1')
        protected = {}
        predecessor = base/'probability_intervals_v1'/db
        if predecessor.exists():
            assert read(predecessor/'exit.json')['returncode'] == 1
            assert not (predecessor/'completion.json').exists()
            for f in predecessor.iterdir():
                if f.is_file() and f.name != 'writer.lock':
                    protected[str(f)] = sha(f)
            oldsource = project/'work'/RUN/'probability_intervals_v1/loss_intervals.py'
            protected[str(oldsource)] = sha(oldsource)
        for folder in (fixed, secondary):
            completion = read(folder/'completion.json')
            active = read(folder/'active-execution.json') if (folder/'active-execution.json').exists() else {}
            receipt = (folder/active.get('receipt','exit.json')).resolve()
            assert receipt.is_relative_to(folder.resolve()) and read(receipt)['returncode'] == 0
            assert completion['complete'] and sha(folder/'verified-results.json') == completion['files']['verified-results.json']
            protected[str(folder/'verified-results.json')] = sha(folder/'verified-results.json')
        primary, sec = read(fixed/'verified-results.json'), read(secondary/'verified-results.json')
        assert sec['primary_sha256'] == sha(fixed/'verified-results.json')
        cfgfile = project/'work'/RUN/db/'basic_v2/work'/RUN/'config.json'
        cfg = read(cfgfile)
        protected[str(cfgfile)] = sha(cfgfile)
        rows, drawfiles, split_info = [], {}, {}
        for split in ('dev', 'test'):
            targetfile = project/cfg['raw_inputs' if split == 'dev' else 'test_inputs']/(split+'_targets.npz')
            protected[str(targetfile)] = sha(targetfile)
            target = dict(np.load(targetfile))
            cluster = target['ids'][:, 0]
            unique, inverse = np.unique(cluster, return_inverse=True)
            n = len(unique)
            counts = np.bincount(inverse, minlength=n)
            keys, values, observed, originals = [], [], [], []
            for e in (0, 1):
                baseline = primary['strongest']['dev'][str(e)]['validation_selected']
                methods = ('CEM-Net', baseline)
                y = target['y'][:, e]
                clinical = next(c for c in sec['clinical'] if (c['split'],c['endpoint'],c['subset']) == (split,e,'all'))
                all_losses = {}
                for method in methods:
                    metafile = fixed/'bootstrap'/('random_'+split+'_'+method.replace('-','_')+'.json')
                    meta = read(metafile)
                    identity = meta['identity']
                    assert identity['rng_seed'] == 20261026+(split == 'test') and identity['replicates'] == B
                    assert identity['cluster_ids_sha256'] == hashlib.sha256(cluster.tobytes()).hexdigest()
                    protected[str(metafile)] = sha(metafile)
                    cachefile = metafile.with_suffix('.npz')
                    assert sha(cachefile) == meta['draws_sha256']
                    protected[str(cachefile)] = meta['draws_sha256']
                    for seed in SEEDS:
                        ref = identity['predictions'][f'{seed}:{e}']
                        assert sha(ref['path']) == ref['sha256']
                        protected[ref['path']] = ref['sha256']
                        z = dict(np.load(ref['path']))
                        np.testing.assert_array_equal(z['ids'], target['ids'])
                        q = z['probs']
                        if q.ndim == 2:
                            q = q[:, e if q.shape[1] == 2 else 0]
                        coef = primary['calibrators'][f'{method}:{seed}:{e}']
                        clipped = np.clip(q, 1e-7, 1-1e-7)
                        cal = expit(coef['coef']*np.log(clipped/(1-clipped)) + coef['intercept'])
                        expected = next(s for s in clinical['methods'][method]['per_seed'] if s['seed'] == seed)
                        for form, p in (('raw', q), ('Platt', cal)):
                            losses = loss_vectors(y, p)
                            for metric, loss in losses.items():
                                np.testing.assert_allclose(loss.mean(), expected[form][metric], rtol=0, atol=1e-12)
                                all_losses[method, seed, form, metric] = loss
                for form in ('raw', 'Platt'):
                    for metric in ('Brier', 'log_loss'):
                        per_seed = np.stack([all_losses['CEM-Net',s,form,metric]-all_losses[baseline,s,form,metric] for s in SEEDS])
                        keys.append((e, baseline, form, metric))
                        values.append(per_seed.mean(axis=0))
                        observed.append(per_seed.mean(axis=1))
                        originals.append({m: [float(all_losses[m,s,form,metric].mean()) for s in SEEDS] for m in methods})
            totals = cluster_totals(values, inverse, n)
            rng = np.random.default_rng(20261026+(split == 'test'))
            draws = np.empty((B, len(keys)))
            for start in range(0, B, 64):
                w = np.stack([np.bincount(rng.integers(0,n,n),minlength=n) for _ in range(min(64,B-start))])
                denom = w @ counts
                draws[start:start+len(w)] = (w @ totals)/denom[:,None]
                if start == 0:
                    direct = np.stack(values) @ w[0,inverse] / denom[0]
                    np.testing.assert_allclose(draws[0], direct, rtol=0, atol=1e-12)
                if start % 2048 == 0:
                    save(out/'progress.json', {'split':split,'completed':start+len(w),'total':B})
            path = out/(split+'_loss_differences.npz')
            np.savez_compressed(path, differences=draws)
            drawfiles[path.name] = sha(path)
            for i, (e, baseline, form, metric) in enumerate(keys):
                a = observed[i]
                rows.append(dict(split=split,endpoint=e,baseline=baseline,form=form,metric=metric,
                                 seed_differences=a.tolist(),mean=float(a.mean()),sample_SD=float(a.std(ddof=1)),
                                 method_values=originals[i],ci95=np.quantile(draws[:,i],[.025,.975]).tolist(),
                                 adjusted_ci=np.quantile(draws[:,i],[.05/(2*FAMILY),1-.05/(2*FAMILY)]).tolist(),
                                 family=FAMILY,negative_favors_CEM=True,validation_resubstitution=split=='dev'))
            split_info[split] = dict(patients=n,stays=len(cluster),rng_seed=20261026+(split=='test'),
                                    cluster_ids_sha256=hashlib.sha256(cluster.tobytes()).hexdigest())
        assert all(sha(p) == h for p,h in protected.items())
        result = dict(complete=True,database=db,rows=rows,replicates=B,source_sha256=sha(__file__),
                      primary_sha256=sha(fixed/'verified-results.json'),secondary_sha256=sha(secondary/'verified-results.json'),
                      split_info=split_info,protected=protected,bootstrap_files=drawfiles,
                      family_definition='24 CEM-minus-validation-selected-baseline contrasts: 3 databases x 2 endpoints x 2 losses x raw/fixed Platt; Validation and Test separate',
                      interpretation='Percentile patient-cluster intervals conditional on fixed models and fixed overall Validation calibrators; Validation is fitting-set description. Lower differences favor CEM.',
                      interval_level=1-.05/FAMILY,calibration_refits=0,training_runs=0,AP_recomputations=0,
                      seed_probability_pooling=False,original_AP_caches_unchanged=True,checks=tests,
                      numpy_version=np.__version__,sklearn_version=sklearn.__version__)
        save(out/'verified-results.json', result)
        save(out/'completion.json', dict(complete=True,source_sha256=sha(__file__),files={'verified-results.json':sha(out/'verified-results.json'),**drawfiles}))
        save(out/'exit.json', {'returncode':0,'finished':time.time()})
        return dict(complete=True,rows=len(rows),replicates=B)
    except Exception:
        save(out/'exit.json', {'returncode':1,'finished':time.time()})
        raise


def launch(db, mode):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from scripts.launch_ema_cross import HOSTS, ssh
    out = root/'runs'/RUN/VERSION/db
    assert Path(os.environ['PROJECT_RUN_DIR']).resolve() == out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    host = 'eicu' if db == 'eicu' else 'mimic'
    d = dict(project=HOSTS[host][1],db=db,mode=mode,run=RUN,version=VERSION,
             source=base64.b64encode(Path(__file__).read_bytes()).decode())
    code = '''import base64,json,os,shutil,subprocess,sys
from pathlib import Path
d=PAYLOAD;p=Path(d['project']).resolve(strict=True)
out=p/'runs'/d['run']/d['version']/d['db']
src=p/'work'/d['run']/d['version'];temp=src/'tmp';cache=p/'cache'/d['run']
for x in (out,src,temp,cache):
 x.mkdir(parents=True,exist_ok=True);assert x.resolve().is_relative_to(p)
assert shutil.disk_usage(p).free>10.5*2**30
source=src/'loss_intervals.py';content=base64.b64decode(d['source'])
assert not source.exists() or source.read_bytes()==content
if d['mode']=='run':
 source.write_bytes(content)
 deps=p/'work/tracehr_tabular_20261004/deps'
 env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[deps,*deps.glob('*.whl'),p/'work/cem_evidence_completion_20261025/test_runtime_v1/deps'])),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),XDG_CACHE_HOME=str(cache))
 command=[sys.executable,'-B',str(source),'--worker','--project',str(p),'--database',d['db']]
 check=subprocess.run([sys.executable,'-B',str(source),'--self-test'],env=env,text=True,capture_output=True)
 assert check.returncode==0,check.stderr
 with (out/'analysis.log').open('xb') as log:
  r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(out/'analysis.log').read_text()[-6000:]
assert json.loads((out/'exit.json').read_text())['returncode']==0
print(json.dumps({n:base64.b64encode((out/n).read_bytes()).decode() for n in ('verified-results.json','completion.json','exit.json')}))
'''.replace('PAYLOAD', repr(d))
    result = ssh(host, code, timeout=1800)
    for name, content in result.items():
        (out/name).write_bytes(base64.b64decode(content))
    assert sha(out/'verified-results.json') == read(out/'completion.json')['files']['verified-results.json']
    print(json.dumps(dict(database=db,complete=True,rows=len(read(out/'verified-results.json')['rows']),summary=str(out))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--database', choices=['mimiciii','mimic','eicu'])
    p.add_argument('--project', type=Path)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--mode', choices=['run','results'], default='run')
    a = p.parse_args()
    if a.self_test:
        print(json.dumps(self_test()))
    elif a.worker:
        print(json.dumps(worker(a.project,a.database)))
    else:
        launch(a.database,a.mode)
