"""Fixed-input calibrated fusion campaign: identities, folds and metrics."""
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold, train_test_split
from scripts.tracehr_multiview_common import read, atomic, sha, inputs, experts, checked_output, scores

RUN = 'tracehr_calibrated_oof_20261010'
FAMILIES = ('A', 'V', 'R', 'T')


def patient_split(ids, y, seed, n_splits=3):
    patients, inverse = np.unique(ids[:, 0], return_inverse=True)
    labels = np.zeros((len(patients), y.shape[1]), np.int8)
    np.maximum.at(labels, inverse, y.astype(np.int8))
    strata = (labels * (2 ** np.arange(y.shape[1]))).sum(1)
    result, assignment = {}, np.full(len(ids), -1, np.int8)
    for fold, (inside, outside) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(patients, strata)):
        fit, val = train_test_split(inside, test_size=.1, stratify=strata[inside], random_state=seed + 100 + fold)
        for key, part in [('fit', fit), ('val', val), ('holdout', outside)]:
            result[f'{key}_{fold}'] = np.flatnonzero(np.isin(inverse, part))
        assignment[result[f'holdout_{fold}']] = fold
        parts = [result[f'{k}_{fold}'] for k in ('fit', 'val', 'holdout')]
        assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(ids)))
        for j, part in enumerate(parts):
            assert all(len(np.unique(y[part, k])) == 2 for k in range(y.shape[1]))
            assert all(not np.intersect1d(ids[part, 0], ids[other, 0]).size for other in parts[j + 1:])
    assert np.all(assignment >= 0)
    result['assignment'] = assignment
    return result


def safe_logit(p):
    return logit(np.clip(np.asarray(p, np.float64), 1e-12, 1 - 1e-12))


def undo_weight(p, weight):
    return expit(safe_logit(p) - np.log(np.asarray(weight)))


def all_scores(y, p):
    from sklearn.metrics import log_loss
    result = scores(y, p)
    result['log_loss'] = [float(log_loss(y[:, j], p[:, j], labels=[0, 1])) for j in range(y.shape[1])]
    result['prevalence'] = y.mean(0).tolist()
    result['mean_auroc'] = float(np.mean(result['auroc']))
    result['mean_brier'] = float(np.mean(result['brier']))
    result['top_fraction'] = {}
    for fraction in (.05, .1):
        values = []
        for j in range(y.shape[1]):
            n = max(1, int(np.ceil(len(y) * fraction)))
            # Fractional inclusion of the boundary tie avoids using row order.
            cutoff = np.sort(p[:, j])[-n]
            higher, tied = p[:, j] > cutoff, p[:, j] == cutoff
            count = float(y[higher, j].sum() + (n-higher.sum()) / tied.sum() * y[tied, j].sum())
            values.append(dict(precision=count/n, recall=count/float(y[:, j].sum()), selected=n))
        result['top_fraction'][str(fraction)] = values
    return result


def original_folder(project, database, family):
    if family == 'T':
        return Path(project)/'runs/tracehr_multiview_20261009'/database/'screen/jobs/temporal_s42'
    return experts(project, database)[family]


def prep_folder(project, database, seed=42):
    registry=Path(project)/'work'/RUN/f'prepared_{database}_s{seed}.json'
    if registry.exists():
        record=read(registry);folder=Path(record['path'])
        assert folder.resolve().is_relative_to((Path(project)/'runs'/RUN/database).resolve())
        assert sha(folder/'completion.json')==record['manifest_sha256']
        return folder
    return Path(project)/'runs'/RUN/database/f'preparation_s{seed}/jobs/prepare'


def grouped_summaries(rows,times,values,samples):
    """Same registered numeric formulas, with no unused pandas dependency."""
    from scripts.tracehr_value_features import TAUS,basis
    starts=np.r_[0,1+np.flatnonzero(np.diff(rows))];ends=np.r_[starts[1:],len(rows)]
    n=ends-starts;last=ends-1
    sums=lambda v:np.add.reduceat(v,starts)
    tv=sums(times);vv=sums(values);den=sums(times*times)-tv*tv/n
    slope=np.divide(sums(times*values)-tv*vv/n,den,out=np.zeros(len(n)),where=den>1e-8)*24.
    simple=np.stack([np.ones(len(n)),np.log1p(n),times[starts]/24,(24-times[last])/24,
        (24-samples[last])/24,vv/n,np.minimum.reduceat(values,starts),np.maximum.reduceat(values,starts),
        values[last],values[last]-values[starts],slope],axis=1).astype(np.float32)
    next_time=np.r_[times[1:],24.];next_time[last]=24.
    mass=(np.exp(-(24-next_time[:,None])/TAUS)-np.exp(-(24-times[:,None])/TAUS))/(1-np.exp(-24/TAUS))
    bv=basis(values);duration=np.empty((len(n),15),np.float32)
    for t in range(3):
        for b in range(5):duration[:,t*5+b]=sums(mass[:,t]*bv[:,b])
    return rows[starts],simple,duration
