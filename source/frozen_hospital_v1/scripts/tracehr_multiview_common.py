"""Shared metadata and immutable input identities for the fixed-input MV campaign."""
import hashlib
import json
import os
from pathlib import Path

RUN = 'tracehr_multiview_20261009'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    os.replace(tmp, path)


def checked_output(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    assert project.is_dir()
    assert output.is_relative_to(project) and output != project
    assert output.drive.upper() != 'C:'
    return output


def inputs(project, database):
    p = Path(project)
    return dict(phi=p/'data/tracehr_values_20261003/full'/f'{database}24',
                values=p/'data/tracehr_values_20261003/full'/f'{database}24_values',
                numeric=p/'data/tracehr_conditional_20261005'/database,
                events=p/'data'/RUN/database)


def experts(project, database, seed=42):
    """Exact known completed checkpoints, never a best-of-directory wildcard."""
    p = Path(project)
    a = p/'runs/tracehr_values_20261003'/database/'round1/runs'/f'{database}24_stat_cfg0_full_s42'
    if seed == 42:
        root = p/'runs/tracehr_tabular_20261004'/database/('round1_r2' if database == 'mimic' else 'round1_r3')/'runs'
        v = root/f'{database}24_tabm_cfg1_full_s42'
        r = root/f'{database}24_realmlp_cfg0_full_s42'
    else:
        root = p/'runs/tracehr_expansion_confirm_20261004'/database/'round1/runs'
        v = root/f'{database}24_tabm_F2_cfg1_full_s{seed}'
        r = root/f'{database}24_realmlp_F2_cfg0_full_s{seed}'
    return dict(A=a, V=v, R=r)


def scores(y, p):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    assert p.shape == y.shape and np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1))
    ap = [float(average_precision_score(y[:, j], p[:, j])) for j in range(y.shape[1])]
    return dict(mean_ap=float(np.mean(ap)), ap=ap,
                auroc=[float(roc_auc_score(y[:, j], p[:, j])) for j in range(y.shape[1])],
                brier=[float(brier_score_loss(y[:, j], p[:, j])) for j in range(y.shape[1])])


def combine(y, pa, pv, pj):
    """A predeclared DEVELOPMENT selection, not OOF stacking or independent testing."""
    import numpy as np
    from sklearn.metrics import average_precision_score
    alphas = [0., .1, .25, .5, 1.]
    betas = [0., .25, .5, .75, 1.]
    final = np.empty_like(pv, dtype=np.float64)
    chosen, rows = [], []
    for task in range(y.shape[1]):
        best = -1.
        for alpha in alphas:
            base = alpha * pa[:, task] + (1-alpha) * pv[:, task]
            for beta in betas:
                pred = (1-beta) * base + beta * pj[:, task]
                ap = float(average_precision_score(y[:, task], pred))
                row = dict(task=task, alpha=alpha, beta=beta, ap=ap)
                rows.append(row)
                if ap > best + 1e-12:
                    best, selection = ap, row.copy()
                    final[:, task] = pred
        chosen.append(selection)
    return final, chosen, rows
