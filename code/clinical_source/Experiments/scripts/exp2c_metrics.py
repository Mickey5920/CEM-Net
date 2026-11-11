import math

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score, roc_auc_score


def safe_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def binary_nll(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    if len(y_true) == 0:
        return None
    return float(-(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)).mean())


def ranking_metrics(y_true, y_prob, top_ks=(10, 50, 100)):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    output = {}
    positives = int(y_true.sum())
    unique = np.unique(y_true)
    output['pairwise_accuracy'] = safe_float(roc_auc_score(y_true, y_prob)) if len(unique) >= 2 else None
    if len(y_true) == 0 or positives == 0:
        for k in top_ks:
            output[f'precision_at_{k}'] = None
            output[f'recall_at_{k}'] = None
            output[f'ndcg_at_{k}'] = None
        return output

    order = np.argsort(-y_prob)
    ranked_labels = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, len(ranked_labels) + 2))
    for k in top_ks:
        limit = min(int(k), len(ranked_labels))
        if limit <= 0:
            output[f'precision_at_{k}'] = None
            output[f'recall_at_{k}'] = None
            output[f'ndcg_at_{k}'] = None
            continue
        top_labels = ranked_labels[:limit]
        hits = int(top_labels.sum())
        dcg = float(np.sum(top_labels * discounts[:limit]))
        ideal_hits = min(positives, limit)
        ideal_dcg = float(np.sum(discounts[:ideal_hits])) if ideal_hits > 0 else 0.0
        output[f'precision_at_{k}'] = safe_float(hits / limit)
        output[f'recall_at_{k}'] = safe_float(hits / positives)
        output[f'ndcg_at_{k}'] = safe_float(dcg / ideal_dcg) if ideal_dcg > 0 else None
    return output


def calibration_bins(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for index in range(n_bins):
        lower = float(bins[index])
        upper = float(bins[index + 1])
        if index == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        count = int(mask.sum())
        if count == 0:
            rows.append({'bin': index, 'lower': lower, 'upper': upper, 'n': 0, 'confidence': None, 'accuracy': None, 'gap': None})
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        gap = abs(confidence - accuracy)
        ece += float(mask.mean()) * gap
        rows.append({'bin': index, 'lower': lower, 'upper': upper, 'n': count, 'confidence': confidence, 'accuracy': accuracy, 'gap': gap})
    return rows, float(ece)


def threshold_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= float(threshold)).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        'accuracy': safe_float(accuracy_score(y_true, y_pred)),
        'f1': safe_float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity': None if tp + fn == 0 else safe_float(tp / (tp + fn)),
        'specificity': None if tn + fp == 0 else safe_float(tn / (tn + fp)),
        'ppv': None if tp + fp == 0 else safe_float(tp / (tp + fp)),
        'npv': None if tn + fn == 0 else safe_float(tn / (tn + fn)),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def binary_metrics(y_true, y_prob, threshold=0.5, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    unique = np.unique(y_true)
    auroc = safe_float(roc_auc_score(y_true, y_prob)) if len(unique) >= 2 else None
    auprc = safe_float(average_precision_score(y_true, y_prob)) if len(unique) >= 2 else None
    bins, ece = calibration_bins(y_true, y_prob, n_bins=n_bins)
    output = {
        'auroc': auroc,
        'auprc': auprc,
        'brier': safe_float(brier_score_loss(y_true, y_prob)),
        'nll': safe_float(binary_nll(y_true, y_prob)),
        'ece': safe_float(ece),
        'single_class': len(unique) < 2,
        'n': int(len(y_true)),
        'positives': int(y_true.sum()),
        'positive_rate': safe_float(y_true.mean()) if len(y_true) else None,
        'threshold': float(threshold),
        'calibration_bins': bins,
    }
    output.update(ranking_metrics(y_true, y_prob))
    output.update(threshold_metrics(y_true, y_prob, threshold))
    return output


def select_thresholds_on_dev(y_true, y_prob, tasks, metric='f1'):
    thresholds = {}
    grid = np.linspace(0.01, 0.99, 99)
    for index, task in enumerate(tasks):
        labels = np.asarray(y_true[:, index], dtype=int)
        probs = np.asarray(y_prob[:, index], dtype=float)
        best_threshold = 0.5
        best_value = -1.0
        for threshold in grid:
            value = threshold_metrics(labels, probs, threshold).get(metric)
            if value is not None and value > best_value:
                best_value = float(value)
                best_threshold = float(threshold)
        thresholds[task] = {'threshold': best_threshold, 'dev_metric_used': metric, 'dev_metric_value': None if best_value < 0 else best_value}
    return thresholds


def evaluate_multitask(y_true, y_prob, tasks, thresholds=None, primary_tasks=None, n_bins=10):
    thresholds = thresholds or {task: {'threshold': 0.5} for task in tasks}
    primary_tasks = primary_tasks or tasks
    task_metrics = {}
    for index, task in enumerate(tasks):
        threshold = thresholds.get(task, {}).get('threshold', 0.5)
        task_metrics[task] = binary_metrics(y_true[:, index], y_prob[:, index], threshold=threshold, n_bins=n_bins)
    aggregate = {}
    for key in [
        'auroc',
        'auprc',
        'brier',
        'ece',
        'nll',
        'pairwise_accuracy',
        'precision_at_10',
        'recall_at_10',
        'ndcg_at_10',
        'precision_at_50',
        'recall_at_50',
        'ndcg_at_50',
        'precision_at_100',
        'recall_at_100',
        'ndcg_at_100',
    ]:
        values = [task_metrics[task].get(key) for task in primary_tasks if task in task_metrics and task_metrics[task].get(key) is not None]
        aggregate[f'mean_primary_{key}'] = None if not values else float(np.mean(values))
    return {'tasks': task_metrics, 'aggregate': aggregate}
