"""Evaluation metrics, Platt calibration, Decision Curve Analysis, and attribution helpers."""
import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss


def ece15(y_true, probs) -> float:
    """15-bin equal-width expected calibration error."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, 16)
    groups = np.minimum(np.searchsorted(edges, p, side="right") - 1, 14)
    ece = 0.0
    for j in range(15):
        mask = groups == j
        if mask.any():
            ece += (mask.sum() / len(y)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def binary_metrics(y_true, probs) -> dict:
    """Compute AP, AUROC, Brier score, log loss, and 15-bin ECE."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    return dict(
        AP=float(average_precision_score(y, p)),
        AUROC=float(roc_auc_score(y, p)),
        Brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        ECE15=ece15(y, p),
    )


def fit_platt(y_val, p_val) -> dict:
    """Fit unregularized logistic Platt recalibration strictly on validation logits."""
    pc = np.clip(np.asarray(p_val, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(pc / (1.0 - pc))[:, None]
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, tol=1e-10).fit(logits, y_val)
    return dict(coef=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))


def apply_platt(probs, calibrator: dict) -> np.ndarray:
    """Apply frozen validation-fitted Platt parameters to held-out probabilities."""
    pc = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(pc / (1.0 - pc))
    return expit(calibrator["coef"] * logits + calibrator["intercept"])


def decision_curve(y_true, probs, thresholds=None) -> dict:
    """Compute clinical net benefit NB(p_t) across threshold probabilities (Vickers & Elkin, 2006)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    ts = np.linspace(0.01, 0.35, 35) if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    pred_pos = (p[:, None] >= ts[None, :]).astype(np.float64)
    n = float(len(y))
    tp = (pred_pos * y[:, None]).sum(axis=0)
    fp = (pred_pos * (1.0 - y[:, None])).sum(axis=0)
    odds = ts / (1.0 - ts)
    nb_model = (tp / n) - (fp / n) * odds
    prev = float(y.mean())
    nb_all = prev - (1.0 - prev) * odds
    return dict(thresholds=ts, net_benefit=nb_model, treat_all=nb_all, treat_none=np.zeros_like(ts))


def centered_grad_x_input(model, x_raw: torch.Tensor) -> np.ndarray:
    """Compute per-feature centered Gradient x Input attribution folded over direct + periodic paths."""
    model.eval()
    with torch.no_grad():
        h0 = model.numerical(x_raw)
        h0_mean = h0.mean(dim=0, keepdim=True)
    h0_req = h0.detach().requires_grad_(True)
    logits = model.forward_trunk(h0_req)
    margin = (logits[..., 1] - logits[..., 0]).mean(dim=1)
    (grad,) = torch.autograd.grad(margin.sum(), h0_req)
    with torch.no_grad():
        attr_D = grad * (h0 - h0_mean)
        d = len(model.numerical.center)
        folded = torch.zeros((len(x_raw), d), dtype=attr_D.dtype, device=attr_D.device)
        folded.scatter_add_(1, model.feature_map.unsqueeze(0).expand(len(x_raw), -1), attr_D)
        return folded.abs().mean(dim=0).cpu().numpy()
