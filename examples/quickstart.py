"""Minimal synthetic smoke test for CEM-Net evidence construction, training, calibration, and DCA."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cem_net.evidence import EventStatistics, simple_lab_summary, duration_lab_summary
from cem_net.model import CEMNet, EMATracker, fit_profile_normalizer
from cem_net.evaluation import binary_metrics, fit_platt, apply_platt, decision_curve, centered_grad_x_input


def make_synthetic_cohort(n_stays=256, n_types=12, n_labs=4, seed=48):
    rng = np.random.default_rng(seed)
    sources = ("chart", "input", "lab", "procedure_icu")
    rows, labels = [], []
    for _ in range(n_stays):
        risk = rng.normal()
        ev = EventStatistics(n_types, sources, window=24.0)
        n_ev = int(rng.integers(5, 25))
        ev.add(rng.integers(0, n_types, n_ev), np.sort(rng.uniform(0.0, 24.0, n_ev)), rng.integers(0, len(sources), n_ev))
        lab_basic, lab_dur = [], []
        for _ in range(n_labs):
            m = int(rng.integers(1, 5))
            ts = np.sort(rng.uniform(0.5, 23.5, m))
            vals = np.clip(rng.normal(0.5 * risk, 1.0, m), -5.0, 5.0)
            lab_basic.append(simple_lab_summary(ts, vals))
            lab_dur.append(duration_lab_summary(ts, vals))
        rows.append(np.concatenate([ev.features(), *lab_basic, *lab_dur]))
        labels.append(int(rng.random() < (1.0 / (1.0 + np.exp(-(1.2 * risk - 1.5))))))
    return np.stack(rows).astype(np.float32), np.asarray(labels, dtype=np.int64), 4 * n_types + 2 * len(sources) + 4


def main():
    torch.manual_seed(48)
    x, y, phi_dim = make_synthetic_cohort()
    x_tr, y_tr = x[:160], y[:160]
    x_va, y_va = x[160:208], y[160:208]
    x_te, y_te = x[208:], y[208:]

    prep = fit_profile_normalizer(x_tr, phi_dim=phi_dim)
    model = CEMNet(prep, members=8, hidden_width=64, depth=3)
    ema = EMATracker(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    x_tr_t, y_tr_t = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    for _ in range(15):
        opt.zero_grad()
        loss = model.loss(model(x_tr_t), y_tr_t)
        loss.backward()
        opt.step()
        ema.update(model)

    with torch.no_grad():
        p_va = ema.shadow.probabilities(torch.from_numpy(x_va)).numpy()
        p_te = ema.shadow.probabilities(torch.from_numpy(x_te)).numpy()

    cal = fit_platt(y_va, p_va)
    p_te_cal = apply_platt(p_te, cal)
    dca = decision_curve(y_te, p_te_cal, thresholds=[0.10, 0.20, 0.30])
    attr = centered_grad_x_input(ema.shadow, torch.from_numpy(x_te))

    print(f"Profile dim: {x.shape[1]} (phi={phi_dim}, labs={x.shape[1]-phi_dim}), params: {sum(p.numel() for p in model.parameters()):,}")
    print("Raw test metrics:  ", {k: round(v, 4) for k, v in binary_metrics(y_te, p_te).items()})
    print("Platt test metrics:", {k: round(v, 4) for k, v in binary_metrics(y_te, p_te_cal).items()})
    print("DCA net benefit @ [0.10, 0.20, 0.30]:", np.round(dca["net_benefit"], 4).tolist())
    print("Top-3 attributed feature indices:", np.argsort(-attr)[:3].tolist())


if __name__ == "__main__":
    main()
