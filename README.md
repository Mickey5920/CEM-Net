# CEM-Net

Official implementation of **CEM-Net: Clinical Evidence Modulation for ICU Outcome Prediction**.

Xiangmin Luo, Da Huang, Ningyi Zhang, Jie Gao, Tao Tan\*, and Yue Sun\*  
Intelligent Medical Computing Laboratory, Faculty of Applied Sciences, Macao Polytechnic University, Macao, China  
(\*Corresponding authors: `taotanjs@gmail.com`, `yuesun@mpu.edu.mo`)

## Overview

CEM-Net predicts prolonged ICU length of stay (>7 days) and ICU mortality at a 24-hour landmark by separating explicit temporal evidence construction from a compact neural predictor:

1. **Temporal Evidence Profile (`cem_net/evidence.py`)**: Enforces source-specific record admissibility at $T=24$ hours, summarizes categorical event burden/recency/decayed activity, and pairs 11-item laboratory summaries with closed-form 15-item recorded-state duration integrals ($\tau \in \{1, 6, 24\}$ h).
2. **Joint Evidence Modulation (`cem_net/model.py`)**: Learns coordinate-wise periodic numerical embeddings (16 cosine bases projected to 3 channels) while retaining the direct scaled profile, modulated by $K=8$ member-specific input factor vectors over a shared 3-layer width-256 pSELU predictor with EMA parameter smoothing.
3. **Evaluation & Interpretability (`cem_net/evaluation.py`)**: Includes AP/AUROC/Brier/log-loss/ECE15 metrics, validation-fitted Platt calibration, Decision Curve Analysis (DCA), and centered Gradient $\times$ Input attribution.

## Repository Structure

```text
CEM-Net/
├── cem_net/
│   ├── evidence.py      # Event sufficient statistics, lab summaries, duration integrals
│   ├── model.py         # TypedPeriodic embeddings, 8-member modulation, shared pSELU trunk, EMA
│   ├── schedule.py      # coslog4 learning-rate/dropout schedules and parameter-group decay
│   └── evaluation.py    # Metrics, Platt calibration, DCA, and Gradient x Input attribution
├── configs/
│   └── cem_net.json     # Model and optimization hyperparameters
├── examples/
│   └── quickstart.py    # Self-contained synthetic smoke test
├── requirements.txt
└── LICENSE
```

## Quick Start

```bash
pip install -r requirements.txt
python examples/quickstart.py
```

## Clinical Data Access

Experiments use three public PhysioNet critical care databases:

- **MIMIC-III v1.4** (`doi:10.13026/C2XW26`)
- **MIMIC-IV v3.1** (`doi:10.13026/kpb9-mt58`)
- **eICU Collaborative Research Database v2.0** (`doi:10.13026/C2WM1R`)

Access requires credentialed PhysioNet approval and acceptance of the corresponding Data Use Agreements. No patient-level records or trained checkpoints are included in this repository.

## License

This project is released under the [MIT License](LICENSE).
