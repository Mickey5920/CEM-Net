# CEM-Net

Reference implementation of **CEM-Net: Clinical Evidence Modulation for ICU Outcome Prediction**.

Xiangmin Luo, Da Huang, Ningyi Zhang, Jie Gao, Tao Tan\*, and Yue Sun\*  
Intelligent Medical Computing Laboratory, Faculty of Applied Sciences, Macao Polytechnic University, Macao, China  
(\*Corresponding authors: `taotanjs@gmail.com`, `yuesun@mpu.edu.mo`)

CEM-Net predicts prolonged ICU stays and ICU mortality at a 24-hour landmark. This repository provides the core feature, model, and evaluation components. The feature functions take deduplicated events with source-specific admissibility times and standardized laboratory values ordered by admissibility time. `configs/cem_net.json` summarizes the study settings.

## Quick Start

```bash
pip install -r requirements.txt
python examples/quickstart.py
```

This example uses synthetic data and a smaller model to exercise the feature, training, and evaluation code.

## Clinical Data Access

The study uses [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/), [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/), and [eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/). Access requires PhysioNet credentials and acceptance of the applicable data use agreements.

## License

This project is released under the [MIT License](LICENSE).
