# CEM-Net

**Clinical Evidence Modulation for ICU Outcome Prediction**

Xiangmin Luo, Da Huang, Ningyi Zhang, Jie Gao, Tao Tan, and Yue Sun.  
Intelligent Medical Computing Laboratory, Faculty of Applied Sciences, Macao Polytechnic University.  
Corresponding authors: Tao Tan and Yue Sun.

| Author | Email | ORCID |
|---|---|---|
| Xiangmin Luo | luoxm5920@163.com | [0000-0002-2915-5677](https://orcid.org/0000-0002-2915-5677) |
| Da Huang | huangda_97@163.com | [0000-0002-6375-0197](https://orcid.org/0000-0002-6375-0197) |
| Ningyi Zhang | p2417803@mpu.edu.mo | [0009-0009-0354-5528](https://orcid.org/0009-0009-0354-5528) |
| Jie Gao | gjttxs@gmail.com | [0009-0005-6274-0973](https://orcid.org/0009-0005-6274-0973) |
| Tao Tan | taotaojs@gmail.com | [0000-0001-5403-0887](https://orcid.org/0000-0001-5403-0887) |
| Yue Sun | yunsun@mpu.edu.mo | [0000-0003-0301-5112](https://orcid.org/0000-0003-0301-5112) |

Public code, configurations, aggregate results, and reproducibility utilities for the ICDE 2027 submission. Prolonged ICU length of stay is the primary outcome; mortality is a complementary outcome.

[Fixed release v1.0.1](https://github.com/Mickey5920/CEM-Net/releases/tag/v1.0.1) · [Code and reproduction ZIP](https://github.com/Mickey5920/CEM-Net/releases/download/v1.0.1/CEM_Net_ICDE2027_code_and_reproduction_v1.0.1.zip)

## Reproduce the reported displays

Clone this repository or extract the release ZIP. Using Python 3.12 and the recorded rendering dependencies:

```sh
python -m pip install -r requirements-reproduction.txt
python -B reproduce.py verify
python -B reproduce.py render --output ../cem_reproduction_output
```

Use a new output directory on a project/data disk. These commands verify the saved evidence and reconstruct tables and plots without clinical records, GPU training, model inference, or new calibration. Pixel-identical plots additionally require **Times New Roman**, as in the recorded environment. All statistical values are read from preserved aggregate results.

The reconstruction checks cover 20 generated TeX assets, one supplied cohort table, and nine PNG figure families. Temporal-evidence gains use an interval plot, and construction costs use annotated heatmaps; the original numerical tables remain available. Some retained displays correspond to secondary analyses rather than the final paper layout. Core comparisons use seeds 48–52; the original three-seed results and recorded uncertainty are retained.

The main AP table reports mean and sample SD. Complete baseline recipes and all 24 hourly-model AP contrast intervals are preserved in [IMPLEMENTATION.md](IMPLEMENTATION.md#7-baseline-architectures-and-training-recipes). Version 1.0.1 synchronizes author information, implementation documentation and the latest display layout; model code, configurations and all aggregate-result files are unchanged from v1.0.0.

## Code map

| Component | Entry points |
|---|---|
| Cohort and labels | [MIMIC-III adapter](code/scripts/adapt_mimiciii.py), [shared landmark-label implementation](code/preprocessing_helpers/build_mimiciv_exp2_cohort_labels_v2.py) |
| Source timestamps and numeric records | [as-of extraction](code/scripts/extract_tracehr_asof.py), [numeric extraction](code/scripts/extract_tracehr_numeric.py) |
| Feature construction | [MIMIC-III profiles](code/scripts/prepare_mimiciii_features.py), [event summaries](code/scripts/tracehr_exact_statistics.py), [laboratory summaries and integrals](code/scripts/tracehr_value_features.py), [laboratory builder](code/scripts/build_tracehr_value_features.py) |
| Model architecture | [numerical responses](code/models/tracehr_unified.py), [shared predictor and input factors](code/models/tracehr_fte_binding.py), [MIMIC-III factory](code/scripts/mimiciii_fixed_common.py), [cross-database factory](code/scripts/ema_cross_common.py) |
| Training and EMA | [five-seed entry](code/scripts/cem_core_five_seed_entry.py), [model/recipe routing](code/scripts/cem_core_five_seed_common.py), [EMA runtime](code/scripts/mimiciii_ema_runtime.py) |
| Configurations | [recorded run recipes](recipes/run_index.json), [runtime configuration](code/work/cem_core_five_seed_20261107/config.json), [environments](environment.json) |
| Statistics and figures | [analysis sources](source/current_analysis/scripts), [five-seed analysis](source/frozen_core_five_seed_v1/local_analysis_v5), [secondary metrics](source/report_cem_secondary_metrics.py), [table/figure renderer](source/paper_integration.py) |
| Saved results | [aggregate results](aggregate/runs) |

[IMPLEMENTATION.md](IMPLEMENTATION.md) explains time boundaries, normalization, duration integrals, tensor shapes, and paired statistical calculations.

## Licensed-data training

Access to MIMIC-III 1.4, MIMIC-IV 3.1 and eICU 2.0 requires the corresponding PhysioNet approvals and data-use agreements. No patient identifiers, raw records, row-level predictions, clinical feature arrays, or fitted weights are included.

The readable `code/` tree is a complete reference runtime, with preprocessing helpers included. Database/stage-specific executed versions share unchanged files and retain their differences in `code_variants/`. To export an exact training source state, first list the available recipes:

```sh
python -B source_loader.py list
python -B source_loader.py export frozen_core_five_seed_v1/local_mimiciii_replacement_v1 --output ../cem_licensed_workspace
```

The export validates every source hash and does not launch training. Use the exported recipe's input preparation, endpoint budget and environment recorded in `config.json` and `environment.json`. Original launchers retain site-specific paths and checks against earlier preparation/checkpoint records; those inputs must be reconstructed and paths mapped for another installation. In particular, the MIMIC-III adapter's `ARCHIVE` and `LABEL_CODE` locations must point to the authorized archive and the included label helper. The public package verifies aggregate reconstruction and source integrity; it is not a tested one-command rerun from raw clinical databases.

Historical class/method identifiers remain in code and result keys so that saved results can be traced to the executed implementation. No model, training recipe, stopping rule, or reported result was changed for this release.

## Contents and license

This release contains code, configuration records, aggregate statistics, source-verification metadata, and display references. It excludes the manuscript PDF/TeX/Markdown and internal writing archives. Repeated encoded source archives are replaced by links to readable source files; the original source hashes are preserved.

The inherited project [LICENSE](LICENSE) and third-party notices in the source trees are retained. Periodic numerical embeddings and multiplicative member factors retain their technical attribution.
