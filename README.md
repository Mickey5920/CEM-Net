# CEM-Net

**CEM-Net: Clinical Evidence Modulation for ICU Outcome Prediction**

Xiangmin Luo, Da Huang, Ningyi Zhang, Jie Gao, Tao Tan, and Yue Sun.  
Intelligent Medical Computing Laboratory, Faculty of Applied Sciences, Macao Polytechnic University.  
Corresponding author: Yue Sun.

Source and aggregate reproducibility materials for the manuscript prepared for ICDE 2027. CEM-Net combines a reusable temporal evidence profile with compact joint response/modulation learning. Prolonged ICU length of stay is the primary outcome; mortality is a complementary outcome.

## Download the complete artifact

[**Fixed release v1.0.0**](https://github.com/Mickey5920/CEM-Net/releases/tag/v1.0.0) · [**Reproducibility ZIP**](https://github.com/Mickey5920/CEM-Net/releases/download/v1.0.0/CEM_Net_ICDE2027_reproducibility.zip)

The release ZIP contains the full frozen source snapshots, configurations, aggregate results, manuscript PDF/TeX/Markdown, tables, figures and reconstruction scripts. All original three-seed results and the five-seed core extension are retained. The repository front page provides the entry-point code and [formula/implementation guide](IMPLEMENTATION.md); **download the ZIP to run reproduction**.

## Quick start

Extract the ZIP on a project/data disk and enter its `reproducibility_v1` directory:

```powershell
python -B reproduce.py verify
python -B reproduce.py render --output ../cem_reproduction_output
```

Use a new output directory. The commands verify aggregate evidence and rebuild displays without clinical records, GPU training, model inference or new calibration. The packaged README documents the recorded Python environment and exact reconstruction scope.

Previously verified display reconstruction covers 20 generated TeX assets, the supplied cohort table and seven PNG figure families. The author version retains 12 body pages plus one acknowledgment/reference page.

## Data and training

MIMIC-III 1.4, MIMIC-IV 3.1 and eICU 2.0 require their respective PhysioNet access approvals and data-use agreements. The artifact includes no patient identifiers, raw clinical data, row-level predictions, clinical feature arrays or fitted weights.

The [implementation guide](IMPLEMENTATION.md) identifies the licensed-data pipeline and frozen job sources. Archived launchers retain host-specific input/cache paths and provenance checks. Aggregate reconstruction was verified; a fresh end-to-end clinical training replay was not performed during packaging.

## Licenses and attribution

Original project and third-party notices are preserved in the artifact snapshots. The inherited project license is included as [LICENSE](LICENSE); individual third-party components retain their own notices. Periodic numerical embeddings and multiplicative member factors are attributed in the manuscript.
