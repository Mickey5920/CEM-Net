# Formula and implementation guide

This guide maps the current manuscript to the supplied implementation. It explains the representation, parameter accounting and statistical reconstruction; the manuscript contains the definitions and evidence needed to assess its claims.

## 1. Source records and the prediction boundary

The landmark is 24 hours after ICU admission. Where both are supplied, a record's admissibility time is the maximum of its measurement/start time and recording time. The source-specific fields and single-timestamp proxies are listed in Section IV-A. MIMIC-III laboratory sampling time is the documented proxy when result-release time is unavailable.

Event tuples are deduplicated upstream. Training data determine the event vocabulary, laboratory keys, transformations and coordinate order. Unseen identities use source-specific unknown entries. The event accumulator receives zero-based type IDs, not padded loader IDs.

Relevant supplied sources:

- code/scripts/extract_tracehr_asof.py: source-time selection for MIMIC-IV/eICU.
- code/scripts/extract_tracehr_numeric.py: numeric extraction and identity handling.
- code/scripts/prepare_mimiciii_features.py: MIMIC-III profile preparation.
- code/scripts/tracehr_exact_statistics.py: EventStatistics.add, merge and features.

For each event identity, the coordinates are log-count, occurrence, latest admissibility time divided by 24, and the sum of exponential recency weights with a 12-hour scale. The array stores these as four vocabulary-wide blocks, not interleaved groups of four. Source log-counts and proportions follow, then four global coordinates.

For an empty event history, the original global count and density use a count floor of one: the global coordinates are [1/512, 0, 0, 1]. Per-type and per-source counts remain zero. This convention is covered by the synthetic interface checks.

## 2. Laboratory normalization and exact duration integrals

The laboratory builder chooses the most-covered recorded unit for each laboratory identifier on training stays, then selects 32 identifiers by training coverage. Distinct units are not numerically pooled. Same-key results with the same stay and admissibility time are combined by median before constructing the ordered path.

Values use training median centering and IQR scaling followed by clipping to [-5, 5]. If IQR is below 1e-8, training standard deviation is used; if that is also below 1e-8, the denominator is one. This value-level normalization precedes summary construction.

Let the unique admissibility times be

$$
0\le t_1<\cdots<t_m\le T,\qquad T=24,\qquad t_{m+1}=T,
$$

and let $z_i$ be the normalized value at $t_i$. The recorded path is constant at $z_i$ on $[t_i,t_{i+1})$, with basis

$$
b(z)=[1,z,z^2,(z-1)_+,(-z-1)_+].
$$

For $\tau\in\{1,6,24\}$, the manuscript's integral has the exact finite-sum implementation

$$
\psi_\tau=
\frac{\sum_{i=1}^{m}
\left[e^{-(T-t_{i+1})/\tau}-e^{-(T-t_i)/\tau}\right]b(z_i)}
{1-e^{-T/\tau}}.
$$

The factor $\tau$ from each integrated interval cancels the same factor in the full-window denominator. The duration_summary function computes interval masses in float64, multiplies by the basis matrix, normalizes, and returns float32. The 15 coordinates are ordered by $\tau=1,6,24$, with the five bases in the order above.

Implementation: code/scripts/tracehr_value_features.py (basis, duration_summary, simple_summary); laboratory selection/scaling: build_tracehr_value_features.py and prepare_mimiciii_features.py in the same directory.

Boundary behavior:

- No observation gives 15 zeros. A separate observation indicator distinguishes missingness from an observed median value.
- There is no contribution before the first observation.
- An observation first available exactly at $T$ has zero duration mass.
- Duplicate times are merged upstream; duration_summary requires strictly increasing times.
- A same-value insertion inside an existing constant segment splits its mass into two terms whose sum is unchanged. This assumes fixed transformations and the same recorded path; an earlier first observation or changed value need not preserve the integral.

Full-window normalization preserves coverage differences. It does not divide by observed duration or interpolate an unobserved physiological trajectory. Predictive comparisons between duration and alternative aggregation are reported separately in the paper.

## 3. Coordinates, model tensors and parameter accounting

Each of 32 laboratory keys supplies 11 observation/value summaries and 15 duration coordinates. With $V$ event identities and $S$ sources,

$$
d=4V+2S+4+32(11+15)=4V+2S+4+832.
$$

The model applies the training-fitted summary transform after construction. This is a second normalization stage: laboratory summaries use median/IQR scaling, half-range for zero IQR, and zero scale for constants. Soft clipping is $u/\sqrt{1+(u/3)^2}$; binary indicators pass through. This differs from the value-level IQR/SD/one fallback above.

Each eligible nonconstant numerical coordinate learns 16 frequencies, 16 phases, a $16\times3$ projection and three biases:

$$
c_j=\cos(2\pi\tilde x_j\omega_j+\varphi_j),\qquad
e_j=c_jW_j+b_j,\qquad 16+16+48+3=83.
$$

Concatenating the three-coordinate responses with the direct transformed profile gives $D=d+3q$. The direct path contains transformed summaries, not unprocessed source values.

| Database | $d$ | eligible coordinates $q$ | expanded width $D$ |
|---|---:|---:|---:|
| MIMIC-III | 13,672 | 10,417 | 44,923 |
| MIMIC-IV | 12,508 | 9,545 | 41,143 |
| eICU | 18,250 | 13,844 | 59,782 |

For a batch of $B$ stays, the encoded profile has shape $B\times D$. Eight member-factor vectors of shape $8\times D$ produce $B\times8\times D$ modulated inputs. A shared three-layer width-256 predictor produces $B\times8\times2$ logits. Training averages member cross-entropies with the manuscript's factor 0.5. Evaluation averages positive-class probabilities within one fitted model; it does not pool predictions across independent seeds.

Section V-E accounts for parameters as

$$
P=83q+8D+D+256D+133{,}122.
$$

The terms are embeddings, member factors, front scale, shared input matrix and remaining network. Member factors contain $1/32$ as many parameters as the input matrix. Sharing parameters does not eliminate member activations or computation.

EMA updates trainable embedding, member-factor and predictor parameters after optimizer updates, with $\beta_t=\min(0.999,(t+1)/(t+10))$. Optimization uses current parameters; checkpoint evaluation uses averaged parameters. Static buffers are unchanged.

Model-factory entry points are scripts/mimiciii_ema_common.py and scripts/ema_cross_common.py inside the frozen snapshots. The historical EMA_NRE_MLP identifier denotes the manuscript's full CEM-Net. Historical identifiers remain unchanged to match saved results.

## 4. Seed averages, bootstrap intervals and correction families

Core training comparisons use seeds 48--52. Original three-seed results remain available. Fixed-checkpoint member analyses, baseline perturbations, calibration, coverage and hospital influence retain their recorded seed scope.

A paired point estimate averages seed-specific metric differences:

$$
\widehat\Delta=\frac1S\sum_{s=1}^{S}
\left[M(y,p_{A,s})-M(y,p_{B,s})\right].
$$

Patient-cluster samples keep all stays belonging to a sampled patient together. Both methods and all seeds use the same sampled clusters. Each draw recomputes the metric for each seed and then averages differences. Hospital holdout analyses resample patient-connected hospital components.

An interaction uses the paired four-cell contrast $A_{11}-A_{10}-A_{01}+A_{00}$ on each draw. Each result records its own family and replicate count; legacy 4,000-draw analyses retain that designation.

For a two-sided family of size $m$, the adjusted percentile interval uses quantiles $0.05/(2m)$ and $1-0.05/(2m)$. The approximate centered-bootstrap p-value is

$$
p=\frac{1+\#\{b:|\Delta_b^*-\widehat\Delta|\ge|\widehat\Delta|\}}{B_{\rm boot}+1}.
$$

Holm correction is applied within the saved family. It differs from the Bonferroni interval rule, so a Holm p-value below 0.05 can coexist with a Bonferroni interval crossing zero. Bootstrap draws quantify cohort uncertainty conditional on saved fits; they are not additional training seeds.

The new secondary analysis contains 24 raw AUROC/Brier contrasts, five seeds and 20,000 patient-cluster draws, using one 24-comparison family. AP and older calibration analyses retain their original families.

The paired_draws.npz and compressed loss-difference files contain aggregate metric contrasts per draw, not patient predictions. The root verifier reconstructs the supported intervals without clinical records.

Relevant sources:

- code/scripts/cem_icde72_significance.py: centered p-values and Holm correction.
- code/scripts/cem_probability_intervals.py: paired probability-loss statistics.
- source/frozen_core_five_seed_v1/local_analysis_v5/: frozen core extension analysis.
- source/frozen_secondary_metrics_v1/source_v2/cem_secondary_metrics.py: the 24 secondary contrasts.
- source/report_cem_secondary_metrics.py: their manuscript table.

## 5. Licensed-data reconstruction route

Aggregate reproduction uses the root reproduce.py commands in the README. Clinical reconstruction additionally requires authorized MIMIC-III 1.4, MIMIC-IV 3.1 and eICU 2.0 inputs.

| Stage | Supplied sources or records | Required output |
|---|---|---|
| Cohort and partitions | Frozen extraction/adaptation scripts; prepare_mimiciii_features.py; original configurations and cohort audits | Landmark-eligible stays, endpoints and patient-disjoint partitions |
| Events and laboratories | extract_tracehr_asof.py, extract_tracehr_numeric.py, build_tracehr_value_features.py, tracehr_exact_statistics.py | Training vocabulary/normalization and aligned profiles |
| CEM-Net and controls | Frozen scripts/cem_core_five_seed_entry.py and work/cem_core_five_seed_20261107/config.json; model factories above | Seed/endpoint-specific models selected by validation AP under recorded budgets |
| Comparators | source/frozen_matched_v4/, source/frozen_sequence_v3/, source/frozen_medformer_v1/, source/frozen_targeted_v1/ | Original comparator recipes and evaluation outputs |
| Paired evaluation | Frozen statistics sources and saved family specifications | Per-seed metrics and paired aggregate intervals |
| Tables and figures | Root reproduce.py render command | Paper displays from aggregate results |

Executed source snapshots share readable files in `code/` and `code_variants/`. `source_bundle.json` stores file links; `source_loader.py` resolves them and verifies the original `source_manifest.json` hashes. Use `python -B source_loader.py list` and `export` to materialize a selected database/stage recipe into a new licensed-data workspace.

Archived launchers include original host paths, cached-input references and assertions against earlier manifests. Map these to a licensed-data environment and reconstruct their inputs before running workers. A fresh end-to-end clinical training replay was not performed during packaging; aggregate reconstruction is the verified entry point. No patient data, predictions, feature arrays or fitted weights are distributed.

## 6. Construction and inference measurements

Construction groups and orders admissible records, computes fixed-size reductions and materializes profiles. The stated $O(N\log N)$ bound includes sorting; output width is independent of history length. Event statistics merge after deduplication. Equal-time medians require complete groups, and duration segments require correct boundary values and times.

Construction and prediction measurements are separate. Construction trials retain their stay counts and CPU setup; duration/window comparisons use the same inputs. Prediction measurements distinguish workflow processing from resident inference and retain measured batch sizes, precision and hardware. These results describe the measured workloads rather than unmeasured deployment throughput.


## 7. Baseline architectures and training recipes

The complete baseline recipe table is retained here as implementation material. It preserves the configurations used for the reported results. The manuscript specifies the shared inputs, checkpoint selection, principal training budgets and matched-control design.

| Methods | Architecture and input transformation | Optimization, budget and stopping |
|---|---|---|
| RealMLP | Pre-tuned full classifier; complete evidence profile. | Batch 256; 256 epochs, no early stopping. |
| TabM / mini; with / without PLE | Three width-256 layers, 32 members, dropout 0.2; training-quantile transform. PLE: version B, dimension 16, no activation, up to 48 bins; merge duplicate edges; unit interval for constants. | AdamW, learning rate $5\times10^{-4}$, decay $10^{-3}$; effective batch 256 (PLE microbatch 32); cap 256, patience 16. |
| SeFT | Input/output: three width-32 layers. Attention embedding: two width-64 layers; four heads, latent/dot-product width 128. Time embedding 4, maximum scale 100, attention dropout 0.1. | Adam $10^{-3}$, no decay; batch/microbatch 64/4; cap 300, patience 30. Halve rate after 15 stale epochs; cooldown 15, floor $10^{-5}$; unweighted BCE. |
| mTAND encoder | Supervised: hidden 256, learned time embedding 128, one head, 128 reference queries. Sparse and reference dense implementations match in outputs and gradients. | Adam $10^{-4}$, no decay; batch/microbatch 256/4; cap/patience 200; two-class cross-entropy. |
| STraTS (ss$^{-}$) | Author supervised variant: hidden 64, two layers, 16 heads; dropout/attention dropout 0.2. Full observations; extra demographic coordinate zero; no pretraining. | Author AdamW $5\times10^{-4}$, $\epsilon=10^{-6}$, no decay, clip 0.3; batch/microbatch 16/1; cap 50, patience 10; train-ratio-weighted BCE. |
| LightGBM | 4.6.0, LGBM-TD: 50 leaves, unlimited depth, 255 bins, minimum leaf rows 40, Hessian $10^{-7}$, split gain 0. Zero is observed. | BCE; rate 0.04, row fraction 0.75, feature fraction 1, $L_1/L_2=0$, no class weights; cap 1,000, patience 300; four threads, deterministic column-wise layout. |
| XGBoost-TD | 3.2.0, CPU histogram trees: depth 6, 256 bins, minimum child weight $5\times10^{-6}$. Zero is observed. | BCE; rate 0.08, row fraction 0.65, column fraction per level 0.9, $L_1/L_2=0$; cap 1,000, patience 300; four threads. |

*Baseline architectures and training recipes. Checkpoints maximize validation AP; patience is in epochs or boosting rounds.*


### Hourly sequence models

TCN uses four width-128 blocks, kernel 4, dropout 0.2, AdamW, batch 16 and cap 50. ModernTCN uses one block at widths 32/64, patch size/stride 8/4, kernels 13/5, dropout 0.1, Adam, batch 64 and cap 100. Both use rate $5\times10^{-4}$ and AP patience 10. LITEMV is a numerically checked PyTorch translation of a single network: 32 filters, kernels 40/20/10, dilations 1/2/4, batch 64 and 1,500 epochs; Adam starts at $10^{-3}$, halving after 50 stale training-loss epochs to a floor of $10^{-4}$.

Medformer retains the authors' full classification graph: six width-128 layers, eight heads, feedforward width 256, 2/4/8-hour patches and dropout 0.1. Adam uses $10^{-4}$ with batch 128, cap 100 and AP patience 10. Arithmetic weight averaging starts after epoch one; validation AP selects averaged weights. All four use unweighted loss, zero decay and fixed recipes, without pretraining, augmentation or tuning.



## 8. Complete hourly-model AP intervals

These are the original CEM-Net-minus-comparator Test AP intervals, preserved when the main performance table was simplified to mean and sample SD. Columns follow the main table. CNN intervals are 99.8333%; Medformer intervals are 99.1667%. Their existing correction families are unchanged.

| Comparator | MIMIC-III LOS | MIMIC-IV LOS | eICU LOS | MIMIC-III mortality | MIMIC-IV mortality | eICU mortality |
|---|---|---|---|---|---|---|
| TCN | [+0.0309,+0.0770] | [+0.0476,+0.0850] | [+0.0404,+0.0773] | [+0.0383,+0.1133] | [+0.0654,+0.1197] | [+0.0603,+0.1162] |
| ModernTCN | [+0.0494,+0.0950] | [+0.0704,+0.1075] | [+0.0559,+0.0925] | [+0.0835,+0.1679] | [+0.0949,+0.1472] | [+0.0828,+0.1354] |
| LITEMV | [+0.0291,+0.0679] | [+0.0406,+0.0748] | [+0.0329,+0.0667] | [+0.0328,+0.1041] | [+0.0578,+0.1064] | [+0.0486,+0.0988] |
| Medformer | [+0.0139,+0.0532] | [+0.0465,+0.0785] | [+0.0413,+0.0747] | [+0.0258,+0.0929] | [+0.0563,+0.1010] | [+0.0696,+0.1165] |

## Preprocessing source locations

The public `code/scripts/` directory includes the MIMIC-III cohort adapter, source-time extraction, numeric extraction, event-feature construction and laboratory summaries. The adapter imports the landmark-label helper through `LABEL_CODE`; point it to `code/preprocessing_helpers/build_mimiciv_exp2_cohort_labels_v2.py` together with its `mimiciv_utils.py` dependency. Source files retain the executed path and checksum conventions; map data/preparation paths for the licensed local installation.
