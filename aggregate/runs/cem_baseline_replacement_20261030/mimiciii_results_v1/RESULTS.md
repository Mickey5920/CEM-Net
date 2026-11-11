# Fixed two-baseline comparison

Fixed-model patient-cluster conditional inference. Two-sided null-centered bootstrap p=(1+count(|draw-observed|>=|observed|))/(B+1), approximate. Bonferroni family=12 for each candidate and split across all three databases. Exact seed signed ranks are supplemental. Historical test exposure and training/selection uncertainty are not covered.

## Validation

|Database|Endpoint|Method|48|49|50|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|
|mimiciii|Mortality|XGBoost_TD|0.437556|0.429028|0.432507|0.433031 ± 0.004288|
|mimiciii|Mortality|STraTS_supervised|0.451522|0.451143|0.445026|0.449230 ± 0.003646|
|mimiciii|Mortality|EMA_NRE_MLP|0.508035|0.488398|0.492720|0.496385 ± 0.010319|
|mimiciii|Mortality|CEM-Net|0.490891|0.489410|0.488207|0.489503 ± 0.001344|
|mimiciii|LOS|XGBoost_TD|0.512447|0.521578|0.519803|0.517943 ± 0.004841|
|mimiciii|LOS|STraTS_supervised|0.506755|0.514017|0.517467|0.512746 ± 0.005468|
|mimiciii|LOS|EMA_NRE_MLP|0.537479|0.532122|0.534473|0.534691 ± 0.002685|
|mimiciii|LOS|CEM-Net|0.534294|0.528870|0.532288|0.531817 ± 0.002742|

|Database|Endpoint|Candidate minus baseline|Difference|95% CI|Family CI|p|Bonferroni p|Holm p|Seed signed-rank p|
|---|---|---|---:|---|---|---:|---:|---:|---:|
|mimiciii|0|CEM-Net − XGBoost_TD|0.056472|[0.031345, 0.079021]|[0.020558, 0.089736]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimiciii|1|CEM-Net − XGBoost_TD|0.013875|[-0.002024, 0.029521]|[-0.010260, 0.036630]|0.0855457|1|pending three databases|0.25|
|mimiciii|0|CEM-Net − STraTS_supervised|0.040273|[0.016282, 0.064046]|[0.006480, 0.075078]|0.00104995|0.0125994|pending three databases|0.25|
|mimiciii|1|CEM-Net − STraTS_supervised|0.019071|[0.002446, 0.035665]|[-0.005009, 0.044310]|0.0250987|0.301185|pending three databases|0.25|
|mimiciii|0|EMA_NRE_MLP − XGBoost_TD|0.063354|[0.036853, 0.086684]|[0.024318, 0.098597]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimiciii|1|EMA_NRE_MLP − XGBoost_TD|0.016749|[-0.000054, 0.032849]|[-0.008202, 0.040094]|0.0464477|0.557372|pending three databases|0.25|
|mimiciii|0|EMA_NRE_MLP − STraTS_supervised|0.047154|[0.022359, 0.071395]|[0.010355, 0.083455]|0.000449978|0.00539973|pending three databases|0.25|
|mimiciii|1|EMA_NRE_MLP − STraTS_supervised|0.021945|[0.004279, 0.038730]|[-0.003921, 0.047224]|0.0128994|0.154792|pending three databases|0.25|

## Test

|Database|Endpoint|Method|48|49|50|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|
|mimiciii|Mortality|XGBoost_TD|0.446333|0.434081|0.445651|0.442022 ± 0.006885|
|mimiciii|Mortality|STraTS_supervised|0.460636|0.465609|0.436992|0.454413 ± 0.015290|
|mimiciii|Mortality|EMA_NRE_MLP|0.494310|0.485638|0.489187|0.489712 ± 0.004360|
|mimiciii|Mortality|CEM-Net|0.490450|0.481090|0.487356|0.486299 ± 0.004769|
|mimiciii|LOS|XGBoost_TD|0.511158|0.511963|0.508686|0.510603 ± 0.001708|
|mimiciii|LOS|STraTS_supervised|0.514689|0.524444|0.519253|0.519462 ± 0.004881|
|mimiciii|LOS|EMA_NRE_MLP|0.551003|0.549535|0.552543|0.551027 ± 0.001504|
|mimiciii|LOS|CEM-Net|0.546551|0.550021|0.550603|0.549058 ± 0.002191|

|Database|Endpoint|Candidate minus baseline|Difference|95% CI|Family CI|p|Bonferroni p|Holm p|Seed signed-rank p|
|---|---|---|---:|---|---|---:|---:|---:|---:|
|mimiciii|0|CEM-Net − XGBoost_TD|0.044277|[0.025525, 0.062933]|[0.017624, 0.072075]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimiciii|1|CEM-Net − XGBoost_TD|0.038456|[0.026651, 0.049774]|[0.021276, 0.055483]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimiciii|0|CEM-Net − STraTS_supervised|0.031886|[0.014316, 0.049187]|[0.005915, 0.056847]|0.000499975|0.0059997|pending three databases|0.25|
|mimiciii|1|CEM-Net − STraTS_supervised|0.029596|[0.017424, 0.041810]|[0.012001, 0.046887]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimiciii|0|EMA_NRE_MLP − XGBoost_TD|0.047690|[0.028513, 0.066770]|[0.020288, 0.075525]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimiciii|1|EMA_NRE_MLP − XGBoost_TD|0.040424|[0.028038, 0.052288]|[0.022355, 0.057985]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimiciii|0|EMA_NRE_MLP − STraTS_supervised|0.035299|[0.017382, 0.052949]|[0.009417, 0.060877]|9.9995e-05|0.00119994|pending three databases|0.25|
|mimiciii|1|EMA_NRE_MLP − STraTS_supervised|0.031565|[0.018800, 0.044243]|[0.013411, 0.049485]|4.99975e-05|0.00059997|pending three databases|0.25|

