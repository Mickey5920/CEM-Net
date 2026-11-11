# Fixed two-baseline comparison

Fixed-model patient-cluster conditional inference. Two-sided null-centered bootstrap p=(1+count(|draw-observed|>=|observed|))/(B+1), approximate. Bonferroni family=12 for each candidate and split across all three databases. Exact seed signed ranks are supplemental. Historical test exposure and training/selection uncertainty are not covered.

## Validation

|Database|Endpoint|Method|48|49|50|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|
|eicu|Mortality|XGBoost_TD|0.338397|0.307527|0.319266|0.321730 ± 0.015582|
|eicu|Mortality|STraTS_supervised|0.386276|0.381368|0.398888|0.388844 ± 0.009038|
|eicu|Mortality|EMA_NRE_MLP|0.432845|0.438736|0.435804|0.435795 ± 0.002946|
|eicu|Mortality|CEM-Net|0.434527|0.431635|0.430567|0.432243 ± 0.002049|
|eicu|LOS|XGBoost_TD|0.361224|0.347459|0.349654|0.352779 ± 0.007396|
|eicu|LOS|STraTS_supervised|0.364119|0.367058|0.378163|0.369780 ± 0.007407|
|eicu|LOS|EMA_NRE_MLP|0.396759|0.400604|0.404282|0.400548 ± 0.003761|
|eicu|LOS|CEM-Net|0.401118|0.397699|0.398755|0.399191 ± 0.001751|

|Database|Endpoint|Candidate minus baseline|Difference|95% CI|Family CI|p|Bonferroni p|Holm p|Seed signed-rank p|
|---|---|---|---:|---|---|---:|---:|---:|---:|
|eicu|0|CEM-Net − XGBoost_TD|0.110513|[0.088762, 0.131223]|[0.079135, 0.140903]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|1|CEM-Net − XGBoost_TD|0.046412|[0.033528, 0.059263]|[0.027756, 0.066309]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|0|CEM-Net − STraTS_supervised|0.043399|[0.024015, 0.062803]|[0.014945, 0.071363]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|1|CEM-Net − STraTS_supervised|0.029411|[0.017221, 0.041414]|[0.011640, 0.047051]|9.9995e-05|0.00119994|pending three databases|0.25|
|eicu|0|EMA_NRE_MLP − XGBoost_TD|0.114065|[0.090527, 0.136440]|[0.079570, 0.146789]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|1|EMA_NRE_MLP − XGBoost_TD|0.047769|[0.034596, 0.060605]|[0.029268, 0.067604]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|0|EMA_NRE_MLP − STraTS_supervised|0.046951|[0.028860, 0.065413]|[0.020201, 0.073873]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|1|EMA_NRE_MLP − STraTS_supervised|0.030768|[0.018728, 0.042535]|[0.013028, 0.048036]|4.99975e-05|0.00059997|pending three databases|0.25|

## Test

|Database|Endpoint|Method|48|49|50|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|
|eicu|Mortality|XGBoost_TD|0.284242|0.282251|0.293983|0.286825 ± 0.006278|
|eicu|Mortality|STraTS_supervised|0.348916|0.349433|0.348934|0.349094 ± 0.000293|
|eicu|Mortality|EMA_NRE_MLP|0.399064|0.396370|0.400782|0.398739 ± 0.002224|
|eicu|Mortality|CEM-Net|0.393539|0.397945|0.398310|0.396598 ± 0.002655|
|eicu|LOS|XGBoost_TD|0.340674|0.342179|0.331564|0.338139 ± 0.005744|
|eicu|LOS|STraTS_supervised|0.353562|0.362927|0.367218|0.361235 ± 0.006983|
|eicu|LOS|EMA_NRE_MLP|0.397717|0.394530|0.394044|0.395430 ± 0.001995|
|eicu|LOS|CEM-Net|0.392219|0.393073|0.394441|0.393244 ± 0.001121|

|Database|Endpoint|Candidate minus baseline|Difference|95% CI|Family CI|p|Bonferroni p|Holm p|Seed signed-rank p|
|---|---|---|---:|---|---|---:|---:|---:|---:|
|eicu|0|CEM-Net − XGBoost_TD|0.109772|[0.093958, 0.125050]|[0.086780, 0.132197]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|1|CEM-Net − XGBoost_TD|0.055105|[0.045404, 0.065046]|[0.040494, 0.069673]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|0|CEM-Net − STraTS_supervised|0.047504|[0.033760, 0.061197]|[0.027827, 0.067630]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|1|CEM-Net − STraTS_supervised|0.032009|[0.022581, 0.041508]|[0.018003, 0.045707]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|0|EMA_NRE_MLP − XGBoost_TD|0.111914|[0.095400, 0.127262]|[0.087926, 0.135021]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|1|EMA_NRE_MLP − XGBoost_TD|0.057291|[0.047328, 0.067113]|[0.042338, 0.071749]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|0|EMA_NRE_MLP − STraTS_supervised|0.049645|[0.036784, 0.062046]|[0.030791, 0.067821]|4.99975e-05|0.00059997|pending three databases|0.25|
|eicu|1|EMA_NRE_MLP − STraTS_supervised|0.034195|[0.024914, 0.043575]|[0.020372, 0.047491]|4.99975e-05|0.00059997|pending three databases|0.25|

