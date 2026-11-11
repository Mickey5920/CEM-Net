# Fixed two-baseline comparison

Fixed-model patient-cluster conditional inference. Two-sided null-centered bootstrap p=(1+count(|draw-observed|>=|observed|))/(B+1), approximate. Bonferroni family=12 for each candidate and split across all three databases. Exact seed signed ranks are supplemental. Historical test exposure and training/selection uncertainty are not covered.

## Validation

|Database|Endpoint|Method|48|49|50|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|
|mimic|Mortality|XGBoost_TD|0.506421|0.511760|0.514609|0.510930 ± 0.004157|
|mimic|Mortality|STraTS_supervised|0.497263|0.503360|0.511175|0.503933 ± 0.006974|
|mimic|Mortality|EMA_NRE_MLP|0.547319|0.555403|0.544259|0.548994 ± 0.005757|
|mimic|Mortality|CEM-Net|0.547408|0.541267|0.540957|0.543211 ± 0.003638|
|mimic|LOS|XGBoost_TD|0.511558|0.513208|0.516947|0.513904 ± 0.002761|
|mimic|LOS|STraTS_supervised|0.522098|0.525969|0.518081|0.522049 ± 0.003944|
|mimic|LOS|EMA_NRE_MLP|0.562726|0.556884|0.561604|0.560405 ± 0.003100|
|mimic|LOS|CEM-Net|0.557108|0.554462|0.552945|0.554838 ± 0.002107|

|Database|Endpoint|Candidate minus baseline|Difference|95% CI|Family CI|p|Bonferroni p|Holm p|Seed signed-rank p|
|---|---|---|---:|---|---|---:|---:|---:|---:|
|mimic|0|CEM-Net − XGBoost_TD|0.032281|[0.017519, 0.047200]|[0.010988, 0.054124]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|1|CEM-Net − XGBoost_TD|0.040934|[0.028257, 0.053848]|[0.022095, 0.060248]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|0|CEM-Net − STraTS_supervised|0.039278|[0.021823, 0.055846]|[0.014131, 0.063433]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|1|CEM-Net − STraTS_supervised|0.032789|[0.020735, 0.044812]|[0.015058, 0.050427]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|0|EMA_NRE_MLP − XGBoost_TD|0.038063|[0.022908, 0.053345]|[0.016551, 0.060344]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|1|EMA_NRE_MLP − XGBoost_TD|0.046501|[0.033604, 0.059628]|[0.027136, 0.066027]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|0|EMA_NRE_MLP − STraTS_supervised|0.045061|[0.027159, 0.061834]|[0.019337, 0.070280]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|1|EMA_NRE_MLP − STraTS_supervised|0.038356|[0.026054, 0.050581]|[0.019966, 0.055762]|4.99975e-05|0.00059997|pending three databases|0.25|

## Test

|Database|Endpoint|Method|48|49|50|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|
|mimic|Mortality|XGBoost_TD|0.492324|0.499158|0.495551|0.495678 ± 0.003418|
|mimic|Mortality|STraTS_supervised|0.483303|0.486617|0.484922|0.484947 ± 0.001657|
|mimic|Mortality|EMA_NRE_MLP|0.546556|0.545002|0.542763|0.544773 ± 0.001907|
|mimic|Mortality|CEM-Net|0.547095|0.544330|0.542873|0.544766 ± 0.002145|
|mimic|LOS|XGBoost_TD|0.478198|0.478521|0.485842|0.480854 ± 0.004323|
|mimic|LOS|STraTS_supervised|0.502925|0.513287|0.510153|0.508788 ± 0.005314|
|mimic|LOS|EMA_NRE_MLP|0.547090|0.547492|0.546949|0.547177 ± 0.000282|
|mimic|LOS|CEM-Net|0.541155|0.541168|0.540199|0.540841 ± 0.000556|

|Database|Endpoint|Candidate minus baseline|Difference|95% CI|Family CI|p|Bonferroni p|Holm p|Seed signed-rank p|
|---|---|---|---:|---|---|---:|---:|---:|---:|
|mimic|0|CEM-Net − XGBoost_TD|0.049089|[0.038463, 0.059270]|[0.033778, 0.064175]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|1|CEM-Net − XGBoost_TD|0.059987|[0.050017, 0.069234]|[0.046007, 0.073633]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|0|CEM-Net − STraTS_supervised|0.059819|[0.048039, 0.071263]|[0.042433, 0.077050]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|1|CEM-Net − STraTS_supervised|0.032053|[0.022882, 0.041125]|[0.018695, 0.045768]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|0|EMA_NRE_MLP − XGBoost_TD|0.049096|[0.038031, 0.059707]|[0.032776, 0.064668]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|1|EMA_NRE_MLP − XGBoost_TD|0.066323|[0.055893, 0.076044]|[0.051394, 0.080995]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|0|EMA_NRE_MLP − STraTS_supervised|0.059826|[0.047466, 0.071949]|[0.042269, 0.077579]|4.99975e-05|0.00059997|pending three databases|0.25|
|mimic|1|EMA_NRE_MLP − STraTS_supervised|0.038388|[0.028941, 0.047850]|[0.024292, 0.052574]|4.99975e-05|0.00059997|pending three databases|0.25|

