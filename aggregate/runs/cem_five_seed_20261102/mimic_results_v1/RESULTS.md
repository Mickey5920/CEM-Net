# Five-seed Table VI comparison

Five paired fixed fits; common patient-cluster resamples. Original 54-comparison Bonferroni allowance retained for nine fixed Table VI contrasts, separately per split. Null-centered Bootstrap p is approximate and never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Historical Test exposure and training-population/selection uncertainty remain outside these conditional intervals.

## dev

|Database|Endpoint|Method|48|49|50|51|52|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|---:|---:|
|mimic|0|CEM-Net|0.547408|0.541267|0.540957|0.544462|0.545462|0.543911 ± 0.002768|
|mimic|1|CEM-Net|0.557108|0.554462|0.552945|0.558495|0.557048|0.556012 ± 0.002250|
|mimic|0|TabM_PLE|0.532593|0.532583|0.536066|0.538701|0.536019|0.535193 ± 0.002613|
|mimic|1|TabM_PLE|0.551953|0.553943|0.556117|0.551210|0.552440|0.553133 ± 0.001945|
## test

|Database|Endpoint|Method|48|49|50|51|52|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|---:|---:|
|mimic|0|CEM-Net|0.547095|0.544330|0.542873|0.543473|0.541859|0.543926 ± 0.001987|
|mimic|1|CEM-Net|0.541155|0.541168|0.540199|0.541606|0.542081|0.541242 ± 0.000696|
|mimic|0|TabM_PLE|0.537051|0.540581|0.539043|0.540183|0.542841|0.539940 ± 0.002124|
|mimic|1|TabM_PLE|0.534689|0.534239|0.535456|0.530673|0.530456|0.533103 ± 0.002359|

All paired differences, nominal/corrected conditional intervals and p values are in verified-results.json. Original three-seed analyses remain unchanged.
