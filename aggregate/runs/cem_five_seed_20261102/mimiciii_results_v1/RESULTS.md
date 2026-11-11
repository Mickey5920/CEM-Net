# Five-seed Table VI comparison

Five paired fixed fits; common patient-cluster resamples. Original 54-comparison Bonferroni allowance retained for nine fixed Table VI contrasts, separately per split. Null-centered Bootstrap p is approximate and never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Historical Test exposure and training-population/selection uncertainty remain outside these conditional intervals.

## dev

|Database|Endpoint|Method|48|49|50|51|52|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|---:|---:|
|mimiciii|0|CEM-Net|0.490891|0.489410|0.488207|0.482168|0.478397|0.485815 ± 0.005309|
|mimiciii|1|CEM-Net|0.534294|0.528870|0.532288|0.532083|0.534585|0.532424 ± 0.002287|
|mimiciii|1|LightGBM_full|0.526799|0.516767|0.522410|0.529648|0.529487|0.525022 ± 0.005465|
|mimiciii|1|TabM-mini|0.525176|0.524697|0.521449|0.527287|0.521380|0.523998 ± 0.002552|
|mimiciii|0|TabM_PLE|0.476779|0.472779|0.473961|0.478255|0.479194|0.476193 ± 0.002750|
|mimiciii|0|TabM_mini_PLE|0.472446|0.469515|0.475402|0.471499|0.472861|0.472345 ± 0.002142|
## test

|Database|Endpoint|Method|48|49|50|51|52|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|---:|---:|
|mimiciii|0|CEM-Net|0.490450|0.481090|0.487356|0.479051|0.489366|0.485463 ± 0.005097|
|mimiciii|1|CEM-Net|0.546551|0.550021|0.550603|0.550880|0.551568|0.549924 ± 0.001966|
|mimiciii|1|LightGBM_full|0.530051|0.536435|0.536014|0.527674|0.529927|0.532020 ± 0.003956|
|mimiciii|1|TabM-mini|0.533134|0.510369|0.527032|0.531369|0.531253|0.526631 ± 0.009364|
|mimiciii|0|TabM_PLE|0.485740|0.472591|0.472029|0.479220|0.468979|0.475712 ± 0.006736|
|mimiciii|0|TabM_mini_PLE|0.484515|0.477548|0.477714|0.473403|0.476093|0.477855 ± 0.004105|

All paired differences, nominal/corrected conditional intervals and p values are in verified-results.json. Original three-seed analyses remain unchanged.
