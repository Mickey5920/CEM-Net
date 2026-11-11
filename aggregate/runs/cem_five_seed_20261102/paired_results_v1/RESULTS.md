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
|mimic|0|CEM-Net|0.547408|0.541267|0.540957|0.544462|0.545462|0.543911 ± 0.002768|
|mimic|1|CEM-Net|0.557108|0.554462|0.552945|0.558495|0.557048|0.556012 ± 0.002250|
|mimic|0|TabM_PLE|0.532593|0.532583|0.536066|0.538701|0.536019|0.535193 ± 0.002613|
|mimic|1|TabM_PLE|0.551953|0.553943|0.556117|0.551210|0.552440|0.553133 ± 0.001945|
|eicu|0|CEM-Net|0.434527|0.431635|0.430567|0.438381|0.432087|0.433439 ± 0.003120|
|eicu|1|CEM-Net|0.401118|0.397699|0.398755|0.402662|0.401367|0.400320 ± 0.002033|
|eicu|1|TabM_PLE|0.401285|0.399827|0.399644|0.401364|0.398912|0.400206 ± 0.001077|
|eicu|0|TabM_mini_PLE|0.439761|0.432539|0.435682|0.434413|0.435658|0.435611 ± 0.002651|
|eicu|1|TabM_mini_PLE|0.400370|0.401897|0.398581|0.401905|0.400145|0.400580 ± 0.001389|
## test

|Database|Endpoint|Method|48|49|50|51|52|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|---:|---:|
|mimiciii|0|CEM-Net|0.490450|0.481090|0.487356|0.479051|0.489366|0.485463 ± 0.005097|
|mimiciii|1|CEM-Net|0.546551|0.550021|0.550603|0.550880|0.551568|0.549924 ± 0.001966|
|mimiciii|1|LightGBM_full|0.530051|0.536435|0.536014|0.527674|0.529927|0.532020 ± 0.003956|
|mimiciii|1|TabM-mini|0.533134|0.510369|0.527032|0.531369|0.531253|0.526631 ± 0.009364|
|mimiciii|0|TabM_PLE|0.485740|0.472591|0.472029|0.479220|0.468979|0.475712 ± 0.006736|
|mimiciii|0|TabM_mini_PLE|0.484515|0.477548|0.477714|0.473403|0.476093|0.477855 ± 0.004105|
|mimic|0|CEM-Net|0.547095|0.544330|0.542873|0.543473|0.541859|0.543926 ± 0.001987|
|mimic|1|CEM-Net|0.541155|0.541168|0.540199|0.541606|0.542081|0.541242 ± 0.000696|
|mimic|0|TabM_PLE|0.537051|0.540581|0.539043|0.540183|0.542841|0.539940 ± 0.002124|
|mimic|1|TabM_PLE|0.534689|0.534239|0.535456|0.530673|0.530456|0.533103 ± 0.002359|
|eicu|0|CEM-Net|0.393539|0.397945|0.398310|0.382758|0.388228|0.392156 ± 0.006653|
|eicu|1|CEM-Net|0.392219|0.393073|0.394441|0.390958|0.395489|0.393236 ± 0.001788|
|eicu|1|TabM_PLE|0.393863|0.394373|0.392126|0.396614|0.397145|0.394824 ± 0.002061|
|eicu|0|TabM_mini_PLE|0.404346|0.404116|0.405729|0.402389|0.403822|0.404080 ± 0.001196|
|eicu|1|TabM_mini_PLE|0.392446|0.392415|0.390288|0.393642|0.398193|0.393397 ± 0.002941|

All paired differences, nominal/corrected conditional intervals and p values are in verified-results.json. Original three-seed analyses remain unchanged.
