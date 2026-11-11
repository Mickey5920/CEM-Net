# Five-seed Table VI comparison

Five paired fixed fits; common patient-cluster resamples. Original 54-comparison Bonferroni allowance retained for nine fixed Table VI contrasts, separately per split. Null-centered Bootstrap p is approximate and never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Historical Test exposure and training-population/selection uncertainty remain outside these conditional intervals.

## dev

|Database|Endpoint|Method|48|49|50|51|52|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|---:|---:|
|eicu|0|CEM-Net|0.434527|0.431635|0.430567|0.438381|0.432087|0.433439 ± 0.003120|
|eicu|1|CEM-Net|0.401118|0.397699|0.398755|0.402662|0.401367|0.400320 ± 0.002033|
|eicu|1|TabM_PLE|0.401285|0.399827|0.399644|0.401364|0.398912|0.400206 ± 0.001077|
|eicu|0|TabM_mini_PLE|0.439761|0.432539|0.435682|0.434413|0.435658|0.435611 ± 0.002651|
|eicu|1|TabM_mini_PLE|0.400370|0.401897|0.398581|0.401905|0.400145|0.400580 ± 0.001389|
## test

|Database|Endpoint|Method|48|49|50|51|52|Mean ± sample SD|
|---|---|---|---:|---:|---:|---:|---:|---:|
|eicu|0|CEM-Net|0.393539|0.397945|0.398310|0.382758|0.388228|0.392156 ± 0.006653|
|eicu|1|CEM-Net|0.392219|0.393073|0.394441|0.390958|0.395489|0.393236 ± 0.001788|
|eicu|1|TabM_PLE|0.393863|0.394373|0.392126|0.396614|0.397145|0.394824 ± 0.002061|
|eicu|0|TabM_mini_PLE|0.404346|0.404116|0.405729|0.402389|0.403822|0.404080 ± 0.001196|
|eicu|1|TabM_mini_PLE|0.392446|0.392415|0.390288|0.393642|0.398193|0.393397 ± 0.002941|

All paired differences, nominal/corrected conditional intervals and p values are in verified-results.json. Original three-seed analyses remain unchanged.
