# Fixed five-seed comparator results

Five paired fixed fits; 20000 common patient-cluster draws. Six fixed CEM-Net versus Medformer comparisons per split form an independent family; all previous families remain unchanged. Centered Bootstrap p is approximate, never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Intervals condition on fitted models and do not cover training-population or historical selection uncertainty. No new calibration or reselection of prior secondary baselines.

|Database|Split|Endpoint|Method|AP mean|Sample SD|
|---|---|---|---|---:|---:|
|mimic|dev|0|CEM-Net|0.543911|0.002768|
|mimic|dev|1|CEM-Net|0.556012|0.002250|
|mimic|dev|0|Medformer|0.458469|0.004288|
|mimic|dev|1|Medformer|0.496407|0.001727|
|mimic|test|0|CEM-Net|0.543926|0.001987|
|mimic|test|1|CEM-Net|0.541242|0.000696|
|mimic|test|0|Medformer|0.465313|0.001831|
|mimic|test|1|Medformer|0.478426|0.005384|
