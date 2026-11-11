# Fixed five-seed comparator results

Five paired fixed fits; 20000 common patient-cluster draws. Six fixed CEM-Net versus Medformer comparisons per split form an independent family; all previous families remain unchanged. Centered Bootstrap p is approximate, never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Intervals condition on fitted models and do not cover training-population or historical selection uncertainty. No new calibration or reselection of prior secondary baselines.

|Database|Split|Endpoint|Method|AP mean|Sample SD|
|---|---|---|---|---:|---:|
|eicu|dev|0|CEM-Net|0.433439|0.003120|
|eicu|dev|1|CEM-Net|0.400320|0.002033|
|eicu|dev|0|Medformer|0.336701|0.004254|
|eicu|dev|1|Medformer|0.342153|0.002212|
|eicu|test|0|CEM-Net|0.392156|0.006653|
|eicu|test|1|CEM-Net|0.393236|0.001788|
|eicu|test|0|Medformer|0.298942|0.002185|
|eicu|test|1|Medformer|0.335424|0.002801|
