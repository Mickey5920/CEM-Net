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
|mimic|dev|0|CEM-Net|0.543911|0.002768|
|mimic|dev|1|CEM-Net|0.556012|0.002250|
|mimic|dev|0|Medformer|0.458469|0.004288|
|mimic|dev|1|Medformer|0.496407|0.001727|
|mimic|test|0|CEM-Net|0.543926|0.001987|
|mimic|test|1|CEM-Net|0.541242|0.000696|
|mimic|test|0|Medformer|0.465313|0.001831|
|mimic|test|1|Medformer|0.478426|0.005384|
|mimiciii|dev|0|CEM-Net|0.485815|0.005309|
|mimiciii|dev|1|CEM-Net|0.532424|0.002287|
|mimiciii|dev|0|Medformer|0.388621|0.003645|
|mimiciii|dev|1|Medformer|0.512582|0.003239|
|mimiciii|test|0|CEM-Net|0.485463|0.005097|
|mimiciii|test|1|CEM-Net|0.549924|0.001966|
|mimiciii|test|0|Medformer|0.425773|0.004632|
|mimiciii|test|1|Medformer|0.516279|0.003297|
