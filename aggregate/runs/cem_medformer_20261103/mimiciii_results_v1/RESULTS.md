# Fixed five-seed comparator results

Five paired fixed fits; 20000 common patient-cluster draws. Six fixed CEM-Net versus Medformer comparisons per split form an independent family; all previous families remain unchanged. Centered Bootstrap p is approximate, never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Intervals condition on fitted models and do not cover training-population or historical selection uncertainty. No new calibration or reselection of prior secondary baselines.

|Database|Split|Endpoint|Method|AP mean|Sample SD|
|---|---|---|---|---:|---:|
|mimiciii|dev|0|CEM-Net|0.485815|0.005309|
|mimiciii|dev|1|CEM-Net|0.532424|0.002287|
|mimiciii|dev|0|Medformer|0.388621|0.003645|
|mimiciii|dev|1|Medformer|0.512582|0.003239|
|mimiciii|test|0|CEM-Net|0.485463|0.005097|
|mimiciii|test|1|CEM-Net|0.549924|0.001966|
|mimiciii|test|0|Medformer|0.425773|0.004632|
|mimiciii|test|1|Medformer|0.516279|0.003297|
