# Fixed five-seed comparator results

Five paired fixed fits; 20000 common patient-cluster draws. The original 30-comparison allowance is retained separately by split, including all pending FT and deferred TVNet contrasts. Bonferroni inference is valid before FT finishes; full-family Holm is computed only after all four active methods are complete. Centered Bootstrap p is approximate, never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Intervals condition on fitted models and do not cover training-population or historical selection uncertainty. No new calibration or reselection of prior secondary baselines.

|Database|Split|Endpoint|Method|AP mean|Sample SD|
|---|---|---|---|---:|---:|
|mimiciii|dev|0|CEM-Net|0.485815|0.005309|
|mimiciii|dev|1|CEM-Net|0.532424|0.002287|
|mimiciii|dev|0|LITEMV|0.401916|0.006522|
|mimiciii|dev|1|LITEMV|0.489660|0.006998|
|mimiciii|dev|0|ModernTCN|0.324580|0.008893|
|mimiciii|dev|1|ModernTCN|0.473224|0.006452|
|mimiciii|dev|0|TCN|0.379724|0.011476|
|mimiciii|dev|1|TCN|0.491230|0.006560|
|mimiciii|test|0|CEM-Net|0.485463|0.005097|
|mimiciii|test|1|CEM-Net|0.549924|0.001966|
|mimiciii|test|0|LITEMV|0.416984|0.011133|
|mimiciii|test|1|LITEMV|0.500975|0.014797|
|mimiciii|test|0|ModernTCN|0.361039|0.014671|
|mimiciii|test|1|ModernTCN|0.477358|0.011962|
|mimiciii|test|0|TCN|0.409518|0.005935|
|mimiciii|test|1|TCN|0.495263|0.011909|
