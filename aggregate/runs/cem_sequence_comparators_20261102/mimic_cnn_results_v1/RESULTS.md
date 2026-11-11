# Fixed five-seed comparator results

Five paired fixed fits; 20000 common patient-cluster draws. The original 30-comparison allowance is retained separately by split, including all pending FT and deferred TVNet contrasts. Bonferroni inference is valid before FT finishes; full-family Holm is computed only after all four active methods are complete. Centered Bootstrap p is approximate, never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Intervals condition on fitted models and do not cover training-population or historical selection uncertainty. No new calibration or reselection of prior secondary baselines.

|Database|Split|Endpoint|Method|AP mean|Sample SD|
|---|---|---|---|---:|---:|
|mimic|dev|0|CEM-Net|0.543911|0.002768|
|mimic|dev|1|CEM-Net|0.556012|0.002250|
|mimic|dev|0|LITEMV|0.476798|0.014591|
|mimic|dev|1|LITEMV|0.504850|0.007363|
|mimic|dev|0|ModernTCN|0.424536|0.015080|
|mimic|dev|1|ModernTCN|0.483475|0.010003|
|mimic|dev|0|TCN|0.441788|0.007575|
|mimic|dev|1|TCN|0.494810|0.004746|
|mimic|test|0|CEM-Net|0.543926|0.001987|
|mimic|test|1|CEM-Net|0.541242|0.000696|
|mimic|test|0|LITEMV|0.462216|0.005291|
|mimic|test|1|LITEMV|0.483570|0.008616|
|mimic|test|0|ModernTCN|0.422406|0.018410|
|mimic|test|1|ModernTCN|0.452193|0.013639|
|mimic|test|0|TCN|0.450755|0.008366|
|mimic|test|1|TCN|0.474128|0.002869|
