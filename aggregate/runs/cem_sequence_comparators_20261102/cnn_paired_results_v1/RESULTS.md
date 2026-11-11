# Fixed five-seed comparator results

Five paired fixed fits; 20000 common patient-cluster draws. The original 30-comparison allowance is retained separately by split, including all pending FT and deferred TVNet contrasts. Bonferroni inference is valid before FT finishes; full-family Holm is computed only after all four active methods are complete. Centered Bootstrap p is approximate, never zero; exact five-seed signed-rank is supplemental (minimum two-sided p=.0625). Intervals condition on fitted models and do not cover training-population or historical selection uncertainty. No new calibration or reselection of prior secondary baselines.

|Database|Split|Endpoint|Method|AP mean|Sample SD|
|---|---|---|---|---:|---:|
|eicu|dev|0|CEM-Net|0.433439|0.003120|
|eicu|dev|1|CEM-Net|0.400320|0.002033|
|eicu|dev|0|LITEMV|0.349374|0.009514|
|eicu|dev|1|LITEMV|0.348894|0.003763|
|eicu|dev|0|ModernTCN|0.314584|0.013492|
|eicu|dev|1|ModernTCN|0.332822|0.006819|
|eicu|dev|0|TCN|0.335801|0.003481|
|eicu|dev|1|TCN|0.337493|0.003612|
|eicu|test|0|CEM-Net|0.392156|0.006653|
|eicu|test|1|CEM-Net|0.393236|0.001788|
|eicu|test|0|LITEMV|0.318204|0.006553|
|eicu|test|1|LITEMV|0.343611|0.002846|
|eicu|test|0|ModernTCN|0.282542|0.011816|
|eicu|test|1|ModernTCN|0.319576|0.005720|
|eicu|test|0|TCN|0.303529|0.007131|
|eicu|test|1|TCN|0.334693|0.003449|
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
