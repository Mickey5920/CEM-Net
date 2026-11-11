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
