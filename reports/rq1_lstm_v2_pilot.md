# RQ1 LSTM v2 pilot

Device: cpu. Validation selection uses seeds [42]. Test rows do not enter hyperparameter or epoch selection.

| Fold | Condition | Model | LL bits/coda | Accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|
| test-2016-04-24 | A | lstm | -1.38685 | 68.750% | 0.1913 |
| test-2016-04-24 | A | logistic | -1.68434 | 75.000% | 0.3939 |
| test-2016-04-24 | A | repeat_own_last | -2.49738 | 75.000% | 0.3939 |
| test-2016-04-24 | A_B_S | lstm | -1.56005 | 68.750% | 0.1913 |
| test-2016-04-24 | A_B_S | logistic | -1.58036 | 81.250% | 0.4058 |
| test-2016-04-24 | A_B_S | repeat_own_last | -2.49738 | 75.000% | 0.3939 |
| test-2015-03-02 | A | lstm | -1.49549 | 79.944% | 0.2750 |
| test-2015-03-02 | A | logistic | -1.37308 | 79.661% | 0.2747 |
| test-2015-03-02 | A | repeat_own_last | -1.70297 | 76.836% | 0.2977 |
| test-2015-03-02 | A_B_S | lstm | -1.97076 | 63.842% | 0.2260 |
| test-2015-03-02 | A_B_S | logistic | -1.58008 | 74.011% | 0.2564 |
| test-2015-03-02 | A_B_S | repeat_own_last | -1.70297 | 76.836% | 0.2977 |
| test-2015-03-31 | A | lstm | -0.74333 | 89.716% | 0.3233 |
| test-2015-03-31 | A | logistic | -0.72171 | 90.071% | 0.3355 |
| test-2015-03-31 | A | repeat_own_last | -0.92706 | 89.362% | 0.3475 |
| test-2015-03-31 | A_B_S | lstm | -0.78100 | 90.071% | 0.3262 |
| test-2015-03-31 | A_B_S | logistic | -0.72173 | 90.071% | 0.3695 |
| test-2015-03-31 | A_B_S | repeat_own_last | -0.92706 | 89.362% | 0.3475 |
| test-2015-03-26 | A | lstm | -0.84892 | 87.109% | 0.2449 |
| test-2015-03-26 | A | logistic | -0.80810 | 87.891% | 0.3347 |
| test-2015-03-26 | A | repeat_own_last | -1.02482 | 87.891% | 0.4630 |
| test-2015-03-26 | A_B_S | lstm | -0.87415 | 86.719% | 0.2524 |
| test-2015-03-26 | A_B_S | logistic | -0.80991 | 88.672% | 0.3801 |
| test-2015-03-26 | A_B_S | repeat_own_last | -1.02482 | 87.891% | 0.4630 |

## Selected fits

| Fold | Condition | Seed | Setting | Beta | Gamma | Best epoch | Epochs | Hit cap |
|---|---|---:|---|---:|---:|---:|---:|---|
| test-2016-04-24 | A | 42 | {'hidden_dim': 64, 'dropout': 0.4, 'weight_decay': 0.001} | 0.44823208451271057 | None | 42 | 62 | False |
| test-2016-04-24 | A_B_S | 42 | {'hidden_dim': 32, 'dropout': 0.4, 'weight_decay': 0.0001, 's_weight_decay': 0.001} | 0.7076165676116943 | 0.24189439415931702 | 102 | 122 | False |
| test-2015-03-02 | A | 42 | {'hidden_dim': 32, 'dropout': 0.4, 'weight_decay': 1e-05} | 0.4170508086681366 | None | 65 | 85 | False |
| test-2015-03-02 | A_B_S | 42 | {'hidden_dim': 32, 'dropout': 0.4, 'weight_decay': 1e-05, 's_weight_decay': 0.001} | 0.27653536200523376 | 0.11781271547079086 | 43 | 63 | False |
| test-2015-03-31 | A | 42 | {'hidden_dim': 64, 'dropout': 0.4, 'weight_decay': 1e-05} | 0.18925727903842926 | None | 22 | 42 | False |
| test-2015-03-31 | A_B_S | 42 | {'hidden_dim': 32, 'dropout': 0.4, 'weight_decay': 0.0001, 's_weight_decay': 0.1} | 0.21581287682056427 | 0.10987024754285812 | 22 | 42 | False |
| test-2015-03-26 | A | 42 | {'hidden_dim': 64, 'dropout': 0, 'weight_decay': 0.001} | 0.16655293107032776 | None | 15 | 35 | False |
| test-2015-03-26 | A_B_S | 42 | {'hidden_dim': 32, 'dropout': 0.4, 'weight_decay': 0.001, 's_weight_decay': 0.1} | 0.4645635187625885 | 0.2222248613834381 | 48 | 68 | False |

## Example learning curve

test-2016-04-24, A, seed 42.

| Epoch | Training LL | Validation LL |
|---:|---:|---:|
| 1 | -1.76244 | -4.41225 |
| 20 | -0.83325 | -4.45473 |
| 40 | -0.75592 | -4.18475 |
| 42 | -0.75469 | -4.09010 |
| 60 | -0.71266 | -4.14311 |
| 62 | -0.70917 | -4.20907 |

## CPU timing

| Seed | Seconds | Validation LL |
|---:|---:|---:|
| 42 | 4.09 | -4.09010 |
| 43 | 1.46 | -4.19170 |
| 44 | 3.68 | -3.81733 |

Full search: 17388 fits; estimated 2.48 CPU wall hours with 6 concurrent fits. Ideal parallel scaling from one fold/condition; excludes worker startup and variation across settings.

CUDA available: False.

## Gate

PASS

0 of 432 candidate fits hit the epoch cap. The selected-setting cap check is reported above.
