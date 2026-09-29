# LSTM v2 validation-only S suppression check

Fold: test-2015-03-02; device: cpu; setting: {'hidden_dim': 32, 'dropout': 0.0, 'weight_decay': 0.0001, 's_weight_decay': 10.0}.

| Condition | Seed | Validation bits/coda | Best epoch | Epochs | Seconds |
|---|---:|---:|---:|---:|---:|
| A_B | 42 | -0.658949 | 23 | 43 | 2.59 |
| A_B | 43 | -0.626203 | 26 | 46 | 2.68 |
| A_B | 44 | -0.649734 | 23 | 43 | 2.51 |
| A_B_S | 42 | -0.657995 | 23 | 43 | 2.62 |
| A_B_S | 43 | -0.620328 | 28 | 48 | 2.96 |
| A_B_S | 44 | -0.650796 | 23 | 43 | 2.63 |

Mean A_B_S minus A_B: 0.001922 bits/coda; predeclared absolute tolerance: 0.01. Check: PASS.

This diagnostic uses training and validation rows only. Test outcomes do not select or alter the grid.
