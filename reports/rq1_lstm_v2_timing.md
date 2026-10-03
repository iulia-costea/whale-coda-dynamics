# RQ1 LSTM v2 timing and determinism check (DEC-17 stage 2)

Fold 2015-03-02, condition fields ['A_history', 'B_history', 'turn_category', 'gap_s', 'overlap_elapsed_s'], the pilot's chosen setting (dropout 0.4, hidden 32, weight decay history 1e-05, weight decay S 1e-05), seeds [42, 43, 44].

## cpu

| Seed | Seconds | Validation LL bits/coda |
|---:|---:|---:|
| 42 | 15.09 | -0.60529 |
| 43 | 25.80 | -0.59181 |
| 44 | 16.71 | -0.61189 |

## cuda: not available

## Stage 3 wall time estimate

Full study: 14 folds x 8 conditions (3 without S x grid, 4 with S x grid x S grid) x 3 seeds = 11340 fits.

| Device | Seconds/fit | Workers | Estimated hours |
|---|---:|---:|---:|
| cpu | 19.20 | 6 | 10.08 |

Ideal scaling from one fold/condition/setting; excludes worker startup and variation across settings and folds.
