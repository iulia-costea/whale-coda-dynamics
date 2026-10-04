# RQ3 self-model calibration

Each coda is scored by its own clan's model excluding that coda's deterministic recording-group fold.
Model settings are selected using the same clan's held-out folds only.

## Selected models

| Dataset | Clan | Source codas | Model | Context order | Alpha | Held-out codas | Mean bits/coda | Mean bits/symbol |
|---|---|---:|---|---:|---:|---:|---:|---:|
| caribbean | EC1 | 7742 | ngram | 4 | 0.1 | 7742 | -8.695923 | -1.658672 |
| caribbean | EC2 | 946 | ngram | 4 | 0.1 | 946 | -8.822502 | -1.630095 |
| pacific | FP | 2556 | ngram | 2 | 0.1 | 2556 | -14.643846 | -2.871254 |
| pacific | PALI | 2052 | ngram | 3 | 0.1 | 2052 | -11.692300 | -2.004227 |
| pacific | PO | 1985 | ngram | 3 | 0.1 | 1985 | -11.175805 | -1.848356 |
| pacific | REG | 8107 | ngram | 4 | 0.1 | 8107 | -8.681947 | -1.443874 |
| pacific | RI | 1747 | ngram | 2 | 0.1 | 1747 | -11.737846 | -2.221911 |
| pacific | SH | 5138 | ngram | 2 | 0.1 | 5138 | -10.020354 | -2.355519 |
| pacific | SI | 1206 | ngram | 2 | 0.1 | 1206 | -8.786183 | -1.956451 |

## Score coverage

| Dataset | Clan | Cross-fitted codas |
|---|---|---:|
| caribbean | EC1 | 7742 |
| caribbean | EC2 | 946 |
| pacific | FP | 2556 |
| pacific | PALI | 2052 |
| pacific | PO | 1985 |
| pacific | REG | 8107 |
| pacific | RI | 1747 |
| pacific | SH | 5138 |
| pacific | SI | 1206 |

The complete candidate-fold table is in `data/results/rq3/calibration/self_calibration.csv`; coda-level own scores are in `data/results/rq3/calibration/self_scores.csv`.
