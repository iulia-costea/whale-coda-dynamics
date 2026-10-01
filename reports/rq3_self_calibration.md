# RQ3 self-model calibration

Each coda is scored by its own clan's model excluding that coda's deterministic recording-group fold.
Model settings are selected using the same clan's held-out folds only.

## Selected models

| Dataset | Clan | Model | Context order | Alpha | Held-out codas | Mean bits/coda | Mean bits/symbol |
|---|---|---|---:|---:|---:|---:|---:|
| caribbean | EC1 | ngram | 4 | 0.1 | 7742 | -8.699694 | -1.661529 |
| caribbean | EC2 | ngram | 4 | 0.1 | 946 | -8.843677 | -1.635778 |
| pacific | FP | ngram | 2 | 0.1 | 2556 | -14.725189 | -2.928978 |
| pacific | PALI | ngram | 3 | 0.1 | 2052 | -11.692300 | -2.095520 |
| pacific | PO | ngram | 3 | 0.1 | 1985 | -11.143864 | -1.840511 |
| pacific | REG | ngram | 4 | 0.1 | 8107 | -8.727398 | -1.446075 |
| pacific | RI | ngram | 2 | 0.1 | 1747 | -11.725358 | -2.328553 |
| pacific | SH | ngram | 2 | 0.1 | 5138 | -10.035126 | -2.291833 |
| pacific | SI | ngram | 2 | 0.1 | 1206 | -8.753158 | -1.863796 |

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

The complete candidate-fold table is in `rq3_self_calibration.csv`; coda-level own scores are in `rq3_self_scores.csv`.
