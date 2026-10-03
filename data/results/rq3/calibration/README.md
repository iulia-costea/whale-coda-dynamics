# Own-clan calibration

`self_calibration.csv` evaluates all candidate models on each clan's own held-out folds. `self_scores.csv` contains one cross-fitted own-clan score per eligible coda. These own scores are the reference used in every later transfer-loss calculation.

## Selected models

| Clan | Model | Context order | Alpha | Held-out bits/symbol |
|---|---|---:|---:|---:|
| EC1 | ngram | 4 | 0.1 | -1.658672 |
| EC2 | ngram | 4 | 0.1 | -1.630095 |
| FP | ngram | 2 | 0.1 | -2.871254 |
| PALI | ngram | 3 | 0.1 | -2.004227 |
| PO | ngram | 3 | 0.1 | -1.848356 |
| REG | ngram | 4 | 0.1 | -1.443874 |
| RI | ngram | 2 | 0.1 | -2.221911 |
| SH | ngram | 2 | 0.1 | -2.355519 |
| SI | ngram | 2 | 0.1 | -1.956451 |
