# Own-clan calibration

`self_calibration.csv` evaluates all candidate models on each clan's own held-out folds. `self_scores.csv` contains one cross-fitted own-clan score per eligible coda. These own scores are the reference used in every later transfer-loss calculation.

## Selected models

| Clan | Model | Context order | Alpha | Held-out bits/symbol |
|---|---|---:|---:|---:|
| EC1 | ngram | 4 | 0.1 | -1.661529 |
| EC2 | ngram | 4 | 0.1 | -1.635778 |
| FP | ngram | 2 | 0.1 | -2.928978 |
| PALI | ngram | 3 | 0.1 | -2.095520 |
| PO | ngram | 3 | 0.1 | -1.840511 |
| REG | ngram | 4 | 0.1 | -1.446075 |
| RI | ngram | 2 | 0.1 | -2.328553 |
| SH | ngram | 2 | 0.1 | -2.291833 |
| SI | ngram | 2 | 0.1 | -1.863796 |
