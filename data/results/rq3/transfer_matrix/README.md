# Primary 9 × 9 transfer matrix

`fold_scores.csv` contains all source-fold predictions for the 72 off-diagonal directions. `scores.csv` contains paired per-coda results for all 81 cells, including zero-loss own-clan diagonal references. `summary.csv` contains the aggregate matrix.

All 72 cross-clan directions have positive loss, from 0.317484 to 2.815491 bits/symbol.

## Derived analyses

- [coverage/](coverage/README.md): whether the source-fold training data contains the complete target timing pattern.
- [uncertainty/](uncertainty/README.md): 1,000 target-group bootstrap draws and 95% intervals.
- [identity/](identity/README.md): Pacific target-coda identity/non-identity breakdown.
- [pacific_overlap/](pacific_overlap/README.md): exploratory Table S4 spatial-overlap association.
