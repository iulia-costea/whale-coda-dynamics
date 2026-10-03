# Initial Caribbean transfer

This is the first EC1 ↔ EC2 comparison. `fold_scores.csv` contains the individual source-fold scores; `scores.csv` contains paired per-coda own and averaged-transfer scores; `summary.csv` contains the aggregate result.

| Direction | Own bits/symbol | Transfer bits/symbol | Loss bits/symbol |
|---|---:|---:|---:|
| EC1 → EC2 | -1.551977 | -3.313055 | 1.761078 |
| EC2 → EC1 | -1.658672 | -4.019187 | 2.360515 |

Both directions have positive loss: each Caribbean clan predicts its own codas better than the other clan does.
