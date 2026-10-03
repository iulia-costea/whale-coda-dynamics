# Initial Caribbean transfer

This is the first EC1 ↔ EC2 comparison. `fold_scores.csv` contains the individual source-fold scores; `scores.csv` contains paired per-coda own and averaged-transfer scores; `summary.csv` contains the aggregate result.

| Direction | Own bits/symbol | Transfer bits/symbol | Loss bits/symbol |
|---|---:|---:|---:|
| EC1 → EC2 | -1.630095 | -3.332531 | 1.702436 |
| EC2 → EC1 | -1.658672 | -3.979414 | 2.320742 |

Both directions have positive loss: each Caribbean clan predicts its own codas better than the other clan does.
