# Initial Caribbean transfer

This is the first EC1 ↔ EC2 comparison. `fold_scores.csv` contains the individual source-fold scores; `scores.csv` contains paired per-coda own and averaged-transfer scores; `summary.csv` contains the aggregate result.

| Direction | Own bits/symbol | Transfer bits/symbol | Loss bits/symbol |
|---|---:|---:|---:|
| EC1 → EC2 | -1.635778 | -3.375786 | 1.740008 |
| EC2 → EC1 | -1.661529 | -4.010152 | 2.348623 |

Both directions have positive loss: each Caribbean clan predicts its own codas better than the other clan does.
