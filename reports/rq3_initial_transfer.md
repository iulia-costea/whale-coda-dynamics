# RQ3 initial Caribbean transfer

Each target coda has a cross-fitted own-clan score and an averaged transfer score from every source-fold model.
The paired transfer loss is own-clan score minus transfer score; positive values mean the source model predicts the target worse.

| Source | Target | Target codas | Source models | Own bits/symbol | Transfer bits/symbol | Loss bits/symbol |
|---|---|---:|---:|---:|---:|---:|
| EC1 | EC2 | 946 | 5 | -1.630095 | -3.332531 | 1.702436 |
| EC2 | EC1 | 7742 | 6 | -1.658672 | -3.979414 | 2.320742 |

These are initial EC1/EC2 results only. They do not yet include bootstrap intervals, coverage strata, or the full 9 x 9 matrix.
