# RQ1 LSTM v2 validation-only S-suppression diagnostic (DEC-20)

Fold 2015-03-02, without-S fields ['A_history', 'B_history'], with-S fields ['A_history', 'B_history', 'turn_category', 'gap_s', 'overlap_elapsed_s'], S weight decay fixed at 1.0 (the top of the grid actually used), history weight decay 0.0001, dropout 0.2, hidden 32, seeds [42, 43, 44], tolerance 0.01 bits/coda. If the with-S validation log-likelihood is not close to the without-S one even at this penalty, the S encoder cannot actually be switched off in practice, and DEC-20's fix would not generalise from logistic regression to the LSTM.

| Seed | Without-S validation LL | With-S validation LL | Difference |
|---:|---:|---:|---:|
| 42 | -0.64529 | -0.63888 | 0.00641 |
| 43 | -0.61592 | -0.61890 | -0.00298 |
| 44 | -0.63835 | -0.62779 | 0.01056 |

Mean difference: 0.00466 bits/coda. Passed: within the 0.01 bits/coda tolerance.
