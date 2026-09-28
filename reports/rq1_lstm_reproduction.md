# RQ1 partner-only Assayag reproduction check

Condition B (partner history only) refit with the architecture and hyperparameters in Assayag et al. 2026 (Bioacoustics 35(4)), Table 1 and Figure 4: embedding 32 (not stated in the paper for this task), two stacked LSTMs with 2 layers each at hidden 64 then 32, dropout 0.05, Adam learning rate 0.001, batch 16. Run on our fourteen leave-one-day-out folds, not their single random 70/15/15 split. This is a standalone check, not part of the primary eight-condition lattice in reports/rq1_lstm.md.

The target is Table 3's LSTM "Paired Coda Type (Accuracy)" row: predicting the second whale's next coda type from the first whale's preceding codas, which is exactly this condition. The paper reports one accuracy per preceding-coda count (0 through 8), not a single figure, so the comparison below is against that range (0.373 to 0.766). codaspred.ipynb's own LSTMModel_CODA cell uses different hyperparameters (hidden 128, embedding 32, dropout 0.15, lr 0.01, batch 4) than this published table; this check follows the paper.

## Result

| Metric | Value |
|---|---:|
| LL bits/coda | -2.60616 |
| Accuracy | 59.211% |
| Macro-F1 | 0.0610 |
| repeat-own-last accuracy (for scale) | 82.547% |
| majority-class accuracy (for scale) | 57.207% |

## Target (Assayag et al. 2026, Table 3, LSTM, Paired Coda Type)

| Preceding codas | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.593 | 0.766 | 0.626 | 0.633 | 0.606 | 0.706 | 0.373 | 0.720 | 0.766 |

Our accuracy (59.211%) uses up to 10 preceding partner codas per row (whatever each row actually has), not a fixed count, so it is not paired one to one with a single column above; it is reported here for comparison against the range as a whole, not interpreted further.

42 fits (14 folds times 3 seeds); epochs trained ranged 11 to 70, 42 stopped early on patience. Source, fold-manifest and configuration SHA-256 hashes are recorded in evaluation/summary.json.
