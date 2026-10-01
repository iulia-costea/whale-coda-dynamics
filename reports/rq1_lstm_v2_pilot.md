# RQ1 LSTM v2 pilot (DEC-17)

Conditions ['A', 'A_B_S'] on test days ['2015-03-02', '2015-03-31', '2015-03-26', '2016-04-24'], seed(s) [42], full grid: dropout [0.0, 0.2, 0.4] x hidden [32, 64] x history weight decay [1e-05, 0.0001, 0.001], and where S is present, S weight decay [1e-05, 0.001, 1.0]. Same architecture and optimiser as the eventual full run (Adam, lr 0.001, batch 32, up to 300 epochs, patience 20). Device: cpu. This checks the mechanism works before spending the full study's compute; it is not evaluated against repeat-own-last or logistic regression as a final result, only as a gate.

## Metrics

| Condition | LL bits/coda | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| A | -1.06646 | 82.819% | 0.2178 |
| A_B_S | -1.23723 | 80.837% | 0.2066 |
| majority | n/a | 64.427% | 0.0412 |
| repeat_own_last | -1.28480 | 83.811% | 0.3812 |

## vs repeat-own-last, over the pilot folds only

| Condition | LL contrast | CI | Positive days | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|---:|
| A | 0.21833 | [0.1951, 0.3340] | 4/4 | 82.819% | 0.2178 |
| A_B_S | 0.04757 | [-0.0703, 0.2325] | 3/4 | 80.837% | 0.2066 |

## LSTM v2 pilot vs logistic regression, by condition [U1]

| Condition | Logistic LL bits/coda | LSTM v2 LL bits/coda | Logistic Accuracy | LSTM v2 Accuracy |
|---|---:|---:|---:|---:|
| A | -1.09690 | -1.06646 | 82.353% | 82.819% |
| A_B_S | -1.13261 | -1.23723 | 82.288% | 80.837% |

## Chosen setting and fitted beta/gamma per condition

| Condition | Dropout | Hidden | WD history | WD S | Beta range | Gamma range | Epochs trained | Hit epoch cap |
|---|---:|---:|---:|---:|---|---|---|---|
| A | [0.2, 0.4] | [32, 64] | [1e-05, 0.0001, 0.001] | n/a | 0.1883 to 0.4617 | n/a | 37-73 | 0/4 |
| A_B_S | [0.0, 0.4] | [32] | [1e-05, 0.001] | [1e-05, 0.001, 1.0] | 0.2253 to 1.2500 | 0.0930 to 0.5767 | 48-171 | 0/4 |

## Learning curve, one fold/condition/setting (train vs validation log-likelihood)

Fold test-2016-04-24, condition A, seed 42, the chosen setting (dropout 0.4, hidden 64, weight decay history 0.001, weight decay S None):

| Epoch | Train LL bits/coda | Validation LL bits/coda |
|---:|---:|---:|
| 1 | -2.80366 | -4.42583 |
| 4 | -1.14713 | -4.68634 |
| 7 | -1.00200 | -4.77521 |
| 10 | -0.97609 | -4.68318 |
| 13 | -0.91223 | -4.62300 |
| 16 | -0.89479 | -4.56758 |
| 19 | -0.85892 | -4.53261 |
| 22 | -0.87575 | -4.36194 |
| 25 | -0.85613 | -4.31678 |
| 28 | -0.88305 | -4.35353 |
| 31 | -0.83476 | -4.23734 |
| 34 | -0.84187 | -4.17383 |
| 37 | -0.82413 | -4.16762 |
| 40 | -0.78735 | -4.15409 |
| 43 | -0.79188 | -4.13340 |
| 46 | -0.78562 | -4.11414 |
| 49 | -0.78957 | -4.14456 |
| 52 | -0.78204 | -4.16713 |
| 55 | -0.76499 | -4.10042 |
| 58 | -0.76576 | -4.16605 |
| 61 | -0.75982 | -4.12439 |
| 64 | -0.75438 | -4.10346 |

## Gate 2: does not hold

The following stop conditions were met. No interpretation or conclusions are written below; the tables above are the complete result. Per the plan, settings are not retuned in response to this: the pilot folds are test days.

- A_B_S: LSTM reliably below logistic regression (CI [-0.2275, -0.0669])

Per-epoch training and validation log-likelihood for every grid-search fit (not only the chosen setting) are in training_curves.jsonl; the full grid is in lstm_selection.jsonl. Model weights are not saved. Source, fold-manifest and configuration SHA-256 hashes are recorded in evaluation/summary.json.
