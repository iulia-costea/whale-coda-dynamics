# RQ1 LSTM main study

One LSTM encoder per enabled history stream (embedding 32, hidden 64) and a feed-forward encoder for S (hidden 64), concatenated into a per-condition output layer; an absent input has no encoder at all. Adam, learning rate 0.0001, weight decay 1e-05, batch 32, early stopping on validation-day log-likelihood with patience 5 and a cap of 100 epochs. Identical settings across all conditions and folds; only which encoders exist differs. Seeds [42, 43, 44] are averaged per coda before contrasts, by the shared evaluation module.

## Metrics

| Condition | LL bits/coda | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| none | -2.60254 | 57.207% | 0.0364 |
| S | -2.61916 | 56.992% | 0.0383 |
| A | -1.21333 | 81.297% | 0.2154 |
| B | -2.61448 | 59.901% | 0.0664 |
| A_B | -1.31962 | 78.711% | 0.1832 |
| A_S | -1.26959 | 79.918% | 0.1912 |
| B_S | -2.63969 | 58.608% | 0.0661 |
| A_B_S | -1.37277 | 77.677% | 0.1698 |
| majority | n/a | 57.207% | 0.0364 |
| repeat_own_last | -1.39276 | 82.547% | 0.4070 |

## A-conditions vs repeat-own-last

| Condition | LL contrast | CI | Positive days | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|---:|
| A | 0.17943 | [0.11445718740109259, 0.2660857053843938] | 13/14 | 81.297% | 0.2154 |
| A_B | 0.07313 | [-0.04207820803227102, 0.2197934389371159] | 11/14 | 78.711% | 0.1832 |
| A_S | 0.12317 | [0.00746594272468818, 0.22548161942764935] | 11/14 | 79.918% | 0.1912 |
| A_B_S | 0.01999 | [-0.11372557291542838, 0.18445177109416436] | 9/14 | 77.677% | 0.1698 |

## All twelve add-one-input contrasts

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | -0.01662 | 0.05181 | [-0.17901, 0.22160] | 7/14 |
| adding A to no inputs | 1.38921 | 1.46987 | [1.12752, 1.75616] | 14/14 |
| adding B to no inputs | -0.01194 | 0.20272 | [-0.46197, 0.60149] | 8/14 |
| adding B to A | -0.10629 | -0.07172 | [-0.19643, -0.02636] | 5/14 |
| adding A to B | 1.29486 | 1.19543 | [0.88662, 1.64292] | 13/14 |
| adding A to S | 1.34957 | 1.31140 | [1.12099, 1.63822] | 14/14 |
| adding S to A | -0.05626 | -0.10666 | [-0.12144, -0.02409] | 4/14 |
| adding B to S | -0.02054 | 0.12073 | [-0.51050, 0.53008] | 9/14 |
| adding S to B | -0.02521 | -0.03018 | [-0.08635, 0.03413] | 5/14 |
| adding S to A and B | -0.05314 | -0.12447 | [-0.10249, -0.01460] | 6/14 |
| adding B to A and S | -0.10318 | -0.08952 | [-0.20040, 0.00434] | 6/14 |
| adding A to B and S | 1.26692 | 1.10114 | [0.83625, 1.65158] | 13/14 |

## LSTM vs logistic regression (v2), by condition [U1]

| Condition | Logistic LL bits/coda | LSTM LL bits/coda | Logistic Accuracy | LSTM Accuracy |
|---|---:|---:|---:|---:|
| S | -2.51416 | -2.61916 | 56.949% | 56.992% |
| A | -1.08791 | -1.21333 | 82.741% | 81.297% |
| B | -2.46488 | -2.61448 | 59.147% | 59.901% |
| A_B | -1.09819 | -1.31962 | 82.353% | 78.711% |
| A_S | -1.11937 | -1.26959 | 82.999% | 79.918% |
| B_S | -2.50527 | -2.63969 | 57.725% | 58.608% |
| A_B_S | -1.14421 | -1.37277 | 82.094% | 77.677% |

294 model fits (14 folds times 7 non-unigram conditions times 3 seeds); epochs trained ranged 6 to 100, 206 stopped early on patience, the rest hit the 100-epoch cap. Full per-run convergence is in convergence.jsonl.

## Stop: results need review before interpretation

The following stop conditions were met. No interpretation or conclusions are written below; the tables above are the complete result.

- A: accuracy 81.297% below repeat-own-last 82.547%
- A_B: accuracy 78.711% below repeat-own-last 82.547%
- A_S: accuracy 79.918% below repeat-own-last 82.547%
- A_B_S: accuracy 77.677% below repeat-own-last 82.547%
- add_input:A_B-A: contrast CI entirely below zero [-0.19642622375704047, -0.026358516784770754]
- add_input:A_S-A: contrast CI entirely below zero [-0.12143584171871069, -0.02408784163682116]
- add_input:A_B_S-A_B: contrast CI entirely below zero [-0.10249438862796066, -0.014601383046887666]
- B: LSTM reliably below logistic regression (CI [-0.22672771730693567, -0.061925037046827475])
- A_S: LSTM reliably below logistic regression (CI [-0.34183671868551163, -0.05176300656964149])
- A: LSTM reliably below logistic regression (CI [-0.22178151034069715, -0.052474014761711096])
- B_S: LSTM reliably below logistic regression (CI [-0.21992134320750464, -0.05085857160677808])
- A_B: LSTM reliably below logistic regression (CI [-0.38786212568245904, -0.0937588405898292])
- A_B_S: LSTM reliably below logistic regression (CI [-0.4139996685479761, -0.10154847993416179])

Statistical uncertainty and restricted training data limit inference from this model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 270 non-repeat codas and is exploratory. Model weights are not saved (parameter count makes this impractical to keep in version control); convergence.jsonl records per-run epoch counts and best validation log-likelihood, sufficient to audit training without re-fitting. Source, fold-manifest and configuration SHA-256 hashes are recorded in evaluation/summary.json.
