# RQ1 residual LSTM: checks before the full run (DEC-21, PLAN [M4])

Training and validation days only; no residual model is scored on a test day. The full run reads the gates from this run's summary.json.

## a. Wiring (PLAN [M4] checks 1 and 2), all folds and conditions

Per condition, over the 14 folds: the largest difference between the softmax of the recomputed offsets (the network with every block absent) and logistic v4's stored test probabilities (v4 against v4; no residual output, no labels), and how many folds agree bit for bit; the largest difference between the offset-only validation log-likelihood and the one v4 recorded when it chose its penalties; the largest difference between the untrained network's epoch-0 validation log-likelihood (every block on) and v4's recorded one; and whether the untrained network's logits equal the offsets exactly. Logistic v4 was fitted on another machine, so recomputed values can differ from its stored ones in the last bits; wherever the full run uses the offset alone on a test day (logistic v4 chosen, or a seed whose best epoch is 0), it uses v4's stored test probabilities, which these offsets reproduce.

| Condition | Max offset diff (test) | Bitwise folds | Max offset-only validation diff (bits) | Max epoch-0 diff (bits) | Epoch-0 logits = offset | Passed |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.00e+00 | 14/14 | n/a (prior) | n/a (prior) | n/a | 14/14 |
| S | 6.66e-16 | 0/14 | 2.22e-16 | 8.88e-16 | 14/14 | 14/14 |
| A | 5.55e-16 | 2/14 | 0.00e+00 | 8.88e-16 | 14/14 | 14/14 |
| B | 6.66e-16 | 3/14 | 4.44e-16 | 8.88e-16 | 14/14 | 14/14 |
| A_B | 8.88e-16 | 0/14 | 4.44e-16 | 8.88e-16 | 14/14 | 14/14 |
| A_S | 8.88e-16 | 0/14 | 1.11e-16 | 8.88e-16 | 14/14 | 14/14 |
| B_S | 7.77e-16 | 0/14 | 2.22e-16 | 4.44e-16 | 14/14 | 14/14 |
| A_B_S | 1.44e-15 | 0/14 | 2.22e-16 | 3.33e-16 | 14/14 | 14/14 |

Tolerances: offsets 1e-09, epoch 0 1e-12 bits. Gate: passed.

## b. Learnability (PLAN [M4] check 3)

Every block on, hidden 64, dropout 0.0, 300 epochs without early stopping, seeds [42, 43, 44, 45, 46]. Training log-likelihood is measured in eval mode on the full training set after each epoch; epoch 0 is the offset (logistic v4). Pass: the final training log-likelihood is at least 0.1 bits/coda above the offset's in every run.

| Fold | Condition | Seed | Offset train LL | Final train LL | Gain | Offset validation LL | Final validation LL | Best validation epoch | Passed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test-2015-03-02 | A | 42 | -0.73954 | -0.27331 | 0.46623 | -0.67053 | -1.14381 | 5 | True |
| test-2015-03-02 | A | 43 | -0.73954 | -0.27509 | 0.46445 | -0.67053 | -1.05452 | 11 | True |
| test-2015-03-02 | A | 44 | -0.73954 | -0.27285 | 0.46669 | -0.67053 | -0.98559 | 5 | True |
| test-2015-03-02 | A | 45 | -0.73954 | -0.27483 | 0.46472 | -0.67053 | -1.18080 | 8 | True |
| test-2015-03-02 | A | 46 | -0.73954 | -0.27488 | 0.46466 | -0.67053 | -1.19500 | 4 | True |
| test-2015-03-02 | A_B_S | 42 | -0.56361 | -0.05884 | 0.50477 | -0.61587 | -1.41407 | 9 | True |
| test-2015-03-02 | A_B_S | 43 | -0.56361 | -0.06124 | 0.50237 | -0.61587 | -1.38230 | 11 | True |
| test-2015-03-02 | A_B_S | 44 | -0.56361 | -0.07288 | 0.49072 | -0.61587 | -1.61721 | 5 | True |
| test-2015-03-02 | A_B_S | 45 | -0.56361 | -0.06142 | 0.50218 | -0.61587 | -1.81601 | 8 | True |
| test-2015-03-02 | A_B_S | 46 | -0.56361 | -0.06123 | 0.50237 | -0.61587 | -1.47123 | 7 | True |
| test-2015-03-26 | A | 42 | -0.79089 | -0.31164 | 0.47925 | -2.24484 | -3.66439 | 13 | True |
| test-2015-03-26 | A | 43 | -0.79089 | -0.30605 | 0.48484 | -2.24484 | -3.47987 | 9 | True |
| test-2015-03-26 | A | 44 | -0.79089 | -0.32209 | 0.46879 | -2.24484 | -4.00005 | 13 | True |
| test-2015-03-26 | A | 45 | -0.79089 | -0.31030 | 0.48059 | -2.24484 | -3.55909 | 17 | True |
| test-2015-03-26 | A | 46 | -0.79089 | -0.32313 | 0.46776 | -2.24484 | -3.70898 | 10 | True |
| test-2015-03-26 | A_B_S | 42 | -0.78653 | -0.08062 | 0.70591 | -2.23859 | -6.11552 | 4 | True |
| test-2015-03-26 | A_B_S | 43 | -0.78653 | -0.07924 | 0.70728 | -2.23859 | -5.17390 | 3 | True |
| test-2015-03-26 | A_B_S | 44 | -0.78653 | -0.08403 | 0.70250 | -2.23859 | -5.39464 | 3 | True |
| test-2015-03-26 | A_B_S | 45 | -0.78653 | -0.08809 | 0.69843 | -2.23859 | -5.28681 | 4 | True |
| test-2015-03-26 | A_B_S | 46 | -0.78653 | -0.07958 | 0.70695 | -2.23859 | -5.50910 | 4 | True |

Curves, mean over seeds (train LL in eval mode / validation LL, bits per coda):

| Fold | Condition | Epoch 0 | Epoch 1 | Epoch 5 | Epoch 10 | Epoch 25 | Epoch 50 | Epoch 100 | Epoch 150 | Epoch 200 | Epoch 250 | Epoch 300 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test-2015-03-02 | A | -0.740 / -0.671 | -0.738 / -0.670 | -0.714 / -0.666 | -0.657 / -0.676 | -0.526 / -0.742 | -0.412 / -0.848 | -0.323 / -1.001 | -0.295 / -1.074 | -0.284 / -1.102 | -0.285 / -1.105 | -0.274 / -1.112 |
| test-2015-03-02 | A_B_S | -0.564 / -0.616 | -0.562 / -0.614 | -0.538 / -0.609 | -0.464 / -0.615 | -0.285 / -0.714 | -0.169 / -0.896 | -0.100 / -1.175 | -0.080 / -1.331 | -0.070 / -1.426 | -0.067 / -1.443 | -0.063 / -1.540 |
| test-2015-03-26 | A | -0.791 / -2.245 | -0.789 / -2.244 | -0.770 / -2.220 | -0.715 / -2.190 | -0.567 / -2.311 | -0.443 / -2.664 | -0.361 / -3.159 | -0.331 / -3.407 | -0.321 / -3.543 | -0.316 / -3.619 | -0.315 / -3.682 |
| test-2015-03-26 | A_B_S | -0.787 / -2.239 | -0.774 / -2.231 | -0.696 / -2.245 | -0.581 / -2.380 | -0.371 / -2.749 | -0.225 / -3.382 | -0.136 / -4.324 | -0.105 / -4.828 | -0.093 / -5.173 | -0.086 / -5.318 | -0.082 / -5.496 |

Gate: passed.

## c. Planted signal (PLAN [M4] check 4)

Fold test-2015-03-02, condition A, training and validation days only. Eligible targets have at least 6 own codas, and the own coda at lag 6 differs from lags 1-3. A fraction f of them (nested subsets from one RandomState(42) permutation per split) is relabelled as the own coda at lag 6; histories are unchanged. Logistic regression is refitted on the planted data with v4's procedure, and condition A's full residual selection runs on its offsets. Mean log2-probabilities are over the planted validation targets, seed-averaged. f = 0 plants nothing: it is a reference for what condition A's selection does on this fold without a planted signal, and is not part of the gate. Whether A is selected therefore only shows detection where it differs from f = 0; the planted targets' log-probabilities under the residual network and the offset show whether the planted signal itself was picked up.

| f | Planted train/validation | Refit L2 | A selected | Chosen | Chosen gain over LR (SE) | Best A candidate | Its gain over LR | Planted: offset | Planted: chosen | Planted: best A |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|
| 0.0 | 0/61, 0/13 | 0.001 | yes | A/h16/d0.4 | 0.01666 (0.00000) | A/h16/d0.4 | 0.01666 | n/a | n/a | n/a |
| 0.05 | 3/61, 1/13 | 0.001 | yes | A/h16/d0.4 | 0.02422 (0.00000) | A/h16/d0.4 | 0.02422 | -3.320 | -3.579 | -3.579 |
| 0.1 | 6/61, 1/13 | 0.001 | yes | A/h16/d0.4 | 0.02463 (0.00000) | A/h16/d0.4 | 0.02463 | -3.321 | -3.487 | -3.487 |
| 0.2 | 12/61, 3/13 | 0.001 | yes | A/h16/d0.4 | 0.03648 (0.00000) | A/h16/d0.4 | 0.03648 | -6.184 | -5.054 | -5.054 |
| 0.4 | 24/61, 5/13 | 0.001 | yes | A/h32/d0.4 | 0.04475 (0.00465) | A/h32/d0.2 | 0.04877 | -6.660 | -6.074 | -5.771 |

Gate (A's residual selected at the largest fraction): passed.

