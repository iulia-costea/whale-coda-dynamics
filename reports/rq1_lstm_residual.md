# RQ1 residual LSTM on top of logistic v4 (DEC-21)

Logistic v4 (reports/rq1_logistic_v4.md) gives the headline lattice; this network is the enlarged-family check. Per fold and condition the logits are logistic v4's logits (a fixed offset, recomputed from its committed fits) plus one zero-initialised, bias-free projection per input block that is on: an LSTM (embedding 32) per coda history over up to 10 codas, and a feed-forward encoder for S over turn category and the two onset-available timing values (standardised on training targets), each with dropout on its output. Absent blocks are not built; with every block absent the model is logistic v4. AdamW (learning rate 0.001, decoupled weight decay 0.1 on all parameters), batch 32, up to 300 epochs, patience 20, restoring the best validation epoch; epoch 0 (the offset alone) is a candidate.

Candidates per fold and condition: hidden [16, 32, 64] x dropout [0.0, 0.2, 0.4] x each block on or absent (at least one on), plus logistic v4 alone. Selection on the validation day: per-target validation log-likelihood averaged over seeds [42, 43, 44, 45, 46]; the simplest candidate whose mean paired difference to the best is within one standard error (SD of the per-target differences, ddof 1, over sqrt(n)); simplicity order: fewer blocks on, smaller hidden size, more dropout; equally simple candidates go to the higher validation log-likelihood. The SE is a selection rule, not a reported interval; reported intervals are day-bootstrap intervals (10000 replicates). Device: cpu with 18 processes (step 6d). Test days are scored once, with the chosen candidate only.

## Per condition: residual network vs logistic v4

Residual gain = residual network minus logistic v4, paired per coda (seeds averaged per coda first), day-bootstrap CI. Zero days: days whose gain is exactly zero (logistic v4 chosen on that fold, or every seed at epoch 0).

| Condition | v4 LL | Residual LL | v4 accuracy | Residual accuracy | v4 macro-F1 | Residual macro-F1 | Gain | Gain unweighted | 95% CI | Positive days | Zero days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| none | -2.60254 | -2.60254 | 57.207% | 57.207% | 0.0364 | 0.0364 | 0.00000 | 0.00000 | [0.0000, 0.0000] | 0/14 | 14 |
| S | -2.55388 | -2.56350 | 57.272% | 57.104% | 0.0441 | 0.0440 | -0.00962 | -0.02105 | [-0.0297, 0.0014] | 3/14 | 6 |
| A | -1.09690 | -1.11495 | 82.353% | 82.133% | 0.2819 | 0.2896 | -0.01805 | -0.03971 | [-0.0556, 0.0071] | 4/14 | 5 |
| B | -2.47278 | -2.57856 | 59.017% | 59.470% | 0.0729 | 0.0779 | -0.10578 | -0.07802 | [-0.2369, 0.0237] | 4/14 | 5 |
| A_B | -1.11237 | -1.14605 | 82.482% | 81.900% | 0.3270 | 0.3058 | -0.03368 | -0.12212 | [-0.1402, 0.0121] | 7/14 | 1 |
| A_S | -1.10695 | -1.11819 | 82.159% | 81.642% | 0.2824 | 0.2746 | -0.01124 | -0.01603 | [-0.0361, 0.0140] | 5/14 | 5 |
| B_S | -2.47630 | -2.59783 | 57.983% | 58.358% | 0.0691 | 0.0850 | -0.12153 | -0.09898 | [-0.2667, 0.0257] | 3/14 | 4 |
| A_B_S | -1.13261 | -1.15619 | 82.288% | 81.383% | 0.3248 | 0.3054 | -0.02359 | -0.08993 | [-0.1037, 0.0149] | 7/14 | 2 |
| majority | n/a | n/a | 57.207% | 57.207% | 0.0364 | 0.0364 | | | | | |
| repeat_own_last | -1.39276 | -1.39276 | 82.547% | 82.547% | 0.4070 | 0.4070 | | | | | |

Repetition: 1277 of 1547 comparison targets (82.5%) repeat the whale's own previous coda (1277 after UNK mapping).

## Lattices side by side

### [M2] contrasts

| Contrast | Model | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part | Switch-choice part |
|---|---|---:|---:|---|---:|---:|---:|
| own_history (A_B_S - B_S) | logistic v4 | 1.34369 | 1.25568 | [0.9445, 1.7317] | 14/14 | 1.20951 | 0.13418 |
| own_history (A_B_S - B_S) | residual | 1.44164 | 1.26473 | [0.9531, 1.9614] | 14/14 | 1.27051 | 0.17113 |
| partner_content (A_B_S - A_S) | logistic v4 | -0.02566 | -0.00144 | [-0.0942, 0.0635] | 6/14 | -0.02081 | -0.00485 |
| partner_content (A_B_S - A_S) | residual | -0.03801 | -0.07534 | [-0.1235, 0.0376] | 7/14 | -0.03117 | -0.00684 |
| timing (A_B_S - A_B) | logistic v4 | -0.02024 | -0.00885 | [-0.0464, 0.0014] | 7/14 | -0.01875 | -0.00148 |
| timing (A_B_S - A_B) | residual | -0.01014 | 0.02334 | [-0.0402, 0.0334] | 7/14 | -0.01202 | 0.00188 |

### All twelve add-one-input pairs

| Pair | v4 bits/coda | v4 unweighted | v4 CI | v4 positive days | Residual bits/coda | Residual unweighted | Residual CI | Residual positive days |
|---|---:|---:|---|---:|---:|---:|---|---:|
| adding S to no inputs | 0.04866 | 0.12024 | [-0.0804, 0.2992] | 6/14 | 0.03904 | 0.09919 | [-0.0948, 0.2865] | 6/14 |
| adding A to no inputs | 1.50564 | 1.59874 | [1.1967, 1.9137] | 14/14 | 1.48759 | 1.55903 | [1.1697, 1.8991] | 14/14 |
| adding B to no inputs | 0.12976 | 0.34451 | [-0.2908, 0.6998] | 10/14 | 0.02398 | 0.26649 | [-0.4992, 0.6880] | 9/14 |
| adding B to A | -0.01547 | 0.00720 | [-0.0747, 0.0665] | 7/14 | -0.03110 | -0.07521 | [-0.1210, 0.0411] | 9/14 |
| adding A to B | 1.36041 | 1.26143 | [0.9493, 1.7408] | 14/14 | 1.43251 | 1.21733 | [0.9219, 1.9303] | 14/14 |
| adding A to S | 1.44693 | 1.47828 | [1.1993, 1.7507] | 14/14 | 1.44531 | 1.48330 | [1.1876, 1.7569] | 14/14 |
| adding S to A | -0.01005 | -0.00021 | [-0.0248, 0.0037] | 7/14 | -0.00324 | 0.02347 | [-0.0239, 0.0286] | 9/14 |
| adding B to S | 0.07758 | 0.22116 | [-0.2833, 0.4581] | 10/14 | -0.03433 | 0.14324 | [-0.5300, 0.4715] | 10/14 |
| adding S to B | -0.00352 | -0.00311 | [-0.1085, 0.1404] | 9/14 | -0.01927 | -0.02406 | [-0.1447, 0.1348] | 7/14 |
| adding S to A and B | -0.02024 | -0.00885 | [-0.0464, 0.0014] | 7/14 | -0.01014 | 0.02334 | [-0.0402, 0.0334] | 7/14 |
| adding B to A and S | -0.02566 | -0.00144 | [-0.0942, 0.0635] | 6/14 | -0.03801 | -0.07534 | [-0.1235, 0.0376] | 7/14 |
| adding A to B and S | 1.34369 | 1.25568 | [0.9445, 1.7317] | 14/14 | 1.44164 | 1.26473 | [0.9531, 1.9614] | 14/14 |

## Chosen candidates

Per condition over the 14 folds. Best epochs are those of the chosen candidate's seeds (folds where logistic v4 alone was chosen have none). "Best not LR" counts folds where the best candidate, before the one-SE rule, was a trained network.

| Condition | LR-only chosen | Best not LR | Blocks on | Hidden | Dropout | Best epochs (min / median / max) | Seeds at epoch 0 | Validation gain over LR (min / median / max) |
|---|---:|---:|---|---|---|---|---:|---|
| none | 14/14 | 0/14 | - | - | - | - | 0 | 0.0000 / 0.0000 / 0.0000 |
| S | 6/14 | 11/14 | S: 8 | 16: 6, 64: 2 | 0.0: 2, 0.4: 6 | 0 / 4 / 129 | 4 | 0.0000 / 0.0032 / 0.0552 |
| A | 5/14 | 14/14 | A: 9 | 16: 8, 64: 1 | 0.0: 1, 0.2: 3, 0.4: 5 | 0 / 45 / 168 | 2 | 0.0000 / 0.0361 / 0.0653 |
| B | 5/14 | 10/14 | B: 9 | 16: 6, 32: 2, 64: 1 | 0.0: 3, 0.2: 2, 0.4: 4 | 0 / 24 / 92 | 3 | 0.0000 / 0.0767 / 0.7721 |
| A_B | 1/14 | 14/14 | A: 10, A+B: 1, B: 2 | 16: 9, 32: 2, 64: 2 | 0.0: 1, 0.2: 3, 0.4: 9 | 0 / 46 / 238 | 6 | 0.0000 / 0.0590 / 0.1199 |
| A_S | 5/14 | 14/14 | A: 9 | 16: 8, 32: 1 | 0.2: 4, 0.4: 5 | 2 / 47 / 155 | 0 | 0.0000 / 0.0354 / 0.0690 |
| B_S | 4/14 | 13/14 | B: 8, S: 2 | 16: 5, 32: 4, 64: 1 | 0.0: 2, 0.2: 3, 0.4: 5 | 1 / 22 / 139 | 0 | 0.0000 / 0.0659 / 0.7140 |
| A_B_S | 2/14 | 14/14 | A: 10, A+B: 1, B: 1 | 16: 8, 32: 3, 64: 1 | 0.0: 2, 0.4: 10 | 0 / 60 / 288 | 6 | 0.0000 / 0.0432 / 0.1565 |

Chosen candidate per fold (LR = logistic v4 alone; otherwise blocks/hidden/dropout):

| Fold | S | A | B | A_B | A_S | B_S | A_B_S |
|---|---|---|---|---|---|---|---|
| test-2014-04-16 | S/h64/d0.4 | A/h16/d0.2 | B/h16/d0.2 | A/h16/d0.2 | A/h16/d0.2 | B/h16/d0.2 | A/h32/d0.4 |
| test-2015-04-01 | S/h16/d0.4 | LR | B/h16/d0.4 | B/h16/d0.4 | LR | B/h16/d0.4 | LR |
| test-2015-04-10 | LR | LR | LR | LR | LR | LR | LR |
| test-2016-05-13 | S/h64/d0.4 | A/h64/d0.0 | B/h32/d0.0 | A+B/h64/d0.4 | A/h32/d0.4 | B/h32/d0.2 | A+B/h32/d0.0 |
| test-2014-04-24 | S/h16/d0.0 | A/h16/d0.4 | B/h32/d0.0 | A/h16/d0.4 | A/h16/d0.2 | B/h16/d0.0 | A/h16/d0.4 |
| test-2015-04-07 | LR | A/h16/d0.4 | B/h16/d0.4 | A/h32/d0.2 | A/h16/d0.4 | S/h32/d0.4 | A/h16/d0.4 |
| test-2016-05-12 | S/h16/d0.4 | A/h16/d0.4 | LR | A/h16/d0.4 | A/h16/d0.4 | LR | A/h16/d0.4 |
| test-2014-04-29 | S/h16/d0.0 | LR | B/h16/d0.2 | A/h16/d0.4 | LR | B/h64/d0.0 | A/h16/d0.4 |
| test-2015-03-19 | LR | A/h16/d0.2 | B/h64/d0.4 | B/h64/d0.4 | A/h16/d0.2 | B/h32/d0.2 | B/h64/d0.4 |
| test-2016-04-24 | LR | LR | LR | A/h16/d0.2 | LR | LR | A/h16/d0.4 |
| test-2016-04-12 | LR | LR | LR | A/h32/d0.0 | LR | S/h32/d0.4 | A/h32/d0.0 |
| test-2015-03-02 | S/h16/d0.4 | A/h16/d0.4 | B/h16/d0.0 | A/h16/d0.4 | A/h16/d0.4 | B/h16/d0.4 | A/h16/d0.4 |
| test-2015-03-31 | S/h16/d0.4 | A/h16/d0.4 | LR | A/h16/d0.4 | A/h16/d0.2 | LR | A/h16/d0.4 |
| test-2015-03-26 | LR | A/h16/d0.2 | B/h16/d0.4 | A/h16/d0.4 | A/h16/d0.4 | B/h16/d0.4 | A/h16/d0.4 |

## A-conditions vs repeat-own-last (residual network)

| Condition | LL contrast | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| A | 0.27781 | 0.32090 | [0.2115, 0.3590] | 13/14 |
| A_B | 0.24671 | 0.24570 | [0.1175, 0.3732] | 13/14 |
| A_S | 0.27457 | 0.34438 | [0.2186, 0.3628] | 13/14 |
| A_B_S | 0.23656 | 0.26904 | [0.1298, 0.3723] | 13/14 |

## Stop rules (PLAN [M4])

Stop and investigate before writing conclusions if any of these has a day-bootstrap CI entirely below zero: an A-condition against repeat-own-last; an add-one-input contrast; the residual gain over logistic v4 on a condition (which now signals selection overfitting to validation days, DEC-21).

None fires.

## Checks before the full run (step 6)

### a. Wiring (PLAN [M4] checks 1 and 2), all folds and conditions

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

### b. Learnability (PLAN [M4] check 3)

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

### c. Planted signal (PLAN [M4] check 4)

Fold test-2015-03-02, condition A, training and validation days only. Eligible targets have at least 6 own codas, and the own coda at lag 6 differs from lags 1-3. A fraction f of them (nested subsets from one RandomState(42) permutation per split) is relabelled as the own coda at lag 6; histories are unchanged. Logistic regression is refitted on the planted data with v4's procedure, and condition A's full residual selection runs on its offsets. Mean log2-probabilities are over the planted validation targets, seed-averaged. f = 0 plants nothing: it is a reference for what condition A's selection does on this fold without a planted signal, and is not part of the gate. Whether A is selected therefore only shows detection where it differs from f = 0; the planted targets' log-probabilities under the residual network and the offset show whether the planted signal itself was picked up.

| f | Planted train/validation | Refit L2 | A selected | Chosen | Chosen gain over LR (SE) | Best A candidate | Its gain over LR | Planted: offset | Planted: chosen | Planted: best A |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|
| 0.0 | 0/61, 0/13 | 0.001 | yes | A/h16/d0.4 | 0.01666 (0.00000) | A/h16/d0.4 | 0.01666 | n/a | n/a | n/a |
| 0.05 | 3/61, 1/13 | 0.001 | yes | A/h16/d0.4 | 0.02422 (0.00000) | A/h16/d0.4 | 0.02422 | -3.320 | -3.579 | -3.579 |
| 0.1 | 6/61, 1/13 | 0.001 | yes | A/h16/d0.4 | 0.02463 (0.00000) | A/h16/d0.4 | 0.02463 | -3.321 | -3.487 | -3.487 |
| 0.2 | 12/61, 3/13 | 0.001 | yes | A/h16/d0.4 | 0.03648 (0.00000) | A/h16/d0.4 | 0.03648 | -6.184 | -5.054 | -5.054 |
| 0.4 | 24/61, 5/13 | 0.001 | yes | A/h32/d0.4 | 0.04475 (0.00465) | A/h32/d0.2 | 0.04877 | -6.660 | -6.074 | -5.771 |

Gate (A's residual selected at the largest fraction): passed.

f = 0 already selects A, and the residual only improves the planted targets at f >= 0.2, so the design detects strong patterns only.

### d. Device and throughput

Fold test-2015-03-02, condition A_B_S, every block on, hidden 64, dropout 0.0, normal early stopping (patience 20). Each worker fits seeds [42, 43, 44] once; throughput = fits / wall time, workers started beforehand. CPU workers use one torch thread each. The GPU is eligible only if two fits with the same seed, in two processes under torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG (here ':4096:8'), give identical curves and weights. Ties within 5% go to the CPU.

| Device | Processes | Fits | Wall seconds | Fits/hour | Mean epochs/fit | Eligible |
|---|---:|---:|---:|---:|---:|---|
| cpu | 18 | 54 | 37.7 | 5150.8 | 28.3 | yes |
| cuda | 1 | 3 | 8.6 | 1257.6 | 28.3 | yes |
| cuda | 2 | 6 | 9.8 | 2206.7 | 28.3 | yes |
| cuda | 4 | 12 | 13.7 | 3162.0 | 28.3 | yes |

| Determinism check | Seed | Identical | Note |
|---|---:|---|---|
| cpu | 42 | True |  |
| cuda | 42 | True |  |

Chosen: cpu with 18 processes (5150.8 fits/hour on the largest candidate). Full run: 11970 fits; estimated wall time 2.3 hours (the largest candidate is the slowest per epoch, so this is conservative per epoch; epochs per fit vary).

## Context check against Assayag et al. (PLAN [M4])

Their split is random (70/15/15), ours holds out whole days, so their accuracies are optimistic next to ours; no model is retrained for this comparison.

Condition B (partner history) on the 1086 switch and overlap targets: residual network 61.123%, logistic v4 60.129%, majority class (each fold's training majority) 57.551%. Assayag et al. Table 3, LSTM, paired coda type: 0.373 to 0.766 across 0-8 preceding codas.

Condition A (own history), all 1547 targets: residual network 82.133%, logistic v4 82.353%. By own-history length, next to Assayag et al. Table 3, LSTM, single-whale coda type (column t-k uses the current coda and k before it, i.e. k+1 own codas; their window is fixed, ours is whatever each target has, capped at 10, so the columns are not paired one to one):

| Own codas available | Targets | Residual accuracy | v4 accuracy | Assayag column | Assayag accuracy |
|---:|---:|---:|---:|---|---:|
| 1 | 223 | 78.027% | 78.027% | t-0 | 0.800 |
| 2 | 236 | 76.695% | 77.542% | t-1 | 0.824 |
| 3 | 200 | 79.100% | 79.500% | t-2 | 0.821 |
| 4 | 158 | 82.785% | 82.911% | t-3 | 0.889 |
| 5 | 136 | 79.559% | 80.147% | t-4 | 0.859 |
| 6 | 114 | 83.158% | 81.579% | t-5 | 0.886 |
| 7 | 86 | 89.070% | 90.698% | t-6 | 0.922 |
| 8 | 75 | 90.667% | 90.667% | t-7 | 0.746 |
| 9 | 56 | 88.929% | 89.286% | t-8 | 0.845 |
| 10 | 263 | 87.148% | 87.072% | - | - |

## Reading (PLAN §9)

- Headline numbers come from logistic v4; the residual network's lattice and residual gains are the enlarged-family check.
- A contrast is v4's contrast plus the difference between the two conditions' residual gains, so the contrasts are not monotone by construction.
- On validation days the residual network and logistic regression agree by construction, so their agreement is not evidence against an optimisation failure; that evidence is the learnability checks above.
- Logistic regression is a softmax network without hidden layers; the residual LSTM's family contains v4 exactly, as the member whose residual is switched off.
- Limited data and capacity push contrasts toward zero: a positive contrast is robust, a null might just mean too little data.

| Condition | Residual gain CI | Reading |
|---|---|---|
| S | [-0.0297, 0.0014] | includes zero: no further structure detectable in the enlarged family at this data size, not that none exists |
| A | [-0.0556, 0.0071] | includes zero: no further structure detectable in the enlarged family at this data size, not that none exists |
| B | [-0.2369, 0.0237] | includes zero: no further structure detectable in the enlarged family at this data size, not that none exists |
| A_B | [-0.1402, 0.0121] | includes zero: no further structure detectable in the enlarged family at this data size, not that none exists |
| A_S | [-0.0361, 0.0140] | includes zero: no further structure detectable in the enlarged family at this data size, not that none exists |
| B_S | [-0.2667, 0.0257] | includes zero: no further structure detectable in the enlarged family at this data size, not that none exists |
| A_B_S | [-0.1037, 0.0149] | includes zero: no further structure detectable in the enlarged family at this data size, not that none exists |

Not run here: the old-delay (retrospective) sensitivity run of the residual network; the logistic v4 old-delay run is reports/rq1_retrospective_sensitivity_v4.md.

This report does not establish grammar, syntax or causal mechanisms. Claims are about two-whale exchanges only. The switch-choice component of every contrast rests on about 270 non-repeat codas and is exploratory; 14 days give rough intervals.

Files (data/results/rq1_lstm_residual/): predictions.jsonl (test probabilities, every seed), selection.jsonl (every candidate's seed-averaged validation log-likelihood, paired difference to the best and SE, per fold and condition), validation_scores.jsonl (per-target validation log-likelihoods behind the selection), fits.jsonl (best epoch, epochs trained and validation log-likelihoods of every fit), evaluation/ (shared-evaluator output; summary.json records SHA-256 of every input). training_curves.jsonl (per-epoch training and validation log-likelihood of every fit) is regenerated by the run and not committed (size). Model weights are not saved.
