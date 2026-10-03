# RQ1 own-history length sweep (PLAN [P2], DEC-25)

Logistic v4 unchanged except the number of own-history lags (A_history): k in [1, 2, 3, 4, 5, 6, 7, 8, 9]. The partner keeps 3 lags; copy weights, the L2 grid [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], the joint history and S penalties (DEC-20), the selection rule (validation day, ties to the larger value), the 14 folds, the targets, the UNK policy and the labels (`guy_coda_type`) are v4's. Every k fits all eight conditions: the shared evaluation needs every condition, and DEC-16's stop rules use the add-one-input contrasts. Conditions without own history (none, S, B, B_S) are refitted at every k and are identical to the k = 3 run (checked exactly). All k, including the reference k = 3, were fitted in one environment (Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, one BLAS thread per fit): 44226 fits, all converged (fit() raises otherwise).

Contrasts are LL(k) - LL(3) per condition, paired per coda against this environment's k = 3 fit (not the committed v4 run), with the [P1] repeat and switch parts. Intervals are day-bootstrap (10,000 replicates, seed 42); the unweighted day mean is the mean of the 14 day-level contrasts.

## Gate: k = 3 here against the committed logistic v4 run

Chosen penalties agree on 98 of 98 fold and condition fits, and every condition's mean log-likelihood is within 1e-06 bits/coda of v4's. The gate (DEC-25 addendum) requires at most 1 differing penalty and mean log-likelihoods within 1e-05 bits/coda: passed. Predicted probabilities are reported, not enforced, because v4's fits reproduce only to about 1e-4 at the smallest penalties (optimizer tolerance gtol 1e-6; the BLAS thread count alone moves them that much): largest absolute difference 9.73e-05 over all fits (9.73e-05 over the fits whose penalties agree); 15 fits differ by more than 1e-05 (listed below).

| Condition | Fits | Penalties differing | Max abs probability difference | LL here | LL v4 | Difference |
|---|---:|---:|---:|---:|---:|---:|
| none | 14 | 0 | 0 | -2.6025399 | -2.6025399 | 0 |
| S | 14 | 0 | 0 | -2.5538792 | -2.5538792 | 0 |
| A | 14 | 0 | 9.05e-05 | -1.0968981 | -1.0968984 | 2.6e-07 |
| B | 14 | 0 | 6.02e-05 | -2.4727764 | -2.4727774 | 1e-06 |
| A_B | 14 | 0 | 7.09e-05 | -1.1123698 | -1.1123708 | 1e-06 |
| A_S | 14 | 0 | 8.74e-05 | -1.1069462 | -1.1069467 | 4.4e-07 |
| B_S | 14 | 0 | 2.53e-05 | -2.4762998 | -2.4762999 | 6.6e-08 |
| A_B_S | 14 | 0 | 9.73e-05 | -1.1326057 | -1.1326062 | 5e-07 |

Fits whose predicted probabilities differ from v4's by more than 1e-05, with the chosen penalties (history, S):

| Fold | Condition | History L2 | S L2 | Max abs probability difference |
|---|---|---:|---:|---:|
| test-2016-05-13 | A_B_S | 0.0001 | 1.0 | 9.73e-05 |
| test-2015-03-19 | A | 0.0001 | n/a | 9.05e-05 |
| test-2016-05-13 | A_S | 0.0001 | 1.0 | 8.74e-05 |
| test-2016-05-13 | A_B | 0.0001 | n/a | 7.09e-05 |
| test-2016-05-13 | B | 0.0001 | n/a | 6.02e-05 |
| test-2015-03-19 | A_S | 0.0001 | 1.0 | 4.97e-05 |
| test-2015-03-19 | B | 0.0001 | n/a | 3.71e-05 |
| test-2015-03-19 | A_B_S | 0.0003 | 1.0 | 3.08e-05 |
| test-2014-04-24 | A_S | 0.0001 | 0.0003 | 2.59e-05 |
| test-2015-03-19 | B_S | 0.0001 | 1.0 | 2.53e-05 |
| test-2015-04-07 | B_S | 1.0 | 0.0003 | 1.97e-05 |
| test-2014-04-29 | A_S | 0.0003 | 1.0 | 1.7e-05 |
| test-2014-04-24 | B | 0.0003 | n/a | 1.62e-05 |
| test-2016-05-13 | B_S | 0.003 | 0.0001 | 1.61e-05 |
| test-2016-05-13 | A | 0.0001 | n/a | 1.6e-05 |

The gate was amended after the first k = 3 gate run (DEC-25 addendum). That run required every predicted probability within 1e-5 of v4's where penalties agree and failed at 9.7e-5, with all 98 penalties identical and every condition's mean log-likelihood within 1e-6 bits/coda of v4's. At that point no k = 9 log-likelihood, calibration or contrast number had been seen; the k = 9 run's log line showed only its run time and that no DEC-16 stop rule fired. All nine k and the calibration were then refitted in one batch from a clean tree.

During development the smoke test (a fixed L2 of 0.01, not this grid) printed sweep numbers; no setting of the sweep changed after them.

## Log-likelihood by number of own lags

Held-out log-likelihood (bits/coda, 1,547 targets), accuracy, and the chosen history L2 (median over folds; folds at a grid edge). Repeat-own-last: -1.39276 bits/coda.

| k | A LL | A accuracy | A history L2 (edge) | A_B_S LL | A_B_S accuracy | A_B_S history L2 (edge) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -1.12052 | 81.900% | 0.00065 (5/14) | -1.18846 | 79.897% | 0.001 (2/14) |
| 2 | -1.09694 | 82.418% | 0.001 (3/14) | -1.14618 | 81.319% | 0.001 (1/14) |
| 3 | -1.09690 | 82.353% | 0.001 (3/14) | -1.13261 | 82.288% | 0.001 (1/14) |
| 4 | -1.09331 | 83.452% | 0.001 (2/14) | -1.12542 | 81.577% | 0.001 (1/14) |
| 5 * | -1.09631 | 83.387% | 0.001 (2/14) | -1.12227 | 82.224% | 0.002 (1/14) |
| 6 * | -1.09647 | 83.710% | 0.003 (2/14) | -1.12352 | 82.482% | 0.003 (1/14) |
| 7 | -1.09714 | 83.710% | 0.003 (2/14) | -1.12453 | 82.224% | 0.003 (2/14) |
| 8 | -1.10040 | 83.710% | 0.003 (2/14) | -1.12422 | 82.547% | 0.002 (2/14) |
| 9 | -1.10010 | 83.646% | 0.003 (2/14) | -1.12356 | 82.482% | 0.002 (1/14) |

Rows marked * are the k at which DEC-16's rule (an interval entirely below zero) fires (k = 5, 6). See the stop rules below; no conclusion is drawn from a k at which either rule fires.

## Contrasts LL(k) - LL(3)

Primary (pre-specified): LL(9) - LL(3) in A_B_S. Secondary (pre-specified): LL(1) - LL(3), in each condition. The rest of the curve is descriptive. The primary and secondary contrasts also get the exact two-sided sign-flip p (DEC-23: all 2^14 sign assignments of the day sums, token-weighted mean as the statistic).

| Condition | k | Role | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part [CI] | Switch part [CI] | Sign-flip p (two-sided) |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| A | 1 | **secondary** | -0.02362 | -0.00843 | [-0.0572, 0.0150] | 5/14 | 0.00011 [-0.0252, 0.0267] | -0.02373 [-0.0494, 0.0016] | 0.3295 |
| A | 2 | descriptive | -0.00005 | 0.00551 | [-0.0077, 0.0104] | 5/14 | -0.00059 [-0.0056, 0.0071] | 0.00054 [-0.0058, 0.0066] |  |
| A | 4 | descriptive | 0.00359 | 0.02020 | [-0.0136, 0.0393] | 8/14 | 0.00776 [-0.0085, 0.0447] | -0.00418 [-0.0097, 0.0011] |  |
| A | 5 | descriptive * | 0.00058 | 0.01517 | [-0.0168, 0.0349] | 6/14 | 0.00501 [-0.0115, 0.0405] | -0.00443 [-0.0156, 0.0038] |  |
| A | 6 | descriptive * | 0.00043 | 0.00974 | [-0.0200, 0.0339] | 6/14 | 0.00400 [-0.0143, 0.0388] | -0.00357 [-0.0141, 0.0033] |  |
| A | 7 | descriptive | -0.00024 | 0.01163 | [-0.0209, 0.0350] | 7/14 | 0.00297 [-0.0172, 0.0401] | -0.00321 [-0.0105, 0.0018] |  |
| A | 8 | descriptive | -0.00350 | 0.00681 | [-0.0261, 0.0319] | 7/14 | 0.00144 [-0.0206, 0.0398] | -0.00494 [-0.0193, 0.0026] |  |
| A | 9 | descriptive | -0.00321 | 0.00811 | [-0.0254, 0.0319] | 5/14 | 0.00063 [-0.0215, 0.0389] | -0.00383 [-0.0193, 0.0046] |  |
| A_B_S | 1 | **secondary** | -0.05586 | -0.03124 | [-0.1109, 0.0006] | 2/14 | -0.02070 [-0.0630, 0.0189] | -0.03516 [-0.0797, -0.0053] | 0.1377 |
| A_B_S | 2 | descriptive | -0.01358 | -0.01046 | [-0.0238, -0.0039] | 1/14 | -0.00847 [-0.0165, 0.0004] | -0.00511 [-0.0090, -0.0016] |  |
| A_B_S | 4 | descriptive | 0.00719 | 0.00738 | [0.0002, 0.0151] | 7/14 | 0.00459 [-0.0023, 0.0115] | 0.00260 [-0.0012, 0.0075] |  |
| A_B_S | 5 | descriptive * | 0.01033 | -0.00186 | [-0.0083, 0.0333] | 5/14 | 0.00862 [-0.0086, 0.0265] | 0.00171 [-0.0080, 0.0087] |  |
| A_B_S | 6 | descriptive * | 0.00908 | -0.00725 | [-0.0160, 0.0349] | 6/14 | 0.00507 [-0.0143, 0.0224] | 0.00401 [-0.0085, 0.0144] |  |
| A_B_S | 7 | descriptive | 0.00808 | -0.00369 | [-0.0139, 0.0340] | 6/14 | 0.00419 [-0.0128, 0.0207] | 0.00388 [-0.0086, 0.0148] |  |
| A_B_S | 8 | descriptive | 0.00839 | -0.00474 | [-0.0143, 0.0344] | 6/14 | 0.00143 [-0.0140, 0.0165] | 0.00696 [-0.0039, 0.0189] |  |
| A_B_S | 9 | **primary** | 0.00904 | -0.00328 | [-0.0141, 0.0374] | 5/14 | 0.00128 [-0.0144, 0.0170] | 0.00776 [-0.0045, 0.0212] | 0.9806 |

Rows marked * are the k at which DEC-16's rule (an interval entirely below zero) fires (k = 5, 6). See the stop rules below; no conclusion is drawn from a k at which either rule fires.

No contrast with more own lags than 3 has an interval entirely below zero (no fitting flag). A negative LL(k) - LL(3) for k < 3 is not a flag: the smaller model drops lags.

## Calibration: planted lag-6 copies, condition A

The residual LSTM's planted-signal protocol (`residual_planted.py`, reused, not copied): on all 14 folds a fraction f in [0.1, 0.2, 0.4] of the eligible targets on training, validation and test days is relabelled as the whale's own coda at lag 6 (eligible: at least 6 own codas, and the coda at lag 6 differs from lags 1-3); labels only, histories unchanged; nested subsets, one permutation per fold and split (sampling seed 42). Logistic regression is refitted on the planted training and validation days with v4's procedure at k = 9 and at k = 3. Scored on the planted test labels: (1) LR(9) - LR(3); (2) an oracle minus LR(3), the oracle predicting f x 1[own coda at lag 6] + (1 - f) x P_LR(3) on eligible test targets and P_LR(3) elsewhere (it knows the rule and f, not which targets were planted); (3) the recovery ratio r = (1)/(2), a ratio of token-weighted means within each bootstrap replicate over the same resampled days (10,000 replicates). Replicates whose oracle difference is at or below zero have no ratio and are counted (column Non-positive); more than 5 make r unstable at that f.

| f | Planted test targets (share) | (1) LR(9) - LR(3) | CI | Positive days | (2) Oracle - LR(3) | CI | (3) r | r CI | Non-positive |
|---:|---|---:|---|---:|---:|---|---:|---|---:|
| 0.1 | 11/1547 (0.7%) | 0.00403 | [-0.0175, 0.0389] | 7/14 | 0.03513 | [0.0185, 0.0550] | 0.1148 | [-0.5918, 1.224] | 0/10000 |
| 0.2 | 23/1547 (1.5%) | -0.01448 | [-0.0454, 0.0256] | 8/14 | 0.06903 | [0.0479, 0.0896] | -0.2098 | [-0.6361, 0.4318] | 0/10000 |
| 0.4 | 43/1547 (2.8%) | 0.03351 | [0.0156, 0.0551] | 12/14 | 0.13702 | [0.0960, 0.1808] | 0.2446 | [0.1074, 0.4724] | 0/10000 |

Bound. The real-data contrast LL(9) - LL(3) in A has a day-bootstrap upper bound of 0.0319 bits/coda (contrast table above). Where r's lower bound is above zero and the oracle difference stays positive in every replicate, that upper bound divided by r's lower bound is the oracle gain above which a pattern of this form (a deterministic copy of the own coda at lag 6) would have pushed the real-data contrast past its observed upper bound. With an upper bound at or below zero there is nothing to divide and no bound is stated.

- f = 0.1: r's lower bound is -0.5918, at or below zero, so the run gives no bound.
- f = 0.2: r's lower bound is -0.6361, at or below zero, so the run gives no bound.
- f = 0.4: r >= 0.1074, so a lag-6 copy pattern worth more than 0.2966 bits/coda to the oracle would have been detected (here the oracle gains 0.1370).

Planted test targets: mean log2-probability of the planted label.

| f | Targets | LR(3) | LR(9) | Oracle |
|---:|---:|---:|---:|---:|
| 0.1 | 11 | -7.912 | -7.159 | -3.233 |
| 0.2 | 23 | -7.218 | -6.622 | -2.226 |
| 0.4 | 43 | -6.899 | -5.472 | -1.278 |

Planting check against the residual planted run (`data/results/rq1_lstm_residual_planted/scores.jsonl`): eligibility, planted flags and planted labels are identical on all 4,641 fold, fraction and target combinations; the k = 3 refit differs from that run's refitted logistic regression by at most 0.000759 bits (68 targets differ by more than 1e-5).

## Stop rules per k (DEC-16 and DEC-23)

Each k is a full run of the 16 checks: the 12 add-one-input contrasts and the 4 A-conditions against repeat-own-last. DEC-16's rule, under which the sweep was specified (DEC-25), stops when a check's day-bootstrap interval lies entirely below zero. DEC-23's rule, which supersedes it, gives each check an exact one-sided sign-flip test (alternative: mean below zero) on its day sums, with Holm's correction across the 16, and stops when an adjusted p is below 0.05 (`logistic_v5.dec23_stop_rules`). Both are reported. The sweep keeps per-coda values only for its swept conditions, so DEC-23's day sums are each run's stored day means times the day's target count; the means they give match the stored ones to 1e-12.

| k | DEC-16: interval entirely below zero | DEC-23: Holm-adjusted p < 0.05 | Smallest one-sided p, Holm-adjusted (check) |
|---:|---|---|---|
| 1 | no stop rule fires | no stop rule fires | 0.1994, Holm 1.0000 (add_input:A_B_S-A_B) |
| 2 | no stop rule fires | no stop rule fires | 0.1409, Holm 1.0000 (add_input:A_B_S-A_B) |
| 3 | no stop rule fires | no stop rule fires | 0.1194, Holm 1.0000 (add_input:A_B_S-A_B) |
| 4 | no stop rule fires | no stop rule fires | 0.1595, Holm 1.0000 (add_input:A_B_S-A_B) |
| 5 | add_input:A_B_S-A_B: contrast CI entirely below zero [-0.0186, -0.0009] | no stop rule fires | 0.0562, Holm 0.8994 (add_input:A_B_S-A_B) |
| 6 | add_input:A_B_S-A_B: contrast CI entirely below zero [-0.0107, -0.0006] | no stop rule fires | 0.0504, Holm 0.8066 (add_input:A_B_S-A_B) |
| 7 | no stop rule fires | no stop rule fires | 0.1113, Holm 1.0000 (add_input:A_B_S-A_B) |
| 8 | no stop rule fires | no stop rule fires | 0.0949, Holm 1.0000 (add_input:A_B_S-A_B) |
| 9 | no stop rule fires | no stop rule fires | 0.0679, Holm 1.0000 (add_input:A_B_S-A_B) |

## Reading (DEC-25)

Primary, pre-specified: LL(9) - LL(3) in A_B_S: 0.00904 bits/coda (unweighted day mean -0.00328, 95% CI [-0.0141, 0.0374], 5/14 days positive, two-sided sign-flip p = 0.9806). Repeat part 0.00128 [-0.0144, 0.0170]; switch part 0.00776 [-0.0045, 0.0212].
No detectable information beyond 3 own codas for this model at this data size. Calibration (condition A): at f = 0.4, a lag-6 copy pattern worth more than 0.2966 bits/coda to the oracle would have been detected. A null may also mean too little data: limited data and capacity push contrasts toward zero (PLAN §9).
Secondary, pre-specified: LL(1) - LL(3) in A: -0.02362 bits/coda (unweighted day mean -0.00843, 95% CI [-0.0572, 0.0150], 5/14 days positive, two-sided sign-flip p = 0.3295); LL(1) - LL(3) in A_B_S: -0.05586 bits/coda (unweighted day mean -0.03124, 95% CI [-0.1109, 0.0006], 2/14 days positive, two-sided sign-flip p = 0.1377).

## Limits

- The extra lags share one history penalty with lags 1-3, the partner block and the copy weights (v4's rule, kept by DEC-25), so a null may partly reflect that compromise.
- Day-bootstrap intervals are approximate with only 14 days. The switch part rests on about 270 non-repeat codas and is exploratory.
- The calibration covers one pattern form (a copy of the own coda at lag 6) in condition A only; it says nothing about structure of other forms.
- Claims are about dyadic exchanges only; this sweep says nothing about grammar or syntax.

Per-k outputs are in `data/results/rq1_history_length/k<k>/` (summary.json with the stop rules and input SHA-256, l2_selection.jsonl, scored.jsonl for the swept conditions), the calibration in `data/results/rq1_history_length/calibration/`; `data/results/rq1_history_length/summary.json` records SHA-256 of every input to this report.
