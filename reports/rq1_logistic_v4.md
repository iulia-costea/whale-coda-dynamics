# RQ1 logistic regression v4: separate penalties

Joint validation-day selection over base L2 [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] and S L2 [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]. History lag weights, copy weights and the shared intercept use base L2; all timing and turn-category coefficients use S L2. Ties prefer larger base L2, then larger S L2. No-S conditions retain v3's fit exactly. Training excludes validation and test days. Preprocessing, folds, vocabulary and shared evaluation are unchanged.

## Metrics, v3 vs v4

| Condition | v3 LL bits/coda | v4 LL bits/coda | v3 Accuracy | v4 Accuracy | v3 Macro-F1 | v4 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| none | -2.60254 | -2.60254 | 57.207% | 57.207% | 0.0364 | 0.0364 |
| S | -2.52372 | -2.55388 | 56.820% | 57.272% | 0.0437 | 0.0441 |
| A | -1.09690 | -1.09690 | 82.353% | 82.353% | 0.2819 | 0.2819 |
| B | -2.47278 | -2.47278 | 59.017% | 59.017% | 0.0729 | 0.0729 |
| A_B | -1.11237 | -1.11237 | 82.482% | 82.482% | 0.3270 | 0.3270 |
| A_S | -1.13053 | -1.10695 | 82.612% | 82.159% | 0.3181 | 0.2824 |
| B_S | -2.50494 | -2.47630 | 57.660% | 57.983% | 0.0668 | 0.0691 |
| A_B_S | -1.16249 | -1.13261 | 82.094% | 82.288% | 0.3422 | 0.3248 |
| majority | n/a | n/a | 57.207% | 57.207% | 0.0364 | 0.0364 |
| repeat_own_last | -1.39276 | -1.39276 | 82.547% | 82.547% | 0.4070 | 0.4070 |

## Chosen L2 per condition

| Condition | Min | Median | Max | Folds at grid edge |
|---|---:|---:|---:|---|
| S | 0.0001 | 0.002 | 1.0 | 10/14 (test-2016-05-13, test-2014-04-24, test-2015-04-07, test-2016-05-12, test-2014-04-29, test-2015-03-19, test-2016-04-24, test-2015-03-02, test-2015-03-31, test-2015-03-26) |
| A | 0.0001 | 0.001 | 0.03 | 3/14 (test-2016-05-13, test-2014-04-24, test-2015-03-19) |
| B | 0.0001 | 0.003 | 0.3 | 2/14 (test-2016-05-13, test-2015-03-19) |
| A_B | 0.0001 | 0.001 | 0.01 | 1/14 (test-2016-05-13) |
| A_S | 0.0001 | 0.001 | 0.03 | 3/14 (test-2016-05-13, test-2014-04-24, test-2015-03-19) |
| B_S | 0.0001 | 0.006500000000000001 | 1.0 | 5/14 (test-2015-04-10, test-2015-04-07, test-2016-05-12, test-2015-03-19, test-2016-04-12) |
| A_B_S | 0.0001 | 0.001 | 0.01 | 1/14 (test-2016-05-13) |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.3813 | 3.4481 | n/a | n/a |
| B | n/a | n/a | 0.7672 | 1.7782 |
| A_B | 2.3328 | 3.0183 | 0.7258 | 1.3087 |
| A_S | 2.3773 | 3.4470 | n/a | n/a |
| B_S | n/a | n/a | 0.1497 | 1.7799 |
| A_B_S | 2.3320 | 3.0178 | 0.7463 | 1.3098 |

## A-conditions vs repeat-own-last, v3 vs v4

Repeat-own-last: 82.547% accuracy, 0.4070 macro-F1, -1.39276 bits/coda.

| Condition | v3 LL contrast | v3 CI | v3 +days | v4 LL contrast | v4 CI | v4 +days | v4 Accuracy | v4 Macro-F1 |
|---|---:|---|---:|---:|---|---:|---:|---:|
| A | 0.29586 | [0.2420, 0.3719] | 14/14 | 0.29586 | [0.2420, 0.3719] | 14/14 | 82.353% | 0.2819 |
| A_B | 0.28039 | [0.2158, 0.4092] | 13/14 | 0.28039 | [0.2158, 0.4092] | 13/14 | 82.482% | 0.3270 |
| A_S | 0.26223 | [0.2089, 0.3632] | 13/14 | 0.28581 | [0.2386, 0.3663] | 14/14 | 82.159% | 0.2824 |
| A_B_S | 0.23027 | [0.1502, 0.3838] | 13/14 | 0.26015 | [0.1855, 0.4047] | 13/14 | 82.288% | 0.3248 |

## All twelve add-one-input contrasts (v4)

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | 0.04866 | 0.12024 | [-0.0804, 0.2992] | 6/14 |
| adding A to no inputs | 1.50564 | 1.59874 | [1.1967, 1.9137] | 14/14 |
| adding B to no inputs | 0.12976 | 0.34452 | [-0.2908, 0.6998] | 10/14 |
| adding B to A | -0.01547 | 0.00720 | [-0.0747, 0.0665] | 7/14 |
| adding A to B | 1.36041 | 1.26143 | [0.9493, 1.7408] | 14/14 |
| adding A to S | 1.44693 | 1.47829 | [1.1993, 1.7507] | 14/14 |
| adding S to A | -0.01005 | -0.00021 | [-0.0248, 0.0037] | 7/14 |
| adding B to S | 0.07758 | 0.22117 | [-0.2833, 0.4581] | 10/14 |
| adding S to B | -0.00352 | -0.00311 | [-0.1085, 0.1404] | 9/14 |
| adding S to A and B | -0.02024 | -0.00885 | [-0.0464, 0.0014] | 7/14 |
| adding B to A and S | -0.02566 | -0.00144 | [-0.0942, 0.0635] | 6/14 |
| adding A to B and S | 1.34369 | 1.25568 | [0.9445, 1.7317] | 14/14 |

## Main contrasts, v3 vs v4

| Contrast | v3 bits/coda | v4 bits/coda | v4 unweighted | v4 CI | v4 +days | v4 Repeat part | v4 Switch part |
|---|---:|---:|---:|---|---:|---:|---:|
| own_history | 1.34245 | 1.34369 | 1.25568 | [0.9445, 1.7317] | 14/14 | 1.20951 | 0.13418 |
| partner_content | -0.03196 | -0.02566 | -0.00144 | [-0.0942, 0.0635] | 6/14 | -0.02081 | -0.00485 |
| timing | -0.05012 | -0.02024 | -0.00885 | [-0.0464, 0.0014] | 7/14 | -0.01875 | -0.00148 |

All 4914 fits (14 folds times 7 non-unigram conditions; candidate counts are recorded per fit) converged; fit() raises immediately on non-convergence, so reaching this report already confirms it. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

## Result

No A-condition is reliably below repeat-own-last in log-likelihood, and no add-one-input contrast has a confidence interval entirely below zero.

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 270 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq1_folds.md reports UNK rates per fold.

## Penalty distributions and gate 1

| Condition | Penalty | Value: fold count | Lower edge | Upper edge |
|---|---|---|---:|---:|
| S | chosen_l2 | 0.0001: 6, 0.001: 1, 0.003: 1, 0.03: 1, 0.1: 1, 1.0: 4 | 6/14 | 4/14 |
| S | chosen_s_l2 | 0.0001: 2, 0.001: 2, 0.003: 3, 0.01: 1, 0.03: 3, 0.1: 1, 1.0: 2 | 2/14 | 2/14 |
| A | chosen_l2 | 0.0001: 3, 0.0003: 3, 0.001: 4, 0.003: 3, 0.03: 1 | 3/14 | 0/14 |
| B | chosen_l2 | 0.0001: 2, 0.0003: 1, 0.001: 3, 0.003: 2, 0.01: 3, 0.03: 1, 0.1: 1, 0.3: 1 | 2/14 | 0/14 |
| A_B | chosen_l2 | 0.0001: 1, 0.0003: 2, 0.001: 5, 0.003: 4, 0.01: 2 | 1/14 | 0/14 |
| A_S | chosen_l2 | 0.0001: 3, 0.0003: 3, 0.001: 4, 0.003: 3, 0.03: 1 | 3/14 | 0/14 |
| A_S | chosen_s_l2 | 0.0003: 1, 0.01: 3, 0.03: 3, 0.3: 1, 1.0: 6 | 0/14 | 6/14 |
| B_S | chosen_l2 | 0.0001: 1, 0.0003: 1, 0.001: 3, 0.003: 2, 0.01: 2, 0.3: 1, 1.0: 4 | 1/14 | 4/14 |
| B_S | chosen_s_l2 | 0.0001: 1, 0.0003: 1, 0.001: 1, 0.01: 2, 0.03: 4, 0.3: 1, 1.0: 4 | 1/14 | 4/14 |
| A_B_S | chosen_l2 | 0.0001: 1, 0.0003: 2, 0.001: 5, 0.003: 4, 0.01: 2 | 1/14 | 0/14 |
| A_B_S | chosen_s_l2 | 0.0003: 1, 0.003: 2, 0.01: 2, 0.03: 2, 0.1: 1, 0.3: 2, 1.0: 4 | 0/14 | 4/14 |

DEC-16 and grid checks: PASS.


## Complete likelihood contrasts versus reference

| Contrast | Run | Weighted bits/coda | Unweighted day mean | 95% CI | Positive days |
|---|---|---:|---:|---|---:|
| own_history | reference | 1.34245 | 1.20798 | [0.9026, 1.7584] | 14/14 |
| own_history | v4 | 1.34369 | 1.25568 | [0.9445, 1.7317] | 14/14 |
| partner_content | reference | -0.03196 | -0.01506 | [-0.0927, 0.0361] | 5/14 |
| partner_content | v4 | -0.02566 | -0.00144 | [-0.0942, 0.0635] | 6/14 |
| timing | reference | -0.05012 | -0.03829 | [-0.0816, -0.0142] | 5/14 |
| timing | v4 | -0.02024 | -0.00885 | [-0.0464, 0.0014] | 7/14 |
| add_input:S-none | reference | 0.07882 | 0.16250 | [-0.0626, 0.3564] | 8/14 |
| add_input:S-none | v4 | 0.04866 | 0.12024 | [-0.0804, 0.2992] | 6/14 |
| add_input:A-none | reference | 1.50564 | 1.59874 | [1.1967, 1.9137] | 14/14 |
| add_input:A-none | v4 | 1.50564 | 1.59874 | [1.1967, 1.9137] | 14/14 |
| add_input:B-none | reference | 0.12976 | 0.34451 | [-0.2908, 0.6998] | 10/14 |
| add_input:B-none | v4 | 0.12976 | 0.34452 | [-0.2908, 0.6998] | 10/14 |
| add_input:A_B-A | reference | -0.01547 | 0.00720 | [-0.0747, 0.0665] | 7/14 |
| add_input:A_B-A | v4 | -0.01547 | 0.00720 | [-0.0747, 0.0665] | 7/14 |
| add_input:A_B-B | reference | 1.36041 | 1.26143 | [0.9493, 1.7408] | 14/14 |
| add_input:A_B-B | v4 | 1.36041 | 1.26143 | [0.9493, 1.7408] | 14/14 |
| add_input:A_S-S | reference | 1.39319 | 1.42021 | [1.1145, 1.7287] | 14/14 |
| add_input:A_S-S | v4 | 1.44693 | 1.47829 | [1.1993, 1.7507] | 14/14 |
| add_input:A_S-A | reference | -0.03363 | -0.01603 | [-0.0620, 0.0119] | 4/14 |
| add_input:A_S-A | v4 | -0.01005 | -0.00021 | [-0.0248, 0.0037] | 7/14 |
| add_input:B_S-S | reference | 0.01878 | 0.19718 | [-0.3766, 0.4555] | 10/14 |
| add_input:B_S-S | v4 | 0.07758 | 0.22117 | [-0.2833, 0.4581] | 10/14 |
| add_input:B_S-B | reference | -0.03216 | 0.01517 | [-0.0997, 0.0470] | 7/14 |
| add_input:B_S-B | v4 | -0.00352 | -0.00311 | [-0.1085, 0.1404] | 9/14 |
| add_input:A_B_S-A_B | reference | -0.05012 | -0.03829 | [-0.0816, -0.0142] | 5/14 |
| add_input:A_B_S-A_B | v4 | -0.02024 | -0.00885 | [-0.0464, 0.0014] | 7/14 |
| add_input:A_B_S-A_S | reference | -0.03196 | -0.01506 | [-0.0927, 0.0361] | 5/14 |
| add_input:A_B_S-A_S | v4 | -0.02566 | -0.00144 | [-0.0942, 0.0635] | 6/14 |
| add_input:A_B_S-B_S | reference | 1.34245 | 1.20798 | [0.9026, 1.7584] | 14/14 |
| add_input:A_B_S-B_S | v4 | 1.34369 | 1.25568 | [0.9445, 1.7317] | 14/14 |
