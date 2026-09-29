# RQ1 logistic-regression reference results, v4

Same fourteen leave-one-day-out folds as v3. Conditions containing A get a shared scalar added to the logit of the class matching the whale's own last coda, and conditions containing B get a shared scalar added to the logit of the class matching the partner's last coda, both penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] by the highest mean log-likelihood on that fold's validation day (ties go to the larger value), then the model trained on the training days only is scored on the test day, with no refit on train plus validation. Bootstrap replicates are 10,000. Preprocessing, folds, the UNK policy and the S definition are unchanged; v3 results are kept for comparison, not overwritten.

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

## Chosen L2 per condition (DEC-20: history and S penalties chosen separately)

| Condition | History min | History median | History max | History at grid edge | S min | S median | S max | S at grid edge |
|---|---:|---:|---:|---|---:|---:|---:|---|
| S | 0.0001 | 0.002 | 1.0 | 10/14 (test-2016-05-13, test-2014-04-24, test-2015-04-07, test-2016-05-12, test-2014-04-29, test-2015-03-19, test-2016-04-24, test-2015-03-02, test-2015-03-31, test-2015-03-26) | 0.0001 | 0.006500000000000001 | 1.0 | 4/14 |
| A | 0.0001 | 0.001 | 0.03 | 3/14 (test-2016-05-13, test-2014-04-24, test-2015-03-19) | n/a | n/a | n/a | n/a |
| B | 0.0001 | 0.003 | 0.3 | 2/14 (test-2016-05-13, test-2015-03-19) | n/a | n/a | n/a | n/a |
| A_B | 0.0001 | 0.001 | 0.01 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| A_S | 0.0001 | 0.001 | 0.03 | 3/14 (test-2016-05-13, test-2014-04-24, test-2015-03-19) | 0.0003 | 0.16499999999999998 | 1.0 | 6/14 |
| B_S | 0.0001 | 0.006500000000000001 | 1.0 | 5/14 (test-2015-04-10, test-2015-04-07, test-2016-05-12, test-2015-03-19, test-2016-04-12) | 0.0001 | 0.03 | 1.0 | 5/14 |
| A_B_S | 0.0001 | 0.001 | 0.01 | 1/14 (test-2016-05-13) | 0.0003 | 0.065 | 1.0 | 4/14 |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.3813 | 3.4481 | n/a | n/a |
| B | n/a | n/a | 0.7672 | 1.7782 |
| A_B | 2.3328 | 3.0183 | 0.7258 | 1.3088 |
| A_S | 2.3773 | 3.4469 | n/a | n/a |
| B_S | n/a | n/a | 0.1497 | 1.7799 |
| A_B_S | 2.3320 | 3.0179 | 0.7463 | 1.3097 |

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
| adding B to no inputs | 0.12976 | 0.34451 | [-0.2908, 0.6998] | 10/14 |
| adding B to A | -0.01547 | 0.00720 | [-0.0747, 0.0665] | 7/14 |
| adding A to B | 1.36041 | 1.26143 | [0.9493, 1.7408] | 14/14 |
| adding A to S | 1.44693 | 1.47828 | [1.1993, 1.7507] | 14/14 |
| adding S to A | -0.01005 | -0.00021 | [-0.0248, 0.0037] | 7/14 |
| adding B to S | 0.07758 | 0.22116 | [-0.2833, 0.4581] | 10/14 |
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

All 4914 fits converged, across 14 folds and 7 non-unigram conditions: conditions without S search 9 history L2 values, conditions with S jointly search 9 history values times 9 S values (DEC-20). fit() raises immediately on non-convergence, so reaching this report already confirms it. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

## Result

No A-condition is reliably below repeat-own-last in log-likelihood, and no add-one-input contrast has a confidence interval entirely below zero.

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 270 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq1_folds.md reports UNK rates per fold.
