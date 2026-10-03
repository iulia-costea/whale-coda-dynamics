# RQ1 logistic-regression reference results, v3

Same fourteen leave-one-day-out folds as v2. Conditions containing A get a shared scalar added to the logit of the class matching the whale's own last coda, and conditions containing B get a shared scalar added to the logit of the class matching the partner's last coda, both penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] by the highest mean log-likelihood on that fold's validation day (ties go to the larger value), then the model trained on the training days only is scored on the test day, with no refit on train plus validation. Bootstrap replicates are 10,000. Preprocessing, folds, the UNK policy and the S definition are unchanged; v2 results are kept for comparison, not overwritten.

## Metrics, v2 vs v3

| Condition | v2 LL bits/coda | v3 LL bits/coda | v2 Accuracy | v3 Accuracy | v2 Macro-F1 | v3 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| none | -2.60254 | -2.60254 | 57.207% | 57.207% | 0.0364 | 0.0364 |
| S | -2.51416 | -2.52372 | 56.949% | 56.820% | 0.0438 | 0.0437 |
| A | -1.08791 | -1.09690 | 82.741% | 82.353% | 0.2926 | 0.2819 |
| B | -2.46488 | -2.47278 | 59.147% | 59.017% | 0.0730 | 0.0729 |
| A_B | -1.09819 | -1.11237 | 82.353% | 82.482% | 0.3277 | 0.3270 |
| A_S | -1.11937 | -1.13053 | 82.999% | 82.612% | 0.3297 | 0.3181 |
| B_S | -2.50527 | -2.50494 | 57.725% | 57.660% | 0.0670 | 0.0668 |
| A_B_S | -1.14421 | -1.16249 | 82.094% | 82.094% | 0.3428 | 0.3422 |
| majority | n/a | n/a | 57.207% | 57.207% | 0.0364 | 0.0364 |
| repeat_own_last | -1.39276 | -1.39276 | 82.547% | 82.547% | 0.4070 | 0.4070 |

## Chosen L2 per condition

| Condition | Min | Median | Max | Folds at grid edge |
|---|---:|---:|---:|---|
| S | 0.0001 | 0.003 | 1.0 | 3/14 (test-2014-04-16, test-2016-05-13, test-2016-04-24) |
| A | 0.0001 | 0.001 | 0.03 | 3/14 (test-2016-05-13, test-2014-04-24, test-2015-03-19) |
| B | 0.0001 | 0.003 | 0.3 | 2/14 (test-2016-05-13, test-2015-03-19) |
| A_B | 0.0001 | 0.001 | 0.01 | 1/14 (test-2016-05-13) |
| A_S | 0.0001 | 0.003 | 0.03 | 2/14 (test-2016-05-13, test-2014-04-24) |
| B_S | 0.0003 | 0.01 | 0.3 | 0/14 |
| A_B_S | 0.0001 | 0.002 | 0.01 | 1/14 (test-2016-05-13) |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.3813 | 3.4481 | n/a | n/a |
| B | n/a | n/a | 0.7672 | 1.7781 |
| A_B | 2.3328 | 3.0183 | 0.7258 | 1.3087 |
| A_S | 2.3311 | 3.3763 | n/a | n/a |
| B_S | n/a | n/a | 0.7207 | 1.5949 |
| A_B_S | 2.2878 | 3.1306 | 0.7251 | 1.4503 |

## A-conditions vs repeat-own-last, v2 vs v3

Repeat-own-last: 82.547% accuracy, 0.4070 macro-F1, -1.39276 bits/coda.

| Condition | v2 LL contrast | v2 CI | v2 +days | v3 LL contrast | v3 CI | v3 +days | v3 Accuracy | v3 Macro-F1 |
|---|---:|---|---:|---:|---|---:|---:|---:|
| A | 0.30485 | [0.2517, 0.3860] | 14/14 | 0.29586 | [0.2420, 0.3719] | 14/14 | 82.353% | 0.2819 |
| A_B | 0.29457 | [0.2356, 0.4292] | 14/14 | 0.28039 | [0.2158, 0.4092] | 13/14 | 82.482% | 0.3270 |
| A_S | 0.27339 | [0.2207, 0.3815] | 14/14 | 0.26223 | [0.2089, 0.3632] | 13/14 | 82.612% | 0.3181 |
| A_B_S | 0.24855 | [0.1693, 0.4116] | 14/14 | 0.23027 | [0.1502, 0.3838] | 13/14 | 82.094% | 0.3422 |

## All twelve add-one-input contrasts (v3)

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | 0.07882 | 0.16250 | [-0.0626, 0.3564] | 8/14 |
| adding A to no inputs | 1.50564 | 1.59874 | [1.1967, 1.9137] | 14/14 |
| adding B to no inputs | 0.12976 | 0.34451 | [-0.2908, 0.6998] | 10/14 |
| adding B to A | -0.01547 | 0.00720 | [-0.0747, 0.0665] | 7/14 |
| adding A to B | 1.36041 | 1.26143 | [0.9493, 1.7408] | 14/14 |
| adding A to S | 1.39319 | 1.42021 | [1.1145, 1.7287] | 14/14 |
| adding S to A | -0.03363 | -0.01603 | [-0.0620, 0.0119] | 4/14 |
| adding B to S | 0.01878 | 0.19718 | [-0.3766, 0.4555] | 10/14 |
| adding S to B | -0.03216 | 0.01517 | [-0.0997, 0.0470] | 7/14 |
| adding S to A and B | -0.05012 | -0.03829 | [-0.0816, -0.0142] | 5/14 |
| adding B to A and S | -0.03196 | -0.01506 | [-0.0927, 0.0361] | 5/14 |
| adding A to B and S | 1.34245 | 1.20798 | [0.9026, 1.7584] | 14/14 |

## Main contrasts, v2 vs v3

| Contrast | v2 bits/coda | v3 bits/coda | v3 unweighted | v3 CI | v3 +days | v3 Repeat part | v3 Switch part |
|---|---:|---:|---:|---|---:|---:|---:|
| own_history | 1.36105 | 1.34245 | 1.20798 | [0.9026, 1.7584] | 14/14 | 1.23959 | 0.10286 |
| partner_content | -0.02485 | -0.03196 | -0.01506 | [-0.0927, 0.0361] | 5/14 | -0.03089 | -0.00106 |
| timing | -0.04603 | -0.05012 | -0.03829 | [-0.0816, -0.0142] | 5/14 | -0.03020 | -0.01992 |

All 882 fits (14 folds times 7 non-unigram conditions times 9 grid values) converged; fit() raises immediately on non-convergence, so reaching this report already confirms it. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

## Stop: results need review before interpretation

The following stop conditions from the task were met. No interpretation or conclusions are written below; the tables above are the complete result.

- add_input:A_B_S-A_B: contrast CI entirely below zero [-0.0816, -0.0142]

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 270 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq1_folds.md reports UNK rates per fold.
