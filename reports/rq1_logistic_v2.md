# RQ1 logistic-regression reference results, v2 (copy features and tuned L2)

Same fourteen leave-one-day-out folds as v1. Two changes to the reference model only: conditions containing A get a shared scalar added to the logit of the class matching the whale's own last coda, and conditions containing B get a shared scalar added to the logit of the class matching the partner's last coda, both penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] by the highest mean log-likelihood on that fold's validation day (ties go to the larger value), then the model trained on the training days only is scored on the test day, with no refit on train plus validation. Bootstrap replicates are 10,000. Preprocessing, folds, the UNK policy and the S definition are unchanged; v1 results are kept for comparison, not overwritten.

## Metrics, v1 vs v2

| Condition | v1 LL bits/coda | v2 LL bits/coda | v1 Accuracy | v2 Accuracy | v1 Macro-F1 | v2 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| none | -2.60254 | -2.60254 | 57.207% | 57.207% | 0.0364 | 0.0364 |
| S | -2.54635 | -2.51416 | 56.884% | 56.949% | 0.0404 | 0.0438 |
| A | -1.36275 | -1.08791 | 79.186% | 82.741% | 0.1821 | 0.2926 |
| B | -2.42974 | -2.46488 | 60.633% | 59.147% | 0.0594 | 0.0730 |
| A_B | -1.38930 | -1.09819 | 77.957% | 82.353% | 0.1677 | 0.3277 |
| A_S | -1.38304 | -1.11937 | 78.410% | 82.999% | 0.1733 | 0.3297 |
| B_S | -2.43562 | -2.50527 | 58.953% | 57.725% | 0.0580 | 0.0670 |
| A_B_S | -1.40876 | -1.14421 | 76.665% | 82.094% | 0.1596 | 0.3428 |
| majority | n/a | n/a | 57.207% | 57.207% | 0.0364 | 0.0364 |
| repeat_own_last | -1.39276 | -1.39276 | 82.547% | 82.547% | 0.4070 | 0.4070 |

## Chosen L2 per condition

| Condition | Min | Median | Max | Folds at grid edge |
|---|---:|---:|---:|---|
| S | 0.001 | 0.003 | 1.0 | 3/14 (test-2014-04-16, test-2016-05-13, test-2016-04-24) |
| A | 0.001 | 0.001 | 0.03 | 10/14 (test-2014-04-16, test-2015-04-01, test-2016-05-13, test-2014-04-24, test-2015-04-07, test-2014-04-29, test-2015-03-19, test-2016-04-12, test-2015-03-02, test-2015-03-26) |
| B | 0.001 | 0.003 | 0.3 | 6/14 (test-2014-04-16, test-2016-05-13, test-2014-04-24, test-2014-04-29, test-2015-03-19, test-2015-03-02) |
| A_B | 0.001 | 0.001 | 0.01 | 8/14 (test-2014-04-16, test-2015-04-01, test-2016-05-13, test-2014-04-24, test-2015-04-07, test-2014-04-29, test-2015-03-19, test-2015-03-02) |
| A_S | 0.001 | 0.003 | 0.03 | 6/14 (test-2014-04-16, test-2015-04-01, test-2016-05-13, test-2014-04-24, test-2015-04-07, test-2015-03-26) |
| B_S | 0.001 | 0.01 | 0.3 | 3/14 (test-2016-05-13, test-2014-04-24, test-2015-03-02) |
| A_B_S | 0.001 | 0.002 | 0.01 | 7/14 (test-2014-04-16, test-2015-04-01, test-2016-05-13, test-2014-04-24, test-2015-04-07, test-2015-03-19, test-2015-03-02) |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.3813 | 2.8023 | n/a | n/a |
| B | n/a | n/a | 0.7672 | 1.4364 |
| A_B | 2.3328 | 2.5570 | 0.7258 | 1.0098 |
| A_S | 2.3311 | 2.7847 | n/a | n/a |
| B_S | n/a | n/a | 0.7207 | 1.4342 |
| A_B_S | 2.2878 | 2.5768 | 0.7251 | 1.0311 |

## A-conditions vs repeat-own-last, v1 vs v2

Repeat-own-last: 82.547% accuracy, 0.4070 macro-F1, -1.39276 bits/coda.

| Condition | v1 LL contrast | v1 CI | v1 +days | v2 LL contrast | v2 CI | v2 +days | v2 Accuracy | v2 Macro-F1 |
|---|---:|---|---:|---:|---|---:|---:|---:|
| A | 0.03001 | [-0.039842224985290546, 0.1405069988629949] | 9/14 | 0.30485 | [0.25168448079227507, 0.3860045653312211] | 14/14 | 82.741% | 0.2926 |
| A_B | 0.00346 | [-0.07759767634400976, 0.13763424134872349] | 9/14 | 0.29457 | [0.2355514989194937, 0.429189356111791] | 14/14 | 82.353% | 0.3277 |
| A_S | 0.00972 | [-0.06458697071716024, 0.12294957497038707] | 9/14 | 0.27339 | [0.22069359603288488, 0.38153289887548425] | 14/14 | 82.999% | 0.3297 |
| A_B_S | -0.01600 | [-0.10462822780879909, 0.12276618896356559] | 9/14 | 0.24855 | [0.16930690836264572, 0.41155823053676477] | 14/14 | 82.094% | 0.3428 |

## All twelve add-one-input contrasts (v2)

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | 0.08838 | 0.18690 | [-0.04411, 0.36846] | 8/14 |
| adding A to no inputs | 1.51463 | 1.63108 | [1.21242, 1.92514] | 14/14 |
| adding B to no inputs | 0.13766 | 0.37276 | [-0.27779, 0.71062] | 10/14 |
| adding B to A | -0.01028 | 0.02583 | [-0.06765, 0.07230] | 7/14 |
| adding A to B | 1.36669 | 1.28414 | [0.95836, 1.74782] | 14/14 |
| adding A to S | 1.39480 | 1.43607 | [1.11865, 1.72687] | 14/14 |
| adding S to A | -0.03146 | -0.00812 | [-0.05981, 0.01453] | 3/14 |
| adding B to S | 0.00890 | 0.17223 | [-0.38584, 0.43860] | 10/14 |
| adding S to B | -0.04039 | -0.01363 | [-0.10609, 0.03266] | 7/14 |
| adding S to A and B | -0.04603 | -0.02706 | [-0.07721, -0.00927] | 5/14 |
| adding B to A and S | -0.02485 | 0.00689 | [-0.08554, 0.04473] | 7/14 |
| adding A to B and S | 1.36105 | 1.27072 | [0.92965, 1.77484] | 14/14 |

## Main contrasts, v1 vs v2

| Contrast | v1 bits/coda | v2 bits/coda | v2 unweighted | v2 CI | v2 +days | v2 Repeat part | v2 Switch part |
|---|---:|---:|---:|---|---:|---:|---:|
| own_history | 1.02687 | 1.36105 | 1.27072 | [0.9296544434079919, 1.7748439533346758] | 14/14 | 1.24732 | 0.11374 |
| partner_content | -0.02572 | -0.02485 | 0.00689 | [-0.08553862787234563, 0.04473176389624954] | 7/14 | -0.02884 | 0.00399 |
| timing | -0.01946 | -0.04603 | -0.02706 | [-0.0772124079144048, -0.009267742881333444] | 5/14 | -0.03022 | -0.01581 |

All 686 fits (14 folds times 7 non-unigram conditions times 7 grid values) converged; fit() raises immediately on non-convergence, so reaching this report already confirms it. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

## Stop: results need review before interpretation

The following stop conditions from the task were met. No interpretation or conclusions are written below; the tables above are the complete result.

- A_B: accuracy 82.353% below repeat-own-last 82.547%
- A_B_S: accuracy 82.094% below repeat-own-last 82.547%
- add_input:A_B_S-A_B: contrast CI entirely below zero [-0.0772124079144048, -0.009267742881333444]

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 270 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq1_folds.md reports UNK rates per fold.
