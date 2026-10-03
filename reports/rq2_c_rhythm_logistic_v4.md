# RQ2 logistic-regression reference results, v4, inventory C

Fourteen leave-one-day-out folds (`data/processed/rq2_c_rhythm_folds/manifest.json`). Conditions containing A get a shared scalar added to the logit of the class matching the whale's own last coda, and conditions containing B get a shared scalar added to the logit of the class matching the partner's last coda, both penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] by the highest mean log-likelihood on that fold's validation day (ties go to the larger value), then the model trained on the training days only is scored on the test day, with no refit on train plus validation. Bootstrap replicates are 10,000. Inventory C: Sharma rhythm (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy, the model and its selection rule are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v4.md. With skip_nonconverged_candidates on, an L2 candidate that fails the convergence check is left out of the validation-day selection and counted below (DEC-22).

## Metrics, v4, inventory C

| Condition | LL bits/coda | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| none | -2.15209 | 61.926% | 0.0450 |
| S | -2.16882 | 61.345% | 0.0487 |
| A | -1.20748 | 78.410% | 0.2291 |
| B | -2.05256 | 65.223% | 0.0907 |
| A_B | -1.23267 | 77.957% | 0.2038 |
| A_S | -1.21169 | 78.539% | 0.2251 |
| B_S | -2.09691 | 63.736% | 0.0877 |
| A_B_S | -1.23585 | 77.957% | 0.2050 |
| majority | n/a | 61.926% | 0.0450 |
| repeat_own_last | -1.42054 | 78.539% | 0.3848 |

## Chosen L2 per condition (DEC-20: history and S penalties chosen separately)

| Condition | History min | History median | History max | History at grid edge | S min | S median | S max | S at grid edge |
|---|---:|---:|---:|---|---:|---:|---:|---|
| S | 0.0001 | 0.003 | 1.0 | 8/14 (test-2014-04-16, test-2015-04-10, test-2016-05-13, test-2015-04-07, test-2014-04-29, test-2015-03-19, test-2015-03-31, test-2015-03-26) | 0.0001 | 0.006500000000000001 | 1.0 | 3/14 |
| A | 0.0001 | 0.003 | 0.01 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| B | 0.0001 | 0.01 | 0.1 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| A_B | 0.0001 | 0.003 | 0.03 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| A_S | 0.0001 | 0.003 | 0.03 | 1/14 (test-2016-05-13) | 0.0001 | 0.3 | 1.0 | 6/14 |
| B_S | 0.0001 | 0.01 | 1.0 | 4/14 (test-2015-04-10, test-2016-05-13, test-2016-05-12, test-2016-04-12) | 0.0001 | 0.02 | 1.0 | 4/14 |
| A_B_S | 0.0001 | 0.003 | 0.03 | 1/14 (test-2016-05-13) | 0.0001 | 0.2 | 1.0 | 7/14 |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.0454 | 2.8086 | n/a | n/a |
| B | n/a | n/a | 0.5237 | 0.9752 |
| A_B | 1.7948 | 2.7224 | 0.4615 | 0.8079 |
| A_S | 1.8393 | 2.8129 | n/a | n/a |
| B_S | n/a | n/a | 0.1207 | 1.0614 |
| A_B_S | 1.7439 | 2.8044 | 0.4482 | 0.8042 |

## A-conditions vs repeat-own-last, v4, inventory C

Repeat-own-last: 78.539% accuracy, 0.3848 macro-F1, -1.42054 bits/coda.

| Condition | LL contrast | CI | +days | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|---:|
| A | 0.21307 | [0.1638, 0.2877] | 14/14 | 78.410% | 0.2291 |
| A_B | 0.18787 | [0.0927, 0.2976] | 12/14 | 77.957% | 0.2038 |
| A_S | 0.20886 | [0.1600, 0.2809] | 14/14 | 78.539% | 0.2251 |
| A_B_S | 0.18470 | [0.0887, 0.2936] | 12/14 | 77.957% | 0.2050 |

## All twelve add-one-input contrasts (v4, inventory C)

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | -0.01674 | -0.00647 | [-0.0485, 0.0216] | 6/14 |
| adding A to no inputs | 0.94461 | 1.00025 | [0.6806, 1.2047] | 14/14 |
| adding B to no inputs | 0.09952 | 0.19889 | [-0.1195, 0.3332] | 10/14 |
| adding B to A | -0.02520 | -0.00951 | [-0.0873, 0.0336] | 8/14 |
| adding A to B | 0.81989 | 0.79185 | [0.5188, 1.1634] | 14/14 |
| adding A to S | 0.95714 | 0.98850 | [0.6920, 1.2182] | 14/14 |
| adding S to A | -0.00421 | -0.01822 | [-0.0198, 0.0054] | 8/14 |
| adding B to S | 0.07191 | 0.12405 | [-0.1334, 0.2785] | 10/14 |
| adding S to B | -0.04435 | -0.08131 | [-0.0899, -0.0188] | 5/14 |
| adding S to A and B | -0.00317 | -0.01248 | [-0.0137, 0.0046] | 7/14 |
| adding B to A and S | -0.02416 | -0.00377 | [-0.0859, 0.0361] | 7/14 |
| adding A to B and S | 0.86106 | 0.86068 | [0.5594, 1.2086] | 14/14 |

## Main contrasts, v4, inventory C

| Contrast | bits/coda | Unweighted | CI | +days | Repeat part | Switch part |
|---|---:|---:|---|---:|---:|---:|
| own_history | 0.86106 | 0.86068 | [0.5594, 1.2086] | 14/14 | 0.76988 | 0.09118 |
| partner_content | -0.02416 | -0.00377 | [-0.0859, 0.0361] | 7/14 | -0.02395 | -0.00021 |
| timing | -0.00317 | -0.01248 | [-0.0137, 0.0046] | 7/14 | -0.00274 | -0.00043 |

4914 fits were run, across 14 folds and 7 non-unigram conditions: conditions without S search 9 history L2 values, conditions with S jointly search 9 history values times 9 S values (DEC-20). With skip_nonconverged_candidates on (DEC-22), a candidate that fails fit()'s convergence check is left out of the validation-day selection; 0 of the 4914 did, and every selected model converged. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

No candidate failed the convergence check in any fold or condition.

## Stop: results need review before interpretation

The following stop conditions from the task were met. No interpretation or conclusions are written below; the tables above are the complete result.

- add_input:B_S-B: contrast CI entirely below zero [-0.0899, -0.0188]

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 332 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq2_c_rhythm_folds.md reports UNK rates per fold.
