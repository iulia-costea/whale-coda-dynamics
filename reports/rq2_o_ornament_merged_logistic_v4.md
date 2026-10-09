# RQ2 logistic-regression reference results, v4, inventory O

Fourteen leave-one-day-out folds (`data/processed/rq2_o_ornament_merged_folds/manifest.json`). Conditions containing A get a shared scalar added to the logit of the class matching the whale's own last coda, and conditions containing B get a shared scalar added to the logit of the class matching the partner's last coda, both penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] by the highest mean log-likelihood on that fold's validation day (ties go to the larger value), then the model trained on the training days only is scored on the test day, with no refit on train plus validation. Bootstrap replicates are 10,000. Inventory O: guy_coda_type with Sharma-flagged ornamented codas relabelled to their neighbour's type (PLAN [P7], DEC-31). Only the labels differ from RQ1: targets, folds, S, the UNK policy, the model and its selection rule are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's; the comparison with inventory A is reports/rq1_ornament_merge.md. With skip_nonconverged_candidates on, an L2 candidate that fails the convergence check is left out of the validation-day selection and counted below.

## Metrics, v4, inventory O

| Condition | LL bits/coda | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| none | -2.54991 | 57.595% | 0.0365 |
| S | -2.49574 | 57.401% | 0.0438 |
| A | -0.94641 | 86.296% | 0.4440 |
| B | -2.45130 | 59.341% | 0.0725 |
| A_B | -0.96540 | 86.037% | 0.4242 |
| A_S | -0.95331 | 86.231% | 0.4138 |
| B_S | -2.47638 | 58.048% | 0.0678 |
| A_B_S | -0.98159 | 85.520% | 0.4242 |
| majority | n/a | 57.595% | 0.0365 |
| repeat_own_last | -1.13150 | 86.490% | 0.4743 |

## Chosen L2 per condition (DEC-20: history and S penalties chosen separately)

| Condition | History min | History median | History max | History at grid edge | S min | S median | S max | S at grid edge |
|---|---:|---:|---:|---|---:|---:|---:|---|
| S | 0.0001 | 0.01 | 1.0 | 7/14 (test-2016-05-13, test-2015-04-07, test-2016-05-12, test-2016-04-24, test-2015-03-02, test-2015-03-31, test-2015-03-26) | 0.0001 | 0.01 | 1.0 | 4/14 |
| A | 0.0001 | 0.001 | 0.03 | 2/14 (test-2016-05-13, test-2015-04-07) | n/a | n/a | n/a | n/a |
| B | 0.0001 | 0.01 | 0.3 | 2/14 (test-2016-05-13, test-2015-03-19) | n/a | n/a | n/a | n/a |
| A_B | 0.0001 | 0.002 | 0.03 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| A_S | 0.0001 | 0.001 | 0.03 | 1/14 (test-2016-05-13) | 0.0003 | 0.65 | 1.0 | 7/14 |
| B_S | 0.0001 | 0.01 | 1.0 | 6/14 (test-2015-04-10, test-2016-05-13, test-2015-04-07, test-2016-05-12, test-2015-03-19, test-2016-04-12) | 0.0001 | 0.02 | 1.0 | 5/14 |
| A_B_S | 0.0001 | 0.002 | 0.03 | 2/14 (test-2015-04-01, test-2016-05-13) | 0.0001 | 0.3 | 1.0 | 6/14 |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.6022 | 3.9398 | n/a | n/a |
| B | n/a | n/a | 0.7693 | 1.9483 |
| A_B | 2.4665 | 3.6096 | 0.6012 | 1.3581 |
| A_S | 2.5988 | 3.9388 | n/a | n/a |
| B_S | n/a | n/a | 0.1472 | 2.0157 |
| A_B_S | 2.4594 | 3.7258 | 0.5987 | 1.3582 |

## A-conditions vs repeat-own-last, v4, inventory O

Repeat-own-last: 86.490% accuracy, 0.4743 macro-F1, -1.13150 bits/coda.

| Condition | LL contrast | CI | +days | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|---:|
| A | 0.18510 | [0.1305, 0.2782] | 12/14 | 86.296% | 0.4440 |
| A_B | 0.16610 | [0.0843, 0.3040] | 12/14 | 86.037% | 0.4242 |
| A_S | 0.17820 | [0.1271, 0.2798] | 12/14 | 86.231% | 0.4138 |
| A_B_S | 0.14991 | [0.0603, 0.2992] | 12/14 | 85.520% | 0.4242 |

## All twelve add-one-input contrasts (v4, inventory O)

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | 0.05417 | 0.12721 | [-0.0742, 0.2994] | 6/14 |
| adding A to no inputs | 1.60351 | 1.72607 | [1.2824, 2.0176] | 14/14 |
| adding B to no inputs | 0.09861 | 0.31739 | [-0.3360, 0.6759] | 10/14 |
| adding B to A | -0.01899 | -0.00511 | [-0.0662, 0.0363] | 8/14 |
| adding A to B | 1.48590 | 1.40358 | [1.0325, 1.9163] | 14/14 |
| adding A to S | 1.54244 | 1.61088 | [1.2889, 1.8466] | 14/14 |
| adding S to A | -0.00690 | 0.01202 | [-0.0228, 0.0160] | 8/14 |
| adding B to S | 0.01936 | 0.12237 | [-0.3650, 0.3940] | 9/14 |
| adding S to B | -0.02508 | -0.06781 | [-0.1808, 0.1293] | 9/14 |
| adding S to A and B | -0.01619 | -0.00286 | [-0.0311, 0.0034] | 10/14 |
| adding B to A and S | -0.02829 | -0.01999 | [-0.0753, 0.0264] | 7/14 |
| adding A to B and S | 1.49478 | 1.46852 | [1.0500, 1.9438] | 14/14 |

## Main contrasts, v4, inventory O

| Contrast | bits/coda | Unweighted | CI | +days | Repeat part | Switch part |
|---|---:|---:|---|---:|---:|---:|
| own_history | 1.49478 | 1.46852 | [1.0500, 1.9438] | 14/14 | 1.43176 | 0.06302 |
| partner_content | -0.02829 | -0.01999 | [-0.0753, 0.0264] | 7/14 | -0.04208 | 0.01379 |
| timing | -0.01619 | -0.00286 | [-0.0311, 0.0034] | 10/14 | -0.01231 | -0.00388 |

4914 fits were run, across 14 folds and 7 non-unigram conditions: conditions without S search 9 history L2 values, conditions with S jointly search 9 history values times 9 S values (DEC-20). With skip_nonconverged_candidates on (DEC-22), a candidate that fails fit()'s convergence check is left out of the validation-day selection; 0 of the 4914 did, and every selected model converged. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

No candidate failed the convergence check in any fold or condition.

## Result

No A-condition is reliably below repeat-own-last in log-likelihood, and no add-one-input contrast has a confidence interval entirely below zero.

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 210 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq2_o_ornament_merged_folds.md reports UNK rates per fold.
