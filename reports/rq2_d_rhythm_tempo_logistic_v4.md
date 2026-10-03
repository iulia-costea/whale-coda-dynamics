# RQ2 logistic-regression reference results, v4, inventory D

Fourteen leave-one-day-out folds (`data/processed/rq2_d_rhythm_tempo_folds/manifest.json`). Conditions containing A get a shared scalar added to the logit of the class matching the whale's own last coda, and conditions containing B get a shared scalar added to the logit of the class matching the partner's last coda, both penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] by the highest mean log-likelihood on that fold's validation day (ties go to the larger value), then the model trained on the training days only is scored on the test day, with no refit on train plus validation. Bootstrap replicates are 10,000. Inventory D: Sharma rhythm × tempo (KDE cuts) (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy, the model and its selection rule are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v4.md. With skip_nonconverged_candidates on, an L2 candidate that fails the convergence check is left out of the validation-day selection and counted below (DEC-22).

## Metrics, v4, inventory D

| Condition | LL bits/coda | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| none | -3.68461 | 39.948% | 0.0178 |
| S | -3.83344 | 38.268% | 0.0212 |
| A | -1.69981 | 73.562% | 0.2854 |
| B | -3.11125 | 48.610% | 0.0585 |
| A_B | -1.72460 | 72.786% | 0.2585 |
| A_S | -1.70237 | 73.497% | 0.2847 |
| B_S | -3.14666 | 48.158% | 0.0595 |
| A_B_S | -1.73282 | 72.980% | 0.2652 |
| majority | n/a | 39.948% | 0.0178 |
| repeat_own_last | -2.12358 | 74.208% | 0.3358 |

## Chosen L2 per condition (DEC-20: history and S penalties chosen separately)

| Condition | History min | History median | History max | History at grid edge | S min | S median | S max | S at grid edge |
|---|---:|---:|---:|---|---:|---:|---:|---|
| S | 0.0001 | 0.051500000000000004 | 1.0 | 8/14 (test-2016-05-13, test-2014-04-24, test-2015-04-07, test-2014-04-29, test-2015-03-19, test-2016-04-24, test-2015-03-02, test-2015-03-31) | 0.0001 | 0.03 | 1.0 | 4/14 |
| A | 0.0001 | 0.001 | 0.03 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| B | 0.0001 | 0.003 | 0.1 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| A_B | 0.0001 | 0.003 | 0.03 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| A_S | 0.0001 | 0.001 | 0.03 | 1/14 (test-2016-05-13) | 0.001 | 1.0 | 1.0 | 10/14 |
| B_S | 0.0001 | 0.003 | 0.1 | 1/14 (test-2016-05-13) | 0.0001 | 0.2 | 1.0 | 6/14 |
| A_B_S | 0.0001 | 0.003 | 0.03 | 1/14 (test-2016-05-13) | 0.0001 | 0.65 | 1.0 | 8/14 |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.8268 | 3.3371 | n/a | n/a |
| B | n/a | n/a | 1.2016 | 1.6615 |
| A_B | 2.3818 | 2.7941 | 0.7619 | 1.0851 |
| A_S | 2.8260 | 3.3368 | n/a | n/a |
| B_S | n/a | n/a | 1.0408 | 1.7506 |
| A_B_S | 2.3812 | 2.9322 | 0.7656 | 1.1772 |

## A-conditions vs repeat-own-last, v4, inventory D

Repeat-own-last: 74.208% accuracy, 0.3358 macro-F1, -2.12358 bits/coda.

| Condition | LL contrast | CI | +days | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|---:|
| A | 0.42377 | [0.3526, 0.5332] | 13/14 | 73.562% | 0.2854 |
| A_B | 0.39899 | [0.3023, 0.5540] | 13/14 | 72.786% | 0.2585 |
| A_S | 0.42122 | [0.3503, 0.5296] | 13/14 | 73.497% | 0.2847 |
| A_B_S | 0.39077 | [0.2891, 0.5484] | 13/14 | 72.980% | 0.2652 |

## All twelve add-one-input contrasts (v4, inventory D)

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | -0.14883 | -0.04508 | [-0.4447, 0.1157] | 4/14 |
| adding A to no inputs | 1.98479 | 2.02461 | [1.7639, 2.2831] | 14/14 |
| adding B to no inputs | 0.57335 | 0.74881 | [0.2486, 1.0170] | 12/14 |
| adding B to A | -0.02478 | 0.02639 | [-0.0706, 0.0531] | 6/14 |
| adding A to B | 1.38666 | 1.30218 | [1.1007, 1.6433] | 14/14 |
| adding A to S | 2.13107 | 2.06719 | [1.8634, 2.4085] | 14/14 |
| adding S to A | -0.00255 | -0.00250 | [-0.0053, -0.0002] | 4/14 |
| adding B to S | 0.68677 | 0.73346 | [0.3032, 1.0665] | 12/14 |
| adding S to B | -0.03541 | -0.06043 | [-0.0861, -0.0049] | 6/14 |
| adding S to A and B | -0.00822 | -0.00685 | [-0.0176, 0.0015] | 6/14 |
| adding B to A and S | -0.03045 | 0.02203 | [-0.0819, 0.0509] | 6/14 |
| adding A to B and S | 1.41385 | 1.35576 | [1.1414, 1.6583] | 14/14 |

## Main contrasts, v4, inventory D

| Contrast | bits/coda | Unweighted | CI | +days | Repeat part | Switch part |
|---|---:|---:|---|---:|---:|---:|
| own_history | 1.41385 | 1.35576 | [1.1414, 1.6583] | 14/14 | 1.23800 | 0.17585 |
| partner_content | -0.03045 | 0.02203 | [-0.0819, 0.0509] | 6/14 | -0.02105 | -0.00940 |
| timing | -0.00822 | -0.00685 | [-0.0176, 0.0015] | 6/14 | -0.00431 | -0.00391 |

4914 fits were run, across 14 folds and 7 non-unigram conditions: conditions without S search 9 history L2 values, conditions with S jointly search 9 history values times 9 S values (DEC-20). With skip_nonconverged_candidates on (DEC-22), a candidate that fails fit()'s convergence check is left out of the validation-day selection; 0 of the 4914 did, and every selected model converged. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

No candidate failed the convergence check in any fold or condition.

## Stop: results need review before interpretation

The following stop conditions from the task were met. No interpretation or conclusions are written below; the tables above are the complete result.

- add_input:A_S-A: contrast CI entirely below zero [-0.0053, -0.0002]
- add_input:B_S-B: contrast CI entirely below zero [-0.0861, -0.0049]

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 405 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq2_d_rhythm_tempo_folds.md reports UNK rates per fold.
