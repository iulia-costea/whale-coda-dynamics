# RQ2 logistic-regression reference results, v4, inventory E

Fourteen leave-one-day-out folds (`data/processed/rq2_e_type_tempo_folds/manifest.json`). Conditions containing A get a shared scalar added to the logit of the class matching the whale's own last coda, and conditions containing B get a shared scalar added to the logit of the class matching the partner's last coda, both penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] by the highest mean log-likelihood on that fold's validation day (ties go to the larger value), then the model trained on the training days only is scored on the test day, with no refit on train plus validation. Bootstrap replicates are 10,000. Inventory E: guy_coda_type × tempo (KDE cuts) (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy, the model and its selection rule are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v4.md. With skip_nonconverged_candidates on, an L2 candidate that fails the convergence check is left out of the validation-day selection and counted below (DEC-22).

## Metrics, v4, inventory E

| Condition | LL bits/coda | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| none | -3.56101 | 39.884% | 0.0173 |
| S | -3.70003 | 37.557% | 0.0216 |
| A | -1.53773 | 77.505% | 0.3056 |
| B | -3.04276 | 53.264% | 0.0851 |
| A_B | -1.57978 | 75.178% | 0.2510 |
| A_S | -1.54019 | 77.246% | 0.3007 |
| B_S | -3.08437 | 49.580% | 0.0678 |
| A_B_S | -1.57882 | 75.372% | 0.2471 |
| majority | n/a | 39.884% | 0.0173 |
| repeat_own_last | -1.90389 | 78.087% | 0.3339 |

## Chosen L2 per condition (DEC-20: history and S penalties chosen separately)

| Condition | History min | History median | History max | History at grid edge | S min | S median | S max | S at grid edge |
|---|---:|---:|---:|---|---:|---:|---:|---|
| S | 0.0001 | 0.002 | 1.0 | 8/14 (test-2014-04-16, test-2016-05-13, test-2014-04-24, test-2015-04-07, test-2015-03-19, test-2016-04-24, test-2015-03-02, test-2015-03-31) | 0.0001 | 0.03 | 1.0 | 4/14 |
| A | 0.0001 | 0.001 | 0.03 | 3/14 (test-2016-05-13, test-2014-04-24, test-2015-03-19) | n/a | n/a | n/a | n/a |
| B | 0.0001 | 0.001 | 0.1 | 2/14 (test-2016-05-13, test-2015-03-19) | n/a | n/a | n/a | n/a |
| A_B | 0.0001 | 0.001 | 0.03 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| A_S | 0.0001 | 0.001 | 0.03 | 3/14 (test-2016-05-13, test-2014-04-24, test-2015-03-19) | 0.003 | 1.0 | 1.0 | 11/14 |
| B_S | 0.0001 | 0.002 | 0.3 | 2/14 (test-2016-05-13, test-2015-03-19) | 0.0001 | 0.1 | 1.0 | 6/14 |
| A_B_S | 0.0001 | 0.001 | 0.03 | 1/14 (test-2016-05-13) | 0.001 | 1.0 | 1.0 | 8/14 |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.8502 | 3.3980 | n/a | n/a |
| B | n/a | n/a | 1.3307 | 1.8547 |
| A_B | 2.4640 | 2.7417 | 0.8542 | 1.4056 |
| A_S | 2.8496 | 3.3998 | n/a | n/a |
| B_S | n/a | n/a | 0.6557 | 1.9416 |
| A_B_S | 2.4621 | 2.7387 | 0.8538 | 1.4059 |

## A-conditions vs repeat-own-last, v4, inventory E

Repeat-own-last: 78.087% accuracy, 0.3339 macro-F1, -1.90389 bits/coda.

| Condition | LL contrast | CI | +days | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|---:|
| A | 0.36616 | [0.3122, 0.4329] | 13/14 | 77.505% | 0.3056 |
| A_B | 0.32411 | [0.2295, 0.4810] | 13/14 | 75.178% | 0.2510 |
| A_S | 0.36371 | [0.3110, 0.4273] | 13/14 | 77.246% | 0.3007 |
| A_B_S | 0.32508 | [0.2304, 0.4817] | 13/14 | 75.372% | 0.2471 |

## All twelve add-one-input contrasts (v4, inventory E)

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | -0.13903 | -0.03781 | [-0.4522, 0.1498] | 4/14 |
| adding A to no inputs | 2.02327 | 2.05150 | [1.8107, 2.2974] | 14/14 |
| adding B to no inputs | 0.51825 | 0.81338 | [0.0859, 1.0774] | 12/14 |
| adding B to A | -0.04205 | 0.03007 | [-0.1389, 0.0835] | 8/14 |
| adding A to B | 1.46298 | 1.26819 | [1.1189, 1.7951] | 14/14 |
| adding A to S | 2.15985 | 2.08586 | [1.8816, 2.4391] | 14/14 |
| adding S to A | -0.00245 | -0.00345 | [-0.0064, -0.0001] | 5/14 |
| adding B to S | 0.61566 | 0.74790 | [0.1172, 1.0906] | 12/14 |
| adding S to B | -0.04161 | -0.10329 | [-0.1562, 0.0109] | 8/14 |
| adding S to A and B | 0.00097 | 0.00141 | [-0.0023, 0.0057] | 8/14 |
| adding B to A and S | -0.03863 | 0.03494 | [-0.1356, 0.0862] | 8/14 |
| adding A to B and S | 1.50555 | 1.37290 | [1.1616, 1.8381] | 14/14 |

## Main contrasts, v4, inventory E

| Contrast | bits/coda | Unweighted | CI | +days | Repeat part | Switch part |
|---|---:|---:|---|---:|---:|---:|
| own_history | 1.50555 | 1.37290 | [1.1616, 1.8381] | 14/14 | 1.32867 | 0.17689 |
| partner_content | -0.03863 | 0.03494 | [-0.1356, 0.0862] | 8/14 | -0.05908 | 0.02045 |
| timing | 0.00097 | 0.00141 | [-0.0023, 0.0057] | 8/14 | 0.00034 | 0.00063 |

4914 fits were run, across 14 folds and 7 non-unigram conditions: conditions without S search 9 history L2 values, conditions with S jointly search 9 history values times 9 S values (DEC-20). With skip_nonconverged_candidates on (DEC-22), a candidate that fails fit()'s convergence check is left out of the validation-day selection; 0 of the 4914 did, and every selected model converged. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

No candidate failed the convergence check in any fold or condition.

## Stop: results need review before interpretation

The following stop conditions from the task were met. No interpretation or conclusions are written below; the tables above are the complete result.

- add_input:A_S-A: contrast CI entirely below zero [-0.0064, -0.0001]

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 343 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq2_e_type_tempo_folds.md reports UNK rates per fold.
