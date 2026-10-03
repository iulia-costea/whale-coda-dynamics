# RQ2 logistic-regression reference results, v4, inventory E′

Fourteen leave-one-day-out folds (`data/processed/rq2_e_type_tempo_tdict_folds/manifest.json`). Conditions containing A get a shared scalar added to the logit of the class matching the whale's own last coda, and conditions containing B get a shared scalar added to the logit of the class matching the partner's last coda, both penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] by the highest mean log-likelihood on that fold's validation day (ties go to the larger value), then the model trained on the training days only is scored on the test day, with no refit on train plus validation. Bootstrap replicates are 10,000. Inventory E′: guy_coda_type × tempo (tempos-dict.p cuts) (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy, the model and its selection rule are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v4.md. With skip_nonconverged_candidates on, an L2 candidate that fails the convergence check is left out of the validation-day selection and counted below (DEC-22).

## Metrics, v4, inventory E′

| Condition | LL bits/coda | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| none | -3.59777 | 39.431% | 0.0162 |
| S | -3.73306 | 36.910% | 0.0201 |
| A | -1.53485 | 79.444% | 0.2967 |
| B | -3.03745 | 52.489% | 0.0664 |
| A_B | -1.57736 | 76.406% | 0.2579 |
| A_S | -1.53877 | 78.862% | 0.2952 |
| B_S | -3.05593 | 50.549% | 0.0669 |
| A_B_S | -1.58067 | 76.018% | 0.2484 |
| majority | n/a | 39.431% | 0.0162 |
| repeat_own_last | -1.87884 | 78.539% | 0.3349 |

## Chosen L2 per condition (DEC-20: history and S penalties chosen separately)

| Condition | History min | History median | History max | History at grid edge | S min | S median | S max | S at grid edge |
|---|---:|---:|---:|---|---:|---:|---:|---|
| S | 0.0001 | 0.003 | 1.0 | 8/14 (test-2014-04-16, test-2016-05-13, test-2014-04-24, test-2015-04-07, test-2014-04-29, test-2015-03-19, test-2016-04-24, test-2015-03-02) | 0.0001 | 0.03 | 1.0 | 4/14 |
| A | 0.0001 | 0.001 | 0.03 | 3/14 (test-2016-05-13, test-2014-04-24, test-2015-03-19) | n/a | n/a | n/a | n/a |
| B | 0.0001 | 0.001 | 0.1 | 2/14 (test-2016-05-13, test-2015-03-19) | n/a | n/a | n/a | n/a |
| A_B | 0.0001 | 0.001 | 0.03 | 2/14 (test-2016-05-13, test-2015-03-19) | n/a | n/a | n/a | n/a |
| A_S | 0.0001 | 0.001 | 0.03 | 3/14 (test-2016-05-13, test-2014-04-24, test-2015-03-19) | 0.0003 | 1.0 | 1.0 | 9/14 |
| B_S | 0.0001 | 0.001 | 0.1 | 2/14 (test-2016-05-13, test-2015-03-19) | 0.0001 | 0.2 | 1.0 | 6/14 |
| A_B_S | 0.0001 | 0.001 | 0.03 | 2/14 (test-2016-05-13, test-2015-03-19) | 0.003 | 0.65 | 1.0 | 7/14 |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.9013 | 3.3809 | n/a | n/a |
| B | n/a | n/a | 1.3159 | 1.8890 |
| A_B | 2.4757 | 2.7941 | 0.8150 | 1.2896 |
| A_S | 2.9007 | 3.4115 | n/a | n/a |
| B_S | n/a | n/a | 1.0456 | 1.9181 |
| A_B_S | 2.4783 | 2.7912 | 0.8147 | 1.2900 |

## A-conditions vs repeat-own-last, v4, inventory E′

Repeat-own-last: 78.539% accuracy, 0.3349 macro-F1, -1.87884 bits/coda.

| Condition | LL contrast | CI | +days | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|---:|
| A | 0.34399 | [0.2847, 0.4390] | 13/14 | 79.444% | 0.2967 |
| A_B | 0.30147 | [0.1861, 0.4844] | 13/14 | 76.406% | 0.2579 |
| A_S | 0.34007 | [0.2799, 0.4375] | 13/14 | 78.862% | 0.2952 |
| A_B_S | 0.29817 | [0.1805, 0.4819] | 13/14 | 76.018% | 0.2484 |

## All twelve add-one-input contrasts (v4, inventory E′)

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | -0.13529 | -0.04159 | [-0.4565, 0.1612] | 6/14 |
| adding A to no inputs | 2.06292 | 2.11458 | [1.8426, 2.3636] | 14/14 |
| adding B to no inputs | 0.56032 | 0.83683 | [0.1589, 1.0801] | 12/14 |
| adding B to A | -0.04251 | 0.02169 | [-0.1181, 0.0556] | 6/14 |
| adding A to B | 1.46009 | 1.29945 | [1.1116, 1.7794] | 14/14 |
| adding A to S | 2.19429 | 2.15369 | [1.9170, 2.4899] | 14/14 |
| adding S to A | -0.00392 | -0.00249 | [-0.0088, 0.0000] | 5/14 |
| adding B to S | 0.67714 | 0.82360 | [0.2110, 1.1519] | 12/14 |
| adding S to B | -0.01847 | -0.05482 | [-0.0765, 0.0111] | 8/14 |
| adding S to A and B | -0.00330 | -0.00379 | [-0.0082, 0.0012] | 8/14 |
| adding B to A and S | -0.04190 | 0.02039 | [-0.1165, 0.0545] | 6/14 |
| adding A to B and S | 1.47526 | 1.35047 | [1.1322, 1.7934] | 14/14 |

## Main contrasts, v4, inventory E′

| Contrast | bits/coda | Unweighted | CI | +days | Repeat part | Switch part |
|---|---:|---:|---|---:|---:|---:|
| own_history | 1.47526 | 1.35047 | [1.1322, 1.7934] | 14/14 | 1.33532 | 0.13993 |
| partner_content | -0.04190 | 0.02039 | [-0.1165, 0.0545] | 6/14 | -0.06687 | 0.02498 |
| timing | -0.00330 | -0.00379 | [-0.0082, 0.0012] | 8/14 | -0.00209 | -0.00121 |

4914 fits were run, across 14 folds and 7 non-unigram conditions: conditions without S search 9 history L2 values, conditions with S jointly search 9 history values times 9 S values (DEC-20). With skip_nonconverged_candidates on (DEC-22), a candidate that fails fit()'s convergence check is left out of the validation-day selection; 0 of the 4914 did, and every selected model converged. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

No candidate failed the convergence check in any fold or condition.

## Result

No A-condition is reliably below repeat-own-last in log-likelihood, and no add-one-input contrast has a confidence interval entirely below zero.

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 337 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq2_e_type_tempo_tdict_folds.md reports UNK rates per fold.
