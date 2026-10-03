# RQ2 logistic-regression reference results, v4, inventory D′

Fourteen leave-one-day-out folds (`data/processed/rq2_d_rhythm_tempo_tdict_folds/manifest.json`). Conditions containing A get a shared scalar added to the logit of the class matching the whale's own last coda, and conditions containing B get a shared scalar added to the logit of the class matching the partner's last coda, both penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] by the highest mean log-likelihood on that fold's validation day (ties go to the larger value), then the model trained on the training days only is scored on the test day, with no refit on train plus validation. Bootstrap replicates are 10,000. Inventory D′: Sharma rhythm × tempo (tempos-dict.p cuts) (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy, the model and its selection rule are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v4.md. With skip_nonconverged_candidates on, an L2 candidate that fails the convergence check is left out of the validation-day selection and counted below (DEC-22).

## Metrics, v4, inventory D′

| Condition | LL bits/coda | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| none | -3.72492 | 39.496% | 0.0157 |
| S | -3.83483 | 37.880% | 0.0188 |
| A | -1.70989 | 75.436% | 0.2674 |
| B | -3.16085 | 48.158% | 0.0509 |
| A_B | -1.73488 | 73.368% | 0.2514 |
| A_S | -1.71337 | 75.048% | 0.2656 |
| B_S | -3.20006 | 47.705% | 0.0495 |
| A_B_S | -1.73971 | 73.239% | 0.2478 |
| majority | n/a | 39.496% | 0.0157 |
| repeat_own_last | -2.12127 | 74.467% | 0.3044 |

## Chosen L2 per condition (DEC-20: history and S penalties chosen separately)

| Condition | History min | History median | History max | History at grid edge | S min | S median | S max | S at grid edge |
|---|---:|---:|---:|---|---:|---:|---:|---|
| S | 0.0001 | 0.006500000000000001 | 1.0 | 8/14 (test-2015-04-01, test-2016-05-13, test-2015-04-07, test-2016-05-12, test-2014-04-29, test-2015-03-19, test-2016-04-24, test-2015-03-02) | 0.0001 | 0.03 | 1.0 | 4/14 |
| A | 0.0001 | 0.001 | 0.03 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| B | 0.0001 | 0.003 | 0.1 | 2/14 (test-2016-05-13, test-2016-05-12) | n/a | n/a | n/a | n/a |
| A_B | 0.0001 | 0.003 | 0.03 | 1/14 (test-2016-05-13) | n/a | n/a | n/a | n/a |
| A_S | 0.0001 | 0.001 | 0.03 | 1/14 (test-2016-05-13) | 0.0003 | 1.0 | 1.0 | 10/14 |
| B_S | 0.0001 | 0.003 | 0.1 | 2/14 (test-2016-05-13, test-2016-05-12) | 0.0001 | 0.2 | 1.0 | 7/14 |
| A_B_S | 0.0001 | 0.003 | 0.03 | 1/14 (test-2016-05-13) | 0.0001 | 0.65 | 1.0 | 8/14 |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.8552 | 3.3097 | n/a | n/a |
| B | n/a | n/a | 1.2914 | 1.7629 |
| A_B | 2.4478 | 2.8422 | 0.7986 | 1.0864 |
| A_S | 2.8625 | 3.3094 | n/a | n/a |
| B_S | n/a | n/a | 1.0313 | 1.8072 |
| A_B_S | 2.4473 | 2.8859 | 0.7997 | 1.1861 |

## A-conditions vs repeat-own-last, v4, inventory D′

Repeat-own-last: 74.467% accuracy, 0.3044 macro-F1, -2.12127 bits/coda.

| Condition | LL contrast | CI | +days | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|---:|
| A | 0.41138 | [0.3180, 0.5543] | 14/14 | 75.436% | 0.2674 |
| A_B | 0.38639 | [0.2774, 0.5595] | 14/14 | 73.368% | 0.2514 |
| A_S | 0.40790 | [0.3104, 0.5542] | 14/14 | 75.048% | 0.2656 |
| A_B_S | 0.38156 | [0.2661, 0.5597] | 14/14 | 73.239% | 0.2478 |

## All twelve add-one-input contrasts (v4, inventory D′)

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | -0.10991 | -0.03221 | [-0.3198, 0.1046] | 4/14 |
| adding A to no inputs | 2.01503 | 2.08467 | [1.7918, 2.3363] | 14/14 |
| adding B to no inputs | 0.56407 | 0.72940 | [0.2392, 1.0026] | 11/14 |
| adding B to A | -0.02498 | 0.00075 | [-0.0508, 0.0219] | 6/14 |
| adding A to B | 1.42598 | 1.35602 | [1.1248, 1.6933] | 14/14 |
| adding A to S | 2.12146 | 2.11552 | [1.8946, 2.3821] | 14/14 |
| adding S to A | -0.00348 | -0.00136 | [-0.0088, 0.0006] | 4/14 |
| adding B to S | 0.63477 | 0.70257 | [0.2870, 0.9818] | 12/14 |
| adding S to B | -0.03921 | -0.05905 | [-0.1020, -0.0034] | 7/14 |
| adding S to A and B | -0.00484 | 0.00420 | [-0.0144, 0.0076] | 8/14 |
| adding B to A and S | -0.02634 | 0.00632 | [-0.0534, 0.0232] | 6/14 |
| adding A to B and S | 1.46035 | 1.41927 | [1.1679, 1.7256] | 14/14 |

## Main contrasts, v4, inventory D′

| Contrast | bits/coda | Unweighted | CI | +days | Repeat part | Switch part |
|---|---:|---:|---|---:|---:|---:|
| own_history | 1.46035 | 1.41927 | [1.1679, 1.7256] | 14/14 | 1.29113 | 0.16921 |
| partner_content | -0.02634 | 0.00632 | [-0.0534, 0.0232] | 6/14 | -0.03086 | 0.00452 |
| timing | -0.00484 | 0.00420 | [-0.0144, 0.0076] | 8/14 | -0.00360 | -0.00123 |

4914 fits were run, across 14 folds and 7 non-unigram conditions: conditions without S search 9 history L2 values, conditions with S jointly search 9 history values times 9 S values (DEC-20). With skip_nonconverged_candidates on (DEC-22), a candidate that fails fit()'s convergence check is left out of the validation-day selection; 0 of the 4914 did, and every selected model converged. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

No candidate failed the convergence check in any fold or condition.

## Stop: results need review before interpretation

The following stop conditions from the task were met. No interpretation or conclusions are written below; the tables above are the complete result.

- add_input:B_S-B: contrast CI entirely below zero [-0.1020, -0.0034]

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 400 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq2_d_rhythm_tempo_tdict_folds.md reports UNK rates per fold.
