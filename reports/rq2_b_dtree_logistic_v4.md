# RQ2 logistic-regression reference results, v4, inventory B

Fourteen leave-one-day-out folds (`data/processed/rq2_b_dtree_folds/manifest.json`). Conditions containing A get a shared scalar added to the logit of the class matching the whale's own last coda, and conditions containing B get a shared scalar added to the logit of the class matching the partner's last coda, both penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] by the highest mean log-likelihood on that fold's validation day (ties go to the larger value), then the model trained on the training days only is scored on the test day, with no refit on train plus validation. Bootstrap replicates are 10,000. Inventory B: decision_tree_coda_type (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy, the model and its selection rule are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v4.md. With skip_nonconverged_candidates on, an L2 candidate that fails the convergence check is left out of the validation-day selection and counted below (DEC-22).

## Metrics, v4, inventory B

| Condition | LL bits/coda | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| none | -2.65044 | 56.949% | 0.0316 |
| S | -2.63183 | 56.755% | 0.0383 |
| A | -1.12205 | 82.224% | 0.2774 |
| B | -2.70155 | 58.112% | 0.0820 |
| A_B | -1.18298 | 81.125% | 0.2451 |
| A_S | -1.13239 | 81.771% | 0.2569 |
| B_S | -2.68881 | 56.884% | 0.0755 |
| A_B_S | -1.20064 | 81.254% | 0.2477 |
| majority | n/a | 56.949% | 0.0316 |
| repeat_own_last | -1.43264 | 82.741% | 0.3320 |

## Chosen L2 per condition (DEC-20: history and S penalties chosen separately)

| Condition | History min | History median | History max | History at grid edge | S min | S median | S max | S at grid edge |
|---|---:|---:|---:|---|---:|---:|---:|---|
| S | 0.0001 | 0.002 | 1.0 | 7/14 (test-2016-05-13, test-2014-04-24, test-2014-04-29, test-2015-03-19, test-2016-04-24, test-2015-03-02, test-2015-03-31) | 0.0001 | 0.02 | 1.0 | 6/14 |
| A | 0.0001 | 0.001 | 0.03 | 4/14 (test-2016-05-13, test-2014-04-24, test-2014-04-29, test-2015-03-19) | n/a | n/a | n/a | n/a |
| B | 0.0001 | 0.002 | 0.3 | 4/14 (test-2016-05-13, test-2016-05-12, test-2014-04-29, test-2015-03-19) | n/a | n/a | n/a | n/a |
| A_B | 0.0001 | 0.002 | 0.03 | 2/14 (test-2016-05-13, test-2014-04-29) | n/a | n/a | n/a | n/a |
| A_S | 0.0001 | 0.001 | 0.03 | 4/14 (test-2016-05-13, test-2014-04-24, test-2014-04-29, test-2015-03-19) | 0.0001 | 0.515 | 1.0 | 8/14 |
| B_S | 0.0001 | 0.006500000000000001 | 1.0 | 6/14 (test-2015-04-10, test-2016-05-13, test-2015-04-07, test-2014-04-29, test-2015-03-19, test-2016-04-12) | 0.0001 | 0.03 | 1.0 | 6/14 |
| A_B_S | 0.0001 | 0.001 | 0.03 | 2/14 (test-2016-05-13, test-2014-04-29) | 0.0001 | 0.16499999999999998 | 1.0 | 7/14 |

## Fitted beta and gamma per condition (range over folds)

| Condition | Beta min | Beta max | Gamma min | Gamma max |
|---|---:|---:|---:|---:|
| S | n/a | n/a | n/a | n/a |
| A | 2.4669 | 3.3811 | n/a | n/a |
| B | n/a | n/a | 0.7968 | 1.6557 |
| A_B | 2.3164 | 2.9511 | 0.6166 | 1.0469 |
| A_S | 2.4629 | 3.4528 | n/a | n/a |
| B_S | n/a | n/a | 0.1580 | 1.6557 |
| A_B_S | 2.3086 | 2.9508 | 0.6182 | 1.0483 |

## A-conditions vs repeat-own-last, v4, inventory B

Repeat-own-last: 82.741% accuracy, 0.3320 macro-F1, -1.43264 bits/coda.

| Condition | LL contrast | CI | +days | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|---:|
| A | 0.31060 | [0.2441, 0.4314] | 13/14 | 82.224% | 0.2774 |
| A_B | 0.24967 | [0.1244, 0.4433] | 12/14 | 81.125% | 0.2451 |
| A_S | 0.30025 | [0.2320, 0.4218] | 13/14 | 81.771% | 0.2569 |
| A_B_S | 0.23200 | [0.0982, 0.4342] | 12/14 | 81.254% | 0.2477 |

## All twelve add-one-input contrasts (v4, inventory B)

| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |
|---|---:|---:|---|---:|
| adding S to no inputs | 0.01861 | 0.05951 | [-0.0942, 0.1969] | 6/14 |
| adding A to no inputs | 1.52839 | 1.59916 | [1.2190, 1.9207] | 14/14 |
| adding B to no inputs | -0.05111 | 0.19584 | [-0.6176, 0.6216] | 10/14 |
| adding B to A | -0.06093 | -0.02463 | [-0.1325, 0.0196] | 7/14 |
| adding A to B | 1.51857 | 1.37869 | [0.9730, 2.0393] | 14/14 |
| adding A to S | 1.49944 | 1.53168 | [1.2232, 1.8092] | 14/14 |
| adding S to A | -0.01034 | -0.00797 | [-0.0199, -0.0010] | 4/14 |
| adding B to S | -0.05698 | 0.10177 | [-0.5552, 0.4134] | 10/14 |
| adding S to B | 0.01274 | -0.03455 | [-0.1415, 0.2180] | 7/14 |
| adding S to A and B | -0.01767 | -0.01102 | [-0.0341, -0.0006] | 5/14 |
| adding B to A and S | -0.06825 | -0.02768 | [-0.1454, 0.0185] | 7/14 |
| adding A to B and S | 1.48817 | 1.40223 | [0.9725, 2.0115] | 14/14 |

## Main contrasts, v4, inventory B

| Contrast | bits/coda | Unweighted | CI | +days | Repeat part | Switch part |
|---|---:|---:|---|---:|---:|---:|
| own_history | 1.48817 | 1.40223 | [0.9725, 2.0115] | 14/14 | 1.30606 | 0.18211 |
| partner_content | -0.06825 | -0.02768 | [-0.1454, 0.0185] | 7/14 | -0.04644 | -0.02182 |
| timing | -0.01767 | -0.01102 | [-0.0341, -0.0006] | 5/14 | -0.01296 | -0.00471 |

4914 fits were run, across 14 folds and 7 non-unigram conditions: conditions without S search 9 history L2 values, conditions with S jointly search 9 history values times 9 S values (DEC-20). With skip_nonconverged_candidates on (DEC-22), a candidate that fails fit()'s convergence check is left out of the validation-day selection; 0 of the 4914 did, and every selected model converged. Full per-grid-value convergence is in l2_selection.jsonl; the 98 selected models' convergence is in convergence.jsonl.

No candidate failed the convergence check in any fold or condition.

## Stop: results need review before interpretation

The following stop conditions from the task were met. No interpretation or conclusions are written below; the tables above are the complete result.

- add_input:A_S-A: contrast CI entirely below zero [-0.0199, -0.0010]
- add_input:A_B_S-A_B: contrast CI entirely below zero [-0.0341, -0.0006]

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. The switch-choice component of every contrast rests on about 267 non-repeat codas and is exploratory. Source, fold-manifest and configuration SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; reports/rq2_b_dtree_folds.md reports UNK rates per fold.
