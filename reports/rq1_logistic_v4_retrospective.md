# RQ1 logistic regression v4: separate penalties

Joint validation-day selection over base L2 [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] and S L2 [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]. History lag weights, copy weights and the shared intercept use base L2; all timing and turn-category coefficients use S L2. Ties prefer larger base L2, then larger S L2. No-S conditions retain v3's fit exactly. Training excludes validation and test days. Preprocessing, folds, vocabulary and shared evaluation are unchanged.

Sensitivity only: S uses the retrospective end-to-start delay. The primary and retrospective runs reproduce all no-S conditions exactly on this machine.

## S-containing conditions and repeat-own-last, primary vs retrospective delay

| Condition | Primary LL bits/coda | Retrospective LL bits/coda | Primary Accuracy | Retrospective Accuracy | Primary Macro-F1 | Retrospective Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| S | -2.55388 | -2.54672 | 57.272% | 56.884% | 0.0441 | 0.0363 |
| A_S | -1.10695 | -1.09779 | 82.159% | 82.288% | 0.2824 | 0.2858 |
| B_S | -2.47630 | -2.46632 | 57.983% | 58.824% | 0.0691 | 0.0709 |
| A_B_S | -1.13261 | -1.11539 | 82.288% | 82.482% | 0.3248 | 0.3270 |
| repeat_own_last | -1.39276 | -1.39276 | 82.547% | 82.547% | 0.4070 | 0.4070 |

## Timing contrast and A_B_S vs repeat-own-last, primary vs retrospective delay

| Comparison | Primary bits/coda | Primary CI | Primary +days | Retrospective bits/coda | Retrospective CI | Retrospective +days |
|---|---:|---|---:|---:|---|---:|
| timing (A_B_S - A_B) | -0.02024 | [-0.0464, 0.0014] | 7/14 | -0.00302 | [-0.0099, 0.0034] | 8/14 |
| A_B_S vs repeat-own-last | 0.26015 | [0.1855, 0.4047] | 13/14 | 0.27737 | [0.2096, 0.4086] | 13/14 |

The A, B and A_B conditions do not use S and are refit identically to the primary run (verified exactly equal before this report is written).

## Result

No A-condition falls below repeat-own-last in log-likelihood or accuracy on this run, and no add-one-input contrast has a confidence interval entirely below zero.

Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. Source, fold-manifest and configuration SHA-256 hashes and full model settings are recorded in evaluation/summary.json alongside this report.

## Penalty distributions and gate 1

| Condition | Penalty | Value: fold count | Lower edge | Upper edge |
|---|---|---|---:|---:|
| S | chosen_l2 | 0.0001: 6, 0.003: 2, 0.01: 3, 0.1: 1, 0.3: 1, 1.0: 1 | 6/14 | 1/14 |
| S | chosen_s_l2 | 0.0001: 2, 0.01: 2, 0.03: 2, 0.1: 2, 0.3: 1, 1.0: 5 | 2/14 | 5/14 |
| A | chosen_l2 | 0.0001: 3, 0.0003: 3, 0.001: 4, 0.003: 3, 0.03: 1 | 3/14 | 0/14 |
| B | chosen_l2 | 0.0001: 2, 0.0003: 1, 0.001: 3, 0.003: 2, 0.01: 3, 0.03: 1, 0.1: 1, 0.3: 1 | 2/14 | 0/14 |
| A_B | chosen_l2 | 0.0001: 1, 0.0003: 2, 0.001: 5, 0.003: 4, 0.01: 2 | 1/14 | 0/14 |
| A_S | chosen_l2 | 0.0001: 3, 0.0003: 2, 0.001: 5, 0.003: 3, 0.03: 1 | 3/14 | 0/14 |
| A_S | chosen_s_l2 | 0.0001: 2, 0.0003: 1, 0.03: 2, 0.1: 3, 0.3: 1, 1.0: 5 | 2/14 | 5/14 |
| B_S | chosen_l2 | 0.0001: 1, 0.0003: 1, 0.001: 3, 0.003: 3, 0.01: 1, 0.3: 2, 1.0: 3 | 1/14 | 3/14 |
| B_S | chosen_s_l2 | 0.0001: 4, 0.003: 1, 0.01: 2, 0.03: 2, 0.3: 1, 1.0: 4 | 4/14 | 4/14 |
| A_B_S | chosen_l2 | 0.0001: 1, 0.0003: 2, 0.001: 5, 0.003: 4, 0.01: 2 | 1/14 | 0/14 |
| A_B_S | chosen_s_l2 | 0.0001: 3, 0.0003: 1, 0.003: 1, 0.01: 2, 0.1: 2, 0.3: 1, 1.0: 4 | 3/14 | 4/14 |

DEC-16 and grid checks: PASS.


## Complete likelihood contrasts versus reference

| Contrast | Run | Weighted bits/coda | Unweighted day mean | 95% CI | Positive days |
|---|---|---:|---:|---|---:|
| own_history | reference | 1.34369 | 1.25568 | [0.9445, 1.7317] | 14/14 |
| own_history | v4 | 1.35093 | 1.26332 | [0.9528, 1.7425] | 14/14 |
| partner_content | reference | -0.02566 | -0.00144 | [-0.0942, 0.0635] | 6/14 |
| partner_content | v4 | -0.01760 | 0.00300 | [-0.0802, 0.0667] | 8/14 |
| timing | reference | -0.02024 | -0.00885 | [-0.0464, 0.0014] | 7/14 |
| timing | v4 | -0.00302 | -0.00349 | [-0.0099, 0.0034] | 8/14 |
| add_input:S-none | reference | 0.04866 | 0.12024 | [-0.0804, 0.2992] | 6/14 |
| add_input:S-none | v4 | 0.05582 | 0.13103 | [-0.0452, 0.3013] | 5/14 |
| add_input:A-none | reference | 1.50564 | 1.59874 | [1.1967, 1.9137] | 14/14 |
| add_input:A-none | v4 | 1.50564 | 1.59874 | [1.1967, 1.9137] | 14/14 |
| add_input:B-none | reference | 0.12976 | 0.34452 | [-0.2908, 0.6998] | 10/14 |
| add_input:B-none | v4 | 0.12976 | 0.34452 | [-0.2908, 0.6998] | 10/14 |
| add_input:A_B-A | reference | -0.01547 | 0.00720 | [-0.0747, 0.0665] | 7/14 |
| add_input:A_B-A | v4 | -0.01547 | 0.00720 | [-0.0747, 0.0665] | 7/14 |
| add_input:A_B-B | reference | 1.36041 | 1.26143 | [0.9493, 1.7408] | 14/14 |
| add_input:A_B-B | v4 | 1.36041 | 1.26143 | [0.9493, 1.7408] | 14/14 |
| add_input:A_S-S | reference | 1.44693 | 1.47829 | [1.1993, 1.7507] | 14/14 |
| add_input:A_S-S | v4 | 1.44893 | 1.46843 | [1.1776, 1.7556] | 14/14 |
| add_input:A_S-A | reference | -0.01005 | -0.00021 | [-0.0248, 0.0037] | 7/14 |
| add_input:A_S-A | v4 | -0.00090 | 0.00071 | [-0.0088, 0.0060] | 9/14 |
| add_input:B_S-S | reference | 0.07758 | 0.22117 | [-0.2833, 0.4581] | 10/14 |
| add_input:B_S-S | v4 | 0.08040 | 0.20810 | [-0.2727, 0.4744] | 10/14 |
| add_input:B_S-B | reference | -0.00352 | -0.00311 | [-0.1085, 0.1404] | 9/14 |
| add_input:B_S-B | v4 | 0.00646 | -0.00538 | [-0.0930, 0.1349] | 8/14 |
| add_input:A_B_S-A_B | reference | -0.02024 | -0.00885 | [-0.0464, 0.0014] | 7/14 |
| add_input:A_B_S-A_B | v4 | -0.00302 | -0.00349 | [-0.0099, 0.0034] | 8/14 |
| add_input:A_B_S-A_S | reference | -0.02566 | -0.00144 | [-0.0942, 0.0635] | 6/14 |
| add_input:A_B_S-A_S | v4 | -0.01760 | 0.00300 | [-0.0802, 0.0667] | 8/14 |
| add_input:A_B_S-B_S | reference | 1.34369 | 1.25568 | [0.9445, 1.7317] | 14/14 |
| add_input:A_B_S-B_S | v4 | 1.35093 | 1.26332 | [0.9528, 1.7425] | 14/14 |
