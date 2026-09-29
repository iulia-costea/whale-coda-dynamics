# RQ1 retrospective-delay sensitivity run

Same fourteen leave-one-day-out folds, vocabulary and v4 model (shared copy features, L2 tuned per fold and condition on the validation day) as the primary results. Only the timing fields feeding S, A_S, B_S and A_B_S change: turn_category is kept, but gap_s and overlap_elapsed_s are replaced by retrospective_delay_s, the onset-to-previous-interlocutor-offset delay used before DEC-1. That field can be negative (the predecessor coda had not finished when the target started) and is only standardized, not log1p transformed, since it goes below -1 in this data. Conditions without S (none, A, B, A_B) are refit identically to the primary run for a complete lattice, and should reproduce it exactly. This run is a sensitivity check only; it does not alter the primary preprocessing, model, or conclusions.

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
