# RQ1 retrospective-delay sensitivity run

Same fourteen leave-one-day-out folds, vocabulary and v2 model (shared copy features, L2 tuned per fold and condition on the validation day) as the primary results. Only the timing fields feeding S, A_S, B_S and A_B_S change: turn_category is kept, but gap_s and overlap_elapsed_s are replaced by retrospective_delay_s, the onset-to-previous-interlocutor-offset delay used before DEC-1. That field can be negative (the predecessor coda had not finished when the target started) and is only standardized, not log1p transformed, since it goes below -1 in this data. Conditions without S (none, A, B, A_B) are refit identically to the primary run for a complete lattice, and should reproduce it exactly. This run is a sensitivity check only; it does not alter the primary preprocessing, model, or conclusions.

## S-containing conditions and repeat-own-last, primary vs retrospective delay

| Condition | Primary LL bits/coda | Retrospective LL bits/coda | Primary Accuracy | Retrospective Accuracy | Primary Macro-F1 | Retrospective Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| S | -2.51416 | -2.54714 | 56.949% | 56.884% | 0.0438 | 0.0363 |
| A_S | -1.11937 | -1.10374 | 82.999% | 82.676% | 0.3297 | 0.3159 |
| B_S | -2.50527 | -2.50168 | 57.725% | 59.147% | 0.0670 | 0.0739 |
| A_B_S | -1.14421 | -1.12348 | 82.094% | 82.224% | 0.3428 | 0.3367 |
| repeat_own_last | -1.39276 | -1.39276 | 82.547% | 82.547% | 0.4070 | 0.4070 |

## Timing contrast and A_B_S vs repeat-own-last, primary vs retrospective delay

| Comparison | Primary bits/coda | Primary CI | Primary +days | Retrospective bits/coda | Retrospective CI | Retrospective +days |
|---|---:|---|---:|---:|---|---:|
| timing (A_B_S - A_B) | -0.04603 | [-0.0772124079144048, -0.009267742881333444] | 5/14 | -0.02529 | [-0.044108695593226936, -0.0061985138577028975] | 4/14 |
| A_B_S vs repeat-own-last | 0.24855 | [0.16930690836264572, 0.41155823053676477] | 14/14 | 0.26928 | [0.20276366245670394, 0.4119912653187393] | 14/14 |

The A, B and A_B conditions do not use S and are refit identically to the primary run (verified exactly equal before this report is written).

## Stop: results need review before interpretation

The following stop conditions from the task were met on this retrospective-delay run. No interpretation or conclusions are written below; the tables above are the complete result.

- A_B: accuracy 82.353% below repeat-own-last 82.547%
- A_B_S: accuracy 82.224% below repeat-own-last 82.547%
- add_input:A_B_S-A_B: contrast CI entirely below zero [-0.044108695593226936, -0.0061985138577028975]

Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. Source, fold-manifest and configuration SHA-256 hashes and full model settings are recorded in evaluation/summary.json alongside this report.
