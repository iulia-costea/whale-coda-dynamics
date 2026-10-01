# RQ1 retrospective-delay sensitivity run

Same fourteen leave-one-day-out folds, vocabulary and v3 model (shared copy features, L2 tuned per fold and condition on the validation day) as the primary results. Only the timing fields feeding S, A_S, B_S and A_B_S change: turn_category is kept, but gap_s and overlap_elapsed_s are replaced by retrospective_delay_s, the onset-to-previous-interlocutor-offset delay used before DEC-1. That field can be negative (the predecessor coda had not finished when the target started) and is only standardized, not log1p transformed, since it goes below -1 in this data. Conditions without S (none, A, B, A_B) are refit identically to the primary run for a complete lattice, and should reproduce it exactly. This run is a sensitivity check only; it does not alter the primary preprocessing, model, or conclusions.

## S-containing conditions and repeat-own-last, primary vs retrospective delay

| Condition | Primary LL bits/coda | Retrospective LL bits/coda | Primary Accuracy | Retrospective Accuracy | Primary Macro-F1 | Retrospective Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| S | -2.52372 | -2.55347 | 56.820% | 56.884% | 0.0437 | 0.0363 |
| A_S | -1.13053 | -1.11321 | 82.612% | 82.547% | 0.3181 | 0.3092 |
| B_S | -2.50494 | -2.50980 | 57.660% | 59.082% | 0.0668 | 0.0738 |
| A_B_S | -1.16249 | -1.13783 | 82.094% | 82.159% | 0.3422 | 0.3343 |
| repeat_own_last | -1.39276 | -1.39276 | 82.547% | 82.547% | 0.4070 | 0.4070 |

## Timing contrast and A_B_S vs repeat-own-last, primary vs retrospective delay

| Comparison | Primary bits/coda | Primary CI | Primary +days | Retrospective bits/coda | Retrospective CI | Retrospective +days |
|---|---:|---|---:|---:|---|---:|
| timing (A_B_S - A_B) | -0.05012 | [-0.0816, -0.0142] | 5/14 | -0.02546 | [-0.0438, -0.0072] | 4/14 |
| A_B_S vs repeat-own-last | 0.23027 | [0.1502, 0.3838] | 13/14 | 0.25493 | [0.1847, 0.3919] | 13/14 |

The A, B and A_B conditions do not use S and are refit identically to the primary run (verified exactly equal before this report is written).

## Stop: results need review before interpretation

The following stop conditions from the task were met on this retrospective-delay run. No interpretation or conclusions are written below; the tables above are the complete result.

- add_input:A_B_S-A_B: contrast CI entirely below zero [-0.0438, -0.0072]

Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. Source, fold-manifest and configuration SHA-256 hashes and full model settings are recorded in evaluation/summary.json alongside this report.
