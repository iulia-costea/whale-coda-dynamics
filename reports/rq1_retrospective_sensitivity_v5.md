# RQ1 old-delay sensitivity run, logistic v5 (DEC-23)

**Logistic v4 remains RQ1's headline (DEC-23; reports/rq1_logistic_v4.md). This report is the old-delay sensitivity run under v5, part of the robustness line.**

Logistic v5 is logistic v4 with optional ignorance by selection (DEC-23, PLAN `[M4]`). For each fold and condition, the candidates are v4's chosen fits of every sub-condition (A, B and S each on or absent; all absent is the fold's smoothed prior, condition none). The candidate with the highest validation-day log-likelihood is chosen (exact ties: fewer blocks, then the order A, B, S), and its stored v4 test-day predictions become the condition's. Each sub-condition's penalties were already chosen by v4's rule, so nothing is refitted: everything comes from `data/results/rq1_logistic_v4_retrospective/` (models.jsonl, l2_selection.jsonl, predictions.jsonl). Sensitivity: DEC-21's one-SE rule (the fewest blocks within one standard error of the best; SE of the per-target paired validation differences). Every log-likelihood below is in bits per coda, from the shared evaluation module. Old-delay sensitivity run (PLAN §9, DEC-1): S, A_S, B_S and A_B_S use retrospective_delay_s in place of gap_s and overlap_elapsed_s (turn_category is kept); conditions without S are those of the primary run.

## Checks before any test-day number

- Reproduction (gate 1): the validation log-likelihoods of all 98 v4 fits, recomputed from models.jsonl, match l2_selection.jsonl to 4.44e-16 bits at most (tolerance 1e-09); their test-day probabilities, and the prior's, match predictions.jsonl to 8.88e-16 at most.
- Validation monotonicity (gate 2): under the max rule all 168 add-one-input contrasts (12 pairs × 14 validation days) are ≥ 0 on the validation day; the smallest is 0 bits.
- The smallest validation-day gap between a chosen candidate and the runner-up is 2.52e-05 bits (test-2016-04-12, condition A_S).
- Every v5 test-day log-likelihood equals v4's for the chosen sub-condition on the same coda to 1.78e-15 bits at most (tolerance 1e-12; recomputing log2 on another machine can change the last bit).

## Stop rules

DEC-23's rule: each of the 16 checks (12 add-one-input contrasts, 4 A-conditions against repeat-own-last) gets an exact one-sided sign-flip test on the day sums of per-coda differences (alternative: mean below zero; all 2^14 sign assignments; statistic: token-weighted mean), with Holm's correction across the 16; a check fires when its adjusted p < 0.05. DEC-16's old rule (day-bootstrap CI entirely below zero) is shown next to it.

| Model | DEC-23 rule (sign-flip, Holm) | DEC-16 rule (CI below zero) |
|---|---|---|
| v4 | none fired | none fired |
| v5 (max) | none fired | none fired |
| v5 (one-SE) | none fired | none fired |

Governing status (DEC-23's rule on v4): none fired.

## Lattice, v4 and v5

| Condition | v4 LL | v5 LL | v5 − v4 | v5 one-SE LL | v4 accuracy | v5 accuracy | v4 macro-F1 | v5 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | -2.60254 | -2.60254 | 0.00000 | -2.60254 | 57.207% | 57.207% | 0.0364 | 0.0364 |
| S | -2.54672 | -2.54340 | 0.00332 | -2.53552 | 56.884% | 57.078% | 0.0363 | 0.0363 |
| A | -1.09690 | -1.09690 | 0.00000 | -1.09690 | 82.353% | 82.353% | 0.2819 | 0.2819 |
| B | -2.47278 | -2.42671 | 0.04607 | -2.42671 | 59.017% | 59.341% | 0.0729 | 0.0705 |
| A_B | -1.11237 | -1.13119 | -0.01882 | -1.13287 | 82.482% | 81.836% | 0.3270 | 0.2842 |
| A_S | -1.09779 | -1.09780 | -0.00001 | -1.09689 | 82.288% | 82.288% | 0.2858 | 0.2858 |
| B_S | -2.46632 | -2.45259 | 0.01372 | -2.47457 | 58.824% | 59.405% | 0.0709 | 0.0712 |
| A_B_S | -1.11539 | -1.13390 | -0.01850 | -1.13438 | 82.482% | 81.642% | 0.3270 | 0.2826 |
| majority | n/a | n/a | — | n/a | 57.207% | 57.207% | 0.0364 | 0.0364 |
| repeat_own_last | -1.39276 | -1.39276 | — | -1.39276 | 82.547% | 82.547% | 0.4070 | 0.4070 |

## Sub-conditions chosen

Folds (of 14) on which each sub-condition was chosen: max rule, with the one-SE rule in parentheses. A dash marks a sub-condition that is not a candidate for that condition.

| Condition | none | A | B | S | A_B | A_S | B_S | A_B_S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 14 (14) | — | — | — | — | — | — | — |
| S | 2 (5) | — | — | 12 (9) | — | — | — | — |
| A | 0 (0) | 14 (14) | — | — | — | — | — | — |
| B | 3 (3) | — | 11 (11) | — | — | — | — | — |
| A_B | 0 (0) | 7 (9) | 0 (0) | — | 7 (5) | — | — | — |
| A_S | 0 (0) | 5 (13) | — | 0 (0) | — | 9 (1) | — | — |
| B_S | 0 (1) | — | 3 (5) | 3 (3) | — | — | 8 (5) | — |
| A_B_S | 0 (0) | 1 (7) | 0 (0) | 0 (0) | 2 (6) | 6 (1) | 0 (0) | 5 (0) |

Per fold: the chosen sub-condition under the max rule, with the one-SE choice after an arrow where it differs.

| Test day | S | A | B | A_B | A_S | B_S | A_B_S |
|---|---|---|---|---|---|---|---|
| 2014-04-16 | none | A | B | A_B | A | B | A_B_S → A_B |
| 2014-04-24 | S | A | B | A | A_S | B_S | A_S |
| 2014-04-29 | S → none | A | B | A_B → A | A_S → A | B_S | A_B_S → A_B |
| 2015-03-02 | S | A | B | A_B | A | B_S | A_B_S → A_B |
| 2015-03-19 | S → none | A | B | A_B | A | B | A_B |
| 2015-03-26 | none | A | B | A | A_S → A | B_S → B | A_S → A |
| 2015-03-31 | S | A | B | A | A_S → A | B_S → B | A_S → A |
| 2015-04-01 | S | A | B | A_B → A | A_S → A | B_S | A_B_S → A |
| 2015-04-07 | S | A | none | A | A_S → A | S | A_S → A |
| 2015-04-10 | S | A | none | A | A_S → A | S | A_S → A |
| 2016-04-12 | S | A | B | A | A_S → A | B_S → S | A_S → A |
| 2016-04-24 | S | A | B | A_B | A | B | A_B |
| 2016-05-12 | S → none | A | none | A | A | S → none | A |
| 2016-05-13 | S | A | B | A_B | A_S → A | B_S | A_B_S → A_B |

## RQ1 contrasts (`[M2]`)

Paired per-coda differences; positive days, with the days whose difference is exactly zero; day-bootstrap 95% CI (10,000 replicates); exact two-sided sign-flip p on the day sums (DEC-23).

| Contrast | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |
|---|---|---:|---:|---|---:|---:|
| own_history | v4 | 1.35092 | 1.26332 | [0.9528, 1.7425] | 14/14 | 0.0001221 |
| own_history | v5 (max) | 1.31870 | 1.23661 | [0.9166, 1.7221] | 14/14 | 0.0001221 |
| own_history | v5 (one-SE) | 1.34020 | 1.28047 | [0.9368, 1.7457] | 14/14 | 0.0001221 |
| partner_content | v4 | -0.01760 | 0.00300 | [-0.0802, 0.0667] | 8/14 | 0.6787 |
| partner_content | v5 (max) | -0.03610 | -0.03242 | [-0.0875, 0.0105] | 3/14 (7 at 0) | 0.4062 |
| partner_content | v5 (one-SE) | -0.03748 | -0.03250 | [-0.0838, 0.0032] | 2/14 (8 at 0) | 0.2188 |
| timing | v4 | -0.00302 | -0.00348 | [-0.0099, 0.0034] | 8/14 | 0.3853 |
| timing | v5 (max) | -0.00270 | -0.00183 | [-0.0098, 0.0049] | 7/14 (3 at 0) | 0.5205 |
| timing | v5 (one-SE) | -0.00151 | -0.00346 | [-0.0062, 0.0000] | 1/14 (12 at 0) | 1 |

Sign of each token-weighted mean and where its CI lies:

| Contrast | v4 | v5 (max) | v5 (one-SE) |
|---|---|---|---|
| own_history | +, CI > 0 | +, CI > 0 | +, CI > 0 |
| partner_content | −, CI spans 0 | −, CI spans 0 | −, CI spans 0 |
| timing | −, CI spans 0 | −, CI spans 0 | −, CI spans 0 |

The sign and CI position of every `[M2]` contrast are the same under v4 and v5.

## The sixteen stop-rule checks

Per-coda differences, token-weighted means, day-bootstrap CI and positive days (with the days whose difference is exactly zero); DEC-16: CI entirely below zero; one-sided sign-flip p (alternative: mean below zero) and its Holm adjustment across the 16; DEC-23: adjusted p below alpha.

### v4

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.05582 | [-0.0452, 0.3013] | 5/14 | — | 0.5042 | 1 | — |
| adding A to none (A − none) | 1.50564 | [1.1967, 1.9137] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.12976 | [-0.2908, 0.6998] | 10/14 | — | 0.7002 | 1 | — |
| adding B to A (A_B − A) | -0.01547 | [-0.0747, 0.0665] | 7/14 | — | 0.3362 | 1 | — |
| adding A to B (A_B − B) | 1.36041 | [0.9493, 1.7408] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.44893 | [1.1776, 1.7556] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00089 | [-0.0088, 0.0060] | 9/14 | — | 0.3407 | 1 | — |
| adding B to S (B_S − S) | 0.08040 | [-0.2727, 0.4744] | 10/14 | — | 0.6384 | 1 | — |
| adding S to B (B_S − B) | 0.00646 | [-0.0930, 0.1349] | 8/14 | — | 0.5369 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.00302 | [-0.0099, 0.0034] | 8/14 | — | 0.1926 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.01760 | [-0.0802, 0.0667] | 8/14 | — | 0.3394 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.35092 | [0.9528, 1.7425] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.29586 | [0.2420, 0.3719] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.28039 | [0.2158, 0.4092] | 13/14 | — | 0.9998 | 1 | — |
| A_S − repeat-own-last | 0.29497 | [0.2389, 0.3735] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.27737 | [0.2096, 0.4086] | 13/14 | — | 0.9998 | 1 | — |

### v5 (max)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.05914 | [-0.0423, 0.3035] | 5/14 (2 at 0) | — | 0.5098 | 1 | — |
| adding A to none (A − none) | 1.50564 | [1.1967, 1.9137] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.17583 | [-0.2239, 0.7162] | 8/14 (3 at 0) | — | 0.7666 | 1 | — |
| adding B to A (A_B − A) | -0.03430 | [-0.0823, 0.0089] | 3/14 (7 at 0) | — | 0.1719 | 1 | — |
| adding A to B (A_B − B) | 1.29551 | [0.8913, 1.6908] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.44560 | [1.1711, 1.7545] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00090 | [-0.0088, 0.0060] | 6/14 (5 at 0) | — | 0.3398 | 1 | — |
| adding B to S (B_S − S) | 0.09081 | [-0.2647, 0.4720] | 8/14 (3 at 0) | — | 0.6572 | 1 | — |
| adding S to B (B_S − B) | -0.02589 | [-0.0908, 0.0145] | 6/14 (3 at 0) | — | 0.1792 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.00270 | [-0.0098, 0.0049] | 7/14 (3 at 0) | — | 0.2603 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.03610 | [-0.0875, 0.0105] | 3/14 (7 at 0) | — | 0.2031 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.31870 | [0.9166, 1.7221] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.29586 | [0.2420, 0.3719] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.26156 | [0.2126, 0.3496] | 13/14 | — | 0.9998 | 1 | — |
| A_S − repeat-own-last | 0.29496 | [0.2389, 0.3735] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.25886 | [0.2061, 0.3509] | 13/14 | — | 0.9998 | 1 | — |

### v5 (one-SE)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.06702 | [-0.0307, 0.3174] | 5/14 (5 at 0) | — | 0.5527 | 1 | — |
| adding A to none (A − none) | 1.50564 | [1.1967, 1.9137] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.17583 | [-0.2239, 0.7162] | 8/14 (3 at 0) | — | 0.7666 | 1 | — |
| adding B to A (A_B − A) | -0.03597 | [-0.0820, 0.0050] | 2/14 (9 at 0) | — | 0.1562 | 1 | — |
| adding A to B (A_B − B) | 1.29384 | [0.8925, 1.6906] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.43863 | [1.1602, 1.7487] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | 0.00001 | [0.0000, 0.0000] | 1/14 (13 at 0) | — | 1 | 1 | — |
| adding B to S (B_S − S) | 0.06095 | [-0.2907, 0.4362] | 7/14 (4 at 0) | — | 0.6123 | 1 | — |
| adding S to B (B_S − B) | -0.04787 | [-0.1758, 0.0134] | 4/14 (6 at 0) | — | 0.1641 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.00151 | [-0.0062, 0.0000] | 1/14 (12 at 0) | — | 0.5 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.03748 | [-0.0838, 0.0032] | 2/14 (8 at 0) | — | 0.1094 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.34020 | [0.9368, 1.7457] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.29586 | [0.2420, 0.3719] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.25989 | [0.2107, 0.3484] | 13/14 | — | 0.9998 | 1 | — |
| A_S − repeat-own-last | 0.29587 | [0.2420, 0.3719] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.25838 | [0.2087, 0.3469] | 13/14 | — | 0.9998 | 1 | — |

## Primary timing vs the old delay, v5

The timing contrast (A_B_S − A_B) in the primary v5 run and in this run. Conditions without S are identical in both runs (checked before this report was written).

| Run | Token-weighted mean | 95% CI | Positive days |
|---|---:|---|---:|
| primary (onset-available timing) | -0.01842 | [-0.0453, 0.0035] | 4/14 (5 at 0) |
| this run (old delay) | -0.00270 | [-0.0098, 0.0049] | 7/14 (3 at 0) |

## Notes

- Day-bootstrap CIs are approximate with 14 days, and limited data and capacity push contrasts toward zero: a CI that spans zero may mean too little data (PLAN §9).
- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly favours keeping S; the one-SE rule counters this (DEC-23).
- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).
- This report establishes no grammar or causal mechanism.
- Per fold, condition and rule, the candidates and their validation log-likelihoods are in `data/results/rq1_logistic_v5_retrospective/selection.jsonl`; v5's predictions are rebuilt from it and `data/results/rq1_logistic_v4_retrospective/predictions.jsonl` (`logistic_v5.rebuild_predictions`).

## Inputs

SHA-256 of every input:

- `config/rq1_logistic_v5_retrospective.json`: `8299ba88cd4d709f2a36a0d6844167c6a90a010cd53c9b4d9b92073fe89e962e`
- `config/rq1_logistic_v4_retrospective.json`: `8b3585bafbb8589ee73e1bf112e4561b8b206b40fd2c96fb30ebf8e5f0ef1e6a`
- `config/rq1_evaluation_v4_retrospective.json`: `6bc70e9a44e270bda0d2e1774f48a6c7f4c72b1c2096fe7e0c7b8af9894c6254`
- `data/processed/rq1_folds_retrospective/manifest.json`: `782649caf37c706dee0a1b3e37745e47c12ebb723038724fab34e08881f9f1a7`
- `data/processed/rq1_targets.jsonl`: `9c7245c36fcc392dec0e67db08b7d943cd89107eb1b34e3b2b8cc55e1934be02`
- `data/results/rq1_logistic_v4_retrospective/models.jsonl`: `687f497ccb3e2ba24a47ddb48183705fa60abb43ffd2b05705d594cb6857520a`
- `data/results/rq1_logistic_v4_retrospective/l2_selection.jsonl`: `0d010485fee0f1c599039a4df3ad203ff1b2eb57ca9bc6dec8effc8f088009b5`
- `data/results/rq1_logistic_v4_retrospective/predictions.jsonl`: `4df2a7328ad5bf3485f7d81f2e3c19a15529674a55901ec60507e861133a4fb8`
- `data/results/rq1_logistic_v4_retrospective/baseline_predictions.jsonl`: `b294dec536b7a8b418448c16ce2e646587a991a49e7daf6da36df00b124edb20`
- `data/results/rq1_logistic_v4_retrospective/evaluation/summary.json`: `52741e862a00feb6070cd86676f0eaf45dac2481f855e2dd57c30dbd4bd1344b`
- `data/results/rq1_logistic_v4_retrospective/evaluation/scored_predictions.jsonl`: `f56965c54344206d3cab73d0f79cd605c6dfcb093fc07dec92ea48f276f6c68a`
- `data/results/rq1_logistic_v5/evaluation/summary.json`: `3ab2fb36b18fcd9f6c26a2a8adc103d3bca741f2f83986ae52a0d12b90042e3d`
