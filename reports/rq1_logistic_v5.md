# RQ1 logistic regression v5 (DEC-23)

**Logistic v4 remains RQ1's headline (DEC-23; reports/rq1_logistic_v4.md). This report is the robustness line.**

Logistic v5 is logistic v4 with optional ignorance by selection (DEC-23, PLAN `[M4]`). For each fold and condition, the candidates are v4's chosen fits of every sub-condition (A, B and S each on or absent; all absent is the fold's smoothed prior, condition none). The candidate with the highest validation-day log-likelihood is chosen (exact ties: fewer blocks, then the order A, B, S), and its stored v4 test-day predictions become the condition's. Each sub-condition's penalties were already chosen by v4's rule, so nothing is refitted: everything comes from `data/results/rq1_logistic_v4/` (models.jsonl, l2_selection.jsonl, predictions.jsonl). Sensitivity: DEC-21's one-SE rule (the fewest blocks within one standard error of the best; SE of the per-target paired validation differences). Every log-likelihood below is in bits per coda, from the shared evaluation module.

## Checks before any test-day number

- Reproduction (gate 1): the validation log-likelihoods of all 98 v4 fits, recomputed from models.jsonl, match l2_selection.jsonl to 4.44e-16 bits at most (tolerance 1e-09); their test-day probabilities, and the prior's, match predictions.jsonl to 1.44e-15 at most.
- Validation monotonicity (gate 2): under the max rule all 168 add-one-input contrasts (12 pairs × 14 validation days) are ≥ 0 on the validation day; the smallest is 0 bits.
- The smallest validation-day gap between a chosen candidate and the runner-up is 4.38e-05 bits (test-2016-05-13, condition A_S).
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
| S | -2.55388 | -2.55501 | -0.00113 | -2.55484 | 57.272% | 57.272% | 0.0441 | 0.0441 |
| A | -1.09690 | -1.09690 | 0.00000 | -1.09690 | 82.353% | 82.353% | 0.2819 | 0.2819 |
| B | -2.47278 | -2.42671 | 0.04607 | -2.42671 | 59.017% | 59.341% | 0.0729 | 0.0705 |
| A_B | -1.11237 | -1.13119 | -0.01882 | -1.13287 | 82.482% | 81.836% | 0.3270 | 0.2842 |
| A_S | -1.10695 | -1.10702 | -0.00007 | -1.09696 | 82.159% | 82.224% | 0.2824 | 0.2825 |
| B_S | -2.47630 | -2.48956 | -0.01326 | -2.50698 | 57.983% | 57.660% | 0.0691 | 0.0681 |
| A_B_S | -1.13261 | -1.14962 | -0.01701 | -1.15144 | 82.288% | 81.319% | 0.3248 | 0.2750 |
| majority | n/a | n/a | — | n/a | 57.207% | 57.207% | 0.0364 | 0.0364 |
| repeat_own_last | -1.39276 | -1.39276 | — | -1.39276 | 82.547% | 82.547% | 0.4070 | 0.4070 |

## Sub-conditions chosen

Folds (of 14) on which each sub-condition was chosen: max rule, with the one-SE rule in parentheses. A dash marks a sub-condition that is not a candidate for that condition.

| Condition | none | A | B | S | A_B | A_S | B_S | A_B_S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 14 (14) | — | — | — | — | — | — | — |
| S | 1 (3) | — | — | 13 (11) | — | — | — | — |
| A | 0 (0) | 14 (14) | — | — | — | — | — | — |
| B | 3 (3) | — | 11 (11) | — | — | — | — | — |
| A_B | 0 (0) | 7 (9) | 0 (0) | — | 7 (5) | — | — | — |
| A_S | 0 (0) | 6 (12) | — | 0 (0) | — | 8 (2) | — | — |
| B_S | 1 (1) | — | 4 (5) | 1 (3) | — | — | 8 (5) | — |
| A_B_S | 0 (0) | 3 (8) | 0 (0) | 0 (0) | 2 (3) | 4 (1) | 0 (0) | 5 (2) |

Per fold: the chosen sub-condition under the max rule, with the one-SE choice after an arrow where it differs.

| Test day | S | A | B | A_B | A_S | B_S | A_B_S |
|---|---|---|---|---|---|---|---|
| 2014-04-16 | S | A | B | A_B | A_S → A | B | A_B_S → A_B |
| 2014-04-24 | S | A | B | A | A_S | B_S | A_S |
| 2014-04-29 | S → none | A | B | A_B → A | A | B_S → B | A_B_S → A |
| 2015-03-02 | S | A | B | A_B | A_S → A | B_S | A_B_S |
| 2015-03-19 | S | A | B | A_B | A | B | A_B |
| 2015-03-26 | S → none | A | B | A | A_S → A | B_S | A_S → A |
| 2015-03-31 | S | A | B | A | A | B | A |
| 2015-04-01 | S | A | B | A_B → A | A_S | B_S | A_B_S → A |
| 2015-04-07 | none | A | none | A | A_S → A | none | A_S → A |
| 2015-04-10 | S | A | none | A | A_S → A | S | A_S → A |
| 2016-04-12 | S | A | B | A | A | B_S → S | A |
| 2016-04-24 | S | A | B | A_B | A_S → A | B | A_B_S |
| 2016-05-12 | S | A | none | A | A | B_S → S | A |
| 2016-05-13 | S | A | B | A_B | A | B_S | A_B |

## RQ1 contrasts (`[M2]`)

Paired per-coda differences; positive days, with the days whose difference is exactly zero; day-bootstrap 95% CI (10,000 replicates); exact two-sided sign-flip p on the day sums (DEC-23).

| Contrast | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |
|---|---|---:|---:|---|---:|---:|
| own_history | v4 | 1.34369 | 1.25568 | [0.9445, 1.7317] | 14/14 | 0.0001221 |
| own_history | v5 (max) | 1.33994 | 1.25408 | [0.9390, 1.7252] | 14/14 | 0.0001221 |
| own_history | v5 (one-SE) | 1.35554 | 1.29268 | [0.9592, 1.7427] | 14/14 | 0.0001221 |
| partner_content | v4 | -0.02566 | -0.00144 | [-0.0942, 0.0635] | 6/14 | 0.6279 |
| partner_content | v5 (max) | -0.04260 | -0.03512 | [-0.1018, 0.0095] | 3/14 (7 at 0) | 0.3594 |
| partner_content | v5 (one-SE) | -0.05447 | -0.03761 | [-0.1267, 0.0042] | 2/14 (8 at 0) | 0.2812 |
| timing | v4 | -0.02024 | -0.00885 | [-0.0464, 0.0014] | 7/14 | 0.239 |
| timing | v5 (max) | -0.01842 | -0.00567 | [-0.0453, 0.0035] | 4/14 (5 at 0) | 0.4375 |
| timing | v5 (one-SE) | -0.01857 | -0.00889 | [-0.0458, 0.0000] | 0/14 (11 at 0) | 0.25 |

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
| adding S to none (S − none) | 0.04866 | [-0.0804, 0.2992] | 6/14 | — | 0.5596 | 1 | — |
| adding A to none (A − none) | 1.50564 | [1.1967, 1.9137] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.12976 | [-0.2908, 0.6998] | 10/14 | — | 0.7002 | 1 | — |
| adding B to A (A_B − A) | -0.01547 | [-0.0747, 0.0665] | 7/14 | — | 0.3362 | 1 | — |
| adding A to B (A_B − B) | 1.36041 | [0.9493, 1.7408] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.44693 | [1.1993, 1.7507] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.01005 | [-0.0248, 0.0037] | 7/14 | — | 0.2472 | 1 | — |
| adding B to S (B_S − S) | 0.07758 | [-0.2833, 0.4581] | 10/14 | — | 0.6263 | 1 | — |
| adding S to B (B_S − B) | -0.00352 | [-0.1085, 0.1404] | 9/14 | — | 0.4747 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.02024 | [-0.0464, 0.0014] | 7/14 | — | 0.1195 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.02566 | [-0.0942, 0.0635] | 6/14 | — | 0.314 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.34369 | [0.9445, 1.7317] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.29586 | [0.2420, 0.3719] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.28039 | [0.2158, 0.4092] | 13/14 | — | 0.9998 | 1 | — |
| A_S − repeat-own-last | 0.28581 | [0.2386, 0.3663] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.26015 | [0.1855, 0.4047] | 13/14 | — | 0.9998 | 1 | — |

### v5 (max)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.04753 | [-0.0819, 0.2971] | 5/14 (1 at 0) | — | 0.5537 | 1 | — |
| adding A to none (A − none) | 1.50564 | [1.1967, 1.9137] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.17583 | [-0.2239, 0.7162] | 8/14 (3 at 0) | — | 0.7666 | 1 | — |
| adding B to A (A_B − A) | -0.03430 | [-0.0823, 0.0089] | 3/14 (7 at 0) | — | 0.1719 | 1 | — |
| adding A to B (A_B − B) | 1.29551 | [0.8913, 1.6908] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.44799 | [1.1996, 1.7524] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.01012 | [-0.0248, 0.0037] | 4/14 (6 at 0) | — | 0.2344 | 1 | — |
| adding B to S (B_S − S) | 0.06545 | [-0.2941, 0.4449] | 8/14 (2 at 0) | — | 0.613 | 1 | — |
| adding S to B (B_S − B) | -0.06285 | [-0.1482, 0.0022] | 5/14 (5 at 0) | — | 0.08203 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.01842 | [-0.0453, 0.0035] | 4/14 (5 at 0) | — | 0.2188 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.04260 | [-0.1018, 0.0095] | 3/14 (7 at 0) | — | 0.1797 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.33994 | [0.9390, 1.7252] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.29586 | [0.2420, 0.3719] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.26156 | [0.2126, 0.3496] | 13/14 | — | 0.9998 | 1 | — |
| A_S − repeat-own-last | 0.28574 | [0.2386, 0.3660] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.24314 | [0.1818, 0.3486] | 13/14 | — | 0.9998 | 1 | — |

### v5 (one-SE)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.04770 | [-0.0816, 0.2977] | 5/14 (3 at 0) | — | 0.5552 | 1 | — |
| adding A to none (A − none) | 1.50564 | [1.1967, 1.9137] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.17583 | [-0.2239, 0.7162] | 8/14 (3 at 0) | — | 0.7666 | 1 | — |
| adding B to A (A_B − A) | -0.03597 | [-0.0820, 0.0050] | 2/14 (9 at 0) | — | 0.1562 | 1 | — |
| adding A to B (A_B − B) | 1.29384 | [0.8925, 1.6906] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.45788 | [1.2029, 1.7576] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00006 | [-0.0003, 0.0000] | 1/14 (12 at 0) | — | 0.5 | 1 | — |
| adding B to S (B_S − S) | 0.04786 | [-0.3120, 0.4171] | 7/14 (4 at 0) | — | 0.5986 | 1 | — |
| adding S to B (B_S − B) | -0.08028 | [-0.2136, 0.0016] | 4/14 (6 at 0) | — | 0.08203 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.01857 | [-0.0458, 0.0000] | 0/14 (11 at 0) | — | 0.125 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.05447 | [-0.1267, 0.0042] | 2/14 (8 at 0) | — | 0.1406 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.35554 | [0.9592, 1.7427] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.29586 | [0.2420, 0.3719] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.25989 | [0.2107, 0.3484] | 13/14 | — | 0.9998 | 1 | — |
| A_S − repeat-own-last | 0.29580 | [0.2419, 0.3717] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.24132 | [0.1809, 0.3448] | 13/14 | — | 0.9998 | 1 | — |

## Notes

- Day-bootstrap CIs are approximate with 14 days, and limited data and capacity push contrasts toward zero: a CI that spans zero may mean too little data (PLAN §9).
- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly favours keeping S; the one-SE rule counters this (DEC-23).
- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).
- This report establishes no grammar or causal mechanism.
- Per fold, condition and rule, the candidates and their validation log-likelihoods are in `data/results/rq1_logistic_v5/selection.jsonl`; v5's predictions are rebuilt from it and `data/results/rq1_logistic_v4/predictions.jsonl` (`logistic_v5.rebuild_predictions`).

## Inputs

SHA-256 of every input:

- `config/rq1_logistic_v5.json`: `664c09f32f4c3ede20a72fac2ab52685e67204b7b96e38e3d36626ec17c23e67`
- `config/rq1_logistic_v4.json`: `3cb6fe984dce1ae2d4f0a675df1f1ef01cdfb68b2927f241b070762a391ad4d1`
- `config/rq1_evaluation_v4.json`: `17ea02e16f8dbebb4fde8a3ea6c0785be1ed8ef1b03d89535cd920043aa5d6a0`
- `data/processed/rq1_folds/manifest.json`: `66fb93262c981d2c3fa98f95d9e4d4ee091c9dc492fa7c36cca5b62a57fc676f`
- `data/processed/rq1_targets.jsonl`: `9c7245c36fcc392dec0e67db08b7d943cd89107eb1b34e3b2b8cc55e1934be02`
- `data/results/rq1_logistic_v4/models.jsonl`: `51aab96c3eeda17565a704dd86c5b3f8368782a3000fbef3a9ca5912827bda5c`
- `data/results/rq1_logistic_v4/l2_selection.jsonl`: `9c4d3888cd39b444ab5680a501523268c6b3ef2bd170a574c1867f48815be39e`
- `data/results/rq1_logistic_v4/predictions.jsonl`: `3572cd4c3f48189f628baf3a7e0162882d04e2071ecad7bd85c09fd902c79e51`
- `data/results/rq1_logistic_v4/baseline_predictions.jsonl`: `b294dec536b7a8b418448c16ce2e646587a991a49e7daf6da36df00b124edb20`
- `data/results/rq1_logistic_v4/evaluation/summary.json`: `b782c2705b001a346d350c6043b725b047bf2bb6c7e0a801546d33fa0b998980`
- `data/results/rq1_logistic_v4/evaluation/scored_predictions.jsonl`: `cfaf3a8a600bb22d7789d72a33eb5cf3b6aa078902cd5b9715d7afacac488926`
