# RQ2 logistic regression v5, inventory C (DEC-23)

Logistic v5 is logistic v4 with optional ignorance by selection (DEC-23, PLAN `[M4]`). For each fold and condition, the candidates are v4's chosen fits of every sub-condition (A, B and S each on or absent; all absent is the fold's smoothed prior, condition none). The candidate with the highest validation-day log-likelihood is chosen (exact ties: fewer blocks, then the order A, B, S), and its stored v4 test-day predictions become the condition's. Each sub-condition's penalties were already chosen by v4's rule, so nothing is refitted: everything comes from `data/results/rq2_c_rhythm_logistic_v4/` (models.jsonl, l2_selection.jsonl, predictions.jsonl). Sensitivity: DEC-21's one-SE rule (the fewest blocks within one standard error of the best; SE of the per-target paired validation differences). Every log-likelihood below is in bits per coda, from the shared evaluation module. Inventory C: Sharma rhythm (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy and the model are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v5.md.

## Checks before any test-day number

- Reproduction (gate 1): the validation log-likelihoods of all 98 v4 fits, recomputed from models.jsonl, match l2_selection.jsonl to 0 bits at most (tolerance 1e-09); their test-day probabilities, and the prior's, match predictions.jsonl to 0 at most.
- Validation monotonicity (gate 2): under the max rule all 168 add-one-input contrasts (12 pairs × 14 validation days) are ≥ 0 on the validation day; the smallest is 0 bits.
- The smallest validation-day gap between a chosen candidate and the runner-up is 0.000108 bits (test-2015-04-10, condition A_S).
- Every v5 test-day log-likelihood equals v4's for the chosen sub-condition on the same coda to 0 bits at most (tolerance 1e-12; recomputing log2 on another machine can change the last bit).

## Stop rules

DEC-23's rule: each of the 16 checks (12 add-one-input contrasts, 4 A-conditions against repeat-own-last) gets an exact one-sided sign-flip test on the day sums of per-coda differences (alternative: mean below zero; all 2^14 sign assignments; statistic: token-weighted mean), with Holm's correction across the 16; a check fires when its adjusted p < 0.05. DEC-16's old rule (day-bootstrap CI entirely below zero) is shown next to it.

| Model | DEC-23 rule (sign-flip, Holm) | DEC-16 rule (CI below zero) |
|---|---|---|
| v4 | none fired | add_input:B_S-B: contrast CI entirely below zero [-0.0899, -0.0188] |
| v5 (max) | none fired | add_input:B_S-B: contrast CI entirely below zero [-0.1141, -0.0120] |
| v5 (one-SE) | none fired | none fired |

Governing status (DEC-23's rule on v5 (max)): none fired.

## Lattice, v4 and v5

| Condition | v4 LL | v5 LL | v5 − v4 | v5 one-SE LL | v4 accuracy | v5 accuracy | v4 macro-F1 | v5 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | -2.15209 | -2.15209 | 0.00000 | -2.15209 | 61.926% | 61.926% | 0.0450 | 0.0450 |
| S | -2.16882 | -2.16627 | 0.00255 | -2.16556 | 61.345% | 61.345% | 0.0487 | 0.0487 |
| A | -1.20748 | -1.20748 | 0.00000 | -1.20748 | 78.410% | 78.410% | 0.2291 | 0.2291 |
| B | -2.05256 | -2.07215 | -0.01958 | -2.10057 | 65.223% | 64.900% | 0.0907 | 0.0882 |
| A_B | -1.23267 | -1.24525 | -0.01258 | -1.24901 | 77.957% | 77.957% | 0.2038 | 0.1958 |
| A_S | -1.21169 | -1.21234 | -0.00066 | -1.20915 | 78.539% | 78.539% | 0.2251 | 0.2251 |
| B_S | -2.09691 | -2.11714 | -0.02023 | -2.10320 | 63.736% | 63.219% | 0.0877 | 0.0848 |
| A_B_S | -1.23585 | -1.24926 | -0.01341 | -1.24901 | 77.957% | 78.087% | 0.2050 | 0.1970 |
| majority | n/a | n/a | — | n/a | 61.926% | 61.926% | 0.0450 | 0.0450 |
| repeat_own_last | -1.42054 | -1.42054 | — | -1.42054 | 78.539% | 78.539% | 0.3848 | 0.3848 |

## Sub-conditions chosen

Folds (of 14) on which each sub-condition was chosen: max rule, with the one-SE rule in parentheses. A dash marks a sub-condition that is not a candidate for that condition.

| Condition | none | A | B | S | A_B | A_S | B_S | A_B_S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 14 (14) | — | — | — | — | — | — | — |
| S | 3 (4) | — | — | 11 (10) | — | — | — | — |
| A | 0 (0) | 14 (14) | — | — | — | — | — | — |
| B | 3 (4) | — | 11 (10) | — | — | — | — | — |
| A_B | 0 (0) | 6 (12) | 0 (0) | — | 8 (2) | — | — | — |
| A_S | 0 (0) | 5 (10) | — | 0 (0) | — | 9 (4) | — | — |
| B_S | 2 (2) | — | 2 (7) | 2 (2) | — | — | 8 (3) | — |
| A_B_S | 0 (0) | 2 (10) | 0 (0) | 0 (0) | 3 (1) | 4 (2) | 0 (0) | 5 (1) |

Per fold: the chosen sub-condition under the max rule, with the one-SE choice after an arrow where it differs.

| Test day | S | A | B | A_B | A_S | B_S | A_B_S |
|---|---|---|---|---|---|---|---|
| 2014-04-16 | S | A | B | A | A_S → A | B_S → B | A_S → A |
| 2014-04-24 | S | A | B | A_B → A | A_S → A | B_S → B | A_B_S → A |
| 2014-04-29 | S → none | A | B | A_B → A | A | B_S → B | A_B → A |
| 2015-03-02 | S | A | B | A_B | A_S → A | B_S → B | A_B_S → A_B |
| 2015-03-19 | S | A | B | A_B → A | A | B | A_B → A |
| 2015-03-26 | none | A | B | A_B → A | A | B | A_B → A |
| 2015-03-31 | S | A | B | A | A_S | B_S | A_S |
| 2015-04-01 | S | A | B | A | A_S | B_S | A_S |
| 2015-04-07 | none | A | none | A_B → A | A_S → A | none | A_B_S → A |
| 2015-04-10 | S | A | none | A | A | S | A |
| 2016-04-12 | S | A | B → none | A | A_S → A | S | A_S → A |
| 2016-04-24 | S | A | B | A_B → A | A_S | B_S → B | A_B_S → A |
| 2016-05-12 | none | A | none | A | A | none | A |
| 2016-05-13 | S | A | B | A_B | A_S | B_S | A_B_S |

## RQ1 contrasts (`[M2]`)

Paired per-coda differences; positive days, with the days whose difference is exactly zero; day-bootstrap 95% CI (10,000 replicates); exact two-sided sign-flip p on the day sums (DEC-23).

| Contrast | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |
|---|---|---:|---:|---|---:|---:|
| own_history | v4 | 0.86106 | 0.86068 | [0.5594, 1.2086] | 14/14 | 0.0001221 |
| own_history | v5 (max) | 0.86788 | 0.93087 | [0.5500, 1.2329] | 14/14 | 0.0001221 |
| own_history | v5 (one-SE) | 0.85419 | 0.89772 | [0.5422, 1.2094] | 14/14 | 0.0001221 |
| partner_content | v4 | -0.02416 | -0.00377 | [-0.0859, 0.0361] | 7/14 | 0.8042 |
| partner_content | v5 (max) | -0.03692 | -0.01546 | [-0.0908, 0.0095] | 4/14 (6 at 0) | 0.4219 |
| partner_content | v5 (one-SE) | -0.03986 | -0.02749 | [-0.0927, 0.0014] | 1/14 (11 at 0) | 0.5 |
| timing | v4 | -0.00317 | -0.01248 | [-0.0137, 0.0046] | 7/14 | 0.4388 |
| timing | v5 (max) | -0.00401 | -0.01473 | [-0.0171, 0.0048] | 5/14 (5 at 0) | 0.4219 |
| timing | v5 (one-SE) | -0.00000 | -0.00643 | [-0.0082, 0.0065] | 2/14 (11 at 0) | 1 |

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
| adding S to none (S − none) | -0.01674 | [-0.0485, 0.0216] | 6/14 | — | 0.1898 | 1 | — |
| adding A to none (A − none) | 0.94461 | [0.6806, 1.2047] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.09952 | [-0.1195, 0.3332] | 10/14 | — | 0.79 | 1 | — |
| adding B to A (A_B − A) | -0.02520 | [-0.0873, 0.0336] | 8/14 | — | 0.3839 | 1 | — |
| adding A to B (A_B − B) | 0.81989 | [0.5188, 1.1634] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 0.95714 | [0.6920, 1.2182] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00421 | [-0.0198, 0.0054] | 8/14 | — | 0.21 | 1 | — |
| adding B to S (B_S − S) | 0.07191 | [-0.1334, 0.2785] | 10/14 | — | 0.7322 | 1 | — |
| adding S to B (B_S − B) | -0.04435 | [-0.0899, -0.0188] | 5/14 | fires | 0.003906 | 0.0625 | — |
| adding S to A_B (A_B_S − A_B) | -0.00317 | [-0.0137, 0.0046] | 7/14 | — | 0.2194 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.02416 | [-0.0859, 0.0361] | 7/14 | — | 0.4021 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 0.86106 | [0.5594, 1.2086] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.21307 | [0.1638, 0.2877] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.18787 | [0.0927, 0.2976] | 12/14 | — | 0.9996 | 1 | — |
| A_S − repeat-own-last | 0.20886 | [0.1600, 0.2809] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.18470 | [0.0887, 0.2936] | 12/14 | — | 0.9991 | 1 | — |

### v5 (max)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.01419 | [-0.0464, 0.0243] | 5/14 (3 at 0) | — | 0.2295 | 1 | — |
| adding A to none (A − none) | 0.94461 | [0.6806, 1.2047] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.07994 | [-0.1377, 0.2954] | 8/14 (3 at 0) | — | 0.7437 | 1 | — |
| adding B to A (A_B − A) | -0.03777 | [-0.0919, 0.0096] | 4/14 (6 at 0) | — | 0.2148 | 1 | — |
| adding A to B (A_B − B) | 0.82689 | [0.5205, 1.1816] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 0.95393 | [0.6862, 1.2178] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00486 | [-0.0204, 0.0047] | 5/14 (5 at 0) | — | 0.1797 | 1 | — |
| adding B to S (B_S − S) | 0.04913 | [-0.1531, 0.2365] | 7/14 (4 at 0) | — | 0.6709 | 1 | — |
| adding S to B (B_S − B) | -0.04499 | [-0.1141, -0.0120] | 2/14 (4 at 0) | fires | 0.01172 | 0.1875 | — |
| adding S to A_B (A_B_S − A_B) | -0.00401 | [-0.0171, 0.0048] | 5/14 (5 at 0) | — | 0.2109 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.03692 | [-0.0908, 0.0095] | 4/14 (6 at 0) | — | 0.2109 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 0.86788 | [0.5500, 1.2329] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.21307 | [0.1638, 0.2877] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.17529 | [0.0858, 0.2848] | 12/14 | — | 0.9996 | 1 | — |
| A_S − repeat-own-last | 0.20820 | [0.1598, 0.2799] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.17129 | [0.0808, 0.2791] | 12/14 | — | 0.9991 | 1 | — |

### v5 (one-SE)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.01347 | [-0.0456, 0.0253] | 5/14 (4 at 0) | — | 0.2441 | 1 | — |
| adding A to none (A − none) | 0.94461 | [0.6806, 1.2047] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.05152 | [-0.1613, 0.2471] | 7/14 (4 at 0) | — | 0.666 | 1 | — |
| adding B to A (A_B − A) | -0.04153 | [-0.0949, 0.0000] | 0/14 (12 at 0) | — | 0.25 | 1 | — |
| adding A to B (A_B − B) | 0.85157 | [0.5315, 1.2137] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 0.95641 | [0.6878, 1.2219] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00167 | [-0.0139, 0.0057] | 2/14 (10 at 0) | — | 0.375 | 1 | — |
| adding B to S (B_S − S) | 0.06236 | [-0.1332, 0.2458] | 7/14 (4 at 0) | — | 0.7178 | 1 | — |
| adding S to B (B_S − B) | -0.00263 | [-0.0250, 0.0240] | 2/14 (9 at 0) | — | 0.4688 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.00000 | [-0.0082, 0.0065] | 2/14 (11 at 0) | — | 0.5 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.03986 | [-0.0927, 0.0014] | 1/14 (11 at 0) | — | 0.25 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 0.85419 | [0.5422, 1.2094] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.21307 | [0.1638, 0.2877] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.17154 | [0.0834, 0.2769] | 12/14 | — | 0.9996 | 1 | — |
| A_S − repeat-own-last | 0.21140 | [0.1631, 0.2829] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.17154 | [0.0824, 0.2766] | 12/14 | — | 0.9994 | 1 | — |

## Notes

- Day-bootstrap CIs are approximate with 14 days, and limited data and capacity push contrasts toward zero: a CI that spans zero may mean too little data (PLAN §9).
- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly favours keeping S; the one-SE rule counters this (DEC-23).
- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).
- This report establishes no grammar or causal mechanism.
- Per fold, condition and rule, the candidates and their validation log-likelihoods are in `data/results/rq2_c_rhythm_logistic_v5/selection.jsonl`; v5's predictions are rebuilt from it and `data/results/rq2_c_rhythm_logistic_v4/predictions.jsonl` (`logistic_v5.rebuild_predictions`).

## Inputs

SHA-256 of every input:

- `config/rq2_c_rhythm_logistic_v5.json`: `7d4ddfb5b175d7ccd56bd7d4b22ac3b2c0428a12904704db8d1ea06ac4654daa`
- `config/rq2_c_rhythm_logistic_v4.json`: `31156b90253831aeb6c019cd3aadff241dee0e8a11a08f3dde5d9c6b6e8e87b3`
- `config/rq2_c_rhythm_evaluation_v4.json`: `1340383e1ec9a9eee677e546b488611ac6a1debd2c7530f725c188160be44a4f`
- `data/processed/rq2_c_rhythm_folds/manifest.json`: `7878a2c4d2c7a4c006173f72ae0477ae6c3389d55cb966428f0729df4e1a64f0`
- `data/processed/rq2_c_rhythm_targets.jsonl`: `a315defd8ae4c63f4f47963138e26436b34110b44bcfd1e52a2274b71dceb090`
- `data/results/rq2_c_rhythm_logistic_v4/models.jsonl`: `b1eb1cc5623e51bf9cd3c2e3c25e4776c0ef17145eec53ae6132b5c3bce4c78c`
- `data/results/rq2_c_rhythm_logistic_v4/l2_selection.jsonl`: `03902c127a670c4accaadbef5da10870b56e88599129a10d9c43dd0bf434a9e8`
- `data/results/rq2_c_rhythm_logistic_v4/predictions.jsonl`: `9610179a0f0ddde095ef87acc227cb6dd2dd97d6aafe700937ffaee6ed206777`
- `data/results/rq2_c_rhythm_logistic_v4/baseline_predictions.jsonl`: `b8bee5eeb5a880f3e3c0faf5e80fcf552b46ec99637077757a6f02b43af5482c`
- `data/results/rq2_c_rhythm_logistic_v4/evaluation/summary.json`: `0962d18528fcea5b064c5dc2cf3269677efe1f35502eb4fa4d8a998412d9d4e1`
- `data/results/rq2_c_rhythm_logistic_v4/evaluation/scored_predictions.jsonl`: `497e39df2eb53d8c039b7a09e9e34c7f1c5ddb55a25865c2a9db654efd1d2b5d`
