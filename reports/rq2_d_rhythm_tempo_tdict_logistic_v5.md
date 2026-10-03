# RQ2 logistic regression v5, inventory D′ (DEC-23)

Logistic v5 is logistic v4 with optional ignorance by selection (DEC-23, PLAN `[M4]`). For each fold and condition, the candidates are v4's chosen fits of every sub-condition (A, B and S each on or absent; all absent is the fold's smoothed prior, condition none). The candidate with the highest validation-day log-likelihood is chosen (exact ties: fewer blocks, then the order A, B, S), and its stored v4 test-day predictions become the condition's. Each sub-condition's penalties were already chosen by v4's rule, so nothing is refitted: everything comes from `data/results/rq2_d_rhythm_tempo_tdict_logistic_v4/` (models.jsonl, l2_selection.jsonl, predictions.jsonl). Sensitivity: DEC-21's one-SE rule (the fewest blocks within one standard error of the best; SE of the per-target paired validation differences). Every log-likelihood below is in bits per coda, from the shared evaluation module. Inventory D′: Sharma rhythm × tempo (tempos-dict.p cuts) (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy and the model are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v5.md.

## Checks before any test-day number

- Reproduction (gate 1): the validation log-likelihoods of all 98 v4 fits, recomputed from models.jsonl, match l2_selection.jsonl to 0 bits at most (tolerance 1e-09); their test-day probabilities, and the prior's, match predictions.jsonl to 0 at most.
- Validation monotonicity (gate 2): under the max rule all 168 add-one-input contrasts (12 pairs × 14 validation days) are ≥ 0 on the validation day; the smallest is 0 bits.
- The smallest validation-day gap between a chosen candidate and the runner-up is 8.96e-05 bits (test-2016-05-12, condition A_S).
- Every v5 test-day log-likelihood equals v4's for the chosen sub-condition on the same coda to 0 bits at most (tolerance 1e-12; recomputing log2 on another machine can change the last bit).

## Stop rules

DEC-23's rule: each of the 16 checks (12 add-one-input contrasts, 4 A-conditions against repeat-own-last) gets an exact one-sided sign-flip test on the day sums of per-coda differences (alternative: mean below zero; all 2^14 sign assignments; statistic: token-weighted mean), with Holm's correction across the 16; a check fires when its adjusted p < 0.05. DEC-16's old rule (day-bootstrap CI entirely below zero) is shown next to it.

| Model | DEC-23 rule (sign-flip, Holm) | DEC-16 rule (CI below zero) |
|---|---|---|
| v4 | none fired | add_input:B_S-B: contrast CI entirely below zero [-0.1020, -0.0034] |
| v5 (max) | none fired | add_input:B_S-B: contrast CI entirely below zero [-0.1028, -0.0042] |
| v5 (one-SE) | none fired | add_input:B_S-B: contrast CI entirely below zero [-0.0994, -0.0002] |

Governing status (DEC-23's rule on v5 (max)): none fired.

## Lattice, v4 and v5

| Condition | v4 LL | v5 LL | v5 − v4 | v5 one-SE LL | v4 accuracy | v5 accuracy | v4 macro-F1 | v5 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | -3.72492 | -3.72492 | 0.00000 | -3.72492 | 39.496% | 39.496% | 0.0157 | 0.0157 |
| S | -3.83483 | -3.83559 | -0.00076 | -3.72845 | 37.880% | 37.880% | 0.0188 | 0.0188 |
| A | -1.70989 | -1.70989 | 0.00000 | -1.70989 | 75.436% | 75.436% | 0.2674 | 0.2674 |
| B | -3.16085 | -3.16085 | 0.00000 | -3.16085 | 48.158% | 48.158% | 0.0509 | 0.0509 |
| A_B | -1.73488 | -1.71070 | 0.02418 | -1.71243 | 73.368% | 75.048% | 0.2514 | 0.2460 |
| A_S | -1.71337 | -1.71338 | -0.00001 | -1.70972 | 75.048% | 75.048% | 0.2656 | 0.2655 |
| B_S | -3.20006 | -3.20071 | -0.00065 | -3.19597 | 47.705% | 47.770% | 0.0495 | 0.0497 |
| A_B_S | -1.73971 | -1.71235 | 0.02736 | -1.71226 | 73.239% | 74.661% | 0.2478 | 0.2441 |
| majority | n/a | n/a | — | n/a | 39.496% | 39.496% | 0.0157 | 0.0157 |
| repeat_own_last | -2.12127 | -2.12127 | — | -2.12127 | 74.467% | 74.467% | 0.3044 | 0.3044 |

## Sub-conditions chosen

Folds (of 14) on which each sub-condition was chosen: max rule, with the one-SE rule in parentheses. A dash marks a sub-condition that is not a candidate for that condition.

| Condition | none | A | B | S | A_B | A_S | B_S | A_B_S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 14 (14) | — | — | — | — | — | — | — |
| S | 1 (4) | — | — | 13 (10) | — | — | — | — |
| A | 0 (0) | 14 (14) | — | — | — | — | — | — |
| B | 0 (0) | — | 14 (14) | — | — | — | — | — |
| A_B | 0 (0) | 8 (12) | 0 (0) | — | 6 (2) | — | — | — |
| A_S | 0 (0) | 10 (13) | — | 0 (0) | — | 4 (1) | — | — |
| B_S | 0 (0) | — | 6 (8) | 0 (0) | — | — | 8 (6) | — |
| A_B_S | 0 (0) | 4 (11) | 0 (0) | 0 (0) | 3 (2) | 4 (1) | 0 (0) | 3 (0) |

Per fold: the chosen sub-condition under the max rule, with the one-SE choice after an arrow where it differs.

| Test day | S | A | B | A_B | A_S | B_S | A_B_S |
|---|---|---|---|---|---|---|---|
| 2014-04-16 | S | A | B | A | A_S → A | B | A_S → A |
| 2014-04-24 | S | A | B | A | A_S → A | B_S | A_S → A |
| 2014-04-29 | S → none | A | B | A_B → A | A | B | A_B_S → A |
| 2015-03-02 | S | A | B | A | A_S → A | B_S | A_S → A |
| 2015-03-19 | S | A | B | A | A | B | A |
| 2015-03-26 | S → none | A | B | A_B → A | A | B_S → B | A_B → A |
| 2015-03-31 | S | A | B | A | A | B_S → B | A |
| 2015-04-01 | S | A | B | A | A_S | B_S | A_S |
| 2015-04-07 | S | A | B | A_B → A | A | B | A_B → A |
| 2015-04-10 | S → none | A | B | A | A | B | A |
| 2016-04-12 | S | A | B | A | A | B_S | A |
| 2016-04-24 | S | A | B | A_B | A | B | A_B |
| 2016-05-12 | none | A | B | A_B → A | A | B_S | A_B_S → A |
| 2016-05-13 | S | A | B | A_B | A | B_S | A_B_S → A_B |

## RQ1 contrasts (`[M2]`)

Paired per-coda differences; positive days, with the days whose difference is exactly zero; day-bootstrap 95% CI (10,000 replicates); exact two-sided sign-flip p on the day sums (DEC-23).

| Contrast | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |
|---|---|---:|---:|---|---:|---:|
| own_history | v4 | 1.46035 | 1.41927 | [1.1679, 1.7256] | 14/14 | 0.0001221 |
| own_history | v5 (max) | 1.48836 | 1.42281 | [1.1854, 1.7596] | 14/14 | 0.0001221 |
| own_history | v5 (one-SE) | 1.48371 | 1.40917 | [1.1751, 1.7560] | 14/14 | 0.0001221 |
| partner_content | v4 | -0.02634 | 0.00632 | [-0.0534, 0.0232] | 6/14 | 0.3029 |
| partner_content | v5 (max) | 0.00103 | 0.00762 | [-0.0113, 0.0174] | 4/14 (8 at 0) | 0.875 |
| partner_content | v5 (one-SE) | -0.00254 | -0.00567 | [-0.0143, 0.0037] | 1/14 (12 at 0) | 1 |
| timing | v4 | -0.00484 | 0.00420 | [-0.0144, 0.0076] | 8/14 | 0.6086 |
| timing | v5 (max) | -0.00165 | 0.00583 | [-0.0078, 0.0076] | 3/14 (7 at 0) | 0.7031 |
| timing | v5 (one-SE) | 0.00017 | 0.00015 | [0.0000, 0.0006] | 1/14 (13 at 0) | 1 |

Sign of each token-weighted mean and where its CI lies:

| Contrast | v4 | v5 (max) | v5 (one-SE) |
|---|---|---|---|
| own_history | +, CI > 0 | +, CI > 0 | +, CI > 0 |
| partner_content | −, CI spans 0 | +, CI spans 0 | −, CI spans 0 |
| timing | −, CI spans 0 | −, CI spans 0 | +, CI spans 0 |

The sign or CI position differs between v4 and v5 for: partner_content. Both are reported.

## The sixteen stop-rule checks

Per-coda differences, token-weighted means, day-bootstrap CI and positive days (with the days whose difference is exactly zero); DEC-16: CI entirely below zero; one-sided sign-flip p (alternative: mean below zero) and its Holm adjustment across the 16; DEC-23: adjusted p below alpha.

### v4

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.10991 | [-0.3198, 0.1046] | 4/14 | — | 0.2482 | 1 | — |
| adding A to none (A − none) | 2.01503 | [1.7918, 2.3363] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.56407 | [0.2392, 1.0026] | 11/14 | — | 0.9979 | 1 | — |
| adding B to A (A_B − A) | -0.02498 | [-0.0508, 0.0219] | 6/14 | — | 0.1438 | 1 | — |
| adding A to B (A_B − B) | 1.42598 | [1.1248, 1.6933] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 2.12146 | [1.8946, 2.3821] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00348 | [-0.0088, 0.0006] | 4/14 | — | 0.2633 | 1 | — |
| adding B to S (B_S − S) | 0.63477 | [0.2870, 0.9818] | 12/14 | — | 0.9991 | 1 | — |
| adding S to B (B_S − B) | -0.03921 | [-0.1020, -0.0034] | 7/14 | fires | 0.0531 | 0.8496 | — |
| adding S to A_B (A_B_S − A_B) | -0.00484 | [-0.0144, 0.0076] | 8/14 | — | 0.3043 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.02634 | [-0.0534, 0.0232] | 6/14 | — | 0.1514 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.46035 | [1.1679, 1.7256] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.41138 | [0.3180, 0.5543] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.38639 | [0.2774, 0.5595] | 14/14 | — | 1 | 1 | — |
| A_S − repeat-own-last | 0.40790 | [0.3104, 0.5542] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.38156 | [0.2661, 0.5597] | 14/14 | — | 1 | 1 | — |

### v5 (max)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.11067 | [-0.3205, 0.1042] | 3/14 (1 at 0) | — | 0.2471 | 1 | — |
| adding A to none (A − none) | 2.01503 | [1.7918, 2.3363] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.56407 | [0.2392, 1.0026] | 11/14 | — | 0.9979 | 1 | — |
| adding B to A (A_B − A) | -0.00081 | [-0.0164, 0.0167] | 4/14 (8 at 0) | — | 0.5 | 1 | — |
| adding A to B (A_B − B) | 1.45015 | [1.1441, 1.7231] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 2.12222 | [1.8953, 2.3823] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00348 | [-0.0088, 0.0004] | 1/14 (10 at 0) | — | 0.3125 | 1 | — |
| adding B to S (B_S − S) | 0.63488 | [0.2887, 0.9806] | 12/14 | — | 0.9991 | 1 | — |
| adding S to B (B_S − B) | -0.03986 | [-0.1028, -0.0042] | 3/14 (6 at 0) | fires | 0.04688 | 0.75 | — |
| adding S to A_B (A_B_S − A_B) | -0.00165 | [-0.0078, 0.0076] | 3/14 (7 at 0) | — | 0.3516 | 1 | — |
| adding B to A_S (A_B_S − A_S) | 0.00103 | [-0.0113, 0.0174] | 4/14 (8 at 0) | — | 0.5781 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.48836 | [1.1854, 1.7596] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.41138 | [0.3180, 0.5543] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.41057 | [0.3161, 0.5601] | 14/14 | — | 1 | 1 | — |
| A_S − repeat-own-last | 0.40789 | [0.3107, 0.5542] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.40892 | [0.3110, 0.5620] | 14/14 | — | 1 | 1 | — |

### v5 (one-SE)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.00352 | [-0.0724, 0.1314] | 2/14 (4 at 0) | — | 0.4912 | 1 | — |
| adding A to none (A − none) | 2.01503 | [1.7918, 2.3363] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.56407 | [0.2392, 1.0026] | 11/14 | — | 0.9979 | 1 | — |
| adding B to A (A_B − A) | -0.00254 | [-0.0143, 0.0037] | 1/14 (12 at 0) | — | 0.5 | 1 | — |
| adding A to B (A_B − B) | 1.44842 | [1.1419, 1.7172] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 2.01872 | [1.8024, 2.3055] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | 0.00017 | [0.0000, 0.0006] | 1/14 (13 at 0) | — | 1 | 1 | — |
| adding B to S (B_S − S) | 0.53247 | [0.2380, 0.8833] | 12/14 | — | 0.9991 | 1 | — |
| adding S to B (B_S − B) | -0.03512 | [-0.0994, -0.0002] | 2/14 (8 at 0) | fires | 0.0625 | 1 | — |
| adding S to A_B (A_B_S − A_B) | 0.00017 | [0.0000, 0.0006] | 1/14 (13 at 0) | — | 1 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.00254 | [-0.0143, 0.0037] | 1/14 (12 at 0) | — | 0.5 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.48371 | [1.1751, 1.7560] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.41138 | [0.3180, 0.5543] | 14/14 | — | 1 | 1 | — |
| A_B − repeat-own-last | 0.40884 | [0.3149, 0.5519] | 14/14 | — | 1 | 1 | — |
| A_S − repeat-own-last | 0.41155 | [0.3180, 0.5544] | 14/14 | — | 1 | 1 | — |
| A_B_S − repeat-own-last | 0.40901 | [0.3149, 0.5524] | 14/14 | — | 1 | 1 | — |

## Notes

- Day-bootstrap CIs are approximate with 14 days, and limited data and capacity push contrasts toward zero: a CI that spans zero may mean too little data (PLAN §9).
- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly favours keeping S; the one-SE rule counters this (DEC-23).
- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).
- This report establishes no grammar or causal mechanism.
- Per fold, condition and rule, the candidates and their validation log-likelihoods are in `data/results/rq2_d_rhythm_tempo_tdict_logistic_v5/selection.jsonl`; v5's predictions are rebuilt from it and `data/results/rq2_d_rhythm_tempo_tdict_logistic_v4/predictions.jsonl` (`logistic_v5.rebuild_predictions`).

## Inputs

SHA-256 of every input:

- `config/rq2_d_rhythm_tempo_tdict_logistic_v5.json`: `d25aed13dbb445872db6b7b95d78ea48f5e025c7a2db9eb3c3b77087e0f82a6b`
- `config/rq2_d_rhythm_tempo_tdict_logistic_v4.json`: `82c83bae9dab723da3f1e9a40e70fe212abe246d67a5cf265ed279743986f269`
- `config/rq2_d_rhythm_tempo_tdict_evaluation_v4.json`: `6b91f58519ec55839cd40a5ba44312d91016fdfdf6f116d1f61ac9480abe916d`
- `data/processed/rq2_d_rhythm_tempo_tdict_folds/manifest.json`: `957c7cf00d4303a195d5b612db0f32036c6b1d960f35a00ac994b4470e02da47`
- `data/processed/rq2_d_rhythm_tempo_tdict_targets.jsonl`: `154eac17a3ea34bbe27f66d3c50b7019b1c62df4c5ee5262b6a7f190fdafa9d6`
- `data/results/rq2_d_rhythm_tempo_tdict_logistic_v4/models.jsonl`: `136be135896822e68358a2f190069eb6f9eed4e564b7df4f08d5d32c33854f47`
- `data/results/rq2_d_rhythm_tempo_tdict_logistic_v4/l2_selection.jsonl`: `57cedeaa012dd9b1016880eb6951613dbaebaabf87bfed773638df882915e6e9`
- `data/results/rq2_d_rhythm_tempo_tdict_logistic_v4/predictions.jsonl`: `19cfd4e06db0e75657950c4cbb8e6580e65dd7318064811ec3bdea4b5595a3a7`
- `data/results/rq2_d_rhythm_tempo_tdict_logistic_v4/baseline_predictions.jsonl`: `cc796ae545d854d482f6332d1eab51a473f9598c0b0254f53af800d9c80f44e5`
- `data/results/rq2_d_rhythm_tempo_tdict_logistic_v4/evaluation/summary.json`: `505dcf8cac982a3848a590fcd296df60704dfc5933ca9fd3758a9d9659d9c047`
- `data/results/rq2_d_rhythm_tempo_tdict_logistic_v4/evaluation/scored_predictions.jsonl`: `06356cd5614f8d84b303b9f5d45618e014569fc01c5bb87f94c4f3ab8382fa47`
