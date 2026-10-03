# RQ2 logistic regression v5, inventory D (DEC-23)

Logistic v5 is logistic v4 with optional ignorance by selection (DEC-23, PLAN `[M4]`). For each fold and condition, the candidates are v4's chosen fits of every sub-condition (A, B and S each on or absent; all absent is the fold's smoothed prior, condition none). The candidate with the highest validation-day log-likelihood is chosen (exact ties: fewer blocks, then the order A, B, S), and its stored v4 test-day predictions become the condition's. Each sub-condition's penalties were already chosen by v4's rule, so nothing is refitted: everything comes from `data/results/rq2_d_rhythm_tempo_logistic_v4/` (models.jsonl, l2_selection.jsonl, predictions.jsonl). Sensitivity: DEC-21's one-SE rule (the fewest blocks within one standard error of the best; SE of the per-target paired validation differences). Every log-likelihood below is in bits per coda, from the shared evaluation module. Inventory D: Sharma rhythm × tempo (KDE cuts) (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy and the model are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v5.md.

## Checks before any test-day number

- Reproduction (gate 1): the validation log-likelihoods of all 98 v4 fits, recomputed from models.jsonl, match l2_selection.jsonl to 0 bits at most (tolerance 1e-09); their test-day probabilities, and the prior's, match predictions.jsonl to 0 at most.
- Validation monotonicity (gate 2): under the max rule all 168 add-one-input contrasts (12 pairs × 14 validation days) are ≥ 0 on the validation day; the smallest is 0 bits.
- The smallest validation-day gap between a chosen candidate and the runner-up is 7.59e-05 bits (test-2016-05-13, condition A_B_S).
- Every v5 test-day log-likelihood equals v4's for the chosen sub-condition on the same coda to 0 bits at most (tolerance 1e-12; recomputing log2 on another machine can change the last bit).

## Stop rules

DEC-23's rule: each of the 16 checks (12 add-one-input contrasts, 4 A-conditions against repeat-own-last) gets an exact one-sided sign-flip test on the day sums of per-coda differences (alternative: mean below zero; all 2^14 sign assignments; statistic: token-weighted mean), with Holm's correction across the 16; a check fires when its adjusted p < 0.05. DEC-16's old rule (day-bootstrap CI entirely below zero) is shown next to it.

| Model | DEC-23 rule (sign-flip, Holm) | DEC-16 rule (CI below zero) |
|---|---|---|
| v4 | none fired | add_input:A_S-A: contrast CI entirely below zero [-0.0053, -0.0002]; add_input:B_S-B: contrast CI entirely below zero [-0.0861, -0.0049] |
| v5 (max) | none fired | add_input:A_S-A: contrast CI entirely below zero [-0.0053, -0.0002]; add_input:B_S-B: contrast CI entirely below zero [-0.0867, -0.0052]; add_input:A_B_S-A_B: contrast CI entirely below zero [-0.0055, -0.0005] |
| v5 (one-SE) | none fired | add_input:B_S-B: contrast CI entirely below zero [-0.0859, -0.0064] |

Governing status (DEC-23's rule on v5 (max)): none fired.

## Lattice, v4 and v5

| Condition | v4 LL | v5 LL | v5 − v4 | v5 one-SE LL | v4 accuracy | v5 accuracy | v4 macro-F1 | v5 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | -3.68461 | -3.68461 | 0.00000 | -3.68461 | 39.948% | 39.948% | 0.0178 | 0.0178 |
| S | -3.83344 | -3.83259 | 0.00085 | -3.68006 | 38.268% | 38.268% | 0.0212 | 0.0212 |
| A | -1.69981 | -1.69981 | 0.00000 | -1.69981 | 73.562% | 73.562% | 0.2854 | 0.2854 |
| B | -3.11125 | -3.11125 | 0.00000 | -3.11125 | 48.610% | 48.610% | 0.0585 | 0.0585 |
| A_B | -1.72460 | -1.70488 | 0.01972 | -1.69889 | 72.786% | 73.174% | 0.2585 | 0.2572 |
| A_S | -1.70237 | -1.70233 | 0.00003 | -1.70094 | 73.497% | 73.562% | 0.2847 | 0.2848 |
| B_S | -3.14666 | -3.14692 | -0.00026 | -3.14749 | 48.158% | 48.158% | 0.0595 | 0.0595 |
| A_B_S | -1.73282 | -1.70763 | 0.02518 | -1.70002 | 72.980% | 73.174% | 0.2652 | 0.2567 |
| majority | n/a | n/a | — | n/a | 39.948% | 39.948% | 0.0178 | 0.0178 |
| repeat_own_last | -2.12358 | -2.12358 | — | -2.12358 | 74.208% | 74.208% | 0.3358 | 0.3358 |

## Sub-conditions chosen

Folds (of 14) on which each sub-condition was chosen: max rule, with the one-SE rule in parentheses. A dash marks a sub-condition that is not a candidate for that condition.

| Condition | none | A | B | S | A_B | A_S | B_S | A_B_S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 14 (14) | — | — | — | — | — | — | — |
| S | 1 (4) | — | — | 13 (10) | — | — | — | — |
| A | 0 (0) | 14 (14) | — | — | — | — | — | — |
| B | 0 (0) | — | 14 (14) | — | — | — | — | — |
| A_B | 0 (0) | 8 (11) | 0 (0) | — | 6 (3) | — | — | — |
| A_S | 0 (0) | 10 (13) | — | 0 (0) | — | 4 (1) | — | — |
| B_S | 0 (0) | — | 5 (8) | 0 (0) | — | — | 9 (6) | — |
| A_B_S | 0 (0) | 4 (10) | 0 (0) | 0 (0) | 5 (3) | 4 (1) | 0 (0) | 1 (0) |

Per fold: the chosen sub-condition under the max rule, with the one-SE choice after an arrow where it differs.

| Test day | S | A | B | A_B | A_S | B_S | A_B_S |
|---|---|---|---|---|---|---|---|
| 2014-04-16 | S | A | B | A | A_S → A | B | A_S → A |
| 2014-04-24 | S | A | B | A | A_S → A | B_S | A_S → A |
| 2014-04-29 | S → none | A | B | A_B | A | B | A_B |
| 2015-03-02 | S | A | B | A | A_S → A | B_S | A_S → A |
| 2015-03-19 | S | A | B | A | A | B | A |
| 2015-03-26 | S → none | A | B | A_B → A | A | B_S → B | A_B → A |
| 2015-03-31 | S | A | B | A | A | B_S | A |
| 2015-04-01 | S | A | B | A | A_S | B_S | A_S |
| 2015-04-07 | S | A | B | A_B → A | A | B | A_B → A |
| 2015-04-10 | S → none | A | B | A | A | B_S → B | A |
| 2016-04-12 | S | A | B | A | A | B_S | A |
| 2016-04-24 | S | A | B | A_B | A | B | A_B |
| 2016-05-12 | none | A | B | A_B → A | A | B_S → B | A_B_S → A |
| 2016-05-13 | S | A | B | A_B | A | B_S | A_B |

## RQ1 contrasts (`[M2]`)

Paired per-coda differences; positive days, with the days whose difference is exactly zero; day-bootstrap 95% CI (10,000 replicates); exact two-sided sign-flip p on the day sums (DEC-23).

| Contrast | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |
|---|---|---:|---:|---|---:|---:|
| own_history | v4 | 1.41385 | 1.35576 | [1.1414, 1.6583] | 14/14 | 0.0001221 |
| own_history | v5 (max) | 1.43929 | 1.33648 | [1.1381, 1.7193] | 14/14 | 0.0001221 |
| own_history | v5 (one-SE) | 1.44747 | 1.34333 | [1.1462, 1.7239] | 14/14 | 0.0001221 |
| partner_content | v4 | -0.03045 | 0.02203 | [-0.0819, 0.0509] | 6/14 | 0.5426 |
| partner_content | v5 (max) | -0.00530 | 0.00101 | [-0.0180, 0.0052] | 3/14 (8 at 0) | 0.6875 |
| partner_content | v5 (one-SE) | 0.00093 | 0.00643 | [-0.0014, 0.0049] | 2/14 (11 at 0) | 0.75 |
| timing | v4 | -0.00822 | -0.00685 | [-0.0176, 0.0015] | 6/14 | 0.2749 |
| timing | v5 (max) | -0.00275 | -0.00254 | [-0.0055, -0.0005] | 0/14 (9 at 0) | 0.0625 |
| timing | v5 (one-SE) | -0.00113 | -0.00099 | [-0.0040, 0.0000] | 0/14 (13 at 0) | 1 |

Sign of each token-weighted mean and where its CI lies:

| Contrast | v4 | v5 (max) | v5 (one-SE) |
|---|---|---|---|
| own_history | +, CI > 0 | +, CI > 0 | +, CI > 0 |
| partner_content | −, CI spans 0 | −, CI spans 0 | +, CI spans 0 |
| timing | −, CI spans 0 | −, CI < 0 | −, CI spans 0 |

The sign or CI position differs between v4 and v5 for: timing. Both are reported.

## The sixteen stop-rule checks

Per-coda differences, token-weighted means, day-bootstrap CI and positive days (with the days whose difference is exactly zero); DEC-16: CI entirely below zero; one-sided sign-flip p (alternative: mean below zero) and its Holm adjustment across the 16; DEC-23: adjusted p below alpha.

### v4

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.14883 | [-0.4447, 0.1157] | 4/14 | — | 0.2536 | 1 | — |
| adding A to none (A − none) | 1.98479 | [1.7639, 2.2831] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.57335 | [0.2486, 1.0170] | 12/14 | — | 0.9991 | 1 | — |
| adding B to A (A_B − A) | -0.02478 | [-0.0706, 0.0531] | 6/14 | — | 0.2832 | 1 | — |
| adding A to B (A_B − B) | 1.38666 | [1.1007, 1.6433] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 2.13107 | [1.8634, 2.4085] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00255 | [-0.0053, -0.0002] | 4/14 | fires | 0.03271 | 0.5234 | — |
| adding B to S (B_S − S) | 0.68677 | [0.3032, 1.0665] | 12/14 | — | 0.9997 | 1 | — |
| adding S to B (B_S − B) | -0.03541 | [-0.0861, -0.0049] | 6/14 | fires | 0.0473 | 0.7095 | — |
| adding S to A_B (A_B_S − A_B) | -0.00822 | [-0.0176, 0.0015] | 6/14 | — | 0.1375 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.03045 | [-0.0819, 0.0509] | 6/14 | — | 0.2713 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.41385 | [1.1414, 1.6583] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.42377 | [0.3526, 0.5332] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.39899 | [0.3023, 0.5540] | 13/14 | — | 0.9999 | 1 | — |
| A_S − repeat-own-last | 0.42122 | [0.3503, 0.5296] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.39077 | [0.2891, 0.5484] | 13/14 | — | 0.9999 | 1 | — |

### v5 (max)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.14798 | [-0.4445, 0.1165] | 4/14 (1 at 0) | — | 0.2543 | 1 | — |
| adding A to none (A − none) | 1.98479 | [1.7639, 2.2831] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.57335 | [0.2486, 1.0170] | 12/14 | — | 0.9991 | 1 | — |
| adding B to A (A_B − A) | -0.00507 | [-0.0180, 0.0057] | 3/14 (8 at 0) | — | 0.3594 | 1 | — |
| adding A to B (A_B − B) | 1.40637 | [1.0930, 1.6991] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 2.13026 | [1.8618, 2.4078] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00252 | [-0.0053, -0.0002] | 0/14 (10 at 0) | fires | 0.0625 | 0.875 | — |
| adding B to S (B_S − S) | 0.68567 | [0.3017, 1.0657] | 12/14 | — | 0.9997 | 1 | — |
| adding S to B (B_S − B) | -0.03567 | [-0.0867, -0.0052] | 3/14 (5 at 0) | fires | 0.04492 | 0.6738 | — |
| adding S to A_B (A_B_S − A_B) | -0.00275 | [-0.0055, -0.0005] | 0/14 (9 at 0) | fires | 0.03125 | 0.5 | — |
| adding B to A_S (A_B_S − A_S) | -0.00530 | [-0.0180, 0.0052] | 3/14 (8 at 0) | — | 0.3438 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.43929 | [1.1381, 1.7193] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.42377 | [0.3526, 0.5332] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.41870 | [0.3485, 0.5308] | 13/14 | — | 0.9999 | 1 | — |
| A_S − repeat-own-last | 0.42125 | [0.3504, 0.5296] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.41595 | [0.3464, 0.5273] | 13/14 | — | 0.9999 | 1 | — |

### v5 (one-SE)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.00455 | [-0.0687, 0.1463] | 3/14 (4 at 0) | — | 0.5059 | 1 | — |
| adding A to none (A − none) | 1.98479 | [1.7639, 2.2831] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.57335 | [0.2486, 1.0170] | 12/14 | — | 0.9991 | 1 | — |
| adding B to A (A_B − A) | 0.00093 | [-0.0014, 0.0049] | 2/14 (11 at 0) | — | 0.75 | 1 | — |
| adding A to B (A_B − B) | 1.41236 | [1.0980, 1.6996] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.97911 | [1.7739, 2.2368] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00113 | [-0.0040, 0.0000] | 0/14 (13 at 0) | — | 0.5 | 1 | — |
| adding B to S (B_S − S) | 0.53257 | [0.2442, 0.8899] | 12/14 | — | 0.9997 | 1 | — |
| adding S to B (B_S − B) | -0.03623 | [-0.0859, -0.0064] | 1/14 (8 at 0) | fires | 0.04688 | 0.75 | — |
| adding S to A_B (A_B_S − A_B) | -0.00113 | [-0.0040, 0.0000] | 0/14 (13 at 0) | — | 0.5 | 1 | — |
| adding B to A_S (A_B_S − A_S) | 0.00093 | [-0.0014, 0.0049] | 2/14 (11 at 0) | — | 0.75 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.44747 | [1.1462, 1.7239] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.42377 | [0.3526, 0.5332] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.42469 | [0.3539, 0.5347] | 13/14 | — | 0.9999 | 1 | — |
| A_S − repeat-own-last | 0.42264 | [0.3522, 0.5302] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.42356 | [0.3534, 0.5320] | 13/14 | — | 0.9999 | 1 | — |

## Notes

- Day-bootstrap CIs are approximate with 14 days, and limited data and capacity push contrasts toward zero: a CI that spans zero may mean too little data (PLAN §9).
- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly favours keeping S; the one-SE rule counters this (DEC-23).
- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).
- This report establishes no grammar or causal mechanism.
- Per fold, condition and rule, the candidates and their validation log-likelihoods are in `data/results/rq2_d_rhythm_tempo_logistic_v5/selection.jsonl`; v5's predictions are rebuilt from it and `data/results/rq2_d_rhythm_tempo_logistic_v4/predictions.jsonl` (`logistic_v5.rebuild_predictions`).

## Inputs

SHA-256 of every input:

- `config/rq2_d_rhythm_tempo_logistic_v5.json`: `afb9d07819e111a37e52f6e80fd4ab2d6ed0a6f8758868d599f387577fd3f119`
- `config/rq2_d_rhythm_tempo_logistic_v4.json`: `5f133529df8301c2ca92dd1dfb583dc567d338f37bfb8d63c08e41703cc286c1`
- `config/rq2_d_rhythm_tempo_evaluation_v4.json`: `ff8fd09c40dc663e19c6576689fc1b2b96b1ffdaab01dabb5e5f4becdba3366a`
- `data/processed/rq2_d_rhythm_tempo_folds/manifest.json`: `5c79d957695a6b39784961a01de5343f3411fe12f794c5033fb8570ed40bae88`
- `data/processed/rq2_d_rhythm_tempo_targets.jsonl`: `4b49578dfaab640d9b278b18ebe58f03331616e52faf70ddb6a3083febdc05e5`
- `data/results/rq2_d_rhythm_tempo_logistic_v4/models.jsonl`: `265e5b43d56e6c43705ac165204a17ab79e5b38a6a5223072ee8b98e00250d8b`
- `data/results/rq2_d_rhythm_tempo_logistic_v4/l2_selection.jsonl`: `b1de99f0c2adde43177cc85ba185bce8abd23bbbb8c53ba0743ea3c87ce1ef61`
- `data/results/rq2_d_rhythm_tempo_logistic_v4/predictions.jsonl`: `3e01623bc1a9741ec79e0cdf2ec1c76f7edba99beec5c4385f2f26123edca3f6`
- `data/results/rq2_d_rhythm_tempo_logistic_v4/baseline_predictions.jsonl`: `1e39404c8446c27d1d7a397db29239a5731b21f690f4f3261f0527bc5584d162`
- `data/results/rq2_d_rhythm_tempo_logistic_v4/evaluation/summary.json`: `dc859705e8aa066e32b13dca8330a42724276ed52a2e644d9abe638ed8dbff10`
- `data/results/rq2_d_rhythm_tempo_logistic_v4/evaluation/scored_predictions.jsonl`: `aae34d4ee835bcbc6843e718963a92911b9e2c64760298f63d01d1ad2e63778b`
