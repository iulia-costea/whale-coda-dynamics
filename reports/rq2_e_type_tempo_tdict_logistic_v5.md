# RQ2 logistic regression v5, inventory E′ (DEC-23)

Logistic v5 is logistic v4 with optional ignorance by selection (DEC-23, PLAN `[M4]`). For each fold and condition, the candidates are v4's chosen fits of every sub-condition (A, B and S each on or absent; all absent is the fold's smoothed prior, condition none). The candidate with the highest validation-day log-likelihood is chosen (exact ties: fewer blocks, then the order A, B, S), and its stored v4 test-day predictions become the condition's. Each sub-condition's penalties were already chosen by v4's rule, so nothing is refitted: everything comes from `data/results/rq2_e_type_tempo_tdict_logistic_v4/` (models.jsonl, l2_selection.jsonl, predictions.jsonl). Sensitivity: DEC-21's one-SE rule (the fewest blocks within one standard error of the best; SE of the per-target paired validation differences). Every log-likelihood below is in bits per coda, from the shared evaluation module. Inventory E′: guy_coda_type × tempo (tempos-dict.p cuts) (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy and the model are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v5.md.

## Checks before any test-day number

- Reproduction (gate 1): the validation log-likelihoods of all 98 v4 fits, recomputed from models.jsonl, match l2_selection.jsonl to 0 bits at most (tolerance 1e-09); their test-day probabilities, and the prior's, match predictions.jsonl to 0 at most.
- Validation monotonicity (gate 2): under the max rule all 168 add-one-input contrasts (12 pairs × 14 validation days) are ≥ 0 on the validation day; the smallest is 0 bits.
- The smallest validation-day gap between a chosen candidate and the runner-up is 6.02e-05 bits (test-2015-04-10, condition A_S).
- Every v5 test-day log-likelihood equals v4's for the chosen sub-condition on the same coda to 0 bits at most (tolerance 1e-12; recomputing log2 on another machine can change the last bit).

## Stop rules

DEC-23's rule: each of the 16 checks (12 add-one-input contrasts, 4 A-conditions against repeat-own-last) gets an exact one-sided sign-flip test on the day sums of per-coda differences (alternative: mean below zero; all 2^14 sign assignments; statistic: token-weighted mean), with Holm's correction across the 16; a check fires when its adjusted p < 0.05. DEC-16's old rule (day-bootstrap CI entirely below zero) is shown next to it.

| Model | DEC-23 rule (sign-flip, Holm) | DEC-16 rule (CI below zero) |
|---|---|---|
| v4 | none fired | none fired |
| v5 (max) | none fired | add_input:A_S-A: contrast CI entirely below zero [-0.0089, -0.0002] |
| v5 (one-SE) | none fired | none fired |

Governing status (DEC-23's rule on v5 (max)): none fired.

## Lattice, v4 and v5

| Condition | v4 LL | v5 LL | v5 − v4 | v5 one-SE LL | v4 accuracy | v5 accuracy | v4 macro-F1 | v5 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | -3.59777 | -3.59777 | 0.00000 | -3.59777 | 39.431% | 39.431% | 0.0162 | 0.0162 |
| S | -3.73306 | -3.73196 | 0.00110 | -3.56790 | 36.910% | 36.910% | 0.0201 | 0.0201 |
| A | -1.53485 | -1.53485 | 0.00000 | -1.53485 | 79.444% | 79.444% | 0.2967 | 0.2967 |
| B | -3.03745 | -3.03745 | 0.00000 | -3.03745 | 52.489% | 52.489% | 0.0664 | 0.0664 |
| A_B | -1.57736 | -1.58086 | -0.00349 | -1.53383 | 76.406% | 76.535% | 0.2579 | 0.2617 |
| A_S | -1.53877 | -1.53886 | -0.00009 | -1.53521 | 78.862% | 78.862% | 0.2952 | 0.2952 |
| B_S | -3.05593 | -3.05642 | -0.00049 | -3.05635 | 50.549% | 50.485% | 0.0669 | 0.0668 |
| A_B_S | -1.58067 | -1.58467 | -0.00401 | -1.53401 | 76.018% | 76.212% | 0.2484 | 0.2531 |
| majority | n/a | n/a | — | n/a | 39.431% | 39.431% | 0.0162 | 0.0162 |
| repeat_own_last | -1.87884 | -1.87884 | — | -1.87884 | 78.539% | 78.539% | 0.3349 | 0.3349 |

## Sub-conditions chosen

Folds (of 14) on which each sub-condition was chosen: max rule, with the one-SE rule in parentheses. A dash marks a sub-condition that is not a candidate for that condition.

| Condition | none | A | B | S | A_B | A_S | B_S | A_B_S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 14 (14) | — | — | — | — | — | — | — |
| S | 1 (4) | — | — | 13 (10) | — | — | — | — |
| A | 0 (0) | 14 (14) | — | — | — | — | — | — |
| B | 0 (0) | — | 14 (14) | — | — | — | — | — |
| A_B | 0 (0) | 4 (12) | 0 (0) | — | 10 (2) | — | — | — |
| A_S | 0 (0) | 8 (12) | — | 0 (0) | — | 6 (2) | — | — |
| B_S | 0 (0) | — | 4 (10) | 0 (0) | — | — | 10 (4) | — |
| A_B_S | 0 (0) | 2 (11) | 0 (0) | 0 (0) | 5 (2) | 2 (1) | 0 (0) | 5 (0) |

Per fold: the chosen sub-condition under the max rule, with the one-SE choice after an arrow where it differs.

| Test day | S | A | B | A_B | A_S | B_S | A_B_S |
|---|---|---|---|---|---|---|---|
| 2014-04-16 | S | A | B | A_B → A | A_S → A | B | A_B_S → A |
| 2014-04-24 | S | A | B | A | A_S | B_S | A_S |
| 2014-04-29 | S → none | A | B | A_B → A | A | B_S → B | A_B_S → A |
| 2015-03-02 | S | A | B | A_B → A | A_S → A | B_S | A_B_S → A |
| 2015-03-19 | S | A | B | A_B → A | A | B | A_B → A |
| 2015-03-26 | S → none | A | B | A_B → A | A | B_S → B | A_B → A |
| 2015-03-31 | S | A | B | A | A | B_S → B | A |
| 2015-04-01 | S | A | B | A_B → A | A_S | B_S | A_B_S → A |
| 2015-04-07 | S → none | A | B | A_B | A | B | A_B |
| 2015-04-10 | S | A | B | A | A_S → A | B_S → B | A_S → A |
| 2016-04-12 | S | A | B | A | A | B_S | A |
| 2016-04-24 | S | A | B | A_B | A | B | A_B |
| 2016-05-12 | none | A | B | A_B → A | A_S → A | B_S → B | A_B_S → A |
| 2016-05-13 | S | A | B | A_B → A | A | B_S → B | A_B → A |

## RQ1 contrasts (`[M2]`)

Paired per-coda differences; positive days, with the days whose difference is exactly zero; day-bootstrap 95% CI (10,000 replicates); exact two-sided sign-flip p on the day sums (DEC-23).

| Contrast | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |
|---|---|---:|---:|---|---:|---:|
| own_history | v4 | 1.47526 | 1.35047 | [1.1322, 1.7934] | 14/14 | 0.0001221 |
| own_history | v5 (max) | 1.47174 | 1.32871 | [1.1304, 1.7824] | 14/14 | 0.0001221 |
| own_history | v5 (one-SE) | 1.52234 | 1.31987 | [1.1300, 1.9012] | 14/14 | 0.0001221 |
| partner_content | v4 | -0.04190 | 0.02039 | [-0.1165, 0.0545] | 6/14 | 0.6426 |
| partner_content | v5 (max) | -0.04581 | -0.00330 | [-0.1207, 0.0274] | 4/14 (4 at 0) | 0.6191 |
| partner_content | v5 (one-SE) | 0.00120 | 0.00738 | [-0.0001, 0.0050] | 2/14 (11 at 0) | 0.5 |
| timing | v4 | -0.00330 | -0.00379 | [-0.0082, 0.0012] | 8/14 | 0.4031 |
| timing | v5 (max) | -0.00381 | -0.00380 | [-0.0085, 0.0005] | 2/14 (7 at 0) | 0.2656 |
| timing | v5 (one-SE) | -0.00018 | -0.00070 | [-0.0008, 0.0000] | 0/14 (13 at 0) | 1 |

Sign of each token-weighted mean and where its CI lies:

| Contrast | v4 | v5 (max) | v5 (one-SE) |
|---|---|---|---|
| own_history | +, CI > 0 | +, CI > 0 | +, CI > 0 |
| partner_content | −, CI spans 0 | −, CI spans 0 | +, CI spans 0 |
| timing | −, CI spans 0 | −, CI spans 0 | −, CI spans 0 |

The sign and CI position of every `[M2]` contrast are the same under v4 and v5.

## The sixteen stop-rule checks

Per-coda differences, token-weighted means, day-bootstrap CI and positive days (with the days whose difference is exactly zero); DEC-16: CI entirely below zero; one-sided sign-flip p (alternative: mean below zero) and its Holm adjustment across the 16; DEC-23: adjusted p below alpha.

### v4

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.13529 | [-0.4565, 0.1612] | 6/14 | — | 0.2742 | 1 | — |
| adding A to none (A − none) | 2.06292 | [1.8426, 2.3636] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.56032 | [0.1589, 1.0801] | 12/14 | — | 0.9916 | 1 | — |
| adding B to A (A_B − A) | -0.04251 | [-0.1181, 0.0556] | 6/14 | — | 0.3217 | 1 | — |
| adding A to B (A_B − B) | 1.46009 | [1.1116, 1.7794] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 2.19429 | [1.9170, 2.4899] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00392 | [-0.0088, 0.0000] | 5/14 | — | 0.06268 | 1 | — |
| adding B to S (B_S − S) | 0.67714 | [0.2110, 1.1519] | 12/14 | — | 0.9939 | 1 | — |
| adding S to B (B_S − B) | -0.01847 | [-0.0765, 0.0111] | 8/14 | — | 0.1901 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.00330 | [-0.0082, 0.0012] | 8/14 | — | 0.2015 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.04190 | [-0.1165, 0.0545] | 6/14 | — | 0.3213 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.47526 | [1.1322, 1.7934] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.34399 | [0.2847, 0.4390] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.30147 | [0.1861, 0.4844] | 13/14 | — | 0.9999 | 1 | — |
| A_S − repeat-own-last | 0.34007 | [0.2799, 0.4375] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.29817 | [0.1805, 0.4819] | 13/14 | — | 0.9999 | 1 | — |

### v5 (max)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.13419 | [-0.4550, 0.1621] | 6/14 (1 at 0) | — | 0.2765 | 1 | — |
| adding A to none (A − none) | 2.06292 | [1.8426, 2.3636] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.56032 | [0.1589, 1.0801] | 12/14 | — | 0.9916 | 1 | — |
| adding B to A (A_B − A) | -0.04600 | [-0.1224, 0.0287] | 4/14 (4 at 0) | — | 0.3213 | 1 | — |
| adding A to B (A_B − B) | 1.45659 | [1.1036, 1.7710] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 2.19310 | [1.9151, 2.4894] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00401 | [-0.0089, -0.0002] | 1/14 (8 at 0) | fires | 0.04688 | 0.75 | — |
| adding B to S (B_S − S) | 0.67554 | [0.2093, 1.1504] | 12/14 | — | 0.9938 | 1 | — |
| adding S to B (B_S − B) | -0.01897 | [-0.0773, 0.0106] | 5/14 (4 at 0) | — | 0.1816 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.00381 | [-0.0085, 0.0005] | 2/14 (7 at 0) | — | 0.1328 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.04581 | [-0.1207, 0.0274] | 4/14 (4 at 0) | — | 0.3096 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.47174 | [1.1304, 1.7824] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.34399 | [0.2847, 0.4390] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.29798 | [0.1882, 0.4523] | 13/14 | — | 0.9999 | 1 | — |
| A_S − repeat-own-last | 0.33998 | [0.2799, 0.4371] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.29417 | [0.1816, 0.4499] | 13/14 | — | 0.9999 | 1 | — |

### v5 (one-SE)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.02988 | [-0.0567, 0.1932] | 5/14 (4 at 0) | — | 0.5615 | 1 | — |
| adding A to none (A − none) | 2.06292 | [1.8426, 2.3636] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.56032 | [0.1589, 1.0801] | 12/14 | — | 0.9916 | 1 | — |
| adding B to A (A_B − A) | 0.00102 | [-0.0001, 0.0047] | 1/14 (12 at 0) | — | 0.75 | 1 | — |
| adding A to B (A_B − B) | 1.50362 | [1.1019, 1.8870] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 2.03269 | [1.8375, 2.2708] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00036 | [-0.0011, 0.0000] | 0/14 (12 at 0) | — | 0.25 | 1 | — |
| adding B to S (B_S − S) | 0.51155 | [0.1599, 0.9139] | 13/14 | — | 0.9945 | 1 | — |
| adding S to B (B_S − B) | -0.01890 | [-0.0676, 0.0043] | 1/14 (10 at 0) | — | 0.1875 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.00018 | [-0.0008, 0.0000] | 0/14 (13 at 0) | — | 0.5 | 1 | — |
| adding B to A_S (A_B_S − A_S) | 0.00120 | [-0.0001, 0.0050] | 2/14 (11 at 0) | — | 0.875 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.52234 | [1.1300, 1.9012] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.34399 | [0.2847, 0.4390] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.34501 | [0.2856, 0.4411] | 13/14 | — | 0.9999 | 1 | — |
| A_S − repeat-own-last | 0.34363 | [0.2845, 0.4381] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.34483 | [0.2854, 0.4408] | 13/14 | — | 0.9999 | 1 | — |

## Notes

- Day-bootstrap CIs are approximate with 14 days, and limited data and capacity push contrasts toward zero: a CI that spans zero may mean too little data (PLAN §9).
- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly favours keeping S; the one-SE rule counters this (DEC-23).
- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).
- This report establishes no grammar or causal mechanism.
- Per fold, condition and rule, the candidates and their validation log-likelihoods are in `data/results/rq2_e_type_tempo_tdict_logistic_v5/selection.jsonl`; v5's predictions are rebuilt from it and `data/results/rq2_e_type_tempo_tdict_logistic_v4/predictions.jsonl` (`logistic_v5.rebuild_predictions`).

## Inputs

SHA-256 of every input:

- `config/rq2_e_type_tempo_tdict_logistic_v5.json`: `fc242c7cb5f78666e481101a4f3b9f0226aa2edfc7b3ad57b884998bd09f7a86`
- `config/rq2_e_type_tempo_tdict_logistic_v4.json`: `a23f06bdd8f1b5b33315bef34258be36a0b7213c0e78ec79f9aa0182e98b0fe7`
- `config/rq2_e_type_tempo_tdict_evaluation_v4.json`: `693677284eb01c3ca4e4a2f9a89474c9297588874c8b179f107036450d374200`
- `data/processed/rq2_e_type_tempo_tdict_folds/manifest.json`: `0714659a95576573f2cd3457afb345df1956fe6277eacc742166821e120a651c`
- `data/processed/rq2_e_type_tempo_tdict_targets.jsonl`: `bd54f96b8e6a0e78f8da68e7f6f10bf7cfb13da21b1b6e756616b81feb7bbeb4`
- `data/results/rq2_e_type_tempo_tdict_logistic_v4/models.jsonl`: `1181d113c4ab3c18ef5b5bf32759af6a3baa22679f42202bbff3dd5bdf1d5281`
- `data/results/rq2_e_type_tempo_tdict_logistic_v4/l2_selection.jsonl`: `00b53bf4b22468a44c6883201a4df8bc02e00b4d8890635279e571f1bfa7cc57`
- `data/results/rq2_e_type_tempo_tdict_logistic_v4/predictions.jsonl`: `6c501c01025bcffd7df23b3ec93fc8b381a3050230d74b19a56e81a32a7e039b`
- `data/results/rq2_e_type_tempo_tdict_logistic_v4/baseline_predictions.jsonl`: `62135d7fe7e00c61bf4aa9bb3b84a173467b98c149ba4def68ca728f22285acd`
- `data/results/rq2_e_type_tempo_tdict_logistic_v4/evaluation/summary.json`: `45b9cc25f902cc396355c077ec10e6be2450943c17d390669b49c91e5d17d151`
- `data/results/rq2_e_type_tempo_tdict_logistic_v4/evaluation/scored_predictions.jsonl`: `0ac1c21c25f984427d2f3a528d5d631f48d097a262770bfe70ad91e130ebdf2f`
