# RQ2 logistic regression v5, inventory E (DEC-23)

Logistic v5 is logistic v4 with optional ignorance by selection (DEC-23, PLAN `[M4]`). For each fold and condition, the candidates are v4's chosen fits of every sub-condition (A, B and S each on or absent; all absent is the fold's smoothed prior, condition none). The candidate with the highest validation-day log-likelihood is chosen (exact ties: fewer blocks, then the order A, B, S), and its stored v4 test-day predictions become the condition's. Each sub-condition's penalties were already chosen by v4's rule, so nothing is refitted: everything comes from `data/results/rq2_e_type_tempo_logistic_v4/` (models.jsonl, l2_selection.jsonl, predictions.jsonl). Sensitivity: DEC-21's one-SE rule (the fewest blocks within one standard error of the best; SE of the per-target paired validation differences). Every log-likelihood below is in bits per coda, from the shared evaluation module. Inventory E: guy_coda_type × tempo (KDE cuts) (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy and the model are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v5.md.

## Checks before any test-day number

- Reproduction (gate 1): the validation log-likelihoods of all 98 v4 fits, recomputed from models.jsonl, match l2_selection.jsonl to 0 bits at most (tolerance 1e-09); their test-day probabilities, and the prior's, match predictions.jsonl to 0 at most.
- Validation monotonicity (gate 2): under the max rule all 168 add-one-input contrasts (12 pairs × 14 validation days) are ≥ 0 on the validation day; the smallest is 0 bits.
- The smallest validation-day gap between a chosen candidate and the runner-up is 3.49e-05 bits (test-2015-03-02, condition A_S).
- Every v5 test-day log-likelihood equals v4's for the chosen sub-condition on the same coda to 0 bits at most (tolerance 1e-12; recomputing log2 on another machine can change the last bit).

## Stop rules

DEC-23's rule: each of the 16 checks (12 add-one-input contrasts, 4 A-conditions against repeat-own-last) gets an exact one-sided sign-flip test on the day sums of per-coda differences (alternative: mean below zero; all 2^14 sign assignments; statistic: token-weighted mean), with Holm's correction across the 16; a check fires when its adjusted p < 0.05. DEC-16's old rule (day-bootstrap CI entirely below zero) is shown next to it.

| Model | DEC-23 rule (sign-flip, Holm) | DEC-16 rule (CI below zero) |
|---|---|---|
| v4 | none fired | add_input:A_S-A: contrast CI entirely below zero [-0.0064, -0.0001] |
| v5 (max) | none fired | add_input:A_S-A: contrast CI entirely below zero [-0.0067, -0.0004] |
| v5 (one-SE) | none fired | none fired |

Governing status (DEC-23's rule on v5 (max)): none fired.

## Lattice, v4 and v5

| Condition | v4 LL | v5 LL | v5 − v4 | v5 one-SE LL | v4 accuracy | v5 accuracy | v4 macro-F1 | v5 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | -3.56101 | -3.56101 | 0.00000 | -3.56101 | 39.884% | 39.884% | 0.0173 | 0.0173 |
| S | -3.70003 | -3.69890 | 0.00113 | -3.53861 | 37.557% | 37.557% | 0.0216 | 0.0216 |
| A | -1.53773 | -1.53773 | 0.00000 | -1.53773 | 77.505% | 77.505% | 0.3056 | 0.3056 |
| B | -3.04276 | -3.04276 | 0.00000 | -3.04276 | 53.264% | 53.264% | 0.0851 | 0.0851 |
| A_B | -1.57978 | -1.60064 | -0.02086 | -1.53664 | 75.178% | 74.273% | 0.2510 | 0.2438 |
| A_S | -1.54019 | -1.54036 | -0.00018 | -1.53889 | 77.246% | 77.246% | 0.3007 | 0.3007 |
| B_S | -3.08437 | -3.08487 | -0.00050 | -3.09024 | 49.580% | 49.580% | 0.0678 | 0.0679 |
| A_B_S | -1.57882 | -1.60035 | -0.02153 | -1.53664 | 75.372% | 74.402% | 0.2471 | 0.2400 |
| majority | n/a | n/a | — | n/a | 39.884% | 39.884% | 0.0173 | 0.0173 |
| repeat_own_last | -1.90389 | -1.90389 | — | -1.90389 | 78.087% | 78.087% | 0.3339 | 0.3339 |

## Sub-conditions chosen

Folds (of 14) on which each sub-condition was chosen: max rule, with the one-SE rule in parentheses. A dash marks a sub-condition that is not a candidate for that condition.

| Condition | none | A | B | S | A_B | A_S | B_S | A_B_S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 14 (14) | — | — | — | — | — | — | — |
| S | 1 (4) | — | — | 13 (10) | — | — | — | — |
| A | 0 (0) | 14 (14) | — | — | — | — | — | — |
| B | 0 (0) | — | 14 (14) | — | — | — | — | — |
| A_B | 0 (0) | 4 (11) | 0 (0) | — | 10 (3) | — | — | — |
| A_S | 0 (0) | 9 (13) | — | 0 (0) | — | 5 (1) | — | — |
| B_S | 0 (0) | — | 5 (10) | 0 (0) | — | — | 9 (4) | — |
| A_B_S | 0 (0) | 2 (11) | 0 (0) | 0 (0) | 6 (3) | 2 (0) | 0 (0) | 4 (0) |

Per fold: the chosen sub-condition under the max rule, with the one-SE choice after an arrow where it differs.

| Test day | S | A | B | A_B | A_S | B_S | A_B_S |
|---|---|---|---|---|---|---|---|
| 2014-04-16 | S | A | B | A_B → A | A_S → A | B | A_B_S → A |
| 2014-04-24 | S | A | B | A | A_S → A | B_S → B | A_S → A |
| 2014-04-29 | S → none | A | B | A_B → A | A | B | A_B_S → A |
| 2015-03-02 | S | A | B | A_B → A | A_S → A | B_S | A_B_S → A |
| 2015-03-19 | S | A | B | A_B → A | A | B | A_B → A |
| 2015-03-26 | S → none | A | B | A_B → A | A | B_S → B | A_B → A |
| 2015-03-31 | S | A | B | A | A | B_S → B | A |
| 2015-04-01 | S | A | B | A_B → A | A_S | B_S | A_B_S → A |
| 2015-04-07 | S → none | A | B | A_B | A | B | A_B |
| 2015-04-10 | S | A | B | A_B → A | A | B_S → B | A_B → A |
| 2016-04-12 | S | A | B | A | A | B_S | A |
| 2016-04-24 | S | A | B | A_B | A | B | A_B |
| 2016-05-12 | none | A | B | A | A_S → A | B_S → B | A_S → A |
| 2016-05-13 | S | A | B | A_B | A | B_S | A_B |

## RQ1 contrasts (`[M2]`)

Paired per-coda differences; positive days, with the days whose difference is exactly zero; day-bootstrap 95% CI (10,000 replicates); exact two-sided sign-flip p on the day sums (DEC-23).

| Contrast | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |
|---|---|---:|---:|---|---:|---:|
| own_history | v4 | 1.50555 | 1.37290 | [1.1616, 1.8381] | 14/14 | 0.0001221 |
| own_history | v5 (max) | 1.48452 | 1.32842 | [1.1446, 1.8130] | 14/14 | 0.0001221 |
| own_history | v5 (one-SE) | 1.55360 | 1.35037 | [1.1472, 1.9630] | 14/14 | 0.0001221 |
| partner_content | v4 | -0.03863 | 0.03494 | [-0.1356, 0.0862] | 8/14 | 0.7701 |
| partner_content | v5 (max) | -0.05999 | -0.01150 | [-0.1476, 0.0221] | 5/14 (4 at 0) | 0.4785 |
| partner_content | v5 (one-SE) | 0.00226 | 0.00842 | [-0.0001, 0.0071] | 3/14 (10 at 0) | 0.25 |
| timing | v4 | 0.00097 | 0.00141 | [-0.0023, 0.0057] | 8/14 | 0.6873 |
| timing | v5 (max) | 0.00029 | 0.00141 | [-0.0026, 0.0050] | 2/14 (8 at 0) | 0.9688 |
| timing | v5 (one-SE) | 0.00000 | 0.00000 | [0.0000, 0.0000] | 0/14 (14 at 0) | 1 |

Sign of each token-weighted mean and where its CI lies:

| Contrast | v4 | v5 (max) | v5 (one-SE) |
|---|---|---|---|
| own_history | +, CI > 0 | +, CI > 0 | +, CI > 0 |
| partner_content | −, CI spans 0 | −, CI spans 0 | +, CI spans 0 |
| timing | +, CI spans 0 | +, CI spans 0 | 0, CI spans 0 |

The sign and CI position of every `[M2]` contrast are the same under v4 and v5.

## The sixteen stop-rule checks

Per-coda differences, token-weighted means, day-bootstrap CI and positive days (with the days whose difference is exactly zero); DEC-16: CI entirely below zero; one-sided sign-flip p (alternative: mean below zero) and its Holm adjustment across the 16; DEC-23: adjusted p below alpha.

### v4

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.13903 | [-0.4522, 0.1498] | 4/14 | — | 0.2616 | 1 | — |
| adding A to none (A − none) | 2.02327 | [1.8107, 2.2974] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.51825 | [0.0859, 1.0774] | 12/14 | — | 0.9807 | 1 | — |
| adding B to A (A_B − A) | -0.04205 | [-0.1389, 0.0835] | 8/14 | — | 0.3594 | 1 | — |
| adding A to B (A_B − B) | 1.46298 | [1.1189, 1.7951] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 2.15985 | [1.8816, 2.4391] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00245 | [-0.0064, -0.0001] | 5/14 | fires | 0.05328 | 0.8525 | — |
| adding B to S (B_S − S) | 0.61566 | [0.1172, 1.0906] | 12/14 | — | 0.9781 | 1 | — |
| adding S to B (B_S − B) | -0.04161 | [-0.1562, 0.0109] | 8/14 | — | 0.1459 | 1 | — |
| adding S to A_B (A_B_S − A_B) | 0.00097 | [-0.0023, 0.0057] | 8/14 | — | 0.6564 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.03863 | [-0.1356, 0.0862] | 8/14 | — | 0.3851 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.50555 | [1.1616, 1.8381] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.36616 | [0.3122, 0.4329] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.32411 | [0.2295, 0.4810] | 13/14 | — | 0.9999 | 1 | — |
| A_S − repeat-own-last | 0.36371 | [0.3110, 0.4273] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.32508 | [0.2304, 0.4817] | 13/14 | — | 0.9999 | 1 | — |

### v5 (max)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | -0.13790 | [-0.4517, 0.1510] | 4/14 (1 at 0) | — | 0.2632 | 1 | — |
| adding A to none (A − none) | 2.02327 | [1.8107, 2.2974] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.51825 | [0.0859, 1.0774] | 12/14 | — | 0.9807 | 1 | — |
| adding B to A (A_B − A) | -0.06291 | [-0.1511, 0.0199] | 5/14 (4 at 0) | — | 0.1982 | 1 | — |
| adding A to B (A_B − B) | 1.44212 | [1.0848, 1.7767] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 2.15854 | [1.8792, 2.4384] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00263 | [-0.0067, -0.0004] | 0/14 (9 at 0) | fires | 0.03125 | 0.5 | — |
| adding B to S (B_S − S) | 0.61404 | [0.1165, 1.0895] | 12/14 | — | 0.9779 | 1 | — |
| adding S to B (B_S − B) | -0.04211 | [-0.1571, 0.0103] | 4/14 (5 at 0) | — | 0.1426 | 1 | — |
| adding S to A_B (A_B_S − A_B) | 0.00029 | [-0.0026, 0.0050] | 2/14 (8 at 0) | — | 0.5312 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.05999 | [-0.1476, 0.0221] | 5/14 (4 at 0) | — | 0.2393 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.48452 | [1.1446, 1.8130] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.36616 | [0.3122, 0.4329] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.30325 | [0.2211, 0.4223] | 13/14 | — | 0.9999 | 1 | — |
| A_S − repeat-own-last | 0.36353 | [0.3108, 0.4270] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.30354 | [0.2208, 0.4230] | 13/14 | — | 0.9999 | 1 | — |

### v5 (one-SE)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.02240 | [-0.0585, 0.1815] | 3/14 (4 at 0) | — | 0.5283 | 1 | — |
| adding A to none (A − none) | 2.02327 | [1.8107, 2.2974] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.51825 | [0.0859, 1.0774] | 12/14 | — | 0.9807 | 1 | — |
| adding B to A (A_B − A) | 0.00110 | [-0.0003, 0.0048] | 2/14 (11 at 0) | — | 0.875 | 1 | — |
| adding A to B (A_B − B) | 1.50612 | [1.0781, 1.9259] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.99972 | [1.8072, 2.2198] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00116 | [-0.0041, 0.0000] | 0/14 (13 at 0) | — | 0.5 | 1 | — |
| adding B to S (B_S − S) | 0.44837 | [0.0643, 0.8707] | 12/14 | — | 0.9774 | 1 | — |
| adding S to B (B_S − B) | -0.04748 | [-0.1582, 0.0024] | 1/14 (10 at 0) | — | 0.125 | 1 | — |
| adding S to A_B (A_B_S − A_B) | 0.00000 | [0.0000, 0.0000] | 0/14 (14 at 0) | — | 1 | 1 | — |
| adding B to A_S (A_B_S − A_S) | 0.00226 | [-0.0001, 0.0071] | 3/14 (10 at 0) | — | 0.9375 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.55360 | [1.1472, 1.9630] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.36616 | [0.3122, 0.4329] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.36726 | [0.3136, 0.4351] | 13/14 | — | 0.9999 | 1 | — |
| A_S − repeat-own-last | 0.36500 | [0.3114, 0.4301] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.36726 | [0.3136, 0.4351] | 13/14 | — | 0.9999 | 1 | — |

## Notes

- Day-bootstrap CIs are approximate with 14 days, and limited data and capacity push contrasts toward zero: a CI that spans zero may mean too little data (PLAN §9).
- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly favours keeping S; the one-SE rule counters this (DEC-23).
- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).
- This report establishes no grammar or causal mechanism.
- Per fold, condition and rule, the candidates and their validation log-likelihoods are in `data/results/rq2_e_type_tempo_logistic_v5/selection.jsonl`; v5's predictions are rebuilt from it and `data/results/rq2_e_type_tempo_logistic_v4/predictions.jsonl` (`logistic_v5.rebuild_predictions`).

## Inputs

SHA-256 of every input:

- `config/rq2_e_type_tempo_logistic_v5.json`: `6e6e3cd2a18de739b004bbb987c80c9e810fc9c1d91a37a50469c85f73eb4d7e`
- `config/rq2_e_type_tempo_logistic_v4.json`: `0dd4d7565fcf3b52be53f60b558e21a0c80a253b52052f8d370d5696a65c4a5e`
- `config/rq2_e_type_tempo_evaluation_v4.json`: `815410d52c11a5189146f5b114e5a4b0981103c03457938f62afb32f0d2b5b2c`
- `data/processed/rq2_e_type_tempo_folds/manifest.json`: `521fc45410c050460fdad848df445124fbbedd78db61c0150d061d8b400bea9c`
- `data/processed/rq2_e_type_tempo_targets.jsonl`: `dbfb2fda49169aaa60e01a8e059f0d44ddce921118ee6cd0d1af268c2643c7f2`
- `data/results/rq2_e_type_tempo_logistic_v4/models.jsonl`: `1879fe4c1e9db012fa1820731a806f50097ed5d3af2bc5e9a0c2cc003e11458d`
- `data/results/rq2_e_type_tempo_logistic_v4/l2_selection.jsonl`: `8beaf427259c7fdc43d21a7c40cb87e0d9af07f8a669f4ac8de1f2695f9c01e8`
- `data/results/rq2_e_type_tempo_logistic_v4/predictions.jsonl`: `967ce986105d97d0daeed6c91640aae2b5b500c85ba8ac2b65da4e4f14a6caac`
- `data/results/rq2_e_type_tempo_logistic_v4/baseline_predictions.jsonl`: `52bed6e9c4e775c1400bbd184f08464c9ef9d90da4ac3c1b7af49866639099f2`
- `data/results/rq2_e_type_tempo_logistic_v4/evaluation/summary.json`: `6a8b757739f5ec4f94e6f69c13b6473cb080bd4f42974b346489bbe2ca28a115`
- `data/results/rq2_e_type_tempo_logistic_v4/evaluation/scored_predictions.jsonl`: `cb492f9d13870a06700edd9c1115c4f1bb348eb77216c74685a6d4767cea9d28`
