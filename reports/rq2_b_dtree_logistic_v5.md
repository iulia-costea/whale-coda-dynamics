# RQ2 logistic regression v5, inventory B (DEC-23)

Logistic v5 is logistic v4 with optional ignorance by selection (DEC-23, PLAN `[M4]`). For each fold and condition, the candidates are v4's chosen fits of every sub-condition (A, B and S each on or absent; all absent is the fold's smoothed prior, condition none). The candidate with the highest validation-day log-likelihood is chosen (exact ties: fewer blocks, then the order A, B, S), and its stored v4 test-day predictions become the condition's. Each sub-condition's penalties were already chosen by v4's rule, so nothing is refitted: everything comes from `data/results/rq2_b_dtree_logistic_v4/` (models.jsonl, l2_selection.jsonl, predictions.jsonl). Sensitivity: DEC-21's one-SE rule (the fewest blocks within one standard error of the best; SE of the per-target paired validation differences). Every log-likelihood below is in bits per coda, from the shared evaluation module. Inventory B: decision_tree_coda_type (PLAN §7, DEC-22). Only the labels differ from RQ1: targets, folds, S, the UNK policy and the model are RQ1's. The log-likelihoods, accuracies and macro-F1 values in this report are not comparable with any other inventory's (PLAN [R2-3]); the comparison across inventories is reports/rq2_inventories_v5.md.

## Checks before any test-day number

- Reproduction (gate 1): the validation log-likelihoods of all 98 v4 fits, recomputed from models.jsonl, match l2_selection.jsonl to 0 bits at most (tolerance 1e-09); their test-day probabilities, and the prior's, match predictions.jsonl to 0 at most.
- Validation monotonicity (gate 2): under the max rule all 168 add-one-input contrasts (12 pairs × 14 validation days) are ≥ 0 on the validation day; the smallest is 0 bits.
- The smallest validation-day gap between a chosen candidate and the runner-up is 0.00012 bits (test-2016-04-24, condition A_S).
- Every v5 test-day log-likelihood equals v4's for the chosen sub-condition on the same coda to 0 bits at most (tolerance 1e-12; recomputing log2 on another machine can change the last bit).

## Stop rules

DEC-23's rule: each of the 16 checks (12 add-one-input contrasts, 4 A-conditions against repeat-own-last) gets an exact one-sided sign-flip test on the day sums of per-coda differences (alternative: mean below zero; all 2^14 sign assignments; statistic: token-weighted mean), with Holm's correction across the 16; a check fires when its adjusted p < 0.05. DEC-16's old rule (day-bootstrap CI entirely below zero) is shown next to it.

| Model | DEC-23 rule (sign-flip, Holm) | DEC-16 rule (CI below zero) |
|---|---|---|
| v4 | none fired | add_input:A_S-A: contrast CI entirely below zero [-0.0199, -0.0010]; add_input:A_B_S-A_B: contrast CI entirely below zero [-0.0341, -0.0006] |
| v5 (max) | none fired | add_input:A_S-A: contrast CI entirely below zero [-0.0198, -0.0010]; add_input:B_S-B: contrast CI entirely below zero [-0.2151, -0.0153] |
| v5 (one-SE) | none fired | none fired |

Governing status (DEC-23's rule on v5 (max)): none fired.

## Lattice, v4 and v5

| Condition | v4 LL | v5 LL | v5 − v4 | v5 one-SE LL | v4 accuracy | v5 accuracy | v4 macro-F1 | v5 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | -2.65044 | -2.65044 | 0.00000 | -2.65044 | 56.949% | 56.949% | 0.0316 | 0.0316 |
| S | -2.63183 | -2.63286 | -0.00102 | -2.64244 | 56.755% | 56.755% | 0.0383 | 0.0383 |
| A | -1.12205 | -1.12205 | 0.00000 | -1.12205 | 82.224% | 82.224% | 0.2774 | 0.2774 |
| B | -2.70155 | -2.61352 | 0.08803 | -2.72052 | 58.112% | 58.824% | 0.0820 | 0.0800 |
| A_B | -1.18298 | -1.18178 | 0.00119 | -1.12866 | 81.125% | 80.931% | 0.2451 | 0.2524 |
| A_S | -1.13239 | -1.13237 | 0.00002 | -1.12437 | 81.771% | 81.771% | 0.2569 | 0.2569 |
| B_S | -2.68881 | -2.70231 | -0.01350 | -2.71483 | 56.884% | 56.755% | 0.0755 | 0.0749 |
| A_B_S | -1.20064 | -1.19487 | 0.00577 | -1.13099 | 81.254% | 80.802% | 0.2477 | 0.2517 |
| majority | n/a | n/a | — | n/a | 56.949% | 56.949% | 0.0316 | 0.0316 |
| repeat_own_last | -1.43264 | -1.43264 | — | -1.43264 | 82.741% | 82.741% | 0.3320 | 0.3320 |

## Sub-conditions chosen

Folds (of 14) on which each sub-condition was chosen: max rule, with the one-SE rule in parentheses. A dash marks a sub-condition that is not a candidate for that condition.

| Condition | none | A | B | S | A_B | A_S | B_S | A_B_S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 14 (14) | — | — | — | — | — | — | — |
| S | 1 (2) | — | — | 13 (12) | — | — | — | — |
| A | 0 (0) | 14 (14) | — | — | — | — | — | — |
| B | 3 (4) | — | 11 (10) | — | — | — | — | — |
| A_B | 0 (0) | 7 (12) | 0 (0) | — | 7 (2) | — | — | — |
| A_S | 0 (0) | 6 (12) | — | 0 (0) | — | 8 (2) | — | — |
| B_S | 1 (1) | — | 4 (6) | 1 (3) | — | — | 8 (4) | — |
| A_B_S | 0 (0) | 3 (10) | 0 (0) | 0 (0) | 3 (2) | 4 (2) | 0 (0) | 4 (0) |

Per fold: the chosen sub-condition under the max rule, with the one-SE choice after an arrow where it differs.

| Test day | S | A | B | A_B | A_S | B_S | A_B_S |
|---|---|---|---|---|---|---|---|
| 2014-04-16 | S | A | B | A_B → A | A_S → A | B | A_B_S → A |
| 2014-04-24 | S | A | B | A | A_S → A | B_S | A_S → A |
| 2014-04-29 | S | A | B | A_B → A | A | B_S → B | A_B → A |
| 2015-03-02 | S | A | B | A_B → A | A_S → A | B_S | A_B_S → A |
| 2015-03-19 | S | A | B | A_B → A | A | B | A_B → A |
| 2015-03-26 | S → none | A | B | A_B → A | A_S → A | B_S → B | A_B_S → A |
| 2015-03-31 | S | A | B | A | A | B | A |
| 2015-04-01 | S | A | B | A | A_S | B_S | A_S |
| 2015-04-07 | none | A | none | A | A_S → A | none | A_S → A |
| 2015-04-10 | S | A | none | A | A | S | A |
| 2016-04-12 | S | A | B → none | A | A | B_S → S | A |
| 2016-04-24 | S | A | B | A_B | A_S → A | B | A_B_S → A_B |
| 2016-05-12 | S | A | none | A | A_S | B_S → S | A_S |
| 2016-05-13 | S | A | B | A_B | A | B_S | A_B |

## RQ1 contrasts (`[M2]`)

Paired per-coda differences; positive days, with the days whose difference is exactly zero; day-bootstrap 95% CI (10,000 replicates); exact two-sided sign-flip p on the day sums (DEC-23).

| Contrast | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |
|---|---|---:|---:|---|---:|---:|
| own_history | v4 | 1.48817 | 1.40223 | [0.9725, 2.0115] | 14/14 | 0.0001221 |
| own_history | v5 (max) | 1.50744 | 1.42781 | [0.9930, 2.0244] | 14/14 | 0.0001221 |
| own_history | v5 (one-SE) | 1.58384 | 1.48573 | [1.0134, 2.1828] | 14/14 | 0.0001221 |
| partner_content | v4 | -0.06825 | -0.02768 | [-0.1454, 0.0185] | 7/14 | 0.2396 |
| partner_content | v5 (max) | -0.06250 | -0.03600 | [-0.1444, 0.0076] | 3/14 (7 at 0) | 0.2812 |
| partner_content | v5 (one-SE) | -0.00662 | -0.02098 | [-0.0315, 0.0034] | 1/14 (12 at 0) | 1 |
| timing | v4 | -0.01767 | -0.01102 | [-0.0341, -0.0006] | 5/14 | 0.1094 |
| timing | v5 (max) | -0.01309 | -0.00567 | [-0.0261, 0.0001] | 2/14 (6 at 0) | 0.1406 |
| timing | v5 (one-SE) | -0.00233 | -0.00226 | [-0.0086, 0.0002] | 1/14 (12 at 0) | 1 |

Sign of each token-weighted mean and where its CI lies:

| Contrast | v4 | v5 (max) | v5 (one-SE) |
|---|---|---|---|
| own_history | +, CI > 0 | +, CI > 0 | +, CI > 0 |
| partner_content | −, CI spans 0 | −, CI spans 0 | −, CI spans 0 |
| timing | −, CI < 0 | −, CI spans 0 | −, CI spans 0 |

The sign or CI position differs between v4 and v5 for: timing. Both are reported.

## The sixteen stop-rule checks

Per-coda differences, token-weighted means, day-bootstrap CI and positive days (with the days whose difference is exactly zero); DEC-16: CI entirely below zero; one-sided sign-flip p (alternative: mean below zero) and its Holm adjustment across the 16; DEC-23: adjusted p below alpha.

### v4

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.01861 | [-0.0942, 0.1969] | 6/14 | — | 0.558 | 1 | — |
| adding A to none (A − none) | 1.52839 | [1.2190, 1.9207] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | -0.05111 | [-0.6176, 0.6216] | 10/14 | — | 0.4508 | 1 | — |
| adding B to A (A_B − A) | -0.06093 | [-0.1325, 0.0196] | 7/14 | — | 0.1355 | 1 | — |
| adding A to B (A_B − B) | 1.51857 | [0.9730, 2.0393] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.49944 | [1.2232, 1.8092] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.01034 | [-0.0199, -0.0010] | 4/14 | fires | 0.03851 | 0.6162 | — |
| adding B to S (B_S − S) | -0.05698 | [-0.5552, 0.4134] | 10/14 | — | 0.4805 | 1 | — |
| adding S to B (B_S − B) | 0.01274 | [-0.1415, 0.2180] | 7/14 | — | 0.5078 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.01767 | [-0.0341, -0.0006] | 5/14 | fires | 0.05469 | 0.8203 | — |
| adding B to A_S (A_B_S − A_S) | -0.06825 | [-0.1454, 0.0185] | 7/14 | — | 0.1198 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.48817 | [0.9725, 2.0115] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.31060 | [0.2441, 0.4314] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.24967 | [0.1244, 0.4433] | 12/14 | — | 0.9992 | 1 | — |
| A_S − repeat-own-last | 0.30025 | [0.2320, 0.4218] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.23200 | [0.0982, 0.4342] | 12/14 | — | 0.9979 | 1 | — |

### v5 (max)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.01758 | [-0.0952, 0.1955] | 5/14 (1 at 0) | — | 0.5544 | 1 | — |
| adding A to none (A − none) | 1.52839 | [1.2190, 1.9207] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | 0.03692 | [-0.5142, 0.6552] | 8/14 (3 at 0) | — | 0.5312 | 1 | — |
| adding B to A (A_B − A) | -0.05974 | [-0.1346, 0.0021] | 3/14 (7 at 0) | — | 0.1016 | 1 | — |
| adding A to B (A_B − B) | 1.43174 | [0.9182, 1.9598] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.50049 | [1.2239, 1.8105] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.01032 | [-0.0198, -0.0010] | 2/14 (6 at 0) | fires | 0.03906 | 0.5859 | — |
| adding B to S (B_S − S) | -0.06945 | [-0.5663, 0.3976] | 8/14 (2 at 0) | — | 0.4749 | 1 | — |
| adding S to B (B_S − B) | -0.08879 | [-0.2151, -0.0153] | 3/14 (5 at 0) | fires | 0.0293 | 0.4688 | — |
| adding S to A_B (A_B_S − A_B) | -0.01309 | [-0.0261, 0.0001] | 2/14 (6 at 0) | — | 0.07031 | 0.9844 | — |
| adding B to A_S (A_B_S − A_S) | -0.06250 | [-0.1444, 0.0076] | 3/14 (7 at 0) | — | 0.1406 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.50744 | [0.9930, 2.0244] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.31060 | [0.2441, 0.4314] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.25086 | [0.1268, 0.4215] | 12/14 | — | 0.9992 | 1 | — |
| A_S − repeat-own-last | 0.30027 | [0.2321, 0.4219] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.23777 | [0.1032, 0.4178] | 12/14 | — | 0.9979 | 1 | — |

### v5 (one-SE)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding S to none (S − none) | 0.00800 | [-0.1014, 0.1873] | 4/14 (2 at 0) | — | 0.5171 | 1 | — |
| adding A to none (A − none) | 1.52839 | [1.2190, 1.9207] | 14/14 | — | 1 | 1 | — |
| adding B to none (B − none) | -0.07008 | [-0.5804, 0.4070] | 7/14 (4 at 0) | — | 0.4629 | 1 | — |
| adding B to A (A_B − A) | -0.00662 | [-0.0315, 0.0034] | 1/14 (12 at 0) | — | 0.5 | 1 | — |
| adding A to B (A_B − B) | 1.59186 | [0.9984, 2.2124] | 14/14 | — | 1 | 1 | — |
| adding A to S (A_S − S) | 1.51806 | [1.2483, 1.8193] | 14/14 | — | 1 | 1 | — |
| adding S to A (A_S − A) | -0.00233 | [-0.0086, 0.0002] | 1/14 (12 at 0) | — | 0.5 | 1 | — |
| adding B to S (B_S − S) | -0.07239 | [-0.5758, 0.3898] | 7/14 (4 at 0) | — | 0.4688 | 1 | — |
| adding S to B (B_S − B) | 0.00569 | [-0.1018, 0.1876] | 3/14 (7 at 0) | — | 0.5312 | 1 | — |
| adding S to A_B (A_B_S − A_B) | -0.00233 | [-0.0086, 0.0002] | 1/14 (12 at 0) | — | 0.5 | 1 | — |
| adding B to A_S (A_B_S − A_S) | -0.00662 | [-0.0315, 0.0034] | 1/14 (12 at 0) | — | 0.5 | 1 | — |
| adding A to B_S (A_B_S − B_S) | 1.58384 | [1.0134, 2.1828] | 14/14 | — | 1 | 1 | — |
| A − repeat-own-last | 0.31060 | [0.2441, 0.4314] | 13/14 | — | 0.9999 | 1 | — |
| A_B − repeat-own-last | 0.30398 | [0.2269, 0.4259] | 13/14 | — | 0.9997 | 1 | — |
| A_S − repeat-own-last | 0.30827 | [0.2420, 0.4279] | 13/14 | — | 0.9999 | 1 | — |
| A_B_S − repeat-own-last | 0.30165 | [0.2247, 0.4238] | 13/14 | — | 0.9997 | 1 | — |

## Notes

- Day-bootstrap CIs are approximate with 14 days, and limited data and capacity push contrasts toward zero: a CI that spans zero may mean too little data (PLAN §9).
- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly favours keeping S; the one-SE rule counters this (DEC-23).
- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).
- This report establishes no grammar or causal mechanism.
- Per fold, condition and rule, the candidates and their validation log-likelihoods are in `data/results/rq2_b_dtree_logistic_v5/selection.jsonl`; v5's predictions are rebuilt from it and `data/results/rq2_b_dtree_logistic_v4/predictions.jsonl` (`logistic_v5.rebuild_predictions`).

## Inputs

SHA-256 of every input:

- `config/rq2_b_dtree_logistic_v5.json`: `dc60bbbb4a5ed2a599942a223a28c8031a7573133c1755bed3c1c2a1b650a956`
- `config/rq2_b_dtree_logistic_v4.json`: `9a4c1bd3d36c1db2a5000827ef519cbd1b9063d4f0c031783e675de2ded3d1a3`
- `config/rq2_b_dtree_evaluation_v4.json`: `20bf7ae50e8baf08835bb2a13ccca09841fbf94a77bb7c9362ec79ad53b3b0d5`
- `data/processed/rq2_b_dtree_folds/manifest.json`: `951fd33f86efa67d70e2b84b5fda8b392ce7d1dab2c861e7ff67e2f792eb347a`
- `data/processed/rq2_b_dtree_targets.jsonl`: `be16e65abda3ee4a13bf7fc3b81e4265d7f8ad5c6639cd41920fcaf872cbe0ec`
- `data/results/rq2_b_dtree_logistic_v4/models.jsonl`: `1870ee5be287fcaebaa7cd384809c717a9da330d9ba74da33177f830757a863b`
- `data/results/rq2_b_dtree_logistic_v4/l2_selection.jsonl`: `ccf698a5ac283ef9d6b5b24bc7e1ba72c33694262d75c604e65241d725163a11`
- `data/results/rq2_b_dtree_logistic_v4/predictions.jsonl`: `641cbb7e017fdc8ec49874411b8f7d830bd1065b8856131af03498f03ef2b917`
- `data/results/rq2_b_dtree_logistic_v4/baseline_predictions.jsonl`: `03dcb0f134b923d67578ea611ba798c365d1a9979c19904b83f821e408676c6a`
- `data/results/rq2_b_dtree_logistic_v4/evaluation/summary.json`: `46df20dc4c95f845e494a5e06d0051c754216ce1c7c42a321d100f85bc552ab3`
- `data/results/rq2_b_dtree_logistic_v4/evaluation/scored_predictions.jsonl`: `084ea07ef9f7ca72238a3ae227aee763c6646aaf0f2c4a9c2642abfdf77087cf`
