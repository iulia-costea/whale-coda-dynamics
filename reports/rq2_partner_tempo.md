# RQ2 partner-tempo ablation (DEC-24)

Does a partner's tempo or a partner's pattern predict a coda's tempo? DEC-24 (PLAN `[R2-8]`) models the chain rule's second factor directly: the target is the coda's tempo class (t0, t1, t2, t3, t4: the KDE classes of DEC-22), given the coda's own pattern, which is always present. The primary run uses Sharma rhythms as patterns (inventory D's labels); the secondary run uses `guy_coda_type` (inventory E's). Rhythm is normalised by duration, so the partner's rhythm and tempo are separable by construction; named types partly encode tempo. Every condition predicts the same five-class target on the same 1,547 comparison targets, with RQ1's 14 leave-one-day-out folds and UNK policy, so every contrast is a difference in bits of the same target and the conditions nest. All log-likelihoods are in bits per coda of the tempo class.

| Condition | Inputs besides the coda's own pattern |
|---|---|
| floor | none (the floor) |
| own | own history: the whale's last 3 tempo classes and last 3 patterns, one-hot by lag, and a shared own-tempo copy weight |
| own_T | own history + partner tempo (last 3 classes and a partner-tempo copy weight) |
| own_P | own history + partner pattern (last 3 patterns) |
| own_S | own history + S |
| own_T_P | own history + partner tempo (last 3 classes and a partner-tempo copy weight) + partner pattern (last 3 patterns) |
| own_T_S | own history + partner tempo (last 3 classes and a partner-tempo copy weight) + S |
| own_P_S | own history + partner pattern (last 3 patterns) + S |
| own_T_P_S | own history + partner tempo (last 3 classes and a partner-tempo copy weight) + partner pattern (last 3 patterns) + S |

DEC-24's base is own_S; +tempo is own_T_S, +pattern own_P_S, both own_T_P_S. Partner codas are those completed before the target's onset (`[D5]`). Tempo classes have no UNK; patterns follow RQ1's UNK policy per fold.

**Fitting.** Logistic v4's procedure (`logistic.grid_search_l2`): every non-S column, both copy weights and the intercept share one penalty, S has its own, both chosen per fold and condition on the validation day from [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] (ties go to the larger value); the floor has one penalty. Candidates that fail the convergence check are left out of the selection and counted (`skip_nonconverged_candidates`, as in every RQ2 run). **Selection (v5, DEC-23).** For each fold and condition the candidates are the fits of every sub-condition (partner tempo, partner pattern and S each on or absent; own history always present); the highest validation-day log-likelihood wins (exact ties: fewer blocks). DEC-21's one-SE rule is the sensitivity, and the own fits (each condition's own fit, no absent candidates) are reported beside v5. DEC-23's stop rule on v5 governs each run.

## Checks before any test-day number

- **rhythm (primary)**: every pattern equals the coarse inventory's label and every tempo is a configured class (equal across runs); the folds equal the coarse inventory's and RQ1's; each fold's rhythm vocabulary (UNK included: 15–17 labels) equals the coarse inventory's. Under the max rule all 168 add-one-block contrasts (12 pairs × 14 validation days) are ≥ 0 on the validation day (smallest 0 bits). 0 of 5,166 penalty candidates failed the convergence check.
- **type (secondary)**: every pattern equals the coarse inventory's label and every tempo is a configured class (equal across runs); the folds equal the coarse inventory's and RQ1's; each fold's type vocabulary (UNK included: 19–22 labels) equals the coarse inventory's. Under the max rule all 168 add-one-block contrasts (12 pairs × 14 validation days) are ≥ 0 on the validation day (smallest 0 bits). 0 of 5,166 penalty candidates failed the convergence check.

## Stop rules

DEC-23's rule on this family's 13 add-one-block contrasts (the 12 pairs among the eight conditions, and own history added to the floor): an exact one-sided sign-flip test per check on the day sums of per-coda differences (alternative: mean below zero; all 2^14 sign assignments; statistic: token-weighted mean), Holm's correction across the 13, firing at an adjusted p < 0.05. DEC-16's old rule (day-bootstrap CI entirely below zero) is shown next to it.

| Run | Model | DEC-23 rule (sign-flip, Holm) | DEC-16 rule (CI below zero) |
|---|---|---|---|
| rhythm | own fit (no absent candidates) | none fired | none fired |
| rhythm | v5 (max) | none fired | none fired |
| rhythm | v5 (one-SE) | none fired | none fired |
| type | own fit (no absent candidates) | none fired | none fired |
| type | v5 (max) | none fired | none fired |
| type | v5 (one-SE) | none fired | none fired |

- Governing status, rhythm (DEC-23's rule on v5 (max)): none fired.
- Governing status, type (DEC-23's rule on v5 (max)): none fired.

## The four contrasts (DEC-24)

Paired per-coda differences in bits of the tempo class; positive days, with the days whose difference is exactly zero; day-bootstrap 95% CI (10,000 replicates); exact two-sided sign-flip p on the day sums (DEC-23).

### rhythm (primary: rhythm)

| Contrast | Conditions | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |
|---|---|---|---:|---:|---|---:|---:|
| partner tempo alone | own_T_S − own_S | own fit (no absent candidates) | 0.00327 | 0.02859 | [-0.0207, 0.0450] | 8/14 | 0.824 |
| partner tempo alone | own_T_S − own_S | v5 (max) | 0.00539 | 0.02049 | [-0.0165, 0.0379] | 6/14 (3 at 0) | 0.6748 |
| partner tempo alone | own_T_S − own_S | v5 (one-SE) | 0.00719 | 0.01923 | [-0.0050, 0.0308] | 5/14 (5 at 0) | 0.4141 |
| partner rhythm alone | own_P_S − own_S | own fit (no absent candidates) | 0.00500 | 0.01262 | [-0.0119, 0.0280] | 10/14 | 0.6154 |
| partner rhythm alone | own_P_S − own_S | v5 (max) | 0.00072 | 0.01231 | [-0.0147, 0.0241] | 7/14 (4 at 0) | 0.9551 |
| partner rhythm alone | own_P_S − own_S | v5 (one-SE) | 0.00689 | 0.01480 | [-0.0011, 0.0256] | 4/14 (9 at 0) | 0.25 |
| tempo beyond rhythm | own_T_P_S − own_P_S | own fit (no absent candidates) | 0.01004 | 0.04542 | [-0.0155, 0.0571] | 11/14 | 0.5223 |
| tempo beyond rhythm | own_T_P_S − own_P_S | v5 (max) | 0.00739 | 0.03614 | [-0.0177, 0.0504] | 11/14 (1 at 0) | 0.6475 |
| tempo beyond rhythm | own_T_P_S − own_P_S | v5 (one-SE) | -0.00608 | 0.01036 | [-0.0267, 0.0238] | 4/14 (3 at 0) | 0.71 |
| rhythm beyond tempo | own_T_P_S − own_T_S | own fit (no absent candidates) | 0.01177 | 0.02945 | [-0.0116, 0.0410] | 10/14 | 0.3879 |
| rhythm beyond tempo | own_T_P_S − own_T_S | v5 (max) | 0.00273 | 0.02796 | [-0.0161, 0.0349] | 7/14 (3 at 0) | 0.7324 |
| rhythm beyond tempo | own_T_P_S − own_T_S | v5 (one-SE) | -0.00638 | 0.00593 | [-0.0221, 0.0179] | 4/14 (5 at 0) | 0.5508 |

### type (secondary: type)

| Contrast | Conditions | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |
|---|---|---|---:|---:|---|---:|---:|
| partner tempo alone | own_T_S − own_S | own fit (no absent candidates) | 0.00982 | 0.02040 | [-0.0120, 0.0464] | 7/14 | 0.5156 |
| partner tempo alone | own_T_S − own_S | v5 (max) | 0.01635 | 0.02254 | [-0.0021, 0.0493] | 6/14 (3 at 0) | 0.208 |
| partner tempo alone | own_T_S − own_S | v5 (one-SE) | 0.00538 | 0.00776 | [-0.0041, 0.0215] | 3/14 (8 at 0) | 0.5625 |
| partner type alone | own_P_S − own_S | own fit (no absent candidates) | -0.00203 | -0.00094 | [-0.0235, 0.0346] | 5/14 | 0.8997 |
| partner type alone | own_P_S − own_S | v5 (max) | 0.00503 | -0.00029 | [-0.0138, 0.0321] | 2/14 (7 at 0) | 0.8906 |
| partner type alone | own_P_S − own_S | v5 (one-SE) | 0.00399 | -0.00401 | [-0.0142, 0.0303] | 1/14 (9 at 0) | 1 |
| tempo beyond type | own_T_P_S − own_P_S | own fit (no absent candidates) | 0.01485 | 0.02442 | [-0.0059, 0.0411] | 11/14 | 0.1824 |
| tempo beyond type | own_T_P_S − own_P_S | v5 (max) | 0.00185 | 0.01885 | [-0.0205, 0.0349] | 8/14 (2 at 0) | 0.895 |
| tempo beyond type | own_T_P_S − own_P_S | v5 (one-SE) | 0.01535 | 0.03296 | [-0.0018, 0.0462] | 9/14 (4 at 0) | 0.125 |
| type beyond tempo | own_T_P_S − own_T_S | own fit (no absent candidates) | 0.00299 | 0.00309 | [-0.0153, 0.0258] | 8/14 | 0.7651 |
| type beyond tempo | own_T_P_S − own_T_S | v5 (max) | -0.00947 | -0.00398 | [-0.0289, 0.0119] | 4/14 (6 at 0) | 0.4297 |
| type beyond tempo | own_T_P_S − own_T_S | v5 (one-SE) | 0.01396 | 0.02120 | [-0.0070, 0.0572] | 6/14 (7 at 0) | 0.5156 |

Sign of each token-weighted mean and where its CI lies, v5 (max):

| Contrast | rhythm | type |
|---|---|---|
| partner tempo alone | +, CI spans 0 | +, CI spans 0 |
| partner pattern alone | +, CI spans 0 | +, CI spans 0 |
| tempo beyond pattern | +, CI spans 0 | +, CI spans 0 |
| pattern beyond tempo | +, CI spans 0 | −, CI spans 0 |

Reading (DEC-24): tempo beyond the pattern > 0 with the pattern beyond tempo ≈ 0 reads as tempo matching; the pattern beyond tempo > 0 reads as the partner's pattern carrying information about the coda's speed beyond the partner's own speed (for instance through a shared behavioural context), with no claim about meaning. Nulls read "not detectable at this data size" (14 days). A run whose governing stop rule fires supports no reading.

## By turn category (v5, descriptive)

The four contrasts on the targets of each turn category; token-weighted means, no CIs and no tests (DEC-24).

| Run | Contrast | Turn category | Targets | Token-weighted mean | Positive days |
|---|---|---|---:|---:|---:|
| rhythm | partner tempo alone | overlap | 441 | -0.00106 | 7/14 (3 at 0) |
| rhythm | partner tempo alone | switch | 645 | 0.02144 | 8/14 (3 at 0) |
| rhythm | partner tempo alone | self_continuation | 461 | -0.01091 | 6/14 (3 at 0) |
| rhythm | partner rhythm alone | overlap | 441 | 0.01199 | 5/14 (4 at 0) |
| rhythm | partner rhythm alone | switch | 645 | -0.00403 | 5/14 (4 at 0) |
| rhythm | partner rhythm alone | self_continuation | 461 | -0.00341 | 6/14 (4 at 0) |
| rhythm | tempo beyond rhythm | overlap | 441 | 0.00701 | 9/14 (1 at 0) |
| rhythm | tempo beyond rhythm | switch | 645 | 0.02531 | 8/14 (1 at 0) |
| rhythm | tempo beyond rhythm | self_continuation | 461 | -0.01730 | 8/14 (1 at 0) |
| rhythm | rhythm beyond tempo | overlap | 441 | 0.02006 | 6/14 (3 at 0) |
| rhythm | rhythm beyond tempo | switch | 645 | -0.00016 | 6/14 (3 at 0) |
| rhythm | rhythm beyond tempo | self_continuation | 461 | -0.00980 | 7/14 (3 at 0) |
| type | partner tempo alone | overlap | 441 | 0.00671 | 6/14 (3 at 0) |
| type | partner tempo alone | switch | 645 | 0.04391 | 10/14 (3 at 0) |
| type | partner tempo alone | self_continuation | 461 | -0.01299 | 6/14 (3 at 0) |
| type | partner type alone | overlap | 441 | 0.00184 | 1/14 (7 at 0) |
| type | partner type alone | switch | 645 | 0.00464 | 2/14 (7 at 0) |
| type | partner type alone | self_continuation | 461 | 0.00863 | 4/14 (7 at 0) |
| type | tempo beyond type | overlap | 441 | 0.00565 | 5/14 (2 at 0) |
| type | tempo beyond type | switch | 645 | 0.02570 | 8/14 (2 at 0) |
| type | tempo beyond type | self_continuation | 461 | -0.03516 | 7/14 (2 at 0) |
| type | type beyond tempo | overlap | 441 | 0.00078 | 4/14 (6 at 0) |
| type | type beyond tempo | switch | 645 | -0.01357 | 3/14 (6 at 0) |
| type | type beyond tempo | self_continuation | 461 | -0.01353 | 3/14 (6 at 0) |

## Lattice

Mean log-likelihood in bits per coda of the tempo class, accuracy and macro-F1. Within a run all conditions predict the same target; the two runs condition on different patterns.

### rhythm

| Condition | Own-fit LL | v5 LL | v5 one-SE LL | Own-fit accuracy | v5 accuracy | Own-fit macro-F1 | v5 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| floor | -1.74148 | -1.74148 | -1.74148 | 60.763% | 60.763% | 0.3098 | 0.3098 |
| own | -0.62239 | -0.62239 | -0.62239 | 87.524% | 87.524% | 0.8066 | 0.8066 |
| own_T | -0.61291 | -0.60788 | -0.61312 | 86.878% | 87.524% | 0.7875 | 0.8084 |
| own_P | -0.61128 | -0.62030 | -0.62005 | 87.330% | 87.460% | 0.7976 | 0.8037 |
| own_S | -0.61326 | -0.61331 | -0.61799 | 87.654% | 87.654% | 0.8101 | 0.8101 |
| own_T_P | -0.61106 | -0.61627 | -0.61823 | 86.942% | 87.201% | 0.7907 | 0.7984 |
| own_T_S | -0.61000 | -0.60792 | -0.61080 | 87.201% | 87.330% | 0.7945 | 0.8024 |
| own_P_S | -0.60826 | -0.61258 | -0.61110 | 87.395% | 87.330% | 0.8060 | 0.8053 |
| own_T_P_S | -0.59822 | -0.60519 | -0.61718 | 87.201% | 87.589% | 0.7869 | 0.7970 |

Test targets per tempo class: t0 321, t1 56, t2 231, t3 160, t4 779.

### type

| Condition | Own-fit LL | v5 LL | v5 one-SE LL | Own-fit accuracy | v5 accuracy | Own-fit macro-F1 | v5 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| floor | -1.18116 | -1.18116 | -1.18116 | 67.938% | 67.938% | 0.5174 | 0.5174 |
| own | -0.55881 | -0.55881 | -0.55881 | 88.623% | 88.623% | 0.8339 | 0.8339 |
| own_T | -0.54642 | -0.54705 | -0.55311 | 87.977% | 88.171% | 0.8156 | 0.8245 |
| own_P | -0.56187 | -0.55128 | -0.55233 | 87.524% | 88.494% | 0.8135 | 0.8324 |
| own_S | -0.55994 | -0.56021 | -0.55977 | 88.365% | 88.365% | 0.8277 | 0.8277 |
| own_T_P | -0.54427 | -0.54864 | -0.54013 | 88.171% | 88.559% | 0.8152 | 0.8266 |
| own_T_S | -0.55012 | -0.54386 | -0.55439 | 88.106% | 88.300% | 0.8214 | 0.8238 |
| own_P_S | -0.56197 | -0.55518 | -0.55578 | 87.718% | 88.365% | 0.8171 | 0.8277 |
| own_T_P_S | -0.54712 | -0.55333 | -0.54043 | 88.235% | 88.171% | 0.8189 | 0.8188 |

Test targets per tempo class: t0 321, t1 56, t2 231, t3 160, t4 779.

## Sub-conditions chosen

Folds (of 14) on which each sub-condition was chosen: max rule, with the one-SE rule in parentheses; a dash marks a sub-condition that is not a candidate.

### rhythm

| Condition | own | own_T | own_P | own_S | own_T_P | own_T_S | own_P_S | own_T_P_S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 14 (14) | — | — | — | — | — | — | — |
| own_T | 5 (7) | 9 (7) | — | — | — | — | — | — |
| own_P | 6 (9) | — | 8 (5) | — | — | — | — | — |
| own_S | 2 (9) | — | — | 12 (5) | — | — | — | — |
| own_T_P | 2 (3) | 2 (4) | 0 (3) | — | 10 (4) | — | — | — |
| own_T_S | 1 (6) | 3 (5) | — | 2 (1) | — | 8 (2) | — | — |
| own_P_S | 0 (5) | — | 3 (4) | 4 (4) | — | — | 7 (1) | — |
| own_T_P_S | 0 (1) | 0 (2) | 0 (2) | 1 (1) | 1 (3) | 2 (2) | 0 (0) | 10 (3) |

### type

| Condition | own | own_T | own_P | own_S | own_T_P | own_T_S | own_P_S | own_T_P_S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 14 (14) | — | — | — | — | — | — | — |
| own_T | 2 (8) | 12 (6) | — | — | — | — | — | — |
| own_P | 8 (10) | — | 6 (4) | — | — | — | — | — |
| own_S | 6 (11) | — | — | 8 (3) | — | — | — | — |
| own_T_P | 2 (4) | 4 (4) | 0 (1) | — | 8 (5) | — | — | — |
| own_T_S | 0 (7) | 5 (3) | — | 3 (1) | — | 6 (3) | — | — |
| own_P_S | 3 (9) | — | 4 (3) | 4 (1) | — | — | 3 (1) | — |
| own_T_P_S | 0 (3) | 1 (2) | 0 (1) | 2 (1) | 3 (4) | 3 (2) | 0 (0) | 5 (1) |

## Penalties and copy weights (own fits)

Chosen penalties over the folds (min / median / max, and how often at an edge of the grid), the ranges of the own-tempo (β) and partner-tempo (γ) copy weights, and penalty candidates that failed the convergence check.

### rhythm

| Condition | History L2 | S L2 | β | γ | Not converged |
|---|---|---|---|---|---:|
| floor | 0.0001 / 0.0002 / 0.3 (7/14 at an edge) | n/a | n/a | n/a | 0/126 |
| own | 0.0001 / 0.002 / 0.01 (1/14 at an edge) | n/a | 1.813 to 2.525 | n/a | 0/126 |
| own_T | 0.0001 / 0.001 / 0.01 (1/14 at an edge) | n/a | 1.675 to 2.418 | 0.428 to 0.677 | 0/126 |
| own_P | 0.0001 / 0.002 / 0.01 (2/14 at an edge) | n/a | 1.775 to 2.748 | n/a | 0/126 |
| own_S | 0.0001 / 0.002 / 0.01 (1/14 at an edge) | 0.0001 / 0.02 / 1 (7/14 at an edge) | 1.810 to 2.687 | n/a | 0/1134 |
| own_T_P | 0.0001 / 0.001 / 0.01 (2/14 at an edge) | n/a | 1.619 to 2.530 | 0.536 to 0.997 | 0/126 |
| own_T_S | 0.0001 / 0.001 / 0.01 (1/14 at an edge) | 0.0001 / 0.01 / 1 (5/14 at an edge) | 1.690 to 2.417 | 0.406 to 0.676 | 0/1134 |
| own_P_S | 0.0001 / 0.002 / 0.01 (2/14 at an edge) | 0.0001 / 0.01 / 1 (4/14 at an edge) | 1.773 to 2.795 | n/a | 0/1134 |
| own_T_P_S | 0.0001 / 0.001 / 0.01 (2/14 at an edge) | 0.0001 / 0.01 / 1 (4/14 at an edge) | 1.635 to 2.557 | 0.532 to 0.999 | 0/1134 |

### type

| Condition | History L2 | S L2 | β | γ | Not converged |
|---|---|---|---|---|---:|
| floor | 0.0001 / 0.0001 / 0.0003 (11/14 at an edge) | n/a | n/a | n/a | 0/126 |
| own | 0.0001 / 0.00065 / 0.01 (5/14 at an edge) | n/a | 1.617 to 1.907 | n/a | 0/126 |
| own_T | 0.0001 / 0.001 / 0.003 (4/14 at an edge) | n/a | 1.476 to 1.699 | 0.312 to 0.684 | 0/126 |
| own_P | 0.0001 / 0.001 / 0.003 (3/14 at an edge) | n/a | 1.605 to 1.819 | n/a | 0/126 |
| own_S | 0.0001 / 0.00065 / 0.01 (5/14 at an edge) | 0.0001 / 0.055 / 1 (9/14 at an edge) | 1.616 to 2.028 | n/a | 0/1134 |
| own_T_P | 0.0001 / 0.001 / 0.01 (3/14 at an edge) | n/a | 1.444 to 1.622 | 0.464 to 0.859 | 0/126 |
| own_T_S | 0.0001 / 0.00065 / 0.003 (5/14 at an edge) | 0.0001 / 0.02 / 1 (8/14 at an edge) | 1.476 to 1.805 | 0.312 to 0.699 | 0/1134 |
| own_P_S | 0.0001 / 0.001 / 0.003 (3/14 at an edge) | 0.0001 / 0.0165 / 1 (6/14 at an edge) | 1.628 to 1.927 | n/a | 0/1134 |
| own_T_P_S | 0.0001 / 0.001 / 0.01 (2/14 at an edge) | 0.0001 / 0.01 / 1 (8/14 at an edge) | 1.444 to 1.762 | 0.467 to 0.936 | 0/1134 |

## The thirteen stop-rule checks

Per-coda differences: token-weighted mean, day-bootstrap CI, positive days (with days at exactly zero), DEC-16 (CI entirely below zero), one-sided sign-flip p (alternative: mean below zero) and its Holm adjustment across the 13, DEC-23 (adjusted p below alpha).

### rhythm, own fit (no absent candidates)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding partner tempo to own (own_T − own) | 0.00947 | [-0.0115, 0.0479] | 9/14 | — | 0.771 | 1 | — |
| adding partner rhythm to own (own_P − own) | 0.01110 | [-0.0053, 0.0371] | 10/14 | — | 0.8742 | 1 | — |
| adding S to own (own_S − own) | 0.00912 | [0.0007, 0.0138] | 10/14 | — | 0.9559 | 1 | — |
| adding partner rhythm to own_T (own_T_P − own_T) | 0.00186 | [-0.0210, 0.0286] | 9/14 | — | 0.5559 | 1 | — |
| adding partner tempo to own_P (own_T_P − own_P) | 0.00023 | [-0.0180, 0.0334] | 8/14 | — | 0.5081 | 1 | — |
| adding S to own_T (own_T_S − own_T) | 0.00292 | [-0.0106, 0.0123] | 10/14 | — | 0.695 | 1 | — |
| adding partner tempo to own_S (own_T_S − own_S) | 0.00327 | [-0.0207, 0.0450] | 8/14 | — | 0.5881 | 1 | — |
| adding S to own_P (own_P_S − own_P) | 0.00302 | [-0.0134, 0.0121] | 10/14 | — | 0.6779 | 1 | — |
| adding partner rhythm to own_S (own_P_S − own_S) | 0.00500 | [-0.0119, 0.0280] | 10/14 | — | 0.6924 | 1 | — |
| adding S to own_T_P (own_T_P_S − own_T_P) | 0.01283 | [0.0034, 0.0241] | 10/14 | — | 0.9883 | 1 | — |
| adding partner rhythm to own_T_S (own_T_P_S − own_T_S) | 0.01177 | [-0.0116, 0.0410] | 10/14 | — | 0.8061 | 1 | — |
| adding partner tempo to own_P_S (own_T_P_S − own_P_S) | 0.01004 | [-0.0155, 0.0571] | 11/14 | — | 0.7389 | 1 | — |
| adding own history to the floor (own − floor) | 1.11910 | [0.9879, 1.3214] | 14/14 | — | 1 | 1 | — |

### rhythm, v5 (max)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding partner tempo to own (own_T − own) | 0.01450 | [0.0003, 0.0396] | 6/14 (5 at 0) | — | 0.9453 | 1 | — |
| adding partner rhythm to own (own_P − own) | 0.00209 | [-0.0096, 0.0242] | 5/14 (6 at 0) | — | 0.668 | 1 | — |
| adding S to own (own_S − own) | 0.00908 | [0.0007, 0.0137] | 9/14 (2 at 0) | — | 0.9543 | 1 | — |
| adding partner rhythm to own_T (own_T_P − own_T) | -0.00839 | [-0.0302, 0.0235] | 6/14 (4 at 0) | — | 0.3213 | 1 | — |
| adding partner tempo to own_P (own_T_P − own_P) | 0.00403 | [-0.0156, 0.0373] | 7/14 (2 at 0) | — | 0.6277 | 1 | — |
| adding S to own_T (own_T_S − own_T) | -0.00004 | [-0.0110, 0.0087] | 6/14 (4 at 0) | — | 0.4941 | 1 | — |
| adding partner tempo to own_S (own_T_S − own_S) | 0.00539 | [-0.0165, 0.0379] | 6/14 (3 at 0) | — | 0.6631 | 1 | — |
| adding S to own_P (own_P_S − own_P) | 0.00771 | [-0.0045, 0.0156] | 9/14 (3 at 0) | — | 0.876 | 1 | — |
| adding partner rhythm to own_S (own_P_S − own_S) | 0.00072 | [-0.0147, 0.0241] | 7/14 (4 at 0) | — | 0.5234 | 1 | — |
| adding S to own_T_P (own_T_P_S − own_T_P) | 0.01108 | [0.0021, 0.0210] | 9/14 (1 at 0) | — | 0.9806 | 1 | — |
| adding partner rhythm to own_T_S (own_T_P_S − own_T_S) | 0.00273 | [-0.0161, 0.0349] | 7/14 (3 at 0) | — | 0.6343 | 1 | — |
| adding partner tempo to own_P_S (own_T_P_S − own_P_S) | 0.00739 | [-0.0177, 0.0504] | 11/14 (1 at 0) | — | 0.6764 | 1 | — |
| adding own history to the floor (own − floor) | 1.11910 | [0.9879, 1.3214] | 14/14 | — | 1 | 1 | — |

### rhythm, v5 (one-SE)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding partner tempo to own (own_T − own) | 0.00927 | [-0.0014, 0.0294] | 5/14 (7 at 0) | — | 0.9062 | 1 | — |
| adding partner rhythm to own (own_P − own) | 0.00234 | [-0.0094, 0.0246] | 3/14 (9 at 0) | — | 0.6875 | 1 | — |
| adding S to own (own_S − own) | 0.00440 | [0.0003, 0.0088] | 4/14 (9 at 0) | — | 0.9688 | 1 | — |
| adding partner rhythm to own_T (own_T_P − own_T) | -0.00511 | [-0.0206, 0.0165] | 3/14 (6 at 0) | — | 0.3086 | 1 | — |
| adding partner tempo to own_P (own_T_P − own_P) | 0.00182 | [-0.0118, 0.0212] | 3/14 (6 at 0) | — | 0.5781 | 1 | — |
| adding S to own_T (own_T_S − own_T) | 0.00231 | [-0.0000, 0.0072] | 2/14 (11 at 0) | — | 0.875 | 1 | — |
| adding partner tempo to own_S (own_T_S − own_S) | 0.00719 | [-0.0050, 0.0308] | 5/14 (5 at 0) | — | 0.7949 | 1 | — |
| adding S to own_P (own_P_S − own_P) | 0.00895 | [0.0004, 0.0198] | 5/14 (9 at 0) | — | 1 | 1 | — |
| adding partner rhythm to own_S (own_P_S − own_S) | 0.00689 | [-0.0011, 0.0256] | 4/14 (9 at 0) | — | 0.9062 | 1 | — |
| adding S to own_T_P (own_T_P_S − own_T_P) | 0.00105 | [-0.0067, 0.0129] | 4/14 (8 at 0) | — | 0.5781 | 1 | — |
| adding partner rhythm to own_T_S (own_T_P_S − own_T_S) | -0.00638 | [-0.0221, 0.0179] | 4/14 (5 at 0) | — | 0.2754 | 1 | — |
| adding partner tempo to own_P_S (own_T_P_S − own_P_S) | -0.00608 | [-0.0267, 0.0238] | 4/14 (3 at 0) | — | 0.355 | 1 | — |
| adding own history to the floor (own − floor) | 1.11910 | [0.9879, 1.3214] | 14/14 | — | 1 | 1 | — |

### type, own fit (no absent candidates)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding partner tempo to own (own_T − own) | 0.01239 | [-0.0089, 0.0483] | 8/14 | — | 0.8083 | 1 | — |
| adding partner type to own (own_P − own) | -0.00306 | [-0.0252, 0.0314] | 4/14 | — | 0.4237 | 1 | — |
| adding S to own (own_S − own) | -0.00113 | [-0.0072, 0.0043] | 7/14 | — | 0.3502 | 1 | — |
| adding partner type to own_T (own_T_P − own_T) | 0.00215 | [-0.0101, 0.0211] | 8/14 | — | 0.6114 | 1 | — |
| adding partner tempo to own_P (own_T_P − own_P) | 0.01759 | [0.0003, 0.0437] | 10/14 | — | 0.9659 | 1 | — |
| adding S to own_T (own_T_S − own_T) | -0.00370 | [-0.0106, 0.0035] | 6/14 | — | 0.1703 | 1 | — |
| adding partner tempo to own_S (own_T_S − own_S) | 0.00982 | [-0.0120, 0.0464] | 7/14 | — | 0.7422 | 1 | — |
| adding S to own_P (own_P_S − own_P) | -0.00010 | [-0.0106, 0.0115] | 8/14 | — | 0.4921 | 1 | — |
| adding partner type to own_S (own_P_S − own_S) | -0.00203 | [-0.0235, 0.0346] | 5/14 | — | 0.4498 | 1 | — |
| adding S to own_T_P (own_T_P_S − own_T_P) | -0.00285 | [-0.0175, 0.0087] | 7/14 | — | 0.3457 | 1 | — |
| adding partner type to own_T_S (own_T_P_S − own_T_S) | 0.00299 | [-0.0153, 0.0258] | 8/14 | — | 0.6175 | 1 | — |
| adding partner tempo to own_P_S (own_T_P_S − own_P_S) | 0.01485 | [-0.0059, 0.0411] | 11/14 | — | 0.9089 | 1 | — |
| adding own history to the floor (own − floor) | 0.62235 | [0.4851, 0.7392] | 14/14 | — | 1 | 1 | — |

### type, v5 (max)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding partner tempo to own (own_T − own) | 0.01175 | [-0.0094, 0.0478] | 6/14 (2 at 0) | — | 0.7932 | 1 | — |
| adding partner type to own (own_P − own) | 0.00752 | [-0.0088, 0.0343] | 2/14 (8 at 0) | — | 0.5938 | 1 | — |
| adding S to own (own_S − own) | -0.00141 | [-0.0073, 0.0041] | 4/14 (6 at 0) | — | 0.3281 | 1 | — |
| adding partner type to own_T (own_T_P − own_T) | -0.00158 | [-0.0113, 0.0152] | 4/14 (6 at 0) | — | 0.418 | 1 | — |
| adding partner tempo to own_P (own_T_P − own_P) | 0.00265 | [-0.0184, 0.0371] | 6/14 (2 at 0) | — | 0.5801 | 1 | — |
| adding S to own_T (own_T_S − own_T) | 0.00319 | [-0.0066, 0.0134] | 5/14 (5 at 0) | — | 0.6875 | 1 | — |
| adding partner tempo to own_S (own_T_S − own_S) | 0.01635 | [-0.0021, 0.0493] | 6/14 (3 at 0) | — | 0.8965 | 1 | — |
| adding S to own_P (own_P_S − own_P) | -0.00390 | [-0.0147, 0.0035] | 3/14 (7 at 0) | — | 0.2344 | 1 | — |
| adding partner type to own_S (own_P_S − own_S) | 0.00503 | [-0.0138, 0.0321] | 2/14 (7 at 0) | — | 0.5625 | 1 | — |
| adding S to own_T_P (own_T_P_S − own_T_P) | -0.00470 | [-0.0186, 0.0033] | 4/14 (4 at 0) | — | 0.2305 | 1 | — |
| adding partner type to own_T_S (own_T_P_S − own_T_S) | -0.00947 | [-0.0289, 0.0119] | 4/14 (6 at 0) | — | 0.2148 | 1 | — |
| adding partner tempo to own_P_S (own_T_P_S − own_P_S) | 0.00185 | [-0.0205, 0.0349] | 8/14 (2 at 0) | — | 0.5527 | 1 | — |
| adding own history to the floor (own − floor) | 0.62235 | [0.4851, 0.7392] | 14/14 | — | 1 | 1 | — |

### type, v5 (one-SE)

| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |
|---|---:|---|---:|---|---:|---:|---|
| adding partner tempo to own (own_T − own) | 0.00570 | [-0.0037, 0.0221] | 3/14 (8 at 0) | — | 0.7344 | 1 | — |
| adding partner type to own (own_P − own) | 0.00648 | [-0.0095, 0.0328] | 1/14 (10 at 0) | — | 0.5625 | 1 | — |
| adding S to own (own_S − own) | -0.00097 | [-0.0065, 0.0032] | 1/14 (11 at 0) | — | 0.375 | 1 | — |
| adding partner type to own_T (own_T_P − own_T) | 0.01298 | [-0.0079, 0.0563] | 5/14 (7 at 0) | — | 0.7422 | 1 | — |
| adding partner tempo to own_P (own_T_P − own_P) | 0.01220 | [-0.0047, 0.0417] | 7/14 (4 at 0) | — | 0.8916 | 1 | — |
| adding S to own_T (own_T_S − own_T) | -0.00128 | [-0.0079, 0.0038] | 1/14 (10 at 0) | — | 0.3125 | 1 | — |
| adding partner tempo to own_S (own_T_S − own_S) | 0.00538 | [-0.0041, 0.0215] | 3/14 (8 at 0) | — | 0.7344 | 1 | — |
| adding S to own_P (own_P_S − own_P) | -0.00345 | [-0.0132, 0.0000] | 0/14 (12 at 0) | — | 0.25 | 1 | — |
| adding partner type to own_S (own_P_S − own_S) | 0.00399 | [-0.0142, 0.0303] | 1/14 (9 at 0) | — | 0.5312 | 1 | — |
| adding S to own_T_P (own_T_P_S − own_T_P) | -0.00030 | [-0.0044, 0.0040] | 1/14 (10 at 0) | — | 0.375 | 1 | — |
| adding partner type to own_T_S (own_T_P_S − own_T_S) | 0.01396 | [-0.0070, 0.0572] | 6/14 (7 at 0) | — | 0.75 | 1 | — |
| adding partner tempo to own_P_S (own_T_P_S − own_P_S) | 0.01535 | [-0.0018, 0.0462] | 9/14 (4 at 0) | — | 0.9385 | 1 | — |
| adding own history to the floor (own − floor) | 0.62235 | [0.4851, 0.7392] | 14/14 | — | 1 | 1 | — |

## Notes

- Day-bootstrap CIs are approximate with 14 days, and limited data and capacity push contrasts toward zero: a CI that spans zero may mean too little data (PLAN §9).
- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly favours keeping S; the one-SE rule counters this (DEC-23).
- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).
- Claims concern dyadic exchanges only; this report establishes no grammar, causal mechanism or meaning.
- Per fold and condition: the chosen fits in `models.jsonl`, every penalty candidate in `l2_selection.jsonl`, the v5 candidates and choices in `selection.jsonl`, the own fits' test predictions in `predictions.jsonl` (v5's are rebuilt from them with `logistic_v5.rebuild_predictions`), per-coda log-likelihoods of every model in `evaluation/scored_predictions.jsonl`; output directories: `data/results/rq2_partner_tempo/`, `data/results/rq2_partner_tempo_type/`.

## Inputs

SHA-256 of every input:

- `config/rq2_partner_tempo.json`: `557f459802654d42caf5e64f7a79adaa959fd33be3041ce0b01f2c953cfe562d`
- `data/processed/rq1_folds/manifest.json`: `66fb93262c981d2c3fa98f95d9e4d4ee091c9dc492fa7c36cca5b62a57fc676f`
- `data/processed/rq1_targets.jsonl`: `9c7245c36fcc392dec0e67db08b7d943cd89107eb1b34e3b2b8cc55e1934be02`
- `data/processed/rq2_c_rhythm_folds/manifest.json`: `7878a2c4d2c7a4c006173f72ae0477ae6c3389d55cb966428f0729df4e1a64f0`
- `data/processed/rq2_c_rhythm_targets.jsonl`: `a315defd8ae4c63f4f47963138e26436b34110b44bcfd1e52a2274b71dceb090`
- `data/processed/rq2_d_rhythm_tempo_folds/manifest.json`: `5c79d957695a6b39784961a01de5343f3411fe12f794c5033fb8570ed40bae88`
- `data/processed/rq2_d_rhythm_tempo_targets.jsonl`: `4b49578dfaab640d9b278b18ebe58f03331616e52faf70ddb6a3083febdc05e5`
- `data/processed/rq2_e_type_tempo_folds/manifest.json`: `521fc45410c050460fdad848df445124fbbedd78db61c0150d061d8b400bea9c`
- `data/processed/rq2_e_type_tempo_targets.jsonl`: `dbfb2fda49169aaa60e01a8e059f0d44ddce921118ee6cd0d1af268c2643c7f2`
