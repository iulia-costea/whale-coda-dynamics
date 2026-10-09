# RQ1 duration lattice (PLAN [P6], DEC-30)

RQ1's eight conditions with a type-free target: the coda's log duration y = ln d (d = `offset_s − onset_s`, in seconds), for RQ1's 1,547 comparison targets, its 14 leave-one-day-out folds with their validation days, exchanges, S and `[D5]` (all unchanged). Every condition has the always-present inputs: the target's click count n (raw `nClicks`) as ln n, and one-hots for the click counts with at least 10 training targets in the fold; variant T adds the target's `guy_coda_type` one-hot (fold vocabulary, UNK per `[M6]`). A and B: for lags 1–3 of the whale's own and the partner's completed codas, ln d and ln n (standardised on the non-missing training values, 0 when missing) and a missing indicator; columns constant on the training rows are dropped. S: logistic v4's S columns. History inputs are type-free in both variants. The floor (`none`) has the always-present inputs only.

**Model.** Ridge regression on y standardised on the fold's training targets, in closed form: mean squared error + λ_h‖w_A, w_B‖² + λ_S‖w_S‖², a fixed 0.0001 on the always-present weights and none on the intercept. Penalties from logistic v4's grid [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], chosen per fold and condition on the validation day by mean held-out log density (ties: the larger value on each axis); an axis is searched only when its block has columns (S: λ_S; A, B, A,B: λ_h; A,S, B,S, A,B,S: every pair). σ̂² is the mean squared training residual. A coda's score is log2 N(y; ŷ, σ̂²) in bits per coda; contrasts are paired differences, so the unit of y cancels. Bits per coda of this continuous target are not comparable with RQ1's bits per coda of a type. Bootstrap: 10,000 day-level replicates (seed 42). Added after the results freeze (2026-10-04) as a final run.

## Data checks

Checked before any model against values computed independently (computed independently before the run (DEC-30 prompt, 2026-10-09)): counts exactly, other values within 0.0005 (agreement at 3 decimals). All reproduce.

| Quantity | Expected | Computed | Reproduced |
|---|---|---|---|
| rows | count: 2075 | count: 2075 | yes |
| comparison_targets | count: 1547 | count: 1547 | yes |
| history_references | count: 7914, unresolved: 0, ending_after_target_onset: 0 | count: 7914, unresolved: 0, ending_after_target_onset: 0 | yes |
| duration_vs_raw | max_abs_below: 1e-09 | max_abs_below: 3.4561242756581123e-12 | yes |
| onset_vs_raw | max_abs: 0 | max_abs: 0.0 | yes |
| nclicks_vs_intervals | count: 2 | count: 2 | yes |
| own_lags | counts: [1547, 1324, 1088] | counts: [1547, 1324, 1088] | yes |
| partner_lags | counts: [1547, 1324, 1084] | counts: [1547, 1324, 1084] | yes |
| targets_by_clicks | counts: {'3': 32, '4': 111, '5': 1222, '6': 52, '7': 22, '8': 21, '9': 19, '10': 9, '11': 2, '12': 4, '13': 2, '14': 2, '15': 6, '16': 1, '17': 2, '18': 3, '19': 10, '20': 8, '21': 7, '22': 3, '23': 7, '24': 1, '25': 1} | counts: {'3': 32, '4': 111, '5': 1222, '6': 52, '7': 22, '8': 21, '9': 19, '10': 9, '11': 2, '12': 4, '13': 2, '14': 2, '15': 6, '16': 1, '17': 2, '18': 3, '19': 10, '20': 8, '21': 7, '22': 3, '23': 7, '24': 1, '25': 1} | yes |
| target_log_duration | mean: -0.1795, sd: 0.5405 | mean: -0.1794679813379479, sd: 0.540451414668645 | yes |
| target_log_duration_residual_sd | click_count: 0.5138, type: 0.2081 | click_count: 0.5137796745424167, type: 0.20810453335316262 | yes |
| lag1_r | own: 0.94, partner: 0.772 | own: 0.9401760329040567, partner: 0.7718206094322729 | yes |
| lag1_r_within_click_count | own: 0.933, partner: 0.769 | own: 0.9327886594321535, partner: 0.7692483090215771 | yes |
| lag1_r_within_type | own: 0.568, partner: 0.369 | own: 0.5678069204951752, partner: 0.36912556881422376 | yes |
| lag1_r_within_type:self_continuation | n: 461, own: 0.616, partner: 0.301 | n: 461, own: 0.6160607479724368, partner: 0.30120130962124947 | yes |
| lag1_r_within_type:switch | n: 645, own: 0.556, partner: 0.376 | n: 645, own: 0.5563710625800287, partner: 0.37643283632309 | yes |
| lag1_r_within_type:overlap | n: 441, own: 0.541, partner: 0.445 | n: 441, own: 0.5408131891541523, partner: 0.4449022599849834 | yes |

Rows whose raw `nClicks` differs from the number of positive ICI1–ICI40 plus one (the run uses `nClicks`):

| coda_id | source_row_index | nClicks | positive intervals | guy_coda_type | comparison target |
|---|---:|---:|---:|---|---|
| assayag:row:1635 | 1635 | 7 | 5 | 6i | yes |
| assayag:row:3606 | 3606 | 5 | 5 | 6R | yes |

Process notes:

- The first implementation of the data checks centred the lag-1 variables on the targets' group means instead of their own (each variable's mean over the rows sharing its coda's click count or type). Those five rows differed from the expected values (within click count: own 0.937, partner 0.769; within type: 0.578, 0.372; self-continuation 0.614, 0.301; switch 0.566, 0.383; overlap 0.558, 0.447). The code was corrected to the stated definition, which reproduces every expected value, before any model was fitted; no expected value was changed.
- During development the smoke test (one penalty value, 0.01, not this grid; 50 bootstrap replicates; f = 0.3 only) produced numbers. Its report was viewed only with every digit masked, to check the layout; that view still showed which smoke contrasts were flagged or detectable. No input, variant, grid, calibration or rule changed after it.
- After the first full run (code 3805633), the planted partner contrasts (+0.053, +0.088, +0.127 bits/coda at f = 0.1, 0.2, 0.3) exceeded DEC-30's nominal sizes even after subtracting the real-data contrast: z_b1 correlates with the A,S residual, so planting along it adds a cross term (A,S's training sigma rose 1.7%, 3.7%, 6.2% against 0.5%, 2.0%, 4.4% for an independent z_b1). The report then gained the increment over the real-data contrast and the sigma ratios, and stopped calling the nominal sizes an upper bound. The planting, fractions, refits and detection rule did not change; every stage was rerun from a clean tree.

## Stop rules

DEC-23's rule on each variant's 16 checks (the 12 add-one-input contrasts, and the 4 A-conditions against copying the own last coda's ln d, scored as a Gaussian with the training mean squared difference as variance): exact one-sided sign-flip test on the day sums (alternative: mean below zero; all 2^14 sign assignments), Holm across the 16, firing at an adjusted p < 0.05. DEC-16's rule (interval entirely below zero) beside it. An add-one-input contrast whose interval lies entirely below zero is a fitting flag and is not interpreted.

| Variant | DEC-23 rule | DEC-16 rule | Fitting flags |
|---|---|---|---|
| F | none fired | none fired | none |
| T | none fired | none fired | none |

## Primary and secondary contrasts

Paired per coda; token-weighted mean (the estimand) and unweighted day mean; day-bootstrap 95% CI; exact two-sided sign-flip p on the day sums; Holm's correction across the three secondaries. Detectable: interval entirely above zero and p < 0.05 (Holm-adjusted for the secondaries).

| Role | Contrast | Variant | Conditions | Token-weighted | Unweighted day mean | 95% CI | Positive days | Two-sided p | Holm p | Detectable |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| primary | partner | F | A_B_S − A_S | 0.02598 | 0.04663 | [-0.0640, 0.0979] | 10/14 | 0.583 | — | no |
| secondary | own history | F | A_B_S − B_S | 0.92580 | 1.10427 | [0.7415, 1.1630] | 13/14 | 0.0003662 | 0.001099 | yes |
| secondary | timing | F | A_B_S − A_B | -0.01116 | -0.00915 | [-0.0405, 0.0031] | 8/14 | 0.4901 | 0.4901 | no |
| secondary | partner | T | A_B_S − A_S | 0.07673 | 0.08414 | [0.0066, 0.1829] | 10/14 | 0.08289 | 0.1658 | no |

## Reading (DEC-30)

- Primary, pre-specified: partner LL(A,B,S) − LL(A,S) in F: +0.0260 bits/coda (unweighted day mean +0.0466, 95% CI [-0.0640, 0.0979], 10/14 days positive, two-sided sign-flip p = 0.583).
- Secondary, pre-specified: own history LL(A,B,S) − LL(B,S) in F: +0.9258 bits/coda (unweighted day mean +1.1043, 95% CI [0.7415, 1.1630], 13/14 days positive, two-sided sign-flip p = 0.0003662, Holm-adjusted p = 0.001099); timing LL(A,B,S) − LL(A,B) in F: -0.0112 bits/coda (unweighted day mean -0.0092, 95% CI [-0.0405, 0.0031], 8/14 days positive, two-sided sign-flip p = 0.4901, Holm-adjusted p = 0.4901); partner LL(A,B,S) − LL(A,S) in T: +0.0767 bits/coda (unweighted day mean +0.0841, 95% CI [0.0066, 0.1829], 10/14 days positive, two-sided sign-flip p = 0.08289, Holm-adjusted p = 0.1658).
- F: own history > 0 with partner and timing not detectable: RQ1's pattern doesn't depend on the coda types.
- Primary null: the partner's previous codas add no detectable information about a coda's duration beyond own history and S — not detectable at this data size. Calibration: the smallest planted partner effect detected is f = 0.3: planted partner contrast +0.127 bits/coda (interval [0.0339, 0.2225]), +0.101 (interval [0.0852, 0.1336]) above the real-data contrast; f = 0.1 (+0.053), f = 0.2 (+0.088) were not detected. DEC-30's nominal sizes, ½·log2(1 + f²), assume z_b1 independent of the reduced model's residual; here the planted increment exceeds them at f = 0.1, 0.2, 0.3 (z_b1 is correlated with that residual, so the plant adds a cross term; see the calibration section). A null may also mean too little data: limited data and capacity push contrasts toward zero (PLAN §9).
- Claims are about dyadic exchanges only; nothing here is about grammar or syntax.

## Calibration (planted partner effect)

Variant F, partner contrast only: y is replaced by y + f·σ̂_AS·z_b1 in every split, where z_b1 is the standardised ln d of the partner's last coda (the A,B,S design's own column, standardised on the training rows) and σ̂_AS the fold's training residual SD of condition A,S (median 0.1696); conditions A,B,S and A,S are refitted with the full penalty search and scored on the planted test day. The smallest f whose interval lies above zero is the detection bound. The planted contrast includes the real data's own partner contrast; the increment over it is the paired per-coda difference between the planted and the real-data contrast. DEC-30's nominal size ½·log2(1 + f²) is the increment if z_b1 (unit variance) were independent of A, S and the A,S residual; when z_b1 correlates with that residual (the real partner signal), planting along it also adds a cross term, which shows as A,S's training σ̂ rising by more than √(1 + f²).

| f | DEC-30 nominal size | Planted contrast | Unweighted day mean | 95% CI | Positive days | Two-sided p | Detected | Increment over real-data contrast [95% CI] | σ̂ ratio A,S (√(1 + f²)) | σ̂ ratio A,B,S |
|---:|---:|---:|---:|---|---:|---:|---|---|---:|---:|
| 0.1 | 0.0072 | 0.05323 | 0.07688 | [-0.0410, 0.1326] | 11/14 | 0.2493 | no | 0.02725 [0.0210, 0.0373] | 1.0164 (1.0050) | 1.0000 |
| 0.2 | 0.0283 | 0.08779 | 0.10912 | [-0.0029, 0.1719] | 12/14 | 0.08179 | no | 0.06181 [0.0509, 0.0814] | 1.0366 (1.0198) | 1.0000 |
| 0.3 | 0.0622 | 0.12730 | 0.17658 | [0.0339, 0.2225] | 12/14 | 0.02649 | yes | 0.10132 [0.0852, 0.1336] | 1.0603 (1.0440) | 1.0000 |

Real-data partner contrast in F (the primary): 0.02598 bits/coda, 95% CI [-0.0640, 0.0979]. σ̂ ratios are medians over folds of the planted fit's training σ̂ over the real fit's.

Calibration: the smallest planted partner effect detected is f = 0.3: planted partner contrast +0.127 bits/coda (interval [0.0339, 0.2225]), +0.101 (interval [0.0852, 0.1336]) above the real-data contrast; f = 0.1 (+0.053), f = 0.2 (+0.088) were not detected. DEC-30's nominal sizes, ½·log2(1 + f²), assume z_b1 independent of the reduced model's residual; here the planted increment exceeds them at f = 0.1, 0.2, 0.3 (z_b1 is correlated with that residual, so the plant adds a cross term; see the calibration section).

## Lattice

Held-out mean log density in bits per coda; the gain over the floor (paired, with day-bootstrap CI); held-out RMS residual of ln d; median training σ̂ over folds; median chosen penalties with the folds at a grid edge. Copy-own-last baseline: 0.29328 bits/coda, held-out RMS residual 0.1887.

### F (type-free)

| Condition | Inputs | Bits/coda | Gain over floor | 95% CI | Held-out RMS residual | Median σ̂ | λ_h median (at edge) | λ_S median (at edge) |
|---|---|---:|---:|---|---:|---:|---|---|
| none | always-present inputs only (the floor) | -1.25466 | — | — | 0.5532 | 0.5148 | n/a | n/a |
| S | S | -1.19893 | 0.05573 | [-0.0026, 0.0956] | 0.5329 | 0.4852 | n/a | 0.002 (6/14) |
| A | A | 0.38364 | 1.63830 | [1.1501, 2.0984] | 0.1840 | 0.1699 | 0.00055 (8/14) | n/a |
| B | B | -0.53620 | 0.71846 | [0.2789, 1.1235] | 0.3407 | 0.3233 | 0.00505 (10/14) | n/a |
| A_B | A, B | 0.41342 | 1.66808 | [1.1676, 2.1526] | 0.1820 | 0.1643 | 0.00155 (8/14) | n/a |
| A_S | A, S | 0.37628 | 1.63094 | [1.1356, 2.0928] | 0.1856 | 0.1696 | 0.00055 (8/14) | 0.065 (6/14) |
| B_S | B, S | -0.52354 | 0.73111 | [0.2927, 1.1331] | 0.3377 | 0.3199 | 0.01505 (10/14) | 0.03 (9/14) |
| A_B_S | A, B, S | 0.40226 | 1.65692 | [1.1418, 2.1438] | 0.1842 | 0.1640 | 0.00155 (8/14) | 0.165 (6/14) |

### T (the target's type given)

| Condition | Inputs | Bits/coda | Gain over floor | 95% CI | Held-out RMS residual | Median σ̂ | λ_h median (at edge) | λ_S median (at edge) |
|---|---|---:|---:|---|---:|---:|---|---|
| none | always-present inputs only (the floor) | -0.02110 | — | — | 0.2388 | 0.2047 | n/a | n/a |
| S | S | -0.02205 | -0.00095 | [-0.0173, 0.0134] | 0.2382 | 0.2009 | n/a | 0.01 (8/14) |
| A | A | 0.66750 | 0.68860 | [0.3996, 0.9167] | 0.1512 | 0.1299 | 0.0001 (9/14) | n/a |
| B | B | 0.18545 | 0.20655 | [0.0107, 0.3285] | 0.2041 | 0.1698 | 0.065 (8/14) | n/a |
| A_B | A, B | 0.73194 | 0.75304 | [0.4782, 0.9742] | 0.1436 | 0.1244 | 0.00505 (8/14) | n/a |
| A_S | A, S | 0.66993 | 0.69103 | [0.4043, 0.9178] | 0.1510 | 0.1299 | 0.0001 (9/14) | 0.2 (8/14) |
| B_S | B, S | 0.21014 | 0.23124 | [0.0254, 0.3584] | 0.2007 | 0.1674 | 0.055 (9/14) | 0.065 (6/14) |
| A_B_S | A, B, S | 0.74665 | 0.76775 | [0.5008, 0.9829] | 0.1430 | 0.1249 | 0.00155 (8/14) | 0.03 (10/14) |

## All twelve add-one-input contrasts

### F

| Pair | Token-weighted | Unweighted day mean | 95% CI | Positive days | Fitting flag |
|---|---:|---:|---|---:|---|
| S − none | 0.05573 | 0.05855 | [-0.0026, 0.0956] | 10/14 | no |
| A − none | 1.63830 | 1.64293 | [1.1501, 2.0984] | 12/14 | no |
| B − none | 0.71846 | 0.56132 | [0.2789, 1.1235] | 11/14 | no |
| A_B − A | 0.02978 | 0.05334 | [-0.0537, 0.1000] | 11/14 | no |
| A_B − B | 0.94962 | 1.13494 | [0.7687, 1.1869] | 13/14 | no |
| A_S − S | 1.57521 | 1.58193 | [1.1030, 2.0516] | 13/14 | no |
| A_S − A | -0.00736 | -0.00245 | [-0.0297, 0.0054] | 7/14 | no |
| B_S − S | 0.67538 | 0.52429 | [0.2569, 1.0904] | 11/14 | no |
| B_S − B | 0.01265 | 0.02152 | [-0.0178, 0.0408] | 10/14 | no |
| A_B_S − A_B | -0.01116 | -0.00915 | [-0.0405, 0.0031] | 8/14 | no |
| A_B_S − A_S | 0.02598 | 0.04663 | [-0.0640, 0.0979] | 10/14 | no |
| A_B_S − B_S | 0.92580 | 1.10427 | [0.7415, 1.1630] | 13/14 | no |

### T

| Pair | Token-weighted | Unweighted day mean | 95% CI | Positive days | Fitting flag |
|---|---:|---:|---|---:|---|
| S − none | -0.00095 | -0.00153 | [-0.0173, 0.0134] | 8/14 | no |
| A − none | 0.68860 | 0.67305 | [0.3996, 0.9167] | 12/14 | no |
| B − none | 0.20655 | 0.10067 | [0.0107, 0.3285] | 10/14 | no |
| A_B − A | 0.06444 | 0.06442 | [-0.0045, 0.1652] | 9/14 | no |
| A_B − B | 0.54649 | 0.63681 | [0.3782, 0.7302] | 13/14 | no |
| A_S − S | 0.69198 | 0.67130 | [0.4012, 0.9207] | 12/14 | no |
| A_S − A | 0.00242 | -0.00328 | [-0.0114, 0.0180] | 9/14 | no |
| B_S − S | 0.23219 | 0.11610 | [0.0324, 0.3569] | 10/14 | no |
| B_S − B | 0.02469 | 0.01390 | [0.0021, 0.0516] | 11/14 | no |
| A_B_S − A_B | 0.01472 | 0.01644 | [0.0051, 0.0310] | 10/14 | no |
| A_B_S − A_S | 0.07673 | 0.08414 | [0.0066, 0.1829] | 10/14 | no |
| A_B_S − B_S | 0.53651 | 0.63934 | [0.3665, 0.7360] | 14/14 | no |

## By turn category and by repeat (descriptive)

RQ1's three contrasts within groups of targets: token-weighted mean and positive days; no CIs or tests. Repeat: the target's stored type equals the whale's own last coda's.

| Variant | Contrast | Group | Targets | Token-weighted | Positive days |
|---|---|---|---:|---:|---:|
| F | own history | self_continuation | 461 | 1.12609 | 13/14 |
| F | own history | switch | 645 | 0.81183 | 12/14 |
| F | own history | overlap | 441 | 0.88313 | 13/14 |
| F | own history | repeats own last type | 1277 | 1.35679 | 14/14 |
| F | own history | switches type | 270 | -1.11260 | 6/14 |
| F | partner | self_continuation | 461 | -0.03097 | 7/14 |
| F | partner | switch | 645 | 0.09902 | 11/14 |
| F | partner | overlap | 441 | -0.02132 | 9/14 |
| F | partner | repeats own last type | 1277 | -0.01250 | 8/14 |
| F | partner | switches type | 270 | 0.20799 | 11/14 |
| F | timing | self_continuation | 461 | -0.00345 | 7/14 |
| F | timing | switch | 645 | -0.01610 | 7/14 |
| F | timing | overlap | 441 | -0.01200 | 9/14 |
| F | timing | repeats own last type | 1277 | -0.00560 | 8/14 |
| F | timing | switches type | 270 | -0.03746 | 4/14 |
| T | own history | self_continuation | 461 | 0.60869 | 12/14 |
| T | own history | switch | 645 | 0.43402 | 13/14 |
| T | own history | overlap | 441 | 0.61096 | 13/14 |
| T | own history | repeats own last type | 1277 | 0.64878 | 14/14 |
| T | own history | switches type | 270 | 0.00553 | 9/14 |
| T | partner | self_continuation | 461 | -0.01755 | 7/14 |
| T | partner | switch | 645 | 0.16167 | 10/14 |
| T | partner | overlap | 441 | 0.05105 | 9/14 |
| T | partner | repeats own last type | 1277 | 0.06896 | 8/14 |
| T | partner | switches type | 270 | 0.11347 | 5/14 |
| T | timing | self_continuation | 461 | -0.00267 | 6/14 |
| T | timing | switch | 645 | 0.01733 | 8/14 |
| T | timing | overlap | 441 | 0.02906 | 9/14 |
| T | timing | repeats own last type | 1277 | 0.00853 | 8/14 |
| T | timing | switches type | 270 | 0.04399 | 6/14 |

## Weights on the last coda's ln d (fold summaries)

Fitted change in predicted ln d per unit of the own (A) or partner (B) last coda's ln d, on the original scale (1 = full copy, given the other inputs); min, median and max over the 14 folds.

| Variant | Condition | Own lag 1 (min / median / max) | Partner lag 1 (min / median / max) |
|---|---|---|---|
| F | A | 0.326 / 0.848 / 0.864 | — |
| F | B | — | 0.264 / 0.575 / 0.645 |
| F | A_B | 0.244 / 0.768 / 0.781 | 0.100 / 0.119 / 0.187 |
| F | A_S | 0.313 / 0.847 / 0.861 | — |
| F | B_S | — | 0.254 / 0.563 / 0.631 |
| F | A_B_S | 0.234 / 0.767 / 0.780 | 0.100 / 0.119 / 0.187 |
| T | A | 0.095 / 0.510 / 0.553 | — |
| T | B | — | 0.074 / 0.196 / 0.261 |
| T | A_B | 0.083 / 0.430 / 0.471 | 0.061 / 0.110 / 0.124 |
| T | A_S | 0.095 / 0.509 / 0.552 | — |
| T | B_S | — | 0.072 / 0.199 / 0.267 |
| T | A_B_S | 0.081 / 0.425 / 0.468 | 0.060 / 0.109 / 0.127 |

## Residual diagnostics

Held-out residuals standardised by each fold's σ̂, pooled over the test days. A Gaussian has skewness 0, excess kurtosis 0, 0.27% beyond 3σ and mean squared standardised residual 1.

| Model | Condition | Skewness | Excess kurtosis | Beyond 3σ̂ | Mean squared standardised residual |
|---|---|---:|---:|---:|---:|
| F | none | -0.970 | -0.329 | 1 (0.06%) | 1.186 |
| F | S | -0.840 | -0.187 | 0 (0.00%) | 1.210 |
| F | A | 2.398 | 31.303 | 32 (2.07%) | 1.000 |
| F | B | -0.395 | 2.582 | 22 (1.42%) | 1.075 |
| F | A_B | 1.745 | 22.201 | 34 (2.20%) | 1.037 |
| F | A_S | 2.323 | 30.062 | 33 (2.13%) | 1.015 |
| F | B_S | -0.432 | 2.214 | 22 (1.42%) | 1.082 |
| F | A_B_S | 1.692 | 21.161 | 34 (2.20%) | 1.055 |
| T | none | -0.370 | 3.906 | 42 (2.71%) | 1.337 |
| T | S | -0.327 | 3.872 | 41 (2.65%) | 1.367 |
| T | A | 0.410 | 12.436 | 34 (2.20%) | 1.200 |
| T | B | -0.287 | 5.597 | 46 (2.97%) | 1.411 |
| T | A_B | 0.129 | 10.884 | 40 (2.59%) | 1.227 |
| T | A_S | 0.395 | 12.411 | 34 (2.20%) | 1.199 |
| T | B_S | -0.285 | 5.561 | 47 (3.04%) | 1.412 |
| T | A_B_S | 0.124 | 10.366 | 43 (2.78%) | 1.213 |
| copy own last | — | 2.304 | 39.510 | 36 (2.33%) | 1.035 |

## v5 line (descriptive)

DEC-23's absent candidates applied to these fits, no new fits: per fold and condition, the sub-condition fit with the highest validation-day mean log density (exact ties: fewer blocks); its test-day scores become the condition's. Not tested; no Holm.

### F

| Condition | v5 bits/coda | Sub-conditions chosen (folds) |
|---|---:|---|
| none | -1.25466 | none: 14 |
| S | -1.20495 | S: 12, none: 2 |
| A | 0.32407 | A: 13, none: 1 |
| B | -0.57042 | B: 11, none: 3 |
| A_B | 0.38081 | A: 3, A_B: 10, none: 1 |
| A_S | 0.32099 | A: 6, A_S: 7, S: 1 |
| B_S | -0.55135 | B: 3, B_S: 8, S: 2, none: 1 |
| A_B_S | 0.37713 | A: 1, A_B: 3, A_B_S: 7, A_S: 2, S: 1 |

| Contrast | Token-weighted | Unweighted day mean | 95% CI | Positive days | Two-sided p |
|---|---:|---:|---|---:|---:|
| own history (A_B_S − B_S) | 0.92847 | 1.16565 | [0.6669, 1.2636] | 12/14 (1 at 0) | 0.0007324 |
| partner (A_B_S − A_S) | 0.05614 | 0.07672 | [0.0232, 0.1103] | 9/14 (4 at 0) | 0.01172 |
| timing (A_B_S − A_B) | -0.00368 | -0.00164 | [-0.0168, 0.0045] | 6/14 (4 at 0) | 0.8027 |

### T

| Condition | v5 bits/coda | Sub-conditions chosen (folds) |
|---|---:|---|
| none | -0.02110 | none: 14 |
| S | -0.02282 | S: 12, none: 2 |
| A | 0.64710 | A: 13, none: 1 |
| B | 0.12940 | B: 12, none: 2 |
| A_B | 0.64986 | A: 8, A_B: 5, none: 1 |
| A_S | 0.64918 | A: 4, A_S: 9, none: 1 |
| B_S | 0.15218 | B: 2, B_S: 10, S: 1, none: 1 |
| A_B_S | 0.65817 | A: 2, A_B: 1, A_B_S: 5, A_S: 5, none: 1 |

| Contrast | Token-weighted | Unweighted day mean | 95% CI | Positive days | Two-sided p |
|---|---:|---:|---|---:|---:|
| own history (A_B_S − B_S) | 0.50599 | 0.60632 | [0.3397, 0.6809] | 13/14 (1 at 0) | 0.0002441 |
| partner (A_B_S − A_S) | 0.00899 | 0.02667 | [-0.0017, 0.0302] | 5/14 (8 at 0) | 0.1875 |
| timing (A_B_S − A_B) | 0.00831 | 0.01458 | [0.0001, 0.0254] | 6/14 (4 at 0) | 0.1211 |

## DEC-23 checks per variant

| Variant | Check | Token-weighted | 95% CI | Positive days | One-sided p | Holm p | DEC-23 fires | DEC-16 fires |
|---|---|---:|---|---:|---:|---:|---|---|
| F | add_input:S-none | 0.05573 | [-0.0026, 0.0956] | 10/14 | 0.9343 | 1 | no | no |
| F | add_input:A-none | 1.63830 | [1.1501, 2.0984] | 12/14 | 0.9991 | 1 | no | no |
| F | add_input:B-none | 0.71846 | [0.2789, 1.1235] | 11/14 | 0.9837 | 1 | no | no |
| F | add_input:A_B-A | 0.02978 | [-0.0537, 0.1000] | 11/14 | 0.7611 | 1 | no | no |
| F | add_input:A_B-B | 0.94962 | [0.7687, 1.1869] | 13/14 | 0.9999 | 1 | no | no |
| F | add_input:A_S-S | 1.57521 | [1.1030, 2.0516] | 13/14 | 0.9991 | 1 | no | no |
| F | add_input:A_S-A | -0.00736 | [-0.0297, 0.0054] | 7/14 | 0.3559 | 1 | no | no |
| F | add_input:B_S-S | 0.67538 | [0.2569, 1.0904] | 11/14 | 0.9833 | 1 | no | no |
| F | add_input:B_S-B | 0.01265 | [-0.0178, 0.0408] | 10/14 | 0.8026 | 1 | no | no |
| F | add_input:A_B_S-A_B | -0.01116 | [-0.0405, 0.0031] | 8/14 | 0.2451 | 1 | no | no |
| F | add_input:A_B_S-A_S | 0.02598 | [-0.0640, 0.0979] | 10/14 | 0.7086 | 1 | no | no |
| F | add_input:A_B_S-B_S | 0.92580 | [0.7415, 1.1630] | 13/14 | 0.9999 | 1 | no | no |
| F | copy_own_last:A | 0.09036 | [-0.1108, 0.2318] | 9/14 | 0.8229 | 1 | no | no |
| F | copy_own_last:A_B | 0.12014 | [-0.0453, 0.2244] | 12/14 | 0.9086 | 1 | no | no |
| F | copy_own_last:A_S | 0.08300 | [-0.1114, 0.2122] | 9/14 | 0.8078 | 1 | no | no |
| F | copy_own_last:A_B_S | 0.10898 | [-0.0625, 0.2127] | 12/14 | 0.8868 | 1 | no | no |
| T | add_input:S-none | -0.00095 | [-0.0173, 0.0134] | 8/14 | 0.4549 | 1 | no | no |
| T | add_input:A-none | 0.68860 | [0.3996, 0.9167] | 12/14 | 0.9984 | 1 | no | no |
| T | add_input:B-none | 0.20655 | [0.0107, 0.3285] | 10/14 | 0.9579 | 1 | no | no |
| T | add_input:A_B-A | 0.06444 | [-0.0045, 0.1652] | 9/14 | 0.9225 | 1 | no | no |
| T | add_input:A_B-B | 0.54649 | [0.3782, 0.7302] | 13/14 | 0.9999 | 1 | no | no |
| T | add_input:A_S-S | 0.69198 | [0.4012, 0.9207] | 12/14 | 0.9984 | 1 | no | no |
| T | add_input:A_S-A | 0.00242 | [-0.0114, 0.0180] | 9/14 | 0.7035 | 1 | no | no |
| T | add_input:B_S-S | 0.23219 | [0.0324, 0.3569] | 10/14 | 0.9643 | 1 | no | no |
| T | add_input:B_S-B | 0.02469 | [0.0021, 0.0516] | 11/14 | 0.9702 | 1 | no | no |
| T | add_input:A_B_S-A_B | 0.01472 | [0.0051, 0.0310] | 10/14 | 0.9877 | 1 | no | no |
| T | add_input:A_B_S-A_S | 0.07673 | [0.0066, 0.1829] | 10/14 | 0.9586 | 1 | no | no |
| T | add_input:A_B_S-B_S | 0.53651 | [0.3665, 0.7360] | 14/14 | 1 | 1 | no | no |
| T | copy_own_last:A | 0.37422 | [0.2274, 0.5298] | 11/14 | 0.9976 | 1 | no | no |
| T | copy_own_last:A_B | 0.43866 | [0.2776, 0.6622] | 12/14 | 0.9984 | 1 | no | no |
| T | copy_own_last:A_S | 0.37665 | [0.2316, 0.5328] | 11/14 | 0.9979 | 1 | no | no |
| T | copy_own_last:A_B_S | 0.45338 | [0.2901, 0.6846] | 12/14 | 0.9988 | 1 | no | no |

## Limits

- The partner's coda that overlaps the target isn't complete at the target's onset and isn't an input, so duration matching within overlaps is outside this design.
- Gaussian residuals: heavy tails affect every condition but can inflate differences; diagnostics above.
- Bits per coda of a continuous target are not comparable with RQ1's bits per coda of a type.
- Ornaments (an extra final click) aren't flagged: conditioning on click count groups an ornamented coda with codas of its click count, and in T many ornamented codas have their own type.
- Added after the results freeze (2026-10-04) as a final run.
- Not done: heavy-tailed or nonlinear models; the residual LSTM; inter-click-interval targets; other inventories in T.

## Run

7,840 penalty candidates in the fit stage (F 3.1 s, T 3.9 s; 13.4 s with data preparation and the baseline), 6,804 in the calibration (12.3 s); Python 3.10.21, NumPy 2.2.6, SciPy 1.15.3, one BLAS thread; code at commit 784b14d5693375ebdf7da4bfbabc71b293b8a4c4. Outputs in `data/results/rq1_duration_lattice/` (`checks/`, `fit/`, `calibration/`); `summary.json` records SHA-256 of every input to this report, as listed in the config.
