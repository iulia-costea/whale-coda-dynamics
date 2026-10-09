# RQ1 switch structure (PLAN [P5], DEC-28)

Inventory O (DEC-31): guy_coda_type with Sharma-flagged ornamented codas relabelled to their neighbour's type. DEC-31's primary is LL(PD) − LL(R) in A_B_S, this report's secondary. The Reading below applies DEC-28's rules to inventory O and is not DEC-31's reading (reports/rq1_ornament_merge.md).

Logistic v4 unchanged (labels `guy_coda_type`, targets, folds, UNK policy, S, the L2 grid and selection rule, the joint history and S penalties, copy weights) plus shared class-feature weights computed from the whale's own earlier codas in the exchange. Each is one scalar, penalized with the history penalty like the copy weights, and added to the logits of the classes it marks: run length (β_r log r on the class of the own last coda, r = the identical own codas ending the history), distinct order (δ1 on d1, the most recent own type different from the current one; δ2 on d2, the most recent own type different from both) and repertoire (ρ log(1 + n_c) on every class c, n_c = the whale's earlier codas of type c in the exchange, excluding the current run). The features use the whale's whole earlier stream in the exchange (its codas with an earlier onset, not only the 10 in A_history), with labels mapped by the fold's vocabulary exactly as A_history is (UNK included); they exist only in conditions with A, and a row without the needed history gets zeros.

Variants (class features): R: none (logistic v4); L: run_length; P: run_length, repertoire; D: run_length, distinct_1, distinct_2; PD: run_length, repertoire, distinct_1, distinct_2. Every variant fits all eight conditions in one environment (Python 3.11.7, NumPy 2.4.2, SciPy 1.17.1, one BLAS thread per fit): 24,570 fits, all converged (fit() raises otherwise). The four conditions without A (none, S, B, B_S) carry no class features; in every variant their penalty searches and test predictions are identical to R's (checked exactly). R is the reference for every contrast.

Contrasts are paired per coda between two variants' fits of one condition, with the [P1] repeat and switch parts; intervals are day-bootstrap (10,000 replicates, seed 42); the unweighted day mean is the mean of the 14 day-level contrasts.

## Descriptive table (no models)

Comparison targets of `rq1_targets.jsonl`, labels as stored in Ct (not fold-mapped). Own last a = A_history[-1], partner last b = B_history[-1]. The whale's stream: its codas in the same exchange with an earlier onset, all of them; r = the identical codas ending it; before = the stream without that run; d1 = before[-1]; d2 = the most recent label in before that is neither a nor d1. A switch is a target that differs from a. Day-level chances use that day's comparison targets: p(b) / (1 − p(a)) for landing on b, p(d1) / (1 − p(a)) for returning to d1, where p is the day's share of targets of that type. Intervals: day bootstrap, 10,000 replicates of the 14 days (seed 42), ratio of sums within each replicate. Run length is computed within A_history only in the two rows that say so.

Check against the expected values (computed independently in a Cowork session before this run, 2026-10-04): counts and point estimates must match at 3 decimals, interval bounds within ±0.015 (different random streams); 3 of 3 agree.

| Quantity | Computed | Expected | Check |
|---|---|---|---|
| Comparison targets | count 1,547 | count 1,547 | agrees |
| Targets that repeat the own last coda (a) | count 1,337 | count 1,337 | agrees |
| Repeat rate after an own coda (self_continuation) | 0.874 [0.810, 0.911]; count 403; n 461 | n/a | n/a |
| Repeat rate right after a partner coda (switch) | 0.857 [0.815, 0.895]; count 553; n 645 | n/a | n/a |
| Repeat rate overlapping a partner coda (overlap) | 0.864 [0.812, 0.906]; count 381; n 441 | n/a | n/a |
| Own last equals partner last (a = b) | 0.554 [0.411, 0.687]; count 857; n 1,547 | n/a | n/a |
| a = b expected from day-level frequencies (sum over types of p(own last) p(partner last)) | 0.495 [0.406, 0.587]; n 1,547 | n/a | n/a |
| a ≠ b: target = a | 0.787 [0.731, 0.815]; count 543; n 690 | n/a | n/a |
| a ≠ b: target = b | 0.090 [0.069, 0.127]; count 62; n 690 | n/a | n/a |
| a ≠ b: target is neither | 0.123 [0.081, 0.161]; count 85; n 690 | n/a | n/a |
| Switches with a ≠ b: lands on b | 0.422 [0.333, 0.585]; count 62; n 147 | n/a | n/a |
| Switches with a ≠ b: day-level chance p(b) / (1 − p(a)) | 0.384 [0.323, 0.500]; n 147 | n/a | n/a |
| Switches with a ≠ b: lands on b minus chance | 0.037 [-0.040, 0.144]; n 147 | n/a | n/a |
| Switches (target ≠ a) | count 210 | count 210 | agrees |
| Switches: a ≠ b and target = b | count 62 | n/a | n/a |
| Switches: a = b | count 63 | n/a | n/a |
| Switches: a ≠ b and target ≠ b | count 85 | n/a | n/a |
| Switches with a d1 (an earlier different own coda) | count 117 | n/a | n/a |
| Switches with a d1: returns to d1 | 0.410 [0.296, 0.538]; count 48; n 117 | n/a | n/a |
| Switches with a d1: day-level chance p(d1) / (1 − p(a)) | 0.280 [0.231, 0.356]; n 117 | n/a | n/a |
| Switches with a d1: returns to d1 minus chance | 0.131 [0.035, 0.223]; n 117 | n/a | n/a |
| Returns to d1, one other type before the run | 0.474 [0.344, 0.597]; chance 0.301; count 36; n 76 | n/a | n/a |
| Returns to d1, two other types before the run | 0.290 [0.133, 0.500]; chance 0.272; count 9; n 31 | n/a | n/a |
| Returns to d1, three or more other types before the run | 0.300 [0.000, 0.556]; chance 0.137; count 3; n 10 | n/a | n/a |
| Switches with a d2 (two or more other own types) | count 41 | n/a | n/a |
| Switches with a d2: returns to d2 | 0.268 [0.100, 0.484]; count 11; n 41 | n/a | n/a |
| Switches with a d2: returns to d1 | 0.293 [0.148, 0.460]; count 12; n 41 | n/a | n/a |
| Targets with run length r ≥ 3 (whole stream) | count 857 | n/a | n/a |
| Targets with r ≥ 10 (whole stream) | count 147 | n/a | n/a |
| Largest r (whole stream) | count 29 | n/a | n/a |
| Repeat rate, r = 1 within A_history (10 codas) | 0.725 [0.660, 0.798]; count 296; n 408 | n/a | n/a |
| Repeat rate, r = 2 within A_history | 0.869 [0.803, 0.907]; count 245; n 282 | n/a | n/a |
| Repeat rate, r = 3 within A_history | 0.900 [0.841, 0.931]; count 181; n 201 | n/a | n/a |
| Repeat rate, r = 4–5 within A_history | 0.920 [0.880, 0.943]; count 243; n 264 | n/a | n/a |
| Repeat rate, r = 6–10 within A_history | 0.949 [0.920, 0.987]; count 372; n 392 | n/a | n/a |
| Within whale and exchange: repeat rate at r ≥ 2 minus at r = 1 (r within A_history), mean over whale-exchanges with both | -0.033 [-0.089, 0.010]; n 209 | n/a | n/a |

## Gate: R here against the committed logistic v4 run

Chosen penalties agree on 98 of 98 fold and condition fits, and every condition's mean log-likelihood is within 0 bits/coda of v4's. DEC-25's amended gate requires at most 1 differing penalty and mean log-likelihoods within 1e-05 bits/coda: passed. Predicted probabilities are reported, not enforced (optimizer tolerance and BLAS threads move them by up to about 1e-4 at the smallest penalties): largest absolute difference 0 over all fits (0 where penalties agree); 0 fits differ by more than 1e-05.

| Condition | Fits | Penalties differing | Max abs probability difference | LL here | LL v4 | Difference |
|---|---:|---:|---:|---:|---:|---:|
| none | 14 | 0 | 0 | -2.5499119 | -2.5499119 | 0 |
| S | 14 | 0 | 0 | -2.4957417 | -2.4957417 | 0 |
| A | 14 | 0 | 0 | -0.9464060 | -0.9464060 | 0 |
| B | 14 | 0 | 0 | -2.4512979 | -2.4512979 | 0 |
| A_B | 14 | 0 | 0 | -0.9653990 | -0.9653990 | 0 |
| A_S | 14 | 0 | 0 | -0.9533052 | -0.9533052 | 0 |
| B_S | 14 | 0 | 0 | -2.4763783 | -2.4763783 | 0 |
| A_B_S | 14 | 0 | 0 | -0.9815934 | -0.9815934 | 0 |

Before any fit with this configuration, during development: the descriptive stage was run (step 1, its intended check); one fold's condition-A fit of variant PD on training and validation days only was run to time the unit tests; and the smoke test (one L2 value, three variants, 50 bootstrap replicates) produced a report on tiny settings, which was looked at only with every digit masked, to check its layout. No feature, variant, grid or rule changed after any of them.

## Log-likelihood and accuracy by variant

Held-out log-likelihood (bits/coda, 1,547 targets), accuracy, the difference to repeat-own-last (-1.13150 bits/coda) with its day-bootstrap CI, and the chosen history L2 (median over folds; folds at a grid edge).

| Variant | A LL | A accuracy | A − repeat-own-last [CI] | A history L2 (edge) | A_B_S LL | A_B_S accuracy | A_B_S − repeat-own-last [CI] | A_B_S history L2 (edge) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R | -0.94641 | 86.296% | 0.18510 [0.1305, 0.2782] | 0.001 (2/14) | -0.98159 | 85.520% | 0.14991 [0.0603, 0.2992] | 0.002 (2/14) |
| L | -0.94168 | 86.361% | 0.18982 [0.1343, 0.2878] | 0.001 (1/14) | -0.98276 | 85.714% | 0.14874 [0.0535, 0.3083] | 0.002 (3/14) |
| P | -0.92531 | 86.684% | 0.20619 [0.1311, 0.3436] | 0.001 (2/14) | -0.95848 | 85.585% | 0.17302 [0.0653, 0.3551] | 0.003 (3/14) |
| D | -0.91274 | 86.619% | 0.21877 [0.1472, 0.3490] | 0.001 (2/14) | -0.95394 | 85.520% | 0.17756 [0.0784, 0.3449] | 0.002 (2/14) |
| PD | -0.90370 | 86.813% | 0.22780 [0.1493, 0.3745] | 0.001 (2/14) | -0.94559 | 85.908% | 0.18592 [0.0787, 0.3688] | 0.002 (2/14) |

## Fitted weights

Min / median / max over the 14 folds of the copy weight β and each class-feature weight, at the chosen penalties. Per-fold values are in `summary.json` (`coefficients`).

| Variant | Condition | β (copy) | β_r | δ1 | δ2 | ρ |
|---|---|---:|---:|---:|---:|---:|
| R | A | 2.602 / 3.149 / 3.940 | — | — | — | — |
| R | A_B_S | 2.459 / 2.888 / 3.726 | — | — | — | — |
| L | A | 2.308 / 3.170 / 4.013 | -0.343 / 0.029 / 0.777 | — | — | — |
| L | A_B_S | 2.170 / 2.841 / 3.852 | -0.569 / 0.165 / 0.757 | — | — | — |
| P | A | 2.313 / 3.189 / 4.086 | -0.008 / 0.327 / 0.805 | — | — | 0.591 / 0.890 / 1.160 |
| P | A_B_S | 2.515 / 2.826 / 3.938 | -0.156 / 0.411 / 0.612 | — | — | 0.659 / 0.793 / 1.068 |
| D | A | 2.743 / 3.323 / 4.225 | 0.054 / 0.345 / 0.564 | 0.894 / 1.580 / 1.993 | 0.387 / 1.203 / 1.790 | — |
| D | A_B_S | 2.596 / 3.002 / 3.990 | -0.185 / 0.374 / 0.573 | 0.728 / 1.351 / 1.683 | 0.354 / 1.039 / 1.877 | — |
| PD | A | 2.743 / 3.325 / 4.217 | 0.165 / 0.376 / 0.623 | 0.585 / 1.273 / 1.608 | 0.265 / 0.970 / 1.455 | 0.382 / 0.489 / 0.800 |
| PD | A_B_S | 2.539 / 2.984 / 3.976 | -0.042 / 0.468 / 0.631 | 0.313 / 1.011 / 1.268 | 0.244 / 0.831 / 1.479 | 0.390 / 0.504 / 0.629 |

## Contrasts

Primary (pre-specified): LL(PD) - LL(P) in A_B_S: does the order of the whale's earlier switches add usable information beyond which types it has used and how often? Secondary (pre-specified): LL(PD) - LL(R) in A_B_S: does the collapsed view of own history add anything to v4? Both get the exact two-sided sign-flip p (DEC-23: all 2^14 sign assignments of the day sums, token-weighted mean as the statistic). Every other row is descriptive. Every contrast adds class features, so an interval entirely below zero is a fitting flag (DEC-28) and is not interpreted.

| Condition | Contrast | Role | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part [CI] | Switch part [CI] | Sign-flip p (two-sided) | Fitting flag |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| A_B_S | PD − P | **primary** | 0.01290 | 0.00688 | [-0.0010, 0.0259] | 8/14 | 0.01103 [-0.0014, 0.0210] | 0.00186 [-0.0032, 0.0090] | 0.1433 | no |
| A_B_S | PD − R | **secondary** | 0.03601 | 0.03840 | [0.0029, 0.0781] | 12/14 | 0.01214 [-0.0043, 0.0270] | 0.02387 [0.0018, 0.0560] | 0.0726 | no |
| A_B_S | L − R | descriptive | -0.00117 | -0.00645 | [-0.0259, 0.0172] | 6/14 | 0.00024 [-0.0155, 0.0119] | -0.00141 [-0.0111, 0.0065] |  | no |
| A_B_S | D − L | descriptive | 0.02882 | 0.03263 | [0.0158, 0.0452] | 12/14 | 0.01097 [0.0021, 0.0211] | 0.01785 [0.0063, 0.0335] |  | no |
| A_B_S | P − L | descriptive | 0.02428 | 0.03797 | [0.0088, 0.0516] | 13/14 | 0.00086 [-0.0153, 0.0201] | 0.02342 [0.0065, 0.0462] |  | no |
| A_B_S | PD − D | descriptive | 0.00836 | 0.01222 | [-0.0003, 0.0248] | 9/14 | 0.00093 [-0.0068, 0.0099] | 0.00743 [0.0003, 0.0187] |  | no |
| A | PD − P | descriptive | 0.02161 | 0.02773 | [0.0110, 0.0340] | 11/14 | 0.01518 [0.0050, 0.0245] | 0.00644 [0.0005, 0.0163] |  | no |
| A | PD − R | descriptive | 0.04271 | 0.06528 | [0.0125, 0.0983] | 12/14 | 0.01118 [-0.0008, 0.0285] | 0.03152 [0.0082, 0.0745] |  | no |
| A | L − R | descriptive | 0.00472 | 0.01080 | [-0.0016, 0.0162] | 8/14 | 0.00047 [-0.0031, 0.0041] | 0.00425 [-0.0005, 0.0141] |  | no |
| A | D − L | descriptive | 0.02895 | 0.04120 | [0.0057, 0.0644] | 10/14 | 0.01158 [0.0050, 0.0210] | 0.01736 [-0.0010, 0.0461] |  | no |
| A | P − L | descriptive | 0.01637 | 0.02676 | [-0.0099, 0.0601] | 8/14 | -0.00446 [-0.0220, 0.0149] | 0.02084 [-0.0027, 0.0561] |  | no |
| A | PD − D | descriptive | 0.00904 | 0.01329 | [-0.0012, 0.0271] | 9/14 | -0.00087 [-0.0087, 0.0072] | 0.00991 [0.0016, 0.0238] |  | no |

No contrast has an interval entirely below zero (no fitting flag).

## Stop rules per variant (DEC-23, with DEC-16 beside it)

Each variant is a full run of the 16 checks: the 12 add-one-input contrasts and the 4 A-conditions against repeat-own-last. DEC-23's rule, which governs (DEC-28), gives each check an exact one-sided sign-flip test (alternative: mean below zero) on its day sums, with Holm's correction across the 16, and fires when an adjusted p is below 0.05 (`logistic_v5.dec23_stop_rules`). DEC-16's rule, reported beside it, fires when a check's day-bootstrap interval lies entirely below zero. Each check's day sums are the variant's stored day means times the day's target count, as in the [P2] sweep.

| Variant | DEC-23: Holm-adjusted p < 0.05 | Smallest one-sided p, Holm-adjusted (check) | DEC-16: interval entirely below zero |
|---|---|---|---|
| R | no stop rule fires | 0.1235, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |
| L | no stop rule fires | 0.1211, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |
| P | no stop rule fires | 0.1106, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |
| D | no stop rule fires | 0.1311, Holm 1.0000 (add_input:A_B_S-A_S) | no stop rule fires |
| PD | no stop rule fires | 0.0974, Holm 1.0000 (add_input:A_S-A) | no stop rule fires |

## Reading (DEC-28, primary only)

Primary, pre-specified: LL(PD) - LL(P) in A_B_S: 0.01290 bits/coda (unweighted day mean 0.00688, 95% CI [-0.0010, 0.0259], 8/14 days positive, two-sided sign-flip p = 0.1433); repeat part 0.01103 [-0.0014, 0.0210], switch part 0.00186 [-0.0032, 0.0090].
Not detectable at this data size. The primary rests on few discriminating switches (only 41 switches have two or more other own types to choose between), so its null is weak; no calibration is run. Limited data and capacity push contrasts toward zero (PLAN §9).
Repertoire (P - L in A_B_S) positive, CI [0.0088, 0.0516], with the primary null: switch choices follow the whale's active repertoire, consistent with stable individual or unit repertoires (Gero, Whitehead & Rendell 2016b).
Secondary, pre-specified (numbers only; no reading): LL(PD) - LL(R) in A_B_S: 0.03601 bits/coda (unweighted day mean 0.03840, 95% CI [0.0029, 0.0781], 12/14 days positive, two-sided sign-flip p = 0.0726); repeat part 0.01214 [-0.0043, 0.0270], switch part 0.02387 [0.0018, 0.0560].

## Limits

- The primary rests on few discriminating switches: only 41 switches have two or more other own types to choose between, so a null is weak.
- No calibration (planted signal) was run, so no size of order effect can be ruled out.
- Run after the results freeze (2026-10-04) as a final run; the descriptive numbers that motivated it were seen first (DEC-28). The features, variants, grid and rules were fixed before any fit.
- The class-feature weights share one history penalty with the lag one-hots, the partner block and the copy weights (v4's rule), so a null may partly reflect that compromise.
- Day-bootstrap intervals are approximate with only 14 days; the switch part rests on about 270 non-repeat codas and is exploratory.
- Claims are about dyadic exchanges only; nothing here is evidence of grammar or syntax.

Per-variant outputs are in `data/results/rq1_switch_structure_ornament_merged/<variant>/` (summary.json with both stop-rule inputs and input SHA-256, l2_selection.jsonl, scored.jsonl for A and A_B_S, coefficients.jsonl), the descriptive table in `data/results/rq1_switch_structure_ornament_merged/descriptive/`; `data/results/rq1_switch_structure_ornament_merged/summary.json` records SHA-256 of every input to this report.
