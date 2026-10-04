# RQ1 switch structure (PLAN [P5], DEC-28)

Logistic v4 unchanged (labels `guy_coda_type`, targets, folds, UNK policy, S, the L2 grid and selection rule, the joint history and S penalties, copy weights) plus shared class-feature weights computed from the whale's own earlier codas in the exchange. Each is one scalar, penalized with the history penalty like the copy weights, and added to the logits of the classes it marks: run length (β_r log r on the class of the own last coda, r = the identical own codas ending the history), distinct order (δ1 on d1, the most recent own type different from the current one; δ2 on d2, the most recent own type different from both) and repertoire (ρ log(1 + n_c) on every class c, n_c = the whale's earlier codas of type c in the exchange, excluding the current run). The features use the whale's whole earlier stream in the exchange (its codas with an earlier onset, not only the 10 in A_history), with labels mapped by the fold's vocabulary exactly as A_history is (UNK included); they exist only in conditions with A, and a row without the needed history gets zeros.

Variants (class features): R: none (logistic v4); L: run_length; P: run_length, repertoire; D: run_length, distinct_1, distinct_2; PD: run_length, repertoire, distinct_1, distinct_2. Every variant fits all eight conditions in one environment (Python 3.10.21, NumPy 2.2.6, SciPy 1.15.3, one BLAS thread per fit): 24,570 fits, all converged (fit() raises otherwise). The four conditions without A (none, S, B, B_S) carry no class features; in every variant their penalty searches and test predictions are identical to R's (checked exactly). R is the reference for every contrast.

Contrasts are paired per coda between two variants' fits of one condition, with the [P1] repeat and switch parts; intervals are day-bootstrap (10,000 replicates, seed 42); the unweighted day mean is the mean of the 14 day-level contrasts.

## Descriptive table (no models)

Comparison targets of `rq1_targets.jsonl`, labels as stored in Ct (not fold-mapped). Own last a = A_history[-1], partner last b = B_history[-1]. The whale's stream: its codas in the same exchange with an earlier onset, all of them; r = the identical codas ending it; before = the stream without that run; d1 = before[-1]; d2 = the most recent label in before that is neither a nor d1. A switch is a target that differs from a. Day-level chances use that day's comparison targets: p(b) / (1 − p(a)) for landing on b, p(d1) / (1 − p(a)) for returning to d1, where p is the day's share of targets of that type. Intervals: day bootstrap, 10,000 replicates of the 14 days (seed 42), ratio of sums within each replicate. Run length is computed within A_history only in the two rows that say so.

Check against the expected values (computed independently in a Cowork session before this run, 2026-10-04): counts and point estimates must match at 3 decimals, interval bounds within ±0.015 (different random streams); 36 of 36 agree.

| Quantity | Computed | Expected | Check |
|---|---|---|---|
| Comparison targets | count 1,547 | count 1,547 | agrees |
| Targets that repeat the own last coda (a) | count 1,277 | count 1,277 | agrees |
| Repeat rate after an own coda (self_continuation) | 0.822 [0.744, 0.885]; count 379; n 461 | 0.822 [0.741, 0.884]; n 461 | agrees |
| Repeat rate right after a partner coda (switch) | 0.825 [0.771, 0.871]; count 532; n 645 | 0.825 [0.770, 0.872]; n 645 | agrees |
| Repeat rate overlapping a partner coda (overlap) | 0.830 [0.775, 0.874]; count 366; n 441 | 0.830 [0.773, 0.873]; n 441 | agrees |
| Own last equals partner last (a = b) | 0.548 [0.405, 0.681]; count 847; n 1,547 | 0.548 [0.404, 0.681]; count 847 | agrees |
| a = b expected from day-level frequencies (sum over types of p(own last) p(partner last)) | 0.488 [0.399, 0.579]; n 1,547 | 0.488 [0.399, 0.578] | agrees |
| a ≠ b: target = a | 0.716 [0.663, 0.750]; count 501; n 700 | 0.716 [0.663, 0.750]; n 700 | agrees |
| a ≠ b: target = b | 0.107 [0.085, 0.147]; count 75; n 700 | 0.107 [0.085, 0.148]; n 700 | agrees |
| a ≠ b: target is neither | 0.177 [0.122, 0.218]; count 124; n 700 | 0.177 [0.122, 0.219]; n 700 | agrees |
| Switches with a ≠ b: lands on b | 0.377 [0.298, 0.529]; count 75; n 199 | 0.377; n 199 | agrees |
| Switches with a ≠ b: day-level chance p(b) / (1 − p(a)) | 0.375 [0.307, 0.499]; n 199 | 0.375; n 199 | agrees |
| Switches with a ≠ b: lands on b minus chance | 0.002 [-0.071, 0.085]; n 199 | 0.002 [-0.072, 0.086]; n 199 | agrees |
| Switches (target ≠ a) | count 270 | count 270 | agrees |
| Switches: a ≠ b and target = b | count 75 | count 75 | agrees |
| Switches: a = b | count 71 | count 71 | agrees |
| Switches: a ≠ b and target ≠ b | count 124 | count 124 | agrees |
| Switches with a d1 (an earlier different own coda) | count 168 | count 168 | agrees |
| Switches with a d1: returns to d1 | 0.476 [0.407, 0.567]; count 80; n 168 | 0.476; n 168 | agrees |
| Switches with a d1: day-level chance p(d1) / (1 − p(a)) | 0.268 [0.224, 0.344]; n 168 | 0.268; n 168 | agrees |
| Switches with a d1: returns to d1 minus chance | 0.208 [0.148, 0.261]; n 168 | 0.208 [0.149, 0.260]; n 168 | agrees |
| Returns to d1, one other type before the run | 0.545 [0.462, 0.633]; chance 0.291; count 60; n 110 | 0.545; chance 0.291; n 110 | agrees |
| Returns to d1, two other types before the run | 0.372 [0.276, 0.500]; chance 0.237; count 16; n 43 | 0.372; chance 0.237; n 43 | agrees |
| Returns to d1, three or more other types before the run | 0.267 [0.111, 0.500]; chance 0.190; count 4; n 15 | 0.267; chance 0.190; n 15 | agrees |
| Switches with a d2 (two or more other own types) | count 58 | count 58 | agrees |
| Switches with a d2: returns to d2 | 0.207 [0.069, 0.314]; count 12; n 58 | 0.207; n 58 | agrees |
| Switches with a d2: returns to d1 | 0.345 [0.263, 0.459]; count 20; n 58 | 0.345; n 58 | agrees |
| Targets with run length r ≥ 3 (whole stream) | count 808 | count 808 | agrees |
| Targets with r ≥ 10 (whole stream) | count 113 | count 113 | agrees |
| Largest r (whole stream) | count 21 | count 21 | agrees |
| Repeat rate, r = 1 within A_history (10 codas) | 0.651 [0.580, 0.735]; count 298; n 458 | 0.651; n 458 | agrees |
| Repeat rate, r = 2 within A_history | 0.861 [0.806, 0.900]; count 242; n 281 | 0.861; n 281 | agrees |
| Repeat rate, r = 3 within A_history | 0.881 [0.831, 0.914]; count 178; n 202 | 0.881; n 202 | agrees |
| Repeat rate, r = 4–5 within A_history | 0.900 [0.858, 0.932]; count 235; n 261 | 0.900; n 261 | agrees |
| Repeat rate, r = 6–10 within A_history | 0.939 [0.894, 0.987]; count 324; n 345 | 0.939; n 345 | agrees |
| Within whale and exchange: repeat rate at r ≥ 2 minus at r = 1 (r within A_history), mean over whale-exchanges with both | -0.007 [-0.056, 0.040]; n 206 | -0.007 [-0.058, 0.040]; n 206 | agrees |

## Gate: R here against the committed logistic v4 run

Chosen penalties agree on 98 of 98 fold and condition fits, and every condition's mean log-likelihood is within 2.4e-06 bits/coda of v4's. DEC-25's amended gate requires at most 1 differing penalty and mean log-likelihoods within 1e-05 bits/coda: passed. Predicted probabilities are reported, not enforced (optimizer tolerance and BLAS threads move them by up to about 1e-4 at the smallest penalties): largest absolute difference 0.000113 over all fits (0.000113 where penalties agree); 17 fits differ by more than 1e-05.

| Condition | Fits | Penalties differing | Max abs probability difference | LL here | LL v4 | Difference |
|---|---:|---:|---:|---:|---:|---:|
| none | 14 | 0 | 0 | -2.6025399 | -2.6025399 | 0 |
| S | 14 | 0 | 2.63e-05 | -2.5538779 | -2.5538792 | 1.2e-06 |
| A | 14 | 0 | 0.000105 | -1.0968960 | -1.0968984 | 2.4e-06 |
| B | 14 | 0 | 6.55e-05 | -2.4727769 | -2.4727774 | 4.8e-07 |
| A_B | 14 | 0 | 5.51e-05 | -1.1123701 | -1.1123708 | 7.4e-07 |
| A_S | 14 | 0 | 9.12e-05 | -1.1069470 | -1.1069467 | -3.2e-07 |
| B_S | 14 | 0 | 5.05e-05 | -2.4762998 | -2.4762999 | 4.5e-09 |
| A_B_S | 14 | 0 | 0.000113 | -1.1326053 | -1.1326062 | 9.4e-07 |

Before any fit with this configuration, during development: the descriptive stage was run (step 1, its intended check); one fold's condition-A fit of variant PD on training and validation days only was run to time the unit tests; and the smoke test (one L2 value, three variants, 50 bootstrap replicates) produced a report on tiny settings, which was looked at only with every digit masked, to check its layout. No feature, variant, grid or rule changed after any of them.

## Log-likelihood and accuracy by variant

Held-out log-likelihood (bits/coda, 1,547 targets), accuracy, the difference to repeat-own-last (-1.39276 bits/coda) with its day-bootstrap CI, and the chosen history L2 (median over folds; folds at a grid edge).

| Variant | A LL | A accuracy | A − repeat-own-last [CI] | A history L2 (edge) | A_B_S LL | A_B_S accuracy | A_B_S − repeat-own-last [CI] | A_B_S history L2 (edge) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R | -1.09690 | 82.353% | 0.29586 [0.2420, 0.3719] | 0.001 (3/14) | -1.13261 | 82.288% | 0.26015 [0.1855, 0.4047] | 0.001 (1/14) |
| L | -1.08841 | 82.999% | 0.30435 [0.2435, 0.4004] | 0.001 (3/14) | -1.13498 | 82.224% | 0.25778 [0.1814, 0.4034] | 0.001 (1/14) |
| P | -1.06188 | 83.452% | 0.33088 [0.2542, 0.4680] | 0.001 (2/14) | -1.09401 | 82.999% | 0.29875 [0.2117, 0.4716] | 0.001 (1/14) |
| D | -1.04989 | 83.258% | 0.34287 [0.2625, 0.4728] | 0.001 (1/14) | -1.09937 | 82.159% | 0.29339 [0.2030, 0.4669] | 0.001 (1/14) |
| PD | -1.03892 | 83.775% | 0.35383 [0.2668, 0.4994] | 0.001 (1/14) | -1.08091 | 83.193% | 0.31185 [0.2198, 0.4940] | 0.001 (1/14) |

## Fitted weights

Min / median / max over the 14 folds of the copy weight β and each class-feature weight, at the chosen penalties. Per-fold values are in `summary.json` (`coefficients`).

| Variant | Condition | β (copy) | β_r | δ1 | δ2 | ρ |
|---|---|---:|---:|---:|---:|---:|
| R | A | 2.381 / 2.754 / 3.448 | — | — | — | — |
| R | A_B_S | 2.332 / 2.451 / 3.018 | — | — | — | — |
| L | A | 2.135 / 2.775 / 3.572 | -0.532 / -0.227 / 0.691 | — | — | — |
| L | A_B_S | 2.283 / 2.508 / 3.160 | -0.552 / -0.110 / 0.390 | — | — | — |
| P | A | 2.135 / 2.789 / 3.625 | -0.184 / 0.215 / 0.733 | — | — | 0.677 / 0.827 / 1.167 |
| P | A_B_S | 2.296 / 2.549 / 3.230 | -0.216 / 0.221 / 0.524 | — | — | 0.691 / 0.788 / 1.089 |
| D | A | 2.533 / 2.959 / 3.805 | -0.013 / 0.281 / 0.531 | 1.284 / 1.625 / 1.789 | 0.333 / 1.085 / 1.475 | — |
| D | A_B_S | 2.424 / 2.719 / 3.442 | -0.101 / 0.299 / 0.501 | 1.257 / 1.493 / 1.705 | 0.333 / 0.889 / 1.557 | — |
| PD | A | 2.522 / 2.944 / 3.809 | 0.079 / 0.319 / 0.594 | 0.971 / 1.366 / 1.542 | 0.228 / 0.857 / 1.235 | 0.340 / 0.449 / 0.729 |
| PD | A_B_S | 2.456 / 2.703 / 3.445 | 0.005 / 0.389 / 0.520 | 0.924 / 1.250 / 1.381 | 0.260 / 0.706 / 1.317 | 0.381 / 0.440 / 0.649 |

## Contrasts

Primary (pre-specified): LL(PD) - LL(P) in A_B_S: does the order of the whale's earlier switches add usable information beyond which types it has used and how often? Secondary (pre-specified): LL(PD) - LL(R) in A_B_S: does the collapsed view of own history add anything to v4? Both get the exact two-sided sign-flip p (DEC-23: all 2^14 sign assignments of the day sums, token-weighted mean as the statistic). Every other row is descriptive. Every contrast adds class features, so an interval entirely below zero is a fitting flag (DEC-28) and is not interpreted.

| Condition | Contrast | Role | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part [CI] | Switch part [CI] | Sign-flip p (two-sided) | Fitting flag |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| A_B_S | PD − P | **primary** | 0.01310 | 0.01214 | [0.0025, 0.0267] | 10/14 | 0.01183 [0.0039, 0.0186] | 0.00127 [-0.0072, 0.0123] | 0.0525 | no |
| A_B_S | PD − R | **secondary** | 0.05170 | 0.06171 | [0.0234, 0.0986] | 13/14 | 0.02004 [0.0091, 0.0335] | 0.03166 [0.0101, 0.0675] | 0.0084 | no |
| A_B_S | L − R | descriptive | -0.00238 | -0.00313 | [-0.0065, 0.0008] | 7/14 | -0.00198 [-0.0059, 0.0015] | -0.00040 [-0.0022, 0.0009] |  | no |
| A_B_S | D − L | descriptive | 0.03561 | 0.04320 | [0.0167, 0.0707] | 11/14 | 0.01359 [0.0069, 0.0239] | 0.02201 [0.0057, 0.0491] |  | no |
| A_B_S | P − L | descriptive | 0.04098 | 0.05270 | [0.0208, 0.0754] | 12/14 | 0.01019 [0.0022, 0.0211] | 0.03078 [0.0119, 0.0596] |  | no |
| A_B_S | PD − D | descriptive | 0.01847 | 0.02163 | [0.0079, 0.0316] | 12/14 | 0.00843 [0.0028, 0.0137] | 0.01004 [0.0026, 0.0210] |  | no |
| A | PD − P | descriptive | 0.02295 | 0.01703 | [0.0032, 0.0389] | 9/14 | 0.02015 [0.0058, 0.0344] | 0.00280 [-0.0087, 0.0145] |  | no |
| A | PD − R | descriptive | 0.05797 | 0.08796 | [0.0111, 0.1483] | 10/14 | 0.02964 [0.0069, 0.0809] | 0.02833 [0.0024, 0.0679] |  | no |
| A | L − R | descriptive | 0.00849 | 0.02141 | [-0.0037, 0.0405] | 6/14 | 0.00892 [-0.0053, 0.0461] | -0.00043 [-0.0061, 0.0027] |  | no |
| A | D − L | descriptive | 0.03851 | 0.04615 | [0.0087, 0.0812] | 9/14 | 0.01779 [0.0084, 0.0289] | 0.02072 [-0.0021, 0.0545] |  | no |
| A | P − L | descriptive | 0.02653 | 0.04951 | [0.0022, 0.0775] | 10/14 | 0.00057 [-0.0149, 0.0200] | 0.02596 [0.0060, 0.0626] |  | no |
| A | PD − D | descriptive | 0.01097 | 0.02039 | [0.0015, 0.0292] | 9/14 | 0.00293 [-0.0032, 0.0111] | 0.00804 [0.0017, 0.0200] |  | no |

No contrast has an interval entirely below zero (no fitting flag).

## Stop rules per variant (DEC-23, with DEC-16 beside it)

Each variant is a full run of the 16 checks: the 12 add-one-input contrasts and the 4 A-conditions against repeat-own-last. DEC-23's rule, which governs (DEC-28), gives each check an exact one-sided sign-flip test (alternative: mean below zero) on its day sums, with Holm's correction across the 16, and fires when an adjusted p is below 0.05 (`logistic_v5.dec23_stop_rules`). DEC-16's rule, reported beside it, fires when a check's day-bootstrap interval lies entirely below zero. Each check's day sums are the variant's stored day means times the day's target count, as in the [P2] sweep.

| Variant | DEC-23: Holm-adjusted p < 0.05 | Smallest one-sided p, Holm-adjusted (check) | DEC-16: interval entirely below zero |
|---|---|---|---|
| R | no stop rule fires | 0.1195, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |
| L | no stop rule fires | 0.1186, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |
| P | no stop rule fires | 0.1394, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |
| D | no stop rule fires | 0.1334, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |
| PD | no stop rule fires | 0.1555, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |

## Reading (DEC-28, primary only)

Primary, pre-specified: LL(PD) - LL(P) in A_B_S: 0.01310 bits/coda (unweighted day mean 0.01214, 95% CI [0.0025, 0.0267], 10/14 days positive, two-sided sign-flip p = 0.0525); repeat part 0.01183 [0.0039, 0.0186], switch part 0.00127 [-0.0072, 0.0123].
The order of the whale's earlier switches carries information beyond its active repertoire (for example A→B→A alternation). Not grammar. It lives in the repeat part (interval above zero).
Secondary, pre-specified (numbers only; no reading): LL(PD) - LL(R) in A_B_S: 0.05170 bits/coda (unweighted day mean 0.06171, 95% CI [0.0234, 0.0986], 13/14 days positive, two-sided sign-flip p = 0.0084); repeat part 0.02004 [0.0091, 0.0335], switch part 0.03166 [0.0101, 0.0675].

## Limits

- The primary rests on few discriminating switches: only 58 switches have two or more other own types to choose between, so a null is weak.
- No calibration (planted signal) was run, so no size of order effect can be ruled out.
- Run after the results freeze (2026-10-04) as a final run; the descriptive numbers that motivated it were seen first (DEC-28). The features, variants, grid and rules were fixed before any fit.
- The class-feature weights share one history penalty with the lag one-hots, the partner block and the copy weights (v4's rule), so a null may partly reflect that compromise.
- Day-bootstrap intervals are approximate with only 14 days; the switch part rests on about 270 non-repeat codas and is exploratory.
- Claims are about dyadic exchanges only; nothing here is evidence of grammar or syntax.

Per-variant outputs are in `data/results/rq1_switch_structure/<variant>/` (summary.json with both stop-rule inputs and input SHA-256, l2_selection.jsonl, scored.jsonl for A and A_B_S, coefficients.jsonl), the descriptive table in `data/results/rq1_switch_structure/descriptive/`; `data/results/rq1_switch_structure/summary.json` records SHA-256 of every input to this report.
