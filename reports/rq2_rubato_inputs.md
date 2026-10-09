# RQ2 rubato inputs (PLAN [R2-9], DEC-29)

Logistic v4 unchanged on inventories D (Sharma rhythm × tempo, KDE cuts), A (guy_coda_type, RQ1), E′ (guy_coda_type × tempo, tempos-dict.p cuts): each inventory's committed v4 configuration (targets, folds, UNK policy, S, copy weights, the L2 grid and selection rule, the joint history and S penalties, convergence handling), plus the previous codas' durations as continuous inputs in the history blocks. Own: for lags 1–3 of A_history, the coda's log duration ln d (d = offset_s − onset_s), one column per lag, standardised on the column's non-missing training values and 0 when the lag is missing, with per-class weights like the lag one-hots, in the A block under the history penalty; no missing indicators (a missing lag already has all-zero one-hots). Partner: the same for lags 1–3 of B_history (the partner's codas completed before the target's onset, [D5]), in the B block. Durations come from `rq1_targets.jsonl` by history coda ID; every referenced coda ends at or before the target's onset.

Variants (duration inputs): R: none (logistic v4); DurA: A_log_duration; Dur: A_log_duration, B_log_duration. All fitted in one environment (Python 3.10.21, NumPy 2.2.6, SciPy 1.15.3, one BLAS thread per fitting process). DurA fits the conditions with A and Dur those with A or B; the other conditions reuse R's fits, after a refit on fold test-2014-04-16 gave identical penalty searches, models and test predictions. Dur's A and A_S have DurA's inputs; both were fitted and agree exactly on every fold. Raw log-likelihoods are not compared across inventories (PLAN [R2-3]); every contrast pairs two fits of one inventory.

Contrasts are paired per coda, with the [P1] repeat and switch parts; intervals are day-bootstrap (10,000 replicates, seed 42); the unweighted day mean is the mean of the 14 day-level contrasts.

## Data checks (no models)

Checked before any model against the expected values (computed independently before the run (DEC-29 prompt, 2026-10-09)): counts exactly, other statistics within 0.0005 (3 decimals); 13 of 13 agree.

Inventory files (data/processed): every inventory has the same rows as `rq1_targets.jsonl` in the same order, with the same coda_id, onset_s, offset_s, comparison_eligible, A_history_ids and B_history_ids; only label fields differ. Fold manifests: the same folds, days and comparison IDs.

| Check | Computed | Expected | Result |
|---|---|---|---|
| Inventory D: targets file against rq1_targets.jsonl | rows 2075; mismatched_rows 0; other_fields_differing []; label_fields_differing A_history / B_history / Ct / Ct_raw / inventory | rows 2075; mismatched_rows 0; other_fields_differing [] | agrees |
| Inventory D: fold manifest against inventory A's | identical True | identical True | agrees |
| Inventory E′: targets file against rq1_targets.jsonl | rows 2075; mismatched_rows 0; other_fields_differing []; label_fields_differing A_history / B_history / Ct / Ct_raw / inventory | rows 2075; mismatched_rows 0; other_fields_differing [] | agrees |
| Inventory E′: fold manifest against inventory A's | identical True | identical True | agrees |
| Rows in rq1_targets.jsonl | count 2075 | count 2075 | agrees |
| Comparison targets | count 1547 | count 1547 | agrees |
| References in the last 3 lags of A_history_ids and B_history_ids (comparison targets) | count 7914; unresolved 0; ending_after_target_onset 0 | count 7914; unresolved 0; ending_after_target_onset 0 | agrees |
| max abs(offset_s − onset_s − raw Duration) (all rows) | max_abs_below 3.456e-12 | max_abs_below 1e-09 | agrees |
| Targets with own lag 1 / 2 / 3 | counts 1547 / 1324 / 1088 | counts 1547 / 1324 / 1088 | agrees |
| Targets with partner lag 1 / 2 / 3 | counts 1547 / 1324 / 1084 | counts 1547 / 1324 / 1084 | agrees |
| Target ln d: mean; population SD | mean -0.1795; sd 0.5405 | mean -0.1795; sd 0.5405 | agrees |
| Pearson r of target ln d with own lag-1 ln d; with partner lag-1 ln d | own 0.9402; partner 0.7718 | own 0.9400; partner 0.7720 | agrees |
| The same, each variable centred on its mean over the 1,547 rows with its coda's label (Ct) | own 0.5678; partner 0.3691 | own 0.5680; partner 0.3690 | agrees |

## Gates: R here against each committed logistic v4 run

DEC-25's amended gate: chosen penalties identical (at most one differing) and every condition's mean log-likelihood within 1e-5 bits/coda of v4's; predicted probabilities are reported, not enforced (optimizer tolerance and BLAS threads move them by up to about 1e-4 at the smallest penalties).

| Inventory | Fits compared | Penalties differing | Max abs LL difference (bits/coda) | Max abs probability difference | Same penalty: max abs probability difference | Fits above 1e-5 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| D | 98 | 0 | 0 | 0 | 0 | 0 | passed |
| A | 98 | 0 | 2.4e-06 | 0.000113 | 0.000113 | 17 | passed |
| E′ | 98 | 0 | 0 | 0 | 0 | 0 | passed |

Penalty candidates that failed the convergence check (left out of the selection where the inventory's v4 configuration skips them, DEC-22): D R 0; D DurA 0; D Dur 0; A R 0; A DurA 0; A Dur 0; E′ R 0; E′ DurA 0; E′ Dur 0.

Implementation choices, fixed before any fit: in a condition's field list each duration field follows its history field; a duration column is standardised with the mean and population SD of its non-missing training values (Encoder's rule for timing; SD 0 becomes 1); the reuse check refits every unchanged condition on the first manifest fold (test-2014-04-16) and requires identical penalty searches, models and test predictions; Dur's A and A_S, whose inputs are DurA's, are fitted, not reused, and must equal DurA's on every fold; 'detected' (for the Reading) means an interval entirely above zero and p < 0.05 (two-sided for the primary, Holm-adjusted for the secondaries), DEC-30's and DEC-31's rule, since DEC-29 does not define one; DEC-29's DurA − R bullet is applied descriptively (interval entirely above zero, no test), since DEC-29 gives it no test; the fitting flag is checked for DurA − R, Dur − DurA and Dur − R in every condition where the two variants differ; data checks: counts exactly, other statistics within 0.0005.

Before any fit with this configuration: the data checks were computed (a scratch computation, then the checks stage, their intended first step) and reproduced every expected value; to time the run, one fold of inventory D was fitted under Dur on its training and validation days only (conditions A and A_B_S, two penalty values; only the time and iteration counts were looked at); unit tests ran on synthetic rows, and variant R (logistic v4 itself) on one fold against the committed v4 run; the smoke test (one L2 value, 0.01, not this grid; 50 bootstrap replicates) produced reports on tiny settings that were checked by assertions and viewed only with every digit masked, to check the layout; that view still showed the signs of the smoke contrasts and which were flagged (one penalty value, so not this run's fits). No input, variant, inventory, grid or rule changed after any of them.

## Log-likelihood and accuracy

Held-out log-likelihood (bits/coda; comparable within an inventory only), accuracy, macro-F1, and the difference to repeat-own-last with its day-bootstrap CI, for A and A,B,S. Every condition's numbers are in `summary.json` (`metrics`).

| Inventory | Variant | A LL | A accuracy | A macro-F1 | A − repeat-own-last [CI] | A,B,S LL | A,B,S accuracy | A,B,S macro-F1 | A,B,S − repeat-own-last [CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D | R | -1.69981 | 73.562% | 0.2854 | 0.42377 [0.3526, 0.5332] | -1.73282 | 72.980% | 0.2652 | 0.39077 [0.2891, 0.5484] |
| D | DurA | -1.70710 | 72.915% | 0.2853 | 0.41648 [0.3029, 0.5463] | -1.72943 | 71.946% | 0.2447 | 0.39415 [0.2767, 0.5608] |
| D | Dur | -1.70710 | 72.915% | 0.2853 | 0.41648 [0.3029, 0.5463] | -1.77655 | 70.976% | 0.2384 | 0.34703 [0.2027, 0.5305] |
| A | R | -1.09690 | 82.353% | 0.2819 | 0.29586 [0.2420, 0.3719] | -1.13261 | 82.288% | 0.3248 | 0.26015 [0.1855, 0.4047] |
| A | DurA | -1.08174 | 82.288% | 0.2733 | 0.31102 [0.2427, 0.4051] | -1.12909 | 81.319% | 0.2951 | 0.26367 [0.1947, 0.3983] |
| A | Dur | -1.08174 | 82.288% | 0.2733 | 0.31102 [0.2427, 0.4051] | -1.13897 | 81.125% | 0.2781 | 0.25378 [0.1917, 0.3718] |
| E′ | R | -1.53485 | 79.444% | 0.2967 | 0.34399 [0.2847, 0.4390] | -1.58067 | 76.018% | 0.2484 | 0.29817 [0.1805, 0.4819] |
| E′ | DurA | -1.54611 | 77.376% | 0.2501 | 0.33273 [0.2225, 0.4547] | -1.58008 | 76.277% | 0.2451 | 0.29876 [0.1685, 0.4968] |
| E′ | Dur | -1.54611 | 77.376% | 0.2501 | 0.33273 [0.2225, 0.4547] | -1.57992 | 75.889% | 0.2332 | 0.29892 [0.1785, 0.4788] |

## Contrasts

Primary (pre-specified): the partner contrast LL(A,B,S) − LL(A,S) under Dur on inventory D, with the exact two-sided sign-flip p (DEC-23: all 2^14 sign assignments of the day sums, token-weighted mean as the statistic). Secondary (pre-specified), each with the two-sided sign-flip p and Holm's correction across the five: the partner contrast under Dur on A and E′; the partner's durations beyond everything else, LL_Dur(A,B,S) − LL_DurA(A,B,S), on A, D and E′. Detected (for the Reading): interval entirely above zero and p < 0.05 (two-sided for the primary, Holm-adjusted for the secondaries), the rule of DEC-30 and DEC-31.

| Role | Inventory | Contrast | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part [CI] | Switch part [CI] | Two-sided p | Holm p | Detected |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| **primary** | D | partner: LL_Dur(A_B_S) − LL_Dur(A_S) | -0.05535 | -0.02230 | [-0.1062, 0.0207] | 5/14 | -0.04919 [-0.0897, 0.0036] | -0.00616 [-0.0340, 0.0331] | 0.2133 | n/a | no |
| **secondary** | A | partner: LL_Dur(A_B_S) − LL_Dur(A_S) | -0.04610 | -0.02973 | [-0.0995, 0.0061] | 4/14 | -0.04626 [-0.0861, -0.0057] | 0.00015 [-0.0320, 0.0283] | 0.1884 | 0.7534 | no |
| **secondary** | E′ | partner: LL_Dur(A_B_S) − LL_Dur(A_S) | -0.03431 | 0.01329 | [-0.0821, 0.0364] | 6/14 | -0.04571 [-0.0860, 0.0137] | 0.01140 [-0.0007, 0.0284] | 0.4722 | 0.9443 | no |
| **secondary** | A | partner's durations: LL_Dur(A_B_S) − LL_DurA(A_B_S) | -0.00989 | -0.01972 | [-0.0337, 0.0025] | 8/14 | -0.00604 [-0.0172, 0.0059] | -0.00384 [-0.0320, 0.0123] | 0.2095 | 0.7534 | no |
| **secondary** | D | partner's durations: LL_Dur(A_B_S) − LL_DurA(A_B_S) | -0.04712 | -0.05360 | [-0.0852, -0.0174] | 6/14 | -0.02368 [-0.0474, 0.0028] | -0.02344 [-0.0566, -0.0018] | 0.0250 | 0.1251 | no |
| **secondary** | E′ | partner's durations: LL_Dur(A_B_S) − LL_DurA(A_B_S) | 0.00016 | -0.03095 | [-0.0622, 0.0461] | 5/14 | 0.00999 [-0.0277, 0.0433] | -0.00982 [-0.0360, 0.0047] | 0.9951 | 0.9951 | no |

Descriptive (no tests): DurA − R and Dur − R in A and A,B,S; the own and timing contrasts under Dur; the partner contrast under R.

| Inventory | Contrast | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part [CI] | Switch part [CI] |
|---|---|---:|---:|---|---:|---:|---:|
| D | DurA − R: LL_DurA(A) − LL_R(A) | -0.00728 | -0.00982 | [-0.0848, 0.0444] | 10/14 | -0.02190 [-0.0562, 0.0191] | 0.01462 [-0.0455, 0.0457] |
| D | DurA − R: LL_DurA(A_B_S) − LL_R(A_B_S) | 0.00338 | -0.01132 | [-0.0653, 0.0414] | 11/14 | -0.03170 [-0.0635, -0.0031] | 0.03508 [-0.0189, 0.0732] |
| D | Dur − R: LL_Dur(A) − LL_R(A) | -0.00728 | -0.00982 | [-0.0848, 0.0444] | 10/14 | -0.02190 [-0.0562, 0.0191] | 0.01462 [-0.0455, 0.0457] |
| D | Dur − R: LL_Dur(A_B_S) − LL_R(A_B_S) | -0.04374 | -0.06492 | [-0.1484, 0.0219] | 8/14 | -0.05538 [-0.1085, -0.0050] | 0.01164 [-0.0703, 0.0670] |
| D | own history, under Dur: LL_Dur(A_B_S) − LL_Dur(B_S) | 1.30162 | 1.30514 | [1.0946, 1.4834] | 14/14 | 1.15104 [0.9796, 1.3045] | 0.15057 [0.0850, 0.2117] |
| D | timing, under Dur: LL_Dur(A_B_S) − LL_Dur(A_B) | -0.01557 | -0.01640 | [-0.0458, 0.0002] | 7/14 | -0.00869 [-0.0238, 0.0003] | -0.00688 [-0.0237, 0.0013] |
| D | partner, under R: LL_R(A_B_S) − LL_R(A_S) | -0.03045 | 0.02203 | [-0.0819, 0.0509] | 6/14 | -0.02105 [-0.0496, 0.0349] | -0.00940 [-0.0396, 0.0308] |
| A | DurA − R: LL_DurA(A) − LL_R(A) | 0.01516 | 0.02491 | [-0.0224, 0.0497] | 10/14 | 0.00127 [-0.0180, 0.0275] | 0.01389 [-0.0122, 0.0332] |
| A | DurA − R: LL_DurA(A_B_S) − LL_R(A_B_S) | 0.00352 | 0.01123 | [-0.0328, 0.0345] | 8/14 | -0.01596 [-0.0340, -0.0022] | 0.01947 [-0.0077, 0.0461] |
| A | Dur − R: LL_Dur(A) − LL_R(A) | 0.01516 | 0.02491 | [-0.0224, 0.0497] | 10/14 | 0.00127 [-0.0180, 0.0275] | 0.01389 [-0.0122, 0.0332] |
| A | Dur − R: LL_Dur(A_B_S) − LL_R(A_B_S) | -0.00637 | -0.00848 | [-0.0548, 0.0287] | 7/14 | -0.02200 [-0.0475, 0.0009] | 0.01563 [-0.0220, 0.0466] |
| A | own history, under Dur: LL_Dur(A_B_S) − LL_Dur(B_S) | 1.28806 | 1.26284 | [0.8960, 1.6877] | 14/14 | 1.17556 [0.8160, 1.5399] | 0.11250 [0.0538, 0.1725] |
| A | timing, under Dur: LL_Dur(A_B_S) − LL_Dur(A_B) | -0.01295 | -0.00519 | [-0.0302, 0.0029] | 8/14 | -0.01186 [-0.0278, 0.0013] | -0.00109 [-0.0035, 0.0025] |
| A | partner, under R: LL_R(A_B_S) − LL_R(A_S) | -0.02566 | -0.00144 | [-0.0942, 0.0635] | 6/14 | -0.02081 [-0.0663, 0.0520] | -0.00485 [-0.0367, 0.0251] |
| E′ | DurA − R: LL_DurA(A) − LL_R(A) | -0.01125 | -0.02216 | [-0.0913, 0.0325] | 10/14 | -0.02040 [-0.0523, -0.0017] | 0.00915 [-0.0420, 0.0433] |
| E′ | DurA − R: LL_DurA(A_B_S) − LL_R(A_B_S) | 0.00059 | 0.01061 | [-0.0448, 0.0434] | 9/14 | -0.00999 [-0.0315, 0.0142] | 0.01058 [-0.0284, 0.0399] |
| E′ | Dur − R: LL_Dur(A) − LL_R(A) | -0.01125 | -0.02216 | [-0.0913, 0.0325] | 10/14 | -0.02040 [-0.0523, -0.0017] | 0.00915 [-0.0420, 0.0433] |
| E′ | Dur − R: LL_Dur(A_B_S) − LL_R(A_B_S) | 0.00075 | -0.02034 | [-0.1000, 0.0633] | 8/14 | -0.00000 [-0.0387, 0.0247] | 0.00075 [-0.0625, 0.0413] |
| E′ | own history, under Dur: LL_Dur(A_B_S) − LL_Dur(B_S) | 1.46308 | 1.45742 | [1.1798, 1.7294] | 14/14 | 1.32162 [1.0750, 1.5463] | 0.14146 [0.0838, 0.1977] |
| E′ | timing, under Dur: LL_Dur(A_B_S) − LL_Dur(A_B) | -0.01137 | -0.01030 | [-0.0308, 0.0015] | 9/14 | -0.00634 [-0.0186, -0.0001] | -0.00502 [-0.0128, 0.0020] |
| E′ | partner, under R: LL_R(A_B_S) − LL_R(A_S) | -0.04190 | 0.02039 | [-0.1165, 0.0545] | 6/14 | -0.06687 [-0.1248, -0.0006] | 0.02498 [0.0030, 0.0639] |

## The partner contrast under R and under Dur

LL(A,B,S) − LL(A,S) within each variant, and its per-coda change from R to Dur, [LL_Dur(A,B,S) − LL_Dur(A,S)] − [LL_R(A,B,S) − LL_R(A,S)], with its day-bootstrap CI (descriptive).

| Inventory | Under R [CI] | +days | Under Dur [CI] | +days | Change R → Dur | Change CI | Change +days |
|---|---:|---:|---:|---:|---:|---|---:|
| D | -0.03045 [-0.0819, 0.0509] | 6/14 | -0.05535 [-0.1062, 0.0207] | 5/14 | -0.02490 | [-0.0583, -0.0011] | 5/14 |
| A | -0.02566 [-0.0942, 0.0635] | 6/14 | -0.04610 [-0.0995, 0.0061] | 4/14 | -0.02045 | [-0.0720, 0.0143] | 5/14 |
| E′ | -0.04190 [-0.1165, 0.0545] | 6/14 | -0.03431 [-0.0821, 0.0364] | 6/14 | 0.00759 | [-0.0426, 0.0436] | 6/14 |

## The partner contrast under Dur by turn category (no tests)

Token-weighted means within each turn category of the target ([D5]); positive days among the days with targets in that category.

| Inventory | Turn category | Targets | Bits/coda | Repeat part | Switch part | Positive days |
|---|---|---:|---:|---:|---:|---:|
| D | self_continuation | 461 | -0.06286 | -0.05920 | -0.00366 | 7/14 |
| D | switch | 645 | -0.05556 | -0.05784 | 0.00228 | 6/14 |
| D | overlap | 441 | -0.04719 | -0.02607 | -0.02112 | 9/14 |
| A | self_continuation | 461 | -0.03707 | -0.04340 | 0.00633 | 5/14 |
| A | switch | 645 | -0.03907 | -0.04728 | 0.00821 | 7/14 |
| A | overlap | 441 | -0.06583 | -0.04775 | -0.01809 | 5/14 |
| E′ | self_continuation | 461 | -0.05072 | -0.04569 | -0.00504 | 7/14 |
| E′ | switch | 645 | -0.01641 | -0.05657 | 0.04016 | 5/14 |
| E′ | overlap | 441 | -0.04334 | -0.02984 | -0.01349 | 6/14 |

## Fitted duration weights

Min / median / max over the 14 folds of the L2 norm of the own (A_log_duration, lags 1–3 × classes) and partner (B_log_duration) duration weights at the chosen penalties, per standard deviation of the lag's training log durations. Per-fold weights are in each variant's `coefficients.jsonl`.

| Inventory | Variant | A: own | A,B,S: own | A,B,S: partner |
|---|---|---:|---:|---:|
| D | DurA | 1.643 / 5.062 / 10.147 | 2.167 / 3.342 / 8.770 | — |
| D | Dur | 1.643 / 5.062 / 10.147 | 1.214 / 2.993 / 8.610 | 0.953 / 2.271 / 6.947 |
| A | DurA | 1.140 / 3.395 / 6.475 | 1.472 / 3.277 / 6.238 | — |
| A | Dur | 1.140 / 3.395 / 6.475 | 0.945 / 2.369 / 6.122 | 0.586 / 1.621 / 4.759 |
| E′ | DurA | 1.475 / 4.958 / 9.420 | 1.736 / 3.957 / 8.488 | — |
| E′ | Dur | 1.475 / 4.958 / 9.420 | 1.093 / 3.008 / 8.218 | 0.873 / 2.269 / 6.786 |

## Stop rules (DEC-23, with DEC-16 beside it) and fitting flags

Each inventory and variant is a full run of the 16 checks: the 12 add-one-input contrasts and the 4 A-conditions against repeat-own-last. DEC-23's rule gives each check an exact one-sided sign-flip test (alternative: mean below zero) on its day sums, with Holm's correction across the 16, and fires when an adjusted p is below 0.05 (`logistic_v5.dec23_stop_rules`; the day sums are each run's stored day means times the day's target count, as in the [P2] sweep). DEC-16's rule, beside it, fires when a check's day-bootstrap interval lies entirely below zero.

| Inventory | Variant | DEC-23: Holm-adjusted p < 0.05 | Smallest one-sided p, Holm-adjusted (check) | DEC-16: interval entirely below zero |
|---|---|---|---|---|
| D | R | no stop rule fires | 0.0327, Holm 0.5234 (add_input:A_S-A) | add_input:A_S-A: contrast CI entirely below zero [-0.0053, -0.0002]; add_input:B_S-B: contrast CI entirely below zero [-0.0861, -0.0049] |
| D | DurA | no stop rule fires | 0.0333, Holm 0.5332 (add_input:A_S-A) | add_input:A_S-A: contrast CI entirely below zero [-0.0429, -0.0006]; add_input:B_S-B: contrast CI entirely below zero [-0.0861, -0.0049] |
| D | Dur | no stop rule fires | 0.0284, Holm 0.4551 (add_input:B_S-B) | add_input:A_S-A: contrast CI entirely below zero [-0.0429, -0.0006]; add_input:B_S-B: contrast CI entirely below zero [-0.1240, -0.0130] |
| A | R | no stop rule fires | 0.1195, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |
| A | DurA | no stop rule fires | 0.2162, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |
| A | Dur | no stop rule fires | 0.0942, Holm 1.0000 (add_input:A_B_S-A_S) | no stop rule fires |
| E′ | R | no stop rule fires | 0.0627, Holm 1.0000 (add_input:A_S-A) | no stop rule fires |
| E′ | DurA | no stop rule fires | 0.1901, Holm 1.0000 (add_input:B_S-B) | no stop rule fires |
| E′ | Dur | no stop rule fires | 0.1613, Holm 1.0000 (add_input:A_B_S-A_B) | no stop rule fires |

Added-input contrasts (DEC-29's fitting flag: an interval entirely below zero is a fitting flag and is not interpreted), each in the conditions where the two variants differ:

| Inventory | Contrast | Bits/coda | 95% CI | Positive days | Fitting flag |
|---|---|---:|---|---:|---|
| D | LL_DurA(A) − LL_R(A) | -0.00728 | [-0.0848, 0.0444] | 10/14 | no |
| D | LL_DurA(A_B) − LL_R(A_B) | 0.00104 | [-0.0655, 0.0397] | 11/14 | no |
| D | LL_DurA(A_S) − LL_R(A_S) | -0.01884 | [-0.1040, 0.0395] | 10/14 | no |
| D | LL_DurA(A_B_S) − LL_R(A_B_S) | 0.00338 | [-0.0653, 0.0414] | 11/14 | no |
| D | LL_Dur(B) − LL_DurA(B) | 0.08875 | [-0.0925, 0.2026] | 9/14 | no |
| D | LL_Dur(A_B) − LL_DurA(A_B) | -0.03743 | [-0.0732, -0.0063] | 7/14 | yes |
| D | LL_Dur(B_S) − LL_DurA(B_S) | 0.06850 | [-0.1193, 0.1871] | 8/14 | no |
| D | LL_Dur(A_B_S) − LL_DurA(A_B_S) | -0.04712 | [-0.0852, -0.0174] | 6/14 | yes |
| D | LL_Dur(A) − LL_R(A) | -0.00728 | [-0.0848, 0.0444] | 10/14 | no |
| D | LL_Dur(B) − LL_R(B) | 0.08875 | [-0.0925, 0.2026] | 9/14 | no |
| D | LL_Dur(A_B) − LL_R(A_B) | -0.03639 | [-0.1335, 0.0266] | 8/14 | no |
| D | LL_Dur(A_S) − LL_R(A_S) | -0.01884 | [-0.1040, 0.0395] | 10/14 | no |
| D | LL_Dur(B_S) − LL_R(B_S) | 0.06850 | [-0.1193, 0.1871] | 8/14 | no |
| D | LL_Dur(A_B_S) − LL_R(A_B_S) | -0.04374 | [-0.1484, 0.0219] | 8/14 | no |
| A | LL_DurA(A) − LL_R(A) | 0.01516 | [-0.0224, 0.0497] | 10/14 | no |
| A | LL_DurA(A_B) − LL_R(A_B) | 0.00457 | [-0.0353, 0.0361] | 8/14 | no |
| A | LL_DurA(A_S) − LL_R(A_S) | 0.01408 | [-0.0233, 0.0480] | 10/14 | no |
| A | LL_DurA(A_B_S) − LL_R(A_B_S) | 0.00352 | [-0.0328, 0.0345] | 8/14 | no |
| A | LL_Dur(B) − LL_DurA(B) | -0.02865 | [-0.2469, 0.1284] | 9/14 | no |
| A | LL_Dur(A_B) − LL_DurA(A_B) | -0.01823 | [-0.0398, -0.0004] | 7/14 | yes |
| A | LL_Dur(B_S) − LL_DurA(B_S) | 0.04927 | [-0.1756, 0.1559] | 10/14 | no |
| A | LL_Dur(A_B_S) − LL_DurA(A_B_S) | -0.00989 | [-0.0337, 0.0025] | 8/14 | no |
| A | LL_Dur(A) − LL_R(A) | 0.01516 | [-0.0224, 0.0497] | 10/14 | no |
| A | LL_Dur(B) − LL_R(B) | -0.02865 | [-0.2469, 0.1284] | 9/14 | no |
| A | LL_Dur(A_B) − LL_R(A_B) | -0.01366 | [-0.0599, 0.0227] | 6/14 | no |
| A | LL_Dur(A_S) − LL_R(A_S) | 0.01408 | [-0.0233, 0.0480] | 10/14 | no |
| A | LL_Dur(B_S) − LL_R(B_S) | 0.04927 | [-0.1756, 0.1559] | 10/14 | no |
| A | LL_Dur(A_B_S) − LL_R(A_B_S) | -0.00637 | [-0.0548, 0.0287] | 7/14 | no |
| E′ | LL_DurA(A) − LL_R(A) | -0.01125 | [-0.0913, 0.0325] | 10/14 | no |
| E′ | LL_DurA(A_B) − LL_R(A_B) | 0.00052 | [-0.0447, 0.0422] | 9/14 | no |
| E′ | LL_DurA(A_S) − LL_R(A_S) | -0.00683 | [-0.0866, 0.0435] | 10/14 | no |
| E′ | LL_DurA(A_B_S) − LL_R(A_B_S) | 0.00059 | [-0.0448, 0.0434] | 9/14 | no |
| E′ | LL_Dur(B) − LL_DurA(B) | 0.03482 | [-0.2397, 0.2099] | 9/14 | no |
| E′ | LL_Dur(A_B) − LL_DurA(A_B) | 0.00829 | [-0.0491, 0.0517] | 6/14 | no |
| E′ | LL_Dur(B_S) − LL_DurA(B_S) | 0.01293 | [-0.2887, 0.1998] | 8/14 | no |
| E′ | LL_Dur(A_B_S) − LL_DurA(A_B_S) | 0.00016 | [-0.0622, 0.0461] | 5/14 | no |
| E′ | LL_Dur(A) − LL_R(A) | -0.01125 | [-0.0913, 0.0325] | 10/14 | no |
| E′ | LL_Dur(B) − LL_R(B) | 0.03482 | [-0.2397, 0.2099] | 9/14 | no |
| E′ | LL_Dur(A_B) − LL_R(A_B) | 0.00881 | [-0.0843, 0.0672] | 8/14 | no |
| E′ | LL_Dur(A_S) − LL_R(A_S) | -0.00683 | [-0.0866, 0.0435] | 10/14 | no |
| E′ | LL_Dur(B_S) − LL_R(B_S) | 0.01293 | [-0.2887, 0.1998] | 8/14 | no |
| E′ | LL_Dur(A_B_S) − LL_R(A_B_S) | 0.00075 | [-0.1000, 0.0633] | 8/14 | no |

Fitting flags: D: LL_Dur(A_B) − LL_DurA(A_B); D: LL_Dur(A_B_S) − LL_DurA(A_B_S); A: LL_Dur(A_B) − LL_DurA(A_B).

## Reading (DEC-29)

Chosen mechanically for the primary by rules fixed in the code before any fit (`rubato.reading_lines`): no reading when a gate fails, when DEC-23's stop rule fires on the primary's variants, or when an added-input contrast whose full side is one of the primary's two fits (or a fit with the same inputs) carries a fitting flag; otherwise detected (positive) or null as defined above.

- Primary, pre-specified: LL_Dur(A_B_S) − LL_Dur(A_S) on D: -0.05535 bits/coda (unweighted day mean -0.02230, 95% CI [-0.1062, 0.0207], 5/14 days positive, two-sided sign-flip p = 0.2133); repeat part -0.04919 [-0.0897, 0.0036], switch part -0.00616 [-0.0340, 0.0331].
- No reading: an added-input contrast on a fit the primary rests on has an interval entirely below zero (LL_Dur(A_B_S) − LL_DurA(A_B_S)), a fitting flag.
- Secondary, pre-specified (Holm across the five; numbers only, no reading): LL_Dur(A_B_S) − LL_Dur(A_S) on A: -0.04610 bits/coda (unweighted day mean -0.02973, 95% CI [-0.0995, 0.0061], 4/14 days positive, two-sided sign-flip p = 0.1884, Holm-adjusted p = 0.7534); repeat part -0.04626 [-0.0861, -0.0057], switch part 0.00015 [-0.0320, 0.0283]; LL_Dur(A_B_S) − LL_Dur(A_S) on E′: -0.03431 bits/coda (unweighted day mean 0.01329, 95% CI [-0.0821, 0.0364], 6/14 days positive, two-sided sign-flip p = 0.4722, Holm-adjusted p = 0.9443); repeat part -0.04571 [-0.0860, 0.0137], switch part 0.01140 [-0.0007, 0.0284]; LL_Dur(A_B_S) − LL_DurA(A_B_S) on A: -0.00989 bits/coda (unweighted day mean -0.01972, 95% CI [-0.0337, 0.0025], 8/14 days positive, two-sided sign-flip p = 0.2095, Holm-adjusted p = 0.7534); repeat part -0.00604 [-0.0172, 0.0059], switch part -0.00384 [-0.0320, 0.0123]; LL_Dur(A_B_S) − LL_DurA(A_B_S) on D: -0.04712 bits/coda (unweighted day mean -0.05360, 95% CI [-0.0852, -0.0174], 6/14 days positive, two-sided sign-flip p = 0.0250, Holm-adjusted p = 0.1251); repeat part -0.02368 [-0.0474, 0.0028], switch part -0.02344 [-0.0566, -0.0018]; LL_Dur(A_B_S) − LL_DurA(A_B_S) on E′: 0.00016 bits/coda (unweighted day mean -0.03095, 95% CI [-0.0622, 0.0461], 5/14 days positive, two-sided sign-flip p = 0.9951, Holm-adjusted p = 0.9951); repeat part 0.00999 [-0.0277, 0.0433], switch part -0.00982 [-0.0360, 0.0047].
- DurA − R (descriptive, no test or multiplicity control): no interval lies entirely above zero, so DEC-29's rubato reading does not apply (not detectable at this data size).

## Limits

- The partner's coda that overlaps the target is not complete at the target's onset, so it is not a B input ([D5]); duration matching within overlaps (Sharma et al. 2024a; Assayag et al. 2026) is outside this design.
- Added after the results freeze (2026-10-04) as a final run (DEC-29); the descriptive correlations that motivated it were seen first. The inputs, variants, grid and rules were fixed before any fit.
- No calibration (planted signal) was run, so a null says only "not detectable at this data size".
- The duration columns share one history penalty with the lag one-hots, the other history block and the copy weights (v4's rule), so a null may partly reflect that compromise.
- Day-bootstrap intervals are approximate with only 14 days; the switch part rests on the non-repeat codas and is exploratory.
- Claims are about dyadic exchanges only; nothing here is evidence of grammar or syntax.

Fit stage: 34.9 min wall time on 18 processes (R on all inventories 12.8 min; then D 9.6 min, A 3.7 min, E′ 8.8 min); after D, the estimate for the rest was 11.0 min (the first inventory's DurA and Dur wall time, scaled by R's CPU time per inventory). Outputs: `data/results/rq2_rubato_inputs/checks/summary.json`, `fit/summary.json`, per inventory and variant `fit/<inventory>/<variant>/` (summary.json, l2_selection.jsonl, scored.jsonl, coefficients.jsonl), and `data/results/rq2_rubato_inputs/summary.json`, which records the SHA-256 of every input to this report.
