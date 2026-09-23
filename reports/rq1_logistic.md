# RQ1 logistic-regression reference results

Fourteen leave-one-day-out folds, fixed settings, no test-driven tuning. These reference results precede the neural study.
L2=0.01 on mean cross-entropy plus L2/2 times squared coefficients, including intercepts. Full fold vocabulary is modeled, including classes absent from training targets. No synthetic training rows or probability clipping.
Last three codas are one-hot encoded by lag; missing lags are zero vectors within enabled streams. Timing uses raw and log1p values, each standardized on training comparison rows. Absent sources supply no features. Validation days remain excluded; fixed models use no tuning or early stopping.
Repeat-own-last uses the mapped training comparison repeat rate, with remaining mass distributed by the add-1 prior excluding the last label. Majority class has classification scores only.

| Condition | LL bits/coda | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| none | -2.60254 | 57.207% | 0.0364 |
| S | -2.54635 | 56.884% | 0.0404 |
| A | -1.36275 | 79.186% | 0.1821 |
| B | -2.42974 | 60.633% | 0.0594 |
| A_B | -1.38930 | 77.957% | 0.1677 |
| A_S | -1.38304 | 78.410% | 0.1733 |
| B_S | -2.43562 | 58.953% | 0.0580 |
| A_B_S | -1.40876 | 76.665% | 0.1596 |
| majority | — | 57.207% | 0.0364 |
| repeat_own_last | -1.39276 | 82.547% | 0.4070 |

| Contrast | Bits/coda | 95% day-bootstrap CI | Positive days | Repeat part | Switch-choice part |
|---|---:|---|---:|---:|---:|
| own_history | 1.02687 | [0.7595691447548991, 1.2685936176372359] | 14/14 | 0.93638 | 0.09049 |
| partner_content | -0.02572 | [-0.06994843317925525, 0.025903254830323966] | 8/14 | -0.03171 | 0.00600 |
| timing | -0.01946 | [-0.03489447671416844, -0.0021682331115086264] | 5/14 | -0.01775 | -0.00171 |

All 98 contextual fits converged. Negative add-input comparisons: ['add_input:A_B-A', 'add_input:A_S-A', 'add_input:B_S-B', 'add_input:A_B_S-A_B', 'add_input:A_B_S-A_S'].
Own-history conditions below repeat-own-last in mean LL: ['A_B_S'].
Negative contrasts and baseline failures must be investigated before substantive interpretation. Statistical uncertainty and restricted feature capacity also limit inference; this report does not establish grammar or causal mechanisms.
Raw repeats: 1277/1547; mapped repeats: 1277/1547. Decomposition uses mapped labels and all targets; switch results are exploratory.
Day-bootstrap intervals use 2,000 replicates and are approximate with 14 days. Full per-day values, all lattice comparisons and baseline comparisons are in evaluation/summary.json.
Pending: retrospective-delay sensitivity, neural models and their reproduction/robustness checks. No settings were changed after inspecting these test results.
