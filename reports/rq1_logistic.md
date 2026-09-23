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
| own_history | 1.02687 | [0.7595691447548023, 1.2685936176375194] | 14/14 | 0.93638 | 0.09049 |
| partner_content | -0.02572 | [-0.06994843318063523, 0.025903254830265992] | 8/14 | -0.03171 | 0.00600 |
| timing | -0.01946 | [-0.03489447671420885, -0.002168233111898324] | 5/14 | -0.01775 | -0.00171 |

All 98 contextual fits converged. Negative add-input comparisons: ['add_input:A_B-A', 'add_input:A_S-A', 'add_input:B_S-B', 'add_input:A_B_S-A_B', 'add_input:A_B_S-A_S'].
Own-history conditions below repeat-own-last in mean LL: ['A_B_S'].
Raw repeats: 1277/1547; mapped repeats: 1277/1547. Decomposition uses mapped labels and all targets; switch results are exploratory.
Day-bootstrap intervals use 2,000 replicates and are approximate with 14 days. Full per-day values, all lattice comparisons and baseline comparisons are in evaluation/summary.json.

## Interpreting the negative comparisons

Twelve condition pairs differ by exactly one input source, giving twelve direct checks that adding information does not lower held out performance. Five of the twelve pairs had a negative mean contrast. Repeating the day bootstrap for each of the five shows that three are not distinguishable from zero (adding B to A, adding S to B, adding B to A and S), so these are consistent with the added input contributing nothing rather than causing harm.
Two pairs are reliably negative, meaning the full interval lies below zero: adding S to A (-0.02029 bits per coda, -0.03362 to -0.00642) and adding S to A and B (-0.01946 bits per coda, -0.03489 to -0.00217).
Both reliable cases add the interactional state S to a condition that already contains own history A. The size of this loss is small, about 61 times smaller than the 1.240 bits per coda gained from own history alone. All eight conditions share one L2 penalty of 0.01 rather than a penalty tuned per condition, and S contributes only a handful of turn category indicators and two timing values on top of a much larger set of history features when A is present. Adding a small number of weak features to an already large feature set under one fixed penalty is a plausible source of a small, consistent loss on held out data. This is a property of the fixed penalty reference model, not evidence that timing information is harmful, and the same check should be repeated once the neural model, which fits its S encoder separately, is available.
No change was made to the model, the features, or the penalty after these results were observed. Every condition keeps the same architecture and hyperparameters so the contrasts stay comparable; adjusting the setup only for the conditions that performed worse would mean choosing settings after seeing the outcome.
The full condition (A, B and S) also fell slightly below the repeat own last baseline in mean log likelihood (-0.01600 bits per coda), but the day bootstrap interval, -0.10463 to 0.12277, comfortably includes zero, and 9 of 14 days favoured the full condition. Falling reliably below this baseline would indicate a failed fit; an interval that includes zero indicates the two are indistinguishable, which is a finding rather than a failure. The same pattern, an interval spanning zero, holds for A, for A and B, and for A and S considered on their own.
The repeat and switch decomposition of the own history contrast supports the same reading. Of the 1.027 bits per coda gained by adding own history, 0.936 bits (91 percent) come from predicting whether the whale repeats its previous coda, and 0.090 bits (9 percent) come from predicting what it produces when it does not repeat. At the level of this reference model, most of the value of own history reflects repetition rather than new sequential structure, consistent with the persistence account this study is designed to test.

Statistical uncertainty and restricted feature capacity limit inference from this reference model; this report does not establish grammar or causal mechanisms.
Pending: retrospective-delay sensitivity, neural models and their reproduction/robustness checks.
