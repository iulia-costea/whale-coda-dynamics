# LSTM v2: pooling 3 validation days instead of 1 (experiment, negative result)

Branch: `experiment/lstm-v2-multi-validation-days`, off `feat/rq1-lstm-v2`.
Script: `lstm_v2_multi_validation_days.py`.

## Why

Ardjano approved trying a change to the validation design (2026-09-30) after
Iulia's stage-3 run confirmed the "adding B hurts" failure at scale, our
B-encoder investigation reproduced it on individual folds, and
`lstm_v2_3seed_selection_notes.md` ruled out 3-seed-averaged selection as a
fix. The next cheapest thing to try, before touching `folds.py` or DEC-2's
committed manifest: does hyperparameter selection on a bigger, pooled
validation set (3 days instead of the official single next-day-in-ring day)
give a choice that generalizes better to the test day?

## Method

For each fold, validation is the official validation day plus the next two
days in the same seeded `SHA256(seed:day)` ring `day_folds()` already uses
(k=3, a judgment call — see the script docstring). Those extra days come out
of what was training, shrinking the training set. `prepare_fold`,
`select_lstm_grid_and_fit` and `predict_v2` are reused unchanged; only which
days are labelled `"validation"` in the fold dict passed to `prepare_fold`
differs from the committed manifest. Nothing in `folds.py`, `data/`, or the
official fold manifest is touched.

Same two folds and grids as the 3-seed experiment, for direct comparison.

## Result

| Fold | Condition | wd_history | Validation LL | Test LL |
|---|---|---:|---:|---:|
| 2015-03-02 (1 val. day, prior experiment) | A | 0.0001 | -0.66494 | -1.53344 |
| 2015-03-02 (1 val. day, prior experiment) | A_B | 0.00001 | -0.61821 | -1.76756 |
| 2015-03-02 (3 val. days) | A | 0.001 | -0.98604 | -1.72147 |
| 2015-03-02 (3 val. days) | A_B | 0.001 | -1.03855 | -2.31065 |
| 2015-03-26 (1 val. day, prior experiment) | A | 0.001 | -2.19160 | -0.85502 |
| 2015-03-26 (1 val. day, prior experiment) | A_B | 0.01 | -2.26257 | -0.87939 |
| 2015-03-26 (3 val. days) | A | 0.001 | -1.01658 | -0.89249 |
| 2015-03-26 (3 val. days) | A_B | 0.001 | -1.02615 | -0.93038 |

A_B minus A on the test day:

| Fold | 1 validation day | 3 validation days |
|---|---:|---:|
| 2015-03-02 | -0.234 | -0.589 |
| 2015-03-26 | -0.024 | -0.038 |

## Reading

Pooling more validation days does not close the A_B gap — on 2015-03-02 it
roughly doubles it, on 2015-03-26 it's flat. Overall test log-likelihood also
got worse for both conditions on both folds versus 1-day validation, likely
because the training set shrank by 2 days to make room for the extra
validation days, and the pooled 3-day validation set is not obviously more
representative of the single held-out test day than the original 1 day was.

The grid search also picked the same `wd_history=0.001` for both A and A_B
in three of four rows here, versus different values per condition/fold with
1 validation day — pooling more (heterogeneous) days into the validation set
seems to push selection toward the most conservative setting in the grid
regardless of condition, rather than letting each condition find its own
best point. That is the opposite of what DEC-20's per-block penalty split is
for.

This rules out "just use more validation days, holding the same k-fold-like
mechanics" as a fix, at least at k=3 pooled this way. It does not rule out
genuine k-fold selection that keeps the training-day count fixed (rotate the
validation day across several choices, refit at each, and average the
*selection decision*, not the training set) — that's a different, more
expensive mechanism than this script tests, and like any change to
`folds.py`'s cross-validation design touches DEC-2, so it needs a
`DECISIONS.md` entry and Ardjano's sign-off on the specific mechanism before
it's worth spending Snellius time on.
