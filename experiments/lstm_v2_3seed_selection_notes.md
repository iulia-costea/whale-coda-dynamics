# Experiment: does 3-seed-averaged selection fix A_B's LSTM v2 test-day gap?

Status: negative result. Not a fix; recorded so nobody retries this same idea from scratch.

## Background

The pilot (`reports/rq1_lstm_v2_pilot.md`) selected hyperparameters using validation
log-likelihood from a single seed (42), even though the plan's design (DEC-17) calls
for averaging over 3 seeds at full scale. Iulia's independent full stage-3 run
(`stage3_results_notes.md`, not in this repo) found `A_B` reliably below both `A` and
logistic v4 across all 14 test days. Before proposing anything bigger (e.g. multiple
validation days per fold), the cheapest hypothesis to rule out was: is this just
selection noise from using 1 seed instead of 3?

## Method

`select_lstm_grid_and_fit` already supports multi-seed selection, so no new code was
needed. Ran it with `seeds=[42, 43, 44]` for conditions `A` and `A_B` on two pilot
folds (`2015-03-02`, the largest, and `2015-03-26`), varying `weight_decay_history`
over a small grid centred on each fold's pilot-chosen setting for `A_B_S`, dropout and
hidden size fixed at that same setting. See `lstm_v2_3seed_selection.py`.

## Result

| Fold | Condition | Chosen wd_history | Validation LL | Test LL |
|---|---|---:|---:|---:|
| 2015-03-02 | A | 0.0001 | -0.66494 | -1.53344 |
| 2015-03-02 | A_B | 0.00001 | -0.61821 | -1.76756 |
| 2015-03-26 | A | 0.001 | -2.19160 | -0.85502 |
| 2015-03-26 | A_B | 0.01 | -2.26257 | -0.87939 |

Same pattern as with 1-seed selection, essentially unchanged:
- **2015-03-02**: `A_B` still looks better than `A` on validation but loses on test
  (a validation/test generalisation gap, not a seed-noise artifact).
- **2015-03-26**: `A_B` is worse than `A` on both validation and test.

## Interpretation

3-seed averaging does not close the gap on either fold. This rules out "the pilot's
1-seed selection was just unlucky" as the explanation. The validation/test gap on
2015-03-02 in particular suggests the model has enough capacity (a whole extra LSTM
encoder for B) to fit one validation day's idiosyncrasies regardless of how many
seeds that day's selection is averaged over; averaging seeds reduces training-run
variance, not validation-day sampling variance, which is a different source of
noise. A more structural change to the validation procedure (more validation days
per fold, or k-fold selection within training days) is the next thing worth asking
Ardjano about, not implementing unilaterally, since it changes the leave-one-day-out
design (DEC-2) this project is built on.
