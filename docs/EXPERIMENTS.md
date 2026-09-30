# Experiments tried (and ruled out)

Log of things tried outside the reported pipeline, so nobody re-attempts
something already ruled out. Not for adopted decisions (see
`docs/DECISIONS.md`) or ideas nobody has tried yet (see `docs/LATER.md`).

Each entry: what was tried, why, where the code/write-up lives, the result,
and the takeaway.

## LSTM v2 gate-2 failure: B hurts instead of helping (2026-09-30)

Context: the LSTM v2 pilot (`reports/rq1_lstm_v2_pilot.md`) found `A_B_S`
reliably below logistic regression on 4 folds. A full run confirmed and
sharpened this.

**Iulia's stage-3 run.** Full run: 14 folds x 8 conditions x 3 seeds, on her
own snapshot of the code. Confirmed the pattern at scale with significant
day-cluster CIs: LSTM below logistic v4 for `A, B, A_B, A_S, A_B_S`; `A_B`
below `A` and `A_B_S` below `A_S` (adding B specifically hurts). Notes not
committed to this repo (shared privately) — ask Iulia for
`stage3_results_notes.md` before rerunning; it's about 5 hours on Snellius.

**3-seed-averaged hyperparameter selection.**
Branch `experiment/lstm-v2-3seed-selection`,
`experiments/lstm_v2_3seed_selection_notes.md`.
- **Hypothesis:** selection on 1 seed is noisy; averaging validation LL over
  3 seeds picks a setting that generalizes better.
- **Result:** negative. No change to the `A_B` vs `A` test-day gap on either
  pilot fold tested.

**Pooled 3-day validation set.**
Branch `experiment/lstm-v2-multi-validation-days`,
`experiments/lstm_v2_multi_validation_days_notes.md`.
Closes out the `docs/LATER.md` "one validation day or two pooled?" item.
- **Hypothesis:** a single validation day is too small/noisy for
  hyperparameter selection at this model's capacity; pooling more days into
  validation fixes it.
- **Result:** negative. The gap didn't close (got worse on one fold), and
  overall test LL dropped on both folds, likely from the smaller resulting
  training set.

**Takeaway.** Both ruled-out fixes target noise in hyperparameter selection.
That the failure survives day-cluster bootstrapping over 14 folds already,
and shows up even for `A` alone in the stage-3 run (no B encoder involved),
points to a systematic fitting problem rather than noisy selection. Real
k-fold selection (rotate the validation day, average the choice, keep full
training data) targets the same kind of noise and is expected to have the
same limitation — not tried; it would also need a `DECISIONS.md` entry
first since it changes DEC-2's fold design.

**Open.** Root cause unresolved. Current best guess is optimization/
architecture (e.g. how the copy weights interact with the B encoder), not
validation design. Next step is Ardjano's call.

*For contrast: logistic regression had a comparable "one mechanism doing two
jobs" problem (a single L2 penalty couldn't let the model ignore a weak $S$
while still fitting history well) and DEC-20's split history/S penalty fixed
it cleanly (`reports/rq1_logistic_v4.md`). That it didn't fix the LSTM's gap
here is more evidence the LSTM's problem isn't the same kind as logistic's
was.*
