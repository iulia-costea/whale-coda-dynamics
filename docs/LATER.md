# Things to think about later

Deferred to keep stage 1 simple. Nothing here is required before the first logistic-regression results exist. Each item says when it becomes relevant and why it was raised.

## Inference (revisit once first contrasts exist)
- **Exact day-level sign-flip test.**
  - Method: randomly flip the sign of each held-out day's summed contrast (2¹⁴ = 16,384 flips) and compare the observed mean to that distribution.
  - Why: valid with 14 clusters, whereas bootstrap CIs with few clusters are too narrow.
- **Unweighted day mean** alongside the token-weighted one.
  - Why: day sizes are very unequal, so a few large days can decide the weighted result. A sign disagreement between the two is itself a finding.
- **Pre-specify which is primary** before looking at results. The token-weighted mean is the natural choice, since it is the per-token 𝒱-information estimand.
- **Leave-one-day-out jackknife:** the range of the estimate with each day dropped.

## Hyperparameter tuning (relevant from phase 4, or if fixed LR settings look poor)
- **Tuned settings can't be identical.** Once *k* or the L2 strength (LR), or early stopping (neural), is tuned, the selected values differ per condition.
- **Rule:** identical search space and identical selection rule on the early-stopping day for every condition; report the selected values.
- **Early-stopping data:** one held-out training day or two pooled? Decide from per-day token counts. Pooling two days leaves 11 of 14 days for fitting.

## Neural architecture (phase 4)
- **Template:** one encoder per source ($A$, $B$, $S$), concatenated into a shared output head. An absent source means its encoder is removed. Condition (i) stays the unigram.
- **WhaleLM:** its encoder–decoder LSTM predicts a single coda sequence, whereas these models classify one coda from up to three inputs, and some conditions have no sequence at all. Keep WhaleLM's *hyperparameters* (64-dim, Adam 1e-4, weight decay 1e-5, batch 32), not its architecture, and don't call the models "WhaleLM's configuration".
- **`[U1]`:** each neural model should not fall below logistic regression on the same condition by more than its seed spread. Falling below means it is undertrained.

## Count-based models
- **Kneser-Ney n-gram / VLMC on (iii)/(iv).** Useful for direct comparison with WhaleLM and Assayag et al.
- **Limitation:** both condition on one symbol sequence, so they are undefined for the two-stream conditions (v), (vii), (viii) and for any condition with continuous timing.
- **RQ3:** they remain the intended within-coda (dICI) models there.

## Already in PLAN but only relevant later
- **RQ3 (PLAN §7):**
  - The sympatric vs allopatric comparison is confounded with one-pipeline vs many-pipeline data. It is stated as a difference-in-differences with identity codas, with the stated assumption that pipeline effects hit identity and non-identity codas equally.
  - dICI bins are fitted on the training clan only.
- **RQ2 (PLAN `[R2-6]`, `DEC-15`):**
  - UNK rates are reported per labelling.
  - A tempo fallback from `Duration` applies only if the tempo pickle can't be fetched.
