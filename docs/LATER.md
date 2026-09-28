# Things to think about later

Deferred to keep stage 1 simple. Nothing here is required before the first logistic-regression results exist. Each item says when it becomes relevant and why it was raised.

## Inference (revisit once first contrasts exist)
- **Exact day-level sign-flip test.**
  - Method: randomly flip the sign of each held-out day's summed contrast (2¹⁴ = 16,384 flips) and compare the observed mean to that distribution.
  - Why: valid with 14 clusters, whereas bootstrap CIs with few clusters are too narrow.
- **Leave-one-day-out jackknife:** the range of the estimate with each day dropped.

## Model selection (now done per DEC-16/17; options for later)
- **Validation data:** one held-out training day per fold, or two pooled? Some validation days have only 16–30 targets, so selection can be noisy there. Pooling two days leaves 11 of 14 days for fitting. Decide from the spread of selected settings.
- **Tied output embeddings** as an alternative copy mechanism for the LSTM (output layer shares weights with the input embeddings); a sensitivity check next to the copy weights.

## Count-based models
- **Kneser-Ney n-gram / VLMC on (iii)/(iv).** Useful for direct comparison with WhaleLM and Assayag et al.
- **Limitation:** both condition on one symbol sequence, so they are undefined for the two-stream conditions (v), (vii), (viii) and for any condition with continuous timing.
- **RQ3:** they remain the intended within-coda (dICI) models there.

## Already in PLAN but only relevant later
- **RQ3 (PLAN §8):**
  - The sympatric vs allopatric comparison is confounded with one-pipeline vs many-pipeline data. It is stated as a difference-in-differences with identity codas, with the stated assumption that pipeline effects hit identity and non-identity codas equally.
  - dICI bins are fitted on the training clan only.
- **RQ2 (PLAN `[R2-6]`, `DEC-15`):**
  - UNK rates are reported per labelling.
  - A tempo fallback from `Duration` applies only if the tempo pickle can't be fetched.
