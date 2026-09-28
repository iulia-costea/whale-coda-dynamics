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
- **RQ3 (PLAN §8):** the old difference-in-differences and clan-fitted bins were superseded by DEC-19; see "RQ3, deferred by DEC-19" below.
- **RQ2 (PLAN `[R2-6]`, `DEC-15`):**
  - UNK rates are reported per labelling.
  - A tempo fallback from `Duration` applies only if the tempo pickle can't be fetched.

## RQ3, deferred by DEC-19
- **Pooled training across clans (the backup).**
  - One model on all clans with clan tags and size-balanced sampling (the mBART recipe, Liu et al. 2020, trained from scratch); leave-one-clan-out; with vs without the tag.
  - Compare within one model family: the pooled model against own-clan models of the same kind, with the count models as reference.
  - Uses `[R3-0]`–`[R3-3]` unchanged. Relevant if the lecturer asks for a standard multilingual experiment.
- **Size-matched reruns.** Cap every source clan's training data at the smallest clan's size and redraw it several times, to check whether large source clans transfer better only because they are large.
- **Within-type transfer** (Leitão et al. SI §2.4.2): train and score one coda type at a time across clans, which removes usage and coverage. Feasible for 1–17 shared types per Pacific pair.
- **The Caribbean ↔ Pacific difference-in-differences** (old DEC-14), as a secondary reading of the matrix, if EC1/EC2 identity codas get a source.
