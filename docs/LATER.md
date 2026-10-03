# Things to think about later

Deferred to keep stage 1 simple. Nothing here is required before the first logistic-regression results exist. Each item says when it becomes relevant and why it was raised.

## Inference (revisit once first contrasts exist)
- **Leave-one-day-out jackknife:** the range of the estimate with each day dropped.

## Model selection (now done per DEC-16/17; options for later)
- **Validation data:** one held-out training day per fold, or two pooled? Some validation days have only 16–30 targets, so selection can be noisy there. Pooling two days leaves 11 of 14 days for fitting. Decide from the spread of selected settings. *Tried (pooled 3 days, LSTM v2): no improvement, see `docs/EXPERIMENTS.md`.*
- **Tied output embeddings** as an alternative copy mechanism for the LSTM (output layer shares weights with the input embeddings); a sensitivity check next to the copy weights.

## Count-based models
- **Kneser-Ney n-gram / VLMC on (iii)/(iv).** Useful for direct comparison with WhaleLM and Assayag et al.
- **Limitation:** both condition on one symbol sequence, so they are undefined for the two-stream conditions (v), (vii), (viii) and for any condition with continuous timing.
- **RQ3:** they remain the intended within-coda (dICI) models there.

## Already in PLAN but only relevant later
- **RQ3 (PLAN §8):** the old difference-in-differences and clan-fitted bins were superseded by DEC-19; see "RQ3, deferred by DEC-19" below.
- **RQ2:** see "RQ2, deferred by DEC-22" below.

## RQ3, deferred by DEC-19
- **Pooled training across clans (the backup).**
  - One model on all clans with clan tags and size-balanced sampling (the mBART recipe, Liu et al. 2020, trained from scratch); leave-one-clan-out; with vs without the tag.
  - Compare within one model family: the pooled model against own-clan models of the same kind, with the count models as reference.
  - Uses `[R3-0]`–`[R3-3]` unchanged. Relevant if the lecturer asks for a standard multilingual experiment.
- **Size-matched reruns.** Cap every source clan's training data at the smallest clan's size and redraw it several times, to check whether large source clans transfer better only because they are large.
- **Within-type transfer** (Leitão et al. SI §2.4.2): train and score one coda type at a time across clans, which removes usage and coverage. Feasible for 1–17 shared types per Pacific pair.
- **The Caribbean ↔ Pacific difference-in-differences** (old DEC-14), as a secondary reading of the matrix, if EC1/EC2 identity codas get a source.

## RQ2, deferred by DEC-22
- **Segmentation proper.** A coda as two tokens (rhythm, then tempo), the subword analogue of DEC-22's combined labels; or interval symbols as tokens, as in RQ3's tokeniser `[R3-1]`. Needs a sequence model over sub-coda tokens with a per-coda score, so it is a different question from DEC-22's.
- **Ornaments** as a third dimension (`ornaments.p`, 151 of 3,840 `[D2]` codas). Sharma et al. find them at the start and end of call sequences and before changes in the partner's chorusing: a turn-taking question rather than a next-coda one.
- **Rubato:** the previous codas' durations as continuous inputs to logistic v4. Separates a gradual tempo trend from the tempo class, and is the first check if a model beats logistic v4 on inventory D or E.
- **The residual LSTM (DEC-21) on inventories D and E.**
- **Tempo classes fitted on other clans** (EC2, the Pacific): Sharma et al.'s cuts come from EC1 only, like a tokeniser trained on one language. Links RQ2 to RQ3.
- **Renaming `reports/rq2.md`** (the Sharma-pipeline preprocessing report) so it isn't read as RQ2 results; needs a change in `prepare_rq2.py`, which generates it.
