# Plan — Sequential structure vs interactional dynamics in sperm whale coda exchanges

What the project does. IDs in brackets (e.g. `[D5]`) are referenced by prompts and code.
- **Why** each choice was made, and how this differs from the proposal: `docs/DECISIONS.md`.
- **Deferred** ideas: `docs/LATER.md`.

**Status (2026-09-22):**
- **Done:** the preprocessing pipeline (RQ1: 1,547 targets; Sharma pipeline: 1,318), including the new $S$ (DEC-1).
- **In progress:** the rest of phase 1.
- **Not started:** no models exist yet.

See §12 for the order of work.

---

## 1. Question

Recent models find long-range structure in coda sequences (WhaleLM: dependence on up to eight codas; Sinha & Chavan 2026: second-order dependence). Terwilliger et al. (2025) argue this may be **persistence**, not composition: `a a b` could be a rule, or a whale repeating `a` until it gets a reply. Nobody has conditioned jointly on own history, partner history and timing.

- **RQ1.** Does a whale's own coda history still predict its next coda once the partner's codas and the interaction timing are known?
  - **Primary contrast:** LL(viii) − LL(vii).
- **RQ2.** Does the answer hold under a different coda labelling?
- **RQ3.** Do within-coda patterns transfer across clans, sympatric first and then allopatric?

**Method.** We train one model per combination of inputs. The difference in held-out log-likelihood between models estimates how much usable information an input adds: predictive 𝒱-information (Xu et al., 2020; O'Connor & Andreas, 2021). It is relative to model class and data size, and defined on the real data distribution, so training data is never filtered or reweighted.

**Scope.**
- Claims are about **two-whale (dyadic) exchanges** only.
- Structure that survives the persistence tests is "not persistence". It is not evidence of grammar, and we make no claims about meaning.

---

## 2. Data

| ID | Source | Size | Used for |
|---|---|---|---|
| `[D1]` | Assayag et al. 2026, `data/raw/dswp-2014-2016-retagged-maor-2025.csv` | 3,933 codas, EC1, 14 days (2014-04-16 to 2016-05-13) | RQ1; both RQ2 labellings |
| `[D2]` | Sharma et al. 2024a, `data/raw/sperm-whale-dialogues.csv` + `rhythms.p` | 3,840 codas, 219 recordings | secondary RQ2 labels; standalone "whole-pipeline" robustness check |
| `[D6]` | `data/raw/CarribeanCodas.csv` (probably Leitão et al.'s data) | 8,719 codas, clans EC1 + EC2, 101 dates | RQ3 sympatric test |
| `[D4]` | Hersh et al. 2022, `data/raw/PacificCodas.csv` (OSF `ae6pd`) | 24,237 codas; 23,429 with 3–10 clicks across 7 Pacific clans | RQ3 allopatric extension |

**Facts that matter**
- **Coda types `[D1]`:** two labellings exist. `guy_coda_type` has 23 types and `decision_tree_coda_type` has 25, and they disagree on 265 codas. 21 types appear in the RQ1 comparison set. Never write "27 types"; that figure is from Gubnitsky et al.
- **No global whale ID.**
  - `Whale` is a segment-local caller code in both files.
  - `IDN` in `[D1]` is 0 (unresolved) on 2,510 rows.
- **`pair_category` `[D1]`** is 62% empty. Never use it as a feature; turn category is derived instead.
  - Unlabelled codas are disproportionately non-focal: 66% vs 50%.
  - All 157 missing-`Focal` rows are unlabelled.
- **`[D1]` and `[D2]` overlap heavily.** 214/219 recordings are shared, and 2,769 codas match on (`REC`, `TsTo`, `Duration`). They are not independent corpora.
- **`[D2]` quirks:**
  - It keeps 153 codas with more than 10 clicks, which Sharma et al. excluded. Flag them, and report results with and without them.
  - `tempos-dict.p` and `ornaments.p` are not in the repo; fetch them from Sharma's GitHub.
  - One `Duration=0` row drops the whole 731-row recording `sw061b003` in the Sharma pipeline.
- **`[D6]` encoding:** `CarribeanCodas.csv` starts with a UTF-8 BOM, so read it with `utf-8-sig`.

**`[D7]` Sample.**
- **Segmentation:** exchanges are cut at a 7 s end-to-start gap (as in Assayag et al.; cite their 3/5/7/10 s sweep).
- **Dyads only:** only segments with exactly two callers are kept. This removes about 45% of `[D1]` codas.
- **Robustness:** `strict_focal_subset` (1,131 targets) checks the provisional caller identities.

---

## 3. Interactional state $S$ `[D5]`

**Rule:** $S$ may only contain what is known at the moment the target coda starts. It never contains any coda's duration or content.

| Field | Values / definition |
|---|---|
| `turn_category` | Relative to the immediately preceding coda (of either whale): `exchange_start` / `self_continuation` / `switch` / `overlap` (target starts before that coda ends) |
| `gap_s` | For `self_continuation` and `switch`: target onset − end of preceding coda. 0 otherwise |
| `overlap_elapsed_s` | For `overlap`: target onset − onset of preceding coda. 0 otherwise |

`turn_predecessor_id` records which coda was the preceding one.

**Old delay:** kept as `retrospective_delay_s`, a sensitivity field only. It is used in the `retrospective_signed_end_to_start` convention and is never part of the primary $S$.

**Result (implemented):**
- **RQ1 comparison targets:** 461 `self_continuation`, 645 `switch`, 441 `overlap`. Sharma pipeline: 388 / 557 / 373.
- **No `exchange_start`:** comparison targets need history from both whales, so none are `exchange_start`.
- **Overlaps:** they match the old overlap flag row for row. There are no same-whale overlaps, and none with a non-adjacent coda.
- **Gaps:** `gap_s` is at most 6.97 s. The old continuation delay reached 50 s.

---

## 4. RQ1 design

**`[M1]` Conditions** — one model per row, trained separately:

| ID | Inputs | Role |
|---|---|---|
| (i) | — | Smoothed unigram (training frequencies, add-1 smoothing) |
| (ii) | $S$ | Timing alone |
| (iii) | $A$ | Own history alone |
| (iv) | $B$ | Partner history alone |
| (v) | $A, B$ | Both histories, no timing |
| (vi) | $A, S$ | Own history + timing |
| (vii) | $B, S$ | Everything except own history |
| (viii) | $A, B, S$ | Full model |

$A$ = the whale's own preceding codas, $B$ = the partner's preceding codas, $S$ = interactional state.

**`[M1b]` Baselines**
- **Majority class:** reported for accuracy and macro-F1 only; it has no usable log-likelihood.
- **Repeat-own-last:**
  - It predicts the whale's last coda with probability equal to the training repeat rate; the rest is spread by frequency.
  - An $A$-model *below* it has failed to train. *Matching* it is a finding: own history carries nothing beyond repetition.

**`[M2]` Contrasts**

| Contrast | Question |
|---|---|
| LL(viii) − LL(vii) | **RQ1:** does own history add anything beyond partner and timing? |
| LL(viii) − LL(vi) | Does the partner's *content* add anything beyond timing? |
| LL(viii) − LL(v) | Does timing add anything beyond content? |

Also compare (v) with the better of (iii) and (iv), to see whether the two histories are redundant or complementary.

**`[M3]` Evaluation**
- **Separate models:** one trained model per condition. Never remove an input by masking it at test time.
- **Folds:** leave-one-day-out, 14 folds. In each fold, one training day is held aside for validation and early stopping.
- **Paired contrasts:** all conditions score the same test codas.
  - A contrast is the mean over test codas of the per-coda difference in $\log_2$ probability.
  - Neural models average over 3 seeds per coda.
- **Uncertainty:**
  - day-level bootstrap CI (resample the 14 days, never individual codas);
  - the number of days, out of 14, with a positive contrast.
  - Note in the write-up that 14 days give rough CIs.

**`[M4]` Models**
- **Main model: LSTM.**
  - Reported for all eight conditions.
  - One LSTM encoder per coda-history input ($A$, $B$) and a small feed-forward encoder for $S$, concatenated into a shared output layer. An absent input means its encoder is removed.
  - Settings as in the literature: 64-dim hidden, Adam (lr 1e-4, weight decay 1e-5), batch 32, early stopping on the validation day. They are fixed and identical across conditions.
  - For comparability with Assayag et al., take their exact LSTM spec from their paper and match it where it applies.
- **Reference and debug model: multinomial logistic regression, run first.**
  - Inputs: one-hot encodings of the last 3 codas of each available stream, the turn category, and the two timing values (standardised, plus `log1p`).
  - Fixed L2 penalty, the same for every condition.
  - Why run it first:
    - It can't fail to train, so it tests the evaluation pipeline end-to-end before any neural model.
    - It later serves as the reference the LSTM must not fall below (`[U1]`).
- **Reproduction check:** condition (iv) (partner history only) should roughly reproduce Assayag et al.'s reported cross-whale result. A large mismatch means we investigate before interpreting anything.
- **Later:**
  - a small Transformer with the same per-input structure;
  - Kneser-Ney/VLMC models (`docs/LATER.md`).

**`[M5]` Metrics**
- Report held-out log-likelihood **per coda, in bits**, never summed.
- Also report macro-F1.
- All metrics come from one shared evaluation module whose function names include the units.

**`[M6]` Unknown coda types**
- One shared module, rebuilt per fold.
- **Vocabulary:** types seen fewer than *k* = 2 times in the fold's training data become `UNK`, in training too, so models learn to give `UNK` some probability.
- **Unseen test types** also map to `UNK`.
- The unigram uses the same vocabulary.
- **Report:** `UNK` counts per fold.

---

## 5. Checks

| ID | Check | Why |
|---|---|---|
| `[V1]` | Can turn category be predicted from coda types alone? Report $H(S \mid C)/H(S)$ | If yes, conditioning on $S$ is nearly meaningless |
| `[V2]` | Does timing help predict the next coda given the previous one? | This is where persistence and composition diverge |
| `[V3]` | Are the information estimates stable across bout thresholds? (secondary) | Assayag et al. only checked correlations |
| `[U1]` | The LSTM must not do worse than logistic regression on the same condition (beyond seed spread) | Otherwise it is undertrained |
| `[U2]` | For the 12 "add one input" pairs, the larger model should score at least as well; count violations | Violations signal fitting problems |
| `[U3]` | Hidden size 32/64/128: contrasts should plateau | Capacity limits |
| `[U4]` | Retrain on 25–100% of training days; contrasts should stabilise | Data limits |
| `[U5]` | Variance across seeds | |
| `[U6]` | Train–test gap per condition | |

A negative contrast means a model failed to train, not a finding.

---

## 6. Persistence tests

**The repetition fact** (report it in the results): 82.5% of RQ1 targets repeat the whale's own previous coda. That leaves 270 non-repeats.

**`[P1]` Repeat/switch split.** Each model's log-likelihood splits exactly into two parts, with $a$ = the whale's own last coda:

$$\log p(C_t) = \underbrace{r\log p(a) + (1-r)\log(1-p(a))}_{\text{did it predict whether the whale repeats?}} + \underbrace{(1-r)\log\frac{p(C_t)}{1-p(a)}}_{\text{did it predict what it said when switching?}}$$

- **Application:** applied after training to every model's predictions, and every coda is used.
- **Stratum:** codas with no own history are left out of the split.
- **Power:** the switch part rests on ~270 codas, so it is exploratory.
- **Why not filter:** evaluating only on non-repeats would bias the RQ1 contrast.

**Other persistence tests**
- **`[P2]`:** own-history length 1–9 (the literature disagrees on memory: 8 codas, 2, or 1).
- **`[P3]`:** do timing and the partner predict *whether* the whale breaks a repeat? (the repeat part of `[P1]`)
- **`[P4]`:** pure persistence predicts the partner-content contrast ≈ 0.

---

## 7. RQ2 — labelling robustness

- **`[R2-1]` Primary:** run the full lattice with `guy_coda_type` and with `decision_tree_coda_type`, on identical codas and folds. Only the labels differ. This is a small perturbation (265 disagreements), so say so.
- **`[R2-1b]` Secondary:** relabel the `[D1]` codas that match `[D2]` with Sharma's rhythm × tempo types.
  - Only use exchanges where every coda matched, and compare against the `[D1]` labels on the same subset.
  - If the tempo file can't be fetched, reconstruct tempo from duration; see `DECISIONS.md` DEC-15.
  - Rhythm alone is too coarse to count as a second scheme.
- **`[R2-3]`:** report each contrast as a share of total predictability, [LL(viii) − LL(rest)] / [LL(viii) − LL(i)]. Never compare raw bits or perplexity across labellings.
- **`[R2-4]`:** compare the *pattern* (sign and relative size of contrasts) across labellings.
- **`[R2-5]` / `[R2-6]`:** the same unknown-type policy for both labellings, with `UNK` rates reported per labelling.
- **Sharma pipeline:** the standalone Sharma pipeline differs in too many ways (targets, segmentation, splits) to test labelling. It only checks robustness to the whole preprocessing chain.

---

## 8. RQ3 — cross-clan transfer

- **`[R3-0]` Input eligibility and grouping:** retain 3–10-click codas from the accepted clans only when every required ICI is in [0.005, 1.0) s. Preserve excluded rows and their reasons in the audit artifact.
  - Caribbean `Date` is parsed day-first into canonical ISO `recording_date`; folds must use this field, never the raw date string. A Caribbean group is one `recording_date`; a Pacific group is one `grpvar` repertoire (stored as `source_repertoire_id`).
  - Write this grouping explicitly as `group_id` in the processed RQ3 tables: `recording_date` for Caribbean codas and `grpvar`/`source_repertoire_id` for Pacific codas.
  - Construct deterministic, group-disjoint folds: five folds per clan, except six folds for EC2 (one recording day per fold). Sort groups by decreasing coda count, breaking ties with SHA-256 of `42:<clan>:<group_id>`, then assign each whole group to the currently smallest fold. A group must never span folds.
  - Save the resulting assignments and per-fold counts in `data/processed/rq3/rq3_folds.csv`.
- **`[R3-1]`:** model structure *within* codas, as sequences of discretised inter-click intervals (following Leitão et al.). This needs no speaker IDs.
  - Use the fixed, global `fixed-50ms-v1` tokenizer: 20 ICI symbols for 50 ms-wide bins over [0, 1.0) s, plus `BOS` and `EOS`.
  - Token boundaries are predefined and are never fitted on a source or target clan.
- **`[R3-2]` Transfer tests:**
  - first implement EC1 ↔ EC2 as the initial Caribbean comparison. This is a valid sympatric pair, but it is one asymmetric, relatively low-overlap case: EC2 is only rarely recorded in the Caribbean study area and the clans are socially segregated. Do not use this pair alone to support a general geographic-overlap claim;
  - then complete the directed 9 × 9 clan matrix: nine own-clan cells and 72 cross-clan cells, comprising EC1/EC2 ↔ Pacific and every Pacific ↔ Pacific direction.
  - Choose model family, order, smoothing, and every other model-setting using source-clan training/validation folds only. Target-clan codas are never used to select or tune a model.
  - Measure held-out base-2 log-likelihood on every target clan. Report both bits per coda and bits per predicted symbol (ICI symbols plus EOS); use bits per symbol as the primary cross-clan metric, since it does not mechanically give longer codas more weight.
  - Use one shared aggregation function for bits per coda, bits per symbol, paired transfer loss, coverage strata, and bootstrap draws. Do not duplicate or redefine these calculations across experiments.
  - For a source clan, train one otherwise-identical model per source fold, each excluding that fold. Score every target coda with every source-fold model and average those scores. This cross-validation ensemble avoids making a transfer result depend on one particular source-data subset.
  - Score every target coda with both its averaged transfer score and an own-clan score from a model that did not train on that coda's group. Calculate transfer loss only from these paired scores on the same token sequence.
  - Form an uncertainty interval (an error bar on the final estimate) with 1,000 bootstrap resamples of whole target groups (Caribbean recording days or Pacific repertoires), never individual codas. Recalculate the paired aggregate transfer loss for each draw. This happens after scoring: it neither retrains models nor changes the primary estimate, and can wait until the core matrix works.
  - Save coda-level paired scores and model/fold identifiers, and generate `reports/rq3_transfer.md` with the transfer matrix, coverage breakdown, and heatmap.
- **`[R3-3]` Hypothesis** (from Leitão et al.): transfer degrades less between sympatric clans, and only for non-identity codas.
  - Test it as the interaction (sympatric vs allopatric) × (identity vs non-identity). The Caribbean and Pacific data come from different recording programmes, and the identity contrast is what separates the clan effect from that.
  - For Pacific codas, derive `identity_status` from the clan-specific numeric coda-type mapping in Hersh et al. (2022), Supplementary Table S2: PALI = 48, 411, 412, 73, 711, 82, 94, 95, 102; FP = 611, 614; SI = 39, 413, 69; RI = 45, 715; SH = 313; PO = 515, 68, 613, 79, 712, 89; REG = 51, 512, 62, 71, 74, 81, 92, 96, 108. A listed numeric type is `identity`; every other numeric Pacific type is `non_identity`. Missing/non-numeric source types are `unmapped` and excluded from identity-stratified analyses. Caribbean codas remain unmapped until an authoritative equivalent mapping is available.
  - `identity_status` is an analysis label only, never a predictive input to the timing model.
  - Train the primary own-clan and transfer models on all eligible codas, irrespective of identity status. Test the interaction by partitioning the resulting paired target-coda scores into identity and non-identity strata; do not train separate identity-only or non-identity-only primary models.
  - Treat EC1 ↔ EC2 as an initial Caribbean comparison within this design. The broader, continuous test of whether geographical co-occurrence predicts transfer is the exploratory Pacific overlap analysis `[R3-6]`, not the EC1–EC2 result alone.
  - Assumption: recording differences affect both coda kinds equally.
- **`[R3-4]`:** compare coda-type distributions across clans within each dataset only, as a diagnostic (clan identification is solved). Never treat Caribbean textual labels and Pacific numeric labels as a shared type vocabulary.
- **`[R3-5]` Coverage diagnostic:** for each source-to-target score, count occurrences of the target coda's complete fixed-token sequence in that specific source model's training data.
  - A target coda is *covered* when its sequence occurs at least five times; otherwise it is *uncovered*.
  - Calculate coverage and paired transfer loss separately for every source-fold model, then aggregate them with the same rule used for the transfer score. Do not call a sequence covered because it appears in a source fold excluded from that particular model's training data.
  - Report coverage share and paired transfer loss separately for covered and uncovered target codas. This distinguishes patterns absent from the source training inventory from poor transfer on patterns the source does produce.
- **`[R3-6]` Exploratory Pacific overlap analysis:** use Hersh et al.'s Supplementary Table S4 directional spatial-overlap values recorded in `config/rq3_spatial_overlap.json`. For each of the 21 unordered Pacific clan pairs, use the mean of its two directions.
  - For pair \(a,b\), average the two directed transfer losses \(G(a,b)\) and \(G(b,a)\), then test whether greater mean overlap is associated with lower average transfer loss.
  - Run the association separately for identity and non-identity codas. Use a predeclared exact permutation test over all \(7! = 5,040\) Pacific-clan label permutations.
  - Label the result exploratory: the pairs share clans and the published overlap measure has no temporal component.
- **`[R3-7]` Model sensitivity:** retain the fixed-50ms-v1 representation for the primary analysis. If time permits, repeat the matrix with a VLMC using 20 ms bins, reporting likelihood in bits per second rather than comparing its raw scores directly with the primary representation. This checks whether the primary conclusion depends on the model family and timing resolution.
- **Rules:**
  - No multi-whale conditions on `[D6]` or `[D4]`.
  - Day-level coda order is never "own history".
  - Don't redo Leitão et al.'s descriptive VLMC/KL analysis; `[R3-7]` is instead a held-out transfer-loss sensitivity analysis.

---

## 9. Reporting and interpretation

**Report for every contrast**
- sign, day-bootstrap CI and positive-day count;
- repeat/switch parts;
- baselines alongside;
- the sensitivity run with the old delay;
- the 82.5% repetition rate.

**Caution about nulls:** limited data and capacity push contrasts toward zero. A positive contrast is robust; a null might just mean too little data. Say so whenever reporting a null.

**How to read the outcomes**

| Result | Reading |
|---|---|
| RQ1 contrast small | Own history adds little beyond partner and timing: structure is mainly interactional |
| Partner-content contrast small | Whether/when the partner replied matters more than what it said: consistent with persistence |
| Timing contrast small | Content carries the predictability: the interactional confound doesn't arise |
| All substantial | The channels are complementary |
| RQ1 contrast lives in the repeat part | Own history's value is persistence-shaped |

---

## 10. Out of scope

- Audio models, including WhAM.
- Efficiency laws (Menzerath, Zipf's law of abbreviation) and surprisal-vs-frequency analyses.
- Duality of patterning (already covered by Sinha & Chavan).
- Exchanges with more than two whales.
- Claims about meaning or grammar.

---

## 11. Open questions

- Do the tempo/ornament pickles align with `sperm-whale-dialogues.csv` by row, like `rhythms.p`?
- How many `[D1]` exchanges fully match `[D2]` (RQ2 secondary)?
- Is `[D6]` Leitão et al.'s dataset, and how many EC2 codas does it have?
- Is an authoritative Caribbean identity-coda mapping available (`[R3-3]`)?
- Per-clan coda counts in `[D4]` after filtering.
- Is the switch part of `[P1]` (~270 codas) big enough to show anything? Check after the first numbers.
- Should partner-coda content heard before the target's onset ever enter $S$? Current answer: no.

---

## 12. Order of work

1. **Phase 1 — preprocessing fixes:**
   - new $S$ (DEC-1) — **done**;
   - shared unknown-type module (DEC-7, DEC-10);
   - record the `pair_category` missingness finding in the data dictionary;
   - flag long codas.
2. **Phase 2 — evaluation and folds:** the shared evaluation module (metrics, paired contrasts, bootstrap, `[P1]` split) and the leave-one-day-out harness.
3. **Phase 3 — first numbers:**
   - Logistic regression on all eight conditions, plus the baselines. This is the debug step.
   - Then the LSTM on all eight conditions, 3 seeds each.
   - The Assayag reproduction check on (iv), and checks `[U1]`, `[U2]`.
4. **Phase 4 — robustness:** remaining checks `[U3]`–`[U6]` and `[V1]`–`[V2]`; the Transformer.
5. **RQ2 primary:** swap the label column.
6. **RQ3 initial Caribbean comparison:** EC1 ↔ EC2.
7. **RQ2 secondary, then the remaining RQ3 9 × 9 matrix.**

---

## References

Assayag et al. 2026, *Bioacoustics* 35(4) · Bermant et al. 2019, *Sci. Rep.* · Gubnitsky et al. 2025, *Sci. Rep.* · Hersh et al. 2022, *PNAS* · Leitão et al. 2024, *eLife* · O'Connor & Andreas 2021, *ACL* · Schulz et al. 2008, *Anim. Behav.* · Sharma et al. 2024a, *Nat. Commun.* · Sharma et al. 2024b, *bioRxiv* (WhaleLM) · Sinha & Chavan 2026, *arXiv:2606.16084* · Terwilliger et al. 2025, *NeurIPS workshop* · Xu et al. 2020, *ICLR*
