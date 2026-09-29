# Plan — Sequential structure vs interactional dynamics in sperm whale coda exchanges

What the project does. IDs in brackets (e.g. `[D5]`) are referenced by prompts and code.
- **Why** each choice was made, and how this differs from the proposal: `docs/DECISIONS.md`.
- **Deferred** ideas: `docs/LATER.md`.

**Status (2026-09-29):**
- **Done:** the preprocessing pipeline (RQ1: 1,547 targets; Sharma pipeline: 1,318), including the new $S$ (DEC-1).
- **Done:** Phase 1: shared rare/unseen-type policy (DEC-7/10), verified pair-category missingness documentation, and the Sharma long-coda flag.
- **Done:** Phase 2: 14 leave-one-day-out folds and the shared evaluation module.
- **Done:** logistic-regression reference v2 with copy weights and validation-chosen L2 (DEC-16), plus the old-delay sensitivity run: `reports/rq1_logistic_v2.md`, `reports/rq1_retrospective_sensitivity.md`. A rerun with the extended L2 grid is still to do.
- **Superseded:** logistic regression v1 and LSTM v1 (fixed settings, no copy path); kept as a record, see `reports/README.md`.
- **In progress:** LSTM v2 (DEC-17), pilot first.
- **RQ3:** preprocessing on branch `rq3-cross-clan` (Vlad). Design updated to a descriptive transfer study on shared building blocks (§8, DEC-19); no model code yet.

See §12 for the order of work.

---

## 1. Question

Recent models find long-range structure in coda sequences (WhaleLM: dependence on up to eight codas; Sinha & Chavan 2026: second-order dependence). Terwilliger et al. (2025) argue this may be **persistence**, not composition: `a a b` could be a rule, or a whale repeating `a` until it gets a reply. Nobody has conditioned jointly on own history, partner history and timing.

- **RQ1.** Does a whale's own coda history still predict its next coda once the partner's codas and the interaction timing are known?
  - **Primary contrast:** LL(viii) − LL(vii).
- **RQ2.** Does the answer hold under a different coda labelling?
- **RQ3.** How well does a model of within-coda timing trained on one clan predict another clan's codas, and how much of the shortfall comes from patterns the source clan never produces? Descriptive, with one exploratory test (§8, DEC-19).

**Method.** We train one model per combination of inputs. The difference in held-out log-likelihood between models estimates how much usable information an input adds: predictive 𝒱-information (Xu et al., 2020; O'Connor & Andreas, 2021). It is relative to model class and data size, and defined on the real data distribution, so training data is never filtered or reweighted.

**Scope.**
- Claims are about **two-whale (dyadic) exchanges** only.
- Structure that survives the persistence tests is "not persistence". It is not evidence of grammar, and we make no claims about meaning.

---

## 2. Data

| ID | Source | Size | Used for |
|---|---|---|---|
| `[D1]` | Assayag et al. 2026, `data/raw/dswp-2014-2016-retagged-maor-2025.csv` | 3,933 codas, EC1 except one day (below), 14 days (2014-04-16 to 2016-05-13) | RQ1; both RQ2 labellings |
| `[D2]` | Sharma et al. 2024a, `data/raw/sperm-whale-dialogues.csv` + `rhythms.p` | 3,840 codas, 219 recordings | secondary RQ2 labels; standalone "whole-pipeline" robustness check |
| `[D6]` | Sharma et al. 2024a "Dataset 1" (DSWP), `data/raw/CarribeanCodas.csv` (sic); same counts as the Dominica sets of Bermant et al. 2019 and Leitão et al. 2024 | 8,719 codas: EC1 7,770, EC2 949; 97 recording days | RQ3 |
| `[D4]` | Hersh et al. 2022, `data/raw/PacificCodas.csv` (OSF `ae6pd`); identity coda types and clan overlap from their SI Tables S2 and S4 | 24,237 rows; 23,429 codas with 3–10 clicks, 22,829 of them with a clan; 7 clans, 191 repertoires | RQ3 |

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
- **One `[D1]` day is not EC1.** Test day 2016-04-12 (52 comparison targets) is an unidentified unit (`ZZZ`) that `[D6]` labels EC2 (same `codaNUM2018`). Report RQ1 with and without that day; without it the logistic v2 contrasts barely change (own history +1.363 vs +1.361 bits/coda).
- **`[D6]` encoding:** the file starts with a UTF-8 BOM, so read it with `utf-8-sig`.
- **`[D6]` dates** come in two formats (`dd/mm/yyyy` and `dd-mm-yyyy`, both day first); four days appear in both, so group by the parsed date.
- **`[D6]` types:** 25 named types plus 10 `N-NOISE` classes (codas the type clustering left unassigned; Gero et al. 2016). Unit `ZZZ` (134 codas, one day) is an unidentified unit labelled EC2.
- **`[D4]` types** are numeric codes, one for every 3–10-click coda. The two datasets' type systems differ, so type comparisons stay within a dataset.

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
  - An $A$-model *below* it in log-likelihood (CI entirely below zero) has failed to train. Every $A$-model has the copy weight (`[M4]`), so it can express repetition with one number; *matching* the baseline is therefore a finding: own history carries nothing beyond repetition.

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
  - day-level bootstrap CI (resample the 14 days, never individual codas; 10,000 replicates);
  - the number of days, out of 14, with a positive contrast;
  - the unweighted mean of the 14 day-level contrasts, next to the token-weighted mean. The token-weighted mean is the primary estimand; three days hold 58% of the targets, so the unweighted mean shows whether a result rests on a few days.
  - Note in the write-up that 14 days give rough CIs.

**`[M4]` Models** (DEC-12, DEC-16, DEC-17)
- **Shared copy weights, in every model that uses $A$ or $B$.** For each candidate coda type $c$:
  $$\text{score}(c) = \text{usual per-type terms} + \beta\,\mathbb{1}[c = \text{own last coda}] + \gamma\,\mathbb{1}[c = \text{partner's last coda}]$$
  - $\beta$ exists only in conditions with $A$, $\gamma$ only in conditions with $B$. Each is one number shared across all coda types.
  - Why: without it, a model has to learn "repeat the last coda" separately for every type, and rare types never learn it (DEC-16). It adds no new information, only a cheaper way to use what $A$ or $B$ already contain.
- **Reference model: multinomial logistic regression**, run first; it is the `[U1]` reference.
  - Inputs: one-hot encodings of the last 3 codas of each available stream, the turn category, the two timing values (standardised, plus `log1p`), and the copy weights.
  - v4 (DEC-20): history/base and S L2 are chosen jointly per fold and condition on the validation day from {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}, by validation log-likelihood; ties prefer larger history/base L2, then larger S L2. History features, copy weights and the shared intercept use the base penalty; all S coefficients use S L2. Conditions without S retain v3 exactly. The model trained on training days only is scored on the test day.
- **Main model: LSTM**, reported for all eight conditions with 3 seeds.
  - One 1-layer LSTM encoder per coda-history input ($A$, $B$) and a small feed-forward encoder for $S$, concatenated into a shared output layer, plus the copy weights. An absent input means its encoder and its copy weight are removed.
  - Optimiser: Adam, learning rate 1e-3, batch 32, up to 300 epochs, patience 20, restoring the best validation epoch.
  - Chosen per fold and condition on the validation day, with one grid and one rule for all conditions: dropout {0, 0.2, 0.4} × history/base weight decay {1e-5, 1e-4, 1e-3} × S weight decay {1e-5, 1e-3, 0.1, 1, 10} × hidden size {32, 64} (DEC-20; omit the S axis when S is absent). Criterion: validation log-likelihood averaged over the 3 seeds; ties prefer smaller hidden size, larger dropout, larger history/base decay, then larger S decay. The chosen setting's 3 seeds are the reported runs.
  - **Pilot first:** conditions A and A+B+S on the three largest test days plus one small one (one seed and a reduced grid are fine), logging training and validation log-likelihood per epoch, compared with logistic regression on the same folds. Run the full study only if the pilot is at or above repeat-own-last and not reliably below logistic regression.
- **Stop rules (log-likelihood only).** Stop and investigate before writing conclusions if any of these has a day-bootstrap CI entirely below zero: an $A$-condition against repeat-own-last; an add-one-input contrast; (LSTM) the paired difference to logistic regression on a condition. Accuracy and macro-F1 are reported but not used to stop. The LSTM *matching* logistic regression is acceptable.
- **Context check against Assayag et al.** Their split is random, ours holds out whole days, so their accuracies are optimistic next to ours. Compare like with like, without retraining: condition (iv) on switch/overlap targets only against the majority class on that subset, and condition (iii) against their single-whale results.
- **Later:** a small Transformer with the same per-input structure; Kneser-Ney/VLMC models (`docs/LATER.md`).

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
| `[U1]` | The LSTM must not do worse than logistic regression on the same condition (paired CI entirely below zero) | Otherwise it is undertrained |
| `[U2]` | For the 12 "add one input" pairs, the larger model should score at least as well; count violations | Violations signal fitting problems |
| `[U3]` | Hidden size is selected from 32/64 per fold (`[M4]`); optionally check 128 afterwards. Contrasts should plateau | Capacity limits |
| `[U4]` | Retrain on 25–100% of training days; contrasts should stabilise | Data limits |
| `[U5]` | Variance across seeds | |
| `[U6]` | Train–test gap per condition | |

An add-one-input contrast with a CI entirely below zero signals a fitting problem (optimisation or regularisation), not a finding: investigate before interpreting it.

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

RQ3 is descriptive: a transfer matrix over all nine clans, a coverage breakdown and one exploratory overlap test (DEC-19). Every RQ3 model uses the building blocks `[R3-0]`–`[R3-3]`, including any later pooled model (`docs/LATER.md`).

**Building blocks**
- **`[R3-0]` Processed codas** (`data/processed/rq3/`, Vlad's pipeline):
  - eligible: 3–10 clicks, an assigned clan (EC1, EC2 or one of the seven Pacific clans), finite positive intervals, no non-zero padding;
  - excluded with a recorded reason: any interval under 0.005 s (double-marked clicks) or of 1 s or more;
  - `recording_date` (ISO), with `[D6]` dates parsed day first from both formats; `group_id` = the recording date for `[D6]`, the repertoire (`grpvar`) for `[D4]`;
  - flags, not filters: `noise_type` (`[D6]` `N-NOISE` classes), `unidentified_unit` (`[D6]` unit `ZZZ`);
  - identity status for `[D4]` codas: identity if the coda's type is on its own clan's list in Hersh et al.'s SI Table S2 (a sourced config file), otherwise non-identity. `[D6]` codas get none: no comparable list exists.
- **`[R3-1]` Tokeniser:** one shared function. Symbol = floor(interval / 0.05 s), 0–19, the same for every clan and model and fitted on no data (Leitão et al. 2024). A coda is `BOS s1 … sn EOS`; models predict every interval symbol and the `EOS`. Its ID (`fixed-50ms-v1`) is stored with every score. Sensitivity, if time: 0.02 s bins; scores at different widths are compared per second (subtract log2 of the bin width in seconds per interval symbol).
- **`[R3-2]` Held-out groups:** one committed fold file (`clan, group_id, fold, n_codas`). Per clan, groups are sorted by size (largest first; ties by `sha256("42:<clan>:<group_id>")`) and each goes to the fold with the fewest codas so far: 5 folds per clan, 6 for EC2 (one day each). Own-clan models then train on about 80% of their clan (EC2: 62–93%, set by its day sizes).
- **`[R3-3]` Scores:** one row per scored coda: `run_id, model, tokeniser, train_clans, train_folds, target_clan, rq3_coda_id, target_fold, n_symbols` (intervals + `EOS`), `ll_bits` (log2, whole coda), `ll_eos_bits`; plus a run manifest (config, SHA-256 of the data and the fold file, commit). One shared function aggregates: bits per symbol = Σ ll / Σ n_symbols, bits per coda = mean ll, CIs by bootstrap over the target's groups (1,000 draws).

**Analyses**
- **`[R3-4]` Transfer matrix:** all 9 × 9 source–target pairs.
  - Own-clan score of a coda in fold k: the model trained on its clan's other folds (the diagonal of the matrix).
  - Transfer score from source s: the mean of the log-likelihoods from s's fold models (each trained on s minus one fold), so both scores come from models trained on about 80% of a clan.
  - Transfer loss G(s,t) = own-clan score − transfer score, on the same codas; per symbol and per coda, with CIs.
  - Settings (n-gram order, smoothing) chosen by cross-validation over the training clan's own folds; the target clan is only ever scored.
  - Models: smoothed n-gram (main); VLMC (sensitivity, if time).
  - Report each source's training size beside its row: sources differ up to about 8.6× in size, and there are no size-matched reruns (`docs/LATER.md`).
- **`[R3-5]` Coverage:** for each pair, the share of the target's codas whose full symbol string occurs at least 5 times in the source's data; G reported separately for covered and uncovered codas (after Aycock et al. 2025).
- **`[R3-6]` Exploratory overlap test, Pacific only:** separate models for identity and for non-identity codas (same building blocks). Per clan pair, G averaged over both directions and related to the pair's mean spatial overlap (Hersh et al. SI Table S4); permutation test over the 7 clan labels (5,040 orders), separately for identity and non-identity codas. The prediction (Leitão et al. 2024): overlap lowers G for non-identity codas only. With 21 pairs this is exploratory.
- **`[R3-7]` Coda-type distributions** per clan: diagnostic, within one dataset only.
- **Sensitivity:** the matrix without the `ZZZ` codas. `N-NOISE` codas stay in `[R3-4]` and `[R3-5]` (symbol strings, no types) and are left out of `[R3-6]` and `[R3-7]`.

**Rules**
- No multi-whale conditions on `[D6]` or `[D4]`; day-level coda order is never own history.
- Tuning and model choice use the training clan only.
- Don't repeat Leitão et al.'s VLMC/KL comparison as a result; their bins and VLMC may be used as tools.
- Claims: how well clans' within-coda timing predicts each other's, and how much of that is coverage. No claim that sharing waters causes convergence, nothing about meaning, no "dialects".

---

## 9. Reporting and interpretation

**Report for every contrast**
- sign, token-weighted and unweighted day means, day-bootstrap CI and positive-day count;
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
- Why is the `[D6]` unit `ZZZ` labelled EC2? Only the DSWP can say.
- Is the switch part of `[P1]` (~270 codas) big enough to show anything? Check after the first numbers.
- Should partner-coda content heard before the target's onset ever enter $S$? Current answer: no.

---

## 12. Order of work

1. **Phase 1 — preprocessing fixes: done.**
   - new $S$ (DEC-1) — **done**;
   - shared unknown-type module (DEC-7, DEC-10) — **done**, fitted on the current training split; fold-specific fitting follows in Phase 2;
   - record the `pair_category` missingness finding in the data dictionary — **done**, source counts verified;
   - flag long codas — **done**, 153 Sharma source codas flagged without filtering.
2. **Phase 2 — evaluation and folds: done.** The shared evaluation module (metrics, paired contrasts, bootstrap, `[P1]` split) and the leave-one-day-out harness.
3. **Phase 3 — first numbers:**
   - Logistic regression v1 (fixed L2) — done, superseded. v2 with copy weights and validation-chosen L2 (DEC-16) — done. Rerun with the extended L2 grid — to do.
   - LSTM v1 (fixed WhaleLM settings, no copy path) — done, superseded (DEC-17).
   - LSTM v2 (DEC-17): pilot, then all eight conditions with 3 seeds; checks `[U1]`, `[U2]`; the Assayag context check.
4. **Phase 4 — robustness:** remaining checks `[U3]`–`[U6]` and `[V1]`–`[V2]`; the Transformer.
5. **RQ2 primary:** swap the label column.
6. **RQ3:** building blocks `[R3-0]`–`[R3-3]` (Vlad's preprocessing PR), then `[R3-4]`–`[R3-5]`, then `[R3-6]`–`[R3-7]`.
7. **RQ2 secondary.**

---

## References

Assayag et al. 2026, *Bioacoustics* 35(4) · Aycock et al. 2025, *ICLR* · Bermant et al. 2019, *Sci. Rep.* · Gero et al. 2016, *R. Soc. Open Sci.* · Gubnitsky et al. 2025, *Sci. Rep.* · Hersh et al. 2022, *PNAS* · Leitão et al. 2024, *eLife* · O'Connor & Andreas 2021, *ACL* · Schulz et al. 2008, *Anim. Behav.* · Sharma et al. 2024a, *Nat. Commun.* · Sharma et al. 2024b, *bioRxiv* (WhaleLM) · Sinha & Chavan 2026, *arXiv:2606.16084* · Terwilliger et al. 2025, *NeurIPS workshop* · Xu et al. 2020, *ICLR*