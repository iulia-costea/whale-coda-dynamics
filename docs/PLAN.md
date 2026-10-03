# Plan — Sequential structure vs interactional dynamics in sperm whale coda exchanges

What the project does. IDs in brackets (e.g. `[D5]`) are referenced by prompts and code.
- **Why** each choice was made, and how this differs from the proposal: `docs/DECISIONS.md`.
- **Deferred** ideas: `docs/LATER.md`.

**Status (2026-10-03):**
- **Done:** the preprocessing pipeline (RQ1: 1,547 targets; Sharma pipeline: 1,318), including the new $S$ (DEC-1).
- **Done:** Phase 1: shared rare/unseen-type policy (DEC-7/10), verified pair-category missingness documentation, and the Sharma long-coda flag.
- **Done:** Phase 2: 14 leave-one-day-out folds and the shared evaluation module.
- **Done:** logistic-regression reference v4 with copy weights and separate validation-chosen penalties for the history and $S$ blocks (DEC-16, DEC-20): `reports/rq1_logistic_v4.md`, `reports/rq1_retrospective_sensitivity_v4.md`. Gate 1 fully holds: all 4,914 fits converged, no DEC-16 stop rule fires in either the primary or old-delay run, and no condition sits at a grid edge in a majority of folds for its real feature blocks. The `add_input:A_B_S-A_B` failure that persisted through v2 and v3 is resolved: the timing contrast now spans zero (`-0.020`, CI `[-0.046, 0.001]`) instead of being reliably negative, confirming DEC-20's diagnosis.
- **Superseded:** logistic regression v1, v2 and v3, and LSTM v1 (fixed settings, one shared penalty, or no history/$S$ split); kept as a record, see `reports/README.md`.
- **Superseded:** LSTM v2 (DEC-17, DEC-20): reliably below logistic v4 on 5 of 7 conditions in its full run; kept as a record (DEC-21).
- **Done:** the residual LSTM on top of logistic v4 (DEC-21): `reports/rq1_lstm_residual.md`, with its checks (`reports/rq1_lstm_residual_checks.md`: wiring, learnability and planted-signal gates pass) and device choice (`reports/rq1_lstm_residual_throughput.md`: CPU ×18). No `[M4]` stop rule fires. The residual gain over logistic v4 is below zero in point estimate on all seven fitted conditions (−0.010 to −0.122 bits/coda), with every day-bootstrap CI spanning zero; logistic v4 alone is chosen on 1–6 of 14 folds per condition.
- **RQ2:** a token-inventory study (§7, DEC-22). **Done:** six inventories of the same codas (`reports/rq2_inventories.md`: the `[R2-1]`/`[R2-2]` checks and `[R2-0]` counts reproduce), their folds (UNK counts equal `[R2-0]`'s), logistic v4 on each and the comparison `reports/rq2_inventories_v4.md`. All fits converged. DEC-16 stop rules fire for B, C, D, E and D′ (add-one-input contrasts that add S, CIs just below zero); A and E′ have none. Per `[R2-7]` no conclusion is drawn from the five; no setting was changed.
- **Pending:** logistic v5 (DEC-23): every input block can be absent, chosen on the validation day from v4's committed fits, and stop rules by exact sign-flip tests with Holm's correction; for RQ1 (primary and old-delay) and the six RQ2 inventories. RQ2's comparison uses v5 for every inventory; RQ1's headline stays v4, with v5 as a robustness line.
- **Pending:** the partner-tempo ablation (DEC-24, `[R2-8]`).
- **RQ3:** preprocessing on branch `rq3-cross-clan` (Vlad). Design updated to a descriptive transfer study on shared building blocks (§8, DEC-19); no model code yet.

See §12 for the order of work.

---

## 1. Question

Recent models find long-range structure in coda sequences (WhaleLM: dependence on up to eight codas; Sinha & Chavan 2026: second-order dependence). Terwilliger et al. (2025) argue this may be **persistence**, not composition: `a a b` could be a rule, or a whale repeating `a` until it gets a reply. Nobody has conditioned jointly on own history, partner history and timing.

- **RQ1.** Does a whale's own coda history still predict its next coda once the partner's codas and the interaction timing are known?
  - **Primary contrast:** LL(viii) − LL(vii).
- **RQ2.** Does the answer hold under a different token inventory, i.e. a different way of turning codas into discrete labels? (§7, DEC-22)
- **RQ3.** How well does a model of within-coda timing trained on one clan predict another clan's codas, and how much of the shortfall comes from patterns the source clan never produces? Descriptive, with one exploratory test (§8, DEC-19).

**Method.** We train one model per combination of inputs. The difference in held-out log-likelihood between models estimates how much usable information an input adds: predictive 𝒱-information (Xu et al., 2020; O'Connor & Andreas, 2021). It is relative to model class and data size, and defined on the real data distribution, so training data is never filtered or reweighted.

**Scope.**
- Claims are about **two-whale (dyadic) exchanges** only.
- Structure that survives the persistence tests is "not persistence". It is not evidence of grammar, and we make no claims about meaning.

---

## 2. Data

| ID | Source | Size | Used for |
|---|---|---|---|
| `[D1]` | Assayag et al. 2026, `data/raw/dswp-2014-2016-retagged-maor-2025.csv` | 3,933 codas, EC1 except one day (below), 14 days (2014-04-16 to 2016-05-13) | RQ1; RQ2 (every inventory is computed from its intervals and durations) |
| `[D2]` | Sharma et al. 2024a, `data/raw/sperm-whale-dialogues.csv` + `rhythms.p`; also `mean_codas.p` and `tempos-dict.p`, copied unchanged from `github.com/pratyushasharma/sw-combinatoriality` (commit `7228c8e`) | 3,840 codas, 219 recordings | RQ2 rhythm templates `[R2-1]` and tempo-class sensitivity `[R2-2]`; standalone "whole-pipeline" robustness check |
| `[D6]` | Sharma et al. 2024a "Dataset 1" (DSWP), `data/raw/CarribeanCodas.csv` (sic); same counts as the Dominica sets of Bermant et al. 2019 and Leitão et al. 2024 | 8,719 codas: EC1 7,770, EC2 949; 97 recording days | RQ3; RQ2 tempo boundaries `[R2-2]` |
| `[D4]` | Hersh et al. 2022, `data/raw/PacificCodas.csv` (OSF `ae6pd`); identity coda types and clan overlap from their SI Tables S2 and S4 | 24,237 rows; 23,429 codas with 3–10 clicks, 22,829 of them with a clan; 7 clans, 191 repertoires | RQ3 |

**Facts that matter**
- **Coda types `[D1]`:** two labellings exist (RQ2 adds three computed ones, §7). `guy_coda_type` has 23 types and `decision_tree_coda_type` has 25, and they disagree on 265 codas. 21 types appear in the RQ1 comparison set. Never write "27 types"; that figure is from Gubnitsky et al.
- **No global whale ID.**
  - `Whale` is a segment-local caller code in both files.
  - `IDN` in `[D1]` is 0 (unresolved) on 2,510 rows.
- **`pair_category` `[D1]`** is 62% empty. Never use it as a feature; turn category is derived instead.
  - Unlabelled codas are disproportionately non-focal: 66% vs 50%.
  - All 157 missing-`Focal` rows are unlabelled.
- **`[D1]` and `[D2]` overlap heavily.** 214/219 recordings are shared, and 2,769 codas match on (`REC`, `TsTo`, `Duration`) (rounded to 3 decimals). They are not independent corpora.
- **`[D2]` quirks:**
  - It keeps 153 codas with more than 10 clicks, which Sharma et al. excluded. Flag them, and report results with and without them.
  - `rhythms.p` holds one rhythm label per CSV row (joined by position). `mean_codas.p` holds the 18 rhythm templates. `tempos-dict.p` is not per coda: it lists the durations in each of 5 tempo classes, which fixes only the class boundaries. `ornaments.p` (one flag per row, 151 ornamented) is not used (DEC-22).
  - One `Duration=0` row drops the whole 731-row recording `sw061b003` in the Sharma pipeline.
- **One `[D1]` day is not EC1.** Test day 2016-04-12 (52 comparison targets) is an unidentified unit (`ZZZ`) that `[D6]` labels EC2 (same `codaNUM2018`). Report RQ1 with and without that day; without it the logistic v2 contrasts barely change (own history +1.363 vs +1.361 bits/coda).
- **`[D6]` encoding:** the file starts with a UTF-8 BOM, so read it with `utf-8-sig`.
- **`[D6]` dates** come in two formats (`dd/mm/yyyy` and `dd-mm-yyyy`, both day first); four days appear in both, so group by the parsed date.
- **`[D6]` types:** 25 named types plus 10 `N-NOISE` classes (codas the type clustering left unassigned; Gero et al. 2016a). Unit `ZZZ` (134 codas, one day) is an unidentified unit labelled EC2.
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
  - An $A$-model *below* it in log-likelihood (the stop rule fires: DEC-23, Holm-adjusted one-sided sign-flip p < 0.05) has failed to train. Every $A$-model has the copy weight (`[M4]`), so it can express repetition with one number; *matching* the baseline is therefore a finding: own history carries nothing beyond repetition.

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
  - Neural models average over their seeds per coda (5 for the residual LSTM, DEC-21).
- **Uncertainty:**
  - day-level bootstrap CI (resample the 14 days, never individual codas; 10,000 replicates);
  - the number of days, out of 14, with a positive contrast;
  - the unweighted mean of the 14 day-level contrasts, next to the token-weighted mean. The token-weighted mean is the primary estimand; three days hold 58% of the targets, so the unweighted mean shows whether a result rests on a few days.
  - the exact sign-flip $p$ (DEC-23): all $2^{d}$ sign assignments of the day-level sums over the $d$ test days, with the token-weighted mean as the statistic; two-sided for the three `[M2]` contrasts, one-sided for the stop rules (`[M4]`).
  - Note in the write-up that 14 days give rough CIs.

**`[M4]` Models** (DEC-12, DEC-16, DEC-17, DEC-20, DEC-21, DEC-23)
- **Shared copy weights, in every model that uses $A$ or $B$.** For each candidate coda type $c$:
  $$\text{score}(c) = \text{usual per-type terms} + \beta\,\mathbb{1}[c = \text{own last coda}] + \gamma\,\mathbb{1}[c = \text{partner's last coda}]$$
  - $\beta$ exists only in conditions with $A$, $\gamma$ only in conditions with $B$. Each is one number shared across all coda types.
  - Why: without it, a model has to learn "repeat the last coda" separately for every type, and rare types never learn it (DEC-16). It adds no new information, only a cheaper way to use what $A$ or $B$ already contain.
- **Separate penalties for history and timing (DEC-20).** Every model with $S$ splits its regularisation in two: one for the history blocks ($A$/$B$'s features plus $\beta$/$\gamma$), one for $S$. Both are chosen jointly on the validation day from the same grid; conditions without $S$ keep a single penalty. Why: a single shared penalty cannot suit both a small, possibly-uninformative block and a large, informative one, so the model may be unable to switch $S$ off even when it should (violating the optional-ignorance assumption behind conditional 𝒱-information).
- **Reference model: multinomial logistic regression**, run first; it is the `[U1]` reference.
  - Inputs: one-hot encodings of the last 3 codas of each available stream, the turn category, the two timing values (standardised, plus `log1p`), and the copy weights.
  - L2 for the history blocks and, where $S$ is present, a separate L2 for $S$, both chosen per fold and condition on the validation day from {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}, by validation log-likelihood; ties go to the larger value on each axis. The model trained on training days only is scored on the test day.
- **Logistic v5 (DEC-23): every input block can be absent, chosen on the validation day.** For each fold and condition $c$, the candidates are v4's chosen fits of every sub-condition $s \subseteq c$ ($A$, $B$, $S$ each on or absent; all absent = condition (i)). The candidate with the highest validation-day log-likelihood is chosen (exact ties: fewer blocks), and its test-day predictions are $c$'s.
  - No new fits: computed from v4's committed `models.jsonl`, `l2_selection.jsonl` and `predictions.jsonl`; condition (i) is the fold's prior scored on the validation day.
  - Every v4 run (RQ1 primary and old-delay, the six RQ2 inventories). RQ2's comparison uses v5 for all seven inventories; RQ1's headline lattice stays v4, with v5 as a robustness line.
  - Sensitivity: DEC-21's one-SE rule instead of the maximum (simpler = fewer blocks).
- **Neural model: residual LSTM on top of logistic v4 (DEC-21)**, all eight conditions, 5 seeds (42–46). It is the enlarged-family check; logistic v4 gives RQ1's headline lattice.
  - Per fold and condition $c$: $z(x) = z^{\mathrm{LR}}_c(x) + \sum_{s\in c} U_s\,h_s(x)$. Logistic v4's logits are a fixed offset, precomputed from its committed fits. One 1-layer LSTM encoder per coda history (up to 10 codas) and a small feed-forward encoder for $S$; bias-free output projections $U_s$ initialised at 0; no separate output bias or copy weights (both are in the offset).
  - Each block on or absent (absent = not built); with every block absent the model is logistic v4 exactly.
  - AdamW (decoupled weight decay 0.1), learning rate 1e-3, batch 32, up to 300 epochs, patience 20, restoring the best validation epoch; epoch 0 (= logistic v4) is a candidate.
  - Candidates: hidden size {16, 32, 64} × dropout {0, 0.2, 0.4} × each block on or absent, plus logistic v4. Selection: per-target validation log-likelihood averaged over the seeds; take the simplest candidate within one standard error of the best (SE of the per-target paired differences, target level); simplicity order: fewer blocks on, smaller hidden size, more dropout.
  - **Checks before the full run** (training and validation days only):
    1. epoch 0 reproduces logistic v4's validation log-likelihood to float precision, and the precomputed offsets reproduce v4's stored test probabilities;
    2. every block absent reproduces v4 exactly (unit test);
    3. learnability: with blocks on and no early stopping, training loss falls well below logistic regression's;
    4. planted signal on one fold: relabel some targets (labels only, histories unchanged) as the own coda at lag 6, only where it differs from lags 1–3 and the target has at least 6 own codas; on training and validation days; refit logistic regression on the planted data; the $A$ residual should switch on and recover the signal. Several fractions give a rough detectable effect size.
    Time a pilot before the full run.
- **LSTM v2 (DEC-17, DEC-20) — superseded by DEC-21**, kept as a record: reliably below logistic v4 on 5 of 7 conditions; likely causes in DEC-21.
- **Stop rules (log-likelihood only; DEC-23).** Per run, the 16 checks (the 12 add-one-input contrasts and the 4 $A$-conditions against repeat-own-last) each get an exact one-sided sign-flip test on the day-level sums of per-coda differences (alternative: mean below zero). Holm's correction across the 16 at $\alpha = 0.05$; a rule fires if its Holm-adjusted $p < 0.05$: stop and investigate before writing conclusions. Day-bootstrap CIs stay the reported intervals. Every run reports both this rule and DEC-16's old one (CI entirely below zero), for v4 and v5. (Residual LSTM) a residual gain over logistic v4 with a day-bootstrap CI entirely below zero on a condition signals selection overfitting to validation days rather than a training failure (DEC-21). Accuracy and macro-F1 are reported but not used to stop. A residual gain of zero is acceptable.
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
| `[U1]` | The residual LSTM must not do worse than logistic v4 on the same condition (paired CI entirely below zero) | It can't on validation days by construction (DEC-21); below on test days means selection overfitting to the validation days |
| `[U2]` | For the 12 "add one input" pairs, the larger model should score at least as well; count violations. Under logistic v5 they are ≥ 0 on validation days by construction (DEC-23) | Violations signal fitting problems |
| `[U3]` | Hidden size is selected from 16/32/64 per fold (`[M4]`); optionally check 128 afterwards. Contrasts should plateau | Capacity limits |
| `[U4]` | Retrain on 25–100% of training days; contrasts should stabilise | Data limits |
| `[U5]` | Variance across seeds | |
| `[U6]` | Train–test gap per condition | |

An add-one-input contrast that fires the stop rule (DEC-23: Holm-adjusted one-sided sign-flip p < 0.05) signals a fitting problem (optimisation or regularisation), not a finding: investigate before interpreting it.

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

## 7. RQ2 — robustness to the token inventory (DEC-22)

A coda becomes a token by discretising its click intervals, and which discretisation to use is the analyst's choice: whale data come with no given units. RQ2 asks whether RQ1's conclusions depend on that choice. The design follows discrete-unit spoken language modelling, which crosses unit encoders with codebook sizes and keeps the rest of the system fixed (Lakhotia et al. 2021). Claims stay within interval-based discretisations; RQ2 says nothing about whether codas are the right units (Andreas et al. 2022).

**Only the labels change.** Targets (1,547), exchanges, history IDs, $S$, the 14 folds, the UNK policy (`[M6]`, *k* = 2 per fold), logistic v4 and its selection rule are RQ1's. Each inventory labels every retained `[D1]` coda from that coda's own intervals and duration.

**`[R2-0]` Inventories.** Counts on the 1,547 comparison targets, computed before any fit: labels in use, the most common label's share, targets equal to the whale's own last label, and test targets mapped to UNK over the 14 folds. The fold harness reports the exact UNK counts.

| ID | Inventory | Labels | Top label | Repeats | UNK |
|---|---|---:|---:|---:|---:|
| A | `guy_coda_type` (RQ1; existing run) | 21 | 57.2% | 1,277 | 31 |
| B | `decision_tree_coda_type` | 25 | 56.9% | 1,280 | 57 |
| C | Sharma rhythm `[R2-1]` | 16 | 61.9% | 1,215 | 25 |
| D | Sharma rhythm × tempo `[R2-1]`, `[R2-2]` | 42 | 39.9% | 1,142 | 73 |
| E | `guy_coda_type` × tempo `[R2-2]` | 39 | 39.9% | 1,204 | 67 |
| D′ | D with `tempos-dict.p` tempo classes | 47 | 39.5% | 1,147 | 74 |
| E′ | E with `tempos-dict.p` tempo classes | 41 | 39.4% | 1,210 | 66 |

- **Grid:** scheme (named types: A, E; Sharma rhythm: C, D) × resolution (without tempo: A, C; with tempo: E, D).
- **D is the main test:** an independently published scheme, and the one that changes the task most.
- **B is a sanity check:** 75 targets change label and 17 flip between repeat and switch; one sentence in the report.

**`[R2-1]` Rhythm** (Sharma et al. 2024a; their 18 rhythm types come from Gero et al. 2016b):
- click positions = the cumulative sums of `ICI1`…`ICI(n−1)` divided by their total, with *n* = `nClicks`;
- label = the template in `data/raw/mean_codas.p` with the same number of clicks that is nearest by mean squared difference; *n* < 3 or *n* > 10 → rhythm 17;
- verified: this rule reproduces `rhythms.p` on all 3,840 `[D2]` rows. On `[D1]`'s own intervals it agrees with `rhythms.p` on 2,743 of the 2,769 codas matched between `[D1]` and `[D2]` on (`REC`, `TsTo`, `Duration`) (rounded to 3 decimals); the rest differ in their intervals between the two releases.

**`[R2-2]` Tempo:**
- label = `Duration` cut at 0.45, 0.61, 0.93 and 1.08 s, giving 5 classes;
- the cuts are the minima of a Gaussian kernel density estimate (bandwidth 0.035 s, evaluated from 0.10 to 1.75 s in 0.01 s steps) of the EC1 durations between 0.1 and 1.75 s in `[D6]`, as in Sharma et al.'s notebook `code/2-tempo.ipynb`. The pipeline recomputes them from `data/raw/CarribeanCodas.csv` and must reproduce these cuts and the five modes (0.33, 0.51, 0.80, 1.02, 1.26 s; their Fig. 2A and 3);
- sensitivity (D′, E′): the classes in Sharma et al.'s `tempos-dict.p`, cut midway between adjacent classes' extreme durations (0.392, 0.594, 0.907, 1.091 s). The notebook doesn't document how these classes were made; they agree with the KDE classes on 1,511 of the 1,547 targets;
- label format: rhythm `r<k>`, tempo `t<j>`, combined with `|` (`r2|t4`, `1+1+3|t4`).

**`[R2-3]` Comparing inventories**
- Never compare raw log-likelihood or perplexity across inventories. They keep different information about the same codas, so there is no common string to score per character as text NLP does; Lakhotia et al. (2021, p. 1337) make the same point for discrete speech units.
- Per inventory: the lattice, the three `[M2]` contrasts (token-weighted and unweighted means, CI, positive days), the repeat-own-last baseline, and the total predictability $T = \mathrm{LL(viii)} - \mathrm{LL(i)}$.
- **Shares:** each contrast divided by $T$, with a day-bootstrap CI (ratio of token-weighted means within each replicate).
- **Share differences against A:** paired day bootstrap, with the same resampled days for both inventories.
- **Pattern:** for each contrast and inventory, the sign and whether the CI excludes zero.

**`[R2-4]` Tempo decomposition** (exploratory; nested pairs A → E and C → D only):
$$\log_2 p(\text{type}, \text{tempo}) = \log_2 p(\text{type}) + \log_2 p(\text{tempo} \mid \text{type})$$
- Computed from E's (D's) stored probability vectors: $p(\text{type})$ sums E's in-vocabulary labels with that type (D: that rhythm).
- The type term is compared with A's (C's) log-likelihood on the same targets, paired: does the finer model still predict the type as well?
- The tempo term gets the three `[M2]` contrasts: do own history, the partner or timing predict a coda's tempo once its type is known?
- Targets whose E (D) label maps to UNK are left out and counted. UNK mass is left out of the type marginal, so the type term is a lower bound.

**`[R2-5]` UNK:** the same policy for every inventory. Report, per inventory and fold, the share of comparison targets mapped to UNK: a closed vocabulary can look easier by predicting UNK (Mielke et al. 2019).

**`[R2-6]` Sensitivities:** D′ and E′ (tempo boundaries); every inventory without day 2016-04-12 (`[D1]` facts).

**`[R2-7]` Stop rules:** DEC-23's rules apply to every inventory's run, for v4 and v5; the outcomes of DEC-16's old rule under v4 stay in the report. If one fires, report it and draw no conclusion from that inventory. No setting changes before the freeze.

**`[R2-8]` Partner-tempo ablation (DEC-24):** `[R2-4]` found that the partner's codas predict a coda's tempo class beyond own history and timing (A → E: +0.039 bits/coda, CI [0.022, 0.061]); is that the partner's tempo or the partner's rhythm? A dedicated model for the chain rule's tempo factor: the target is the coda's KDE tempo class given its own Sharma rhythm, with own history always present (last 3 tempo classes and rhythms, own-tempo copy weight) and partner tempo (last 3 classes, partner-tempo copy weight), partner rhythm (last 3 rhythms) and $S$ as blocks, fitted with v4's procedure and v5's absent candidates. Contrasts in bits per coda of the tempo class, with day-bootstrap CI and DEC-23's sign-flip $p$: partner tempo alone, partner rhythm alone, tempo beyond rhythm, rhythm beyond tempo; a descriptive breakdown by turn category; secondary: `guy_coda_type` in place of rhythm. Same targets, folds and UNK policy. Conditions, readings and limits: DEC-24.

**Not in RQ2** (`docs/LATER.md`): segmentation proper (a coda as two tokens, rhythm then tempo; clicks as tokens), ornaments, rubato as a continuous input, the residual LSTM on the new inventories. The standalone Sharma pipeline stays a whole-chain check; `reports/rq2.md` is its preprocessing report, not RQ2 results.

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

**The residual LSTM (DEC-21)**
- RQ1's headline numbers come from logistic v4, with v5 as a robustness line (DEC-23); RQ2's comparison uses v5 for every inventory, with v4's comparison kept as the record. The residual LSTM stays the enlarged-family check of v4: its lattice and each condition's residual gain are reported next to v4's.
- A near-zero residual reads "no further structure detectable in the enlarged family at this data size", not "none exists".
- A positive residual means structure in that block that logistic regression can't express: for $A$ and $B$, lags 4–10 or non-additive patterns within lags 1–3; for $S$, nonlinear timing effects. We don't separate these; don't call it "long-range".
- A contrast is v4's contrast plus the difference between the two conditions' residual gains; don't call the contrasts monotone.
- On validation days the residual network and logistic regression agree by construction, so their agreement isn't evidence against an optimisation failure; that evidence is the learnability checks in `[M4]`.
- Course framing: "Logistic regression is a softmax network without hidden layers; the residual LSTM's family contains v4 exactly, as the member whose residual is switched off."

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

- How were the tempo classes in Sharma et al.'s `tempos-dict.p` made? They don't match the KDE minima in their notebook; only the authors can say. RQ2 uses the KDE minima and keeps `tempos-dict.p` as a sensitivity (`[R2-2]`).
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
   - Logistic regression v1 (fixed L2), v2 (copy weights, validation-chosen L2, DEC-16) and v3 (extended grid) — done, superseded. v4 with separate history and $S$ penalties (DEC-20) — done; the reference.
   - Logistic v5 (DEC-23): absent-block candidates chosen on the validation day from v4's fits, and the sign-flip stop rules with Holm's correction, for RQ1 (primary and old-delay) and the six RQ2 inventories — pending.
   - LSTM v1 (fixed WhaleLM settings, no copy path) — done, superseded (DEC-17).
   - LSTM v2 (DEC-17) — done, superseded (DEC-21).
   - Residual LSTM on top of logistic v4 (DEC-21): checks and pilot timing, then all eight conditions with 5 seeds; `[U1]`, `[U2]`; the Assayag context check — **done** (`reports/rq1_lstm_residual.md`); no stop rule fires.
4. **Phase 4 — robustness:** remaining checks `[U3]`–`[U6]` and `[V1]`–`[V2]`; the Transformer.
5. **RQ2 (DEC-22):** compute the inventories and check `[R2-1]`/`[R2-2]`; folds and logistic v4 for D, E, C, B, then D′, E′; the comparison `[R2-3]`–`[R2-6]` — **done** (`reports/rq2_inventories_v4.md`); stop rules fire for B, C, D, E and D′ (`[R2-7]`). Then v5 and DEC-23's stop rules on every inventory, and the partner-tempo ablation `[R2-8]` (DEC-24) — pending.
6. **RQ3:** building blocks `[R3-0]`–`[R3-3]` (Vlad's preprocessing PR), then `[R3-4]`–`[R3-5]`, then `[R3-6]`–`[R3-7]`.

---

## References

Andreas et al. 2022, *iScience* 25 · Assayag et al. 2026, *Bioacoustics* 35(4) · Aycock et al. 2025, *ICLR* · Bermant et al. 2019, *Sci. Rep.* · Gero et al. 2016a, *R. Soc. Open Sci.* · Gero, Whitehead & Rendell 2016b, *R. Soc. Open Sci.* · Gubnitsky et al. 2025, *Sci. Rep.* · Hersh et al. 2022, *PNAS* · Lakhotia et al. 2021, *TACL* 9 · Leitão et al. 2024, *eLife* · Mielke et al. 2019, *ACL* · O'Connor & Andreas 2021, *ACL* · Schulz et al. 2008, *Anim. Behav.* · Sharma et al. 2024a, *Nat. Commun.* · Sharma et al. 2024b, *bioRxiv* (WhaleLM) · Sinha & Chavan 2026, *arXiv:2606.16084* · Terwilliger et al. 2025, *NeurIPS workshop* · Xu et al. 2020, *ICLR*