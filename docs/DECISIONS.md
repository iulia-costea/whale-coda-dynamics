# Decisions made so far

Why the plan in `docs/PLAN.md` looks the way it does. Each entry gives what was decided, why, and what was rejected. Anyone can reopen a decision by giving a reason. DEC-1 to DEC-8 followed the 2026-09-22 audit; DEC-9 to DEC-15 followed a review of the plan the same day; DEC-16 and DEC-17 followed the first model results (2026-09-28); DEC-19 followed a review of RQ3 (2026-09-29); DEC-20 followed logistic v3's gate 1 (2026-09-29). DEC-18 is reserved for the LSTM v2 run rules.

## After the audit

**DEC-1 — Interactional state $S$ contains only what is known at the target's onset.**
- **Decision:** four turn categories, including explicit overlap. Timing is measured from the immediately preceding coda of either whale, and overlaps are timed from that coda's onset. See PLAN §3.
- **Why:**
  - The old delay was measured to the partner's last coda. For a whale's second coda in a row, that interval contained its own previous coda's duration, which predicts coda type. It also revealed the turn category outright: continuations ran to 50 s, while switches were capped at 6.76 s.
  - For overlaps, the old negative delay used the end of a partner coda that was still running when the target started. Overlapping codas are duration-matched (Sharma et al.; Assayag et al. r = .934), so this hinted at the answer on 28.5% of targets.
  - Both problems bias the RQ1 contrast.
  - The old plan contradicted itself here ("follow Assayag's definitions" vs "measure to the preceding coda").
- **Rejected:**
  - onset-to-onset intervals, which contain the previous coda's duration;
  - dropping or censoring overlap rows.
- **Kept:** the old delay, as a sensitivity field.

**DEC-2 — Leave-one-day-out cross-validation.**
- **Decision:** 14 folds, one validation day per fold, paired per-coda contrasts.
- **Why:** the single 10/2/2-day split (186 test codas) couldn't produce the confidence intervals the proposal promised.

**DEC-3 — RQ2 compares two labellings of the same codas.**
- **Decision:**
  - The primary test is `guy_coda_type` vs `decision_tree_coda_type`.
  - The secondary test is Sharma's rhythm × tempo on matched exchanges.
  - The standalone Sharma pipeline becomes a whole-pipeline robustness check.
- **Why:** the existing RQ1/RQ2 pipelines differed in labels, segmentation, splits and unknown-type handling all at once.

**DEC-4 — RQ3 starts with EC1 ↔ EC2.**
- **Decision:** the first RQ3 test uses `DominicaCodas.csv`, which is already in the repo. Pacific data comes second.
- **Why:** the two sympatric clans test Leitão et al.'s prediction directly.
- *Superseded in part by DEC-19: EC1 ↔ EC2 is now two cells of the full transfer matrix, reported descriptively.*

**DEC-5 — Two-whale exchanges only** (recorded, not new).
- **Decision:** the existing pipeline already kept only exactly-two-caller segments. This makes that explicit as the scope of the claims.
- **Why recorded:** neither the plan nor the proposal stated it, and it drops ~45% of codas.
- **Robustness:** `strict_focal_subset`.

**DEC-6 — Handle the 82.5% repetition by splitting the likelihood, not by filtering.**
- **Decision:** a repeat-own-last baseline, plus the repeat/switch split (PLAN `[P1]`).
- **Why:** evaluating only on non-repeats selects on the outcome and biases the RQ1 contrast downward.

**DEC-7 — One shared unknown-type policy.**
- **Why:** RQ1 had none, and RQ2's used different token sets for the vocabulary and the prior.
- **Amended** by DEC-10.

**DEC-8 — One shared evaluation module**, with the units in the function names.
- **Why:** nothing in the repo said "per coda" or "bits".

## After reviewing the plan

**DEC-9 — The floor model is the smoothed unigram, not majority class.**
- **Why:** majority class gives zero probability to every other type, so its log-likelihood is −∞.
- **Majority class** is kept for accuracy only.
- **Reading the repeat-own-last baseline:** falling below it means a model failed to train; matching it is a finding.

**DEC-10 — Rare types become `UNK` in training too** (threshold *k* = 2, per fold).
- **Why:** otherwise models never see `UNK`, give it near-zero probability, and a few rare test codas dominate the averages.
- **Reporting:** labellings will differ in `UNK` rates, so report those per labelling.

**DEC-11 — Neural architecture: one encoder per input source, shared output head.**
- **Scope:** relevant from phase 4.
- **Why:** the conditions can't literally share one architecture, since (i) has no input and (ii) has no sequence.
- **WhaleLM:** keep its settings (64-dim, Adam 1e-4, etc.), not its encoder–decoder design, which predicts a single sequence rather than one coda from several inputs. *Settings superseded by DEC-17: they are now chosen on the validation day.*

**DEC-12 — Main model is an LSTM; logistic regression runs first as reference and debug step.**
- **Why an LSTM:**
  - The course requires neural models, and the proposal promises LSTMs.
  - There is precedent in Assayag et al., so condition (iv) can be checked against their reported result.
- **Why logistic regression too:**
  - Its convex fit can't fail to train. That makes it the cheapest way to debug the evaluation pipeline.
  - It is the reference that separates "own history adds nothing" from "the LSTM didn't train" (`[U1]`). With ~1,500 targets, that ambiguity is the project's main risk.
  - It handles two histories plus continuous timing, which Kneser-Ney/VLMC models can't.
- **Deferred:** Kneser-Ney/VLMC and the Transformer.

**DEC-13 — Stage-1 uncertainty: day bootstrap plus a count of positive days.**
- **Deferred:** the stronger options (exact sign-flip test, jackknife) are in `docs/LATER.md`. *The unweighted day mean is now reported (DEC-16).*
- **Why:** keep stage 1 simple, and state in the write-up that 14 days give rough CIs.

**DEC-14 — RQ3 hypothesis tested as an interaction.**
- **Decision:** test (sympatric vs allopatric) × (identity vs non-identity codas), with discretisation bins fitted on the training clan only.
- **Why:** the Caribbean vs Pacific difference is also a one-programme vs many-programmes difference. The identity contrast separates the two, assuming recording differences affect both coda kinds equally.
- *Superseded by DEC-19.*

**DEC-15 — Tempo fallback.**
- **Decision:** if Sharma's tempo file can't be fetched, assign each coda to the nearest of the five tempo modes in Sharma et al. 2024a Fig. 3 (0.33, 0.51, 0.80, 1.02, 1.26 s), labelled "reconstructed".
- **Why:** it is closer to their scheme than rhythm alone. It approximates their clustering; it doesn't reproduce it.

## After the first model results (2026-09-28)

**DEC-16 — Logistic-regression reference v2: shared copy weights, L2 chosen on the validation day, log-likelihood stop rules.**
- **Decision:**
  - Conditions with $A$ add one shared weight $\beta$ to the score of the coda type equal to the whale's own last coda; conditions with $B$ add $\gamma$ for the partner's last coda. Each is a single number shared across all coda types, with the same L2 as the other weights.
  - L2 is chosen per fold and condition by validation-day log-likelihood from one grid, now {0.0001 … 1.0}, with the same rule for all conditions; ties go to the larger value. The model trained on training days only is scored on the test day.
  - Stop rules use log-likelihood only: stop and investigate if an $A$-condition's contrast against repeat-own-last, or any add-one-input contrast, has a CI entirely below zero. Accuracy is reported, not used to stop.
  - Also reported: the unweighted mean of the 14 day-level contrasts; 10,000 bootstrap replicates.
- **Why:**
  - v1 (fixed L2 = 0.01, no copy weight) could learn repetition only as a separate weight per coda type, each shrunk by L2, so rare types never learned it. Its $A$-models only matched the repeat baseline (accuracy 79% vs 82.5%, macro-F1 0.18 vs 0.41).
  - With one shared weight, repetition is learned from all targets at once: condition $A$ went from −1.363 to −1.088 bits/coda and beats the repeat baseline by +0.30 bits, positive on 14/14 days.
  - One fixed L2 can't suit conditions with very different numbers of features.
  - v2's accuracy-based stop conditions fired on differences of 3–7 codas out of 1,547; log-likelihood is the estimand.
  - v2 chose the grid's lowest value (0.001) for the $A$-models in 7–10 of 14 folds, so the grid is extended down to 0.0001; this needs a rerun (minutes).
- **Supersedes:** the fixed L2 of the first logistic-regression version and the stage-1 "fixed settings, no tuning" rule.

**DEC-17 — LSTM v2: copy weights, Assayag-style optimiser, regularisation chosen on the validation day, pilot first.**
- **Decision:**
  - Same template (one 1-layer LSTM encoder per history, small feed-forward encoder for $S$, shared output layer; an absent input removes its encoder), plus the copy weights $\beta$, $\gamma$ of DEC-16.
  - Optimiser: Adam, learning rate 1e-3, batch 32, up to 300 epochs, patience 20, restoring the best validation epoch.
  - Chosen per fold and condition on the validation day, one grid and one rule for all conditions: dropout {0, 0.2, 0.4} × weight decay {1e-5, 1e-4, 1e-3} × hidden size {32, 64}. Criterion: validation log-likelihood averaged over the 3 seeds; ties go to the smaller, more regularised setting.
  - Pilot first (conditions A and A+B+S on the three largest test days plus one small one), with per-epoch training and validation log-likelihood, before the full run.
  - Stop rules as in DEC-16, plus `[U1]`: stop if the LSTM's paired difference to logistic regression on a condition has a CI entirely below zero. Matching logistic regression is acceptable.
  - The Assayag comparison becomes a context check, not a reproduction: condition (iv) on switch/overlap targets against the majority class on that subset, and condition (iii) against their single-whale results.
- **Why:**
  - LSTM v1 used WhaleLM's settings (learning rate 1e-4, weight decay 1e-5, batch 32, at most 100 epochs). WhaleLM is a language model that gets a training signal from every token of long sequences; we get one per target (~1,100–1,500 per fold). 27 of 42 condition-$A$ runs hit the epoch cap while others stopped at epoch 6 on small validation days.
  - v1 had no copy path (the same failure as logistic v1), no dropout, and ~25k parameters per encoder for ~1,300 targets. It fell below logistic v2 on 6 of 7 conditions, and adding inputs made it worse.
  - Assayag et al. (learning rate 1e-3, 300 epochs) is the closer precedent for optimisation. Neither paper's regularisation fits our data size, so it is chosen on the validation day, like the L2 in DEC-16.
  - Their random 70/15/15 split puts codas from the same day in training and test, so their accuracies are optimistic next to our held-out days; hence a context check rather than a reproduction.
- **Supersedes:** the fixed WhaleLM settings in DEC-11 and in the first version of PLAN `[M4]`. LSTM v1 results are kept as a record (`reports/README.md`).

## After the RQ3 review (2026-09-29)

**DEC-19 — RQ3 becomes a descriptive transfer study on shared building blocks.**
- **Decision:** PLAN §8, `[R3-0]`–`[R3-7]`:
  - shared building blocks: processed codas with day groups and flags, one fixed tokeniser, one fold file, one score format;
  - the 9 × 9 transfer matrix, with own-clan and transfer scores on the same codas and from models trained on the same share of a clan;
  - a coverage breakdown of every transfer loss;
  - one exploratory overlap test within the Pacific.
- **Why:**
  - Bins fitted on each training clan (DEC-14) give the two scores in a transfer loss different symbols, so the scores can't be compared. Leitão et al. fixed their bins across clans for this reason. Fixed bins also fit nothing to any clan, which is what DEC-14's rule was for.
  - A transfer loss is only fair if both scores cover the same codas and come from models trained on similar amounts of data. Cross-fitting and averaging each source's fold models give both.
  - `[D6]` dates come in two formats; grouping by the raw string would put one recording day on both sides of a split.
  - The sympatry test can't be run with useful power. EC1 ↔ EC2 is a single, weak case: EC2 units visit from a range elsewhere, the clans don't associate, and their codas were marked by different annotators (Gero et al. 2016). Within the Pacific, 21 clan pairs give roughly 30–40% power at Leitão et al.'s effect size (r² = 0.11), and clans that overlap were often recorded in the same regions.
  - Hersh et al.'s SI (Tables S2 and S4) supplies the Pacific identity lists and overlap values the identity analysis was waiting for. EC1/EC2 have no comparable list.
  - The Pacific ↔ Pacific cells need no new code, only more cells.
  - Time: results freeze on 4 October; the report is due 11 October.
- **Kept from the plan on `rq3-cross-clan`:** the preprocessing and its audit trail, the n-gram and VLMC models, transfer loss as own-clan minus transfer score, tuning on the source clan only, both directions reported, type diagnostics within a dataset.
- **Rejected for now:** the sympatric × identity difference-in-differences as the main test; bins fitted per clan; size-matched reruns; pooled training across clans (the backup in `docs/LATER.md`).
- **Supersedes:** DEC-14; the "EC1 ↔ EC2 first" part of DEC-4.

## After logistic v3's gate 1 (2026-09-29)

**DEC-20 — Separate validation-chosen penalties for history and timing blocks.**
- **Decision:** each model that includes $S$ splits its penalty in two: one for the history blocks ($A$/$B$'s features and their copy weights $\beta$/$\gamma$), one for $S$ (turn category and timing). Both are chosen jointly per fold and condition on the validation day from the same grid; ties go to the larger values. Conditions without $S$ keep a single penalty, unchanged. Logistic v4 replaces v3 as the reference for stages 2 and 3.
- **Why:** conditional 𝒱-information (Xu et al. 2020) assumes a model can choose to ignore an input that adds nothing. A single shared penalty can't separately suit a small, possibly-uninformative block ($S$) and a large, informative one (history): in v2/v3, adding $S$ lowered validation-day log-likelihood on 10 of 14 folds even at the best available L2, so the model was structurally unable to turn $S$ off while still fitting history well. Same class of problem as v1's (DEC-16), different dimension: there it was too few coefficients for repetition; here it's one penalty doing two jobs.
- **Extends to LSTM v2 (stage 2):** weight decay splits the same way (history encoders and their output weights vs the $S$ encoder and its output weights), chosen jointly with dropout and hidden size on the validation day. The $S$ grid includes a value strong enough to verify $S$ can be switched off in practice.

## Differences from the submitted proposal

**Content changes**
- Condition (i) is the smoothed unigram, not "majority-class floor" (DEC-9).
- The type inventory is 23/25 types (21 in the comparison set), not 27.
- Caller identities are segment-local, not identified whales.
- The response delay is redefined (DEC-1).
- RQ2's primary test is the two Assayag labellings; Sharma's scheme is secondary (DEC-3).
- RQ3 is a descriptive transfer matrix over nine clans with one exploratory overlap test, not a sympatry test (DEC-19).
- The claim is scoped to two-whale exchanges (DEC-5).

**Added since the proposal**
- Leave-one-day-out folds and day-level uncertainty (DEC-2, DEC-13).
- Logistic regression as a reference model next to the LSTM (DEC-12).
- The persistence tests `[P1]`–`[P4]`.

**Typos and citations**
- `\maketitle` missing.
- §4 says "three outcomes" but lists four.
- "each code" → "each coda".
- Cite Leitão et al. as eLife 2024, not arXiv 2023.
- §6 mentions "identification of varieties", which is no longer in the design.