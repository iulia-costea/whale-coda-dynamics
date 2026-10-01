# Decisions made so far

Why the plan in `docs/PLAN.md` looks the way it does. Each entry gives what was decided, why, and what was rejected. Anyone can reopen a decision by giving a reason. DEC-1 to DEC-8 followed the 2026-09-22 audit; DEC-9 to DEC-15 followed a review of the plan the same day.

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
- **Decision:** the first RQ3 test uses `CarribeanCodas.csv`, which is already in the repo. Pacific analysis comes second.
- **Why:** the two sympatric clans test Leitão et al.'s prediction directly.

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
- **WhaleLM:** keep its settings (64-dim, Adam 1e-4, etc.), not its encoder–decoder design, which predicts a single sequence rather than one coda from several inputs.

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
- **Deferred:** the stronger options (exact sign-flip test, unweighted day mean, jackknife) are in `docs/LATER.md`.
- **Why:** keep stage 1 simple, and state in the write-up that 14 days give rough CIs.

**DEC-14 — RQ3 hypothesis tested as an interaction.**
- **Decision:** test (sympatric vs allopatric) × (identity vs non-identity codas), with the fixed global `fixed-50ms-v1` tokenizer: 50 ms ICI bins over [0, 1.0) s plus `BOS`/`EOS`.
- **Why:** own-clan and transfer models must score the same target token sequence for their likelihood difference to be interpretable as transfer loss. The Caribbean vs Pacific difference is also a one-programme vs many-programmes difference; the identity contrast separates the two, assuming recording differences affect both coda kinds equally.

**DEC-15 — Tempo fallback.**
- **Decision:** if Sharma's tempo file can't be fetched, assign each coda to the nearest of the five tempo modes in Sharma et al. 2024a Fig. 3 (0.33, 0.51, 0.80, 1.02, 1.26 s), labelled "reconstructed".
- **Why:** it is closer to their scheme than rhythm alone. It approximates their clustering; it doesn't reproduce it.

## Differences from the submitted proposal (`docs/proposal.tex`)

**Content changes**
- Condition (i) is the smoothed unigram, not "majority-class floor" (DEC-9).
- The type inventory is 23/25 types (21 in the comparison set), not 27.
- Caller identities are segment-local, not identified whales.
- The response delay is redefined (DEC-1).
- RQ2's primary test is the two Assayag labellings; Sharma's scheme is secondary (DEC-3).
- RQ3 starts with EC1 ↔ EC2 and is framed as an interaction (DEC-4, DEC-14).
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
