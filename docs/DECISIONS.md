# Decisions made so far

Why the plan in `docs/PLAN.md` looks the way it does. Each entry gives what was decided, why, and what was rejected. Anyone can reopen a decision by giving a reason. DEC-1 to DEC-8 followed the 2026-09-22 audit; DEC-9 to DEC-15 followed a review of the plan the same day; DEC-16 and DEC-17 followed the first model results (2026-09-28); DEC-19 followed a review of RQ3 (2026-09-29); DEC-20 followed logistic v3's gate 1 (2026-09-29); DEC-21 followed LSTM v2's full run (2026-09-30); DEC-22 followed a review of RQ2 as a token-inventory question (2026-10-03); DEC-23 and DEC-24 followed RQ2's v4 runs (2026-10-03); DEC-25 added the own-history length sweep to the persistence tests (2026-10-03); DEC-26 and DEC-27 followed the first RQ3 results (2026-10-03). DEC-28 added the switch-structure test after the results freeze (2026-10-04). DEC-30 added the duration lattice after the results freeze (2026-10-04). DEC-31 added the ornament-merge test of DEC-28's switch structure after the results freeze (2026-10-05). DEC-18 is reserved for the LSTM v2 run rules.

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

**DEC-3 — RQ2 compares two labellings of the same codas.** *Tests superseded by DEC-22.*
- **Decision:**
  - The primary test is `guy_coda_type` vs `decision_tree_coda_type`.
  - The secondary test is Sharma's rhythm × tempo on matched exchanges.
  - The standalone Sharma pipeline becomes a whole-pipeline robustness check.
- **Why:** the existing RQ1/RQ2 pipelines differed in labels, segmentation, splits and unknown-type handling all at once.

**DEC-4 — RQ3 starts with EC1 ↔ EC2.**
- **Decision:** the first RQ3 test uses `CarribeanCodas.csv`, which is already in the repo. Pacific data comes second.
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

**DEC-15 — Tempo fallback.** *Superseded by DEC-22.*
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

## After LSTM v2's stage 3 (2026-09-30)

**DEC-21 — The LSTM becomes a residual network on top of a frozen logistic v4; logistic v4 gives the headline lattice.**
- **Decision:**
  - Model, per fold and condition $c$: $z(x) = z^{\mathrm{LR}}_c(x) + \sum_{s \in c} U_s\,h_s(x)$.
    - $z^{\mathrm{LR}}_c$ are logistic v4's logits for that fold and condition, precomputed from its committed fits (`data/results/rq1_logistic_v4/models.jsonl`) and added as a fixed offset. They never train.
    - $h_s$ is block $s$'s encoder (DEC-11 template: a 1-layer LSTM per coda history, up to 10 codas; a small feed-forward encoder for $S$). $U_s$ is its bias-free output projection, initialised at 0. No output bias and no copy weights outside the offset: both are in $z^{\mathrm{LR}}_c$.
  - Each block is either on or absent. Absent means the block isn't built (structurally $U_s = 0$). With every block absent the model is exactly logistic v4.
  - Training: AdamW with decoupled weight decay 0.1, learning rate 1e-3, batch 32, up to 300 epochs, patience 20, restoring the best validation epoch. Epoch 0, the untrained model (= logistic v4), is a candidate.
  - Candidates per fold and condition: hidden size {16, 32, 64} × dropout {0, 0.2, 0.4} × each block on or absent (at least one on), plus logistic v4 itself. Seeds 42–46.
  - Selection on the validation day:
    1. per-target validation log-likelihood, averaged over the 5 seeds;
    2. find the best candidate;
    3. choose the simplest candidate whose mean paired difference to the best is within one standard error, where the SE is the standard deviation of the per-target paired differences divided by $\sqrt n$. Simplicity order: fewer blocks on, then smaller hidden size, then more dropout.
    - This is a selection rule, not a reported CI; reported CIs stay day-clustered. The SE is target-level on every fold because 9 of the 14 validation days hold fewer than 10 exchanges (two hold only 2), too few to resample by exchange.
  - Reporting: logistic v4's lattice is the headline. The residual network is the enlarged-family check: its lattice next to v4's, plus each condition's residual gain (residual network minus v4, paired, day-bootstrap CI), which is `[U1]`'s new form.
- **Why:**
  - A contrast is only fair if every condition uses the same model family and selection rule and gets its own optimum within it (best held-out log-likelihood, approximated on the validation day). Logistic v1–v3 (one L2 penalty) and LSTM v2 (one early-stopping point and other knobs shared across blocks) both broke the "own optimum" part in a way that favoured smaller conditions and pushed add-input contrasts down.
  - LSTM v2's full run (2026-09-30) was reliably below logistic v4 on 5 of 7 conditions (A, B, A_B, A_S, A_B_S) and lower in point estimate on all 7; adding $B$ made it worse (`A_B − A`, `A_B_S − A_S` CIs below zero). Likely causes, from reasoning about training dynamics rather than an ablation:
    - the copy weights barely trained ($\beta$ 0.06–0.46 vs logistic v4's 2.3–3.5): Adam moves one scalar slowly while the encoders learn per-type repetition faster, and early stopping ended training first;
    - there was no "learn nothing" candidate;
    - one stopping epoch, one dropout rate and one history decay served all blocks.
    Pooled validation days and 3-seed selection did not help (`docs/EXPERIMENTS.md`).
  - A plain LSTM can't compute logistic v4 exactly (the 3-lag window and additivity need saturated gates), so the logistic path is added explicitly. It stays frozen: training it would move it off its validated penalty, and the residual would mix logistic regression refitting itself with what the network adds.
  - Optional ignorance (Xu et al. 2020) now holds for the training procedure, not only for the family: the off state exists (absent), training can return to it (epoch 0), and validation can select it.
  - The extra compute goes to seeds (3 → 5) and one smaller hidden size, not a wider grid: more candidates make a lucky validation winner more likely, and the number of candidates grows with condition size (1 for none, 64 for A_B_S), which biases against richer conditions.
- **What it doesn't guarantee:** a contrast is v4's contrast plus the difference between the two conditions' residual gains, which can be negative, so contrasts aren't monotone by construction. On test days the residual network can fall below v4 where validation and test days disagree; a `[U1]` CI entirely below zero now means selection overfitting to validation days, not a training failure.
- **Not done:** cross-fitted logistic offsets for training targets (the three March 2015 days hold 58% of targets, so leaving one out weakens exactly the offsets that dominate training); a weight-decay grid; any "beyond 3 lags" claim.
- **Supersedes:** DEC-17's LSTM architecture, optimiser and grid, and DEC-12's "main model is an LSTM". Logistic regression gives the headline lattice; the residual LSTM is the enlarged-family check. LSTM v2 results are kept as a record.

## After reviewing RQ2 as a token-inventory question (2026-10-03)

**DEC-22 — RQ2 varies the token inventory on fixed data: scheme (named types vs Sharma's rhythms) × resolution (with or without Sharma's tempo classes), on logistic v4.**
- **Decision:** PLAN §7, `[R2-0]`–`[R2-7]`:
  - Only the labels change. Targets, exchanges, history IDs, $S$, the 14 folds, the UNK policy, logistic v4 and its selection rule are RQ1's. Each inventory labels every retained `[D1]` coda from that coda's own intervals and duration.
  - Inventories: A `guy_coda_type` (RQ1's existing run); B `decision_tree_coda_type`; C Sharma rhythm; D rhythm × tempo; E `guy_coda_type` × tempo. D′ and E′ swap in the tempo classes of Sharma's `tempos-dict.p` as a sensitivity.
  - Rhythm: Sharma's rule, recovered and verified (`[R2-1]`): click positions over total length, nearest of the 18 templates in `mean_codas.p` with the same click count; fewer than 3 or more than 10 clicks → rhythm 17. It reproduces `rhythms.p` on all 3,840 `[D2]` rows.
  - Tempo (`[R2-2]`): duration cut at the minima of a Gaussian KDE (bandwidth 0.035 s) of `[D6]`'s EC1 durations, as in Sharma et al.'s `2-tempo.ipynb`: 0.45, 0.61, 0.93, 1.08 s. Recomputed from the committed `CarribeanCodas.csv`, it gives the notebook's minima and the five modes of their Fig. 2A and 3.
  - Reporting (`[R2-3]`–`[R2-6]`): per inventory the RQ1 contrasts as shares of total predictability, share differences against A with a paired day bootstrap, the pattern of signs, UNK shares; for the nested pairs A → E and C → D an exploratory split into a type (rhythm) term and a tempo term.
  - Model: logistic v4 only.
- **Why:**
  - Whale data come without given units, so turning a coda into a token is a discretisation the analyst chooses (Andreas et al. 2022). The closest NLP practice is discrete-unit spoken language modelling, which crosses unit encoders with codebook sizes and keeps the rest fixed (Lakhotia et al. 2021). Scheme × resolution is that design on our data, and it makes RQ2 the project's tokenisation question.
  - DEC-3's primary test is too weak to show much: 75 of 1,547 targets change label and 17 flip between repeat and switch. It stays, as inventory B and one sentence.
  - DEC-3's secondary test needed exchanges whose codas all match `[D2]` (121 of 157). Applying Sharma's rules to `[D1]`'s own intervals keeps every target and fold.
  - D changes the task most (42 labels among the targets, top label 40% instead of 57%, 1,142 repeats instead of 1,277) and comes from an independent publication. E isolates the tempo dimension within our types; C isolates the scheme.
  - Raw log-likelihood can't be compared across inventories: they keep different information about the same codas, so the text-NLP fix (score the same string per character) doesn't apply. Lakhotia et al. (2021, p. 1337) make the same point for discrete speech units. Shares and patterns can be compared, and within a nested pair the chain rule splits the likelihood exactly.
  - KDE minima as the primary tempo rule: it is the method the paper describes (KDE modes, Fig. 2A) and is reproducible from committed data. The classes in `tempos-dict.p` (boundaries 0.392, 0.594, 0.907, 1.091 s) come from a step the notebook doesn't document; they agree with the KDE classes on 1,511 of 1,547 targets, so they serve as the boundary sensitivity.
  - Logistic v4 only: it gives the headline lattice, the residual LSTM found nothing beyond it (DEC-21), and results freeze on 4 October.
- **Rejected:**
  - Type × rhythm × tempo: rhythm mostly restates type (1+1+3 is rhythm 2 in 882 of 885 targets); the extra splits fall mostly where the two classifications disagree, and 30 of its 55 labels would have fewer than 5 targets.
  - Ornaments (`ornaments.p`): 151 of 3,840 `[D2]` codas, 43 matched RQ1 targets, and many already have their own named type (41 of the ornamented codas matched to `[D1]` are `6i`). Sharma et al.'s ornament findings concern turn-taking, not the next coda's type.
  - Rhythm only as *the* alternative scheme: it is one cell of the grid. It isn't simply coarser than the named types, as DEC-3 said: it merges some types and splits others (1,215 repeats, fewer than A's 1,277).
  - Nearest-mode tempo (DEC-15): its cuts fall halfway between the modes, not at the density minima.
  - Segmentation proper (a coda as two tokens; clicks as tokens): a different question, in `docs/LATER.md`.
- **Supersedes:** DEC-3's two tests (the standalone Sharma pipeline keeps its role as a whole-chain check); DEC-15.
- **Addendum (2026-10-03, decided before any RQ2 fit):** the RQ2 logistic runs set `skip_nonconverged_candidates` (off by default, so RQ1 is unchanged): a grid candidate that fails `fit()`'s convergence check is left out of the validation-day selection and counted per fold and condition, and if every candidate for a fold and condition fails, that inventory's run fails and is reported as such. No candidate failed in any RQ2 run.

## After RQ2's v4 runs (2026-10-03)

**DEC-23 — Logistic v5: every input block can be absent, chosen on the validation day; stop rules use exact sign-flip tests with Holm's correction.**
- **Decision:**
  - v5 is v4 with optional ignorance by selection. For each fold and condition $c$, the candidates are v4's chosen fits of every sub-condition $s \subseteq c$ (each of $A$, $B$, $S$ on or absent; all absent = condition (i), the smoothed unigram). The candidate with the highest validation-day log-likelihood is chosen; each sub-condition's penalties were already chosen by v4's rule. Exact ties go to fewer blocks. The chosen candidate's test-day predictions become $c$'s v5 predictions.
  - No new fits: v5 is computed from v4's committed `models.jsonl`, `l2_selection.jsonl` and `predictions.jsonl`; condition (i) is the fold's prior scored on the validation day.
  - Computed for every v4 run: RQ1 (primary and old-delay sensitivity) and the six RQ2 inventories.
  - RQ2's comparison uses v5 for all seven inventories, A included, so it compares like with like; v4's comparison stays reported as the record.
  - RQ1's headline stays logistic v4 (no change to its reporting): it passed its stop rules, and nothing about RQ1 motivates a change. RQ1's v5 numbers are a robustness line. If RQ1's conclusions differ between v4 and v5, report both and say so. The residual LSTM (DEC-21) stays defined on v4 and remains the enlarged-family check.
  - Sensitivity: DEC-21's one-SE rule instead of the maximum (the simplest candidate within one standard error of the best; SE of the per-target paired validation differences; simpler = fewer blocks).
  - Stop rules, replacing the CI-below-zero criterion of DEC-16: per run, the 16 checks (12 add-one-input contrasts, 4 $A$-conditions against repeat-own-last) each get an exact one-sided sign-flip test on the day-level sums of per-coda differences (alternative: mean below zero; all $2^{d}$ sign assignments over the $d$ test days; statistic: token-weighted mean). Holm's correction across the 16 checks at $\alpha = 0.05$; a rule fires if its Holm-adjusted $p < 0.05$. Day-bootstrap CIs stay the reported intervals. The three `[M2]` contrasts also get an exact two-sided sign-flip $p$.
  - Every run reports both rules (old and new), for v4 and v5.
- **Why:**
  - Under v4, stop rules fired for five of the six RQ2 inventories, all on contrasts that add $S$ and all within 0.1 bits of zero; RQ1's own timing CI ended at +0.0014. Conditional 𝒱-information assumes the model can ignore an input (Xu et al. 2020). A penalised $S$ block can be shrunk but not removed, and its penalty is chosen on one small validation day. DEC-21 solved this for the residual LSTM with absent blocks; v5 does the same for logistic regression. On validation days every add-one-input contrast is now ≥ 0 by construction, so a negative test-day contrast means selection noise between validation and test days, not an unusable input.
  - Every sub-condition is already fitted under the same rule, so v5 needs no refit and leaves v4's fits untouched.
  - The old rule had no multiplicity control (16 checks per run, 112 over seven runs) and used percentile bootstrap intervals, which are too narrow with 14 clusters. The exact sign-flip test has been in `docs/LATER.md` since 2026-09-22 and is valid with few clusters; Holm controls the family-wise error within each run.
  - Both changes were decided after seeing v4's RQ2 stop rules. That is why both models and both rules are reported for every run.
- **Known limits:**
  - Conditions with $S$ choose among 81 penalty pairs, those without among 9, so comparing validation maxima slightly favours keeping $S$ (a larger grid gives a luckier best). The one-SE sensitivity counters this.
  - The sign-flip test weights days by size, so the three March 2015 days (58% of targets) dominate it.
- **Rejected:**
  - Extending the $S$ penalty grid upward: a stronger penalty is still not absence, and the ceiling would move again with vocabulary size.
  - Refitting with absent blocks inside the grid search: the same result at more compute, since every sub-condition is already fitted.
  - Exempting add-$S$ contrasts from the stop rules (rejected in DEC-20).
- **Supersedes:** DEC-16's CI-below-zero stop criterion (its log-likelihood-only principle stands); the sign-flip item in `docs/LATER.md`.

**DEC-24 — Partner-tempo ablation: does the partner's tempo or the partner's rhythm predict a coda's tempo?**
- **Question:** RQ2's exploratory split found that the partner's codas predict a coda's tempo class beyond the whale's own history and timing (A → E: +0.039 bits/coda, CI [0.022, 0.061]). Is that tempo matching (the partner's tempo), or does the partner's pattern carry information about the coda's tempo beyond the partner's own tempo?
- **Decision:**
  - A dedicated model for the second chain-rule factor: the target is the coda's tempo class (the 5 KDE classes, DEC-22), given the coda's own Sharma rhythm (one-hot, always present). Rhythm is normalised by duration, so the partner's rhythm and tempo are separable by construction; named types are not (5R1, 5R2 and 5R3 differ in tempo).
  - Blocks:
    - own history, always present: own last 3 tempo classes and last 3 rhythms (one-hot), plus a shared own-tempo copy weight $\beta$ on $\mathbb 1[\text{tempo} = \text{own last tempo}]$;
    - partner tempo: the partner's last 3 tempo classes (one-hot) plus a shared copy weight $\gamma$ on $\mathbb 1[\text{tempo} = \text{partner's last tempo}]$;
    - partner rhythm: the partner's last 3 rhythms (one-hot);
    - $S$.
  - Fitting: v4's procedure (every non-$S$ block shares one penalty, $S$ has its own, both from the same grid on the validation day) and v5's absent candidates for partner tempo, partner rhythm and $S$.
  - Conditions: base (rhythm + own + $S$), + partner tempo, + partner rhythm, + both; the same four without $S$ (v5's candidates); rhythm alone as the floor.
  - Contrasts, in bits per coda of the tempo class, paired, with day-bootstrap CI and DEC-23's sign-flip $p$: partner tempo alone (+tempo − base); partner rhythm alone (+rhythm − base); tempo beyond rhythm (both − +rhythm); rhythm beyond tempo (both − +tempo). Stop rules per DEC-23 on this family's add-one-block contrasts.
  - Descriptive breakdown of the four contrasts by the target's turn category (overlap, switch, self-continuation); no tests.
  - Secondary: the same with `guy_coda_type` in place of rhythm.
  - Same 1,547 targets, folds and UNK policy (rhythms and types; tempo classes have no UNK). Partner codas are those completed before the target's onset, as in `[D5]`.
- **Reading:**
  - Tempo beyond rhythm > 0 with rhythm beyond tempo ≈ 0: tempo matching.
  - Rhythm beyond tempo > 0: the partner's pattern carries information about the coda's speed beyond the partner's own speed, for instance through a shared behavioural context. No claim about meaning.
  - Nulls read "not detectable at this data size" (14 days).
- **Why:** RQ2's split can't separate the two. E's partner labels carry type and tempo jointly, and named types partly encode tempo. Modelling the chain rule's tempo factor directly, given the coda's rhythm, makes every contrast a difference in bits of the same five-class target, and the conditions nest.
- **Not done:** continuous durations (rubato, `docs/LATER.md`); separate penalties per block.
- **Addendum (2026-10-03, decided before any fit):** implementation choices for `tempo_factor.py`.
  - Condition names: `own`, `own_T`, `own_P`, `own_S`, `own_T_P`, `own_T_S`, `own_P_S`, `own_T_P_S` (T = partner tempo, P = partner pattern: rhythm, or type in the secondary run) and `floor`; the base is `own_S`.
  - Stop-rule family: 13 add-one-block contrasts, the 12 pairs among the eight conditions and own history added to the floor (DEC-24 lists own history as a block and the floor as a condition). No repeat-own-last-tempo check: DEC-24 asks for none. DEC-23's rule on v5 (max) governs each run, as for the rest of RQ2; the own fits (no absent candidates) and the one-SE variant are reported beside it, each under both rules.
  - Vocabularies: one per factor. Tempo: the five KDE classes, no UNK. Rhythm (type): RQ1's policy per fold (`fit_vocabulary` on the training days' codas, *k* = 2), shared by the coda's own label and every lag of both whales, as RQ1's encoder shares one vocabulary between $A$ and $B$. The run checks it equals inventory C's (A's) fold vocabulary.
  - `skip_nonconverged_candidates` on, as in every RQ2 run (DEC-22 addendum).
  - The penalty search is logistic v4's own code, `logistic.grid_search_l2`, factored out of `select_l2_and_fit` with no change in behaviour (committed fits refit bit for bit).

## Persistence tests (2026-10-03)

**DEC-25 — Own-history length sweep (`[P2]`): logistic v4 with 1 to 9 own lags.**
- **Decision:**
  - Logistic v4 unchanged except the number of own-history lags, $k \in \{1, \dots, 9\}$. Partner history stays at 3 lags; copy weights, penalties, grid, selection rule, folds, targets and RQ1's labels (`guy_coda_type`) stay as they are. The encoder gets a per-field lag order; with the default it must reproduce v4 (up to the cross-machine numerical noise in repo `CLAUDE.md`, about 1e-6). All $k$, including $k = 3$, are fitted in the same environment, and contrasts use that $k = 3$ fit as the reference.
  - Conditions: $A,B,S$ (primary; RQ1's full model) and $A$ (own history alone, comparable to the single-stream literature).
  - Contrasts, per condition, paired per coda, with day-bootstrap CI, unweighted day mean and positive days: $\mathrm{LL}(k) - \mathrm{LL}(3)$ for every $k \neq 3$, each split into repeat and switch parts (`[P1]`).
    - Primary, pre-specified: $\mathrm{LL}(9) - \mathrm{LL}(3)$ in $A,B,S$: does own history beyond 3 codas add usable information, given the partner and timing?
    - Secondary: $\mathrm{LL}(1) - \mathrm{LL}(3)$: does memory beyond the last coda matter?
    - The curve over $k$ is descriptive.
  - Calibration: the residual LSTM's planted-signal protocol (`residual_planted.py`: lag-6 own-coda copies, $f \in \{0.1, 0.2, 0.4\}$, same eligibility, planting and seeds, condition $A$), scored with logistic regression at $k = 9$ against $k = 3$, both refitted on the planted data. Report the oracle comparison, the recovery ratio and its bootstrap interval as in the residual LSTM's planted-signal report (`reports/rq1_lstm_residual_sensitivity.md`), and a bound (the real-data $A$ contrast $\mathrm{LL}(9) - \mathrm{LL}(3)$'s upper CI divided by $r$'s lower bound) only where $r$'s lower bound is above zero.
- **Why:**
  - The literature disagrees on how far back structure reaches: up to 8 codas (WhaleLM), 1–2 (Assayag et al.), second order (Sinha & Chavan). Logistic v4 sees 3 own lags. The residual LSTM saw 10, but its planted run gave no bound: it did not recover a rare lag-6 pattern at this data size.
  - Logistic regression with explicit lags is the more sensitive device for this one question: it doesn't have to discover where to look. Recommended as a must-have for the persistence argument on 2026-09-28; not run until now.
  - RQ1's headline model is v4, so the sweep uses v4 (not DEC-23's v5) and its stop rules as they stood for RQ1.
- **Reading:**
  - Positive primary contrast: own history beyond 3 codas carries usable information beyond partner and timing. Still not grammar: a stable repertoire or a slowly changing behavioural state also produce it (Gero, Whitehead & Rendell 2016b; Aluma et al. 2026). Say whether it lives in the repeat or the switch part.
  - Null: "no detectable information beyond 3 own codas for this model at this data size", with the calibration's bound, if any, saying what size of lag-6 pattern would have been detected.
- **Not done:** partner-history length; v5 for the sweep; more than 9 lags (`history_limit` is 10); a lag-6 copy weight (too tailored to the planted pattern).
- **Gate amended** after the k = 3 gate (2026-10-03, before any k = 9 log-likelihood, calibration or contrast number was seen; the k = 9 run's log line showed only its run time and that no DEC-16 stop rule fired): v4's committed fits are reproducible only to ~1e-4 in probability at the smallest penalties (optimizer tolerance gtol 1e-6; the BLAS thread count alone moves them by ~1e-4), while penalties and mean log-likelihoods reproduce (all 98 penalties identical; mean LL within 1e-6). The gate now requires identical chosen penalties (at most one differing) and every condition's mean LL within 1e-5 bits/coda of v4's; probability differences are reported (the largest, and which fits exceed 1e-5), not enforced. The gate checks the code path; every contrast compares fits from one environment.

## After the first RQ3 results (2026-10-03)

**DEC-26 — The identity split uses the matrix models' scores.**
- **Decision:** `[R3-6]` splits each Pacific target coda's loss from the `[R3-4]` models (one per clan, all codas) by its own clan's identity list. It no longer trains separate identity and non-identity models. Separate models become a robustness check, if time. This records what `rq3/experiments/identity.py` already does ("models are never retrained").
- **Why:**
  - It asks how well one clan's whole repertoire predicts another clan's identity and non-identity codas.
  - Clans differ in their coda mix (identity codas are 11% of SH's codas and 63% of PO's), but that works mostly at the clan level, which the pair-level reading removes.
  - Separate identity models would learn only one or two coda types for SH, RI and FP.
- **Rejected:** separate models (Leitão et al.'s design) as the primary analysis. It costs a rerun, and the overlap test is exploratory.

**DEC-27 — RQ3 adds the pooled multilingual experiment `[R3-8]`, with an LSTM.**
- **Decision:** "Pooled training across clans" moves from `docs/LATER.md` into PLAN §8 as `[R3-8]`:
  - one LSTM on all clans, with clan tags and size-balanced sampling;
  - leave one clan out;
  - with and without the tag;
  - fine-tuning against training from scratch at several data sizes.

  Own-clan LSTMs are the comparison, and the n-gram is the reference they must match.
- **Why:**
  - It's the standard multilingual-NLP experiment (Liu et al. 2020), so RQ3 maps onto the course directly.
  - The course asks for neural models trained by backpropagation, and RQ3 had none.
  - Compute is small (about 137,000 intervals).
- **Rejected:**
  - comparing the pooled LSTM with the n-gram, which would mix pooling with a change of model family;
  - pretrained mBART weights, which were trained on human text.

  Size-matched reruns stay in LATER.

## Switch structure (2026-10-04)

**DEC-28 — Switch structure (`[P5]`): a collapsed view of own history and the active repertoire, on logistic v4.**
- **Decision:**
  - Logistic v4 unchanged (labels `guy_coda_type`, targets, folds, UNK policy, $S$, penalties, grid, selection rule, copy weights), plus shared class-feature weights computed from the whale's own earlier codas in the exchange. Each is one scalar, penalized with the history penalty like the copy weights, and added to the logits of the classes it marks:
    - run length: $\beta_r \log r$ on the class of the own last coda, where $r$ is the number of identical own codas ending the history;
    - distinct order: $\delta_1$ on $d_1$, the most recent own coda type different from the current one; $\delta_2$ on $d_2$, the most recent own type different from both the current one and $d_1$;
    - repertoire: $\rho \log(1 + n_c)$ on every class $c$, where $n_c$ counts the whale's earlier codas of type $c$ in the exchange, excluding the current run.
  - The features use the whale's whole earlier stream within the exchange (its codas with an earlier onset), not only the 10 codas stored in `A_history` (`history_limit`), with labels mapped by the fold's vocabulary exactly as `A_history` is. They exist only in conditions with $A$; a row without the needed history gets zeros.
  - Five nested variants, all fitted in one environment: R (v4), L (R + run length), P (L + repertoire), D (L + distinct order), PD (L + repertoire + distinct order). R is the reference for every contrast and must reproduce the committed v4 run under DEC-25's amended gate (chosen penalties identical, at most one differing; each condition's mean log-likelihood within 1e-5 bits/coda).
  - Contrasts in $A,B,S$ (RQ1's full model) and $A$, paired per coda, with day-bootstrap CI, unweighted day mean, positive days and the repeat and switch parts (`[P1]`):
    - Primary, pre-specified: $\mathrm{LL(PD)} - \mathrm{LL(P)}$ in $A,B,S$, with the exact two-sided sign-flip $p$: does the order of the whale's earlier switches add usable information beyond which types it has used and how often?
    - Secondary: $\mathrm{LL(PD)} - \mathrm{LL(R)}$ in $A,B,S$, with the two-sided sign-flip $p$: does the collapsed view of own history add anything to v4?
    - Descriptive: L − R, D − L, P − L, PD − D; the fitted $\beta_r$, $\delta_1$, $\delta_2$, $\rho$ per fold.
  - A descriptive table without models, with day-bootstrap intervals: repeat rates by turn category; what the whale produces when its own and the partner's last codas differ; switches landing on the partner's type and on $d_1$, each against day-level chance; the return rate by the number of other types already used; the run-length effect pooled and within whale and exchange.
  - Stop rules: DEC-23's rule on every variant's 16 checks, with DEC-16's reported beside it. An added-feature contrast whose interval lies entirely below zero is a fitting flag and is not interpreted.
- **Why:**
  - 82.5% of targets repeat the whale's own last coda, so in long runs the most recent different coda lies beyond v4's 3 lags (808 of 1,547 targets are in runs of 3 or more; for 113 it lies beyond the 10 stored codas). The `[P2]` sweep added raw lags, which can't share information across lag positions. Discrete-unit speech language models remove repeated units before modelling (Lakhotia et al. 2021); this is the analogous view of own history.
  - Descriptively, when a whale switches and has an earlier different coda (168 switches), it returns to $d_1$ 47.6% of the time against 26.8% expected from day-level frequencies. v4 can express this only through per-type lag-2 and lag-3 weights, the weakness the copy weights fixed for repetition (DEC-16). A shared weight is the parsimonious form.
  - A return is nearly forced when the whale has used only one other type (110 of the 168 switches), so the order test needs the repertoire control. Only 58 switches have two or more other types to choose between.
  - Added after the results freeze (2026-10-04) as a final run; the numbers above are descriptive and the run recomputes them.
- **Reading:**
  - Positive primary: the order of earlier switches carries information beyond the active repertoire (for example A→B→A alternation). Not grammar. Say whether it lives in the switch part.
  - Repertoire (P − L) positive, primary null: switch choices follow the whale's active repertoire, consistent with stable individual or unit repertoires (Gero, Whitehead & Rendell 2016b).
  - Nulls read "not detectable at this data size". The primary rests on few discriminating switches, so its null is weak. No calibration is run.
- **Not done:** a collapsed partner history; v5 or the residual LSTM with these features; a planted-signal calibration; more than two distinct types.

## Duration lattice (2026-10-04)

**DEC-30 — Duration lattice (`[P6]`): RQ1's lattice with a type-free target, the coda's log duration.**
- **Question:** every RQ1 and RQ2 result depends on a coda inventory. Does RQ1's pattern (own history informative; partner and timing not detectable) hold for a property every coda has without categorisation, its duration? And does the partner predict a coda's duration, or its tempo within its type, beyond the whale's own history (rubato and duration matching, Sharma et al. 2024a; Assayag et al. 2026)?
- **Decision:**
  - RQ1's comparison targets (1,547), its 14 folds with their validation days, exchanges, $S$ and `[D5]` are unchanged. The target is $y = \ln d$, $d$ = the coda's duration (`offset_s − onset_s`).
  - Always-present inputs, in every condition including the floor: the target's click count $n$ (`nClicks` of its raw row, `source_row_index` in `data/raw/dswp-2014-2016-retagged-maor-2025.csv`) as $\ln n$, plus one-hots for the counts with at least 10 training targets in the fold. Conditioning on $n$ makes the lattice about how long a coda is given how many clicks it has, as DEC-24 conditions tempo on rhythm.
  - Blocks:
    - $A$: for lags 1–3 of the whale's own history, the coda's $\ln d$ and $\ln n$, each standardised on its non-missing training values and 0 when missing, plus a missing indicator per lag (constant columns dropped);
    - $B$: the same for the partner's last 3 codas completed before the target's onset (`[D5]`);
    - $S$: v4's $S$ features (turn-category one-hots; gap and overlap elapsed, raw and log1p, standardised on training rows).
  - Model: ridge regression on $y$ standardised on the fold's training targets. Objective: mean squared error over training rows $+\ \lambda_h \lVert w_h \rVert^2 + \lambda_S \lVert w_S \rVert^2$, where $A$ and $B$ share $\lambda_h$ and $S$ has $\lambda_S$; the intercept is unpenalised and the always-present inputs get a fixed $10^{-4}$. Penalties come from v4's grid (nine values, 1e-4 to 1; the 81 pairs for conditions with $S$), chosen on the validation day by held-out log density. Variance $\hat\sigma^2$ = the mean squared training residual. A coda's score is $\log_2 \mathcal N(y; \hat y, \hat\sigma^2)$ on the original scale of $y$, in bits per coda; contrasts are differences of scores, so the unit of $y$ cancels.
  - The eight conditions over $A$, $B$, $S$ (`[M1]`); the floor has the always-present inputs only.
  - Two variants: F (as above, type-free) and T (F plus the target's `guy_coda_type` one-hot as an always-present input, labels and UNK per `[M6]` with the fold's vocabulary). F asks about a coda's duration; T about its duration within its type. History inputs are type-free in both.
  - Contrasts, paired per coda, with day-bootstrap CI (10,000 replicates), unweighted day mean and positive days:
    - Primary, pre-specified: the partner contrast $\mathrm{LL}(A,B,S) - \mathrm{LL}(A,S)$ in F, with the exact two-sided sign-flip $p$.
    - Secondary, each with the exact two-sided sign-flip $p$ and Holm's correction across the three: own $\mathrm{LL}(A,B,S) - \mathrm{LL}(B,S)$ and timing $\mathrm{LL}(A,B,S) - \mathrm{LL}(A,B)$ in F; the partner contrast in T.
    - Descriptive: every lattice contrast in both variants; the three RQ1 contrasts by turn category and by whether the target repeats the whale's own last type (no tests); fold summaries of the weights on own and partner lag-1 $\ln d$; residual diagnostics (skewness, excess kurtosis, the share of held-out residuals beyond $3\hat\sigma$); a v5 line (DEC-23's absent candidates applied to these fits, no new fits).
  - Calibration (F, partner contrast only): plant a partner effect by replacing $y$ with $y + f\,\hat\sigma_{AS}\,z_{b1}$ in every split, where $z_{b1}$ is the standardised $\ln d$ of the partner's last coda (0 when missing) and $\hat\sigma_{AS}$ the fold's training residual SD of condition $A,S$, for $f \in \{0.1, 0.2, 0.3\}$ (at most about 0.007, 0.03 and 0.06 bits/coda); refit everything and report the partner contrast with its CI for each $f$. The smallest $f$ whose interval excludes zero is the detection bound.
  - Stop rules: DEC-23's rule on each variant's 16 checks (the 12 add-one-input contrasts, and the 4 $A$-conditions against copying the own last coda's $\ln d$, scored as a Gaussian with the training mean squared difference as variance), with DEC-16's reported beside it. An add-one-input contrast whose interval lies entirely below zero is a fitting flag and is not interpreted.
- **Why:**
  - It removes the categorisation from RQ1: if the pattern were an artefact of how codas are typed, a type-free target would show it.
  - Duration is where coordination between whales has been reported (rubato; duration matching of overlapping codas, Sharma et al. 2024a; Assayag et al. 2026). RQ1 sees duration only through types and RQ2 through five tempo classes; DEC-24 found the partner's tempo class not detectable.
  - Descriptively (2026-10-04), within type the target's log duration correlates 0.568 with the own last coda's and 0.369 with the partner's last; the lattice asks whether any of the partner's part is not redundant with own history and $S$. The run recomputes these.
  - Ridge regression with a Gaussian held-out density is the simplest predictive family for a continuous target; held-out log-density differences are conditional 𝒱-information (Xu et al. 2020) in bits per coda, as in RQ1.
  - Added after the results freeze (2026-10-04) as a final run.
- **Reading:**
  - F: own > 0 with partner and timing not detectable: RQ1's pattern doesn't depend on the coda types.
  - Partner > 0 in F but not in T: the partner predicts which kind of coda (its type-level duration) comes next, not its tempo within type.
  - Partner > 0 in T: the partner's previous codas predict this coda's tempo within its type beyond own history and $S$; consistent with tempo coordination, but onset-available inputs can't separate coordination from both whales responding to the same context.
  - Nulls read "not detectable at this data size", with the calibration bound.
- **Limits:** the partner's coda that overlaps the target isn't complete at the target's onset and isn't an input, so duration matching within overlaps is outside this design; Gaussian residuals (heavy tails affect every condition but can inflate differences; diagnostics reported); bits per coda of a continuous target are not comparable with RQ1's bits per coda of a type; ornaments (an extra final click, Sharma et al. 2024a) aren't flagged: conditioning on click count groups an ornamented coda with codas of its click count, and in T many ornamented codas have their own type (6i, DEC-22).
- **Not done:** heavy-tailed or nonlinear models; the residual LSTM; inter-click-interval targets; other inventories in T.
- **Addendum (2026-10-09, decided before any fit):** implementation choices for `duration_lattice.py`.
  - Penalty axes are searched only where their block has columns: the floor has no search; $S$ searches $\lambda_S$ (9 values), since in $S$ the 81 pairs would be 9 identical fits per $\lambda_S$; $A$, $B$, $A,B$ search $\lambda_h$ (9); $A,S$, $B,S$, $A,B,S$ the 81 pairs. Ties go to the larger value on each axis.
  - The objective is DEC-30's as written (no factor ½), solved in closed form; $y$ and the history inputs are standardised with the population SD of their training (non-missing) values. The always-present inputs are not standardised; T's type one-hot spans the whole fold vocabulary, UNK included. Constant columns are dropped in $A$ and $B$ only; the lag-1 missing indicators are always constant (every comparison target has both lag-1 codas).
  - Detection, for the Reading: interval entirely above zero and $p < 0.05$ (the two-sided sign-flip $p$ for the primary, Holm-adjusted for the secondaries).
  - Calibration: $\hat\sigma_{AS}$ is the fold's real-data F fit of $A,S$ at its chosen penalties; $z_{b1}$ is the $A,B,S$ design's own partner lag-1 $\ln d$ column; only the targets change (history inputs keep the real durations). The detection bound is the smallest $f$ whose interval lies entirely above zero; an interval entirely below zero is reported as a calibration failure.
  - Descriptive choices: "repeats the whale's own last type" compares the stored `Ct` with the stored `A_history[-1]`; lag-1 weights are reported on the original scale (change in predicted $\ln d$ per unit of the lag-1 coda's $\ln d$); residual diagnostics use held-out residuals divided by the fold's $\hat\sigma$, pooled, with population moments; the v5 line uses DEC-23's max rule only (`logistic_v5.select_max`) and reports the lattice, the three RQ1 contrasts with interval and two-sided $p$ (no Holm) and the sub-conditions chosen.
  - Data checks: counts exactly, other statistics within 0.0005 of the expected value.
  - Scores come from `evaluation.gaussian_log_density_bits` (new, standard library), so every log density goes through the shared evaluation module.

## Ornament merge (2026-10-05)

**DEC-31 — Ornament merge (`[P7]`): DEC-28's switch structure on an inventory in which ornamented codas take their neighbour's type.**
- **Question:** DEC-28's secondary, $\mathrm{LL(PD)} - \mathrm{LL(R)}$ in $A,B,S$, sits mostly on targets that complete A A B → A (61 targets, about 77% of the net gain). Many of those single Bs are the surrounding A with one extra final click, which Sharma et al. 2024a call an ornament; named types give an ornamented coda its own type, so A A B A can be a labelling effect. Does the gain survive when the codas Sharma et al. flag as ornamented take their neighbour's type?
- **Decision:**
  - Inventory O: `guy_coda_type` (inventory A) with one change. A coda is ornamented if Sharma et al.'s released flag (`ornaments.p`, `sw-combinatoriality` commit 7228c8e, one flag per row of `sperm-whale-dialogues.csv`, `[D2]`) is 1 for the `[D2]` row it matches one-to-one on (REC, TsTo, Duration) rounded to 3 decimals, as `inventories.py` matches `[D1]` to `[D2]`. Unmatched codas count as unornamented.
  - An ornamented coda takes its neighbour's inventory-A label. The neighbour: among the whale's immediately preceding and following codas in the same exchange (ordered by onset), those whose onset is at most 10 s from its own and that have exactly one fewer click (raw `nClicks`), the nearest; a tie goes to the preceding one. Without such a neighbour it keeps its label. The new label replaces the old one wherever the coda appears (target, `A_history`, `B_history`), as for the RQ2 inventories (DEC-22); targets, folds, $S$ and the UNK policy are unchanged.
  - The released flags are used, not a reimplementation of the rule: the published definition (the final click in a coda with one more click than the nearest preceding or following coda within ten seconds), applied to Sharma et al.'s own file, flags 214 codas and recovers 126 of their 151, so their flags involve steps the paper and code don't document.
  - Logistic v4 with RQ1's settings on O (folds, fits, evaluation as for an RQ2 inventory); then DEC-28's runner on O with DEC-28's features, variants (R, L, P, D, PD), grid and rules unchanged. R must reproduce O's v4 run under DEC-25's amended gate.
  - Primary, pre-specified: $\mathrm{LL(PD)} - \mathrm{LL(R)}$ in $A,B,S$ on O, with day-bootstrap CI (10,000 replicates), unweighted day mean, positive days, the repeat and switch parts (`[P1]`) and the exact two-sided sign-flip $p$.
  - Descriptive: the per-coda change from inventory A, $[\mathrm{LL_O(PD)} - \mathrm{LL_O(R)}] - [\mathrm{LL_A(PD)} - \mathrm{LL_A(R)}]$ in $A,B,S$, paired by coda, with day-bootstrap CI; for each inventory, the gain split into targets right after a switch whose target returns to the type before it, other targets right after a switch, and all other targets; $\mathrm{LL(PD)} - \mathrm{LL(P)}$ on O with its two-sided $p$; the own, partner and timing contrasts of O's v4 lattice beside RQ1's.
  - Stop rules: DEC-23's rule on every variant's 16 checks, with DEC-16's reported beside it; DEC-28's fitting flag.
- **Why:**
  - DEC-28's secondary is the result most likely to depend on the labels: the RQ1 lattice was tested across seven inventories (DEC-22), DEC-28 on named types only.
  - Descriptively (2026-10-05): 1,706 of 2,075 RQ1 rows match a `[D2]` row; 66 are flagged; 55 have a qualifying neighbour, 51 change label (31 comparison targets). On the comparison targets, repeats of the own last coda go from 1,277 to 1,337, and returns right after a switch from 61 to 34. The run recomputes these.
  - DEC-22 rejected ornaments as an RQ2 dimension (rare; many already have their own named type). This run uses them only to test one result.
  - Added after the results freeze (2026-10-05) as a final run.
- **Reading:**
  - Primary positive (interval above zero, $p$ < 0.05): the collapsed view still adds usable information with the flagged ornaments merged. Report the share of DEC-28's gain that remains and whether it still sits on returns. One-click excursions without a flag remain in O, so this doesn't rule out other labelling effects.
  - Primary null: the gain isn't detectable once flagged ornaments are merged; DEC-28's secondary is not evidence of switch-backs beyond ornaments.
  - Change from A with its interval below zero: the flagged ornaments carry a detectable part of DEC-28's gain.
  - O's lattice: say whether RQ1's own / partner / timing pattern holds.
- **Limits:** flags exist only for the 1,706 matched rows; they come from Sharma et al.'s release, whose rule isn't fully documented; one-click excursions without a flag stay separate types; for 31 targets the label differs between A and O, so the change from A is descriptive; post-freeze run.
- **Not done:** the published rule as a sensitivity; ornament merges for other inventories; v5 or the residual LSTM on O.

## Differences from the submitted proposal

**Content changes**
- Condition (i) is the smoothed unigram, not "majority-class floor" (DEC-9).
- The type inventory is 23/25 types (21 in the comparison set), not 27.
- Caller identities are segment-local, not identified whales.
- The response delay is redefined (DEC-1).
- RQ2 compares token inventories of the same codas: named types vs Sharma's rhythms, each with and without Sharma's tempo classes (DEC-22).
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