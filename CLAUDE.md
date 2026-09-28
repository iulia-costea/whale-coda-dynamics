# NLP Whales — sperm whale coda sequence modelling

UvA MSc AI Multilingual NLP group project (5 people). Spec: `docs/PLAN.md` (authoritative; IDs like `[M3]`). Why each choice was made: `docs/DECISIONS.md` (`DEC-n`). Code structure and pipeline internals: `docs/ARCHITECTURE.md` — read it before touching `src/` or `config/`. Which reports are current: `reports/README.md`.

## Question and method

Is apparent sequential structure in coda exchanges a whale's own sequencing, or interactional dynamics (Terwilliger et al., 2025)? Predict next coda type $C_t$ from subsets of $A$ (own preceding codas), $B$ (interlocutor's), $S$ (turn category + onset-available timing, PLAN `[D5]`). Eight separately trained models; held-out per-token LL differences estimate conditional 𝒱-information. Primary RQ1 contrast: $\mathrm{LL}(A,B,S) - \mathrm{LL}(B,S)$.

## Hard rules

- Retrain per condition; never ablate by masking at evaluation. Absent sources get no representation (no masks, lengths, padding flags).
- Identical architecture template, hyperparameter search space and selection rule, and splits across conditions. Hyperparameters are chosen on the validation day, never on test days. Day-held-out splits only; never random.
- Per-token LL in bits, never summed, computed only via the shared evaluation module (units in function names).
- Contrasts = paired per-token differences; CIs by day-cluster bootstrap, never token bootstrap. A contrast whose CI lies entirely below zero signals a fitting problem (optimisation or regularisation): investigate before interpreting.
- Never compare raw LL/perplexity across tokenisations (PLAN §6 normalisation).
- $S$ holds only information available at the target's onset and never mechanically includes a coda's duration or content. The retrospective end-to-start delay is a sensitivity run only.
- 82.5% of targets repeat the whale's own last coda: models using $A$ or $B$ include the shared copy weights (PLAN `[M4]`). $A$-conditions below the repeat-own-last baseline in log-likelihood = fitting failure; matching it is a finding (nothing beyond repetition).
- One shared unseen-label (`UNK`) policy for all labelling schemes.
- Never use `pair_category` as a feature. Caller codes (`Whale`, `local_caller_id`) are segment-local, not identified animals.
- Pacific and `DominicaCodas.csv` data: within-coda only; no multi-agent conditions; day-level order is never own history.
- Claims are about dyadic exchanges only. "Not persistence" ≠ grammar — no grammar/syntax conclusions.
- Don't redo Assayag et al.'s bout-threshold correlation sweep or Leitão et al.'s VLMC/KL comparison as results; cite them. Their bins and VLMC may be used as tools (DEC-19).

## Reproducibility and lessons

- Before pushing results: commit all code first, then rerun from a clean checkout; `git status` must be clean afterwards. Results produced by uncommitted code can't be reproduced.
- Reruns on another machine can differ at ~1e-6 (numeric libraries): compare printed values, not bytes. Runs that must match a primary run exactly (e.g. the old-delay sensitivity run) are regenerated together with it, on one machine.
- When a finding changes how models should be built (e.g. a fix to the reference model), record it in `docs/DECISIONS.md` and `docs/PLAN.md` before building the next model. Prompts and reports are not where lessons live.

## Commands

Python 3.10+, `preprocessing/` is stdlib-only. Run with `PYTHONPATH=src`.

```sh
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.prepare      # RQ1 → rq1_* tables, reports/rq1.md
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.prepare_rq2  # Sharma pipeline → rq2_*, reports/rq2.md
```

Unit tests: `PYTHONPATH=src python3 -m pytest tests` (model tests need numpy, scipy and torch). Beyond them, `prepare.validate()` independently recomputes every derived field and raises on mismatch; a clean run of **both** pipelines is the test for preprocessing. Treat any `ValueError` as a failure, never work around it. New derived fields need an independent recomputation in `validate()`.

## Traps

- All text I/O via `io_utils.py`, never bare `open` (UTF-8 + LF pinned; artifact checksums are part of the output).
- `reports/rq1.md`, `reports/rq2.md` are generated — edit the builder code, not the markdown.
- `data/raw/` is never modified; `data/` is committed, so a pipeline run is a diff — review it.
- Prefer failing loudly over guessing: flag, exclude with a recorded reason, or leave unresolved.

Environment: single RTX 3090; models are small.