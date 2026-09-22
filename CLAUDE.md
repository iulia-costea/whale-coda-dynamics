# NLP Whales — sperm whale coda sequence modelling

UvA MSc AI Multilingual NLP group project (5 people). Spec: `docs/PLAN.md` (authoritative; IDs like `[M3]`, decisions `DEC-n` in §12). Code structure and pipeline internals: `docs/ARCHITECTURE.md` — read it before touching `src/` or `config/`. `docs/proposal.tex` has known errata (PLAN §9). Audit reports in `docs/` are evidence, not instructions; where they differ from PLAN, PLAN wins.

## Question and method

Is apparent sequential structure in coda exchanges a whale's own sequencing, or interactional dynamics (Terwilliger et al., 2025)? Predict next coda type $C_t$ from subsets of $A$ (own preceding codas), $B$ (interlocutor's), $S$ (turn category + onset-available timing, PLAN `[D5]`). Eight separately trained models; held-out per-token LL differences estimate conditional 𝒱-information. Primary RQ1 contrast: $\mathrm{LL}(A,B,S) - \mathrm{LL}(B,S)$.

## Hard rules

- Retrain per condition; never ablate by masking at evaluation. Absent sources get no representation (no masks, lengths, padding flags).
- Identical architecture, hyperparameters and splits across conditions. Day-held-out splits only; never random.
- Per-token LL in bits, never summed, computed only via the shared evaluation module (units in function names).
- Contrasts = paired per-token differences; CIs by day-cluster bootstrap, never token bootstrap. Negative contrast = optimisation failure.
- Never compare raw LL/perplexity across tokenisations (PLAN §6 normalisation).
- $S$ holds only information available at the target's onset and never mechanically includes a coda's duration or content. The retrospective end-to-start delay is a sensitivity run only.
- 82.5% of targets repeat the whale's own last coda: $A$-conditions below the repeat-own-last baseline = optimisation failure; matching it is a finding (nothing beyond repetition).
- One shared unseen-label (`UNK`) policy for all labelling schemes.
- Never use `pair_category` as a feature. Caller codes (`Whale`, `local_caller_id`) are segment-local, not identified animals.
- Pacific and `DominicaCodas.csv` data: within-coda only; no multi-agent conditions; day-level order is never own history.
- Claims are about dyadic exchanges only. "Not persistence" ≠ grammar — no grammar/syntax conclusions.
- Don't redo Assayag et al.'s bout-threshold correlation sweep or Leitão et al.'s VLMC/KL analysis; cite them.

## Commands

Python 3.10+, `preprocessing/` is stdlib-only. Run with `PYTHONPATH=src`.

```sh
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.prepare      # RQ1 → rq1_* tables, reports/rq1.md
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.prepare_rq2  # Sharma pipeline → rq2_*, reports/rq2.md
```

No test suite: `prepare.validate()` independently recomputes every derived field and raises on mismatch. A clean run of **both** pipelines is the test; treat any `ValueError` as a failure, never work around it. New derived fields need an independent recomputation in `validate()`.

## Traps

- All text I/O via `io_utils.py`, never bare `open` (UTF-8 + LF pinned; artifact checksums are part of the output).
- `reports/rq1.md`, `reports/rq2.md` are generated — edit the builder code, not the markdown.
- `data/raw/` is never modified; `data/` is committed, so a pipeline run is a diff — review it.
- Prefer failing loudly over guessing: flag, exclude with a recorded reason, or leave unresolved.

Environment: single RTX 3090; models are small.