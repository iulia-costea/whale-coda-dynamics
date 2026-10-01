# Architecture — preprocessing pipeline and repo conventions

Reference for anyone changing `src/` or `config/`. Rules and scope live in `CLAUDE.md` and `docs/PLAN.md`; this file describes how the code is built. Status: RQ1/RQ2 preparation plus RQ3 preprocessing and all primary RQ3 transfer analyses are implemented; coda-type diagnostics and optional VLMC sensitivity remain.

## RQ1/RQ2 stages

Four stages, each a module with `run(config_path)`, each runnable alone with `--config`:

```sh
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.audit        # normalization only, prints row counts
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.exchanges    # grouping only, prints summary JSON
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.audit_rq2 --config config/preprocessing_rq2.json
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.exchanges --config config/exchanges_rq2.json
```

1. **`audit.py`** — reads the raw CSV, typed conversion per the `FIELDS` column map, row-level issues into `audit_issues`, every original value kept in `raw_values`. Writes `data/interim/codas_normalized.jsonl`. Never drops or reorders rows; nothing imputed.
2. **`exchanges.py`** — `group_exchanges()` builds provisional two-caller exchanges. Hard boundaries `(tag_id, recording_id, source_bout_code)`; new segment when the end-to-start gap exceeds 7 s (also run at 5/10 s for the sample-count table — counts only, not a `[V3]` result; cite Assayag's sweep there). Segments excluded wholesale for missing caller codes, ≠2 distinct callers, tied onsets, or missing coda types. Writes `exchange_assignments.jsonl` and `exchange_segments.jsonl`. Calls `audit.run()` itself and checks the source SHA-256, so it never consumes a stale normalized file.
3. **`features.py`** — `build_features()`, the pure source-agnostic core. Input events: `coda_id/whale_id/exchange_id/coda_type/onset_s/offset_s`. Outputs `Ct`, `A_history`/`B_history` (completed codas only, oldest first, `history_limit=10`), and the $S$ fields. Refuses to run unless the caller passes `provisional_assumptions_accepted` or `definitions_confirmed` (opt-in by design).
   - $S$ fields (DEC-1): `turn_category` ∈ `exchange_start/self_continuation/switch/overlap`, `gap_s`, `overlap_elapsed_s`, all relative to `turn_predecessor_id` (latest earlier onset, either caller).
   - Sensitivity/diagnostic only: `retrospective_delay_s` + `retrospective_predecessor_id` (old delay to the last interlocutor coda), `overlaps_prior_interlocutor`, `delay_uses_post_onset_endpoint`.
4. **`prepare.py`** — orchestrates, assigns splits (`assign_splits`, deterministic `sha256(f"{seed}:{day}")` order; currently one 10/2/2 day split, to be superseded by leave-one-day-out per DEC-2), runs `validate()`, writes `rq1_*` CSV+JSONL, generates `reports/rq1.md` with SHA-256 of every artifact.

## Configs

Chain by reference, not merged: `config/rq1.json` → `config/exchanges.json` → `config/preprocessing.json`. Each stage resolves the repo root as **its config file's `parent.parent`**, so configs must stay one level below root and all internal paths are root-relative.

`prepare.py` validates strictly and fails rather than adapting: `input_conditions` must be exactly the A/B/S power set with exact field lists; `timing_convention` must be exactly `onset_available` (primary) or `retrospective_signed_end_to_start` (sensitivity), and selects the required $S$ field list via `prepare.S_FIELDS` (shared with `prepare_rq2.py`); `provisional_assumptions_accepted` must be literally `true`.

## Sharma pipeline (`prepare_rq2.py`, `audit_rq2.py`)

- Reuses `group_exchanges`, `build_features`, `assign_splits`/`validate`/`write_table` **unchanged**; RQ1 outputs must be unaffected by work here. `run_exchanges_rq2()` exists because `exchanges.py` hardcodes `from .audit import run` — deliberate glue, don't refactor away.
- Role since DEC-3: whole-preprocessing-chain robustness only. Tokenisation tests (primary `guy_coda_type` vs `decision_tree_coda_type`; secondary Sharma rhythm × tempo on matched exchanges) run inside the RQ1 pipeline as label-column swaps.
- Explicit differences from the Assayag release (also in `reports/rq2.md`): no Tag/Bout/Date/Focal → `tag_id = recording_id`, REC is the only hard boundary, `strict_focal_subset` always False; `recording_date` holds the deployment prefix `^sw\d+` (schema parity, no chronological meaning); `rhythms.p` joined by row index; unseen types → `UNK` via `apply_unk_policy` (preserves `Ct_raw`), moving to the shared module per DEC-7.
- One `Duration=0` row excludes all of `sw061b003` (731 rows) — expected conservative behaviour, not a bug.

## RQ3 preprocessing (`src/rq3/preprocessing/`)

- `prepare.py` reads `CarribeanCodas.csv` and `PacificCodas.csv`, preserves every source row in an interim audit JSONL, and writes two continuous-ICI tables for accepted 3–10-click codas whose required ICIs are in [0.005, 1.0) s. Caribbean `recording_date` is canonical ISO, while raw `source_date` remains preserved. `tokenize.py` defines the global `fixed-50ms-v1` tokenizer (20 ICI symbols plus `BOS`/`EOS`); it does not fit bin edges to data.
- `folds.py` assigns `group_id` values (Caribbean canonical `recording_date`; Pacific `grpvar`) to deterministic, group-disjoint clan folds in `rq3_folds.csv`.
- `config/rq3_identity_mapping.json` records Hersh et al.'s Table S2 Pacific identity-type mapping. `prepare.py` writes its derived `identity_status` to the interim audit and metadata table; it is never a model input.
- `config/rq3_spatial_overlap.json` records Hersh et al.'s Table S4 directional Pacific repertoire-overlap matrix. It is experiment metadata for the exploratory overlap analysis, not a model input or preprocessing field.
- `validate.py` is read-only. It independently checks raw-source coverage, eligibility, ICI sequences, deterministic group/fold assignment, and agreement between the interim, core, metadata, and fold artifacts.
- Run from the repository root with `PYTHONPATH=src python3 -m rq3.preprocessing.prepare --config config/preprocessing_rq3.json`, then `PYTHONPATH=src python3 -m rq3.preprocessing.validate --config config/preprocessing_rq3.json`.

## RQ3 experiments (`src/rq3/experiments/`)

- `data.py` loads and validates the prepared core, metadata, folds, identity mapping, and spatial-overlap metadata. `models.py` implements additive-smoothed unigram/n-gram models; `scoring.py` records base-2 coda and per-symbol scores; `evaluation.py` owns all shared aggregation and later transfer-loss/bootstrap helpers.
- `runner.py --stage calibration` selects one candidate model per clan using only that clan's held-out folds, then writes one cross-fitted own score per eligible coda. `runner.py --stage initial-transfer` runs only EC1 <-> EC2. `runner.py --stage matrix` fits each selected source clan's fold models, scores every other clan, pairs every averaged transfer score with its existing own-clan score, and counts each target token sequence in each source fold's actual training data for the coverage diagnostic. `runner.py --stage bootstrap` resamples whole target groups 1,000 times without retraining. `runner.py --stage identity` adds the Pacific Table S2 identity/non-identity labels to existing paired scores and reruns that bootstrap by stratum. `runner.py --stage overlap` runs the exploratory Table S4 Pacific association with all 5,040 clan-label permutations. `transfer.py`, `uncertainty.py`, `identity.py`, and `overlap.py` hold this logic; `report.py` generates the reports.
- `runner.py --stage summaries` generates `README.md` files at the RQ3 results root and every analysis subfolder. Each is a navigable, data-derived description of its direct artifacts and child analyses.
- Run calibration from the repository root with `PYTHONPATH=src python3 -m rq3.experiments.runner --config config/rq3_experiments.json`. Run the initial comparison with `PYTHONPATH=src python3 -m rq3.experiments.runner --stage initial-transfer --config config/rq3_experiments.json`, the full matrix with `PYTHONPATH=src python3 -m rq3.experiments.runner --stage matrix --config config/rq3_experiments.json`, or uncertainty intervals with `PYTHONPATH=src python3 -m rq3.experiments.runner --stage bootstrap --config config/rq3_experiments.json`.

## Conventions

- **`io_utils.py`** for all text I/O (`read_text/write_text/read_json/read_jsonl/write_jsonl`): UTF-8 (configs contain Hebrew month aliases) and `\n` newlines, because checksums are reported output. `.gitattributes` pins LF.
- **Reports:** `reports/rq1.md`, `reports/rq2.md`, and `reports/rq3_preprocessing.md` are generated; `reports/data_dictionary.md` is hand-written — the reference for field semantics, provenance, and questions open with source authors.
- **Data:** `data/raw/`, `data/interim/`, `data/processed/` all committed; raw never modified.
- **RQ1/RQ2 exchange identity:** `local_caller_id`, `identity_status: provisional_local_code`, exchange IDs like `provisional:g7:row0`. This convention does not apply to RQ3's coda-type-derived `identity` / `non_identity` / `unmapped` labels. `coda_id`s are not stable across source versions.
- **Failing loudly:** unparseable → `None` + audit flag; ambiguous segments excluded with a reason; unestablished facts (absolute timestamps, individual identity, Bout scope) stay unresolved.

## Data files at a glance

Detail and verified counts: PLAN §2.

| File | Role |
|---|---|
| `data/raw/dswp-2014-2016-retagged-maor-2025.csv` | Assayag et al. 2026; RQ1 + both RQ2 labellings |
| `data/raw/sperm-whale-dialogues.csv`, `rhythms.p` | Sharma et al. 2024; secondary RQ2 labels, standalone pipeline. `tempos-dict.p`, `ornaments.p` still to fetch |
| `data/raw/CarribeanCodas.csv` | EC1 + EC2; RQ3 sympatric transfer source |
| `data/raw/PacificCodas.csv` | Hersh et al. 2022 Pacific data; RQ3 allopatric extension |
| `data/interim/rq3/rq3_codas_normalized.jsonl` | All-row RQ3 normalization and audit artifact |
| `data/processed/rq3/rq3_codas.csv` | Core RQ3 model-ready continuous-ICI table |
| `data/processed/rq3/rq3_codas_metadata.csv` | RQ3 core rows plus provenance and diagnostic metadata |
| `data/processed/rq3/rq3_folds.csv` | Deterministic, group-disjoint RQ3 fold assignments |
| `data/results/rq3/calibration/self_scores.csv` | One cross-fitted own-clan likelihood score per eligible RQ3 coda |
| `data/results/rq3/calibration/self_calibration.csv` | Held-out model-candidate scores and selected settings per clan |
| `data/results/rq3/initial_caribbean/` | Per-source-fold, paired-score, and summary outputs for EC1 <-> EC2 |
| `data/results/rq3/transfer_matrix/fold_scores.csv` | Per-source-fold scores for all 72 cross-clan transfer cells |
| `data/results/rq3/transfer_matrix/scores.csv` | Paired own/transfer scores for all 81 matrix cells |
| `data/results/rq3/transfer_matrix/summary.csv` | Aggregate primary 9 x 9 transfer matrix |
| `data/results/rq3/transfer_matrix/coverage/summary.csv` | Covered/uncovered fold-specific transfer-loss strata |
| `data/results/rq3/transfer_matrix/uncertainty/` | All bootstrap draws and 95% interval summary |
| `data/results/rq3/transfer_matrix/identity/` | Pacific-target identity/non-identity scores, draws, and summary |
| `data/results/rq3/transfer_matrix/pacific_overlap/` | Table S4 pair values and exact permutation results |

## Known gotchas

- DEC-1 changed the $S$ fields (and renamed `response_delay_s` → `retrospective_delay_s`) — say so in the commit.
- Old `retrospective_delay_s` for continuations runs to 50 s while switches are capped at 6.76 s by the 7 s gap, so the old delay identifies turn category. Why it isn't the primary $S$.
