# Architecture — preprocessing pipeline and repo conventions

Reference for anyone changing `src/` or `config/`. Rules and scope live in `CLAUDE.md` and `docs/PLAN.md`; this file describes how the code is built. Status as of 2026-09-28: preprocessing, the leave-one-day-out folds and the shared evaluation are complete. Model code lives in `src/whale_coda_dynamics/`: `evaluation.py` (shared metrics and contrasts), `logistic.py` (reference model), `logistic_v5.py` (v5 and DEC-23's stop rules, computed from v4's fits; see "Logistic v5"), `neural.py` (LSTM v1/v2, superseded), `residual.py` (residual LSTM on a frozen logistic v4, DEC-21: model, checks, device measurement and full run), `history_length.py` (own-history length sweep on logistic v4, DEC-25, PLAN `[P2]`); their design is PLAN `[M4]`. RQ2 (DEC-22) adds `sharma_inventory.py` and `rq2_compare.py` (see "RQ2 token inventories"), and DEC-24 `tempo_factor.py` (see "Partner-tempo ablation"). This file covers preprocessing and repo conventions.

## Stages

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
- Role since DEC-3: whole-preprocessing-chain robustness only. RQ2's token inventories (DEC-22) are computed from `[D1]`'s own intervals and run inside the RQ1 pipeline; see "RQ2 token inventories" below.
- Explicit differences from the Assayag release (also in `reports/rq2.md`): no Tag/Bout/Date/Focal → `tag_id = recording_id`, REC is the only hard boundary, `strict_focal_subset` always False; `recording_date` holds the deployment prefix `^sw\d+` (schema parity, no chronological meaning); `rhythms.p` joined by row index; rare/unseen types → `UNK` via shared `vocabulary.py` (preserves `Ct_raw`), per DEC-7/10.
- One `Duration=0` row excludes all of `sw061b003` (731 rows) — expected conservative behaviour, not a bug.

## Conventions

- **`io_utils.py`** for all text I/O (`read_text/write_text/read_json/read_jsonl/write_jsonl`): UTF-8 (configs contain Hebrew month aliases) and `\n` newlines, because checksums are reported output. `.gitattributes` pins LF.
- **Reports:** `reports/rq1.md`, `reports/rq2.md` generated; `reports/data_dictionary.md` hand-written — the reference for field semantics, provenance, and questions open with source authors.
- **Data:** `data/raw/`, `data/interim/`, `data/processed/` all committed; raw never modified.
- **Identity:** `local_caller_id`, `identity_status: provisional_local_code`, exchange IDs like `provisional:g7:row0`. `coda_id`s are not stable across source versions.
- **Failing loudly:** unparseable → `None` + audit flag; ambiguous segments excluded with a reason; unestablished facts (absolute timestamps, individual identity, Bout scope) stay unresolved.

## Data files at a glance

Detail and verified counts: PLAN §2.

| File | Role |
|---|---|
| `data/raw/dswp-2014-2016-retagged-maor-2025.csv` | Assayag et al. 2026; RQ1 + both RQ2 labellings |
| `data/raw/sperm-whale-dialogues.csv`, `rhythms.p` | Sharma et al. 2024a; standalone pipeline; `rhythms.p` checks the RQ2 rhythm rule |
| `data/raw/mean_codas.p`, `tempos-dict.p` | Sharma et al. 2024a, copied unchanged from `github.com/pratyushasharma/sw-combinatoriality` (commit `7228c8e`): the 18 rhythm templates and the tempo classes (RQ2, PLAN `[R2-1]`, `[R2-2]`). `ornaments.p` not used |
| `data/raw/CarribeanCodas.csv` | EC1 + EC2; RQ3; its EC1 durations fix the RQ2 tempo cuts |
| Hersh et al. 2022 (not in repo) | RQ3 allopatric extension |

## RQ2 token inventories (DEC-22)

Only the labels change: every stage after labelling is RQ1's code, run on a relabelled copy of RQ1's targets. Names: `b_dtree`, `c_rhythm`, `d_rhythm_tempo`, `e_type_tempo`, `d_rhythm_tempo_tdict`, `e_type_tempo_tdict` (PLAN `[R2-0]`'s B, C, D, E, D′, E′). Inventory A is RQ1's own `rq1_logistic_v4` run.

```sh
PYTHONPATH=src python3 -m whale_coda_dynamics.sharma_inventory                 # once, needs NumPy; output committed
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.inventories        # config/rq2_inventories.json
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.folds --config config/rq2_<name>_folds.json
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 -m whale_coda_dynamics.logistic --config config/rq2_<name>_logistic_v4.json
PYTHONPATH=src python3 -m whale_coda_dynamics.rq2_compare                      # config/rq2_compare.json
```

- **`sharma_inventory.py`** (NumPy, outside `preprocessing/`), `config/sharma_inventory.json`: unpickles `data/raw/mean_codas.p` and `tempos-dict.p` and writes `data/interim/sharma_inventory.json`: the 18 templates in their original order (index = rhythm label) with click counts; per `tempos-dict.p` class its count, shortest and longest duration; the boundaries midway between adjacent classes; SHA-256 of both pickles; the source commit. `tempos-dict.p` has a sixth key (5) that is empty; it is recorded and left out. The JSON is committed, so preprocessing stays standard-library only; `tests/test_inventories.py` re-extracts it byte for byte when NumPy is present.
- **`preprocessing/inventories.py`** (standard library), `config/rq2_inventories.json`. Reads `rq1_targets.jsonl`, the `raw_values` and `alternative_coda_type` of `codas_normalized.jsonl`, `sharma_inventory.json`, `CarribeanCodas.csv` (`utf-8-sig`, via `io_utils.read_csv_rows`), `sperm-whale-dialogues.csv` and `rhythms.p`.
  - Raises unless the KDE of the EC1 durations gives the configured minima and maxima (`kde_cuts`) and the rhythm rule reproduces every `rhythms.p` label (`check_rhythm_reproduction`).
  - Labels all 2,075 rows by `coda_id` from their own raw values (`coda_labels`, `rhythm_label`, `tempo_class`); raises on any missing or unparseable value a rule needs and on labels equal to the UNK token or with a stray `|`. Zero intervals are parsed values, not missing ones: the rule is applied as written and the report lists those codas.
  - Writes `data/processed/rq2_<name>_targets.jsonl`: `rq1_targets.jsonl` with `Ct_raw` = `Ct` = the label and `A_history`/`B_history` rebuilt from the history IDs, plus `inventory`. `check_relabelled` verifies the written files (same rows, order, IDs and every other field).
  - `reports/rq2_inventories.md`: the checks, the `[D1]`/`[D2]` agreement (matched one-to-one on REC with TsTo and Duration rounded to 3 decimals), the KDE vs `tempos-dict.p` agreement, per-inventory counts (`[R2-0]`) and SHA-256 of inputs and outputs.
  - `prepare_rq2.py` checksums its own files by name: the inventory files share its `rq2_` prefix in `data/processed/`.
- **Folds, unchanged code:** `config/rq2_<name>_folds.json` (preparation config `config/rq1.json`, 14 days, validation seed 42, an optional report `title`), writing `data/processed/rq2_<name>_folds/manifest.json` and `reports/rq2_<name>_folds.md`. The inventories stage and the six folds runs are in `PREPROCESSING` (`tests/test_committed_artifacts.py`).
- **Logistic v4, unchanged model and selection:** `config/rq2_<name>_logistic_v4.json` and `config/rq2_<name>_evaluation_v4.json` (settings as `rq1_logistic_v4`, 10,000 replicates, no `compare_against`), writing `data/results/rq2_<name>_logistic_v4/` and `reports/rq2_<name>_logistic_v4.md`. RQ2-only config keys, all off by default: `skip_nonconverged_candidates` (DEC-22: a grid candidate that fails `fit()`'s convergence check, `NonConvergenceError`, is left out of the validation-day selection and counted per fold and condition; all failing fails the run), `report_title`, `report_note`. Without `compare_against`, `write_v2_report` drops the comparison columns. Run with one BLAS thread: OpenBLAS's default (one thread per core) is over ten times slower on these small matrices, and the thread count changes summation order, so a byte-for-byte rerun needs the same setting.
- **`rq2_compare.py`**, `config/rq2_compare.json`: reads each inventory's manifest, `evaluation/summary.json`, `evaluation/scored_predictions.jsonl` and `l2_selection.jsonl`, and the fine inventories' `predictions.jsonl` for `[R2-4]`. One set of resampled days per replicate is shared by all inventories (`shared_day_draws`, drawn exactly as `evaluation.day_bootstrap_bits` draws them; the run checks that it reproduces each inventory's recorded contrast CIs). Shares are ratios of token-weighted means within each replicate. Writes `data/results/rq2_compare/summary.json` (with `input_sha256`) and `reports/rq2_inventories_v4.md`. Stop rules come from `logistic.check_stop_conditions`. No raw log-likelihood is compared across inventories.

## Logistic v5 (DEC-23)

v5 refits nothing: it reads a v4 run's committed outputs and chooses, per fold and condition, among v4's chosen fits of every sub-condition.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 -m whale_coda_dynamics.logistic_v5 --config config/rq1_logistic_v5.json
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 -m whale_coda_dynamics.logistic_v5 --config config/rq1_logistic_v5_retrospective.json  # after the primary run
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 -m whale_coda_dynamics.logistic_v5 --config config/rq2_<name>_logistic_v5.json
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 -m whale_coda_dynamics.rq2_compare --config config/rq2_compare_v5.json
```

- **Config** (`config/<v4 run name with v4 → v5>.json`): `v4_config` (the v4 run's config, which gives the manifest, the evaluation config with its bootstrap settings, and the v4 output directory), `output_directory`, `report`, `report_title`, optional `headline_note` and `report_note`, `governing_model` (`v4` for RQ1, whose headline stays v4; `v5` for the RQ2 inventories), `reproduction_tolerance_bits` (1e-9), `alpha` (0.05). The old-delay run also names `primary_v5_summary` and checks that its conditions without S equal the primary v5 run's. `load()` refuses a v4 run whose recorded input hashes (targets, manifest, v4 config) differ from the committed files.
- **Order inside `run()`**; nothing is written if a check fails:
  1. `prepare_folds`: `prepare_fold` per fold; the policy and the validation and test IDs must equal the manifest's.
  2. Gate 1, `reproduce_v4`: for every fold and condition, validation and test probabilities are recomputed from `models.jsonl` with `Encoder.from_record` (stored vocabulary, fields, order, per-field orders if recorded, turn categories and standardisation; no refit) and `stored_fit_probabilities`. The mean validation log-likelihood must equal `l2_selection.jsonl`'s value for the chosen penalties, and the test probabilities `predictions.jsonl`'s, to the tolerance (the prior's too, for none). The models' and selection's chosen penalties must agree.
  3. Selection and gate 2, `select_all`: candidates are `subconditions(blocks)` with their recomputed per-target validation log-likelihoods. `select_max` takes the highest mean; exact ties go to fewer blocks, then the block order A, B, S. `select_one_se` is DEC-21's rule (best = `select_max`'s; within one SE if the mean paired difference to the best ≤ sd(ddof=1)/√n; fewest blocks, then higher mean, then block order). Every add-one-input contrast of the max-rule choices must be ≥ 0 on every validation day.
  4. v5 predictions: `rebuild_predictions(selection, v4_predictions, rule)` gives each condition its chosen sub-condition's stored v4 test predictions; both rules are scored with `evaluate()` and v4's evaluation settings. Per-coda log-likelihoods must equal v4's for the chosen sub-condition, and v4's scored log-likelihoods must equal log2 of its stored probabilities, both to 1e-12. Not exactly: RQ1's v4 run was made on another machine, and its stored log2 values differ from this machine's in the last bit (up to 1.8e-15). The RQ2 runs agree exactly.
  5. Stop rules, `stop_rules()`, for v4, v5 and the one-SE variant: DEC-16's (`logistic.check_stop_conditions`) and DEC-23's (`dec23_stop_rules`: `evaluation.sign_flip_p`, alternative `less`, on each check's day sums; `evaluation.holm` across the 16). The three `[M2]` contrasts also get the two-sided sign-flip p.
- **Outputs**, in `data/results/<v4 run name with v4 → v5>/`:
  - `selection.jsonl`: one row per fold, condition and rule (`max`, `one_se`), with the chosen sub-condition, every candidate's validation log-likelihood and, for one-SE, its difference to the best, SE and whether it is within one SE;
  - `evaluation/scored_predictions.jsonl` and `paired_contrasts.jsonl` (max rule);
  - `evaluation/summary.json`: `evaluate()`'s summary for the max rule, plus `gates`, `selection_counts`, `stop_rules` (per model, `old` and `new`), `m2_sign_flip_two_sided_p`, `governing_stop_rule`, `one_se` (metrics, contrasts, repeat checks) and `input_sha256` of every v4 file read.
  - There is no `predictions.jsonl`: it would duplicate v4's, and `rebuild_predictions` recreates it.
- **Report** (`reports/<v4 report name with v4 → v5>.md`): the gates, both stop rules for the three models with the governing status, the v4/v5 lattices, how often each sub-condition was chosen (and per fold), the `[M2]` contrasts with sign-flip p and the v4-vs-v5 sign pattern, the 16 checks per model. Positive-day counts also give the days at exactly zero: under v5 both conditions of a contrast often share a sub-condition on a fold.
- **Reuse (DEC-24):** `subconditions`, `select_max`, `select_one_se` and `dec23_stop_rules` take arbitrary block names and a block order; `evaluation.sign_flip_p` and `evaluation.holm` are standard library.
- **`rq2_compare.py` with `"model": "v5"`** (`config/rq2_compare_v5.json`): each inventory spec gives `results_directory` (v5) and `v4_results_directory`. Stop marks come from DEC-23's rule on v5; the inventory table also shows DEC-16's rule and both rules on v4. Contrast tables gain a two-sided sign-flip p and the days at zero. `[R2-4]` uses `rebuild_predictions` on the fine inventory's `selection.jsonl` and v4 `predictions.jsonl`. A nested pair may carry a `note` (the A → E′ boundary sensitivity). Without the key, the v4 comparison regenerates byte for byte.

## Partner-tempo ablation (DEC-24)

`tempo_factor.py` fits the chain rule's tempo factor as its own model: the target is a coda's KDE tempo class (`t0`–`t4`), given the coda's own pattern. The pattern is the Sharma rhythm in the primary run and `guy_coda_type` in the secondary run. One command fits both runs and writes one report:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python3 -m whale_coda_dynamics.tempo_factor --config config/rq2_partner_tempo.json
```

- **Config** (`config/rq2_partner_tempo.json`):
  - top-level settings: `tempo_classes`, `reference_manifest` (RQ1's folds), v4's settings (`l2_grid`, `history_order`, `max_iterations`, `gradient_tolerance`, `skip_nonconverged_candidates`), the bootstrap settings, `alpha`, `governing_model` (must be `v5`), `report` and `report_title`;
  - one entry in `runs` per run: `name`, `role`, `pattern_label`, `manifest` (the fine inventory: D or E), `pattern_manifest` (the coarse inventory: C, or RQ1's folds for types) and `output_directory`.
- **Labels.** The fine inventory's `Ct_raw` (`<pattern>|<tempo>`) is split by `split_label`.
  - `factor_labels` (gate 1) requires, for every coda:
    - the pattern equals the coarse file's `Ct_raw`;
    - the tempo is a configured class;
    - histories, S fields, day and eligibility equal the coarse file's.
  - Tempo labels must be equal across runs.
  - `check_folds` (gate 2) requires each run's folds, S fields and preparation settings to equal the coarse manifest's and RQ1's.
- **Folds.** `prepare_factor_fold` runs `prepare_fold` (the fine manifest's policy must reproduce).
  - It fits the pattern vocabulary with `fit_vocabulary` on the training days' patterns (*k* = 2, UNK), which must equal the coarse manifest's fold vocabulary.
  - It builds per-split comparison rows: `tempo`, `pattern` and the four streams `own_tempo`, `own_pattern`, `partner_tempo`, `partner_pattern`, each from `A_history_ids`/`B_history_ids` (oldest first), plus the S fields.
  - Tempo has no UNK.
- **`FactorEncoder`**, fitted on training rows only. Its columns, in order:
  - the coda's own pattern (one-hot);
  - lags 1–3 of each present stream (one-hot in that factor's vocabulary; missing lags are zero vectors);
  - S, built by `logistic.Encoder` on the manifest's S fields (the same columns and standardisation as v4), without its intercept;
  - one intercept.
  - `timing_mask` marks the S columns for the S penalty.
  - `copy_indices` gives β's class (the own last tempo; absent in the floor) and γ's class (the partner's last tempo; only with T).
  - Labels outside a vocabulary raise.
- **Conditions.** `floor`, and own history with every subset of {T, P, S}: `own`, `own_T`, …, `own_T_P_S`.
- **Fitting.** `fit_condition` calls `logistic.grid_search_l2`, which is v4's grid loop, factored out of `select_l2_and_fit` unchanged. It runs on training and validation rows only and records per-target validation log-likelihoods.
- **Selection.**
  - `select_fold` gives v5's candidates for each condition: the own fits of its sub-conditions, with `logistic_v5.subconditions(…, BLOCKS)`. It selects with `select_max` and `select_one_se` (block order T, P, S).
  - `validation_monotonicity` is gate 3.
  - `model_predictions` rebuilds v5's test predictions with `logistic_v5.rebuild_predictions`; the floor keeps its own fit.
- **Evaluation**, only after every gate of both runs passes:
  - `score`: per-coda `evaluation.log_likelihood_bits`, argmax ties broken on sorted labels;
  - `contrast_stats`: `evaluation.day_bootstrap_bits`, days at zero and `evaluation.sign_flip_p`, two-sided;
  - `stop_rules`: `logistic_v5.dec23_stop_rules` on the 13 checks of `stop_family()`, plus DEC-16's rule;
  - `turn_category_breakdown`: descriptive, for the four `CONTRASTS`;
  - all of the above for the own fits, v5 and v5 one-SE.
  - `own` and `floor` must be identical under all three.
- **Outputs**, per run, in `output_directory`; nothing is written until both runs are done:
  - `predictions.jsonl`: the own fits' test probabilities;
  - `models.jsonl`: encoder record, weights, β, γ and the chosen penalties;
  - `l2_selection.jsonl`: every penalty candidate;
  - `selection.jsonl`: v5's candidates and choices per rule;
  - `evaluation/scored_predictions.jsonl`: every model, condition and coda;
  - `evaluation/summary.json`: gates, fit summaries, selection counts, per-model metrics, contrasts, checks, stop rules and turn-category breakdown, governing status, BLAS environment and `input_sha256` of every file read.
- **Report** (`reports/rq2_partner_tempo.md`), for both runs: checks, stop rules (both rules, three models, governing status), the four contrasts with sign pattern and DEC-24's reading rules (no automatic conclusion), turn categories, lattices, sub-conditions chosen, penalties and copy weights, the 13 checks per model.

## Known gotchas

- DEC-1 changed the $S$ fields (and renamed `response_delay_s` → `retrospective_delay_s`) — say so in the commit.
- Old `retrospective_delay_s` for continuations runs to 50 s while switches are capped at 6.76 s by the 7 s gap, so the old delay identifies turn category. Why it isn't the primary $S$.

## Shared vocabulary and completed audit items

`vocabulary.py` is used by both preparation orchestrators after original feature validation. `fit_vocabulary` receives unique retained training events, counts each coda once (including context-only codas), and retains types with frequency at least `unk_min_count` (default 2). `apply_vocabulary` maps targets and both histories in every split; `validate_mapping` independently checks labels against original events. `Ct_raw` and history IDs preserve recoverability for future folds. Priors count mapped training comparison targets with additive smoothing over the same vocabulary. Both pipelines export a label-prior JSON and mapping counts in their generated reports. Current splits are unchanged; refitting per leave-one-day-out fold remains Phase 2.

The hand-written data dictionary records source-verified pair-category missingness (1,597/2,425 unlabelled versus 756/1,508 labelled codas non-focal; all 157 missing-Focal codas unlabelled).

Sharma normalization exports `long_coda` (`nClicks > 10`, null if invalid/missing). The RQ2 orchestrator independently checks it against raw click counts and copies it into both prepared tables without changing the shared grouping or feature core. It is diagnostic metadata, not an input or exclusion rule. The generated report counts flagged source, retained, and comparison rows.

## RQ1 leave-one-day-out harness

Run `PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.folds` with `config/rq1_folds.json`. It reads the complete prepared targets table, restores original target/history labels, and assigns 14 folds. The next day in cyclic SHA-256(seed:day) order is validation; the other 12 days train. Every day is test and validation once. No labels or outcomes choose the validation day.

`prepare_fold(rows, fold, preparation)` returns fresh mapped rows and the fitted vocabulary/prior. It reruns the shared history/timing validator and independent mapping validator. Model consumers select comparison-eligible rows and only declared input columns; numerical transforms remain training-only work for the model encoder. The original fixed-split preparation tables are not rewritten.

`data/processed/rq1_folds/manifest.json` records all fold assignments, comparison IDs shared by all conditions, training counts, vocabularies, priors, UNK statistics and input/config hashes. `reports/rq1_folds.md` summarizes sizes. Exactly-once coverage currently concerns test assignments; prediction coverage will be checked by the future evaluator. Shared evaluation is implemented; model code remains unimplemented.

## RQ3 pipeline

The RQ3 preprocessing pipeline in `src/whale_coda_dynamics/rq3/preprocessing/` normalizes the Caribbean and Pacific coda sources, applies the fixed eligibility rules and tokenizer, and writes the processed tables and deterministic group-disjoint folds. Its read-only validator independently checks source coverage, derived coda fields, and fold assignments. RQ3 experiment stages in `src/whale_coda_dynamics/rq3/experiments/` perform source-clan calibration, cross-clan transfer scoring, bootstrap uncertainty, identity strata, and exploratory overlap analysis.

## RQ3 pooled LSTM (`rq3_neural/`, DEC-27, PLAN `[R3-8]`)

`src/whale_coda_dynamics/rq3_neural/` runs the pooled multi-clan experiment on top of Vlad's RQ3 code, which it imports and never copies or edits: codas and folds through `rq3.experiments.data` (`read_csv`, `parse_codas`, `build_fold_lookup`), tokens through `rq3.preprocessing.tokenize.tokenize_ici_sequence`, and every aggregate and CI through `rq3.experiments.evaluation` (`paired_transfer_loss`, `aggregate_transfer_loss`, `bootstrap_group_mean`) with `uncertainty.pair_seed`/`percentile`. Config: `config/rq3_neural.json`.

- `data.py`: `Context` (codas, folds, encodings, index sets per clan and fold), the held-out plans (own-clan fold k; pooled run k; leave one clan out) and the leakage checks, which raise. Every model has the same input vocabulary (20 intervals, `BOS`, nine `BOS_<clan>`, padding), so a tagged and an untagged model with the same seed start from identical weights.
- `model.py`: one LSTM class for every arm, the training sampler (clan by n_c^alpha, coda uniformly, with replacement; one generator per fit, seeded), fitting with early stopping or for a fixed number of epochs, scoring in bits, and the spawn worker pool (one torch thread per worker, deterministic algorithms, as `residual.py`).
- `runner.py`: the stages `pilot`, `arm1`, `arm23`, `capacity`, `arm4`, `arm5`, `report` (`--stage <name>` or `all`), each writing `data/results/rq3_neural/<stage>/` with a `summary.json` (config, input SHA-256, commit, timing; checked by `test_committed_artifacts.py`). Later stages read the earlier stages' outputs: device and worker count and the arm 5 plan from the pilot, the arm 1 run that passed gate 1, arm 3's primary alpha, arm 4's chosen settings and checkpoints. Committed results need a clean tree (`require_clean_tree`); commit each stage's outputs before starting the next.
- `contrasts.py`: thin wrappers that put scores in Vlad's row format; `report.py`: every contrast, the predictions table (verdict rules fixed in `PREDICTION_RULES`) and `reports/rq3_neural.md`; `plots.py`: the figures (matplotlib, not needed by the tests).

Score rows follow PLAN `[R3-3]`: `ll_bits` is the coda's log2 probability (Vlad's `bits_per_coda`), `n_symbols` his `token_count`; a row is the mean over the refit seeds. Arm 5 stores sums per (model, target group) instead (`arm5/group_sums.csv`). Arm 4's weights go to `data/results/rq3_neural/checkpoints/` (gitignored; their state SHA-256 is in `arm4/summary.json` and is checked when arm 5 loads them). Tests: `tests/test_rq3_neural.py` (checks, plans, Vlad's numbers) and a smoke run of every stage in `tests/test_pipeline_smoke.py`; a smoke config (`smoke_test: true`) records a failed gate instead of stopping and is refused if it would write into the repository.

## Shared evaluation in bits

`evaluation.py` exposes `evaluate(predictions, manifest, targets, seeds, ...)`, `log_likelihood_bits`, `repeat_switch_bits`, and `day_bootstrap_bits`. It fits no models. Run `PYTHONPATH=src python3 -m whale_coda_dynamics.evaluation --config config/rq1_evaluation.json` once actual predictions exist; the default prediction file is not generated yet.

Prediction JSONL schema: `coda_id`, `fold_id`, `condition`, integer `seed`, and `probabilities` (label-to-probability object containing exactly the fold vocabulary). Every comparison target must appear once for every condition and configured seed. Use one seed for deterministic reference models, three for neural models. Predictor metadata cannot override the target, day or own-last label, which are recovered from the manifest and prepared rows. The CLI rejects stale target hashes, missing/extra/duplicate predictions, mismatched vocabularies, non-normalized distributions, and zero/nonfinite probabilities. Zero is rejected rather than silently clipped; models must provide finite positive probabilities. Majority-class hard predictions are evaluated separately through `classification_metrics`, not given artificial likelihoods.

Likelihoods are averaged across seeds per coda before pairing conditions. Accuracy and macro-F1 are computed per seed and averaged; macro-F1 uses the union of observed true/predicted labels for that seed. Argmax ties use sorted labels. The 2,000-replicate seeded percentile bootstrap samples complete days with replacement, recomputing token-weighted means. Reports include positive-day counts, three primary contrasts, all 12 add-one-input pairs and negative-contrast flags. The additional A_B versus better A/B comparison is descriptive: its interval conditions on the selected single-history reference.

Repeat/switch decomposition uses the fold's mapped vocabulary so the algebra matches the scored likelihood; raw and mapped repeat counts are both reported. Switch-choice contributions are averaged over all eligible targets, not just switches. Targets lacking own history retain their total likelihood but are excluded from decomposition. Prepared RQ1 comparison targets all have own history. CIs with 14 days remain approximate.

Outputs: `scored_predictions.jsonl`, `paired_contrasts.jsonl`, and `summary.json`, with configuration and input hashes. The original prediction JSONL retains full probability vectors. No synthetic validation predictions are saved as research outputs. Tests include mathematical identities, seed log-averaging, bootstrap weighting, and invalid/missing predictions; the complete 14-fold manifest was also exercised with 12,376 synthetic uniform predictions.

## Logistic-regression reference runner

Install the optional modeling dependencies with `python -m pip install '.[modeling]'` in a virtual environment. Run `PYTHONPATH=src .venv/bin/python -m whale_coda_dynamics.logistic` using `config/rq1_logistic.json`. NumPy and SciPy are modeling-only dependencies; preparation and evaluation remain standard-library code.

The runner reconstructs each fold with `prepare_fold`, verifies its policy against the manifest, and trains seven contextual multinomial models plus the unigram condition. It minimizes mean cross-entropy plus fixed L2 (0.01/2 times squared coefficients, including intercepts) with L-BFGS-B. Penalizing intercepts keeps probabilities finite for full-vocabulary classes absent from training targets, without adding synthetic targets. It checks solver success and gradient magnitude, and fails on nonconvergence. The last three history lags use one-hot labels; missing lags are zero vectors only within enabled streams. Timing includes raw and log1p features, each standardized using training comparison targets. Turn categories are fitted on training only; unseen categories yield zero vectors. Validation days remain excluded because no hyperparameter tuning or early stopping is performed.

The repeat baseline estimates mapped-label repetition frequency from training comparison targets, then allocates the remaining mass by the fold's add-1 prior excluding own-last. Degenerate repeat rates fail explicitly. Majority-class predictions get accuracy/macro-F1 only. Saved models include feature names, vocabulary, scaling, categories and weights; convergence diagnostics are saved per fit.

`data/results/rq1/` holds probabilities, baseline predictions, models and convergence records; `evaluation/` holds shared-evaluator outputs with versions/configuration/input hashes. `reports/rq1_logistic.md` is generated. All 98 contextual fits converged in the first run, but five add-input contrasts were negative and the full model fell slightly below repeat-own-last in LL. These diagnostics remain unresolved research/model-capacity checks, not evidence that partner content or timing is harmful. No settings were changed after inspecting test results. Retrospective-delay sensitivity and neural models remain pending.

**Per-field lag order (DEC-25).** `Encoder(..., order, history_orders)` and the config key `history_orders` (e.g. `{"A_history": 9}`) give a history field its own number of lags; fields not listed keep `history_order`. Without `history_orders` the features, their order and every output are byte-identical to before; `models.jsonl` records `history_orders` only when it is set, and `Encoder.from_record` reads it back (absent: `order` for every field). The v2–v4 fold loop is `fit_v2_lattice(config, manifest, rows, evaluation_config)` (no file output); `_fit_and_evaluate_v2` wraps it and writes the files.

**Own-history length sweep (`history_length.py`, PLAN `[P2]`).** Config `config/rq1_history_length.json`; every fitting process needs `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `OMP_NUM_THREADS` set to 1 (it refuses otherwise). `--workers` (default 4) sets the number of processes; each imports torch through `residual.py` (about 230 MB before any data), so raise it only with RAM to spare. `--stage fit [--orders k ...]` runs one spawned process per k, each fitting all eight conditions with `fit_v2_lattice` and writing `data/results/rq1_history_length/k<k>/` (`summary.json`, `l2_selection.jsonl`, `scored.jsonl` for A and A_B_S); the k = 3 process also compares itself with the committed v4 run (the gate). `--stage calibrate` reuses `residual_planted.py`'s planting per fold and fraction. `--stage report` reads only those outputs and writes `summary.json` and `reports/rq1_history_length.md`, with both stop rules per k (DEC-16's, and DEC-23's through `logistic_v5.dec23_stop_rules` on each check's stored day means times the day's target count) and the two-sided sign-flip p for the primary and secondary contrasts; it regenerates byte-identically without refitting. No `predictions.jsonl` or `models.jsonl` is written.
