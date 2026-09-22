# whale-coda-dynamics

RQ1 asks whether sperm whale coda prediction improves when interlocutor history
is added to the caller's own history.

## Run

Python 3.10+; no third-party runtime dependencies or `.env` required.
Place the original [Assayag dataset](https://github.com/MaorAssayag/coda-sequences-relations-and-prediction/tree/main/dataset)
in `data/raw/dswp-2014-2016-retagged-maor-2025.csv`, then run from the repository root:

```sh
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.prepare
```

This rebuilds normalization, provisional exchanges, features and day-grouped
splits. Built-in validation checks history timing, caller and exchange boundaries,
response alignment, coverage, and split separation. There is no separate test suite.
The original CSV is never modified. Raw data, prepared tables, and the preparation
report are included in the repository for the project handoff. Re-running the
pipeline may update generated files; review those changes before committing.

## Files to use

- **`data/processed/rq1_comparison.csv`**: 1,547 shared targets for all eight input conditions.
- `data/processed/rq1_targets.jsonl`: all 2,075 retained codas, including context
  rows without enough history to be comparison targets.
- `data/processed/rq1_day_splits.csv`: recording-day split assignments.
- **`reports/rq1.md`**: the single report with counts, audit flags, exclusions,
  threshold sensitivity, validation, configuration and checksums.

The three RQ1 tables also have CSV/JSONL counterparts. CSV history cells are JSON
arrays: parse with `json.loads`. JSONL preserves lists and null values directly.

Intermediate files support traceability: `data/interim/codas_normalized.jsonl`
preserves all 3,933 source rows, original values and row-level audit issues;
`data/processed/exchange_assignments.jsonl` and `exchange_segments.jsonl` record
provisional groups and exclusion decisions. No separate audit or exchange reports
are generated. Re-running replaces generated outputs.

## Study assumptions

- Hard boundaries: `(Tag, REC, Bout)`. Split when the gap from the immediately
  preceding coda's end exceeds 7 seconds.
- Retain segments with exactly two local Whale codes, valid timing/types, and
  no tied onsets. These are provisional caller identities, not identified animals.
- Histories contain up to 10 completed prior codas per caller, oldest first.
  The comparison sample requires at least one from each caller.
- Relative timing is used; the absolute timestamp origin remains unresolved.

The settings live in `config/preprocessing.json`, `config/exchanges.json`, and
`config/rq1.json`. The report compares grouping thresholds of 5, 7, and 10 seconds;
final tables use seven seconds. `strict_focal_subset` marks a stricter caller subset.

## Modeling

| Split | Recording days | Tags | Targets |
|---|---:|---:|---:|
| Train | 10 | 14 | 1,247 |
| Validation | 2 | 3 | 114 |
| Test | 2 | 4 | 186 |

Splits hash seed 42 plus normalized source Date, allocating fractions by day
count. All tags recorded on the same day stay together. Dates are source recording
labels; we have not reconstructed wall-clock timestamps or midnight rollovers.
The pipeline rejects missing dates and tags spanning multiple date labels.
Day separation does not imply different whales across partitions.

### Eight RQ1 conditions

The exact input columns are specified in `config/rq1.json`:

| Condition | Inputs |
|---|---|
| None | Training-only label prior |
| S | Turn category and signed response delay |
| A | Own history |
| B | Interlocutor history |
| A, B | Both histories |
| A, S | Own history and timing |
| B, S | Interlocutor history and timing |
| A, B, S | Both histories and timing |

All conditions use the same targets and splits. The main RQ1 contrast is held-out
log-likelihood of **A,B,S minus B,S**: the additional predictive value of own
history conditional on the partner and timing. Partner and timing contrasts are
A,B,S minus A,S and A,B,S minus A,B respectively.

### Timing convention

S contains only what is known at the target's onset (`timing_convention` =
`onset_available`, PLAN [D5]): `turn_category` (exchange_start /
self_continuation / switch / overlap, relative to the immediately preceding coda
of either whale), `gap_s` (end-to-start, continuations and switches) and
`overlap_elapsed_s` (onset-to-onset, overlaps). The old signed delay to the last
interlocutor coda is kept as `retrospective_delay_s` for a sensitivity run only
(`timing_convention` = `retrospective_signed_end_to_start`); negative values use
an endpoint after target onset, flagged by `delay_uses_post_onset_endpoint`.
Histories still contain only completed codas strictly preceding target onset.

Fit model parameters, input encoders and scaling on training only; tune on
validation and reserve test for final evaluation. Validation contains a coda type
(`1+31`) absent from training: define a consistent unseen-label/output-vocabulary
policy for all eight conditions without moving or silently dropping held-out rows.
The no-context likelihood baseline needs a smoothed training label distribution,
not just a hard majority-class prediction.

The current manifest is one reproducible day-held-out split. Multiple folds and
model seeds, confidence intervals, and the proposal's timing/content separability
checks remain downstream modeling work. No models have been trained yet.

## RQ2: a different coda-type labeling scheme

RQ2 tests whether the RQ1 contrasts hold under Sharma, Gero, Payne, et al. (2024)'s
rhythm/tempo coda-type labeling, on a substantially overlapping recording set.
Place `sperm-whale-dialogues.csv` and `rhythms.p` in `data/raw/` (already present),
then run:

```sh
PYTHONPATH=src python3 -m whale_coda_dynamics.preprocessing.prepare_rq2
```

This reuses RQ1's exchange grouping, feature construction, split assignment, and
validation logic unchanged (`exchanges.py`, `features.py`, `prepare.py`); only
normalization (`preprocessing/audit_rq2.py`) and orchestration
(`preprocessing/prepare_rq2.py`) are new, since Sharma's release has a different
schema. Several RQ1 assumptions do not transfer cleanly: there is no Tag, Bout,
Date, or Focal column, and no calendar date, so day-held-out splitting is replaced
with deployment-held-out splitting (grouped by a deployment prefix parsed from
REC, not a date). Coda types unseen in training are mapped to a shared `UNK`
token with a smoothed training label prior, rather than dropped. See
**`reports/rq2.md`** for the full accounting of these deviations, counts, and
checksums. Outputs are `data/processed/rq2_comparison.csv`,
`rq2_targets.jsonl`, `rq2_day_splits.csv`, and `rq2_label_prior.json`.

## Source documentation

- [Data dictionary, provenance, and assumptions](reports/data_dictionary.md)
- [Assayag et al. paper](https://doi.org/10.1080/09524622.2026.2684665)
