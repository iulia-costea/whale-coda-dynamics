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

This preparation uses **retrospective conditional analysis**. S includes the
signed gap from the most recently started earlier interlocutor coda's end to the
target onset, plus the category based on the immediately preceding caller.
Negative delays use an endpoint after target onset. The
`delay_uses_post_onset_endpoint` flag identifies those rows and
`timing_convention` records this choice. This is not a real-time forecasting
claim, and timing-conditioned gains must be interpreted accordingly. Histories
still contain only completed codas strictly preceding target onset.

Fit model parameters, input encoders and scaling on training only; tune on
validation and reserve test for final evaluation. Validation contains a coda type
(`1+31`) absent from training: define a consistent unseen-label/output-vocabulary
policy for all eight conditions without moving or silently dropping held-out rows.
The no-context likelihood baseline needs a smoothed training label distribution,
not just a hard majority-class prediction.

The current manifest is one reproducible day-held-out split. Multiple folds and
model seeds, confidence intervals, and the proposal's timing/content separability
checks remain downstream modeling work. No models have been trained yet.

## Source documentation

- [Data dictionary, provenance, and assumptions](reports/data_dictionary.md)
- [Assayag et al. paper](https://doi.org/10.1080/09524622.2026.2684665)
