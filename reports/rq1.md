# RQ1 preparation

Raw codas: 3933. Retained target rows: 2075. Comparison targets: 1547.
Excluded for insufficient history: 528. Strict focal subset: 1131 comparison targets.

History: up to 10 completed codas per caller, oldest first; at least 1 per caller for comparison.

| Split | Days | Tags | Exchanges with targets | Comparison targets |
|---|---:|---:|---:|---:|
| train | 10 | 14 | 128 | 1247 |
| validation | 2 | 3 | 12 | 114 |
| test | 2 | 4 | 17 | 186 |

## Validation

passed: coverage, history timing/identity/order, target alignment, day/tag/exchange separation

See reports/data_dictionary.md for feature definitions and README.md for modeling limitations.

Types in train absent from training: [].
Types in validation absent from training: ['1+31'].
Types in test absent from training: [].

## RQ1 input conditions

All eight conditions use identical comparison targets and day-held-out splits.

- none: no context (training-only label prior)
- S: turn_category, gap_s, overlap_elapsed_s
- A: A_history
- B: B_history
- A_B: A_history, B_history
- A_S: A_history, turn_category, gap_s, overlap_elapsed_s
- B_S: B_history, turn_category, gap_s, overlap_elapsed_s
- A_B_S: A_history, B_history, turn_category, gap_s, overlap_elapsed_s

Primary RQ1 contrast: held-out log-likelihood(A_B_S) minus log-likelihood(B_S).
Timing convention: onset_available. S holds only what is known at target onset, relative to the immediately preceding coda of either caller: turn_category, gap_s (end-to-start, continuations and switches) and overlap_elapsed_s (onset-to-onset, overlaps). retrospective_delay_s is a sensitivity field in no S list.
Recording day is the normalized source Date, not a reconstructed absolute timestamp. Model folds and training seeds remain downstream evaluation work.

## Interactional state

| turn_category | Comparison targets | Max gap_s | Max overlap_elapsed_s |
|---|---:|---:|---:|
| exchange_start | 0 | 0.000 | 0.000 |
| self_continuation | 461 | 6.967 | 0.000 |
| switch | 645 | 6.755 | 0.000 |
| overlap | 441 | 0.000 | 2.256 |

Same-caller overlaps (preceding coda is the target caller's own): 0.
Targets starting after the preceding coda ends but before an earlier coda ends: 0.
Sensitivity field: targets whose retrospective_delay_s uses a post-onset endpoint: 441; overlaps_prior_interlocutor: 441.

## Audit flags

Counts can overlap; row-level details are in codas_normalized.jsonl.

- coda_label_disagreement: 265
- history_crosses_bout: 72
- history_crosses_rec: 6
- missing_focal: 157
- missing_or_invalid_tag_on_time: 52
- past_history_crosses_whale_code: 946
- shared_tag_onset: 14
- unresolved_individual_id: 2510

## Exchange exclusions

- not_exactly_two_callers: 1790 codas
- tied_onsets: 156 codas

## Gap sensitivity

| Gap (s) | Exchanges | Codas | Targets with both histories |
|---:|---:|---:|---:|
| 5 | 214 | 2108 | 1492 |
| 7 | 172 | 2075 | 1547 |
| 10 | 123 | 1857 | 1446 |

## Reproducibility

Source SHA-256: `17b37605e56a21fb96bff301a2d43c4139693f7def23d4e3801d046026069856`

Run configuration:

```json
{
  "rq1": {
    "exchange_config": "config/exchanges.json",
    "history_limit": 10,
    "minimum_history_per_caller": 1,
    "provisional_assumptions_accepted": true,
    "split_seed": 42,
    "output_directory": "data/processed",
    "report_directory": "reports",
    "validation_day_fraction": 0.15,
    "test_day_fraction": 0.15,
    "timing_convention": "onset_available",
    "input_conditions": {
      "none": [],
      "S": [
        "turn_category",
        "gap_s",
        "overlap_elapsed_s"
      ],
      "A": [
        "A_history"
      ],
      "B": [
        "B_history"
      ],
      "A_B": [
        "A_history",
        "B_history"
      ],
      "A_S": [
        "A_history",
        "turn_category",
        "gap_s",
        "overlap_elapsed_s"
      ],
      "B_S": [
        "B_history",
        "turn_category",
        "gap_s",
        "overlap_elapsed_s"
      ],
      "A_B_S": [
        "A_history",
        "B_history",
        "turn_category",
        "gap_s",
        "overlap_elapsed_s"
      ]
    }
  },
  "exchanges": {
    "normalization_config": "config/preprocessing.json",
    "input": "data/interim/codas_normalized.jsonl",
    "output_directory": "data/processed",
    "gap_threshold_s": 7,
    "sensitivity_thresholds_s": [
      5,
      7,
      10
    ],
    "hard_boundary_fields": [
      "tag_id",
      "recording_id",
      "source_bout_code"
    ],
    "caller_field": "source_whale_code"
  },
  "normalization": {
    "input": "data/raw/dswp-2014-2016-retagged-maor-2025.csv",
    "output_directory": "data/interim",
    "expected_rows": 3933,
    "date_month_aliases": {
      "מרץ": "03",
      "מאי": "05"
    },
    "missing_tokens": [
      "",
      "nan",
      "null",
      "none",
      "na"
    ],
    "source_url": "https://github.com/MaorAssayag/coda-sequences-relations-and-prediction/blob/main/dataset/dswp-2014-2016-retagged-maor-2025.csv"
  }
}
```

Output SHA-256 checksums:

- `rq1_comparison.csv`: `e02858ded71b496a8c7a1a58235b136680118f17cca9e31a8d2fc58ee3aa5489`
- `rq1_comparison.jsonl`: `7ce8b3c8a293f57c1996f5179fb34999f08e183ad88de0f9aea191e6de9aaee6`
- `rq1_day_splits.csv`: `51641340353ba5b12c24e602c32c5bbc7a22406b6fd63ddb9787cc1e49a4171e`
- `rq1_day_splits.jsonl`: `17c707260ceaf1d867ce19a1e338981b4e0ce510201b5ca42963997530b45b3b`
- `rq1_targets.csv`: `113190459bb1da68cefa966bb7d68ab1aac496bae313f6c6d065cdcb591366a5`
- `rq1_targets.jsonl`: `bef05da838d3e4d5c080944466185c1e6ea5c1573f020e48b0b1274bdc49937a`
