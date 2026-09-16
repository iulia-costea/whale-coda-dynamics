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
- S: turn_category, response_delay_s
- A: A_history
- B: B_history
- A_B: A_history, B_history
- A_S: A_history, turn_category, response_delay_s
- B_S: B_history, turn_category, response_delay_s
- A_B_S: A_history, B_history, turn_category, response_delay_s

Primary RQ1 contrast: held-out log-likelihood(A_B_S) minus log-likelihood(B_S).
Timing convention: retrospective signed end-to-start delay to the most recently started prior interlocutor coda. Negative delay uses an endpoint after target onset; this is not online onset-time prediction.
Comparison targets with post-onset endpoints: 441.
Recording day is the normalized source Date, not a reconstructed absolute timestamp. Model folds and training seeds remain downstream evaluation work.

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
    "timing_convention": "retrospective_signed_end_to_start",
    "input_conditions": {
      "none": [],
      "S": [
        "turn_category",
        "response_delay_s"
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
        "response_delay_s"
      ],
      "B_S": [
        "B_history",
        "turn_category",
        "response_delay_s"
      ],
      "A_B_S": [
        "A_history",
        "B_history",
        "turn_category",
        "response_delay_s"
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

- `rq1_comparison.csv`: `783538aba334ee98f6ad25d4f5cf8128c5e4c31304df637a169f96ce904a44b5`
- `rq1_comparison.jsonl`: `b86f11d7f1dc840eee4ac6915dbce7cf11ea49e78acf2d398463cb40d43c3fe4`
- `rq1_day_splits.csv`: `51641340353ba5b12c24e602c32c5bbc7a22406b6fd63ddb9787cc1e49a4171e`
- `rq1_day_splits.jsonl`: `17c707260ceaf1d867ce19a1e338981b4e0ce510201b5ca42963997530b45b3b`
- `rq1_targets.csv`: `cb136f805907687c6d718a32e7fc1f2ff7bae40715a016ee4030e33bb122d921`
- `rq1_targets.jsonl`: `b4876668f962c4878f49f8c6ae79f545f4ed7ae021edeb933769d61eb941084a`
