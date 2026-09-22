# RQ2 preparation (Sharma, Gero, Payne, et al. rhythm/tempo labeling)

Mirrors reports/rq1.md's structure and reuses RQ1's exchange grouping, feature construction, split
assignment, and validation logic unchanged. See the Deviations section below before comparing counts
directly against RQ1: several source assumptions do not transfer to this release.

Raw codas: 3840. Retained target rows: 1742. Comparison targets: 1318.
Excluded for insufficient history: 424. Strict focal subset: 0 comparison targets (see Deviations: no Focal column exists, so this is always 0).

History: up to 10 completed codas per caller, oldest first; at least 1 per caller for comparison.

| Split | Deployments | Recordings (REC) | Exchanges with targets | Comparison targets |
|---|---:|---:|---:|---:|
| train | 9 | 61 | 72 | 854 |
| validation | 2 | 39 | 45 | 332 |
| test | 2 | 10 | 14 | 132 |

## Deviations from the RQ1 pipeline

- **No Tag/Bout/Date/Focal columns.** Sharma's release provides only REC, Whale, TsTo, Duration,
  nClicks, and ICI columns. `tag_id` is set equal to `recording_id` (REC) rather than a distinct
  concept; bout/exchange boundaries use REC alone as the sole hard-boundary field (no Bout to add).
  `strict_focal_subset` is therefore always False here (no Focal column to define it from).
- **No calendar date.** Day-held-out splitting is replaced with deployment-held-out splitting: the
  `recording_date` column (kept under that name only for schema parity with `rq1_day_splits.csv`)
  holds a deployment prefix parsed from REC with the pattern `^sw\\d+` (e.g. `sw061` from
  `sw061b001_124`), giving 13 deployment groups. This is not a date and carries no
  chronological meaning; it exists only so recordings from the same probable tag deployment stay
  together across splits, in the same spirit as RQ1's same-day rule.
- **Wider caller-code range.** Whale codes span 1-11 in this release (not just 1/2 as initially
  described); the same exactly-two-distinct-callers filter is applied, so this excludes
  proportionally more multi-caller segments than in the Assayag file. Retained codes remain
  provisional local caller codes, not confirmed individual whales.
- **No shared key for the rhythms.p join.** Coda-type labels come from `rhythms.p`, a plain list
  aligned to `sperm-whale-dialogues.csv` strictly by row index; there is no key column to verify
  this join independently, unlike Assayag's `codaNUM2018`.
- **No per-row source identifier.** Unlike `codaNUM2018`, there is no natural per-row ID in this
  release; `coda_id`/`source_coda_id` are built from the zero-based row index (`sharma:row:<i>`).
- **One corrupt-duration row can exclude a large recording.** The same conservative rule as RQ1
  (`exchanges.py`: any invalid onset/duration in a hard-boundary group excludes the whole group)
  is unchanged here, but its impact is larger: a single `Duration=0` row in `sw061b003` (731 rows,
  the largest REC in the file) excludes that entire recording. See Exchange exclusions below.
- **UNK policy for unseen coda types.** See the Label vocabulary section below.

## Validation

passed: coverage, history timing/identity/order, target alignment, deployment/recording/exchange separation

See reports/data_dictionary.md for the RQ1 feature definitions this module reuses and README.md
for modeling limitations.

Types in train absent from training (post-UNK-mapping): [].
Types in validation absent from training (post-UNK-mapping): [].
Types in test absent from training (post-UNK-mapping): [].

## RQ2 input conditions

All eight conditions use identical comparison targets, deployment-held-out splits, and UNK mapping.

- none: no context (training-only label prior)
- S: turn_category, gap_s, overlap_elapsed_s
- A: A_history
- B: B_history
- A_B: A_history, B_history
- A_S: A_history, turn_category, gap_s, overlap_elapsed_s
- B_S: B_history, turn_category, gap_s, overlap_elapsed_s
- A_B_S: A_history, B_history, turn_category, gap_s, overlap_elapsed_s

Primary RQ2 contrast (parallel to RQ1): held-out log-likelihood(A_B_S) minus log-likelihood(B_S).
Timing convention: onset_available. S holds only what is known at target onset, relative to the immediately preceding coda of either caller: turn_category, gap_s (end-to-start, continuations and switches) and overlap_elapsed_s (onset-to-onset, overlaps). retrospective_delay_s is a sensitivity field in no S list.
Deployment group replaces recording day; see Deviations above. Model folds and training seeds remain downstream evaluation work.

## Interactional state

| turn_category | Comparison targets | Max gap_s | Max overlap_elapsed_s |
|---|---:|---:|---:|
| exchange_start | 0 | 0.000 | 0.000 |
| self_continuation | 388 | 6.967 | 0.000 |
| switch | 557 | 6.755 | 0.000 |
| overlap | 373 | 0.000 | 1.461 |

Same-caller overlaps (preceding coda is the target caller's own): 0.
Targets starting after the preceding coda ends but before an earlier coda ends: 0.
Sensitivity field: targets whose retrospective_delay_s uses a post-onset endpoint: 373; overlaps_prior_interlocutor: 373.

## Label vocabulary and UNK policy

Raw rhythm-cluster labels: 18 distinct values (0-17).
Training vocabulary (targets and history entries observed in train): 18 labels, plus the catch-all `UNK` token.
Validation/test target labels remapped to `UNK`: 0.
Validation/test history entries remapped to `UNK`: 0.
Smoothing: additive (alpha=1.0) over training comparison-target counts, applied identically across all eight conditions' shared no-context baseline (`data/processed/rq2_label_prior.json`), not as a separate per-condition parameter.
`Ct_raw` preserves the pre-UNK-mapping label on every row for audit; `Ct`, `A_history`, and `B_history` hold the mapped values used for modeling.


## Audit flags

Counts can overlap; row-level details are in codas_normalized_rq2.jsonl.

- invalid_coda_click_count: 14
- invalid_duration: 8
- shared_recording_onset: 14

## Exchange exclusions

- not_exactly_two_callers: 1164 codas
- group_invalid_timing: 875 codas
- tied_onsets: 107 codas

## Gap sensitivity

| Gap (s) | Exchanges | Codas | Targets with both histories |
|---:|---:|---:|---:|
| 5 | 178 | 1770 | 1275 |
| 7 | 143 | 1742 | 1318 |
| 10 | 104 | 1583 | 1255 |

## Reproducibility

Combined source SHA-256 (CSV + rhythms.p): `ca49183e39c27b1a9db4aa98c9d8ecb636ec087eb8a652bb0d712946cb53d5a5`

Run configuration:

```json
{
  "rq2": {
    "exchange_config": "config/exchanges_rq2.json",
    "history_limit": 10,
    "minimum_history_per_caller": 1,
    "provisional_assumptions_accepted": true,
    "split_seed": 42,
    "output_directory": "data/processed",
    "report_directory": "reports",
    "validation_day_fraction": 0.15,
    "test_day_fraction": 0.15,
    "timing_convention": "onset_available",
    "unk_token": "UNK",
    "unk_smoothing_alpha": 1.0,
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
    "normalization_config": "config/preprocessing_rq2.json",
    "input": "data/interim/codas_normalized_rq2.jsonl",
    "output_directory": "data/processed",
    "gap_threshold_s": 7,
    "sensitivity_thresholds_s": [
      5,
      7,
      10
    ],
    "hard_boundary_fields": [
      "recording_id"
    ],
    "caller_field": "source_whale_code"
  },
  "normalization": {
    "input": "data/raw/sperm-whale-dialogues.csv",
    "rhythms_input": "data/raw/rhythms.p",
    "output_directory": "data/interim",
    "expected_rows": 3840,
    "missing_tokens": [
      "",
      "nan",
      "null",
      "none",
      "na"
    ],
    "source_url": "https://github.com/pratyushasharma/sw-combinatoriality/tree/main/data"
  }
}
```

Output SHA-256 checksums:

- `rq2_comparison.csv`: `b44b1324a0bc21fb2f7996406e5875b56a1aa29dae14654299587f01a6508c02`
- `rq2_comparison.jsonl`: `a44fab75b256cdf80e6bacc77fcb158ffc0709e4ad31b8dc1cb9138a7442a63d`
- `rq2_day_splits.csv`: `81e5ace2c9d0ab0966f12deda83e8564485334d1496af2b837e051bf7788a2fb`
- `rq2_day_splits.jsonl`: `cf78530ac7c249d26dd5d204887df592b82a0ca4e9faa332cf813a1d328ad4e2`
- `rq2_exchange_assignments.jsonl`: `15abbd0a11108ee802cf1dec559fdf6f46231da9749da10020c018e164782dad`
- `rq2_exchange_segments.jsonl`: `319480aa91d4265167ff544895797002869deef4d47f1c8034e45d1b4d2a00cb`
- `rq2_label_prior.json`: `5dffcbe9485836e0f43f94ae1a8102dec855753ff4d223944584ee1306baffd6`
- `rq2_targets.csv`: `82bcd0028f92e6d3a3242e670258cc0fa11053a56eee0de37db9bb283b477e1e`
- `rq2_targets.jsonl`: `9d79b368862ffdf7d7ffbd573dd61d00c2683c5b766ffd8eeca8ade67eb430e3`
