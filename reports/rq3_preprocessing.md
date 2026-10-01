# RQ3 preprocessing

Raw codas: 32956. Primary-eligible codas: 31479.

## Sources

| Dataset | Raw rows | SHA-256 |
|---|---:|---|
| caribbean (`data/raw/CarribeanCodas.csv`) | 8719 | `c8ad7b47e6bc102db5a40d7dd2bbbebe8026e4271d135a8e952d68f1baddebd5` |
| pacific (`data/raw/PacificCodas.csv`) | 24237 | `96bbf264286eadc94c3d2d68229b133fa135733f759c0bad915c66ccd7a84acf` |

## Primary eligibility

3–10 clicks; accepted clan; required ICIs in [0.005, 1.0) s; no non-zero padding.
The exported `ici_sequence_s` values are continuous and unpadded. Fixed 50 ms tokenisation occurs downstream from this table.

| Dataset | Clan | Eligible codas |
|---|---|---:|
| caribbean | EC1 | 7742 |
| caribbean | EC2 | 946 |
| pacific | FP | 2556 |
| pacific | PALI | 2052 |
| pacific | PO | 1985 |
| pacific | REG | 8107 |
| pacific | RI | 1747 |
| pacific | SH | 5138 |
| pacific | SI | 1206 |

## Grouping and folds

`group_id` is Caribbean canonical `recording_date` or Pacific source `grpvar`. Each group is assigned whole to one deterministic fold.

| Dataset | Clan | Groups | Fold coda counts |
|---|---|---:|---|
| caribbean | EC1 | 91 | 0: 1550, 1: 1549, 2: 1547, 3: 1549, 4: 1547 |
| caribbean | EC2 | 6 | 0: 363, 1: 143, 2: 133, 3: 129, 4: 116, 5: 62 |
| pacific | FP | 26 | 0: 508, 1: 506, 2: 505, 3: 534, 4: 503 |
| pacific | PALI | 15 | 0: 409, 1: 408, 2: 409, 3: 404, 4: 422 |
| pacific | PO | 15 | 0: 442, 1: 380, 2: 382, 3: 405, 4: 376 |
| pacific | REG | 50 | 0: 1962, 1: 1522, 2: 1541, 3: 1542, 4: 1540 |
| pacific | RI | 19 | 0: 346, 1: 342, 2: 349, 3: 340, 4: 370 |
| pacific | SH | 50 | 0: 1027, 1: 1022, 2: 1031, 3: 1029, 4: 1029 |
| pacific | SI | 16 | 0: 237, 1: 251, 2: 249, 3: 232, 4: 237 |

## Caribbean recording-date normalization

Caribbean `Date` values are parsed day-first from `dd/mm/yyyy` or `dd-mm-yyyy` into canonical ISO `recording_date` values. The raw `source_date` is preserved unchanged. `group_id` is `recording_date` for Caribbean codas and the source `grpvar` repertoire ID for Pacific codas.

## Exclusions

- click_count_outside_primary_range: 823
- ici_at_or_above_maximum: 13
- ici_below_minimum: 43
- invalid_required_ici: 5
- missing_or_unaccepted_clan: 682
- nonzero_padding: 2

## Eligible click counts

| Click count | Codas |
|---:|---:|
| 3 | 3965 |
| 4 | 4193 |
| 5 | 13249 |
| 6 | 3491 |
| 7 | 2698 |
| 8 | 2261 |
| 9 | 1085 |
| 10 | 537 |

## Identity labels

Pacific identity labels use `config/rq3_identity_mapping.json` (Hersh et al. (2022), Supplementary Information, Table S2). Caribbean codas are `unmapped`; Pacific `#N/A` source types are also `unmapped` and are excluded from identity-stratified analyses.

| Identity status | Eligible codas |
|---|---:|
| identity | 8160 |
| non_identity | 14631 |
| unmapped | 8688 |

## Reproducibility

Run configuration:

```json
{
  "caribbean_input": "data/raw/CarribeanCodas.csv",
  "pacific_input": "data/raw/PacificCodas.csv",
  "interim_output": "data/interim/rq3/rq3_codas_normalized.jsonl",
  "processed_output": "data/processed/rq3/rq3_codas.csv",
  "metadata_output": "data/processed/rq3/rq3_codas_metadata.csv",
  "folds_output": "data/processed/rq3/rq3_folds.csv",
  "identity_mapping_input": "config/rq3_identity_mapping.json",
  "report_output": "reports/rq3_preprocessing.md",
  "primary_min_clicks": 3,
  "primary_max_clicks": 10,
  "primary_min_ici_s": 0.005,
  "primary_max_ici_s": 1.0,
  "accepted_caribbean_clans": [
    "EC1",
    "EC2"
  ],
  "accepted_pacific_clans": [
    "FP",
    "PALI",
    "PO",
    "REG",
    "RI",
    "SH",
    "SI"
  ]
}
```

Output SHA-256 checksums:

- `rq3_codas.csv`: `82ab3cc2de8e0cd5e31ddc18e238bd1ab1a55776e5091cc75afbb1ae3e022ffa`
- `rq3_codas_metadata.csv`: `6bccfdf0056806383e24a57bedf0f952ab64184b568b45aea1875c9fa236c08d`
- `rq3_codas_normalized.jsonl`: `035a1e1e4f50cf843696d07cab244389d0935661c3f268389b86a68dab42a58b`
- `rq3_folds.csv`: `272bb592515d6f11918a6914d2b087d4f530f0b2a60c98e6bb066248d52cb9ce`
