# RQ3 preprocessing

Raw codas: 32956. Primary-eligible codas: 31528.

## Sources

| Dataset | Raw rows | SHA-256 |
|---|---:|---|
| caribbean (`data/raw/CarribeanCodas.csv`) | 8719 | `c8ad7b47e6bc102db5a40d7dd2bbbebe8026e4271d135a8e952d68f1baddebd5` |
| pacific (`data/raw/PacificCodas.csv`) | 24237 | `96bbf264286eadc94c3d2d68229b133fa135733f759c0bad915c66ccd7a84acf` |

## Primary eligibility

3–10 clicks; accepted clan; finite positive required ICIs; no non-zero padding.
The exported `ici_sequence_s` values are continuous and unpadded. No global dICI bins were fitted.

| Dataset | Clan | Eligible codas |
|---|---|---:|
| caribbean | EC1 | 7751 |
| caribbean | EC2 | 948 |
| pacific | FP | 2564 |
| pacific | PALI | 2059 |
| pacific | PO | 1986 |
| pacific | REG | 8108 |
| pacific | RI | 1750 |
| pacific | SH | 5153 |
| pacific | SI | 1209 |

## Exclusions

- click_count_outside_primary_range: 823
- invalid_required_ici: 5
- missing_or_unaccepted_clan: 682
- nonzero_padding: 2

## Eligible click counts

| Click count | Codas |
|---:|---:|
| 3 | 3966 |
| 4 | 4197 |
| 5 | 13252 |
| 6 | 3504 |
| 7 | 2711 |
| 8 | 2267 |
| 9 | 1088 |
| 10 | 543 |

## Identity labels

Identity labels are not included in this artifact. An authoritative clan/type mapping is required before producing the identity-versus-non-identity analysis dataset.

## Reproducibility

Run configuration:

```json
{
  "caribbean_input": "data/raw/CarribeanCodas.csv",
  "pacific_input": "data/raw/PacificCodas.csv",
  "interim_output": "data/interim/rq3/rq3_codas_normalized.jsonl",
  "processed_output": "data/processed/rq3/rq3_codas.csv",
  "metadata_output": "data/processed/rq3/rq3_codas_metadata.csv",
  "report_output": "reports/rq3_preprocessing.md",
  "primary_min_clicks": 3,
  "primary_max_clicks": 10,
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

- `rq3_codas.csv`: `c96aaa945aea78fe59ba4cd3531965a1d2ba186efb8e92bc6ad6a4ad41ca95d1`
- `rq3_codas_metadata.csv`: `33087d690cacef2f7b56b35494698b366cd8cc0d911bfc66a77a25d585786072`
- `rq3_codas_normalized.jsonl`: `6b7854748433048a9156f3caee093410454277f4d820ac46fb19d96c7b713148`
