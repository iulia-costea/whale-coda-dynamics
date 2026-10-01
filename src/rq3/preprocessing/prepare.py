"""Load and schema-check the raw RQ3 coda sources without filtering them."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from whale_coda_dynamics.preprocessing.io_utils import read_json, write_jsonl, write_text

from .folds import FOLD_COLUMNS, build_fold_rows


REQUIRED_CONFIG_KEYS = {
    "caribbean_input",
    "pacific_input",
    "interim_output",
    "processed_output",
    "metadata_output",
    "folds_output",
    "identity_mapping_input",
    "report_output",
    "primary_min_clicks",
    "primary_max_clicks",
    "primary_min_ici_s",
    "primary_max_ici_s",
    "accepted_caribbean_clans",
    "accepted_pacific_clans",
}

CARIBBEAN_REQUIRED_COLUMNS = {
    "codaNUM2018",
    "Date",
    "nClicks",
    "Duration",
    "CodaType",
    "Clan",
    "Unit",
    "UnitNum",
    "IDN",
    *(f"ICI{index}" for index in range(1, 10)),
}

PACIFIC_REQUIRED_COLUMNS = {
    "clan_name",
    "within_clan_correlation",
    "coda_type",
    "grpvar",
    "loc",
    "year",
    "date",
    "latitude",
    "longitude",
    "coda_number",
    "nclicks",
    "duration",
    *(f"ICI{index}" for index in range(1, 31)),
}

SOURCE_SPECS = {
    "caribbean": {
        "config_key": "caribbean_input",
        "required_columns": CARIBBEAN_REQUIRED_COLUMNS,
        "source_id_column": "codaNUM2018",
        "field_map": {
            "codaNUM2018": "source_coda_id",
            "Clan": "clan",
            "nClicks": "click_count",
            "Duration": "duration_s",
            "CodaType": "source_coda_type",
        },
        "text_fields": {
            "Clan": "clan",
            "CodaType": "source_coda_type",
            "Date": "source_date",
            "Unit": "source_unit",
            "UnitNum": "source_unit_number",
            "IDN": "source_individual_id",
        },
        "integer_fields": {"nClicks": "click_count"},
        "number_fields": {"Duration": "duration_s"},
        "ici_column_count": 9,
    },
    "pacific": {
        "config_key": "pacific_input",
        "required_columns": PACIFIC_REQUIRED_COLUMNS,
        "source_id_column": "coda_number",
        "field_map": {
            "coda_number": "source_coda_id",
            "clan_name": "clan",
            "nclicks": "click_count",
            "duration": "duration_s",
            "coda_type": "source_coda_type",
        },
        "text_fields": {
            "clan_name": "clan",
            "coda_type": "source_coda_type",
            "grpvar": "source_repertoire_id",
            "loc": "source_location",
            "year": "source_year",
            "date": "source_date",
            "within_clan_correlation": "source_within_clan_correlation",
        },
        "integer_fields": {"nclicks": "click_count"},
        "number_fields": {
            "duration": "duration_s",
            "latitude": "source_latitude",
            "longitude": "source_longitude",
        },
        "ici_column_count": 30,
    },
}

CORE_COLUMNS = [
    "rq3_coda_id",
    "source_dataset",
    "clan",
    "group_id",
    "click_count",
    "ici_sequence_s",
]

METADATA_COLUMNS = [
    *CORE_COLUMNS,
    "source_coda_id",
    "source_coda_type",
    "duration_s",
    "source_date",
    "recording_date",
    "identity_status",
    "source_unit",
    "source_unit_number",
    "source_individual_id",
    "source_repertoire_id",
    "source_location",
    "source_latitude",
    "source_longitude",
    "source_within_clan_correlation",
    "source_file",
    "source_sha256",
]


def validate_config(config):
    """Reject incomplete or malformed RQ3 preprocessing configurations."""
    missing = REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(f"RQ3 preprocessing configuration is missing keys: {sorted(missing)}")
    for key in ("caribbean_input", "pacific_input", "interim_output", "processed_output", "metadata_output", "folds_output", "identity_mapping_input", "report_output"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"Configuration value {key!r} must be a non-empty string")
    minimum, maximum = config["primary_min_clicks"], config["primary_max_clicks"]
    if type(minimum) is not int or type(maximum) is not int or minimum < 1 or minimum > maximum:
        raise ValueError("Primary click-count range must contain positive integer bounds")
    minimum_ici, maximum_ici = config["primary_min_ici_s"], config["primary_max_ici_s"]
    if (type(minimum_ici) not in (int, float) or type(maximum_ici) not in (int, float)
            or not math.isfinite(minimum_ici) or not math.isfinite(maximum_ici)
            or minimum_ici <= 0 or minimum_ici >= maximum_ici):
        raise ValueError("Primary ICI range must contain finite positive bounds in ascending order")
    for key in ("accepted_caribbean_clans", "accepted_pacific_clans"):
        clans = config[key]
        if not isinstance(clans, list) or not clans or any(not isinstance(clan, str) or not clan for clan in clans):
            raise ValueError(f"Configuration value {key!r} must be a non-empty list of clan names")
        if len(clans) != len(set(clans)):
            raise ValueError(f"Configuration value {key!r} contains duplicate clan names")


def read_source_rows(path, required_columns):
    """Read one raw CSV losslessly after checking its required schema."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns:
            raise ValueError(f"Source CSV has no header: {path}")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Source CSV has duplicate header names: {path}")
        missing = required_columns - set(columns)
        if missing:
            raise ValueError(f"Required source columns missing from {path}: {sorted(missing)}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row in {path}; no output generated")
    return rows, columns


def preliminary_record(dataset, source_file, source_row_index, source_coda_id, raw_values):
    """Create the common lossless record shared by both raw sources."""
    return {
        "source_dataset": dataset,
        "source_file": source_file,
        "source_row_index": source_row_index,
        "source_coda_id": source_coda_id,
        "raw_values": dict(raw_values),
    }


def parse_text(value, issues, field):
    """Parse optional text without changing its preserved raw representation."""
    parsed = value.strip()
    if parsed:
        return parsed
    issues.append({"code": "missing_text", "field": field, "detail": "Empty source value"})
    return None


def parse_number(value, issues, field):
    """Parse a finite numeric value, flagging rather than raising for bad rows."""
    text = value.strip()
    if not text:
        issues.append({"code": "missing_number", "field": field, "detail": "Empty source value"})
        return None
    try:
        parsed = float(text)
    except ValueError:
        issues.append({"code": "invalid_number", "field": field, "detail": text})
        return None
    if not math.isfinite(parsed):
        issues.append({"code": "nonfinite_number", "field": field, "detail": text})
        return None
    return parsed


def parse_integer(value, issues, field):
    """Parse a finite exact integer, flagging non-integral source values."""
    parsed = parse_number(value, issues, field)
    if parsed is None:
        return None
    if not parsed.is_integer():
        issues.append({"code": "invalid_integer", "field": field, "detail": value.strip()})
        return None
    return int(parsed)


def parse_caribbean_recording_date(value, issues):
    """Parse a Caribbean source date day-first into a canonical ISO calendar day."""
    text = value.strip()
    if not text:
        return None
    date_format = "%d/%m/%Y" if "/" in text else "%d-%m-%Y" if "-" in text else None
    if date_format is None:
        issues.append({"code": "invalid_recording_date", "field": "Date", "detail": text})
        return None
    try:
        return datetime.strptime(text, date_format).date().isoformat()
    except ValueError:
        issues.append({"code": "invalid_recording_date", "field": "Date", "detail": text})
        return None


def load_identity_mapping(path, accepted_pacific_clans):
    """Load and strictly validate the sourced Pacific identity-coda mapping."""
    mapping = read_json(path)
    if set(mapping) != {"source", "pacific_identity_coda_types"} or not isinstance(mapping["source"], str) or not mapping["source"]:
        raise ValueError("RQ3 identity mapping must contain a non-empty source and Pacific type mapping")
    types_by_clan = mapping["pacific_identity_coda_types"]
    if set(types_by_clan) != set(accepted_pacific_clans):
        raise ValueError("RQ3 identity mapping clans must exactly match accepted Pacific clans")
    for clan, types in types_by_clan.items():
        if (not isinstance(types, list) or not types or any(type(value) is not int or value <= 0 for value in types)
                or len(types) != len(set(types))):
            raise ValueError(f"RQ3 identity mapping has invalid coda types for {clan}")
    return mapping


def identity_status(dataset, clan, source_coda_type, mapping):
    """Classify Pacific coda types without treating missing source types as non-identity."""
    if dataset != "pacific":
        return "unmapped"
    identity_types = mapping["pacific_identity_coda_types"].get(clan)
    if identity_types is None:
        return "unmapped"
    try:
        coda_type = int(source_coda_type)
    except (TypeError, ValueError):
        return "unmapped"
    return "identity" if coda_type in identity_types else "non_identity"


def normalize_preliminary_record(record):
    """Build a typed, lossless record for one raw Caribbean or Pacific coda."""
    dataset = record["source_dataset"]
    spec = SOURCE_SPECS[dataset]
    raw = record["raw_values"]
    issues = []
    normalized = {
        "rq3_coda_id": f"rq3:{dataset}:row:{record['source_row_index']}",
        "source_dataset": dataset,
        "source_file": record["source_file"],
        "source_row_index": record["source_row_index"],
        "source_coda_id": parse_text(record["source_coda_id"], issues, spec["source_id_column"]),
        "audit_issues": issues,
        "raw_values": raw,
    }
    for source_field, normalized_field in spec["text_fields"].items():
        normalized[normalized_field] = parse_text(raw[source_field], issues, source_field)
    for source_field, normalized_field in spec["integer_fields"].items():
        normalized[normalized_field] = parse_integer(raw[source_field], issues, source_field)
    for source_field, normalized_field in spec["number_fields"].items():
        normalized[normalized_field] = parse_number(raw[source_field], issues, source_field)
    normalized["recording_date"] = (
        parse_caribbean_recording_date(raw["Date"], issues)
        if dataset == "caribbean"
        else None
    )
    normalized["group_id"] = (
        normalized["recording_date"]
        if dataset == "caribbean"
        else normalized["source_repertoire_id"]
    )
    normalized["candidate_ici_values_s"] = [
        parse_number(raw[f"ICI{index}"], issues, f"ICI{index}")
        for index in range(1, spec["ici_column_count"] + 1)
    ]
    return normalized


def normalize_raw_records(loaded, identity_mapping):
    """Normalize all loaded rows without applying the RQ3 eligibility policy."""
    return {
        dataset: {
            "columns": source["columns"],
            "records": [
                {
                    **(normalized_record := normalize_preliminary_record(record)),
                    "identity_status": identity_status(
                        normalized_record["source_dataset"],
                        normalized_record["clan"],
                        normalized_record["source_coda_type"],
                        identity_mapping,
                    ),
                }
                for record in source["records"]
            ],
        }
        for dataset, source in loaded.items()
    }


def append_eligibility_issue(record, code, detail):
    """Record a primary-analysis exclusion in the row-level audit trail."""
    record["audit_issues"].append({"code": code, "field": "", "detail": detail})


def apply_primary_eligibility(record, config):
    """Extract an unpadded ICI sequence and record primary-transfer eligibility."""
    result = dict(record)
    result["audit_issues"] = list(record["audit_issues"])
    reasons = []
    click_count = result["click_count"]
    click_count_in_range = click_count is not None and config["primary_min_clicks"] <= click_count <= config["primary_max_clicks"]
    if click_count is None:
        reasons.append("missing_or_invalid_click_count")
    elif not click_count_in_range:
        reasons.append("click_count_outside_primary_range")

    accepted_clans = config[f"accepted_{result['source_dataset']}_clans"]
    if result["clan"] not in accepted_clans:
        reasons.append("missing_or_unaccepted_clan")

    sequence = None
    if click_count_in_range:
        required = result["candidate_ici_values_s"][:click_count - 1]
        padding = result["candidate_ici_values_s"][click_count - 1:]
        if any(value is None or value <= 0 for value in required):
            reasons.append("invalid_required_ici")
        elif any(value < config["primary_min_ici_s"] for value in required):
            reasons.append("ici_below_minimum")
        elif any(value >= config["primary_max_ici_s"] for value in required):
            reasons.append("ici_at_or_above_maximum")
        if any(value is not None and value != 0 for value in padding):
            reasons.append("nonzero_padding")
        if not reasons:
            sequence = required

    for reason in reasons:
        append_eligibility_issue(result, reason, "Excluded from the primary RQ3 transfer dataset")
    result["eligible_for_primary_transfer"] = not reasons
    result["exclusion_reasons"] = reasons
    result["ici_sequence_s"] = sequence
    result["ici_count"] = len(sequence) if sequence is not None else None
    return result


def apply_primary_eligibility_to_all(normalized, config):
    """Apply the configured primary RQ3 eligibility policy without dropping rows."""
    return {
        dataset: {
            "columns": source["columns"],
            "records": [apply_primary_eligibility(record, config) for record in source["records"]],
        }
        for dataset, source in normalized.items()
    }


def load_raw_records(config_path):
    """Load both source schemas into a common preliminary in-memory representation."""
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    validate_config(config)
    root = config_path.parent.parent
    loaded = {}
    for dataset, spec in SOURCE_SPECS.items():
        source_file = config[spec["config_key"]]
        source_path = root / source_file
        if not source_path.is_file():
            raise FileNotFoundError(f"Configured {dataset} input does not exist: {source_path}")
        rows, columns = read_source_rows(source_path, spec["required_columns"])
        records = [
            preliminary_record(dataset, source_file, index, row[spec["source_id_column"]], row)
            for index, row in enumerate(rows)
        ]
        loaded[dataset] = {"columns": columns, "records": records}
    return config, loaded


def prepare_records(config_path):
    """Load, normalize, and assess all raw RQ3 rows without writing artifacts."""
    config, loaded = load_raw_records(config_path)
    root = Path(config_path).resolve().parent.parent
    identity_mapping = load_identity_mapping(root / config["identity_mapping_input"], config["accepted_pacific_clans"])
    normalized = normalize_raw_records(loaded, identity_mapping)
    return config, apply_primary_eligibility_to_all(normalized, config)


def source_checksums(config, root):
    """Return SHA-256 checksums keyed by RQ3 source dataset."""
    return {
        dataset: hashlib.sha256((root / config[spec["config_key"]]).read_bytes()).hexdigest()
        for dataset, spec in SOURCE_SPECS.items()
    }


def add_source_checksums(loaded, checksums):
    """Attach source checksums to every normalized record."""
    result = {}
    for dataset, source in loaded.items():
        records = []
        for record in source["records"]:
            enriched = dict(record)
            enriched["source_sha256"] = checksums[dataset]
            records.append(enriched)
        result[dataset] = {"columns": source["columns"], "records": records}
    return result


def validate_for_writing(loaded, config):
    """Check core output invariants before replacing any generated artifact."""
    records = [record for source in loaded.values() for record in source["records"]]
    identifiers = [record["rq3_coda_id"] for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("RQ3 normalized records have duplicate IDs")
    for record in records:
        if not record["eligible_for_primary_transfer"]:
            continue
        sequence = record["ici_sequence_s"]
        if sequence is None or record["ici_count"] != len(sequence):
            raise ValueError(f"Eligible coda has an invalid ICI sequence: {record['rq3_coda_id']}")
        if record["ici_count"] != record["click_count"] - 1:
            raise ValueError(f"Eligible coda has inconsistent click/ICI counts: {record['rq3_coda_id']}")
        if any(value is None or not math.isfinite(value)
               or not config["primary_min_ici_s"] <= value < config["primary_max_ici_s"]
               for value in sequence):
            raise ValueError(f"Eligible coda has an ICI outside the configured primary range: {record['rq3_coda_id']}")
        if not record["group_id"]:
            raise ValueError(f"Eligible coda has no group ID: {record['rq3_coda_id']}")


def table_row(record, columns):
    """Project one eligible audit record onto a declared output-table schema."""
    return {column: record.get(column) for column in columns}


def write_processed_csv(path, rows, columns):
    """Write an eligible-coda CSV with JSON-encoded continuous ICI sequence cells."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            if "ici_sequence_s" in columns:
                serialized["ici_sequence_s"] = json.dumps(row["ici_sequence_s"], ensure_ascii=False, allow_nan=False)
            writer.writerow(serialized)


def build_report(config, checksums, all_records, eligible_records, artifact_checksums, identity_mapping):
    """Render the human-readable RQ3 preprocessing provenance report."""
    raw_counts = Counter(record["source_dataset"] for record in all_records)
    eligible_by_clan = Counter((record["source_dataset"], record["clan"]) for record in eligible_records)
    exclusions = Counter(reason for record in all_records for reason in record["exclusion_reasons"])
    click_counts = Counter(record["click_count"] for record in eligible_records)
    fold_rows = build_fold_rows(eligible_records)
    folds_by_clan = {}
    for row in fold_rows:
        folds_by_clan.setdefault((row["source_dataset"], row["clan"]), []).append(row)
    lines = [
        "# RQ3 preprocessing",
        "",
        f"Raw codas: {len(all_records)}. Primary-eligible codas: {len(eligible_records)}.",
        "",
        "## Sources",
        "",
        "| Dataset | Raw rows | SHA-256 |",
        "|---|---:|---|",
    ]
    for dataset, spec in SOURCE_SPECS.items():
        lines.append(f"| {dataset} (`{config[spec['config_key']]}`) | {raw_counts[dataset]} | `{checksums[dataset]}` |")
    lines += ["", "## Primary eligibility", "", f"3–10 clicks; accepted clan; required ICIs in [{config['primary_min_ici_s']}, {config['primary_max_ici_s']}) s; no non-zero padding.",
              "The exported `ici_sequence_s` values are continuous and unpadded. Fixed 50 ms tokenisation occurs downstream from this table.", "",
              "| Dataset | Clan | Eligible codas |", "|---|---|---:|"]
    for (dataset, clan), count in sorted(eligible_by_clan.items()):
        lines.append(f"| {dataset} | {clan} | {count} |")
    lines += ["", "## Grouping and folds", "", "`group_id` is Caribbean canonical `recording_date` or Pacific source `grpvar`. Each group is assigned whole to one deterministic fold.",
              "", "| Dataset | Clan | Groups | Fold coda counts |", "|---|---|---:|---|"]
    for (dataset, clan), rows in sorted(folds_by_clan.items()):
        fold_counts = Counter()
        for row in rows:
            fold_counts[row["fold"]] += row["n_codas"]
        counts_text = ", ".join(f"{fold}: {count}" for fold, count in sorted(fold_counts.items()))
        lines.append(f"| {dataset} | {clan} | {len(rows)} | {counts_text} |")
    lines += ["", "## Caribbean recording-date normalization", "",
              "Caribbean `Date` values are parsed day-first from `dd/mm/yyyy` or `dd-mm-yyyy` into canonical ISO `recording_date` values. The raw `source_date` is preserved unchanged. `group_id` is `recording_date` for Caribbean codas and the source `grpvar` repertoire ID for Pacific codas.",
              "", "## Exclusions", ""]
    lines += [f"- {reason}: {count}" for reason, count in sorted(exclusions.items())] or ["- None"]
    lines += ["", "## Eligible click counts", "", "| Click count | Codas |", "|---:|---:|"]
    lines += [f"| {click_count} | {count} |" for click_count, count in sorted(click_counts.items())]
    identity_counts = Counter(record["identity_status"] or "unmapped" for record in eligible_records)
    lines += ["", "## Identity labels", "", f"Pacific identity labels use `{config['identity_mapping_input']}` ({identity_mapping['source']}). Caribbean codas are `unmapped`; Pacific `#N/A` source types are also `unmapped` and are excluded from identity-stratified analyses.",
              "", "| Identity status | Eligible codas |", "|---|---:|"]
    lines += [f"| {status} | {count} |" for status, count in sorted(identity_counts.items())]
    lines += [
              "", "## Reproducibility", "", "Run configuration:", "", "```json", json.dumps(config, indent=2, ensure_ascii=False), "```", "",
              "Output SHA-256 checksums:", ""]
    lines += [f"- `{name}`: `{checksum}`" for name, checksum in sorted(artifact_checksums.items())]
    return "\n".join(lines) + "\n"


def run(config_path):
    """Prepare and write the auditable RQ3 continuous-ICI data artifacts."""
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config, loaded = prepare_records(config_path)
    identity_mapping = load_identity_mapping(root / config["identity_mapping_input"], config["accepted_pacific_clans"])
    checksums = source_checksums(config, root)
    loaded = add_source_checksums(loaded, checksums)
    validate_for_writing(loaded, config)
    all_records = [record for source in loaded.values() for record in source["records"]]
    eligible_records = [record for record in all_records if record["eligible_for_primary_transfer"]]
    interim_path = root / config["interim_output"]
    processed_path = root / config["processed_output"]
    metadata_path = root / config["metadata_output"]
    folds_path = root / config["folds_output"]
    report_path = root / config["report_output"]
    for path in (interim_path, processed_path, metadata_path, folds_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(interim_path, all_records)
    write_processed_csv(processed_path, [table_row(record, CORE_COLUMNS) for record in eligible_records], CORE_COLUMNS)
    write_processed_csv(metadata_path, [table_row(record, METADATA_COLUMNS) for record in eligible_records], METADATA_COLUMNS)
    write_processed_csv(folds_path, build_fold_rows(eligible_records), FOLD_COLUMNS)
    artifact_checksums = {
        interim_path.name: hashlib.sha256(interim_path.read_bytes()).hexdigest(),
        processed_path.name: hashlib.sha256(processed_path.read_bytes()).hexdigest(),
        metadata_path.name: hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        folds_path.name: hashlib.sha256(folds_path.read_bytes()).hexdigest(),
    }
    write_text(report_path, build_report(config, checksums, all_records, eligible_records, artifact_checksums, identity_mapping))
    return {
        "raw_codas": len(all_records),
        "eligible_codas": len(eligible_records),
        "artifact_sha256": artifact_checksums,
    }


def main():
    parser = argparse.ArgumentParser(description="Load and validate raw RQ3 source schemas")
    parser.add_argument("--config", default="config/preprocessing_rq3.json")
    args = parser.parse_args()
    report = run(args.config)
    print(f"Prepared {report['eligible_codas']} primary-eligible RQ3 codas.")


if __name__ == "__main__":
    main()
