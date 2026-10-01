"""Independently validate the generated RQ3 preprocessing artifacts."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl

from .prepare import CORE_COLUMNS, METADATA_COLUMNS, SOURCE_SPECS, validate_config


EXPECTED_ELIGIBLE_BY_CLAN = {
    ("caribbean", "EC1"): 7742,
    ("caribbean", "EC2"): 946,
    ("pacific", "FP"): 2556,
    ("pacific", "PALI"): 2052,
    ("pacific", "PO"): 1985,
    ("pacific", "REG"): 8107,
    ("pacific", "RI"): 1747,
    ("pacific", "SH"): 5138,
    ("pacific", "SI"): 1206,
}

FOLD_COLUMNS = ["source_dataset", "clan", "group_id", "fold", "n_codas"]


def source_text(value):
    """Apply the source-text interpretation used by the normalization contract."""
    stripped = value.strip()
    return stripped if stripped else None


def source_number(value):
    """Parse one source numeric cell independently of the preparation module."""
    stripped = value.strip()
    if not stripped:
        return None
    try:
        number = float(stripped)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def source_integer(value):
    """Parse one source integer cell independently of the preparation module."""
    number = source_number(value)
    return int(number) if number is not None and number.is_integer() else None


def source_caribbean_recording_date(value):
    """Independently parse a Caribbean source date day-first to ISO format."""
    text = value.strip()
    if not text:
        return None
    date_format = "%d/%m/%Y" if "/" in text else "%d-%m-%Y" if "-" in text else None
    if date_format is None:
        return None
    try:
        return datetime.strptime(text, date_format).date().isoformat()
    except ValueError:
        return None


def expected_group_id(dataset, raw):
    """Derive the fold grouping field independently from raw source values."""
    return source_caribbean_recording_date(raw["Date"]) if dataset == "caribbean" else source_text(raw["grpvar"])


def load_identity_mapping_for_validation(path, accepted_pacific_clans):
    """Read and independently validate the sourced Pacific identity mapping."""
    mapping = read_json(path)
    if set(mapping) != {"source", "pacific_identity_coda_types"} or not isinstance(mapping["source"], str) or not mapping["source"]:
        raise ValueError("Invalid RQ3 identity mapping structure")
    types_by_clan = mapping["pacific_identity_coda_types"]
    if set(types_by_clan) != set(accepted_pacific_clans):
        raise ValueError("RQ3 identity mapping clans differ from accepted Pacific clans")
    for clan, types in types_by_clan.items():
        if (not isinstance(types, list) or not types or any(type(value) is not int or value <= 0 for value in types)
                or len(types) != len(set(types))):
            raise ValueError(f"Invalid RQ3 identity mapping types for {clan}")
    return mapping


def expected_identity_status(dataset, clan, raw, mapping):
    """Independently derive the audit identity label from source type and mapping."""
    if dataset != "pacific" or clan not in mapping["pacific_identity_coda_types"]:
        return "unmapped"
    try:
        coda_type = int(source_text(raw["coda_type"]))
    except (TypeError, ValueError):
        return "unmapped"
    return "identity" if coda_type in mapping["pacific_identity_coda_types"][clan] else "non_identity"


def read_raw_csv(path, required_columns):
    """Read and schema-check a raw source without using the preparation loader."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"Invalid raw CSV header: {path}")
        missing = required_columns - set(columns)
        if missing:
            raise ValueError(f"Raw CSV is missing required columns: {path}: {sorted(missing)}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed raw CSV row: {path}")
    return rows


def read_generated_csv(path, expected_columns):
    """Read one generated CSV and reject a schema drift before row validation."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"Unexpected CSV schema in {path}: {reader.fieldnames}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed generated CSV row: {path}")
    return rows


def expected_eligibility(record, config):
    """Recompute primary-transfer eligibility from typed interim values."""
    reasons = []
    click_count = record["click_count"]
    in_range = click_count is not None and config["primary_min_clicks"] <= click_count <= config["primary_max_clicks"]
    if click_count is None:
        reasons.append("missing_or_invalid_click_count")
    elif not in_range:
        reasons.append("click_count_outside_primary_range")
    if record["clan"] not in config[f"accepted_{record['source_dataset']}_clans"]:
        reasons.append("missing_or_unaccepted_clan")
    sequence = None
    if in_range:
        candidate = record["candidate_ici_values_s"]
        required, padding = candidate[:click_count - 1], candidate[click_count - 1:]
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
    return reasons, sequence


def validate_interim(config, root, identity_mapping):
    """Validate source coverage, typed core fields, and primary eligibility."""
    interim_path = root / config["interim_output"]
    interim = read_jsonl(interim_path)
    expected_total = 0
    source_rows = {}
    checksums = {}
    for dataset, spec in SOURCE_SPECS.items():
        source_path = root / config[spec["config_key"]]
        rows = read_raw_csv(source_path, spec["required_columns"])
        source_rows[dataset] = rows
        checksums[dataset] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        expected_total += len(rows)
    if len(interim) != expected_total:
        raise ValueError(f"Interim row count is {len(interim)}, expected {expected_total}")
    expected_ids = {
        f"rq3:{dataset}:row:{index}"
        for dataset, rows in source_rows.items()
        for index in range(len(rows))
    }
    actual_ids = [record.get("rq3_coda_id") for record in interim]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("Interim RQ3 IDs do not provide one-to-one raw-source coverage")
    for record in interim:
        dataset = record["source_dataset"]
        if dataset not in SOURCE_SPECS:
            raise ValueError(f"Unexpected source dataset in interim row: {dataset}")
        index = record["source_row_index"]
        if type(index) is not int or not 0 <= index < len(source_rows[dataset]):
            raise ValueError(f"Invalid source-row index for {record['rq3_coda_id']}")
        raw = source_rows[dataset][index]
        spec = SOURCE_SPECS[dataset]
        expected_id = f"rq3:{dataset}:row:{index}"
        if record["rq3_coda_id"] != expected_id:
            raise ValueError(f"Unexpected RQ3 ID: {record['rq3_coda_id']}")
        if record["source_file"] != config[spec["config_key"]] or record["source_sha256"] != checksums[dataset]:
            raise ValueError(f"Source provenance mismatch for {expected_id}")
        if record["raw_values"] != raw:
            raise ValueError(f"Raw-value preservation mismatch for {expected_id}")
        if record["source_coda_id"] != source_text(raw[spec["source_id_column"]]):
            raise ValueError(f"Source coda ID mismatch for {expected_id}")
        click_field = next(iter(spec["integer_fields"]))
        clan_field = next(field for field, target in spec["text_fields"].items() if target == "clan")
        if record["click_count"] != source_integer(raw[click_field]):
            raise ValueError(f"Click-count normalization mismatch for {expected_id}")
        if record["clan"] != source_text(raw[clan_field]):
            raise ValueError(f"Clan normalization mismatch for {expected_id}")
        expected_candidate = [source_number(raw[f"ICI{number}"]) for number in range(1, spec["ici_column_count"] + 1)]
        if record["candidate_ici_values_s"] != expected_candidate:
            raise ValueError(f"Candidate ICI normalization mismatch for {expected_id}")
        expected_recording_date = source_caribbean_recording_date(raw["Date"]) if dataset == "caribbean" else None
        if record.get("recording_date") != expected_recording_date:
            raise ValueError(f"Recording-date normalization mismatch for {expected_id}")
        if record.get("group_id") != expected_group_id(dataset, raw):
            raise ValueError(f"Group-ID normalization mismatch for {expected_id}")
        if record.get("identity_status") != expected_identity_status(dataset, record["clan"], raw, identity_mapping):
            raise ValueError(f"Identity-status normalization mismatch for {expected_id}")
        reasons, sequence = expected_eligibility(record, config)
        if record["exclusion_reasons"] != reasons or record["eligible_for_primary_transfer"] != (not reasons):
            raise ValueError(f"Eligibility mismatch for {expected_id}")
        if record["ici_sequence_s"] != sequence:
            raise ValueError(f"ICI sequence mismatch for {expected_id}")
        if record["ici_count"] != (len(sequence) if sequence is not None else None):
            raise ValueError(f"ICI count mismatch for {expected_id}")
    return interim


def csv_integer(value, field, row_id):
    """Parse an exact generated-CSV integer or raise a row-specific error."""
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field} in {row_id}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid {field} in {row_id}")
    return int(number)


def csv_sequence(value, row_id):
    """Parse one generated JSON ICI array and check its numeric domain."""
    try:
        sequence = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON ICI sequence in {row_id}") from exc
    if not isinstance(sequence, list) or any(type(item) not in (int, float) or not math.isfinite(item) or item <= 0 for item in sequence):
        raise ValueError(f"Invalid ICI sequence values in {row_id}")
    return sequence


def csv_scalar(value):
    """Match csv.DictWriter's representation of nullable scalar metadata."""
    return "" if value is None else str(value)


def validate_core_and_metadata(config, root, interim):
    """Validate both processed tables against the eligible interim rows."""
    core = read_generated_csv(root / config["processed_output"], CORE_COLUMNS)
    metadata = read_generated_csv(root / config["metadata_output"], METADATA_COLUMNS)
    eligible = [record for record in interim if record["eligible_for_primary_transfer"]]
    expected_ids = [record["rq3_coda_id"] for record in eligible]
    core_ids = [row["rq3_coda_id"] for row in core]
    metadata_ids = [row["rq3_coda_id"] for row in metadata]
    if core_ids != expected_ids:
        raise ValueError("Core processed table does not match eligible interim-row coverage and order")
    if metadata_ids != core_ids:
        raise ValueError("Metadata table IDs do not match core processed-table order")
    for row, record in zip(core, eligible):
        row_id = row["rq3_coda_id"]
        if row["source_dataset"] != record["source_dataset"] or row["clan"] != record["clan"]:
            raise ValueError(f"Core provenance mismatch for {row_id}")
        if row["group_id"] != record["group_id"]:
            raise ValueError(f"Core group ID mismatch for {row_id}")
        click_count = csv_integer(row["click_count"], "click_count", row_id)
        sequence = csv_sequence(row["ici_sequence_s"], row_id)
        if click_count != record["click_count"] or sequence != record["ici_sequence_s"]:
            raise ValueError(f"Core sequence mismatch for {row_id}")
        if len(sequence) != click_count - 1:
            raise ValueError(f"Core ICI length mismatch for {row_id}")
    for row, record in zip(metadata, eligible):
        row_id = row["rq3_coda_id"]
        for column in METADATA_COLUMNS:
            if column == "ici_sequence_s":
                if csv_sequence(row[column], row_id) != record[column]:
                    raise ValueError(f"Metadata sequence mismatch for {row_id}")
            elif row[column] != csv_scalar(record.get(column)):
                raise ValueError(f"Metadata field mismatch for {row_id}: {column}")
    counts = Counter((record["source_dataset"], record["clan"]) for record in eligible)
    if counts != EXPECTED_ELIGIBLE_BY_CLAN:
        raise ValueError(f"Eligible per-clan counts differ from the expected RQ3 sample: {dict(counts)}")
    return len(core)


def expected_fold_rows(eligible):
    """Independently reproduce the documented deterministic group-fold assignment."""
    group_counts = Counter((record["source_dataset"], record["clan"], record["group_id"]) for record in eligible)
    by_clan = {}
    for (dataset, clan, group_id), count in group_counts.items():
        if not group_id:
            raise ValueError(f"Eligible coda has no group ID: {dataset}/{clan}")
        by_clan.setdefault((dataset, clan), []).append((group_id, count))

    rows = []
    for (dataset, clan), groups in sorted(by_clan.items()):
        fold_count = 6 if clan == "EC2" else 5
        if len(groups) < fold_count:
            raise ValueError(f"{dataset}/{clan} has too few groups for its configured folds")
        sizes = [0] * fold_count
        ordered = sorted(
            groups,
            key=lambda item: (
                -item[1],
                hashlib.sha256(f"42:{clan}:{item[0]}".encode("utf-8")).hexdigest(),
                dataset,
                clan,
                item[0],
            ),
        )
        for group_id, n_codas in ordered:
            fold = min(range(fold_count), key=lambda index: (sizes[index], index))
            sizes[fold] += n_codas
            rows.append({
                "source_dataset": dataset,
                "clan": clan,
                "group_id": group_id,
                "fold": fold,
                "n_codas": n_codas,
            })
    return sorted(rows, key=lambda row: (row["source_dataset"], row["clan"], row["fold"], row["group_id"]))


def validate_folds(config, root, interim):
    """Validate the fold artifact against an independent deterministic reconstruction."""
    rows = read_generated_csv(root / config["folds_output"], FOLD_COLUMNS)
    actual = []
    for row in rows:
        source_dataset, clan, group_id = row["source_dataset"], row["clan"], row["group_id"]
        if not source_dataset or not clan or not group_id:
            raise ValueError("Fold row contains an empty provenance or group field")
        row_id = f"{source_dataset}/{clan}/{group_id}"
        actual.append({
            "source_dataset": source_dataset,
            "clan": clan,
            "group_id": group_id,
            "fold": csv_integer(row["fold"], "fold", row_id),
            "n_codas": csv_integer(row["n_codas"], "n_codas", row_id),
        })
    eligible = [record for record in interim if record["eligible_for_primary_transfer"]]
    if actual != expected_fold_rows(eligible):
        raise ValueError("RQ3 fold artifact does not match the deterministic group-fold assignment")


def run(config_path):
    """Validate all current RQ3 source and generated artifacts without writing files."""
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    validate_config(config)
    root = config_path.parent.parent
    identity_mapping = load_identity_mapping_for_validation(root / config["identity_mapping_input"], config["accepted_pacific_clans"])
    interim = validate_interim(config, root, identity_mapping)
    eligible_count = validate_core_and_metadata(config, root, interim)
    validate_folds(config, root, interim)
    return {"raw_codas": len(interim), "eligible_codas": eligible_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/preprocessing_rq3.json")
    report = run(parser.parse_args().config)
    print("RQ3 validation passed: source coverage, eligibility, core-table, metadata-table, and fold consistency. "
          f"Raw codas: {report['raw_codas']}; eligible codas: {report['eligible_codas']}.")


if __name__ == "__main__":
    main()
