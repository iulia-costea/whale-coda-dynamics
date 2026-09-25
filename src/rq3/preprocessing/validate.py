"""Independently validate the generated RQ3 preprocessing artifacts."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl

from .prepare import CORE_COLUMNS, METADATA_COLUMNS, SOURCE_SPECS, validate_config


EXPECTED_ELIGIBLE_BY_CLAN = {
    ("caribbean", "EC1"): 7751,
    ("caribbean", "EC2"): 948,
    ("pacific", "FP"): 2564,
    ("pacific", "PALI"): 2059,
    ("pacific", "PO"): 1986,
    ("pacific", "REG"): 8108,
    ("pacific", "RI"): 1750,
    ("pacific", "SH"): 5153,
    ("pacific", "SI"): 1209,
}


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
        if any(value is not None and value != 0 for value in padding):
            reasons.append("nonzero_padding")
        if not reasons:
            sequence = required
    return reasons, sequence


def validate_interim(config, root):
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


def run(config_path):
    """Validate all current RQ3 source and generated artifacts without writing files."""
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    validate_config(config)
    root = config_path.parent.parent
    interim = validate_interim(config, root)
    eligible_count = validate_core_and_metadata(config, root, interim)
    return {"raw_codas": len(interim), "eligible_codas": eligible_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/preprocessing_rq3.json")
    report = run(parser.parse_args().config)
    print("RQ3 validation passed: source coverage, eligibility, core-table and metadata-table consistency. "
          f"Raw codas: {report['raw_codas']}; eligible codas: {report['eligible_codas']}.")


if __name__ == "__main__":
    main()
