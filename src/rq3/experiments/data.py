"""Load and strictly validate RQ3 coda, metadata, and fold artifacts."""
import csv
import io
import json
import math
from collections import Counter
from pathlib import Path

from whale_coda_dynamics.preprocessing.io_utils import read_json, read_text, write_text


CORE_COLUMNS = ["rq3_coda_id", "source_dataset", "clan", "group_id", "click_count", "ici_sequence_s"]
FOLD_COLUMNS = ["source_dataset", "clan", "group_id", "fold", "n_codas"]
SELF_SCORE_COLUMNS = [
    "rq3_coda_id", "source_dataset", "clan", "group_id", "fold", "model_family",
    "context_order", "alpha", "token_count", "coda_log2_probability", "bits_per_coda",
    "bits_per_symbol",
]
CALIBRATION_COLUMNS = [
    "source_dataset", "clan", "candidate_index", "model_family", "context_order", "alpha",
    "validation_coda_count", "mean_bits_per_coda", "mean_bits_per_symbol", "selected",
]


def read_csv(path, expected_columns):
    """Read a UTF-8 CSV through the project's pinned text-I/O helper."""
    reader = csv.DictReader(io.StringIO(read_text(path)))
    if expected_columns is not None and reader.fieldnames != expected_columns:
        raise ValueError(f"Unexpected CSV schema in {path}: {reader.fieldnames}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row in {path}")
    return rows


def write_csv(path, columns, rows):
    """Write a UTF-8/LF CSV through the project's pinned text-I/O helper."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, buffer.getvalue())


def validate_config(config):
    """Reject incomplete or ambiguous RQ3 experiment settings."""
    required = {
        "core_input", "metadata_input", "folds_input", "self_scores_output",
        "calibration_output", "report_output", "identity_mapping_input",
        "spatial_overlap_input", "tokenizer_id", "candidates",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"RQ3 experiment configuration is missing keys: {sorted(missing)}")
    for key in required - {"candidates"}:
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"RQ3 experiment configuration {key!r} must be a non-empty string")
    if config["tokenizer_id"] != "fixed-50ms-v1":
        raise ValueError("Primary RQ3 calibration requires the fixed-50ms-v1 tokenizer")
    candidates = config["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("RQ3 experiment candidates must be a non-empty list")
    for candidate in candidates:
        if set(candidate) != {"family", "context_order", "alpha"}:
            raise ValueError(f"Invalid candidate keys: {candidate}")
        family, order, alpha = candidate["family"], candidate["context_order"], candidate["alpha"]
        if family not in {"unigram", "ngram"} or type(order) is not int or order < 0:
            raise ValueError(f"Invalid model candidate: {candidate}")
        if family == "unigram" and order != 0:
            raise ValueError("Unigram candidates must have context_order 0")
        if family == "ngram" and order < 1:
            raise ValueError("N-gram candidates must have positive context_order")
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid smoothing alpha: {candidate}")


def parse_codas(rows):
    """Parse model rows while checking their sequence/count contract."""
    codas = []
    for row in rows:
        try:
            click_count = int(row["click_count"])
            sequence = json.loads(row["ici_sequence_s"])
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid core coda row: {row['rq3_coda_id']}") from exc
        if (click_count < 3 or not row["group_id"] or not isinstance(sequence, list)
                or len(sequence) != click_count - 1
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in sequence)):
            raise ValueError(f"Invalid core coda contract: {row['rq3_coda_id']}")
        codas.append({**row, "click_count": click_count, "ici_sequence_s": sequence})
    ids = [coda["rq3_coda_id"] for coda in codas]
    if len(ids) != len(set(ids)):
        raise ValueError("RQ3 core coda IDs are not unique")
    return codas


def build_fold_lookup(codas, fold_rows):
    """Check group coverage/counts and return coda ID to deterministic fold."""
    coda_groups = Counter((coda["source_dataset"], coda["clan"], coda["group_id"]) for coda in codas)
    fold_lookup = {}
    for row in fold_rows:
        key = (row["source_dataset"], row["clan"], row["group_id"])
        try:
            fold, n_codas = int(row["fold"]), int(row["n_codas"])
        except ValueError as exc:
            raise ValueError(f"Invalid fold values for {key}") from exc
        if key in fold_lookup or fold < 0 or n_codas != coda_groups.get(key):
            raise ValueError(f"Invalid or duplicate fold assignment for {key}")
        fold_lookup[key] = fold
    if set(fold_lookup) != set(coda_groups):
        raise ValueError("Fold artifact does not cover exactly the core coda groups")
    for dataset, clan in {(c["source_dataset"], c["clan"]) for c in codas}:
        folds = {fold_lookup[(dataset, clan, coda["group_id"])] for coda in codas if coda["source_dataset"] == dataset and coda["clan"] == clan}
        expected = set(range(6 if clan == "EC2" else 5))
        if folds != expected:
            raise ValueError(f"Unexpected folds for {dataset}/{clan}: {sorted(folds)}")
    return {
        coda["rq3_coda_id"]: fold_lookup[(coda["source_dataset"], coda["clan"], coda["group_id"])]
        for coda in codas
    }


def validate_analysis_metadata(metadata_rows, identity_mapping, spatial_overlap):
    """Check metadata needed later for identity and Pacific-overlap experiments."""
    required_metadata = {"rq3_coda_id", "source_dataset", "clan", "identity_status"}
    if not metadata_rows or not required_metadata <= set(metadata_rows[0]):
        raise ValueError("RQ3 metadata lacks identity-analysis fields")
    identity_types = identity_mapping.get("pacific_identity_coda_types")
    if not isinstance(identity_types, dict) or not identity_types:
        raise ValueError("RQ3 identity mapping lacks Pacific identity types")
    allowed_statuses = {"identity", "non_identity", "unmapped"}
    if any(row["identity_status"] not in allowed_statuses for row in metadata_rows):
        raise ValueError("Unexpected RQ3 identity status")
    clans = spatial_overlap.get("clans")
    matrix = spatial_overlap.get("directed_overlap")
    if not isinstance(clans, list) or not isinstance(matrix, dict) or set(clans) != set(matrix):
        raise ValueError("Invalid RQ3 spatial-overlap clan matrix")
    if any(set(matrix[clan]) != set(clans) for clan in clans):
        raise ValueError("RQ3 spatial-overlap matrix is not square")
    if any(type(value) not in (int, float) or not 0 <= value <= 1 for row in matrix.values() for value in row.values()):
        raise ValueError("RQ3 spatial-overlap values must be proportions")


def load_experiment_data(config_path):
    """Load validated RQ3 artifacts and attach each coda's deterministic fold."""
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = read_json(config_path)
    validate_config(config)
    core = parse_codas(read_csv(root / config["core_input"], CORE_COLUMNS))
    metadata = read_csv(root / config["metadata_input"], None)
    if len(metadata) != len(core) or [row["rq3_coda_id"] for row in metadata] != [coda["rq3_coda_id"] for coda in core]:
        raise ValueError("Metadata rows do not match core coda coverage and order")
    validate_analysis_metadata(
        metadata,
        read_json(root / config["identity_mapping_input"]),
        read_json(root / config["spatial_overlap_input"]),
    )
    folds = read_csv(root / config["folds_input"], FOLD_COLUMNS)
    fold_by_coda = build_fold_lookup(core, folds)
    for coda in core:
        coda["fold"] = fold_by_coda[coda["rq3_coda_id"]]
    return config, root, core
