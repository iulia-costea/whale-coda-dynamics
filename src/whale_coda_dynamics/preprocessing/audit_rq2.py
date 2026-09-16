"""Lossless source preservation plus conservative, typed normalization for RQ2.

Mirrors audit.py's contract but adapts it to a different source: Sharma,
Gero, Payne, et al. (2024)'s `sperm-whale-dialogues.csv` release plus a
separately supplied `rhythms.p` label array. Unlike the Assayag release, this
source has no Tag, Bout, Date, Focal, or individual-ID columns, and no
per-row identifier column (no codaNUM2018 analog): row index is the only
handle. `tag_id` is therefore set equal to `recording_id` (REC) rather than
representing a distinct concept. `date_candidate` does not hold a calendar
date here (none exists in this release); it holds a deployment prefix parsed
from REC (`^sw\\d+`), used only so that recordings from the same probable
tag deployment stay together across day-held-out splits, mirroring RQ1's
same-day-stays-together rule without claiming any chronological meaning.
"""
import argparse
import csv
import hashlib
import math
import pickle
import re
from collections import Counter
from pathlib import Path

from .io_utils import read_json, write_jsonl

DEPLOYMENT_PATTERN = re.compile(r"^(sw\d+)")

FIELDS = {
    "REC": ("recording_id", "text"),
    "Whale": ("source_whale_code", "text"),
    "TsTo": ("source_onset_s", "number"),
    "Duration": ("duration_s", "number"),
    "nClicks": ("click_count", "integer"),
}


def convert(value, kind, config):
    value = value.strip()
    if value.lower() in config["missing_tokens"]:
        return None
    if kind == "text":
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    if kind == "integer":
        if not number.is_integer():
            raise ValueError("invalid integer value")
        return int(number)
    return number


def normalize(rows, rhythms, columns, config):
    missing_columns = set(FIELDS) - set(columns)
    if missing_columns:
        raise ValueError(f"Required source columns missing: {sorted(missing_columns)}")
    if len(rhythms) != len(rows):
        raise ValueError("rhythms.p length does not match sperm-whale-dialogues.csv row count")
    result, issues = [], []

    def flag(index, code, field="", detail=""):
        issues.append({"source_row_index": index, "code": code, "field": field, "detail": detail})

    for i, source in enumerate(rows):
        target = {"coda_id": f"sharma:row:{i}", "source_row_index": i,
                  "source_csv_line": i + 2, "raw_values": source}
        for column, (name, kind) in FIELDS.items():
            try:
                target[name] = convert(source[column], kind, config)
            except (ValueError, TypeError, OverflowError) as exc:
                target[name] = None
                flag(i, "parse_error", column, str(exc))
        rhythm_label = rhythms[i]
        if not isinstance(rhythm_label, int):
            flag(i, "non_integer_rhythm_label", "rhythms.p", str(type(rhythm_label)))
        target["coda_type"] = str(rhythm_label)
        target["source_coda_id"] = target["coda_id"]
        # No Bout, Focal, or individual-ID columns exist in this release.
        target["source_bout_code"] = None
        target["focal_status"] = None
        target["source_individual_id"] = None
        # tag_id is not a distinct concept here; REC is the only session-level ID.
        target["tag_id"] = target["recording_id"]
        deployment = DEPLOYMENT_PATTERN.match(target["recording_id"] or "")
        target["deployment_id"] = deployment.group(1) if deployment else None
        if deployment is None:
            flag(i, "unparsed_recording_id", "REC", "Does not match ^sw\\d+ deployment prefix")
        target["date_candidate"] = target["deployment_id"]
        if target["source_whale_code"] in (None, "", "0"):
            flag(i, "missing_or_zero_whale_code", "Whale")
        if target["duration_s"] is None or target["duration_s"] <= 0:
            flag(i, "invalid_duration", "Duration")
        if target["source_onset_s"] is None or target["source_onset_s"] < 0:
            flag(i, "invalid_onset", "TsTo")
        if target["click_count"] is None or target["click_count"] < 3:
            flag(i, "invalid_coda_click_count", "nClicks")
        result.append(target)

    groups = {}
    for r in result:
        groups.setdefault((r["recording_id"], r["source_onset_s"]), []).append(r["source_row_index"])
    for indices in groups.values():
        if len(indices) > 1:
            for i in indices:
                flag(i, "shared_recording_onset", "TsTo", "Possible overlap or duplicate; retained")
    return result, issues


def run(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    root = config_path.parent.parent
    source = root / config["input"]
    rhythms_path = root / config["rhythms_input"]
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed CSV rows; no output generated")
    with rhythms_path.open("rb") as handle:
        rhythms = pickle.load(handle)
    normalized, issues = normalize(rows, rhythms, columns, config)
    csv_checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    rhythms_checksum = hashlib.sha256(rhythms_path.read_bytes()).hexdigest()
    combined_checksum = hashlib.sha256(f"{csv_checksum}:{rhythms_checksum}".encode()).hexdigest()
    for row in normalized:
        row["source_file"] = config["input"]
        row["rhythms_file"] = config["rhythms_input"]
        row["source_sha256"] = combined_checksum
    deployments = {r["deployment_id"] for r in normalized if r["deployment_id"]}
    report = {
        "source": config.get("source_url"), "source_sha256": combined_checksum,
        "source_csv_sha256": csv_checksum, "rhythms_sha256": rhythms_checksum,
        "configuration": config, "row_count": len(rows), "expected_rows": config["expected_rows"],
        "row_count_matches": len(rows) == config["expected_rows"], "excluded_rows": 0,
        "source_columns": columns,
        "source_missing_counts": {c: sum(r[c].strip().lower() in config["missing_tokens"] for r in rows) for c in columns},
        "issue_counts": dict(sorted(Counter(i["code"] for i in issues).items())),
        "unique_recordings": len({r["REC"] for r in rows}),
        "unique_deployments": len(deployments),
        "whale_code_counts": dict(sorted(Counter(r["Whale"] for r in rows).items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1)),
        "rhythm_label_counts": dict(sorted(Counter(rhythms).items())),
        "limitations": [
            "No Tag, Bout, Date, Focal, or individual-ID columns exist in this release; tag_id is set equal to recording_id (REC)",
            "date_candidate holds a deployment prefix parsed from REC (^sw\\d+), used only to keep same-deployment recordings together across splits; it is not a calendar date",
            "Whale codes are provisional local caller codes, not confirmed individual identities, and are not limited to two values across the full file",
            "coda_type is the rhythm-cluster label from rhythms.p, aligned to this CSV by row index only; there is no shared key column to verify the join",
            "There is no per-row source identifier analogous to Assayag's codaNUM2018; row index is the only handle",
        ],
    }
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    row_issues = {}
    for issue in issues:
        row_issues.setdefault(issue["source_row_index"], []).append(issue)
    for row in normalized:
        row["audit_issues"] = row_issues.get(row["source_row_index"], [])
    write_jsonl(output / "codas_normalized_rq2.jsonl", normalized)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/preprocessing_rq2.json")
    args = parser.parse_args()
    report = run(args.config)
    print(f"Normalized {report['row_count']} rows.")


if __name__ == "__main__":
    main()
