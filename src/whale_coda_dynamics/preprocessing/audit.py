"""Lossless source preservation plus conservative, typed normalization."""
import argparse
import ast
import csv
import hashlib
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from .io_utils import read_json, write_jsonl

FIELDS = {
    "Tag": ("tag_id", "text"), "REC": ("recording_id", "text"),
    "codaNUM2018": ("source_coda_id", "text"),
    "guy_coda_type": ("coda_type", "text"),
    "decision_tree_coda_type": ("alternative_coda_type", "text"),
    "Focal": ("focal_status", "binary"), "Whale": ("source_whale_code", "text"),
    "IDN": ("source_individual_id", "text"),
    "Bout": ("source_bout_code", "text"),
    "TsTo": ("source_onset_s", "number"), "Duration": ("duration_s", "number"),
    "nClicks": ("click_count", "integer"), "Date": ("date_candidate", "date"),
    "TagOnTime": ("tag_on_time_candidate", "time"),
    "past_codas": ("source_past_row_indices", "indices"),
    "codas_history": ("source_history_row_indices", "indices"),
}


def convert(value, kind, config):
    value = value.strip()
    if value.lower() in config["missing_tokens"]:
        return None
    if kind == "text":
        return value
    if kind in ("number", "integer", "binary"):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite number")
        if kind != "number":
            if not number.is_integer() or (kind == "binary" and number not in (0, 1)):
                raise ValueError("invalid integer/binary value")
            return int(number)
        return number
    if kind == "indices":
        indices = ast.literal_eval(value)
        if not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("invalid row-index list")
        return indices
    if kind == "date":
        for month, replacement in config["date_month_aliases"].items():
            value = value.replace(month, replacement)
        return datetime.strptime(value, "%d-%m-%Y").date().isoformat()
    if kind == "time":
        return datetime.strptime(value, "%H:%M:%S").time().isoformat()
    raise ValueError("unknown conversion")


def normalize(rows, columns, config):
    missing_columns = set(FIELDS) - set(columns)
    if missing_columns:
        raise ValueError(f"Required source columns missing: {sorted(missing_columns)}")
    result, issues = [], []
    ids = Counter(r["codaNUM2018"] for r in rows)
    def flag(index, code, field="", detail=""):
        issues.append({"source_row_index": index, "code": code, "field": field, "detail": detail})
    for i, source in enumerate(rows):
        target = {"coda_id": f"assayag:row:{i}", "source_row_index": i,
                  "source_csv_line": i + 2, "raw_values": source}
        for column, (name, kind) in FIELDS.items():
            try:
                target[name] = convert(source[column], kind, config)
            except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
                target[name] = None
                flag(i, "parse_error", column, str(exc))
        if target["focal_status"] is None:
            flag(i, "missing_focal", "Focal")
        if target["source_individual_id"] in (None, "0"):
            flag(i, "unresolved_individual_id", "IDN", "Zero preserved; not treated as a known identity")
        if not target["tag_on_time_candidate"]:
            flag(i, "missing_or_invalid_tag_on_time", "TagOnTime")
        if not target["date_candidate"]:
            flag(i, "missing_or_invalid_date", "Date")
        if not target["coda_type"]:
            flag(i, "missing_primary_coda_type", "guy_coda_type")
        elif target["alternative_coda_type"] and target["coda_type"] != target["alternative_coda_type"]:
            flag(i, "coda_label_disagreement", "guy_coda_type")
        if not target["source_coda_id"] or ids[source["codaNUM2018"]] > 1:
            flag(i, "missing_or_duplicate_source_id", "codaNUM2018")
        if target["duration_s"] is None or target["duration_s"] <= 0:
            flag(i, "invalid_duration", "Duration")
        if target["source_onset_s"] is None or target["source_onset_s"] < 0:
            flag(i, "invalid_onset", "TsTo")
        if target["click_count"] is None or target["click_count"] < 3:
            flag(i, "invalid_coda_click_count", "nClicks")
        for name in ("source_past_row_indices", "source_history_row_indices"):
            history = target[name]
            if history is not None and any(j >= i for j in history):
                flag(i, "nonpast_history_reference", name)
            if history is not None:
                valid = [j for j in history if j < len(rows)]
                for field in ("Tag", "Bout", "REC"):
                    if any(rows[j][field] != source[field] for j in valid):
                        flag(i, "history_crosses_" + field.lower(), name)
                if name == "source_past_row_indices" and any(rows[j]["Whale"] != source["Whale"] for j in valid):
                    flag(i, "past_history_crosses_whale_code", name)
        result.append(target)
    groups = {}
    for r in result:
        groups.setdefault((r["tag_id"], r["source_onset_s"]), []).append(r["source_row_index"])
    for indices in groups.values():
        if len(indices) > 1:
            for i in indices:
                flag(i, "shared_tag_onset", "TsTo", "Possible overlap or duplicate; retained")
    return result, issues


def run(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    root = config_path.parent.parent
    source = root / config["input"]
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed CSV rows; no output generated")
    normalized, issues = normalize(rows, columns, config)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    for row in normalized:
        row["source_file"] = config["input"]
        row["source_sha256"] = checksum
    report = {
        "source": config["source_url"], "source_sha256": checksum,
        "configuration": config, "row_count": len(rows), "expected_rows": config["expected_rows"],
        "row_count_matches": len(rows) == config["expected_rows"], "excluded_rows": 0,
        "source_columns": columns,
        "source_missing_counts": {c: sum(r[c].strip().lower() in config["missing_tokens"] for r in rows) for c in columns},
        "issue_counts": dict(sorted(Counter(i["code"] for i in issues).items())),
        "pair_categories": dict(Counter(r["pair_category"] or "unpaired" for r in rows)),
        "unique_tags": len({r["Tag"] for r in rows}),
        "unique_recordings": len({r["REC"] for r in rows}),
        "limitations": ["Whale and Bout code scope unresolved", "TsTo origin and timezone unverified",
                        "Source forward pair features are not target-aligned RQ1 features",
                        "Missing counts cover configured tokens, not undocumented numeric sentinels",
                        "Row IDs refer to this exact file version; retain its SHA-256"],
    }
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    row_issues = {}
    for issue in issues:
        row_issues.setdefault(issue["source_row_index"], []).append(issue)
    for row in normalized:
        row["audit_issues"] = row_issues.get(row["source_row_index"], [])
    write_jsonl(output / "codas_normalized.jsonl", normalized)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/preprocessing.json")
    args = parser.parse_args()
    report = run(args.config)
    print(f"Normalized {report['row_count']} rows; excluded {report['excluded_rows']}.")


if __name__ == "__main__":
    main()
