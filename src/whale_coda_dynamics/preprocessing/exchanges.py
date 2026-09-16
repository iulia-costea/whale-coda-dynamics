"""Provisional grouping and sample-size audit; no confirmed identity claims."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .audit import run as normalize_source


def group_exchanges(rows, threshold, boundary_fields, caller_field):
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Gap threshold must be finite and positive")
    groups = defaultdict(list)
    assignments, segments = [], []

    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    def assignment(row, exchange=None, reasons=(), caller=None):
        return {
            "coda_id": row["coda_id"], "source_coda_id": row["source_coda_id"],
            "source_row_index": row["source_row_index"], "exchange_id": exchange,
            "tag_id": row["tag_id"], "recording_id": row["recording_id"],
            "recording_date": row["date_candidate"],
            "source_bout_code": row["source_bout_code"],
            "local_caller_id": caller, "source_whale_code": row[caller_field],
            "identity_status": "provisional_local_code",
            "source_onset_s": row["source_onset_s"], "duration_s": row["duration_s"],
            "coda_type": row["coda_type"], "retained": not reasons,
            "exclusion_reasons": list(reasons),
        }

    for row in rows:
        key = tuple(row[f] for f in boundary_fields)
        if any(value is None or value == "" for value in key):
            assignments.append(assignment(row, reasons=["missing_boundary"]))
        else:
            groups[key].append(row)

    for key, group in sorted(groups.items()):
        # Never bridge across a row whose position or duration cannot be established.
        if any(not finite(r["source_onset_s"]) or r["source_onset_s"] < 0
               or not finite(r["duration_s"]) or r["duration_s"] <= 0 for r in group):
            assignments.extend(assignment(r, reasons=["group_invalid_timing"]) for r in group)
            continue
        ordered = sorted(group, key=lambda r: (r["source_onset_s"], r["source_row_index"]))
        chunks = [[]]
        for row in ordered:
            previous = chunks[-1][-1] if chunks[-1] else None
            if previous and row["source_onset_s"] - (previous["source_onset_s"] + previous["duration_s"]) > threshold:
                chunks.append([])
            chunks[-1].append(row)
        for chunk in chunks:
            # Row IDs are stable within the exact source snapshot. Threshold is part of ID.
            exchange = f"provisional:g{threshold:g}:row{chunk[0]['source_row_index']}"
            codes = {r[caller_field] for r in chunk if r[caller_field] not in (None, "", "0")}
            reasons = []
            if any(r[caller_field] in (None, "", "0") for r in chunk):
                reasons.append("missing_caller_code")
            if len(codes) != 2:
                reasons.append("not_exactly_two_callers")
            if len({r["source_onset_s"] for r in chunk}) != len(chunk):
                reasons.append("tied_onsets")
            if any(not r["coda_type"] for r in chunk):
                reasons.append("missing_coda_type")
            strict = (not reasons and all(r["focal_status"] in (0, 1) for r in chunk)
                      and {r["focal_status"] for r in chunk} == {0, 1}
                      and all(len({r["focal_status"] for r in chunk if r[caller_field] == code}) == 1 for code in codes))
            eligible_both = 0
            eligible_own = 0
            if not reasons:
                for i, row in enumerate(chunk):
                    completed = [r for r in chunk[:i] if r["source_onset_s"] + r["duration_s"] <= row["source_onset_s"]]
                    own = any(r[caller_field] == row[caller_field] for r in completed)
                    other = any(r[caller_field] != row[caller_field] for r in completed)
                    eligible_own += own
                    eligible_both += own and other
            segments.append({"exchange_id": exchange, "boundary_values": dict(zip(boundary_fields, key)),
                             "coda_count": len(chunk), "caller_codes": sorted(codes),
                             "retained": not reasons, "exclusion_reasons": reasons,
                             "strict_focal_subset": strict,
                             "targets_with_own_history": eligible_own,
                             "targets_with_both_histories": eligible_both})
            for row in chunk:
                caller = f"{exchange}:whale{row[caller_field]}" if not reasons else None
                assignments.append(assignment(row, exchange, reasons, caller))
    assignments.sort(key=lambda r: r["source_row_index"])
    kept = [s for s in segments if s["retained"]]
    summary = {
        "gap_threshold_s": threshold, "input_codas": len(rows),
        "retained_codas": sum(s["coda_count"] for s in kept),
        "excluded_codas": sum(not a["retained"] for a in assignments),
        "candidate_segments": len(segments), "retained_exchanges": len(kept),
        "retained_tags": len({a["tag_id"] for a in assignments if a["retained"]}),
        "exchange_length_distribution": dict(sorted(Counter(s["coda_count"] for s in kept).items())),
        "targets_with_own_history": sum(s["targets_with_own_history"] for s in kept),
        "targets_with_both_histories": sum(s["targets_with_both_histories"] for s in kept),
        "strict_focal_exchanges": sum(s["strict_focal_subset"] for s in kept),
        "strict_focal_codas": sum(s["coda_count"] for s in kept if s["strict_focal_subset"]),
        "strict_focal_targets_with_both_histories": sum(s["targets_with_both_histories"] for s in kept if s["strict_focal_subset"]),
        "excluded_codas_by_reason": dict(Counter(reason for a in assignments for reason in a["exclusion_reasons"])),
    }
    assert len(assignments) == len(rows)
    assert summary["retained_codas"] + summary["excluded_codas"] == len(rows)
    return assignments, segments, summary


def run(config_path):
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = json.loads(path.read_text())
    # Rebuild normalization to prevent silently consuming stale source data.
    audit = normalize_source(root / config["normalization_config"])
    source = root / config["input"]
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if len(rows) != audit["row_count"] or any(r["source_sha256"] != audit["source_sha256"] for r in rows):
        raise ValueError("Normalized input does not match the audited source")
    results = {t: group_exchanges(rows, t, config["hard_boundary_fields"], config["caller_field"])
               for t in sorted(set(config["sensitivity_thresholds_s"] + [config["gap_threshold_s"]]))}
    assignments, segments, summary = results[config["gap_threshold_s"]]
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    for filename, records in [("exchange_assignments.jsonl", assignments), ("exchange_segments.jsonl", segments)]:
        with (output / filename).open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
    report = {"configuration": config, "source_sha256": audit["source_sha256"],
              "normalized_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "primary": summary, "sensitivity": [r[2] for r in results.values()], "audit": audit}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/exchanges.json")
    args = parser.parse_args()
    print(json.dumps(run(args.config)["primary"], indent=2))


if __name__ == "__main__":
    main()
