"""Prepare target-aligned RQ1 tables under explicit provisional assumptions."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .exchanges import run as run_exchanges
from .features import build_features
from .vocabulary import (fit_vocabulary, apply_vocabulary, smoothed_label_prior,
                         validate_mapping, mapping_summary, vocabulary_report)
from .io_utils import read_json, read_jsonl, write_jsonl, write_text

# S field list per timing convention (PLAN [D5], DEC-1). onset_available is primary;
# the retrospective delay to the last interlocutor coda is a sensitivity run only.
S_FIELDS = {"onset_available": ["turn_category", "gap_s", "overlap_elapsed_s"],
            "retrospective_signed_end_to_start": ["turn_category", "retrospective_delay_s"]}
TIMING_DESCRIPTIONS = {
    "onset_available": "Timing convention: onset_available. S holds only what is known at target onset, relative to the "
                       "immediately preceding coda of either caller: turn_category, gap_s (end-to-start, continuations and "
                       "switches) and overlap_elapsed_s (onset-to-onset, overlaps). retrospective_delay_s is a sensitivity field in no S list.",
    "retrospective_signed_end_to_start": "Timing convention: retrospective_signed_end_to_start (sensitivity run). S uses the signed "
                       "end-to-start delay to the most recently started prior interlocutor coda. Negative delay uses an endpoint "
                       "after target onset; this is not online onset-time prediction."}


def interactional_state_summary(comparison, events):
    """Report lines describing S on the comparison targets."""
    lookup = {e["coda_id"]: e for e in events}
    same_whale = sum(r["turn_category"] == "overlap" and lookup[r["turn_predecessor_id"]]["whale_id"] == r["whale_id"]
                     for r in comparison)
    non_adjacent = sum(r["turn_category"] in ("self_continuation", "switch") and any(
        e["exchange_id"] == r["exchange_id"] and e["onset_s"] < r["onset_s"] < e["offset_s"] for e in events)
        for r in comparison)
    lines = ["", "## Interactional state", "", "| turn_category | Comparison targets | Max gap_s | Max overlap_elapsed_s |", "|---|---:|---:|---:|"]
    for category in ("exchange_start", "self_continuation", "switch", "overlap"):
        subset = [r for r in comparison if r["turn_category"] == category]
        lines.append(f"| {category} | {len(subset)} | {max((r['gap_s'] for r in subset), default=0):.3f} | "
                     f"{max((r['overlap_elapsed_s'] for r in subset), default=0):.3f} |")
    lines += ["", f"Same-caller overlaps (preceding coda is the target caller's own): {same_whale}.",
              f"Targets starting after the preceding coda ends but before an earlier coda ends: {non_adjacent}.",
              f"Sensitivity field: targets whose retrospective_delay_s uses a post-onset endpoint: "
              f"{sum(r['delay_uses_post_onset_endpoint'] for r in comparison)}; overlaps_prior_interlocutor: "
              f"{sum(r['overlaps_prior_interlocutor'] for r in comparison)}."]
    return lines


def assign_splits(days, seed, validation_fraction, test_fraction):
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Invalid split fractions")
    days = sorted(set(days), key=lambda day: (hashlib.sha256(f"{seed}:{day}".encode()).hexdigest(), day))
    if len(days) < 3:
        raise ValueError("At least three days are required for separate train/validation/test sets")
    nv, nt = max(1, round(len(days) * validation_fraction)), max(1, round(len(days) * test_fraction))
    if nv + nt >= len(days):
        raise ValueError("Split fractions leave no training days")
    return {day: "test" if i < nt else "validation" if i < nt + nv else "train"
            for i, day in enumerate(days)}


def validate(features, events, limit):
    lookup = {e["coda_id"]: e for e in events}
    if len(lookup) != len(events) or len(features) != len(events):
        raise ValueError("Missing or duplicate event rows")
    if {r["coda_id"] for r in features} != set(lookup):
        raise ValueError("Feature coverage mismatch")
    for row in features:
        target = lookup[row["coda_id"]]
        prior = sorted([e for e in events if e["exchange_id"] == target["exchange_id"]
                        and e["onset_s"] < target["onset_s"]], key=lambda e: e["onset_s"])
        for prefix, same in (("A", True), ("B", False)):
            expected = [e for e in prior if (e["whale_id"] == target["whale_id"]) == same
                        and e["offset_s"] <= target["onset_s"]][-limit:]
            if row[prefix + "_history_ids"] != [e["coda_id"] for e in expected]:
                raise ValueError("History identity, ordering, completeness or boundary violation")
            if row[prefix + "_history"] != [e["coda_type"] for e in expected]:
                raise ValueError("History label mismatch")
            if any(e["tag_id"] != target["tag_id"] or e["split"] != target["split"] for e in expected):
                raise ValueError("History crosses split or tag")
        others = [e for e in prior if e["whale_id"] != target["whale_id"]]
        predecessor = others[-1] if others else None
        delay = target["onset_s"] - predecessor["offset_s"] if predecessor else None
        if row["retrospective_predecessor_id"] != (predecessor["coda_id"] if predecessor else None) or row["retrospective_delay_s"] != delay:
            raise ValueError("Retrospective delay misaligned")
        # Interactional state S: recomputed from the event list, relative to the latest earlier onset.
        immediate = max(prior, key=lambda e: e["onset_s"]) if prior else None
        if immediate is None:
            expected = ("exchange_start", None, 0.0, 0.0)
        elif immediate["offset_s"] > target["onset_s"]:
            expected = ("overlap", immediate["coda_id"], 0.0, target["onset_s"] - immediate["onset_s"])
        else:
            expected = ("self_continuation" if immediate["whale_id"] == target["whale_id"] else "switch",
                        immediate["coda_id"], target["onset_s"] - immediate["offset_s"], 0.0)
        if row["Ct"] != target["coda_type"] or (row["turn_category"], row["turn_predecessor_id"], row["gap_s"], row["overlap_elapsed_s"]) != expected:
            raise ValueError("Target or interactional state mismatch")
        if expected[0] == "overlap" and not 0 < row["overlap_elapsed_s"] < immediate["offset_s"] - immediate["onset_s"]:
            raise ValueError("Overlap elapsed time outside the preceding coda")
        if row["gap_s"] < 0 or row["overlap_elapsed_s"] < 0:
            raise ValueError("Negative interactional timing")
        if row["overlaps_prior_interlocutor"] != any(e["offset_s"] > target["onset_s"] for e in others):
            raise ValueError("Overlap mismatch")
    for field in ("recording_date", "tag_id", "exchange_id"):
        for value in {e[field] for e in events}:
            if len({e["split"] for e in events if e[field] == value}) != 1:
                raise ValueError(f"{field} occurs in multiple splits")


def write_table(path, rows):
    write_jsonl(path.with_suffix(".jsonl"), rows)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)


def run(config_path):
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = read_json(path)
    limit, minimum = config["history_limit"], config["minimum_history_per_caller"]
    if type(limit) is not int or type(minimum) is not int or not 1 <= minimum <= limit:
        raise ValueError("Require 1 <= minimum history <= history limit")
    if config["provisional_assumptions_accepted"] is not True:
        raise ValueError("This source requires explicit acceptance of provisional assumptions")
    if config["timing_convention"] not in S_FIELDS:
        raise ValueError(f"timing_convention must be one of {sorted(S_FIELDS)}")
    sources = {"A": ["A_history"], "B": ["B_history"], "S": S_FIELDS[config["timing_convention"]]}
    expected_conditions = {"_".join(k for i, k in enumerate(sources) if mask & (1 << i)) or "none":
                           [field for i, fields in enumerate(sources.values()) if mask & (1 << i) for field in fields]
                           for mask in range(8)}
    if config["input_conditions"] != expected_conditions:
        raise ValueError("Input conditions must cover exactly the eight A/B/S combinations")
    exchange_report = run_exchanges(root / config["exchange_config"])
    ec = exchange_report["configuration"]
    assignments = read_jsonl(root / ec["output_directory"] / "exchange_assignments.jsonl")
    segments = read_jsonl(root / ec["output_directory"] / "exchange_segments.jsonl")
    strict = {s["exchange_id"]: s["strict_focal_subset"] for s in segments}
    retained = [r for r in assignments if r["retained"]]
    if any(not r["recording_date"] for r in retained):
        raise ValueError("Missing recording date; day-held-out splitting is not possible")
    for tag in {r["tag_id"] for r in retained}:
        if len({r["recording_date"] for r in retained if r["tag_id"] == tag}) != 1:
            raise ValueError("A tag spans multiple dates; review day grouping before splitting")
    splits = assign_splits([r["recording_date"] for r in retained], config["split_seed"],
                           config["validation_day_fraction"], config["test_day_fraction"])
    events = [dict(r, whale_id=r["local_caller_id"], onset_s=r["source_onset_s"],
                   offset_s=r["source_onset_s"] + r["duration_s"], split=splits[r["recording_date"]]) for r in retained]
    features = build_features(events, provisional_assumptions_accepted=True, history_limit=limit)
    validate(features, events, limit)
    lookup = {e["coda_id"]: e for e in events}
    for row in features:
        e = lookup[row["coda_id"]]
        for field in ("source_coda_id", "source_row_index", "tag_id", "recording_id", "source_bout_code", "source_whale_code", "onset_s", "offset_s", "recording_date", "split"):
            row[field] = e[field]
        row["timing_convention"] = config["timing_convention"]
        row["delay_uses_post_onset_endpoint"] = row["retrospective_delay_s"] is not None and row["retrospective_delay_s"] < 0
        row["source_sha256"] = exchange_report["source_sha256"]
        row["strict_focal_subset"] = strict[row["exchange_id"]]
        row["A_history_length"], row["B_history_length"] = len(row["A_history"]), len(row["B_history"])
        row["comparison_eligible"] = min(row["A_history_length"], row["B_history_length"]) >= minimum
        row["comparison_exclusion_reasons"] = [f"insufficient_{prefix}_history" for prefix in ("A", "B") if row[prefix + "_history_length"] < minimum]
    features.sort(key=lambda r: r["source_row_index"])
    unk = fit_vocabulary([e for e in events if e["split"] == "train"],
                         config["unk_min_count"], config["unk_token"])
    features = apply_vocabulary(features, unk)
    validate_mapping(features, events, unk)
    unk["smoothing_alpha"] = config["unk_smoothing_alpha"]
    unk["label_prior"] = smoothed_label_prior(
        [r for r in features if r["split"] == "train" and r["comparison_eligible"]],
        unk, config["unk_smoothing_alpha"])
    unk["mapping_counts"] = mapping_summary(features, unk)
    comparison = [r for r in features if r["comparison_eligible"]]
    if any(r["retrospective_delay_s"] is None for r in comparison):
        raise ValueError("Every shared comparison target must have an observed signed delay")
    stats = {}
    training_types = set(unk["vocabulary"])
    for split in ("train", "validation", "test"):
        subset = [r for r in comparison if r["split"] == split]
        if not subset:
            raise ValueError(f"No comparison targets in {split}; review split configuration")
        stats[split] = {"days": sorted(day for day, assigned in splits.items() if assigned == split),
                        "tags": sorted({r["tag_id"] for r in features if r["split"] == split}),
                        "targets": len(subset), "exchanges": len({r["exchange_id"] for r in subset}),
                        "coda_type_counts": dict(Counter(r["Ct"] for r in subset)),
                        "types_unseen_in_training": sorted({r["Ct"] for r in subset} - training_types)}
    output, reports = root / config["output_directory"], root / config["report_directory"]
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    write_table(output / "rq1_targets", features)
    write_table(output / "rq1_comparison", comparison)
    write_table(output / "rq1_day_splits", [{"recording_date": day, "split": splits[day]} for day in sorted(splits)])
    write_text(output / "rq1_label_prior.json", json.dumps(unk, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    report = {"configuration": config, "exchange_configuration": ec,
              "source_sha256": exchange_report["source_sha256"], "raw_codas": exchange_report["primary"]["input_codas"],
              "retained_target_rows": len(features), "comparison_targets": len(comparison),
              "excluded_for_history": len(features) - len(comparison),
              "strict_focal_comparison_targets": sum(r["strict_focal_subset"] for r in comparison),
              "splits": stats, "unk": unk, "validation": "passed: coverage, history timing/identity/order, target alignment, day/tag/exchange separation",
              "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("rq1_*")) if p.is_file()}}
    lines = ["# RQ1 preparation", "", f"Raw codas: {report['raw_codas']}. Retained target rows: {len(features)}. Comparison targets: {len(comparison)}.",
             f"Excluded for insufficient history: {report['excluded_for_history']}. Strict focal subset: {report['strict_focal_comparison_targets']} comparison targets.",
             "", f"History: up to {limit} completed codas per caller, oldest first; at least {minimum} per caller for comparison.",
             "", "| Split | Days | Tags | Exchanges with targets | Comparison targets |", "|---|---:|---:|---:|---:|"]
    for split, s in stats.items():
        lines.append(f"| {split} | {len(s['days'])} | {len(s['tags'])} | {s['exchanges']} | {s['targets']} |")
    lines += ["", "## Validation", "", report["validation"], "",
              "See reports/data_dictionary.md for feature definitions and README.md for modeling limitations.", ""]
    for split, s in stats.items():
        lines.append(f"Types in {split} absent from training: {s['types_unseen_in_training']}.")
    lines += ["", "## RQ1 input conditions", "", "All eight conditions use identical comparison targets and day-held-out splits.", ""]
    lines += [f"- {name}: {', '.join(fields) or 'no context (training-only label prior)'}" for name, fields in config["input_conditions"].items()]
    lines += ["", "Primary RQ1 contrast: held-out log-likelihood(A_B_S) minus log-likelihood(B_S).",
              TIMING_DESCRIPTIONS[config["timing_convention"]],
              "Recording day is the normalized source Date, not a reconstructed absolute timestamp. Model folds and training seeds remain downstream evaluation work."]
    lines += interactional_state_summary(comparison, events)
    lines += vocabulary_report(unk)
    lines += ["", "## Audit flags", "", "Counts can overlap; row-level details are in codas_normalized.jsonl.", ""]
    lines += [f"- {name}: {count}" for name, count in exchange_report["audit"]["issue_counts"].items()]
    lines += ["", "## Exchange exclusions", ""]
    lines += [f"- {name}: {count} codas" for name, count in exchange_report["primary"]["excluded_codas_by_reason"].items()]
    lines += ["", "## Gap sensitivity", "", "| Gap (s) | Exchanges | Codas | Targets with both histories |", "|---:|---:|---:|---:|"]
    for result in exchange_report["sensitivity"]:
        lines.append(f"| {result['gap_threshold_s']} | {result['retained_exchanges']} | {result['retained_codas']} | {result['targets_with_both_histories']} |")
    lines += ["", "## Reproducibility", "", f"Source SHA-256: `{report['source_sha256']}`", "",
              "Run configuration:", "", "```json", json.dumps({"rq1": config, "exchanges": ec,
              "normalization": exchange_report["audit"]["configuration"]}, indent=2, ensure_ascii=False), "```", "",
              "Output SHA-256 checksums:", ""]
    lines += [f"- `{name}`: `{checksum}`" for name, checksum in report["artifact_sha256"].items()]
    write_text(reports / "rq1.md", "\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rq1.json")
    report = run(parser.parse_args().config)
    print(f"Prepared {report['comparison_targets']} comparison targets. See reports/rq1.md.")


if __name__ == "__main__":
    main()
