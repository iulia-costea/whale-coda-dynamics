"""Prepare target-aligned RQ2 tables: same conventions as prepare.py (RQ1),
applied to the Sharma, Gero, Payne, et al. (2024) rhythm/tempo labeling of a
substantially overlapping recording set, to test whether the RQ1 contrasts
hold under a different coda-type labeling scheme.

This module reuses `group_exchanges` (exchanges.py), `build_features`
(features.py), and `assign_splits`/`validate`/`write_table` (prepare.py)
unchanged: those are the parts of the RQ1 pipeline that do not depend on the
Assayag CSV's specific columns. Normalization is a new module (audit_rq2.py)
because Sharma's release has a genuinely different schema. Neither
exchanges.py nor prepare.py is modified, so RQ1 outputs are unaffected.

Several RQ1 assumptions do not transfer cleanly to this source and are
handled explicitly rather than forced silently; see reports/rq2.md for the
full accounting. In short:

- No Tag, Bout, Date, or Focal columns exist. `tag_id` is set equal to
  `recording_id` (REC); bout/exchange boundaries use REC alone as the sole
  hard-boundary field (no Bout to add); `strict_focal_subset` is therefore
  always False here.
- There is no calendar date, so day-held-out splitting is replaced with
  deployment-held-out splitting: `recording_date` (reused field name, for
  schema parity with rq1_day_splits) holds a deployment prefix parsed from
  REC (`^sw\\d+`), not a date. This keeps recordings from the same probable
  tag deployment together across splits, in the same spirit as RQ1's
  same-day rule, but carries no chronological meaning.
- Whale codes in this release span 1-11, not just 1/2; the same
  exactly-two-distinct-callers filter is applied, excluding more segments
  here than in the Assayag file. Retained Whale codes remain provisional
  local caller codes, not confirmed individual whales.
- rhythms.p is joined to sperm-whale-dialogues.csv strictly by row index;
  there is no shared key column to verify the join independently.
- Coda types unseen in training (in validation/test) are mapped to a
  catch-all UNK token, identically across all eight input conditions, with
  an additive-smoothed training label prior exported for baseline use,
  instead of being dropped.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .audit_rq2 import run as normalize_source_rq2
from .exchanges import group_exchanges
from .features import build_features
from .io_utils import read_json, read_jsonl, write_jsonl, write_text
from .prepare import S_FIELDS, TIMING_DESCRIPTIONS, assign_splits, interactional_state_summary, validate, write_table


def run_exchanges_rq2(config_path):
    # Mirrors exchanges.py's run(), swapping in audit_rq2's normalization.
    # exchanges.py hardcodes `from .audit import run as normalize_source`,
    # so it cannot be called directly against a different schema; this is
    # the minimal glue needed to reuse `group_exchanges` for RQ2.
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = read_json(path)
    audit = normalize_source_rq2(root / config["normalization_config"])
    source = root / config["input"]
    rows = read_jsonl(source)
    if len(rows) != audit["row_count"] or any(r["source_sha256"] != audit["source_sha256"] for r in rows):
        raise ValueError("Normalized input does not match the audited source")
    results = {t: group_exchanges(rows, t, config["hard_boundary_fields"], config["caller_field"])
               for t in sorted(set(config["sensitivity_thresholds_s"] + [config["gap_threshold_s"]]))}
    assignments, segments, summary = results[config["gap_threshold_s"]]
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "rq2_exchange_assignments.jsonl", assignments)
    write_jsonl(output / "rq2_exchange_segments.jsonl", segments)
    return {"configuration": config, "source_sha256": audit["source_sha256"],
            "normalized_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "primary": summary, "sensitivity": [r[2] for r in results.values()], "audit": audit}


def apply_unk_policy(features, unk_token, alpha):
    """Map coda types unseen in training to `unk_token` for validation/test
    rows, identically for the target label and both history lists, so every
    one of the eight input conditions sees a consistent vocabulary. Also
    returns an additive-smoothed training label prior (over the training
    vocabulary plus UNK) for use as the shared no-context/condition-(i)
    baseline distribution, since a hard majority-class prior cannot assign
    any probability to a label it never saw.
    """
    train_vocab = {row["Ct"] for row in features if row["split"] == "train"}
    train_vocab |= {t for row in features if row["split"] == "train" for t in row["A_history"] + row["B_history"]}
    remapped_targets, remapped_history_entries = 0, 0
    for row in features:
        row["Ct_raw"] = row["Ct"]
        if row["split"] == "train":
            continue
        if row["Ct"] not in train_vocab:
            row["Ct"] = unk_token
            remapped_targets += 1
        for key in ("A_history", "B_history"):
            mapped = [t if t in train_vocab else unk_token for t in row[key]]
            remapped_history_entries += sum(a != b for a, b in zip(row[key], mapped))
            row[key] = mapped
    vocabulary = sorted(train_vocab) + [unk_token]
    train_counts = Counter(row["Ct_raw"] for row in features if row["split"] == "train" and row["comparison_eligible"])
    total, v = sum(train_counts.values()), len(vocabulary)
    label_prior = {label: (train_counts.get(label, 0) + alpha) / (total + alpha * v) for label in vocabulary}
    return {"vocabulary": vocabulary, "label_prior": label_prior, "unk_token": unk_token,
            "smoothing_alpha": alpha, "remapped_targets": remapped_targets,
            "remapped_history_entries": remapped_history_entries, "train_vocabulary_size": len(train_vocab)}


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
    exchange_report = run_exchanges_rq2(root / config["exchange_config"])
    ec = exchange_report["configuration"]
    assignments = read_jsonl(root / ec["output_directory"] / "rq2_exchange_assignments.jsonl")
    segments = read_jsonl(root / ec["output_directory"] / "rq2_exchange_segments.jsonl")
    strict = {s["exchange_id"]: s["strict_focal_subset"] for s in segments}
    retained = [r for r in assignments if r["retained"]]
    if any(not r["recording_date"] for r in retained):
        raise ValueError("Missing deployment prefix; deployment-held-out splitting is not possible")
    for tag in {r["tag_id"] for r in retained}:
        if len({r["recording_date"] for r in retained if r["tag_id"] == tag}) != 1:
            raise ValueError("A recording spans multiple deployment prefixes; review grouping before splitting")
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
    comparison = [r for r in features if r["comparison_eligible"]]
    if any(r["retrospective_delay_s"] is None for r in comparison):
        raise ValueError("Every shared comparison target must have an observed signed delay")
    unk = apply_unk_policy(features, config["unk_token"], config["unk_smoothing_alpha"])
    stats = {}
    # Use the same vocabulary the UNK mapping itself is based on (all train
    # targets and history entries), not just comparison-eligible train Ct
    # values, so this stat cannot show a type as "unseen" that the mapping
    # already treated as seen.
    training_types = set(unk["vocabulary"]) - {unk["unk_token"]}
    for split in ("train", "validation", "test"):
        subset = [r for r in comparison if r["split"] == split]
        if not subset:
            raise ValueError(f"No comparison targets in {split}; review split configuration")
        stats[split] = {"deployments": sorted(day for day, assigned in splits.items() if assigned == split),
                        "recordings": sorted({r["tag_id"] for r in features if r["split"] == split}),
                        "targets": len(subset), "exchanges": len({r["exchange_id"] for r in subset}),
                        "coda_type_counts": dict(Counter(r["Ct"] for r in subset)),
                        "types_unseen_in_training": sorted({r["Ct"] for r in subset} - training_types)}
    output, reports = root / config["output_directory"], root / config["report_directory"]
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    write_table(output / "rq2_targets", features)
    write_table(output / "rq2_comparison", comparison)
    write_table(output / "rq2_day_splits", [{"recording_date": day, "split": splits[day]} for day in sorted(splits)])
    write_text(output / "rq2_label_prior.json", json.dumps(
        {"vocabulary": unk["vocabulary"], "label_prior": unk["label_prior"],
         "unk_token": unk["unk_token"], "smoothing_alpha": unk["smoothing_alpha"]}, indent=2) + "\n")
    report = {"configuration": config, "exchange_configuration": ec,
              "source_sha256": exchange_report["source_sha256"], "raw_codas": exchange_report["audit"]["row_count"],
              "retained_target_rows": len(features), "comparison_targets": len(comparison),
              "excluded_for_history": len(features) - len(comparison),
              "strict_focal_comparison_targets": sum(r["strict_focal_subset"] for r in comparison),
              "splits": stats, "unk": unk,
              "validation": "passed: coverage, history timing/identity/order, target alignment, deployment/recording/exchange separation",
              "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("rq2_*")) if p.is_file()}}
    lines = ["# RQ2 preparation (Sharma, Gero, Payne, et al. rhythm/tempo labeling)", "",
             "Mirrors reports/rq1.md's structure and reuses RQ1's exchange grouping, feature construction, split",
             "assignment, and validation logic unchanged. See the Deviations section below before comparing counts",
             "directly against RQ1: several source assumptions do not transfer to this release.", "",
             f"Raw codas: {report['raw_codas']}. Retained target rows: {len(features)}. Comparison targets: {len(comparison)}.",
             f"Excluded for insufficient history: {report['excluded_for_history']}. Strict focal subset: {report['strict_focal_comparison_targets']} comparison targets (see Deviations: no Focal column exists, so this is always 0).",
             "", f"History: up to {limit} completed codas per caller, oldest first; at least {minimum} per caller for comparison.",
             "", "| Split | Deployments | Recordings (REC) | Exchanges with targets | Comparison targets |", "|---|---:|---:|---:|---:|"]
    for split, s in stats.items():
        lines.append(f"| {split} | {len(s['deployments'])} | {len(s['recordings'])} | {s['exchanges']} | {s['targets']} |")
    lines += ["", "## Deviations from the RQ1 pipeline", "",
              "- **No Tag/Bout/Date/Focal columns.** Sharma's release provides only REC, Whale, TsTo, Duration,",
              "  nClicks, and ICI columns. `tag_id` is set equal to `recording_id` (REC) rather than a distinct",
              "  concept; bout/exchange boundaries use REC alone as the sole hard-boundary field (no Bout to add).",
              "  `strict_focal_subset` is therefore always False here (no Focal column to define it from).",
              "- **No calendar date.** Day-held-out splitting is replaced with deployment-held-out splitting: the",
              "  `recording_date` column (kept under that name only for schema parity with `rq1_day_splits.csv`)",
              "  holds a deployment prefix parsed from REC with the pattern `^sw\\\\d+` (e.g. `sw061` from",
              f"  `sw061b001_124`), giving {len({r['recording_date'] for r in retained})} deployment groups. This is not a date and carries no",
              "  chronological meaning; it exists only so recordings from the same probable tag deployment stay",
              "  together across splits, in the same spirit as RQ1's same-day rule.",
              "- **Wider caller-code range.** Whale codes span 1-11 in this release (not just 1/2 as initially",
              "  described); the same exactly-two-distinct-callers filter is applied, so this excludes",
              "  proportionally more multi-caller segments than in the Assayag file. Retained codes remain",
              "  provisional local caller codes, not confirmed individual whales.",
              "- **No shared key for the rhythms.p join.** Coda-type labels come from `rhythms.p`, a plain list",
              "  aligned to `sperm-whale-dialogues.csv` strictly by row index; there is no key column to verify",
              "  this join independently, unlike Assayag's `codaNUM2018`.",
              "- **No per-row source identifier.** Unlike `codaNUM2018`, there is no natural per-row ID in this",
              "  release; `coda_id`/`source_coda_id` are built from the zero-based row index (`sharma:row:<i>`).",
              "- **One corrupt-duration row can exclude a large recording.** The same conservative rule as RQ1",
              "  (`exchanges.py`: any invalid onset/duration in a hard-boundary group excludes the whole group)",
              "  is unchanged here, but its impact is larger: a single `Duration=0` row in `sw061b003` (731 rows,",
              "  the largest REC in the file) excludes that entire recording. See Exchange exclusions below.",
              "- **UNK policy for unseen coda types.** See the Label vocabulary section below.", ""]
    lines += ["## Validation", "", report["validation"], "",
              "See reports/data_dictionary.md for the RQ1 feature definitions this module reuses and README.md",
              "for modeling limitations.", ""]
    for split, s in stats.items():
        lines.append(f"Types in {split} absent from training (post-UNK-mapping): {s['types_unseen_in_training']}.")
    lines += ["", "## RQ2 input conditions", "",
              "All eight conditions use identical comparison targets, deployment-held-out splits, and UNK mapping.", ""]
    lines += [f"- {name}: {', '.join(fields) or 'no context (training-only label prior)'}" for name, fields in config["input_conditions"].items()]
    lines += ["", "Primary RQ2 contrast (parallel to RQ1): held-out log-likelihood(A_B_S) minus log-likelihood(B_S).",
              TIMING_DESCRIPTIONS[config["timing_convention"]],
              "Deployment group replaces recording day; see Deviations above. Model folds and training seeds remain downstream evaluation work."]
    lines += interactional_state_summary(comparison, events)
    lines += ["", "## Label vocabulary and UNK policy", "",
              f"Raw rhythm-cluster labels: {len(exchange_report['audit']['rhythm_label_counts'])} distinct values (0-17).",
              f"Training vocabulary (targets and history entries observed in train): {unk['train_vocabulary_size']} labels, plus the catch-all `{unk['unk_token']}` token.",
              f"Validation/test target labels remapped to `{unk['unk_token']}`: {unk['remapped_targets']}.",
              f"Validation/test history entries remapped to `{unk['unk_token']}`: {unk['remapped_history_entries']}.",
              f"Smoothing: additive (alpha={unk['smoothing_alpha']}) over training comparison-target counts, applied identically across all eight conditions' shared no-context baseline (`data/processed/rq2_label_prior.json`), not as a separate per-condition parameter.",
              "`Ct_raw` preserves the pre-UNK-mapping label on every row for audit; `Ct`, `A_history`, and `B_history` hold the mapped values used for modeling.", ""]
    lines += ["", "## Audit flags", "", "Counts can overlap; row-level details are in codas_normalized_rq2.jsonl.", ""]
    lines += [f"- {name}: {count}" for name, count in exchange_report["audit"]["issue_counts"].items()]
    lines += ["", "## Exchange exclusions", ""]
    lines += [f"- {name}: {count} codas" for name, count in exchange_report["primary"]["excluded_codas_by_reason"].items()]
    lines += ["", "## Gap sensitivity", "", "| Gap (s) | Exchanges | Codas | Targets with both histories |", "|---:|---:|---:|---:|"]
    for result in exchange_report["sensitivity"]:
        lines.append(f"| {result['gap_threshold_s']} | {result['retained_exchanges']} | {result['retained_codas']} | {result['targets_with_both_histories']} |")
    lines += ["", "## Reproducibility", "", f"Combined source SHA-256 (CSV + rhythms.p): `{report['source_sha256']}`", "",
              "Run configuration:", "", "```json", json.dumps({"rq2": config, "exchanges": ec,
              "normalization": {k: v for k, v in exchange_report["audit"]["configuration"].items()}}, indent=2, ensure_ascii=False), "```", "",
              "Output SHA-256 checksums:", ""]
    lines += [f"- `{name}`: `{checksum}`" for name, checksum in report["artifact_sha256"].items()]
    write_text(reports / "rq2.md", "\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rq2.json")
    report = run(parser.parse_args().config)
    print(f"Prepared {report['comparison_targets']} comparison targets. See reports/rq2.md.")


if __name__ == "__main__":
    main()
