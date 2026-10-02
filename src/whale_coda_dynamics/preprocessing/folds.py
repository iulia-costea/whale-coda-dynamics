"""RQ1 leave-one-day-out partitions and training-only vocabularies (no models)."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path

from .io_utils import read_json, read_jsonl, write_text
from .prepare import validate
from .vocabulary import (fit_vocabulary, apply_vocabulary, validate_mapping,
                         smoothed_label_prior, mapping_summary)


def day_folds(days, seed):
    """Seeded day ring: next day is validation; each day serves each role once."""
    ordered = sorted(set(days), key=lambda d: (hashlib.sha256(f"{seed}:{d}".encode()).hexdigest(), d))
    if len(ordered) < 3:
        raise ValueError("Need at least three recording days")
    for i, test in enumerate(ordered):
        validation = ordered[(i + 1) % len(ordered)]
        yield {"fold_id": f"test-{test}", "test_day": test, "validation_day": validation,
               "day_splits": {d: "test" if d == test else "validation" if d == validation else "train"
                              for d in sorted(ordered)}}


def prepare_fold(rows, fold, preparation):
    """Return fresh mapped rows and policy; recover raw labels before every fit.

    Consumers select comparison_eligible rows, then only the condition's declared
    input fields. No numerical encoders are fitted here; those belong to models.
    """
    lookup = {r["coda_id"]: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate source target IDs")
    features, events = [], []
    for row in rows:
        if not row.get("Ct_raw"):
            raise ValueError("Original Ct_raw labels are required for fold fitting")
        day = row["recording_date"]
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError("Require ISO recording dates, not deployment labels")
        if row["timing_convention"] != preparation["timing_convention"]:
            raise ValueError("Prepared timing convention differs from configuration")
        split = fold["day_splits"][day]
        raw = dict(row, split=split, Ct=row["Ct_raw"])
        for prefix in ("A", "B"):
            raw[prefix + "_history"] = [lookup[c]["Ct_raw"] for c in row[prefix + "_history_ids"]]
        expected_eligible = min(len(raw["A_history"]), len(raw["B_history"])) >= preparation["minimum_history_per_caller"]
        if row["comparison_eligible"] != expected_eligible:
            raise ValueError("Comparison eligibility differs from configured history requirement")
        features.append(raw)
        events.append(dict(raw, coda_type=raw["Ct"]))
    # Recompute histories/timing and check day, tag, exchange and context separation.
    validate(features, events, preparation["history_limit"])
    policy = fit_vocabulary([e for e in events if e["split"] == "train"],
                            preparation["unk_min_count"], preparation["unk_token"])
    mapped = apply_vocabulary(features, policy)
    validate_mapping(mapped, events, policy)
    comparison = [r for r in mapped if r["comparison_eligible"]]
    if {r["split"] for r in comparison} != {"train", "validation", "test"}:
        raise ValueError("Every fold partition must contain comparison targets")
    policy["smoothing_alpha"] = preparation["unk_smoothing_alpha"]
    policy["label_prior"] = smoothed_label_prior([r for r in comparison if r["split"] == "train"],
                                                 policy, preparation["unk_smoothing_alpha"])
    policy["mapping_counts"] = mapping_summary(mapped, policy)
    return mapped, policy


def run(config_path):
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = read_json(path)
    prep_path, source = root / config["preparation_config"], root / config["input"]
    preparation, rows = read_json(prep_path), read_jsonl(source)
    days = {r["recording_date"] for r in rows}
    if len(days) != config["expected_days"]:
        raise ValueError("Unexpected number of recording days")
    if len({r["source_sha256"] for r in rows}) != 1:
        raise ValueError("Mixed source versions")
    folds, tested = [], Counter()
    for fold in day_folds(days, config["validation_seed"]):
        mapped, policy = prepare_fold(rows, fold, preparation)
        ids = {s: sorted(r["coda_id"] for r in mapped if r["split"] == s and r["comparison_eligible"])
               for s in ("train", "validation", "test")}
        tested.update(ids["test"])
        folds.append(dict(fold, comparison_ids=ids, comparison_counts={s: len(v) for s, v in ids.items()},
                          vocabulary_policy=policy))
    expected = Counter({r["coda_id"]: 1 for r in rows if r["comparison_eligible"]})
    if tested != expected:
        raise ValueError("Test targets must occur exactly once across folds")
    result = {"configuration": config, "preparation_configuration": preparation,
              "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "preparation_config_sha256": hashlib.sha256(prep_path.read_bytes()).hexdigest(),
              "source_sha256": rows[0]["source_sha256"],
              "validation_rule": "next day in cyclic SHA256(seed:day) order; independent of labels/outcomes",
              "comparison_targets": len(expected), "input_conditions": preparation["input_conditions"],
              "folds": folds}
    output, report = root / config["output"], root / config["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(output, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    lines = [f"# {config.get('title', 'RQ1 leave-one-day-out folds')}", "",
             f"{len(folds)} folds; {len(expected)} comparison targets, each assigned to test exactly once. No predictions or models yet.",
             "Each fold has one test day, one validation day, and all remaining days for training. Validation is the next day in a seeded hash-ordered ring; each day is validation once. This rule does not inspect labels or outcomes.",
             "Original labels are recovered from Ct_raw and history IDs. Vocabulary counts all retained training codas once, including context-only codas. Priors use training comparison targets only. No numerical transformations are fitted at this stage.",
             "All eight conditions share each fold's targets, vocabulary and partitions. Preparation's original fixed-split tables remain unchanged.", "",
             "| Test day | Validation day | Train targets | Validation targets | Test targets | Vocabulary incl. UNK | Train UNK | Validation UNK | Test UNK |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for f in folds:
        n, p = f["comparison_counts"], f["vocabulary_policy"]
        u = p["mapping_counts"]["comparison"]
        lines.append(f"| {f['test_day']} | {f['validation_day']} | {n['train']} | {n['validation']} | {n['test']} | {len(p['vocabulary'])} | {u['train']['targets']} | {u['validation']['targets']} | {u['test']['targets']} |")
    lines += ["", "Validated: original histories and timing recomputed; day/tag/exchange/context separation; mapped labels independently recomputed; nonempty partitions; exact once-only test coverage.",
              f"Manifest: `{config['output']}`. Contains target IDs, configurations, raw training counts, priors, and target/history UNK counts for every fold.",
              f"Input SHA-256: `{result['input_sha256']}`.", ""]
    write_text(report, "\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rq1_folds.json")
    result = run(parser.parse_args().config)
    print(f"Prepared {len(result['folds'])} folds covering {result['comparison_targets']} test targets exactly once.")


if __name__ == "__main__":
    main()
