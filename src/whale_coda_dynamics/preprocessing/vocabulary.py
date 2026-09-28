"""Training-only, coda-counted vocabulary shared by both preparation pipelines."""
from collections import Counter
import math


def fit_vocabulary(training_events, min_count=2, unk_token="UNK"):
    if type(min_count) is not int or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    if not isinstance(unk_token, str) or not unk_token:
        raise ValueError("unk_token must be a nonempty string")
    counts, ids = Counter(), set()
    for event in training_events:
        if event["split"] != "train":
            raise ValueError("Vocabulary fitting requires training events only")
        if event["coda_id"] in ids:
            raise ValueError("Training codas must occur exactly once")
        ids.add(event["coda_id"])
        label = event["coda_type"]
        if not isinstance(label, str) or not label or label == unk_token:
            raise ValueError("Invalid raw label or UNK token collision")
        counts[label] += 1
    if not counts:
        raise ValueError("Empty training corpus")
    return {"vocabulary": sorted(t for t, n in counts.items() if n >= min_count) + [unk_token],
            "training_type_counts": dict(sorted(counts.items())),
            "rare_training_types": sorted(t for t, n in counts.items() if n < min_count),
            "min_count": min_count, "unk_token": unk_token}


def apply_vocabulary(features, policy):
    known = set(policy["vocabulary"]) - {policy["unk_token"]}
    lookup = {r["coda_id"]: r.get("Ct_raw", r["Ct"]) for r in features}
    if len(lookup) != len(features):
        raise ValueError("Duplicate feature IDs")
    if policy["unk_token"] in lookup.values():
        raise ValueError("UNK token collides with a raw label")
    def mapped(label):
        return label if label in known else policy["unk_token"]
    result = []
    for row in features:
        new = dict(row, Ct_raw=lookup[row["coda_id"]], Ct=mapped(lookup[row["coda_id"]]))
        for prefix in ("A", "B"):
            new[prefix + "_history"] = [mapped(lookup[i]) for i in row[prefix + "_history_ids"]]
        result.append(new)
    return result


def smoothed_label_prior(training_comparison_rows, policy, alpha=1.0):
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Smoothing alpha must be finite and positive")
    if not training_comparison_rows or any(r["split"] != "train" or not r["comparison_eligible"]
                                           for r in training_comparison_rows):
        raise ValueError("Prior requires training comparison targets only")
    counts = Counter(r["Ct"] for r in training_comparison_rows)
    vocabulary = policy["vocabulary"]
    if set(counts) - set(vocabulary):
        raise ValueError("Prior targets have not been mapped to the vocabulary")
    denominator = sum(counts.values()) + alpha * len(vocabulary)
    return {t: (counts[t] + alpha) / denominator for t in vocabulary}


def validate_mapping(features, events, policy):
    """Independently recompute expected mappings from original event labels."""
    lookup = {e["coda_id"]: e for e in events}
    counts = Counter(e["coda_type"] for e in events if e["split"] == "train")
    expected_vocab = sorted(t for t, n in counts.items() if n >= policy["min_count"]) + [policy["unk_token"]]
    if policy["vocabulary"] != expected_vocab or policy["training_type_counts"] != dict(counts):
        raise ValueError("Vocabulary does not match training events")
    if len(features) != len(lookup) or {r["coda_id"] for r in features} != set(lookup):
        raise ValueError("Mapped feature coverage mismatch")
    for row in features:
        raw = lookup[row["coda_id"]]["coda_type"]
        expected = raw if counts[raw] >= policy["min_count"] else policy["unk_token"]
        if row["Ct_raw"] != raw or row["Ct"] != expected:
            raise ValueError("Mapped target mismatch")
        for prefix in ("A", "B"):
            labels = [lookup[i]["coda_type"] for i in row[prefix + "_history_ids"]]
            expected_history = [t if counts[t] >= policy["min_count"] else policy["unk_token"] for t in labels]
            if row[prefix + "_history"] != expected_history:
                raise ValueError("Mapped history mismatch")


def mapping_summary(features, policy):
    raw = {r["coda_id"]: r["Ct_raw"] for r in features}
    summary = {}
    for scope in ("all_retained", "comparison"):
        summary[scope] = {}
        for split in ("train", "validation", "test"):
            rows = [r for r in features if r["split"] == split and
                    (scope == "all_retained" or r["comparison_eligible"])]
            targets = Counter(r["Ct_raw"] for r in rows if r["Ct"] == policy["unk_token"])
            histories = Counter(raw[i] for r in rows for p in ("A", "B")
                                for i, t in zip(r[p + "_history_ids"], r[p + "_history"])
                                if t == policy["unk_token"])
            summary[scope][split] = {"targets": sum(targets.values()), "history_entries": sum(histories.values()),
                                     "mapped_target_types": dict(sorted(targets.items())),
                                     "mapped_history_types": dict(sorted(histories.items()))}
    return summary


def vocabulary_report(policy):
    lines = ["", "## Shared label vocabulary and UNK policy", "",
             f"Vocabulary counts each retained training coda once, including context-only codas. Minimum frequency: {policy['min_count']}. Below-threshold and unseen types map to `{policy['unk_token']}` in all partitions, including training.",
             f"Vocabulary size including UNK: {len(policy['vocabulary'])}. Rare training types: {policy['rare_training_types']}.",
             f"Prior: mapped training comparison-target counts with additive alpha={policy['smoothing_alpha']}, over exactly this vocabulary.",
             "Ct_raw preserves original labels; raw histories are recoverable via history IDs in the complete targets table. Refit from original labels for each future fold.", "",
             "| Scope | Split | UNK targets | UNK history entries | Mapped target types | Mapped history types |",
             "|---|---|---:|---:|---|---|"]
    for scope, splits in policy["mapping_counts"].items():
        for split, s in splits.items():
            lines.append(f"| {scope} | {split} | {s['targets']} | {s['history_entries']} | {s['mapped_target_types']} | {s['mapped_history_types']} |")
    return lines
