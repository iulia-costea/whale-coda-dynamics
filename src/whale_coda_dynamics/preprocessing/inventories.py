"""RQ2 token inventories (PLAN §7, DEC-22): relabel RQ1's targets from each coda's own intervals and duration.

Every row of `rq1_targets.jsonl` (context-only codas included) gets one label per inventory, computed
from its own raw values in `codas_normalized.jsonl`. Each inventory's targets file is RQ1's with only
`Ct_raw`, `Ct`, `A_history` and `B_history` replaced (histories rebuilt from the history IDs), plus an
`inventory` field; targets, exchanges, S, row order and coda IDs are RQ1's. Folds and models are then
RQ1's code, unchanged.

Before labelling, the rules are checked against their sources and the stage raises on a mismatch:
the tempo cuts are recomputed from `CarribeanCodas.csv` (`[R2-2]`), and the rhythm rule must
reproduce `rhythms.p` on every `sperm-whale-dialogues.csv` row (`[R2-1]`). Standard library only;
the rhythm templates and `tempos-dict.p` boundaries come from `sharma_inventory.json`.
"""
import argparse
from collections import Counter
import hashlib
import math
import pickle
import re
from pathlib import Path

from .io_utils import read_csv_rows, read_json, read_jsonl, write_jsonl, write_text

LABEL_FIELDS = ("Ct_raw", "Ct", "A_history", "B_history")
TEMPO_SUFFIX = re.compile(r"\|t[0-9]+$")
INTERVAL_TOLERANCE_S = 1e-6  # [D1]/[D2] intervals equal up to float printing differ by ~1e-16


def number(value, what):
    """Finite float from a raw string; raises instead of imputing."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Unparseable {what}: {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {what}: {value!r}")
    return result


def count(value, what):
    result = number(value, what)
    if not result.is_integer():
        raise ValueError(f"Non-integer {what}: {value!r}")
    return int(result)


def kde_extrema(durations, bandwidth, grid):
    """Strict interior local minima and maxima of an unnormalised Gaussian KDE on `grid`.

    Normalisation does not move extrema, so it is left out."""
    density = [math.fsum(math.exp(-0.5 * ((g - d) / bandwidth) ** 2) for d in durations) for g in grid]
    inner = range(1, len(grid) - 1)
    minima = [grid[i] for i in inner if density[i] < density[i - 1] and density[i] < density[i + 1]]
    maxima = [grid[i] for i in inner if density[i] > density[i - 1] and density[i] > density[i + 1]]
    return minima, maxima


def tempo_class(duration, cuts):
    """Number of cuts strictly below the duration (0..len(cuts))."""
    if not duration > 0:
        raise ValueError(f"Tempo needs a positive duration, got {duration!r}")
    return sum(cut < duration for cut in cuts)


def rhythm_label(n_clicks, intervals, templates, rule):
    """Sharma et al.'s rhythm (`[R2-1]`) and the number of other templates tied with it.

    Out-of-range click counts get the catch-all label. Otherwise the click positions (cumulative
    sums of the first n-1 intervals, 0 prepended, divided by their total) are compared with every
    template of n clicks by mean squared difference; ties go to the lowest index."""
    if n_clicks < rule["min_clicks"] or n_clicks > rule["max_clicks"]:
        return rule["out_of_range_label"], 0
    used = intervals[:n_clicks - 1]
    if len(used) != n_clicks - 1 or any(v < 0 for v in used):
        raise ValueError("Rhythm needs n-1 non-negative intervals")
    positions, total = [0.0], 0.0
    for value in used:
        total += value
        positions.append(total)
    if not total > 0:
        raise ValueError("Rhythm needs a positive total interval")
    positions = [p / total for p in positions]
    scores = [(sum((a - b) ** 2 for a, b in zip(positions, template)) / n_clicks, k)
              for k, template in enumerate(templates) if len(template) == n_clicks]
    if not scores:
        raise ValueError(f"No rhythm template with {n_clicks} clicks")
    best = min(scores)
    return best[1], sum(score == best[0] for score, _ in scores) - 1


def needed_intervals(raw, n_clicks, rule, what):
    if not rule["min_clicks"] <= n_clicks <= rule["max_clicks"]:
        return []
    return [number(raw[f"ICI{j}"], f"{what} ICI{j}") for j in range(1, n_clicks)]


def interval_columns(columns):
    return sorted((c for c in columns if re.fullmatch(r"ICI[0-9]+", c)), key=lambda c: int(c[3:]))


def kde_cuts(rows, kde):
    durations = [d for d in (number(r["Duration"], "CarribeanCodas Duration") for r in rows if r["Clan"] == kde["clan"])
                 if kde["duration_above_s"] < d < kde["duration_below_s"]]
    grid = [round(kde["grid_start_s"] + kde["grid_step_s"] * i, kde["grid_decimals"]) for i in range(kde["grid_points"])]
    minima, maxima = kde_extrema(durations, kde["bandwidth_s"], grid)
    if minima != kde["expected_minima_s"] or maxima != kde["expected_maxima_s"]:
        raise ValueError(f"KDE extrema {minima} / {maxima} differ from the expected "
                         f"{kde['expected_minima_s']} / {kde['expected_maxima_s']}")
    return {"durations": len(durations), "grid": [grid[0], grid[-1], len(grid)], "minima_s": minima, "maxima_s": maxima}


def check_rhythm_reproduction(rows, rhythms, templates, rule):
    """The rule must give `rhythms.p`'s label on every `[D2]` row (joined by position)."""
    if not isinstance(rhythms, list) or len(rhythms) != len(rows) or any(type(v) is not int for v in rhythms):
        raise ValueError("rhythms.p must hold one integer label per sperm-whale-dialogues.csv row")
    reproduced, ties = 0, 0
    for i, (row, expected) in enumerate(zip(rows, rhythms)):
        n = count(row["nClicks"], f"[D2] row {i} nClicks")
        label, tied = rhythm_label(n, needed_intervals(row, n, rule, f"[D2] row {i}"), templates, rule)
        reproduced += label == expected
        ties += tied
    if reproduced != len(rows):
        raise ValueError(f"Rhythm rule reproduces rhythms.p on {reproduced} of {len(rows)} rows only")
    return {"rows": len(rows), "reproduced": reproduced, "ties": ties}


def check_d1_agreement(normalized, d2_rows, rhythms, templates, rule, decimals):
    """`[D1]`'s own intervals vs `rhythms.p` on codas matched one-to-one on (REC, TsTo, Duration),
    TsTo and Duration rounded to `decimals` (the two releases print the same floats differently)."""
    def key(rec, onset, duration, what):
        return rec, round(number(onset, f"{what} TsTo"), decimals), round(number(duration, f"{what} Duration"), decimals)
    d2_keys = Counter(key(r["REC"], r["TsTo"], r["Duration"], f"[D2] row {i}") for i, r in enumerate(d2_rows))
    d2_index = {key(r["REC"], r["TsTo"], r["Duration"], f"[D2] row {i}"): i for i, r in enumerate(d2_rows)}
    d1_keys, unmatchable = {}, 0
    for record in normalized:
        raw = record["raw_values"]
        try:
            k = key(raw["REC"], raw["TsTo"], raw["Duration"], record["coda_id"])
        except ValueError:
            unmatchable += 1  # no usable key: cannot be matched, not part of the rule's labelling
            continue
        d1_keys.setdefault(k, []).append(record)
    shared = set(d1_keys) & set(d2_keys)
    if any(len(d1_keys[k]) > 1 or d2_keys[k] > 1 for k in shared):
        raise ValueError("A matched (REC, TsTo, Duration) key occurs more than once; matching is not one-to-one")
    agree, intervals_differ = 0, 0
    for k in shared:
        record, d2_row = d1_keys[k][0], d2_rows[d2_index[k]]
        n = count(record["raw_values"]["nClicks"], f"{record['coda_id']} nClicks")
        own = needed_intervals(record["raw_values"], n, rule, record["coda_id"])
        label, _ = rhythm_label(n, own, templates, rule)
        if label == rhythms[d2_index[k]]:
            agree += 1
            continue
        n2 = count(d2_row["nClicks"], f"[D2] row {d2_index[k]} nClicks")
        other = needed_intervals(d2_row, n2, rule, f"[D2] row {d2_index[k]}")
        intervals_differ += n != n2 or any(abs(a - b) > INTERVAL_TOLERANCE_S for a, b in zip(own, other))
    return {"d1_codas": len(normalized), "unmatchable_d1_codas": unmatchable, "matched": len(shared), "agree": agree,
            "disagreeing_with_different_intervals": intervals_differ}


def coda_labels(record, target, templates, rule, cuts):
    """All scheme and tempo values for one retained `[D1]` coda, from its own raw values."""
    raw, what = record["raw_values"], record["coda_id"]
    named = raw["guy_coda_type"].strip()
    if named != target["Ct_raw"]:
        raise ValueError(f"{what}: guy_coda_type differs from RQ1's Ct_raw")
    alternative = record["alternative_coda_type"]
    if not isinstance(alternative, str) or not alternative.strip():
        raise ValueError(f"{what}: missing decision_tree_coda_type")
    n = count(raw["nClicks"], f"{what} nClicks")
    rhythm, ties = rhythm_label(n, needed_intervals(raw, n, rule, what), templates, rule)
    duration = number(raw["Duration"], f"{what} Duration")
    schemes = {"guy_coda_type": named, "decision_tree_coda_type": alternative.strip(), "rhythm": f"r{rhythm}"}
    tempos = {name: tempo_class(duration, c) for name, c in cuts.items()}
    return schemes, tempos, ties


def odd_intervals(record, rule, columns):
    """Zero intervals among the first n-1, or non-zero intervals beyond them (rule applied as written)."""
    raw = record["raw_values"]
    n = count(raw["nClicks"], f"{record['coda_id']} nClicks")
    if not rule["min_clicks"] <= n <= rule["max_clicks"]:
        return None
    values = [number(raw[c], f"{record['coda_id']} {c}") for c in columns]
    if 0.0 in values[:n - 1] or any(values[n - 1:]):
        return {"coda_id": record["coda_id"], "nClicks": n, "guy_coda_type": raw["guy_coda_type"],
                "intervals": [raw[c] for c in columns[:n - 1]],
                "beyond": [raw[c] for i, c in enumerate(columns) if i >= n - 1 and values[i] != 0.0]}
    return None


def check_label(label, unk_token, combined):
    if not isinstance(label, str) or not label or label == unk_token:
        raise ValueError(f"Invalid label {label!r} (empty or the UNK token)")
    if combined:
        if label.count("|") != 1 or not TEMPO_SUFFIX.search(label):
            raise ValueError(f"Combined label {label!r} is not <scheme>|t<j>")
    elif "|" in label:
        raise ValueError(f"Label {label!r} contains '|' outside the combined format")


def relabel(targets, labels, name):
    return [dict(row, Ct_raw=labels[row["coda_id"]], Ct=labels[row["coda_id"]],
                 A_history=[labels[i] for i in row["A_history_ids"]],
                 B_history=[labels[i] for i in row["B_history_ids"]], inventory=name) for row in targets]


def check_relabelled(original, relabelled, labels, name):
    """Only the four label fields and `inventory` differ; same rows, order and IDs."""
    if len(original) != len(relabelled):
        raise ValueError(f"{name}: row count changed")
    for before, after in zip(original, relabelled):
        if list(after) != list(before) + ["inventory"] or after["inventory"] != name:
            raise ValueError(f"{name}: fields changed for {before['coda_id']}")
        if any(after[k] != v for k, v in before.items() if k not in LABEL_FIELDS):
            raise ValueError(f"{name}: a non-label field changed for {before['coda_id']}")
        if after["Ct_raw"] != labels[before["coda_id"]] or after["Ct"] != after["Ct_raw"]:
            raise ValueError(f"{name}: target label mismatch for {before['coda_id']}")
        for prefix in ("A", "B"):
            if after[prefix + "_history"] != [labels[i] for i in before[prefix + "_history_ids"]]:
                raise ValueError(f"{name}: history labels differ from the labels of their IDs")


def inventory_statistics(targets, labels, reference):
    comparison = [r for r in targets if r["comparison_eligible"]]
    counts = Counter(labels[r["coda_id"]] for r in comparison)
    every = Counter(labels[r["coda_id"]] for r in targets)
    def repeats(lab, row):
        return lab[row["coda_id"]] == lab[row["A_history_ids"][-1]]
    top, top_count = max(sorted(counts.items()), key=lambda item: item[1])
    return {"comparison_labels": len(counts), "top_label": top, "top_share": top_count / len(comparison),
            "repeats": sum(repeats(labels, r) for r in comparison),
            "all_row_labels": len(every), "labels_once": sum(v == 1 for v in every.values()),
            "differs_from_reference": sum(labels[r["coda_id"]] != reference[r["coda_id"]] for r in comparison),
            "repeat_switch_flips": sum(repeats(labels, r) != repeats(reference, r) for r in comparison)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    rule, kde = config["rhythm"], config["kde"]
    unk_token = read_json(root / config["preparation_config"])["unk_token"]
    sharma = read_json(root / config["sharma_inventory"])
    templates = sharma["rhythm_templates"]
    inputs = {name: root / config[name] for name in ("targets", "preparation_config", "normalized_codas",
                                                     "sharma_inventory", "tempo_source", "rhythm_source", "rhythm_labels")}

    _, tempo_rows = read_csv_rows(inputs["tempo_source"], encoding="utf-8-sig")
    kde_check = kde_cuts(tempo_rows, kde)
    cuts = {"kde": kde_check["minima_s"], "tempos_dict": sharma["tempo_boundaries_s"]}
    _, d2_rows = read_csv_rows(inputs["rhythm_source"], encoding="utf-8-sig")
    rhythms = pickle.loads(inputs["rhythm_labels"].read_bytes())  # binary: a list of ints, stdlib unpickles it
    rhythm_check = check_rhythm_reproduction(d2_rows, rhythms, templates, rule)

    targets = read_jsonl(inputs["targets"])
    normalized_rows = read_jsonl(inputs["normalized_codas"])
    match_check = check_d1_agreement(normalized_rows, d2_rows, rhythms, templates, rule, config["match_decimals"])
    normalized = {r["coda_id"]: r for r in normalized_rows}
    if len(normalized) != len(normalized_rows) or len({r["coda_id"] for r in targets}) != len(targets):
        raise ValueError("Duplicate coda IDs")
    reference = {r["coda_id"]: r["Ct_raw"] for r in targets}
    labels = {name: {} for name in config["inventories"]}
    tempo_values, label_ties = {}, 0
    for row in targets:
        if row["coda_id"] not in normalized:
            raise ValueError(f"{row['coda_id']} is missing from the normalized codas")
        schemes, tempos, ties = coda_labels(normalized[row["coda_id"]], row, templates, rule, cuts)
        tempo_values[row["coda_id"]], label_ties = tempos, label_ties + ties
        for name, spec in config["inventories"].items():
            base = schemes[spec["scheme"]]
            check_label(base, unk_token, combined=False)
            label = base if spec["tempo"] is None else f"{base}|t{tempos[spec['tempo']]}"
            check_label(label, unk_token, combined=spec["tempo"] is not None)
            labels[name][row["coda_id"]] = label
    for label in reference.values():
        check_label(label, unk_token, combined=False)
    columns = interval_columns(normalized_rows[0]["raw_values"])
    odd = [o for o in (odd_intervals(normalized[r["coda_id"]], rule, columns) for r in targets) if o]
    comparison = [r for r in targets if r["comparison_eligible"]]
    tempo_agreement = sum(tempo_values[r["coda_id"]]["kde"] == tempo_values[r["coda_id"]]["tempos_dict"] for r in comparison)

    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    outputs, statistics = {}, {"A": inventory_statistics(targets, reference, reference)}
    for name, spec in config["inventories"].items():
        rows = relabel(targets, labels[name], name)
        check_relabelled(targets, rows, labels[name], name)
        outputs[name] = output / f"rq2_{name}_targets.jsonl"
        write_jsonl(outputs[name], rows)
        check_relabelled(targets, read_jsonl(outputs[name]), labels[name], name)
        statistics[spec["id"]] = inventory_statistics(targets, labels[name], reference)

    result = {"configuration": config, "kde": kde_check, "tempo_cuts": cuts, "rhythm_reproduction": rhythm_check,
              "d1_agreement": match_check, "tempo_agreement": {"comparison_targets": len(comparison), "agree": tempo_agreement},
              "label_ties": label_ties, "odd_intervals": odd, "statistics": statistics,
              "input_sha256": {config[k]: sha256(p) for k, p in inputs.items()},
              "output_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in outputs.values()}}
    report = root / config["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, "\n".join(report_lines(result, config, targets, comparison)) + "\n")
    return result


def report_lines(result, config, targets, comparison):
    kde, rhythm, match = result["kde"], result["rhythm_reproduction"], result["d1_agreement"]
    tempo, cuts, rule, k = result["tempo_agreement"], result["tempo_cuts"], config["rhythm"], config["kde"]
    def seconds(values):
        return ", ".join(f"{v:g}" for v in values)
    lines = ["# RQ2 token inventories (DEC-22)", "",
             f"Each inventory relabels all {len(targets):,} rows of `{config['targets']}` ({len(comparison):,} comparison targets "
             "plus context-only codas) from each coda's own raw intervals and duration in "
             f"`{config['normalized_codas']}`. Only `Ct_raw`, `Ct`, `A_history` and `B_history` change (histories rebuilt "
             "from the history IDs), plus an `inventory` field; targets, exchanges, S, row order and coda IDs are RQ1's. "
             "Inventory A is `guy_coda_type`, RQ1's own labels. Labels are counted before any fit.", "",
             "## Rules", "",
             f"- Rhythm (`[R2-1]`): n = nClicks; n < {rule['min_clicks']} or n > {rule['max_clicks']} gives "
             f"r{rule['out_of_range_label']}; otherwise the click positions (cumulative sums of ICI1…ICI(n−1), 0 prepended, "
             "divided by their total) take the label of the template in `mean_codas.p` with n clicks that is nearest by "
             f"mean squared difference (ties: lowest index). Ties among the {len(targets):,} rows: {result['label_ties']}. Format `r<k>`.",
             f"- Tempo (`[R2-2]`): class j = number of cuts strictly below Duration. KDE cuts: {seconds(cuts['kde'])} s. "
             f"`tempos-dict.p` cuts (midpoints between adjacent classes): {seconds(cuts['tempos_dict'])} s. Format `t<j>`, "
             "combined with `|` (`r2|t4`, `1+1+3|t4`).", "",
             "## Checks", "",
             "| Check | Result |", "|---|---|",
             f"| KDE minima (tempo cuts), s | {seconds(kde['minima_s'])} |",
             f"| KDE maxima (modes), s | {seconds(kde['maxima_s'])} |",
             f"| Rhythm rule vs `rhythms.p`, all `[D2]` rows | {rhythm['reproduced']:,} / {rhythm['rows']:,} (ties: {rhythm['ties']}) |",
             f"| Rule on `[D1]`'s intervals vs `rhythms.p`, matched codas | {match['agree']:,} / {match['matched']:,} |",
             f"| KDE vs `tempos-dict.p` tempo class, comparison targets | {tempo['agree']:,} / {tempo['comparison_targets']:,} |", "",
             f"- KDE: unnormalised Gaussian kernel (bandwidth {k['bandwidth_s']} s) over the {kde['durations']:,} "
             f"{k['clan']} durations d with {k['duration_above_s']} < d < {k['duration_below_s']} s in `{config['tempo_source']}`, "
             f"evaluated at {kde['grid'][0]:.2f}, {kde['grid'][0] + k['grid_step_s']:.2f}, …, {kde['grid'][1]:.2f} s "
             f"({kde['grid'][2]} points); strict interior local minima and maxima. The stage raises unless they equal "
             f"{seconds(k['expected_minima_s'])} and {seconds(k['expected_maxima_s'])} s, and unless the rhythm rule "
             "reproduces every `rhythms.p` label.",
             f"- Matching `[D1]` to `[D2]`: one-to-one on (REC, TsTo, Duration) with TsTo and Duration rounded to "
             f"{config['match_decimals']} decimals, because the two releases print the same values differently "
             f"(`0.9277916000000002` vs `0.9277916`). {match['matched']:,} of the {match['d1_codas']:,} `[D1]` codas match "
             f"({match['unmatchable_d1_codas']} without a parseable key). Of the {match['matched'] - match['agree']} that disagree, "
             f"{match['disagreeing_with_different_intervals']} differ between the two releases in click count or in an interval "
             f"used by the rule (by more than {INTERVAL_TOLERANCE_S:g} s).", "",
             "## Codas with irregular intervals", "",
             "Retained codas with 3–10 clicks whose first n−1 intervals include a zero, or with non-zero intervals beyond "
             "them. The rule is applied to the first n−1 intervals as written; these are parsed values, not missing ones.", "",
             "| Coda | Comparison target | nClicks | guy_coda_type | ICI1…ICI(n−1) | Non-zero beyond n−1 |", "|---|---|---:|---|---|---|"]
    eligible = {r["coda_id"] for r in comparison}
    for o in result["odd_intervals"]:
        lines.append(f"| {o['coda_id']} | {'yes' if o['coda_id'] in eligible else 'no'} | {o['nClicks']} | {o['guy_coda_type']} | "
                     f"{', '.join(o['intervals'])} | {', '.join(o['beyond']) or '—'} |")
    lines += ["", "## Inventories", "",
              f"Counts on the {len(comparison):,} comparison targets unless stated. Repeats: targets equal to the whale's own "
              f"last label. Labels over all {len(targets):,} rows include context-only codas. \"Label ≠ A's\" is given for B "
              "only: the other inventories use different label strings. Flips: targets that repeat under one of A and "
              "the inventory and switch under the other.", "",
              "| ID | Inventory | File | Labels | Top label | Top share | Repeats | Labels, all rows | Occurring once | Label ≠ A's | Flips vs A |",
              "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    rows = [("A", "guy_coda_type (RQ1)", config["targets"])]
    rows += [(spec["id"], spec["description"], f"{config['output_directory']}/rq2_{name}_targets.jsonl")
             for name, spec in config["inventories"].items()]
    for ident, description, file in rows:
        s = result["statistics"][ident]
        differs = f"{s['differs_from_reference']:,}" if ident == "B" else "—"
        flips = f"{s['repeat_switch_flips']:,}" if ident != "A" else "—"
        lines.append(f"| {ident} | {description} | `{file}` | {s['comparison_labels']} | `{s['top_label']}` | {s['top_share']:.1%} | "
                     f"{s['repeats']:,} | {s['all_row_labels']} | {s['labels_once']} | {differs} | {flips} |")
    lines += ["", "UNK counts per fold come from each inventory's fold harness (`reports/rq2_<name>_folds.md`).", "",
              "## Reproducibility", "", "Input SHA-256:", ""]
    lines += [f"- `{p}`: `{h}`" for p, h in result["input_sha256"].items()]
    lines += ["", "Output SHA-256:", ""]
    lines += [f"- `{p}`: `{h}`" for p, h in result["output_sha256"].items()]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rq2_inventories.json")
    result = run(parser.parse_args().config)
    print(f"Wrote {len(result['output_sha256'])} inventories; rhythm rule reproduces "
          f"{result['rhythm_reproduction']['reproduced']}/{result['rhythm_reproduction']['rows']} rhythms.p labels.")


if __name__ == "__main__":
    main()
