"""Ornament merge (PLAN [P7], DEC-31): inventory O relabels every coda Sharma et al.'s released flags mark as
ornamented to its neighbour's inventory-A type, then DEC-28's switch-structure runner is rerun on O and
compared with the committed inventory-A run.

Stage `build` (`config/rq2_o_ornament_merged.json`):
- Match every retained `[D1]` coda (`data/processed/rq1_targets.jsonl`, context-only codas included) to a
  `[D2]` row (`data/raw/sperm-whale-dialogues.csv`) one-to-one on (REC, TsTo, Duration), TsTo and Duration
  rounded to `match_decimals`, as `inventories.check_d1_agreement` does; a matched row is ornamented if
  `data/raw/ornaments.p`'s value at that `[D2]` row's position is 1.
- An ornamented coda's neighbour is sought among the whale's immediately preceding and following codas in
  the same exchange (its own stream, sorted by onset, as `switch_structure.own_streams` groups rows): a
  neighbour qualifies if its onset is within `neighbour_window_s` and its raw nClicks is exactly one fewer.
  Both qualify -> the nearer by onset distance, tie -> preceding; one qualifies -> that one; none -> the
  coda keeps its inventory-A label.
- The new label replaces the old one wherever the coda appears (`inventories.relabel`/`check_relabelled`),
  as for the RQ2 inventories (DEC-22); targets, folds, S and the UNK policy are unchanged.
- Every computed count is checked against `config['expected']` (computed independently before this module
  was run, DEC-31); any mismatch raises and reports both sides without adjusting either.

Stage `compare` (`config/rq1_ornament_merge.json`):
- DEC-31's primary (`LL(PD) - LL(R)` in `A_B_S` on O), the per-coda change from inventory A, the `PD - R`
  gain split by switch/return, `LL(PD) - LL(P)` on O beside A, and O's v4 own/partner/timing contrasts beside
  RQ1's, from the committed inventory-A switch-structure run and O's own.

Standard library plus `inventories`'/`switch_structure`'s/`history_length`'s own numeric dependencies
(numpy, scipy) only; no new third-party dependency.
"""
import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from .evaluation import day_bootstrap_bits, sign_flip_p
from .history_length import by_coda, order_contrast
from .preprocessing.inventories import (check_relabelled, count, inventory_statistics, number, relabel, sha256)
from .preprocessing.io_utils import read_csv_rows, read_json, read_jsonl, write_jsonl, write_text

COMPONENTS = ('log_likelihood_bits', 'repeat_bits', 'switch_choice_bits')


# ---------------------------------------------------------------- matching [D1] to [D2] and the flags

def match_d1_to_d2(targets, normalized, d2_rows, decimals):
    """coda_id -> [D2] row index, one-to-one on (REC, round(TsTo, decimals), round(Duration, decimals)), the
    same key `inventories.check_d1_agreement` uses (duplicated here rather than imported, so this module does
    not depend on that function's own signature/behaviour); raises if a shared key is not one-to-one."""
    def key(rec, onset, duration, what):
        return rec, round(number(onset, f"{what} TsTo"), decimals), round(number(duration, f"{what} Duration"), decimals)
    d2_keys = Counter(key(r["REC"], r["TsTo"], r["Duration"], f"[D2] row {i}") for i, r in enumerate(d2_rows))
    d2_index = {key(r["REC"], r["TsTo"], r["Duration"], f"[D2] row {i}"): i for i, r in enumerate(d2_rows)}
    d1_keys, unmatchable = defaultdict(list), 0
    for row in targets:
        raw = normalized[row["coda_id"]]["raw_values"]
        try:
            k = key(raw["REC"], raw["TsTo"], raw["Duration"], row["coda_id"])
        except ValueError:
            unmatchable += 1
            continue
        d1_keys[k].append(row["coda_id"])
    shared = set(d1_keys) & set(d2_keys)
    if any(len(d1_keys[k]) > 1 or d2_keys[k] > 1 for k in shared):
        raise ValueError("A matched (REC, TsTo, Duration) key occurs more than once; matching is not one-to-one")
    return {d1_keys[k][0]: d2_index[k] for k in shared}, unmatchable


def check_ornament_flags(flags, sha, expected_sha, d2_rows):
    """DEC-31 step 1's gate: the released flags are a list of one 0/1 integer per [D2] row, with the
    recorded SHA-256, before anything else is computed from them."""
    if sha != expected_sha:
        raise ValueError(f"ornaments.p SHA-256 {sha} does not match the recorded {expected_sha}")
    if not isinstance(flags, list) or len(flags) != len(d2_rows) or any(v not in (0, 1) for v in flags):
        raise ValueError("ornaments.p must hold one 0/1 flag per sperm-whale-dialogues.csv row")


# ---------------------------------------------------------------- the neighbour rule

def exchange_streams(rows):
    """(exchange_id, whale_id) -> that whale's rows in the exchange, oldest first. Same grouping key as
    switch_structure.own_streams, but keeping every row (not only those earlier than a given one), since the
    neighbour rule looks both backward and forward."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["exchange_id"], r["whale_id"]].append(r)
    return {key: sorted(group, key=lambda r: r["onset_s"]) for key, group in groups.items()}


def find_neighbour(coda_id, index, stream, n_clicks, window_s):
    """The chosen neighbour's coda_id and side ('preceding'/'following'), or (None, None) without one.
    A neighbour qualifies if its onset is within `window_s` and its own nClicks is exactly one fewer than
    `n_clicks`; both qualifying -> nearer by onset distance, tie -> preceding."""
    own_onset = stream[index]["onset_s"]
    candidates = []
    if index > 0:
        candidates.append(("preceding", stream[index - 1]))
    if index + 1 < len(stream):
        candidates.append(("following", stream[index + 1]))

    def qualifies(row):
        return abs(row["onset_s"] - own_onset) <= window_s and n_clicks[row["coda_id"]] == n_clicks[coda_id] - 1

    qualifying = [(side, row, abs(row["onset_s"] - own_onset)) for side, row in candidates if qualifies(row)]
    if not qualifying:
        return None, None
    if len(qualifying) == 2 and qualifying[0][2] > qualifying[1][2]:
        qualifying = qualifying[::-1]  # preceding first; ties keep that order (tie -> preceding)
    side, row, _ = qualifying[0]
    return row["coda_id"], side


def relabel_ornaments(targets, normalized, flagged, window_s):
    """labels (coda_id -> inventory-A label, changed where a flagged coda has a qualifying neighbour) and a
    record per flagged coda (old/new label, chosen neighbour and side, or None)."""
    reference = {r["coda_id"]: r["Ct_raw"] for r in targets}
    streams = exchange_streams(targets)
    position = {r["coda_id"]: (key, i) for key, rows in streams.items() for i, r in enumerate(rows)}
    n_clicks = {cid: count(normalized[cid]["raw_values"]["nClicks"], f"{cid} nClicks") for cid in reference}

    labels, records = dict(reference), []
    for coda_id in sorted(flagged):
        key, index = position[coda_id]
        neighbour_id, side = find_neighbour(coda_id, index, streams[key], n_clicks, window_s)
        new_label = reference[coda_id] if neighbour_id is None else reference[neighbour_id]
        labels[coda_id] = new_label
        records.append(dict(coda_id=coda_id, old_label=reference[coda_id], new_label=new_label,
                            neighbour_id=neighbour_id, side=side))
    return labels, records


# ---------------------------------------------------------------- build stage

def own_lag_counts(targets, comparison, labels):
    """Comparison targets whose own lag 1 (A_history[-1]) differs from lag 2 (A_history[-2]), and of those,
    how many equal lag 2 (a return right after a switch, DEC-31's gain-split population)."""
    history = {r["coda_id"]: [labels[i] for i in r["A_history_ids"]] for r in comparison}
    differ, returns = 0, 0
    for r in comparison:
        hist = history[r["coda_id"]]
        if len(hist) < 2:
            continue
        lag1, lag2 = hist[-1], hist[-2]
        if lag1 != lag2:
            differ += 1
            returns += labels[r["coda_id"]] == lag2
    return differ, returns


def build_statistics(targets, normalized, d2_rows, flags, match, flagged, records, labels, reference, unmatchable):
    comparison = [r for r in targets if r["comparison_eligible"]]
    changed_all = sum(labels[cid] != reference[cid] for cid in reference)
    changed_comparison = sum(labels[r["coda_id"]] != reference[r["coda_id"]] for r in comparison)
    changes = Counter((r["old_label"], r["new_label"]) for r in records if r["new_label"] != r["old_label"])
    top_changes = [[list(pair), n] for pair, n in changes.most_common(10)]
    with_neighbour = [r for r in records if r["neighbour_id"] is not None]
    chosen_flagged = sum(r["neighbour_id"] in flagged for r in with_neighbour)
    stats_a = inventory_statistics(targets, reference, reference)
    stats_o = inventory_statistics(targets, labels, reference)
    ne_a, eq_a = own_lag_counts(targets, comparison, reference)
    ne_o, eq_o = own_lag_counts(targets, comparison, labels)
    return {
        "d1_rows": len(targets), "unmatchable_d1_codas": unmatchable, "matched": len(match),
        "flagged": len(flagged), "flagged_with_neighbour": len(with_neighbour),
        "flagged_with_neighbour_preceding": sum(r["side"] == "preceding" for r in with_neighbour),
        "flagged_with_neighbour_following": sum(r["side"] == "following" for r in with_neighbour),
        "flagged_without_neighbour": len(flagged) - len(with_neighbour),
        "chosen_neighbours_flagged": chosen_flagged,
        "rows_changed": changed_all, "comparison_targets_changed": changed_comparison,
        "top_changes": top_changes,
        "comparison_distinct_labels": stats_o["comparison_labels"],
        "repeats_of_own_last_a": stats_a["repeats"], "repeats_of_own_last_o": stats_o["repeats"],
        "repeat_switch_flips": stats_o["repeat_switch_flips"],
        "own_lag1_ne_lag2_a": ne_a, "own_lag1_ne_lag2_o": ne_o,
        "target_equals_own_lag2_a": eq_a, "target_equals_own_lag2_o": eq_o,
        "statistics_a": stats_a, "statistics_o": stats_o,
    }


def check_expected(stats, expected):
    """Every expected count must match exactly (DEC-31); no field is adjusted to make them agree."""
    mismatches = []
    for name, wanted in expected.items():
        got = stats.get(name)
        if name == "top_changes":
            if got[:len(wanted)] != wanted:
                mismatches.append((name, wanted, got))
        elif got != wanted:
            mismatches.append((name, wanted, got))
    return mismatches


def run_build(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    name, ident = config["name"], config["id"]

    inputs = {k: root / config[k] for k in ("targets", "normalized_codas", "rhythm_source", "ornament_flags")}
    targets = read_jsonl(inputs["targets"])
    normalized = {r["coda_id"]: r for r in read_jsonl(inputs["normalized_codas"])}
    if len({r["coda_id"] for r in targets}) != len(targets):
        raise ValueError("Duplicate coda IDs in targets")
    for row in targets:
        if row["coda_id"] not in normalized:
            raise ValueError(f"{row['coda_id']} is missing from the normalized codas")
    _, d2_rows = read_csv_rows(inputs["rhythm_source"], encoding="utf-8-sig")
    flags_bytes = inputs["ornament_flags"].read_bytes()
    flags = pickle.loads(flags_bytes)  # binary: a list of 0/1 ints, stdlib unpickles it
    check_ornament_flags(flags, sha256(inputs["ornament_flags"]), config["ornament_flags_sha256"], d2_rows)

    match, unmatchable = match_d1_to_d2(targets, normalized, d2_rows, config["match_decimals"])
    flagged = {cid for cid, idx in match.items() if flags[idx] == 1}
    labels, records = relabel_ornaments(targets, normalized, flagged, config["neighbour_window_s"])
    reference = {r["coda_id"]: r["Ct_raw"] for r in targets}

    rows = relabel(targets, labels, name)
    check_relabelled(targets, rows, labels, name)
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    out_path = output / f"rq2_{name}_targets.jsonl"
    write_jsonl(out_path, rows)
    check_relabelled(targets, read_jsonl(out_path), labels, name)

    stats = build_statistics(targets, normalized, d2_rows, flags, match, flagged, records, labels, reference, unmatchable)
    mismatches = check_expected(stats, config["expected"])
    if mismatches:
        lines = "\n".join(f"  {name}: expected {wanted!r}, computed {got!r}" for name, wanted, got in mismatches)
        raise ValueError(f"Inventory O's computed counts differ from DEC-31's expected counts:\n{lines}")

    result = dict(configuration=config, statistics=stats, records=records, flagged=sorted(flagged),
                  input_sha256={config[k]: sha256(p) for k, p in inputs.items()},
                  output_sha256={out_path.relative_to(root).as_posix(): sha256(out_path)})
    report = root / config["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, "\n".join(build_report_lines(result, config, targets)) + "\n")
    return result


def build_report_lines(result, config, targets):
    s = result["statistics"]
    comparison = [r for r in targets if r["comparison_eligible"]]
    lines = ["# RQ2 inventory O: ornament merge (PLAN [P7], DEC-31)", "",
             f"`guy_coda_type` (inventory A) with every coda Sharma et al.'s released flags (`{config['ornament_flags']}`, "
             f"{config['ornament_source']['repository']} commit `{config['ornament_source']['commit']}`) mark as ornamented "
             "relabelled to its neighbour's inventory-A type. Targets, exchanges, S, row order, folds and the UNK policy "
             "are RQ1's; only `Ct_raw`, `Ct`, `A_history` and `B_history` change (histories rebuilt from the history IDs), "
             "plus an `inventory` field.", "",
             "## Matching and flagging", "",
             f"`[D1]` matched one-to-one to `[D2]` on (REC, TsTo, Duration) rounded to {config['match_decimals']} decimals, "
             "as `inventories.check_d1_agreement` does.", "",
             "| Check | Result |", "|---|---|",
             f"| RQ1 rows (context-only codas included) | {s['d1_rows']:,} |",
             f"| Matched one-to-one to [D2] | {s['matched']:,} ({s['unmatchable_d1_codas']} without a parseable key) |",
             f"| Matched rows flagged ornamented | {s['flagged']:,} |",
             f"| Flagged with a qualifying neighbour (preceding / following) | {s['flagged_with_neighbour']:,} "
             f"({s['flagged_with_neighbour_preceding']} / {s['flagged_with_neighbour_following']}) |",
             f"| Flagged without a qualifying neighbour (kept) | {s['flagged_without_neighbour']:,} |",
             f"| Chosen neighbours that are themselves flagged | {s['chosen_neighbours_flagged']:,} |",
             f"| Rows whose label changes | {s['rows_changed']:,} |",
             f"| Comparison targets whose label changes | {s['comparison_targets_changed']:,} |", "",
             f"Neighbour rule: among the whale's immediately preceding and following codas in the same exchange (ordered "
             f"by onset), those within {config['neighbour_window_s']:g} s with exactly one fewer click, the nearer; a tie "
             "goes to the preceding one. Without a qualifying neighbour the coda keeps its label.", "",
             "## Flagged codas", "",
             "| Coda | Old label | New label | Neighbour | Side |", "|---|---|---|---|---|"]
    for r in result["records"]:
        neighbour = r["neighbour_id"] or "—"
        side = r["side"] or "kept: no neighbour"
        lines.append(f"| {r['coda_id']} | `{r['old_label']}` | `{r['new_label']}` | {neighbour} | {side} |")
    lines += ["", "## Inventory statistics (comparison targets unless stated)", "",
             "| | A | O |", "|---|---:|---:|",
             f"| Distinct labels | — | {s['comparison_distinct_labels']} |",
             f"| Repeats of own last | {s['repeats_of_own_last_a']:,} | {s['repeats_of_own_last_o']:,} |",
             f"| Own lag 1 ≠ lag 2 | {s['own_lag1_ne_lag2_a']:,} | {s['own_lag1_ne_lag2_o']:,} |",
             f"| ...of those, target = own lag 2 (return right after a switch) | {s['target_equals_own_lag2_a']:,} | "
             f"{s['target_equals_own_lag2_o']:,} |", "",
             f"Repeat/switch flips vs A: {s['repeat_switch_flips']:,}. Most frequent changes: "
             + ", ".join(f"`{a}`→`{b}` {n}" for (a, b), n in s["top_changes"][:5]) + ".", "",
             f"Comparison targets: {len(comparison):,}.", "",
             "## Reproducibility", "", "Input SHA-256:", ""]
    lines += [f"- `{p}`: `{h}`" for p, h in result["input_sha256"].items()]
    lines += ["", "Output SHA-256:", ""]
    lines += [f"- `{p}`: `{h}`" for p, h in result["output_sha256"].items()]
    return lines


# ---------------------------------------------------------------- compare stage

def gain_split(scored_pd, scored_r, condition, comparison_ids, rows_by_id):
    """PD - R per-coda gain, split into: (a) targets right after a switch (own lag 1 != own lag 2) whose
    target equals own lag 2; (b) other targets right after a switch; (c) everything else. The switch/return
    classification uses the stored (raw, pre-fold-vocabulary) `Ct_raw`/`A_history`, as DEC-28's own
    descriptive stage does ("labels as stored in Ct, not fold-mapped"), not `scored.jsonl`'s UNK-mapped
    `target`/`own_last`. `rows_by_id` maps coda_id -> that inventory's target row; a coda not in
    `comparison_ids` (e.g. UNK-mapped out of a fold) is left out, matching `scored.jsonl`'s own coverage."""
    pd_by, r_by = by_coda(scored_pd, condition), by_coda(scored_r, condition)
    if set(pd_by) != set(r_by):
        raise ValueError('PD and R scored different targets')
    groups = {"return_after_switch": [], "other_after_switch": [], "other": []}
    for coda_id in comparison_ids:
        if coda_id not in pd_by:
            continue
        gain = pd_by[coda_id]['log_likelihood_bits'] - r_by[coda_id]['log_likelihood_bits']
        hist = rows_by_id[coda_id]['A_history']
        if len(hist) >= 2 and hist[-1] != hist[-2]:
            groups["return_after_switch" if rows_by_id[coda_id]['Ct_raw'] == hist[-2] else "other_after_switch"].append(gain)
        else:
            groups["other"].append(gain)
    total = math.fsum(v for vs in groups.values() for v in vs)
    out = {}
    for name, values in groups.items():
        out[name] = dict(count=len(values), share_of_gain=(math.fsum(values) / total if total else None),
                         mean_per_target=(math.fsum(values) / len(values) if values else None))
    return out


def change_from_a(o_pd, o_r, a_pd, a_r, condition, boot):
    """Per-coda [LL_O(PD) - LL_O(R)] - [LL_A(PD) - LL_A(R)], paired by coda_id, day-bootstrapped. O and A
    share the same comparison targets and days; only their labels differ."""
    rows = {name: by_coda(scored, condition) for name, scored in
            (('o_pd', o_pd), ('o_r', o_r), ('a_pd', a_pd), ('a_r', a_r))}
    ids = set(rows['o_pd']) & set(rows['o_r']) & set(rows['a_pd']) & set(rows['a_r'])
    if not ids:
        raise ValueError('No shared comparison targets across O and A scored rows')
    days = defaultdict(list)
    for coda_id in ids:
        o_gain = rows['o_pd'][coda_id]['log_likelihood_bits'] - rows['o_r'][coda_id]['log_likelihood_bits']
        a_gain = rows['a_pd'][coda_id]['log_likelihood_bits'] - rows['a_r'][coda_id]['log_likelihood_bits']
        day = rows['o_pd'][coda_id]['recording_date']
        days[day].append(o_gain - a_gain)
    out = day_bootstrap_bits(dict(days), *boot)
    out['sign_flip_p_two_sided'] = sign_flip_p({d: math.fsum(v) for d, v in days.items()},
                                               {d: len(v) for d, v in days.items()}, 'two-sided')
    return out


def run_compare(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    boot = (config["bootstrap_replicates"], config["bootstrap_seed"], config["confidence"])
    condition = config["condition"]

    def scored(key):
        return read_jsonl(root / config[key])

    o_summary, a_summary = read_json(root / config["o_switch_structure_summary"]), read_json(root / config["a_switch_structure_summary"])
    o_scored = {v: scored(f"o_scored_{v}") for v in ("R", "P", "PD")}
    a_scored = {v: scored(f"a_scored_{v}") for v in ("R", "P", "PD")}
    o_targets = read_jsonl(root / config["o_targets"])
    comparison_ids = [r["coda_id"] for r in o_targets if r["comparison_eligible"]]
    o_rows_by_id = {r["coda_id"]: r for r in o_targets}

    primary = order_contrast(o_scored["PD"], o_scored["R"], condition, boot, sign_flip=True)
    change = change_from_a(o_scored["PD"], o_scored["R"], a_scored["PD"], a_scored["R"], condition, boot)
    split_o = gain_split(o_scored["PD"], o_scored["R"], condition, comparison_ids, o_rows_by_id)
    a_targets = read_jsonl(root / config["a_targets"])
    a_rows_by_id = {r["coda_id"]: r for r in a_targets}
    a_comparison_ids = [r["coda_id"] for r in a_targets if r["comparison_eligible"]]
    split_a = gain_split(a_scored["PD"], a_scored["R"], condition, a_comparison_ids, a_rows_by_id)
    pd_vs_p_o = order_contrast(o_scored["PD"], o_scored["P"], condition, boot, sign_flip=True)
    pd_vs_p_a = order_contrast(a_scored["PD"], a_scored["P"], condition, boot, sign_flip=True)

    def v4_contrasts(summary):
        """own/partner/timing, by name (logistic.py's fit_v2_lattice names them 'own_history',
        'partner_content', 'timing' at the top of a v4 evaluation summary, not per condition)."""
        return {k: summary['contrasts'][k]['components']['log_likelihood_bits']
                for k in ('own_history', 'partner_content', 'timing')}

    o_v4 = read_json(root / config["o_v4_evaluation_summary"])
    a_v4 = read_json(root / config["a_v4_evaluation_summary"])

    result = dict(
        primary=primary, change_from_a=change, gain_split={"O": split_o, "A": split_a},
        pd_vs_p={"O": pd_vs_p_o, "A": pd_vs_p_a},
        v4_contrasts={"O": v4_contrasts(o_v4), "A": v4_contrasts(a_v4)},
        stop_rules={"O": o_summary['stop_rules'], "O_dec23": o_summary['stop_rules_dec23'],
                    "A": a_summary['stop_rules'], "A_dec23": a_summary['stop_rules_dec23']},
        gate=o_summary['gate'], bootstrap=dict(zip(('replicates', 'seed', 'confidence'), boot)),
        condition=condition,
        input_sha256={config[k]: sha256(root / config[k]) for k in config
                      if isinstance(config[k], str) and (root / config[k]).is_file()})
    result['reading'] = compare_reading_lines(result)
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    write_text(output / 'summary.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
    report = root / config["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(compare_report_lines(result, config)))
    return result


def compare_reading_lines(result):
    """DEC-31's Reading applied to the primary, chosen mechanically by its interval; withheld if the gate or a
    relied-on stop rule fails."""
    low, high = result['primary']['log_likelihood_bits']['ci_bits_per_coda']
    p = result['primary']['log_likelihood_bits']['sign_flip_p_two_sided']
    fired = [v for v, rule in result['stop_rules']['O_dec23'].items() if rule['fired']]
    lines = []
    if not result['gate']['passed']:
        lines.append('No reading: O\'s reference variant does not reproduce logistic v4 (gate above).')
    elif any(v in ('R', 'P', 'PD') for v in fired):
        lines.append(f"No reading: DEC-23's stop rule fires on variant(s) {', '.join(v for v in fired if v in ('R', 'P', 'PD'))}.")
    elif high < 0:
        lines.append('No reading: the interval lies entirely below zero. That is a fitting flag, not a finding.')
    elif low > 0 and p < 0.05:
        lines.append('Primary positive: the collapsed view still adds usable information with the flagged ornaments '
                     'merged into inventory O.')
    else:
        qualifier = f' (the interval is above zero, but the two-sided sign-flip p = {p:.4f} is not below 0.05)' if low > 0 else ''
        lines.append("Primary null: the gain isn't detectable once flagged ornaments are merged; DEC-28's secondary "
                     f'is not evidence of switch-backs beyond ornaments{qualifier}.')
    change_low, change_high = result['change_from_a']['ci_bits_per_coda']
    if change_high < 0:
        lines.append('Change from A is below zero: the flagged ornaments carry a detectable part of DEC-28\'s gain.')
    elif change_low > 0:
        lines.append("Change from A is above zero: merging ornaments increased DEC-28's gain.")
    else:
        lines.append("Change from A spans zero: not detectably different from DEC-28's inventory-A gain.")
    return lines


def _stat(stats):
    ci = f"[{stats['ci_bits_per_coda'][0]:.5f}, {stats['ci_bits_per_coda'][1]:.5f}]"
    sign_flip = f", two-sided sign-flip p = {stats['sign_flip_p_two_sided']:.4f}" if 'sign_flip_p_two_sided' in stats else ''
    return (f"{stats['mean_bits_per_coda']:.5f} bits/coda (unweighted day mean {stats['unweighted_mean_bits_per_coda']:.5f}, "
            f"95% CI {ci}, {stats['positive_days']}/{stats['days']} days positive{sign_flip})")


def compare_report_lines(result, config):
    lines = ["# RQ1 ornament merge: inventory O vs A (PLAN [P7], DEC-31)", "",
             "DEC-28's switch-structure runner (`switch_structure.py`) rerun on inventory O (`ornament_merge.py --stage "
             "build`, `reports/rq2_o_ornament_merged.md`) and compared with the committed inventory-A run "
             f"(`{config['a_scored_PD']}` etc.), condition `{result['condition']}` only.", "",
             "## Primary (DEC-31): LL(PD) - LL(R) in A_B_S on O", "",
             f"Total: {_stat(result['primary']['log_likelihood_bits'])}.", "",
             f"Repeat part: {_stat(result['primary']['repeat_bits'])}.", "",
             f"Switch part: {_stat(result['primary']['switch_choice_bits'])}.", "",
             "## Change from inventory A", "",
             f"[LL_O(PD) - LL_O(R)] - [LL_A(PD) - LL_A(R)], paired by coda: {_stat(result['change_from_a'])}.", "",
             "## PD - R gain split", "", "| Inventory | Part | Count | Share of gain | Mean per target |",
             "|---|---|---:|---:|---:|"]
    for inv, split in result['gain_split'].items():
        for part, s in split.items():
            share = f"{s['share_of_gain']:.1%}" if s['share_of_gain'] is not None else "n/a"
            mean = f"{s['mean_per_target']:.5f}" if s['mean_per_target'] is not None else "n/a"
            lines.append(f"| {inv} | {part} | {s['count']} | {share} | {mean} |")
    lines += ["", "## LL(PD) - LL(P)", "",
             f"O: {_stat(result['pd_vs_p']['O']['log_likelihood_bits'])}.", "",
             f"A: {_stat(result['pd_vs_p']['A']['log_likelihood_bits'])}.", "",
             "## O's v4 lattice beside RQ1's", "",
             "From each run's own logistic v4 evaluation output (not the switch-structure runner).", "",
             "| Contrast | O | RQ1 |", "|---|---|---|"]
    for name, label in (('own_history', 'own (A_B_S - B_S)'), ('partner_content', 'partner (A_B_S - A_S)'),
                        ('timing', 'timing (A_B_S - A_B)')):
        lines.append(f"| {label} | {_stat(result['v4_contrasts']['O'][name])} | {_stat(result['v4_contrasts']['A'][name])} |")
    lines += ["", "## Gate (O's R reproducing logistic v4)", "",
             f"Passed: {result['gate']['passed']}.", "",
             "## Stop rules", "", "| Run | Variant | DEC-16 fires | DEC-23 fires |", "|---|---|---|---|"]
    for key in ("O", "A"):
        dec23 = result['stop_rules'][f'{key}_dec23']
        for variant, reasons in result['stop_rules'][key].items():
            lines.append(f"| {key} | {variant} | {len(reasons)} | {dec23.get(variant, {}).get('fired', 'n/a')} |")
    lines += ["", "## Reading", ""]
    lines += [f"- {line}" for line in result['reading']]
    lines += ["", "## Reproducibility", "", "Input SHA-256:", ""]
    lines += [f"- `{p}`: `{h}`" for p, h in result["input_sha256"].items()]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("build", "compare"), default="build")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.stage == "build":
        result = run_build(args.config)
        print(f"Inventory O: {result['statistics']['rows_changed']} of {result['statistics']['d1_rows']} rows "
              f"relabelled; every expected count matches.")
    else:
        result = run_compare(args.config)
        print('\n'.join(result['reading']))


if __name__ == "__main__":
    main()
