"""Feature construction for explicitly defined, two-caller exchanges only.

This module is not applied to the unresolved source CSV by the audit command.
Preparation can explicitly accept provisional assumptions without claiming confirmation.
History eligibility uses completed codas (end <= target onset), strictly earlier
onsets, and a maximum count. Overlap is measured separately from eligibility.

Interactional state S (PLAN [D5], DEC-1) uses only what is known at target onset,
relative to the immediately preceding coda of either caller (latest earlier onset):
turn_category, gap_s (end-to-start, continuations and switches) and
overlap_elapsed_s (onset-to-onset, overlaps). The older retrospective delay to the
last interlocutor coda is kept as a sensitivity field only.
"""
import math
from collections import defaultdict


def build_features(events, *, definitions_confirmed=False, provisional_assumptions_accepted=False, history_limit=10):
    if not definitions_confirmed and not provisional_assumptions_accepted:
        raise ValueError("Caller identities, exchange boundaries and timing must be confirmed")
    if type(history_limit) is not int or history_limit < 1:
        raise ValueError("history_limit must be a positive integer")
    groups = defaultdict(list)
    ids = set()
    for event in events:
        for field in ("coda_id", "whale_id", "exchange_id", "coda_type"):
            if not event.get(field):
                raise ValueError(f"Missing {field}")
        if event["coda_id"] in ids:
            raise ValueError("Duplicate coda ID")
        ids.add(event["coda_id"])
        start, end = event["onset_s"], event["offset_s"]
        if not all(math.isfinite(v) for v in (start, end)) or end <= start:
            raise ValueError("Invalid onset/offset")
        groups[event["exchange_id"]].append(event)
    output = []
    for exchange, group in groups.items():
        callers = {e["whale_id"] for e in group}
        if len(callers) != 2:
            raise ValueError("Each exchange must have exactly two callers")
        group = sorted(group, key=lambda e: e["onset_s"])
        if len({e["onset_s"] for e in group}) != len(group):
            raise ValueError("Tied onsets require an explicit ambiguity policy")
        for i, target in enumerate(group):
            previous = group[:i]
            eligible = [e for e in previous if e["offset_s"] <= target["onset_s"]]
            own = [e for e in eligible if e["whale_id"] == target["whale_id"]][-history_limit:]
            other = [e for e in eligible if e["whale_id"] != target["whale_id"]][-history_limit:]
            other_prior = [e for e in previous if e["whale_id"] != target["whale_id"]]
            predecessor = other_prior[-1] if other_prior else None
            immediate = previous[-1] if previous else None
            gap, elapsed = 0.0, 0.0
            if immediate is None:
                category = "exchange_start"
            elif target["onset_s"] < immediate["offset_s"]:
                category, elapsed = "overlap", target["onset_s"] - immediate["onset_s"]
            else:
                category = "self_continuation" if immediate["whale_id"] == target["whale_id"] else "switch"
                gap = target["onset_s"] - immediate["offset_s"]
            output.append({
                "coda_id": target["coda_id"], "Ct": target["coda_type"], "whale_id": target["whale_id"],
                "identity_status": "confirmed" if definitions_confirmed else "provisional_local_code",
                "exchange_id": exchange, "A_history": [e["coda_type"] for e in own],
                "B_history": [e["coda_type"] for e in other],
                "A_history_ids": [e["coda_id"] for e in own], "B_history_ids": [e["coda_id"] for e in other],
                "turn_predecessor_id": immediate["coda_id"] if immediate else None,
                "turn_category": category, "gap_s": gap, "overlap_elapsed_s": elapsed,
                "retrospective_predecessor_id": predecessor["coda_id"] if predecessor else None,
                "retrospective_delay_s": target["onset_s"] - predecessor["offset_s"] if predecessor else None,
                "overlaps_prior_interlocutor": any(e["offset_s"] > target["onset_s"] for e in other_prior),
            })
    return output
