"""Feature construction for explicitly defined, two-caller exchanges only.

This module is not applied to the unresolved source CSV by the audit command.
Preparation can explicitly accept provisional assumptions without claiming confirmation.
History eligibility uses completed codas (end <= target onset), strictly earlier
onsets, and a maximum count. Overlap is measured separately from eligibility.
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
            category = "exchange_start" if not previous else (
                "same_caller_continuation" if previous[-1]["whale_id"] == target["whale_id"] else "caller_switch")
            output.append({
                "coda_id": target["coda_id"], "Ct": target["coda_type"], "whale_id": target["whale_id"],
                "identity_status": "confirmed" if definitions_confirmed else "provisional_local_code",
                "exchange_id": exchange, "A_history": [e["coda_type"] for e in own],
                "B_history": [e["coda_type"] for e in other],
                "A_history_ids": [e["coda_id"] for e in own], "B_history_ids": [e["coda_id"] for e in other],
                "response_predecessor_id": predecessor["coda_id"] if predecessor else None,
                "response_delay_s": target["onset_s"] - predecessor["offset_s"] if predecessor else None,
                "turn_category": category,
                "overlaps_prior_interlocutor": any(e["offset_s"] > target["onset_s"] for e in other_prior),
            })
    return output
