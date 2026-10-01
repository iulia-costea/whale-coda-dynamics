"""Deterministically assign RQ3 recording groups to clan-specific folds."""
import hashlib
from collections import Counter


FOLD_COLUMNS = ["source_dataset", "clan", "group_id", "fold", "n_codas"]


def fold_count(clan):
    """Return the plan-specified number of group-disjoint folds for one clan."""
    return 6 if clan == "EC2" else 5


def group_sort_key(dataset, clan, group_id, count):
    """Order larger groups first, with a stable seeded tie-breaker."""
    tie_breaker = hashlib.sha256(f"42:{clan}:{group_id}".encode("utf-8")).hexdigest()
    return (-count, tie_breaker, dataset, clan, group_id)


def build_fold_rows(eligible_records):
    """Assign each whole recording group to the smallest available clan fold."""
    group_counts = Counter(
        (record["source_dataset"], record["clan"], record["group_id"])
        for record in eligible_records
    )
    by_clan = {}
    for (dataset, clan, group_id), count in group_counts.items():
        by_clan.setdefault((dataset, clan), []).append((group_id, count))

    rows = []
    for (dataset, clan), groups in sorted(by_clan.items()):
        count = fold_count(clan)
        if len(groups) < count:
            raise ValueError(f"{dataset}/{clan} has {len(groups)} groups, fewer than its required {count} folds")
        sizes = [0] * count
        for group_id, n_codas in sorted(
            groups,
            key=lambda item: group_sort_key(dataset, clan, item[0], item[1]),
        ):
            fold = min(range(count), key=lambda index: (sizes[index], index))
            sizes[fold] += n_codas
            rows.append({
                "source_dataset": dataset,
                "clan": clan,
                "group_id": group_id,
                "fold": fold,
                "n_codas": n_codas,
            })
    return sorted(rows, key=lambda row: (row["source_dataset"], row["clan"], row["fold"], row["group_id"]))
