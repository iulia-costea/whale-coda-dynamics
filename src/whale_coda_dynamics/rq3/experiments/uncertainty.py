"""Target-group bootstrap uncertainty intervals for fixed RQ3 transfer scores."""
import hashlib
from collections import defaultdict

from .data import (
    BOOTSTRAP_COLUMNS,
    TRANSFER_SCORE_COLUMNS,
    UNCERTAINTY_SUMMARY_COLUMNS,
    read_csv,
    write_csv,
)
from .evaluation import aggregate_transfer_loss, bootstrap_group_mean


def percentile(values, proportion):
    """Return the linearly interpolated empirical percentile of non-empty values."""
    if not values or not 0 <= proportion <= 1:
        raise ValueError("Percentile requires values and a proportion in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def pair_seed(base_seed, *labels):
    """Derive a stable independent bootstrap seed for one analysis stratum."""
    value = ":".join([str(base_seed), *labels]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def run_bootstrap(root, config):
    """Bootstrap every off-diagonal transfer cell over whole target groups."""
    scores = read_csv(root / config["transfer_scores_output"], TRANSFER_SCORE_COLUMNS)
    by_pair = defaultdict(list)
    for row in scores:
        if row["source_clan"] == row["target_clan"]:
            continue
        by_pair[(row["source_dataset"], row["source_clan"], row["target_dataset"], row["target_clan"])].append({
            "group_id": row["group_id"],
            "n_symbols": int(row["token_count"]),
            "own_bits_per_coda": float(row["own_bits_per_coda"]),
            "transfer_bits_per_coda": float(row["transfer_bits_per_coda"]),
            "transfer_loss_bits_per_coda": float(row["transfer_loss_bits_per_coda"]),
            "transfer_loss_bits_per_symbol": float(row["transfer_loss_bits_per_symbol"]),
        })
    if len(by_pair) != 72:
        raise ValueError("Bootstrap requires all 72 off-diagonal transfer cells")
    draw_rows, summary_rows = [], []
    for pair, loss_rows in sorted(by_pair.items()):
        samples = bootstrap_group_mean(loss_rows, config["bootstrap_draws"], pair_seed(config["bootstrap_seed"], *pair))
        point = aggregate_transfer_loss(loss_rows)
        for draw, sample in enumerate(samples):
            draw_rows.append({
                "source_dataset": pair[0], "source_clan": pair[1],
                "target_dataset": pair[2], "target_clan": pair[3], "draw": draw,
                **sample,
            })
        summary_rows.append({
            "source_dataset": pair[0], "source_clan": pair[1],
            "target_dataset": pair[2], "target_clan": pair[3],
            "target_group_count": len({row["group_id"] for row in loss_rows}),
            "bootstrap_draws": len(samples),
            "mean_transfer_loss_bits_per_coda": point["mean_transfer_loss_bits_per_coda"],
            "ci95_lower_bits_per_coda": percentile([row["mean_transfer_loss_bits_per_coda"] for row in samples], 0.025),
            "ci95_upper_bits_per_coda": percentile([row["mean_transfer_loss_bits_per_coda"] for row in samples], 0.975),
            "mean_transfer_loss_bits_per_symbol": point["mean_transfer_loss_bits_per_symbol"],
            "ci95_lower_bits_per_symbol": percentile([row["mean_transfer_loss_bits_per_symbol"] for row in samples], 0.025),
            "ci95_upper_bits_per_symbol": percentile([row["mean_transfer_loss_bits_per_symbol"] for row in samples], 0.975),
        })
    write_csv(root / config["transfer_bootstrap_output"], BOOTSTRAP_COLUMNS, draw_rows)
    write_csv(root / config["transfer_uncertainty_summary_output"], UNCERTAINTY_SUMMARY_COLUMNS, summary_rows)
    return draw_rows, summary_rows
