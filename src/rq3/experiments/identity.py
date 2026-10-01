"""Pacific target-coda identity/non-identity transfer-loss breakdown."""
from collections import defaultdict

from .data import (
    IDENTITY_BOOTSTRAP_COLUMNS,
    IDENTITY_SCORE_COLUMNS,
    IDENTITY_SUMMARY_COLUMNS,
    TRANSFER_SCORE_COLUMNS,
    read_csv,
    write_csv,
)
from .evaluation import aggregate_transfer_loss, bootstrap_group_mean
from .uncertainty import pair_seed, percentile


def run_identity_breakdown(root, config):
    """Stratify existing Pacific-target paired scores; models are never retrained."""
    scores = read_csv(root / config["transfer_scores_output"], TRANSFER_SCORE_COLUMNS)
    metadata_rows = read_csv(root / config["metadata_input"], None)
    metadata = {row["rq3_coda_id"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError("RQ3 metadata coda IDs are not unique")
    score_rows, by_stratum = [], defaultdict(list)
    for row in scores:
        if row["source_clan"] == row["target_clan"] or row["target_dataset"] != "pacific":
            continue
        metadata_row = metadata.get(row["rq3_coda_id"])
        if metadata_row is None or metadata_row["source_dataset"] != "pacific":
            raise ValueError(f"Missing Pacific identity metadata for {row['rq3_coda_id']}")
        status = metadata_row["identity_status"]
        if status == "unmapped":
            continue
        if status not in {"identity", "non_identity"}:
            raise ValueError(f"Unexpected Pacific identity status: {status}")
        result = {
            "source_dataset": row["source_dataset"], "source_clan": row["source_clan"],
            "target_dataset": row["target_dataset"], "target_clan": row["target_clan"],
            "rq3_coda_id": row["rq3_coda_id"], "group_id": row["group_id"],
            "target_fold": row["target_fold"], "identity_status": status,
            "token_count": row["token_count"], "own_bits_per_coda": row["own_bits_per_coda"],
            "own_bits_per_symbol": row["own_bits_per_symbol"],
            "transfer_bits_per_coda": row["transfer_bits_per_coda"],
            "transfer_bits_per_symbol": row["transfer_bits_per_symbol"],
            "transfer_loss_bits_per_coda": row["transfer_loss_bits_per_coda"],
            "transfer_loss_bits_per_symbol": row["transfer_loss_bits_per_symbol"],
        }
        score_rows.append(result)
        by_stratum[(result["source_dataset"], result["source_clan"], result["target_dataset"], result["target_clan"], status)].append({
            "group_id": result["group_id"],
            "transfer_loss_bits_per_coda": float(result["transfer_loss_bits_per_coda"]),
            "transfer_loss_bits_per_symbol": float(result["transfer_loss_bits_per_symbol"]),
        })
    directions = {(key[0], key[1], key[2], key[3]) for key in by_stratum}
    if len(directions) != 56:
        raise ValueError("Identity breakdown requires 56 off-diagonal Pacific-target directions")
    draw_rows, summaries = [], []
    for key, loss_rows in sorted(by_stratum.items()):
        samples = bootstrap_group_mean(loss_rows, config["bootstrap_draws"], pair_seed(config["bootstrap_seed"], *key))
        point = aggregate_transfer_loss(loss_rows)
        for draw, sample in enumerate(samples):
            draw_rows.append({
                "source_dataset": key[0], "source_clan": key[1], "target_dataset": key[2],
                "target_clan": key[3], "identity_status": key[4], "draw": draw, **sample,
            })
        summaries.append({
            "source_dataset": key[0], "source_clan": key[1], "target_dataset": key[2],
            "target_clan": key[3], "identity_status": key[4], "target_coda_count": len(loss_rows),
            "target_group_count": len({row["group_id"] for row in loss_rows}), "bootstrap_draws": len(samples),
            "mean_transfer_loss_bits_per_coda": point["mean_transfer_loss_bits_per_coda"],
            "ci95_lower_bits_per_coda": percentile([row["mean_transfer_loss_bits_per_coda"] for row in samples], 0.025),
            "ci95_upper_bits_per_coda": percentile([row["mean_transfer_loss_bits_per_coda"] for row in samples], 0.975),
            "mean_transfer_loss_bits_per_symbol": point["mean_transfer_loss_bits_per_symbol"],
            "ci95_lower_bits_per_symbol": percentile([row["mean_transfer_loss_bits_per_symbol"] for row in samples], 0.025),
            "ci95_upper_bits_per_symbol": percentile([row["mean_transfer_loss_bits_per_symbol"] for row in samples], 0.975),
        })
    score_rows.sort(key=lambda row: (row["source_dataset"], row["source_clan"], row["target_dataset"], row["target_clan"], row["identity_status"], row["rq3_coda_id"]))
    write_csv(root / config["transfer_identity_scores_output"], IDENTITY_SCORE_COLUMNS, score_rows)
    write_csv(root / config["transfer_identity_bootstrap_output"], IDENTITY_BOOTSTRAP_COLUMNS, draw_rows)
    write_csv(root / config["transfer_identity_summary_output"], IDENTITY_SUMMARY_COLUMNS, summaries)
    return score_rows, draw_rows, summaries
