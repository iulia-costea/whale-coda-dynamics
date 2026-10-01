"""Shared RQ3 score aggregation, paired transfer loss, and uncertainty helpers."""
import random


def mean_bits_per_coda(score_rows):
    """Equal-coda mean of total coda log probabilities."""
    if not score_rows:
        raise ValueError("Cannot aggregate an empty score set")
    return sum(row["bits_per_coda"] for row in score_rows) / len(score_rows)


def mean_bits_per_symbol(score_rows):
    """Equal-coda mean of each coda's length-normalised score."""
    if not score_rows:
        raise ValueError("Cannot aggregate an empty score set")
    return sum(row["bits_per_symbol"] for row in score_rows) / len(score_rows)


def paired_transfer_loss(own_score_rows, transfer_score_rows):
    """Pair scores by coda ID and return per-coda own-minus-transfer losses."""
    own = {row["rq3_coda_id"]: row for row in own_score_rows}
    transfer = {row["rq3_coda_id"]: row for row in transfer_score_rows}
    if set(own) != set(transfer):
        raise ValueError("Own and transfer score sets must contain identical target codas")
    return [
        {
            "rq3_coda_id": coda_id,
            "transfer_loss_bits_per_coda": own[coda_id]["bits_per_coda"] - transfer[coda_id]["bits_per_coda"],
            "transfer_loss_bits_per_symbol": own[coda_id]["bits_per_symbol"] - transfer[coda_id]["bits_per_symbol"],
        }
        for coda_id in own
    ]


def aggregate_transfer_loss(loss_rows):
    """Return the equal-coda mean paired transfer losses for one clan pair."""
    if not loss_rows:
        raise ValueError("Cannot aggregate an empty transfer-loss set")
    return {
        "mean_transfer_loss_bits_per_coda": sum(row["transfer_loss_bits_per_coda"] for row in loss_rows) / len(loss_rows),
        "mean_transfer_loss_bits_per_symbol": sum(row["transfer_loss_bits_per_symbol"] for row in loss_rows) / len(loss_rows),
    }


def bootstrap_group_mean(loss_rows, draws, seed):
    """Return bootstrap group means for post-scoring uncertainty intervals."""
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    by_group = {}
    for row in loss_rows:
        by_group.setdefault(row["group_id"], []).append(row)
    groups = sorted(by_group)
    if not groups:
        raise ValueError("Cannot bootstrap an empty loss set")
    generator = random.Random(seed)
    samples = []
    for _ in range(draws):
        sampled_rows = [row for _ in groups for row in by_group[generator.choice(groups)]]
        samples.append(aggregate_transfer_loss(sampled_rows))
    return samples
