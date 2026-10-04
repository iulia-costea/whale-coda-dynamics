"""Shared RQ3 score aggregation, paired transfer loss, and uncertainty helpers."""
import random


def mean_bits_per_coda(score_rows):
    """Mean of total coda log probabilities, giving each coda equal weight."""
    if not score_rows:
        raise ValueError("Cannot aggregate an empty score set")
    return sum(row["bits_per_coda"] for row in score_rows) / len(score_rows)


def mean_bits_per_symbol(score_rows):
    """Aggregate log likelihood over the total number of predicted symbols."""
    if not score_rows:
        raise ValueError("Cannot aggregate an empty score set")
    total_symbols = sum(row["token_count"] for row in score_rows)
    if total_symbols <= 0:
        raise ValueError("Score rows must contain positive token counts")
    return sum(row["bits_per_coda"] for row in score_rows) / total_symbols


def paired_transfer_loss(own_score_rows, transfer_score_rows):
    """Pair scores by coda ID and return per-coda own-minus-transfer losses."""
    own = {row["rq3_coda_id"]: row for row in own_score_rows}
    transfer = {row["rq3_coda_id"]: row for row in transfer_score_rows}
    if set(own) != set(transfer):
        raise ValueError("Own and transfer score sets must contain identical target codas")
    return [
        {
            "rq3_coda_id": coda_id,
            "n_symbols": own[coda_id]["token_count"],
            "own_bits_per_coda": own[coda_id]["bits_per_coda"],
            "transfer_bits_per_coda": transfer[coda_id]["bits_per_coda"],
            "transfer_loss_bits_per_coda": own[coda_id]["bits_per_coda"] - transfer[coda_id]["bits_per_coda"],
            "transfer_loss_bits_per_symbol": own[coda_id]["bits_per_symbol"] - transfer[coda_id]["bits_per_symbol"],
        }
        for coda_id in own
    ]


def aggregate_transfer_loss(loss_rows):
    """Return coda-mean and symbol-weighted paired transfer losses."""
    if not loss_rows:
        raise ValueError("Cannot aggregate an empty transfer-loss set")
    total_symbols = sum(row["n_symbols"] for row in loss_rows)
    if total_symbols <= 0:
        raise ValueError("Transfer-loss rows must contain positive symbol counts")
    return {
        "mean_transfer_loss_bits_per_coda": sum(row["transfer_loss_bits_per_coda"] for row in loss_rows) / len(loss_rows),
        "mean_transfer_loss_bits_per_symbol": (
            sum(row["own_bits_per_coda"] - row["transfer_bits_per_coda"] for row in loss_rows) / total_symbols
        ),
    }


def bootstrap_group_mean(loss_rows, draws, seed):
    """Return bootstrap group means for post-scoring uncertainty intervals."""
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    by_group = {}
    for row in loss_rows:
        summary = by_group.setdefault(row["group_id"], {
            "count": 0, "coda_sum": 0.0, "symbol_numerator": 0.0, "symbol_count": 0,
        })
        summary["count"] += 1
        summary["coda_sum"] += row["transfer_loss_bits_per_coda"]
        summary["symbol_numerator"] += row["own_bits_per_coda"] - row["transfer_bits_per_coda"]
        summary["symbol_count"] += row["n_symbols"]
    groups = sorted(by_group)
    if not groups:
        raise ValueError("Cannot bootstrap an empty loss set")
    generator = random.Random(seed)
    samples = []
    for _ in range(draws):
        sampled = [by_group[generator.choice(groups)] for _ in groups]
        count = sum(row["count"] for row in sampled)
        symbol_count = sum(row["symbol_count"] for row in sampled)
        samples.append({
            "mean_transfer_loss_bits_per_coda": sum(row["coda_sum"] for row in sampled) / count,
            "mean_transfer_loss_bits_per_symbol": sum(row["symbol_numerator"] for row in sampled) / symbol_count,
        })
    return samples
