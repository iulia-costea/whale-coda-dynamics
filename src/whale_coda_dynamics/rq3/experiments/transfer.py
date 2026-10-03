"""Score RQ3 transfer and its fold-specific timing-pattern coverage diagnostic."""
from collections import Counter, defaultdict

from .data import (
    CALIBRATION_COLUMNS,
    COVERAGE_SUMMARY_COLUMNS,
    SELF_SCORE_COLUMNS,
    TRANSFER_FOLD_SCORE_COLUMNS,
    TRANSFER_SCORE_COLUMNS,
    TRANSFER_SUMMARY_COLUMNS,
    read_csv,
    write_csv,
)
from .evaluation import aggregate_transfer_loss, mean_bits_per_coda, mean_bits_per_symbol, paired_transfer_loss
from .models import SmoothedNGram
from .scoring import score_coda_bits


def selected_candidates(root, config):
    """Load exactly one source-only selected setting for each calibrated clan."""
    rows = read_csv(root / config["calibration_output"], CALIBRATION_COLUMNS)
    selected = [row for row in rows if row["selected"] == "true"]
    candidates = {}
    for row in selected:
        key = (row["source_dataset"], row["clan"])
        if key in candidates:
            raise ValueError(f"Duplicate selected candidate for {key}")
        candidates[key] = {
            "family": row["model_family"],
            "context_order": int(row["context_order"]),
            "alpha": float(row["alpha"]),
        }
    if len(candidates) != 9:
        raise ValueError("Calibration output must contain one selected candidate for all nine clans")
    return candidates


def own_scores(root, config, codas):
    """Load and validate the precomputed cross-fitted own-clan reference scores."""
    rows = read_csv(root / config["self_scores_output"], SELF_SCORE_COLUMNS)
    expected_ids = {coda["rq3_coda_id"] for coda in codas}
    by_id = {row["rq3_coda_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != expected_ids:
        raise ValueError("Self-score output must contain exactly one row for every prepared coda")
    for coda in codas:
        row = by_id[coda["rq3_coda_id"]]
        if (row["source_dataset"], row["clan"], row["group_id"], int(row["fold"])) != (
                coda["source_dataset"], coda["clan"], coda["group_id"], coda["fold"]):
            raise ValueError(f"Self-score metadata mismatch for {coda['rq3_coda_id']}")
    return by_id


def score_pair(source_codas, target_codas, source_key, target_key, candidate, own_by_id):
    """Fit source-fold models, score every target coda, then pair averaged scores."""
    source_folds = sorted({coda["fold"] for coda in source_codas})
    fold_rows = []
    scores_by_target = defaultdict(list)
    for source_fold in source_folds:
        training = [coda for coda in source_codas if coda["fold"] != source_fold]
        sequence_counts = Counter(tuple(coda["tokens"]) for coda in training)
        model = SmoothedNGram(candidate["context_order"], candidate["alpha"])
        model.fit(coda["tokens"] for coda in training)
        for target in target_codas:
            score = score_coda_bits(model, target["tokens"])
            own = own_by_id[target["rq3_coda_id"]]
            sequence_occurrences = sequence_counts[tuple(target["tokens"])]
            row = {
                "source_dataset": source_key[0], "source_clan": source_key[1],
                "source_fold": source_fold, "source_training_coda_count": len(training),
                "target_dataset": target_key[0], "target_clan": target_key[1],
                "rq3_coda_id": target["rq3_coda_id"], "group_id": target["group_id"],
                "target_fold": target["fold"], "model_family": candidate["family"],
                "context_order": candidate["context_order"], "alpha": candidate["alpha"], **score,
                "sequence_occurrences": sequence_occurrences,
                "coverage_status": "covered" if sequence_occurrences >= 5 else "uncovered",
                "own_bits_per_coda": float(own["bits_per_coda"]),
                "own_bits_per_symbol": float(own["bits_per_symbol"]),
                "transfer_loss_bits_per_coda": float(own["bits_per_coda"]) - score["bits_per_coda"],
                "transfer_loss_bits_per_symbol": float(own["bits_per_symbol"]) - score["bits_per_symbol"],
            }
            fold_rows.append(row)
            scores_by_target[target["rq3_coda_id"]].append(row)
    aggregate_rows = []
    for target in target_codas:
        per_fold = scores_by_target[target["rq3_coda_id"]]
        if len(per_fold) != len(source_folds):
            raise ValueError(f"Incomplete source-fold scores for {target['rq3_coda_id']}")
        own = own_by_id[target["rq3_coda_id"]]
        if int(own["token_count"]) != per_fold[0]["token_count"]:
            raise ValueError(f"Token-count mismatch for {target['rq3_coda_id']}")
        transfer_coda = mean_bits_per_coda(per_fold)
        transfer_symbol = mean_bits_per_symbol(per_fold)
        aggregate_rows.append({
            "source_dataset": source_key[0], "source_clan": source_key[1],
            "target_dataset": target_key[0], "target_clan": target_key[1],
            "rq3_coda_id": target["rq3_coda_id"], "group_id": target["group_id"],
            "target_fold": target["fold"], "source_model_count": len(source_folds),
            "token_count": per_fold[0]["token_count"],
            "own_bits_per_coda": float(own["bits_per_coda"]),
            "own_bits_per_symbol": float(own["bits_per_symbol"]),
            "transfer_bits_per_coda": transfer_coda,
            "transfer_bits_per_symbol": transfer_symbol,
            "transfer_loss_bits_per_coda": float(own["bits_per_coda"]) - transfer_coda,
            "transfer_loss_bits_per_symbol": float(own["bits_per_symbol"]) - transfer_symbol,
        })
    paired = paired_transfer_loss(
        [{"rq3_coda_id": row["rq3_coda_id"], "bits_per_coda": row["own_bits_per_coda"], "bits_per_symbol": row["own_bits_per_symbol"]} for row in aggregate_rows],
        [{"rq3_coda_id": row["rq3_coda_id"], "bits_per_coda": row["transfer_bits_per_coda"], "bits_per_symbol": row["transfer_bits_per_symbol"]} for row in aggregate_rows],
    )
    losses = aggregate_transfer_loss(paired)
    summary = {
        "source_dataset": source_key[0], "source_clan": source_key[1],
        "target_dataset": target_key[0], "target_clan": target_key[1],
        "target_coda_count": len(aggregate_rows), "source_model_count": len(source_folds),
        "mean_own_bits_per_coda": sum(row["own_bits_per_coda"] for row in aggregate_rows) / len(aggregate_rows),
        "mean_own_bits_per_symbol": sum(row["own_bits_per_symbol"] for row in aggregate_rows) / len(aggregate_rows),
        "mean_transfer_bits_per_coda": sum(row["transfer_bits_per_coda"] for row in aggregate_rows) / len(aggregate_rows),
        "mean_transfer_bits_per_symbol": sum(row["transfer_bits_per_symbol"] for row in aggregate_rows) / len(aggregate_rows),
        **losses,
    }
    return fold_rows, aggregate_rows, summary


def run_initial_caribbean_transfer(root, config, codas):
    """Write all audit and aggregate outputs for EC1 -> EC2 and EC2 -> EC1."""
    candidates = selected_candidates(root, config)
    own_by_id = own_scores(root, config, codas)
    by_clan = defaultdict(list)
    for coda in codas:
        by_clan[(coda["source_dataset"], coda["clan"])].append(coda)
    fold_rows, aggregate_rows, summaries = [], [], []
    for source_clan, target_clan in config["initial_transfer_pairs"]:
        source_key, target_key = ("caribbean", source_clan), ("caribbean", target_clan)
        pair_fold_rows, pair_aggregate_rows, summary = score_pair(
            by_clan[source_key], by_clan[target_key], source_key, target_key,
            candidates[source_key], own_by_id,
        )
        fold_rows.extend(pair_fold_rows)
        aggregate_rows.extend(pair_aggregate_rows)
        summaries.append(summary)
    fold_rows.sort(key=lambda row: (row["source_clan"], row["source_fold"], row["rq3_coda_id"]))
    aggregate_rows.sort(key=lambda row: (row["source_clan"], row["rq3_coda_id"]))
    summaries.sort(key=lambda row: (row["source_clan"], row["target_clan"]))
    write_csv(root / config["initial_transfer_fold_scores_output"], TRANSFER_FOLD_SCORE_COLUMNS, fold_rows)
    write_csv(root / config["initial_transfer_scores_output"], TRANSFER_SCORE_COLUMNS, aggregate_rows)
    write_csv(root / config["initial_transfer_summary_output"], TRANSFER_SUMMARY_COLUMNS, summaries)
    return fold_rows, aggregate_rows, summaries


def diagonal_rows(clan_key, codas, own_by_id):
    """Represent each existing own-clan score as a zero-loss matrix diagonal row."""
    rows = []
    for coda in codas:
        own = own_by_id[coda["rq3_coda_id"]]
        own_coda, own_symbol = float(own["bits_per_coda"]), float(own["bits_per_symbol"])
        rows.append({
            "source_dataset": clan_key[0], "source_clan": clan_key[1],
            "target_dataset": clan_key[0], "target_clan": clan_key[1],
            "rq3_coda_id": coda["rq3_coda_id"], "group_id": coda["group_id"],
            "target_fold": coda["fold"], "source_model_count": 1,
            "token_count": int(own["token_count"]), "own_bits_per_coda": own_coda,
            "own_bits_per_symbol": own_symbol, "transfer_bits_per_coda": own_coda,
            "transfer_bits_per_symbol": own_symbol, "transfer_loss_bits_per_coda": 0.0,
            "transfer_loss_bits_per_symbol": 0.0,
        })
    summary = {
        "source_dataset": clan_key[0], "source_clan": clan_key[1],
        "target_dataset": clan_key[0], "target_clan": clan_key[1],
        "target_coda_count": len(rows), "source_model_count": 1,
        "mean_own_bits_per_coda": sum(row["own_bits_per_coda"] for row in rows) / len(rows),
        "mean_own_bits_per_symbol": sum(row["own_bits_per_symbol"] for row in rows) / len(rows),
        "mean_transfer_bits_per_coda": sum(row["transfer_bits_per_coda"] for row in rows) / len(rows),
        "mean_transfer_bits_per_symbol": sum(row["transfer_bits_per_symbol"] for row in rows) / len(rows),
        "mean_transfer_loss_bits_per_coda": 0.0,
        "mean_transfer_loss_bits_per_symbol": 0.0,
    }
    return rows, summary


def coverage_summaries(fold_rows):
    """Aggregate fold-specific coverage with the same weights as transfer scoring."""
    by_pair = defaultdict(list)
    for row in fold_rows:
        by_pair[(row["source_dataset"], row["source_clan"], row["target_dataset"], row["target_clan"])].append(row)
    summaries = []
    for pair, rows in sorted(by_pair.items()):
        for status in ("covered", "uncovered"):
            stratum = [row for row in rows if row["coverage_status"] == status]
            count = len(stratum)
            summaries.append({
                "source_dataset": pair[0], "source_clan": pair[1],
                "target_dataset": pair[2], "target_clan": pair[3], "coverage_status": status,
                "source_fold_target_score_count": count,
                "source_fold_target_score_share": count / len(rows),
                "mean_own_bits_per_coda": (sum(row["own_bits_per_coda"] for row in stratum) / count) if count else "",
                "mean_own_bits_per_symbol": (sum(row["own_bits_per_symbol"] for row in stratum) / count) if count else "",
                "mean_transfer_bits_per_coda": (sum(row["bits_per_coda"] for row in stratum) / count) if count else "",
                "mean_transfer_bits_per_symbol": (sum(row["bits_per_symbol"] for row in stratum) / count) if count else "",
                "mean_transfer_loss_bits_per_coda": (sum(row["transfer_loss_bits_per_coda"] for row in stratum) / count) if count else "",
                "mean_transfer_loss_bits_per_symbol": (sum(row["transfer_loss_bits_per_symbol"] for row in stratum) / count) if count else "",
            })
    return summaries


def run_full_transfer_matrix(root, config, codas):
    """Write the nine self cells and 72 source-fold-ensemble transfer cells."""
    candidates = selected_candidates(root, config)
    own_by_id = own_scores(root, config, codas)
    by_clan = defaultdict(list)
    for coda in codas:
        by_clan[(coda["source_dataset"], coda["clan"])].append(coda)
    keys = sorted(by_clan)
    if set(keys) != set(candidates):
        raise ValueError("Prepared clans and calibrated-clan settings do not match")
    fold_rows, aggregate_rows, summaries = [], [], []
    for source_key in keys:
        for target_key in keys:
            if source_key == target_key:
                diagonal, summary = diagonal_rows(source_key, by_clan[source_key], own_by_id)
                aggregate_rows.extend(diagonal)
                summaries.append(summary)
                continue
            pair_fold_rows, pair_aggregate_rows, summary = score_pair(
                by_clan[source_key], by_clan[target_key], source_key, target_key,
                candidates[source_key], own_by_id,
            )
            fold_rows.extend(pair_fold_rows)
            aggregate_rows.extend(pair_aggregate_rows)
            summaries.append(summary)
    fold_rows.sort(key=lambda row: (row["source_dataset"], row["source_clan"], row["target_dataset"], row["target_clan"], row["source_fold"], row["rq3_coda_id"]))
    aggregate_rows.sort(key=lambda row: (row["source_dataset"], row["source_clan"], row["target_dataset"], row["target_clan"], row["rq3_coda_id"]))
    summaries.sort(key=lambda row: (row["source_dataset"], row["source_clan"], row["target_dataset"], row["target_clan"]))
    write_csv(root / config["transfer_fold_scores_output"], TRANSFER_FOLD_SCORE_COLUMNS, fold_rows)
    write_csv(root / config["transfer_scores_output"], TRANSFER_SCORE_COLUMNS, aggregate_rows)
    coverage = coverage_summaries(fold_rows)
    write_csv(root / config["transfer_summary_output"], TRANSFER_SUMMARY_COLUMNS, summaries)
    write_csv(root / config["transfer_coverage_summary_output"], COVERAGE_SUMMARY_COLUMNS, coverage)
    return fold_rows, aggregate_rows, summaries, coverage
