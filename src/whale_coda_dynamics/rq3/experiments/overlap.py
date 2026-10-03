"""Exploratory Pacific spatial-overlap association analysis for RQ3."""
from itertools import combinations, permutations
import math

from whale_coda_dynamics.preprocessing.io_utils import read_json

from .data import (
    IDENTITY_SUMMARY_COLUMNS,
    OVERLAP_PAIR_COLUMNS,
    OVERLAP_SUMMARY_COLUMNS,
    read_csv,
    write_csv,
)


def pearson_correlation(x_values, y_values):
    """Return the ordinary Pearson association for equal-length non-constant vectors."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Pearson correlation requires equally sized vectors of length at least two")
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    denominator = math.sqrt(sum(value * value for value in x_centered) * sum(value * value for value in y_centered))
    if denominator == 0:
        raise ValueError("Pearson correlation is undefined for a constant vector")
    return sum(x * y for x, y in zip(x_centered, y_centered)) / denominator


def canonical_pair(clan_a, clan_b):
    """Use lexical pair keys so bidirectional values have one lookup form."""
    return tuple(sorted((clan_a, clan_b)))


def exact_label_permutation_test(clans, overlap_by_pair, loss_by_pair, observed):
    """Permute clan labels exactly for the predeclared negative-overlap hypothesis."""
    pairs = list(combinations(clans, 2))
    overlap = [overlap_by_pair[canonical_pair(*pair)] for pair in pairs]
    statistics = []
    for relabelled in permutations(clans):
        mapping = dict(zip(clans, relabelled))
        relabelled_loss = [loss_by_pair[canonical_pair(mapping[a], mapping[b])] for a, b in pairs]
        statistics.append(pearson_correlation(overlap, relabelled_loss))
    if len(statistics) != math.factorial(len(clans)):
        raise ValueError("Exact Pacific label permutation enumeration is incomplete")
    one_sided = sum(value <= observed for value in statistics) / len(statistics)
    two_sided = sum(abs(value) >= abs(observed) for value in statistics) / len(statistics)
    return one_sided, two_sided, len(statistics)


def run_pacific_overlap(root, config):
    """Associate mean pair overlap with bidirectional loss for each identity stratum."""
    overlap_config = read_json(root / config["spatial_overlap_input"])
    clans = overlap_config["clans"]
    directed_overlap = overlap_config["directed_overlap"]
    if len(clans) != 7 or len(set(clans)) != 7:
        raise ValueError("Pacific overlap analysis requires exactly seven distinct clans")
    summaries = read_csv(root / config["transfer_identity_summary_output"], IDENTITY_SUMMARY_COLUMNS)
    directional = {}
    for row in summaries:
        if row["source_dataset"] != "pacific" or row["target_dataset"] != "pacific":
            continue
        key = (row["identity_status"], row["source_clan"], row["target_clan"])
        if key in directional:
            raise ValueError(f"Duplicate Pacific identity summary: {key}")
        directional[key] = float(row["mean_transfer_loss_bits_per_symbol"])
    pair_rows, result_rows = [], []
    for status in ("identity", "non_identity"):
        expected_directions = {(status, source, target) for source in clans for target in clans if source != target}
        if set(key for key in directional if key[0] == status) != expected_directions:
            raise ValueError(f"Missing Pacific {status} transfer direction")
        overlap_by_pair, loss_by_pair = {}, {}
        for clan_a, clan_b in combinations(clans, 2):
            key = canonical_pair(clan_a, clan_b)
            mean_overlap = (float(directed_overlap[clan_a][clan_b]) + float(directed_overlap[clan_b][clan_a])) / 2
            a_to_b = directional[(status, clan_a, clan_b)]
            b_to_a = directional[(status, clan_b, clan_a)]
            mean_loss = (a_to_b + b_to_a) / 2
            overlap_by_pair[key], loss_by_pair[key] = mean_overlap, mean_loss
            pair_rows.append({
                "identity_status": status, "clan_a": clan_a, "clan_b": clan_b,
                "mean_spatial_overlap": mean_overlap,
                "transfer_loss_a_to_b_bits_per_symbol": a_to_b,
                "transfer_loss_b_to_a_bits_per_symbol": b_to_a,
                "mean_bidirectional_transfer_loss_bits_per_symbol": mean_loss,
            })
        pairs = list(combinations(clans, 2))
        observed = pearson_correlation(
            [overlap_by_pair[canonical_pair(*pair)] for pair in pairs],
            [loss_by_pair[canonical_pair(*pair)] for pair in pairs],
        )
        one_sided, two_sided, permutation_count = exact_label_permutation_test(clans, overlap_by_pair, loss_by_pair, observed)
        result_rows.append({
            "identity_status": status, "pair_count": len(pairs), "association_statistic": observed,
            "alternative": "higher_overlap_associated_with_lower_transfer_loss",
            "exact_one_sided_p_value": one_sided, "exact_two_sided_p_value": two_sided,
            "permutation_count": permutation_count,
        })
    write_csv(root / config["pacific_overlap_pairs_output"], OVERLAP_PAIR_COLUMNS, pair_rows)
    write_csv(root / config["pacific_overlap_summary_output"], OVERLAP_SUMMARY_COLUMNS, result_rows)
    return pair_rows, result_rows
