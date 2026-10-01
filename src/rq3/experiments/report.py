"""Render the generated report for RQ3 self-model calibration."""
from collections import defaultdict

from whale_coda_dynamics.preprocessing.io_utils import write_text


def build_self_calibration_report(calibration_rows, self_score_rows):
    """Render selected model settings and held-out calibration scores by clan."""
    selected = [row for row in calibration_rows if row["selected"] == "true"]
    lines = [
        "# RQ3 self-model calibration",
        "",
        "Each coda is scored by its own clan's model excluding that coda's deterministic recording-group fold.",
        "Model settings are selected using the same clan's held-out folds only.",
        "",
        "## Selected models",
        "",
        "| Dataset | Clan | Model | Context order | Alpha | Held-out codas | Mean bits/coda | Mean bits/symbol |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['source_dataset']} | {row['clan']} | {row['model_family']} | "
            f"{row['context_order']} | {row['alpha']} | {row['validation_coda_count']} | "
            f"{float(row['mean_bits_per_coda']):.6f} | {float(row['mean_bits_per_symbol']):.6f} |"
        )
    by_clan = defaultdict(int)
    for row in self_score_rows:
        by_clan[(row["source_dataset"], row["clan"])] += 1
    lines += ["", "## Score coverage", "", "| Dataset | Clan | Cross-fitted codas |", "|---|---|---:|"]
    for (dataset, clan), count in sorted(by_clan.items()):
        lines.append(f"| {dataset} | {clan} | {count} |")
    lines += ["", "The complete candidate-fold table is in `data/results/rq3/calibration/self_calibration.csv`; coda-level own scores are in `data/results/rq3/calibration/self_scores.csv`.", ""]
    return "\n".join(lines)


def write_self_calibration_report(path, calibration_rows, self_score_rows):
    """Write the generated calibration report."""
    write_text(path, build_self_calibration_report(calibration_rows, self_score_rows))


def build_initial_transfer_report(summary_rows):
    """Render the generated EC1/EC2 initial-transfer summary."""
    lines = [
        "# RQ3 initial Caribbean transfer",
        "",
        "Each target coda has a cross-fitted own-clan score and an averaged transfer score from every source-fold model.",
        "The paired transfer loss is own-clan score minus transfer score; positive values mean the source model predicts the target worse.",
        "",
        "| Source | Target | Target codas | Source models | Own bits/symbol | Transfer bits/symbol | Loss bits/symbol |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['source_clan']} | {row['target_clan']} | {row['target_coda_count']} | "
            f"{row['source_model_count']} | {float(row['mean_own_bits_per_symbol']):.6f} | "
            f"{float(row['mean_transfer_bits_per_symbol']):.6f} | "
            f"{float(row['mean_transfer_loss_bits_per_symbol']):.6f} |"
        )
    lines += [
        "",
        "These are initial EC1/EC2 results only. They do not yet include bootstrap intervals, coverage strata, or the full 9 x 9 matrix.",
        "",
    ]
    return "\n".join(lines)


def write_initial_transfer_report(path, summary_rows):
    """Write the generated EC1/EC2 initial-transfer report."""
    write_text(path, build_initial_transfer_report(summary_rows))


def build_transfer_report(summary_rows, coverage_rows, uncertainty_rows, identity_rows=None, overlap_pairs=None, overlap_results=None):
    """Render the primary 9 x 9 transfer-loss matrix."""
    clans = sorted({row["source_clan"] for row in summary_rows})
    by_pair = {(row["source_clan"], row["target_clan"]): row for row in summary_rows}
    uncertainty_by_pair = {(row["source_clan"], row["target_clan"]): row for row in uncertainty_rows}
    if len(by_pair) != len(clans) ** 2:
        raise ValueError("Transfer report requires one summary row for every directed clan pair")
    lines = [
        "# RQ3 cross-clan transfer matrix",
        "",
        "Each off-diagonal cell is the equal-coda average paired loss in bits per symbol: the target clan's cross-fitted own score minus the source clan's averaged source-fold transfer score.",
        "Positive values mean that the source model predicts the target codas worse. Diagonal cells are zero by definition.",
        "",
        "## Transfer loss matrix (bits per symbol)",
        "",
        "| Source \\ Target | " + " | ".join(clans) + " |",
        "|---|" + "|".join("---:" for _ in clans) + "|",
    ]
    for source in clans:
        values = [f"{float(by_pair[(source, target)]['mean_transfer_loss_bits_per_symbol']):.6f}" for target in clans]
        lines.append(f"| {source} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Directed cross-clan summary",
        "",
        "| Source | Target | Target codas | Source models | Own bits/symbol | Transfer bits/symbol | Loss bits/symbol | 95% group-bootstrap interval |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        interval = "—" if row["source_clan"] == row["target_clan"] or not uncertainty_by_pair else (
            f"[{float(uncertainty_by_pair[(row['source_clan'], row['target_clan'])]['ci95_lower_bits_per_symbol']):.6f}, "
            f"{float(uncertainty_by_pair[(row['source_clan'], row['target_clan'])]['ci95_upper_bits_per_symbol']):.6f}]"
        )
        lines.append(
            f"| {row['source_clan']} | {row['target_clan']} | {row['target_coda_count']} | "
            f"{row['source_model_count']} | {float(row['mean_own_bits_per_symbol']):.6f} | "
            f"{float(row['mean_transfer_bits_per_symbol']):.6f} | "
            f"{float(row['mean_transfer_loss_bits_per_symbol']):.6f} | {interval} |"
        )
    lines += [
        "",
        "## Coverage diagnostic",
        "",
        "For every source-fold model and target coda, `covered` means the target's complete fixed-token sequence occurred at least five times in that model's actual source training codas. Each row below is therefore a source-fold/target-coda-weighted stratum, matching the averaging rule used for transfer scores.",
        "",
        "| Source | Target | Status | Score rows | Share | Loss bits/symbol |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in coverage_rows:
        loss = "" if row["mean_transfer_loss_bits_per_symbol"] == "" else f"{float(row['mean_transfer_loss_bits_per_symbol']):.6f}"
        lines.append(
            f"| {row['source_clan']} | {row['target_clan']} | {row['coverage_status']} | "
            f"{row['source_fold_target_score_count']} | {float(row['source_fold_target_score_share']):.6f} | {loss} |"
        )
    if identity_rows:
        lines += [
            "",
            "## Pacific identity-status breakdown",
            "",
            "Pacific target codas are labelled from Hersh et al.'s Table S2 mapping. Models were trained on all eligible codas; this is a post-scoring split of paired results. Caribbean target codas are unmapped and excluded.",
            "",
            "| Source | Pacific target | Status | Codas | Loss bits/symbol | 95% group-bootstrap interval |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in identity_rows:
            lines.append(
                f"| {row['source_clan']} | {row['target_clan']} | {row['identity_status']} | {row['target_coda_count']} | "
                f"{float(row['mean_transfer_loss_bits_per_symbol']):.6f} | "
                f"[{float(row['ci95_lower_bits_per_symbol']):.6f}, {float(row['ci95_upper_bits_per_symbol']):.6f}] |"
            )
    if overlap_results:
        lines += [
            "",
            "## Exploratory Pacific spatial-overlap analysis",
            "",
            "For each of the 21 unordered Pacific pairs, spatial overlap is the mean of the two Table S4 directions and transfer loss is the mean of the two directed identity-stratified losses. The association is Pearson's r. The exact one-sided permutation p-value tests the predeclared prediction that higher overlap corresponds to lower loss, using all 7! = 5,040 clan-label permutations.",
            "",
            "| Status | Pairs | Pearson r | One-sided exact p | Two-sided exact p |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in overlap_results:
            lines.append(
                f"| {row['identity_status']} | {row['pair_count']} | {float(row['association_statistic']):.6f} | "
                f"{float(row['exact_one_sided_p_value']):.6f} | {float(row['exact_two_sided_p_value']):.6f} |"
            )
        lines += [
            "",
            "| Status | Pair | Mean overlap | Mean bidirectional loss (bits/symbol) |",
            "|---|---|---:|---:|",
        ]
        for row in overlap_pairs:
            lines.append(
                f"| {row['identity_status']} | {row['clan_a']}–{row['clan_b']} | {float(row['mean_spatial_overlap']):.6f} | "
                f"{float(row['mean_bidirectional_transfer_loss_bits_per_symbol']):.6f} |"
            )
    lines += [
        "",
        "Intervals use 1,000 resamples of whole target recording groups (Caribbean days or Pacific repertoires) after scoring; they do not retrain models. The Pacific-overlap result is exploratory because pairs share clans and Table S4 has no temporal co-occurrence component.",
        "",
    ]
    return "\n".join(lines)


def write_transfer_report(path, summary_rows, coverage_rows, uncertainty_rows, identity_rows=None, overlap_pairs=None, overlap_results=None):
    """Write the generated primary transfer-matrix report."""
    write_text(path, build_transfer_report(summary_rows, coverage_rows, uncertainty_rows, identity_rows, overlap_pairs, overlap_results))
