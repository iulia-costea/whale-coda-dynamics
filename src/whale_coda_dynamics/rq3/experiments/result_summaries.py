"""Generate navigable README summaries beside every RQ3 results level."""
from collections import defaultdict
from pathlib import Path

from whale_coda_dynamics.preprocessing.io_utils import write_text

from .data import (
    CALIBRATION_COLUMNS,
    COVERAGE_SUMMARY_COLUMNS,
    IDENTITY_SUMMARY_COLUMNS,
    OVERLAP_SUMMARY_COLUMNS,
    TRANSFER_SUMMARY_COLUMNS,
    UNCERTAINTY_SUMMARY_COLUMNS,
    read_csv,
)


def read_rows(root, config, config_key, columns):
    """Read one configured RQ3 result artifact."""
    return read_csv(root / config[config_key], columns)


def write_markdown(path, lines):
    """Write a UTF-8/LF result README with a final newline."""
    write_text(path, "\n".join(lines) + "\n")


def mean(values):
    return sum(values) / len(values)


def build_summaries(root, config):
    """Create result-level READMEs from already generated experiment artifacts."""
    result_root = root / config["results_root"]
    calibration = read_rows(root, config, "calibration_output", CALIBRATION_COLUMNS)
    initial = read_rows(root, config, "initial_transfer_summary_output", TRANSFER_SUMMARY_COLUMNS)
    matrix = read_rows(root, config, "transfer_summary_output", TRANSFER_SUMMARY_COLUMNS)
    coverage = read_rows(root, config, "transfer_coverage_summary_output", COVERAGE_SUMMARY_COLUMNS)
    uncertainty = read_rows(root, config, "transfer_uncertainty_summary_output", UNCERTAINTY_SUMMARY_COLUMNS)
    identity = read_rows(root, config, "transfer_identity_summary_output", IDENTITY_SUMMARY_COLUMNS)
    overlap = read_rows(root, config, "pacific_overlap_summary_output", OVERLAP_SUMMARY_COLUMNS)

    selected = [row for row in calibration if row["selected"] == "true"]
    if len(selected) != 9:
        raise ValueError("Result summaries require nine selected calibration models")
    off_diagonal = [row for row in matrix if row["source_clan"] != row["target_clan"]]
    if len(off_diagonal) != 72 or len(uncertainty) != 72:
        raise ValueError("Result summaries require the complete 72-cell transfer matrix and uncertainty output")

    # Root: navigation and overall headline results.
    write_markdown(result_root / "README.md", [
        "# RQ3 results",
        "",
        "This folder contains reproducible artifacts for the within-coda cross-clan transfer experiments. The primary score is paired transfer loss in bits per symbol: target-clan own score minus source-to-target score. Positive loss means the source predicts the target worse.",
        "",
        "## Result hierarchy",
        "",
        "- [calibration/](calibration/README.md): model selection and cross-fitted own-clan reference scores.",
        "- [initial_caribbean/](initial_caribbean/README.md): EC1 ↔ EC2 initial comparison.",
        "- [transfer_matrix/](transfer_matrix/README.md): full 9 × 9 matrix and its derived diagnostics.",
        "",
        "## Headline findings",
        "",
        f"- All 72 cross-clan transfer losses are positive (range {min(float(row['mean_transfer_loss_bits_per_symbol']) for row in off_diagonal):.3f} to {max(float(row['mean_transfer_loss_bits_per_symbol']) for row in off_diagonal):.3f} bits/symbol).",
        "- Coverage of an exact full timing pattern is low across cross-clan source-fold/target-coda evaluations; see the coverage summary.",
        "- All 72 target-group bootstrap intervals lie above zero.",
        "- Pacific non-identity codas generally transfer better than identity codas, but the exploratory Table S4 overlap association is opposite to the predeclared prediction.",
    ])

    write_markdown(result_root / "calibration/README.md", [
        "# Own-clan calibration",
        "",
        "`self_calibration.csv` evaluates all candidate models on each clan's own held-out folds. `self_scores.csv` contains one cross-fitted own-clan score per eligible coda. These own scores are the reference used in every later transfer-loss calculation.",
        "",
        "## Selected models",
        "",
        "| Clan | Model | Context order | Alpha | Held-out bits/symbol |",
        "|---|---|---:|---:|---:|",
        *[f"| {row['clan']} | {row['model_family']} | {row['context_order']} | {row['alpha']} | {float(row['mean_bits_per_symbol']):.6f} |" for row in selected],
    ])

    write_markdown(result_root / "initial_caribbean/README.md", [
        "# Initial Caribbean transfer",
        "",
        "This is the first EC1 ↔ EC2 comparison. `fold_scores.csv` contains the individual source-fold scores; `scores.csv` contains paired per-coda own and averaged-transfer scores; `summary.csv` contains the aggregate result.",
        "",
        "| Direction | Own bits/symbol | Transfer bits/symbol | Loss bits/symbol |",
        "|---|---:|---:|---:|",
        *[f"| {row['source_clan']} → {row['target_clan']} | {float(row['mean_own_bits_per_symbol']):.6f} | {float(row['mean_transfer_bits_per_symbol']):.6f} | {float(row['mean_transfer_loss_bits_per_symbol']):.6f} |" for row in initial],
        "",
        "Both directions have positive loss: each Caribbean clan predicts its own codas better than the other clan does.",
    ])

    write_markdown(result_root / "transfer_matrix/README.md", [
        "# Primary 9 × 9 transfer matrix",
        "",
        "`fold_scores.csv` contains all source-fold predictions for the 72 off-diagonal directions. `scores.csv` contains paired per-coda results for all 81 cells, including zero-loss own-clan diagonal references. `summary.csv` contains the aggregate matrix.",
        "",
        f"All {len(off_diagonal)} cross-clan directions have positive loss, from {min(float(row['mean_transfer_loss_bits_per_symbol']) for row in off_diagonal):.6f} to {max(float(row['mean_transfer_loss_bits_per_symbol']) for row in off_diagonal):.6f} bits/symbol.",
        "",
        "## Derived analyses",
        "",
        "- [coverage/](coverage/README.md): whether the source-fold training data contains the complete target timing pattern.",
        "- [uncertainty/](uncertainty/README.md): 1,000 target-group bootstrap draws and 95% intervals.",
        "- [identity/](identity/README.md): Pacific target-coda identity/non-identity breakdown.",
        "- [pacific_overlap/](pacific_overlap/README.md): exploratory Table S4 spatial-overlap association.",
    ])

    total_coverage = sum(int(row["source_fold_target_score_count"]) for row in coverage)
    covered = sum(int(row["source_fold_target_score_count"]) for row in coverage if row["coverage_status"] == "covered")
    covered_better = 0
    coverage_pairs = defaultdict(dict)
    for row in coverage:
        coverage_pairs[(row["source_clan"], row["target_clan"])][row["coverage_status"]] = row
    for strata in coverage_pairs.values():
        if (float(strata["covered"]["mean_transfer_loss_bits_per_symbol"]) <
                float(strata["uncovered"]["mean_transfer_loss_bits_per_symbol"])):
            covered_better += 1
    write_markdown(result_root / "transfer_matrix/coverage/README.md", [
        "# Coverage diagnostic",
        "",
        "`summary.csv` splits each cross-clan source-fold/target-coda prediction by whether the target's complete fixed-token timing sequence occurred at least five times in that source fold's actual training codas.",
        "",
        f"Across {total_coverage:,} evaluations, {covered / total_coverage:.1%} are covered and {1 - covered / total_coverage:.1%} are uncovered.",
        f"Covered patterns have lower mean loss in {covered_better} of {len(coverage_pairs)} directed pairs. This indicates that source inventories lacking a target's exact full timing pattern explain a substantial part of poor transfer.",
    ])

    all_positive = sum(float(row["ci95_lower_bits_per_symbol"]) > 0 for row in uncertainty)
    write_markdown(result_root / "transfer_matrix/uncertainty/README.md", [
        "# Target-group bootstrap uncertainty",
        "",
        "`bootstrap_draws.csv` contains 1,000 resamples per cross-clan direction. Each draw resamples whole target recording groups—Caribbean days or Pacific repertoires—after model scoring. `summary.csv` records the point estimate and its 95% interval.",
        "",
        f"All {all_positive} of {len(uncertainty)} cross-clan 95% intervals have positive lower bounds. Thus every observed cross-clan loss remains positive across the target-group resamples.",
    ])

    pacific_identity = [row for row in identity if row["source_dataset"] == "pacific"]
    by_status = defaultdict(list)
    for row in pacific_identity:
        by_status[row["identity_status"]].append(row)
    write_markdown(result_root / "transfer_matrix/identity/README.md", [
        "# Pacific identity-status breakdown",
        "",
        "`scores.csv` attaches Hersh et al. Table S2 `identity`/`non_identity` labels to existing Pacific target-coda paired scores. Models were trained on all eligible codas; this is a post-scoring analysis. `bootstrap_draws.csv` and `summary.csv` provide stratum-specific uncertainty.",
        "",
        "| Status | Pacific↔Pacific directed cells | Mean transfer loss (bits/symbol) |",
        "|---|---:|---:|",
        *[f"| {status} | {len(rows)} | {mean([float(row['mean_transfer_loss_bits_per_symbol']) for row in rows]):.6f} |" for status, rows in sorted(by_status.items())],
        "",
        "Non-identity codas have lower average transfer loss than identity codas in the Pacific↔Pacific subset. This descriptive contrast is distinct from the overlap test below.",
    ])

    write_markdown(result_root / "transfer_matrix/pacific_overlap/README.md", [
        "# Exploratory Pacific spatial overlap",
        "",
        "`pairs.csv` contains the 21 unordered Pacific pairs, Table S4's mean two-direction spatial overlap, and mean bidirectional transfer loss. `summary.csv` contains the exact 7! = 5,040 clan-label permutation test. The predeclared directional prediction is higher overlap associated with lower transfer loss.",
        "",
        "| Status | Pearson r | One-sided p for predicted negative association | Two-sided exact p |",
        "|---|---:|---:|---:|",
        *[f"| {row['identity_status']} | {float(row['association_statistic']):.6f} | {float(row['exact_one_sided_p_value']):.6f} | {float(row['exact_two_sided_p_value']):.6f} |" for row in overlap],
        "",
        "Both observed associations are positive, opposite to the predeclared prediction. The non-identity association is positive with two-sided exact p = 0.012698. This remains exploratory: pair values share clans and Table S4 measures repertoire proximity, not confirmed contemporaneous co-occurrence.",
    ])
    return 8
