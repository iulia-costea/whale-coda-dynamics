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
    lines += ["", "The complete candidate-fold table is in `rq3_self_calibration.csv`; coda-level own scores are in `rq3_self_scores.csv`.", ""]
    return "\n".join(lines)


def write_self_calibration_report(path, calibration_rows, self_score_rows):
    """Write the generated calibration report."""
    write_text(path, build_self_calibration_report(calibration_rows, self_score_rows))
