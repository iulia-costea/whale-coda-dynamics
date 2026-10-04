"""Report the EC2 ZZZ-exclusion sensitivity against the primary matrix."""
from whale_coda_dynamics.preprocessing.io_utils import write_text

from .data import TRANSFER_SUMMARY_COLUMNS, read_csv


def write_zzz_comparison(config_path):
    """Write changes for every transfer cell involving EC2."""
    from pathlib import Path
    from whale_coda_dynamics.preprocessing.io_utils import read_json

    path = Path(config_path).resolve()
    root = path.parent.parent
    config = read_json(path)
    primary = read_csv(root / config["primary_transfer_summary_input"], TRANSFER_SUMMARY_COLUMNS)
    sensitivity = read_csv(root / config["transfer_summary_output"], TRANSFER_SUMMARY_COLUMNS)
    primary_by_pair = {(row["source_clan"], row["target_clan"]): row for row in primary}
    sensitivity_by_pair = {(row["source_clan"], row["target_clan"]): row for row in sensitivity}
    pairs = sorted(pair for pair in sensitivity_by_pair if "EC2" in pair)
    lines = [
        "# RQ3 EC2 ZZZ sensitivity",
        "",
        "EC2 codas with `source_unit == ZZZ` were excluded before fold construction and model selection. "
        "The sensitivity run therefore has five EC2 one-day folds.",
        "",
        "| Source | Target | Primary loss (bits/symbol) | ZZZ-excluded loss (bits/symbol) | Change |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in pairs:
        primary_value = float(primary_by_pair[pair]["mean_transfer_loss_bits_per_symbol"])
        sensitivity_value = float(sensitivity_by_pair[pair]["mean_transfer_loss_bits_per_symbol"])
        lines.append(
            f"| {pair[0]} | {pair[1]} | {primary_value:.6f} | "
            f"{sensitivity_value:.6f} | {sensitivity_value - primary_value:.6f} |"
        )
    lines.extend(["", "Positive change means the ZZZ-excluded transfer loss is larger.", ""])
    write_text(root / config["sensitivity_comparison_report_output"], "\n".join(lines))
