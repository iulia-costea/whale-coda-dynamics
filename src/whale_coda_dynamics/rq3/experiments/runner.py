"""Run selected RQ3 calibration and initial-transfer stages."""
import argparse
from collections import defaultdict
from pathlib import Path

from whale_coda_dynamics.rq3.preprocessing.tokenize import TOKENIZER_ID, tokenize_ici_sequence

from .data import CALIBRATION_COLUMNS, SELF_SCORE_COLUMNS, load_experiment_data, write_csv
from .evaluation import mean_bits_per_coda, mean_bits_per_symbol
from .models import SmoothedNGram
from .report import write_self_calibration_report
from .scoring import score_coda_bits
from .transfer import run_full_transfer_matrix, run_initial_caribbean_transfer
from .uncertainty import run_bootstrap
from .identity import run_identity_breakdown
from .overlap import run_pacific_overlap
from .result_summaries import build_summaries


def tokenise_codas(codas):
    """Attach fixed primary tokens to every prepared coda."""
    for coda in codas:
        coda["tokens"] = tokenize_ici_sequence(coda["ici_sequence_s"])
    return codas


def fit_and_score(training_codas, scoring_codas, candidate):
    """Fit one candidate model and return coda-level score dictionaries."""
    model = SmoothedNGram(candidate["context_order"], candidate["alpha"])
    model.fit(coda["tokens"] for coda in training_codas)
    rows = []
    for coda in scoring_codas:
        score = score_coda_bits(model, coda["tokens"])
        rows.append({**coda, **score})
    return rows


def calibrate_clan(codas, candidates):
    """Select one candidate by source-fold validation, then cross-fit own scores."""
    folds = sorted({coda["fold"] for coda in codas})
    candidate_rows = []
    for candidate_index, candidate in enumerate(candidates):
        held_out = []
        for fold in folds:
            training = [coda for coda in codas if coda["fold"] != fold]
            scoring = [coda for coda in codas if coda["fold"] == fold]
            held_out.extend(fit_and_score(training, scoring, candidate))
        candidate_rows.append({
            "candidate_index": candidate_index,
            **candidate,
            "validation_coda_count": len(held_out),
            "mean_bits_per_coda": mean_bits_per_coda(held_out),
            "mean_bits_per_symbol": mean_bits_per_symbol(held_out),
        })
    best = max(candidate_rows, key=lambda row: (row["mean_bits_per_symbol"], -row["candidate_index"]))
    own_scores = []
    selected_candidate = candidates[best["candidate_index"]]
    for fold in folds:
        training = [coda for coda in codas if coda["fold"] != fold]
        scoring = [coda for coda in codas if coda["fold"] == fold]
        for score in fit_and_score(training, scoring, selected_candidate):
            own_scores.append({
                "rq3_coda_id": score["rq3_coda_id"],
                "source_dataset": score["source_dataset"],
                "clan": score["clan"],
                "group_id": score["group_id"],
                "fold": score["fold"],
                "model_family": selected_candidate["family"],
                "context_order": selected_candidate["context_order"],
                "alpha": selected_candidate["alpha"],
                "token_count": score["token_count"],
                "coda_log2_probability": score["coda_log2_probability"],
                "bits_per_coda": score["bits_per_coda"],
                "bits_per_symbol": score["bits_per_symbol"],
            })
    return candidate_rows, own_scores


def run_calibration(config_path):
    """Write calibration candidates, cross-fitted own scores, and their report."""
    config, root, codas = load_experiment_data(config_path)
    if config["tokenizer_id"] != TOKENIZER_ID:
        raise ValueError("Experiment configuration tokenizer does not match implementation")
    tokenise_codas(codas)
    by_clan = defaultdict(list)
    for coda in codas:
        by_clan[(coda["source_dataset"], coda["clan"])].append(coda)
    calibration_rows, self_score_rows = [], []
    for (dataset, clan), clan_codas in sorted(by_clan.items()):
        candidate_rows, own_scores = calibrate_clan(clan_codas, config["candidates"])
        selected_index = max(candidate_rows, key=lambda row: (row["mean_bits_per_symbol"], -row["candidate_index"]))["candidate_index"]
        for row in candidate_rows:
            calibration_rows.append({
                "source_dataset": dataset,
                "clan": clan,
                "candidate_index": row["candidate_index"],
                "model_family": row["family"],
                "context_order": row["context_order"],
                "alpha": row["alpha"],
                "validation_coda_count": row["validation_coda_count"],
                "mean_bits_per_coda": row["mean_bits_per_coda"],
                "mean_bits_per_symbol": row["mean_bits_per_symbol"],
                "selected": str(row["candidate_index"] == selected_index).lower(),
            })
        self_score_rows.extend(own_scores)
    calibration_rows.sort(key=lambda row: (row["source_dataset"], row["clan"], row["candidate_index"]))
    self_score_rows.sort(key=lambda row: row["rq3_coda_id"])
    for output in ("self_scores_output", "calibration_output", "report_output"):
        (root / config[output]).parent.mkdir(parents=True, exist_ok=True)
    write_csv(root / config["self_scores_output"], SELF_SCORE_COLUMNS, self_score_rows)
    write_csv(root / config["calibration_output"], CALIBRATION_COLUMNS, calibration_rows)
    write_self_calibration_report(root / config["report_output"], calibration_rows, self_score_rows)
    return {"codas": len(codas), "clans": len(by_clan), "self_scores": len(self_score_rows)}


def run_initial_transfer(config_path):
    """Score the approved first EC1/EC2 transfer comparison from saved calibration."""
    config, root, codas = load_experiment_data(config_path)
    if config["tokenizer_id"] != TOKENIZER_ID:
        raise ValueError("Experiment configuration tokenizer does not match implementation")
    tokenise_codas(codas)
    for output in (
            "initial_transfer_fold_scores_output", "initial_transfer_scores_output",
            "initial_transfer_summary_output", "initial_transfer_report_output"):
        (root / config[output]).parent.mkdir(parents=True, exist_ok=True)
    fold_rows, aggregate_rows, summaries = run_initial_caribbean_transfer(root, config, codas)
    from .report import write_initial_transfer_report
    write_initial_transfer_report(root / config["initial_transfer_report_output"], summaries)
    return {"fold_scores": len(fold_rows), "paired_target_scores": len(aggregate_rows), "pairs": len(summaries)}


def run_full_matrix(config_path):
    """Score the approved complete 9 x 9 primary RQ3 transfer matrix."""
    config, root, codas = load_experiment_data(config_path)
    if config["tokenizer_id"] != TOKENIZER_ID:
        raise ValueError("Experiment configuration tokenizer does not match implementation")
    tokenise_codas(codas)
    for output in ("transfer_fold_scores_output", "transfer_scores_output", "transfer_summary_output", "transfer_coverage_summary_output", "transfer_report_output"):
        (root / config[output]).parent.mkdir(parents=True, exist_ok=True)
    fold_rows, aggregate_rows, summaries, coverage = run_full_transfer_matrix(root, config, codas)
    from .report import write_transfer_report
    uncertainty_rows = []
    uncertainty_path = root / config["transfer_uncertainty_summary_output"]
    if uncertainty_path.exists():
        from .data import UNCERTAINTY_SUMMARY_COLUMNS, read_csv
        uncertainty_rows = read_csv(uncertainty_path, UNCERTAINTY_SUMMARY_COLUMNS)
    write_transfer_report(root / config["transfer_report_output"], summaries, coverage, uncertainty_rows)
    return {"fold_scores": len(fold_rows), "paired_target_scores": len(aggregate_rows), "cells": len(summaries), "coverage_rows": len(coverage)}


def run_uncertainty(config_path):
    """Bootstrap fixed cross-clan transfer scores over target recording groups."""
    config, root, _ = load_experiment_data(config_path)
    for output in ("transfer_bootstrap_output", "transfer_uncertainty_summary_output"):
        (root / config[output]).parent.mkdir(parents=True, exist_ok=True)
    draw_rows, uncertainty_rows = run_bootstrap(root, config)
    from .data import COVERAGE_SUMMARY_COLUMNS, TRANSFER_SUMMARY_COLUMNS, read_csv
    from .report import write_transfer_report
    summaries = read_csv(root / config["transfer_summary_output"], TRANSFER_SUMMARY_COLUMNS)
    coverage = read_csv(root / config["transfer_coverage_summary_output"], COVERAGE_SUMMARY_COLUMNS)
    write_transfer_report(root / config["transfer_report_output"], summaries, coverage, uncertainty_rows)
    return {"draws": len(draw_rows), "cells": len(uncertainty_rows)}


def run_identity(config_path):
    """Add Pacific identity/non-identity strata to the fixed transfer results."""
    config, root, _ = load_experiment_data(config_path)
    for output in ("transfer_identity_scores_output", "transfer_identity_bootstrap_output", "transfer_identity_summary_output"):
        (root / config[output]).parent.mkdir(parents=True, exist_ok=True)
    score_rows, draw_rows, identity_rows = run_identity_breakdown(root, config)
    from .data import COVERAGE_SUMMARY_COLUMNS, TRANSFER_SUMMARY_COLUMNS, UNCERTAINTY_SUMMARY_COLUMNS, read_csv
    from .report import write_transfer_report
    summaries = read_csv(root / config["transfer_summary_output"], TRANSFER_SUMMARY_COLUMNS)
    coverage = read_csv(root / config["transfer_coverage_summary_output"], COVERAGE_SUMMARY_COLUMNS)
    uncertainty = read_csv(root / config["transfer_uncertainty_summary_output"], UNCERTAINTY_SUMMARY_COLUMNS)
    write_transfer_report(root / config["transfer_report_output"], summaries, coverage, uncertainty, identity_rows)
    return {"scores": len(score_rows), "draws": len(draw_rows), "strata": len(identity_rows)}


def run_overlap(config_path):
    """Run the exploratory exact Pacific spatial-overlap analysis."""
    config, root, _ = load_experiment_data(config_path)
    for output in ("pacific_overlap_pairs_output", "pacific_overlap_summary_output"):
        (root / config[output]).parent.mkdir(parents=True, exist_ok=True)
    pair_rows, overlap_results = run_pacific_overlap(root, config)
    from .data import COVERAGE_SUMMARY_COLUMNS, IDENTITY_SUMMARY_COLUMNS, TRANSFER_SUMMARY_COLUMNS, UNCERTAINTY_SUMMARY_COLUMNS, read_csv
    from .report import write_transfer_report
    summaries = read_csv(root / config["transfer_summary_output"], TRANSFER_SUMMARY_COLUMNS)
    coverage = read_csv(root / config["transfer_coverage_summary_output"], COVERAGE_SUMMARY_COLUMNS)
    uncertainty = read_csv(root / config["transfer_uncertainty_summary_output"], UNCERTAINTY_SUMMARY_COLUMNS)
    identity = read_csv(root / config["transfer_identity_summary_output"], IDENTITY_SUMMARY_COLUMNS)
    write_transfer_report(root / config["transfer_report_output"], summaries, coverage, uncertainty, identity, pair_rows, overlap_results)
    return {"pairs": len(pair_rows), "strata": len(overlap_results)}


def run_result_summaries(config_path):
    """Generate navigable README summaries beside each RQ3 results level."""
    config, root, _ = load_experiment_data(config_path)
    return {"readmes": build_summaries(root, config)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rq3_experiments.json")
    parser.add_argument("--stage", choices=("calibration", "initial-transfer", "matrix", "bootstrap", "identity", "overlap", "summaries"), default="calibration")
    args = parser.parse_args()
    if args.stage == "calibration":
        summary = run_calibration(args.config)
        print(f"RQ3 self-calibration complete: {summary['self_scores']} cross-fitted scores across {summary['clans']} clans.")
    elif args.stage == "initial-transfer":
        summary = run_initial_transfer(args.config)
        print(f"RQ3 initial Caribbean transfer complete: {summary['paired_target_scores']} paired target scores across {summary['pairs']} directions.")
    elif args.stage == "matrix":
        summary = run_full_matrix(args.config)
        print(f"RQ3 transfer matrix complete: {summary['paired_target_scores']} paired target scores across {summary['cells']} cells.")
    elif args.stage == "bootstrap":
        summary = run_uncertainty(args.config)
        print(f"RQ3 transfer bootstrap complete: {summary['draws']} draws across {summary['cells']} cross-clan cells.")
    elif args.stage == "identity":
        summary = run_identity(args.config)
        print(f"RQ3 Pacific identity breakdown complete: {summary['scores']} coda scores, {summary['draws']} draws, and {summary['strata']} strata.")
    elif args.stage == "overlap":
        summary = run_overlap(args.config)
        print(f"RQ3 Pacific overlap analysis complete: {summary['pairs']} pair rows across {summary['strata']} identity strata.")
    else:
        summary = run_result_summaries(args.config)
        print(f"RQ3 result summaries complete: {summary['readmes']} README files.")


if __name__ == "__main__":
    main()
