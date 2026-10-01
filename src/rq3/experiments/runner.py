"""Run RQ3 self-model calibration from the prepared coda and fold artifacts."""
import argparse
from collections import defaultdict
from pathlib import Path

from rq3.preprocessing.tokenize import TOKENIZER_ID, tokenize_ici_sequence

from .data import CALIBRATION_COLUMNS, SELF_SCORE_COLUMNS, load_experiment_data, write_csv
from .evaluation import mean_bits_per_coda, mean_bits_per_symbol
from .models import SmoothedNGram
from .report import write_self_calibration_report
from .scoring import score_coda_bits


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


def run(config_path):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rq3_experiments.json")
    args = parser.parse_args()
    summary = run(args.config)
    print(f"RQ3 self-calibration complete: {summary['self_scores']} cross-fitted scores across {summary['clans']} clans.")


if __name__ == "__main__":
    main()
