"""Experiment: does hyperparameter selection on several pooled validation days,
instead of the official single next-day-in-ring validation day, fix A_B's LSTM
v2 test-day underperformance relative to A alone?

Ardjano approved trying a validation-design change (2026-09-30) after seeing
(1) Iulia's stage-3 run confirming the failure at scale, (2) our B-encoder
investigation, and (3) lstm_v2_3seed_selection_notes.md ruling out 3-seed
averaging as a fix. This script is the cheapest way to try it: no change to
folds.py, the committed fold manifest, or DEC-2 itself. It builds an alternate
day_splits locally, for this experiment only, and reuses prepare_fold and
select_lstm_grid_and_fit unchanged -- only which days are labelled
"validation" changes.

K=3 validation days (the official validation day plus the next two in the
same seeded SHA256(seed:day) ring day_folds() already uses) is a judgment
call: enough to average out one day's idiosyncrasies without spending most of
the remaining training days on validation. Picked to match the "3" in the
already-ruled-out 3-seed experiment, not because 3 is derived from anything
in PLAN.md.

If this looks promising, the real fix is a change to folds.py and the
committed manifest (so every condition, logistic included, gets identical
splits per CLAUDE.md), which needs a DECISIONS.md entry before any reported
run -- not something to leave living in this script.

Run with: PYTHONPATH=src python experiments/lstm_v2_multi_validation_days.py
"""
import hashlib

import numpy as np

from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl
from whale_coda_dynamics.preprocessing.folds import prepare_fold
from whale_coda_dynamics.neural import select_lstm_grid_and_fit, predict_v2

MANIFEST = 'data/processed/rq1_folds/manifest.json'
TARGETS = 'data/processed/rq1_targets.jsonl'
SEEDS = [42, 43, 44]
K_VALIDATION_DAYS = 3

# Per-fold config mirrors each fold's pilot-chosen dropout/hidden for A_B_S
# (reports/rq1_lstm_v2_pilot.md, models.jsonl), same grids as the 3-seed
# experiment so the two are comparable.
FOLD_CONFIGS = {
    '2015-03-02': dict(dropout_grid=[0.4], hidden_grid=[32], weight_decay_history_grid=[1e-5, 1e-4, 1e-3]),
    '2015-03-26': dict(dropout_grid=[0.2], hidden_grid=[64], weight_decay_history_grid=[1e-4, 1e-3, 1e-2]),
}
BASE_CONFIG = dict(embedding_dim=32, learning_rate=1e-3, batch_size=32, max_epochs=300, patience=20,
                   weight_decay_s_grid=[1e-4])


def multi_validation_fold(fold, seed, k):
    """Same test day and seeded ring as day_folds(); validation is the next
    k days in that ring instead of just the next one. Everything else that
    was "train" stays train, shrunk by the k-1 extra validation days."""
    days = sorted(fold['day_splits'])
    ordered = sorted(days, key=lambda d: (hashlib.sha256(f"{seed}:{d}".encode()).hexdigest(), d))
    i = ordered.index(fold['test_day'])
    validation_days = {ordered[(i + j) % len(ordered)] for j in range(1, k + 1)}
    assert fold['test_day'] not in validation_days
    day_splits = {d: 'test' if d == fold['test_day'] else 'validation' if d in validation_days else 'train'
                  for d in days}
    return dict(fold, day_splits=day_splits, validation_day=sorted(validation_days))


def test_log_likelihood(fits, vocab, test, fields):
    y = np.array([vocab.index(r['Ct']) for r in test])
    lls = []
    for model, timing_encoder, _ in fits:
        p = predict_v2(model, timing_encoder, test, vocab, fields, max_len=10)
        lls.append(np.mean(np.log2(p[np.arange(len(test)), y])))
    return float(np.mean(lls))


def run_fold(test_day, grid_config):
    manifest = read_json(MANIFEST)
    rows = read_jsonl(TARGETS)
    official_fold = next(f for f in manifest['folds'] if f['test_day'] == test_day)
    fold = multi_validation_fold(official_fold, manifest['configuration']['validation_seed'], K_VALIDATION_DAYS)
    mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
    train = [r for r in mapped if r['split'] == 'train' and r['comparison_eligible']]
    validation = [r for r in mapped if r['split'] == 'validation' and r['comparison_eligible']]
    test = [r for r in mapped if r['split'] == 'test' and r['comparison_eligible']]
    vocab = policy['vocabulary']
    config = dict(BASE_CONFIG, **grid_config)
    print(f"{test_day}  validation days={fold['validation_day']}  "
          f"train={len(train)} validation={len(validation)} test={len(test)}")

    chosen_a, fits_a, _ = select_lstm_grid_and_fit(train, validation, vocab, ['A_history'], config, SEEDS, 10, [])
    chosen_ab, fits_ab, _ = select_lstm_grid_and_fit(train, validation, vocab, ['A_history', 'B_history'], config, SEEDS, 10, [])
    print(f"{test_day}  A:    wd_history={chosen_a['weight_decay_history']}  "
          f"val={chosen_a['validation_log_likelihood_bits']:.5f}  "
          f"test={test_log_likelihood(fits_a, vocab, test, ['A_history']):.5f}")
    print(f"{test_day}  A_B:  wd_history={chosen_ab['weight_decay_history']}  "
          f"val={chosen_ab['validation_log_likelihood_bits']:.5f}  "
          f"test={test_log_likelihood(fits_ab, vocab, test, ['A_history', 'B_history']):.5f}")


if __name__ == '__main__':
    for test_day, grid_config in FOLD_CONFIGS.items():
        run_fold(test_day, grid_config)
