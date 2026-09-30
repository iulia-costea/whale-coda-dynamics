"""Experiment: does averaging validation log-likelihood over 3 seeds (instead
of the pilot's 1) during hyperparameter selection fix A_B's LSTM v2 test-day
underperformance relative to A alone? See lstm_v2_3seed_selection_notes.md
for the result and interpretation. Read-only against committed data; writes
nothing back into data/ or reports/.

Run with: PYTHONPATH=src python experiments/lstm_v2_3seed_selection.py
"""
import numpy as np

from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl
from whale_coda_dynamics.preprocessing.folds import prepare_fold
from whale_coda_dynamics.neural import select_lstm_grid_and_fit, predict_v2

MANIFEST = 'data/processed/rq1_folds/manifest.json'
TARGETS = 'data/processed/rq1_targets.jsonl'
SEEDS = [42, 43, 44]

# Per-fold config mirrors each fold's pilot-chosen dropout/hidden for A_B_S
# (reports/rq1_lstm_v2_pilot.md, models.jsonl), varying only weight_decay_history,
# the axis this experiment is about.
FOLD_CONFIGS = {
    '2015-03-02': dict(dropout_grid=[0.4], hidden_grid=[32], weight_decay_history_grid=[1e-5, 1e-4, 1e-3]),
    '2015-03-26': dict(dropout_grid=[0.2], hidden_grid=[64], weight_decay_history_grid=[1e-4, 1e-3, 1e-2]),
}
BASE_CONFIG = dict(embedding_dim=32, learning_rate=1e-3, batch_size=32, max_epochs=300, patience=20,
                   weight_decay_s_grid=[1e-4])


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
    fold = next(f for f in manifest['folds'] if f['test_day'] == test_day)
    mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
    train = [r for r in mapped if r['split'] == 'train' and r['comparison_eligible']]
    validation = [r for r in mapped if r['split'] == 'validation' and r['comparison_eligible']]
    test = [r for r in mapped if r['split'] == 'test' and r['comparison_eligible']]
    vocab = policy['vocabulary']
    config = dict(BASE_CONFIG, **grid_config)

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
