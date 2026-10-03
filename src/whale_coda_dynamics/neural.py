"""LSTM main study for RQ1 (PLAN [M4], milestone 7).

One encoder per enabled history stream (A, B), a small feed-forward encoder
for S, concatenated into a shared per-condition output layer. An absent
input means its encoder is removed, never masked. Fixed hyperparameters
(hidden size, optimizer, batch size, early stopping) are identical across
conditions and folds; only which encoders exist, and their input size,
differs by condition. Three neural seeds are averaged per coda by the
shared evaluation module before contrasts are computed.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluation import evaluate, classification_metrics, log_likelihood_bits, day_bootstrap_bits
from .logistic import (Encoder, repeat_distribution, check_stop_conditions, classify_add_input_contrasts, describe,
                       last_label_index, format_ci)
from .preprocessing.folds import prepare_fold
from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text

LOG2E = 1.4426950408889634  # 1/ln(2): natural-log cross-entropy -> bits


def encode_history(rows, field, vocab, max_len):
    """Integer token sequences (fold vocabulary index, chronological, already
    <= max_len from preprocessing) padded with pad_idx = len(vocab)."""
    index = {label: i for i, label in enumerate(vocab)}
    pad_idx = len(vocab)
    seqs = np.full((len(rows), max_len), pad_idx, dtype=np.int64)
    lengths = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        history = r[field]
        if not history:
            raise ValueError('Comparison rows must have nonempty history for an enabled stream')
        for j, label in enumerate(history[:max_len]):
            seqs[i, j] = index[label]
        lengths[i] = min(len(history), max_len)
    return seqs, lengths


class HistoryEncoder(nn.Module):
    """dropout=0.0 (the v1 default) makes nn.Dropout an identity in both train
    and eval mode, so v1's ConditionModel is unaffected by this parameter."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs, lengths):
        out, _ = self.lstm(self.embedding(seqs))
        # Padding after each row's true length cannot influence earlier steps
        # (the LSTM is causal), so indexing the true last step needs no packing.
        idx = (lengths - 1).clamp(min=0)
        return self.dropout(out[torch.arange(seqs.shape[0]), idx])


class TimingEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))


class ConditionModel(nn.Module):
    """Encoders present follow `fields` exactly; absent streams have no
    parameters and contribute nothing, matching the no-masking hard rule."""
    def __init__(self, vocab_size, fields, config, pad_idx, timing_dim):
        super().__init__()
        self.fields = fields
        hidden = config['hidden_dim']
        total = 0
        if 'A_history' in fields:
            self.a_encoder = HistoryEncoder(vocab_size, config['embedding_dim'], hidden, pad_idx)
            total += hidden
        if 'B_history' in fields:
            self.b_encoder = HistoryEncoder(vocab_size, config['embedding_dim'], hidden, pad_idx)
            total += hidden
        if timing_dim is not None:
            self.s_encoder = TimingEncoder(timing_dim, hidden)
            total += hidden
        self.output = nn.Linear(total, vocab_size)

    def forward(self, batch):
        parts = []
        if 'A_history' in self.fields:
            parts.append(self.a_encoder(batch['a_seq'], batch['a_len']))
        if 'B_history' in self.fields:
            parts.append(self.b_encoder(batch['b_seq'], batch['b_len']))
        if hasattr(self, 's_encoder'):
            parts.append(self.s_encoder(batch['s_features']))
        return self.output(torch.cat(parts, dim=1))


class ConditionModelV2(nn.Module):
    """LSTM v2 (DEC-17, DEC-20). Two changes from v1's ConditionModel:
    - copy weights beta/gamma, exactly like logistic v2+ (a single shared
      scalar added to the logit of the class matching the own/partner last
      coda; beta only with A, gamma only with B);
    - dropout on each encoder's output, and the shared output layer is
      decomposed into one linear projection per encoder, summed, instead of
      one linear layer over the concatenation. Summing separate projections
      over a partition of the input is mathematically identical to one
      projection over the concatenation (a block-diagonal reweighting of the
      same computation), but it lets history and S carry separate weight
      decay via optimizer parameter groups, which a single shared weight
      matrix could not (DEC-20)."""
    def __init__(self, vocab_size, fields, config, pad_idx, timing_dim):
        super().__init__()
        self.fields = fields
        hidden, dropout = config['hidden_dim'], config['dropout_rate']
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        if 'A_history' in fields:
            self.a_encoder = HistoryEncoder(vocab_size, config['embedding_dim'], hidden, pad_idx, dropout)
            self.a_output = nn.Linear(hidden, vocab_size, bias=False)
            self.beta = nn.Parameter(torch.tensor(0.0))
        if 'B_history' in fields:
            self.b_encoder = HistoryEncoder(vocab_size, config['embedding_dim'], hidden, pad_idx, dropout)
            self.b_output = nn.Linear(hidden, vocab_size, bias=False)
            self.gamma = nn.Parameter(torch.tensor(0.0))
        if timing_dim is not None:
            self.s_encoder = TimingEncoder(timing_dim, hidden, dropout)
            self.s_output = nn.Linear(hidden, vocab_size, bias=False)

    def history_parameters(self):
        params = [self.output_bias]
        for name in ('a_encoder', 'a_output', 'beta', 'b_encoder', 'b_output', 'gamma'):
            if hasattr(self, name):
                attr = getattr(self, name)
                params += list(attr.parameters()) if isinstance(attr, nn.Module) else [attr]
        return params

    def s_parameters(self):
        if not hasattr(self, 's_encoder'):
            return []
        return list(self.s_encoder.parameters()) + list(self.s_output.parameters())

    def forward(self, batch):
        logits = None
        if 'A_history' in self.fields:
            logits = self.a_output(self.a_encoder(batch['a_seq'], batch['a_len']))
        if 'B_history' in self.fields:
            l_b = self.b_output(self.b_encoder(batch['b_seq'], batch['b_len']))
            logits = l_b if logits is None else logits + l_b
        if hasattr(self, 's_encoder'):
            l_s = self.s_output(self.s_encoder(batch['s_features']))
            logits = l_s if logits is None else logits + l_s
        logits = logits + self.output_bias
        if hasattr(self, 'beta') and 'own_idx' in batch:
            mask = batch['own_idx'] >= 0
            boost = torch.zeros_like(logits)
            boost[mask, batch['own_idx'][mask]] = self.beta
            logits = logits + boost
        if hasattr(self, 'gamma') and 'partner_idx' in batch:
            mask = batch['partner_idx'] >= 0
            boost = torch.zeros_like(logits)
            boost[mask, batch['partner_idx'][mask]] = self.gamma
            logits = logits + boost
        return logits


def build_batch(rows, vocab, fields, max_len, timing_encoder, include_copy_targets=False):
    """include_copy_targets=True (LSTM v2 only) adds own_idx/partner_idx: the
    vocabulary index of each row's own/partner last coda (or -1 if absent),
    for the beta/gamma copy weights. v1 never sets this, so it is unaffected."""
    batch = {}
    if 'A_history' in fields:
        seqs, lengths = encode_history(rows, 'A_history', vocab, max_len)
        batch['a_seq'], batch['a_len'] = torch.from_numpy(seqs), torch.from_numpy(lengths)
        if include_copy_targets:
            batch['own_idx'] = torch.from_numpy(last_label_index(rows, 'A_history', vocab))
    if 'B_history' in fields:
        seqs, lengths = encode_history(rows, 'B_history', vocab, max_len)
        batch['b_seq'], batch['b_len'] = torch.from_numpy(seqs), torch.from_numpy(lengths)
        if include_copy_targets:
            batch['partner_idx'] = torch.from_numpy(last_label_index(rows, 'B_history', vocab))
    if timing_encoder is not None:
        batch['s_features'] = torch.from_numpy(timing_encoder.transform(rows)[:, :-1].astype(np.float32))
    return batch


def fit_condition(train, validation, vocab, fields, config, seed, max_len):
    torch.manual_seed(seed)
    timing_fields = [f for f in fields if f in ('turn_category', 'gap_s', 'overlap_elapsed_s')]
    timing_encoder = Encoder(train, vocab, timing_fields) if timing_fields else None
    timing_dim = len(timing_encoder.names) if timing_encoder is not None else None
    pad_idx = len(vocab)
    model = ConditionModel(len(vocab), fields, config, pad_idx, timing_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    y_train = torch.tensor([vocab.index(r['Ct']) for r in train], dtype=torch.long)
    y_val = torch.tensor([vocab.index(r['Ct']) for r in validation], dtype=torch.long)
    val_batch = build_batch(validation, vocab, fields, max_len, timing_encoder)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_val_ll, epochs_since_best, epoch = None, -float('inf'), 0, 0
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(train), config['batch_size']):
            idx = order[start:start + config['batch_size']]
            batch_rows = [train[i] for i in idx.tolist()]
            batch = build_batch(batch_rows, vocab, fields, max_len, timing_encoder)
            logits = model(batch)
            loss = nn.functional.cross_entropy(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_batch)
            val_ll = (-nn.functional.cross_entropy(val_logits, y_val) * LOG2E).item()
        if val_ll > best_val_ll:
            best_val_ll, best_state, epochs_since_best = val_ll, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= config['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    convergence = dict(epochs_trained=epoch, best_validation_log_likelihood_bits=best_val_ll,
                       early_stopped=epochs_since_best >= config['patience'])
    return model, timing_encoder, convergence


def predict(model, timing_encoder, rows, vocab, fields, max_len):
    """Softmax runs in float32, so raw probabilities can miss summing to 1
    within the shared evaluator's 1e-10 tolerance; renormalize in float64."""
    model.eval()
    with torch.no_grad():
        batch = build_batch(rows, vocab, fields, max_len, timing_encoder)
        probs = nn.functional.softmax(model(batch), dim=1).numpy().astype(np.float64)
        return probs / probs.sum(axis=1, keepdims=True)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def fit_condition_v2(train, validation, vocab, fields, config, seed, max_len, device='cpu'):
    """LSTM v2 (DEC-17, DEC-20). Trains one ConditionModelV2 at a fixed
    hyperparameter setting (config['dropout_rate'], config['hidden_dim'],
    config['weight_decay_history'], config['weight_decay_s']); the grid
    search over these lives in select_lstm_grid_and_fit. Logs per-epoch
    train/validation log-likelihood (DEC-17 stage 2), same early-stopping-
    and-restore-best-epoch pattern as LSTM v1."""
    torch.manual_seed(seed)
    timing_fields = [f for f in fields if f in ('turn_category', 'gap_s', 'overlap_elapsed_s')]
    timing_encoder = Encoder(train, vocab, timing_fields) if timing_fields else None
    timing_dim = len(timing_encoder.names) if timing_encoder is not None else None
    pad_idx = len(vocab)
    model = ConditionModelV2(len(vocab), fields, config, pad_idx, timing_dim).to(device)
    groups = [dict(params=model.history_parameters(), weight_decay=config['weight_decay_history'])]
    s_params = model.s_parameters()
    if s_params:
        groups.append(dict(params=s_params, weight_decay=config['weight_decay_s']))
    optimizer = torch.optim.Adam(groups, lr=config['learning_rate'])
    y_train = torch.tensor([vocab.index(r['Ct']) for r in train], dtype=torch.long, device=device)
    y_val = torch.tensor([vocab.index(r['Ct']) for r in validation], dtype=torch.long, device=device)
    val_batch = move_batch(build_batch(validation, vocab, fields, max_len, timing_encoder, include_copy_targets=True), device)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_val_ll, epochs_since_best, epoch, curve = None, -float('inf'), 0, 0, []
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        train_losses = []
        for start in range(0, len(train), config['batch_size']):
            idx = order[start:start + config['batch_size']]
            batch_rows = [train[i] for i in idx.tolist()]
            batch = move_batch(build_batch(batch_rows, vocab, fields, max_len, timing_encoder, include_copy_targets=True), device)
            loss = nn.functional.cross_entropy(model(batch), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_ll = (-nn.functional.cross_entropy(model(val_batch), y_val) * LOG2E).item()
        train_ll = -(sum(train_losses) / len(train_losses)) * LOG2E
        curve.append(dict(epoch=epoch, train_log_likelihood_bits=train_ll, validation_log_likelihood_bits=val_ll))
        if val_ll > best_val_ll:
            best_val_ll, best_state, epochs_since_best = val_ll, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= config['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    convergence = dict(epochs_trained=epoch, best_validation_log_likelihood_bits=best_val_ll,
                       early_stopped=epochs_since_best >= config['patience'])
    return model, timing_encoder, convergence, curve


def predict_v2(model, timing_encoder, rows, vocab, fields, max_len, device='cpu'):
    model.eval()
    with torch.no_grad():
        batch = move_batch(build_batch(rows, vocab, fields, max_len, timing_encoder, include_copy_targets=True), device)
        probs = nn.functional.softmax(model(batch), dim=1).cpu().numpy().astype(np.float64)
        return probs / probs.sum(axis=1, keepdims=True)


def _fit_grid_point(task):
    """Module-level (picklable) worker for select_lstm_grid_and_fit's optional
    process-pool parallelism: fits one (setting, seed) combination."""
    train, validation, vocab, fields, trial, seed, max_len, device = task
    if device == 'cpu':
        torch.set_num_threads(1)  # avoid oversubscription: process parallelism, not torch's own thread pool
    return fit_condition_v2(train, validation, vocab, fields, trial, seed, max_len, device)


def select_lstm_grid_and_fit(train, validation, vocab, fields, config, seeds, max_len, curve_sink, device='cpu',
                             fold_id=None, condition=None, workers=1):
    """DEC-17/DEC-20: joint grid search on the validation day only (`test` is
    never an argument here, so it is structurally impossible for test data to
    influence selection, mirroring select_l2_and_fit's guarantee). Criterion
    is validation log-likelihood averaged over `seeds`; ties go to the
    smaller, more regularised setting: smaller hidden, then larger dropout,
    then larger weight_decay_history, then larger weight_decay_s (this
    priority order is a judgement call where DEC-17 only says "smaller, more
    regularised", not which axis wins a tie first).

    Conditions without S search dropout x hidden x weight_decay_history only
    (weight_decay_s is None throughout); conditions with S jointly search all
    four axes. Every epoch of every fit (not just the chosen setting) is
    appended to curve_sink, per DEC-17 stage 2's per-epoch logging.

    workers > 1 fits every (setting, seed) combination in a process pool
    instead of one at a time (each grid point is independent); this only
    changes wall time, not results, and defaults to 1 (sequential, the
    original behaviour) so existing configs and tests are unaffected."""
    timing_fields = [f for f in fields if f in ('turn_category', 'gap_s', 'overlap_elapsed_s')]
    wd_s_axis = config['weight_decay_s_grid'] if timing_fields else [None]
    settings = [(dropout, hidden, wd_history, wd_s)
               for dropout in config['dropout_grid'] for hidden in config['hidden_grid']
               for wd_history in config['weight_decay_history_grid'] for wd_s in wd_s_axis]
    tasks, task_keys = [], []
    for dropout, hidden, wd_history, wd_s in settings:
        trial = dict(config, dropout_rate=dropout, hidden_dim=hidden, weight_decay_history=wd_history, weight_decay_s=wd_s)
        for seed in seeds:
            tasks.append((train, validation, vocab, fields, trial, seed, max_len, device))
            task_keys.append((dropout, hidden, wd_history, wd_s, seed))
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            fit_results = list(pool.map(_fit_grid_point, tasks))
    else:
        fit_results = [_fit_grid_point(t) for t in tasks]
    by_setting = defaultdict(list)
    for (dropout, hidden, wd_history, wd_s, seed), (model, timing_encoder, conv, curve) in zip(task_keys, fit_results):
        for row in curve:
            curve_sink.append(dict(row, fold_id=fold_id, condition=condition, dropout=dropout, hidden=hidden,
                                  weight_decay_history=wd_history, weight_decay_s=wd_s, seed=seed))
        by_setting[dropout, hidden, wd_history, wd_s].append((model, timing_encoder, conv))

    grid, best = [], None
    for dropout, hidden, wd_history, wd_s in settings:
        seed_fits = by_setting[dropout, hidden, wd_history, wd_s]
        mean_val_ll = sum(c['best_validation_log_likelihood_bits'] for _, _, c in seed_fits) / len(seed_fits)
        entry = dict(dropout=dropout, hidden=hidden, weight_decay_history=wd_history, weight_decay_s=wd_s,
                    validation_log_likelihood_bits=mean_val_ll,
                    convergence=[c for _, _, c in seed_fits])
        key = (mean_val_ll, -hidden, dropout, wd_history, wd_s or 0)
        if best is None or key > best[0]:
            best = (key, entry, seed_fits)
        grid.append(entry)
    _, chosen, chosen_fits = best
    return chosen, chosen_fits, grid


def check_s_suppression(train, validation, vocab, fields_without_s, fields_with_s, config, weight_decay_s, seed, max_len, device='cpu'):
    """DEC-20 sanity check: conditional V-information assumes a model can
    choose to ignore an input that adds nothing (optional ignorance). This
    fits the with-S condition at the strongest S weight decay the grid
    offers (config['weight_decay_s_grid'] should include a value large
    enough to test this) and the without-S condition at the same history
    settings and seed, then compares validation log-likelihood: if a large
    enough S penalty cannot bring the with-S model close to the without-S
    one, the S encoder is not actually suppressible in practice, and the
    grid (or the mechanism) needs revisiting before trusting DEC-20's
    fix generalises from logistic regression to the LSTM."""
    _, _, conv_without_s, _ = fit_condition_v2(train, validation, vocab, fields_without_s, config, seed, max_len, device)
    with_s_config = dict(config, weight_decay_s=weight_decay_s)
    _, _, conv_with_s, _ = fit_condition_v2(train, validation, vocab, fields_with_s, with_s_config, seed, max_len, device)
    without_s_ll = conv_without_s['best_validation_log_likelihood_bits']
    with_s_ll = conv_with_s['best_validation_log_likelihood_bits']
    return dict(weight_decay_s=weight_decay_s, seed=seed,
               without_s_validation_log_likelihood_bits=without_s_ll,
               with_s_validation_log_likelihood_bits=with_s_ll,
               difference_bits=with_s_ll - without_s_ll)


def run_suppression_diagnostic(config_path):
    """Runs check_s_suppression on one fold (config['test_day']), one
    condition pair (config['fields_without_s']/config['fields_with_s']), at
    config['weight_decay_s'] (should be the top of the grid actually used),
    config['hidden_dim']/['dropout_rate']/['weight_decay_history'] fixed,
    and seeds config['seeds'] (report each, plus the mean). Writes
    reports/rq1_lstm_v2_suppression.md. This is a diagnostic, not a model
    run: it does not touch the pilot or full-run configs or results."""
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    manifest = read_json(root / config['manifest'])
    rows = read_jsonl(root / manifest['configuration']['input'])
    max_len = manifest['preparation_configuration']['history_limit']
    fold = next(f for f in manifest['folds'] if f['test_day'] == config['test_day'])
    mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
    train = [r for r in mapped if r['split'] == 'train' and r['comparison_eligible']]
    validation = [r for r in mapped if r['split'] == 'validation' and r['comparison_eligible']]
    vocab = policy['vocabulary']
    device = config.get('device', 'cpu')
    results = [check_s_suppression(train, validation, vocab, config['fields_without_s'], config['fields_with_s'],
                                   config, config['weight_decay_s'], seed, max_len, device)
              for seed in config['seeds']]
    mean_diff = sum(r['difference_bits'] for r in results) / len(results)
    passed = abs(mean_diff) <= config['tolerance_bits']
    lines = ['# RQ1 LSTM v2 validation-only S-suppression diagnostic (DEC-20)', '',
             f"Fold {config['test_day']}, without-S fields {config['fields_without_s']}, with-S fields "
             f"{config['fields_with_s']}, S weight decay fixed at {config['weight_decay_s']} (the top of the grid "
             f"actually used), history weight decay {config['weight_decay_history']}, dropout "
             f"{config['dropout_rate']}, hidden {config['hidden_dim']}, seeds {config['seeds']}, tolerance "
             f"{config['tolerance_bits']} bits/coda. If the with-S validation log-likelihood is not close to the "
             "without-S one even at this penalty, the S encoder cannot actually be switched off in practice, and "
             "DEC-20's fix would not generalise from logistic regression to the LSTM.", '',
             '| Seed | Without-S validation LL | With-S validation LL | Difference |', '|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['seed']} | {r['without_s_validation_log_likelihood_bits']:.5f} | "
                     f"{r['with_s_validation_log_likelihood_bits']:.5f} | {r['difference_bits']:.5f} |")
    lines += ['', f"Mean difference: {mean_diff:.5f} bits/coda. "
             f"{'Passed' if passed else 'Failed'}: {'within' if passed else 'outside'} the "
             f"{config['tolerance_bits']} bits/coda tolerance.", '']
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(lines))
    return dict(results=results, mean_difference_bits=mean_diff, passed=passed)


def run_timing_determinism_check(config_path):
    """DEC-17 stage 2 timing and determinism check. Fits one fold/condition's
    already-chosen pilot setting with config['seeds'] on each of
    config['devices'] ('cpu', and 'cuda' if listed and available), reports
    seconds/fit per device, and estimates stage 3's wall time from the full
    study's fit count (all 8 conditions x 14 folds x grid x 3 seeds).

    GPU determinism: refits the first seed a second time with
    torch.use_deterministic_algorithms(True) and checks it matches exactly.
    PyTorch's nn.LSTM docs require CUBLAS_WORKSPACE_CONFIG to be set before
    the process starts (not at runtime) for deterministic cuDNN RNNs on CUDA
    10.2+, so this raises immediately if 'cuda' is requested without it,
    rather than silently running non-deterministically.

    CPU and GPU are not expected to match exactly (different kernels); the
    check instead requires the CPU-GPU difference in mean validation
    log-likelihood across seeds to be smaller than the spread across seeds
    on one device."""
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    manifest = read_json(root / config['manifest'])
    rows = read_jsonl(root / manifest['configuration']['input'])
    max_len = manifest['preparation_configuration']['history_limit']
    fold = next(f for f in manifest['folds'] if f['test_day'] == config['test_day'])
    mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
    train = [r for r in mapped if r['split'] == 'train' and r['comparison_eligible']]
    validation = [r for r in mapped if r['split'] == 'validation' and r['comparison_eligible']]
    vocab = policy['vocabulary']
    fields = config['fields']
    setting = dict(config, hidden_dim=config['hidden_dim'], dropout_rate=config['dropout_rate'],
                  weight_decay_history=config['weight_decay_history'], weight_decay_s=config['weight_decay_s'])

    device_results = {}
    for device in config['devices']:
        if device == 'cuda' and not torch.cuda.is_available():
            device_results[device] = dict(available=False)
            continue
        if device == 'cuda':
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
                raise ValueError("Set the environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting "
                                "this process (PyTorch nn.LSTM docs); it cannot be set at runtime.")
            torch.use_deterministic_algorithms(True)
        seed_timings = []
        for seed in config['seeds']:
            t0 = time.time()
            _, _, conv, _ = fit_condition_v2(train, validation, vocab, fields, setting, seed, max_len, device)
            seed_timings.append(dict(seed=seed, seconds=time.time() - t0,
                                     validation_log_likelihood_bits=conv['best_validation_log_likelihood_bits']))
        determinism = None
        if device == 'cuda':
            _, _, conv_repeat, _ = fit_condition_v2(train, validation, vocab, fields, setting, config['seeds'][0], max_len, device)
            first_ll = seed_timings[0]['validation_log_likelihood_bits']
            determinism = dict(first=first_ll, repeat=conv_repeat['best_validation_log_likelihood_bits'],
                              exact_match=first_ll == conv_repeat['best_validation_log_likelihood_bits'])
        device_results[device] = dict(available=True, seed_timings=seed_timings, determinism=determinism)

    spread_check = None
    cpu, gpu = device_results.get('cpu'), device_results.get('cuda')
    if cpu and cpu.get('available') and gpu and gpu.get('available'):
        cpu_lls = [s['validation_log_likelihood_bits'] for s in cpu['seed_timings']]
        gpu_lls = [s['validation_log_likelihood_bits'] for s in gpu['seed_timings']]
        cpu_mean, gpu_mean = sum(cpu_lls) / len(cpu_lls), sum(gpu_lls) / len(gpu_lls)
        seed_spread = max(cpu_lls) - min(cpu_lls)
        spread_check = dict(cpu_mean=cpu_mean, gpu_mean=gpu_mean, difference=abs(cpu_mean - gpu_mean),
                           seed_spread=seed_spread, within_seed_spread=abs(cpu_mean - gpu_mean) < seed_spread)

    grid_n = len(config['dropout_grid']) * len(config['hidden_grid']) * len(config['weight_decay_history_grid'])
    grid_n_with_s = grid_n * len(config['weight_decay_s_grid'])
    n_seeds = len(config['full_study_seeds'])
    total_fits = config['full_study_folds'] * (3 * grid_n + 4 * grid_n_with_s) * n_seeds
    estimates = {}
    for device, result in device_results.items():
        if not result.get('available'):
            continue
        mean_seconds = sum(s['seconds'] for s in result['seed_timings']) / len(result['seed_timings'])
        workers = config.get('workers', 1) if device == 'cpu' else 1
        estimates[device] = dict(mean_seconds_per_fit=mean_seconds, workers=workers,
                                 estimated_hours=total_fits * mean_seconds / workers / 3600)

    write_timing_report(root, config, device_results, spread_check, total_fits, estimates)
    return dict(device_results=device_results, spread_check=spread_check, estimates=estimates)


def write_timing_report(root, config, device_results, spread_check, total_fits, estimates):
    lines = ['# RQ1 LSTM v2 timing and determinism check (DEC-17 stage 2)', '',
             f"Fold {config['test_day']}, condition fields {config['fields']}, the pilot's chosen setting "
             f"(dropout {config['dropout_rate']}, hidden {config['hidden_dim']}, weight decay history "
             f"{config['weight_decay_history']}, weight decay S {config['weight_decay_s']}), seeds "
             f"{config['seeds']}.", '']
    for device, result in device_results.items():
        if not result.get('available'):
            lines += [f"## {device}: not available", '']
            continue
        lines += [f'## {device}', '', '| Seed | Seconds | Validation LL bits/coda |', '|---:|---:|---:|']
        for s in result['seed_timings']:
            lines.append(f"| {s['seed']} | {s['seconds']:.2f} | {s['validation_log_likelihood_bits']:.5f} |")
        if result['determinism'] is not None:
            d = result['determinism']
            lines += ['', f"Determinism (seed {config['seeds'][0]} refit): first {d['first']:.5f}, repeat "
                     f"{d['repeat']:.5f}, exact match: {d['exact_match']}."]
        lines.append('')

    if spread_check is not None:
        lines += ['## CPU vs GPU spread check', '',
                 f"CPU mean validation LL {spread_check['cpu_mean']:.5f}, GPU mean {spread_check['gpu_mean']:.5f}, "
                 f"difference {spread_check['difference']:.5f}, spread across seeds {spread_check['seed_spread']:.5f}. "
                 f"Within seed spread: {spread_check['within_seed_spread']}.", '']

    lines += ['## Stage 3 wall time estimate', '',
             f"Full study: {config['full_study_folds']} folds x 8 conditions (3 without S x grid, 4 with S x "
             f"grid x S grid) x {len(config['full_study_seeds'])} seeds = {total_fits} fits.", '',
             '| Device | Seconds/fit | Workers | Estimated hours |', '|---|---:|---:|---:|']
    for device, e in estimates.items():
        lines.append(f"| {device} | {e['mean_seconds_per_fit']:.2f} | {e['workers']} | {e['estimated_hours']:.2f} |")
    lines += ['', 'Ideal scaling from one fold/condition/setting; excludes worker startup and variation across '
             'settings and folds.', '']

    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(lines))


def run(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    manifest_path = root / config['manifest']
    manifest = read_json(manifest_path)
    source = root / manifest['configuration']['input']
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Stale fold manifest')
    evaluation_config = read_json(root / config['evaluation_config'])
    if evaluation_config['seeds'] != config['seeds']:
        raise ValueError('Evaluation seed configuration mismatch')
    rows = read_jsonl(source)
    max_len = manifest['preparation_configuration']['history_limit']
    predictions, baselines, convergence = [], [], []
    for fold in manifest['folds']:
        mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
        if policy != fold['vocabulary_policy']:
            raise ValueError('Recomputed fold policy differs from manifest')
        train = [r for r in mapped if r['split'] == 'train' and r['comparison_eligible']]
        validation = [r for r in mapped if r['split'] == 'validation' and r['comparison_eligible']]
        test = [r for r in mapped if r['split'] == 'test' and r['comparison_eligible']]
        if not validation:
            raise ValueError('Fold has no validation comparison targets for early stopping')
        vocab, prior = policy['vocabulary'], policy['label_prior']
        rate = sum(r['Ct'] == r['A_history'][-1] for r in train) / len(train)
        majority = max(sorted(prior), key=prior.get)
        for row in test:
            p = repeat_distribution(row['A_history'][-1], rate, prior)
            baselines.append(dict(coda_id=row['coda_id'], fold_id=fold['fold_id'], recording_date=row['recording_date'],
                                  target=row['Ct'], majority=majority, repeat_probabilities=p,
                                  repeat_prediction=max(sorted(p), key=p.get),
                                  repeat_log_likelihood_bits=log_likelihood_bits(p[row['Ct']]), training_repeat_rate=rate))
        for condition, fields in manifest['input_conditions'].items():
            if condition == 'none':
                p = np.array([[prior[t] for t in vocab]] * len(test))
                for seed in config['seeds']:
                    for row, probs in zip(test, p):
                        predictions.append(dict(coda_id=row['coda_id'], fold_id=fold['fold_id'], condition=condition,
                                                seed=seed, probabilities=dict(zip(vocab, map(float, probs)))))
                continue
            for seed in config['seeds']:
                model, timing_encoder, conv = fit_condition(train, validation, vocab, fields, config, seed, max_len)
                p = predict(model, timing_encoder, test, vocab, fields, max_len)
                convergence.append(dict(fold_id=fold['fold_id'], condition=condition, seed=seed, **conv))
                for row, probs in zip(test, p):
                    predictions.append(dict(coda_id=row['coda_id'], fold_id=fold['fold_id'], condition=condition,
                                            seed=seed, probabilities=dict(zip(vocab, map(float, probs)))))
        print(f"Finished {fold['fold_id']}: {len(test)} test targets", flush=True)
    summary, scored, paired = evaluate(predictions, manifest, rows, config['seeds'], evaluation_config['bootstrap_replicates'],
                                       evaluation_config['bootstrap_seed'], evaluation_config['confidence'])
    baseline_metrics = {'majority': classification_metrics([r['target'] for r in baselines], [r['majority'] for r in baselines]),
                        'repeat_own_last': classification_metrics([r['target'] for r in baselines], [r['repeat_prediction'] for r in baselines])}
    baseline_metrics['repeat_own_last']['mean_log_likelihood_bits'] = sum(r['repeat_log_likelihood_bits'] for r in baselines) / len(baselines)
    indexed = {r['coda_id']: r for r in baselines}
    baseline_checks = {}
    for condition in ('A', 'A_B', 'A_S', 'A_B_S'):
        days = {}
        for r in scored:
            if r['condition'] == condition:
                days.setdefault(r['recording_date'], []).append(r['log_likelihood_bits'] - indexed[r['coda_id']]['repeat_log_likelihood_bits'])
        baseline_checks[condition] = day_bootstrap_bits(days, evaluation_config['bootstrap_replicates'], evaluation_config['bootstrap_seed'], evaluation_config['confidence'])

    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'predictions.jsonl', predictions)
    write_jsonl(output / 'baseline_predictions.jsonl', baselines)
    write_jsonl(output / 'convergence.jsonl', convergence)
    eval_output = output / 'evaluation'
    eval_output.mkdir(exist_ok=True)
    write_jsonl(eval_output / 'scored_predictions.jsonl', scored)
    write_jsonl(eval_output / 'paired_contrasts.jsonl', paired)
    summary.update(configuration=config, baseline_metrics=baseline_metrics, repeat_baseline_checks=baseline_checks,
                   versions={'numpy': np.__version__, 'torch': torch.__version__},
                   input_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, manifest_path, path)})
    write_text(eval_output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')

    stop_reasons = check_stop_conditions(summary, baseline_metrics, baseline_checks)
    logistic_summary = read_json(root / config['compare_against_logistic']) if config.get('compare_against_logistic') else None
    underperform = check_against_logistic(scored, logistic_summary, root, evaluation_config) if logistic_summary else []
    write_report(root, config, summary, baseline_metrics, baseline_checks, convergence, stop_reasons, underperform, logistic_summary)
    return summary, stop_reasons, underperform


def check_against_logistic(lstm_scored, logistic_summary, root, evaluation_config):
    """PLAN [U1]: the LSTM must not do worse than logistic regression on the
    same condition. Averages the LSTM's per-coda log-likelihood across its
    three seeds first (logistic has only one), then pairs per coda and
    day-bootstraps, same as every other contrast in this study."""
    logistic_scored = {(r['coda_id'], r['condition']): r['log_likelihood_bits']
                       for r in read_jsonl(root / logistic_summary['configuration']['output_directory'] / 'evaluation' / 'scored_predictions.jsonl')}
    grouped = defaultdict(list)
    for r in lstm_scored:
        grouped[r['condition'], r['coda_id']].append(r)
    reasons = []
    for condition in {c for c, _ in grouped} - {'none'}:
        days = {}
        for (cond, coda_id), rs in grouped.items():
            if cond != condition:
                continue
            key = (coda_id, condition)
            if key not in logistic_scored:
                continue
            lstm_ll = sum(r['log_likelihood_bits'] for r in rs) / len(rs)
            days.setdefault(rs[0]['recording_date'], []).append(lstm_ll - logistic_scored[key])
        if not days:
            continue
        check = day_bootstrap_bits(days, evaluation_config['bootstrap_replicates'], evaluation_config['bootstrap_seed'], evaluation_config['confidence'])
        if check['ci_bits_per_coda'][1] < 0:
            reasons.append(f"{condition}: LSTM reliably below logistic regression (CI {format_ci(check['ci_bits_per_coda'])})")
    return reasons


def _fit_lstm_v2(rows, manifest, folds, conditions, config, device):
    """Shared fold/condition fitting loop for both run_lstm_v2 (full study,
    all 8 conditions) and run_lstm_v2_pilot (2 conditions, 4 folds). Returns
    predictions, baselines, models, selections, curves and a coda_id -> mapped
    target dict (the last is only needed by the pilot's simpler scoring,
    since evaluate() derives targets itself for the full run)."""
    max_len = manifest['preparation_configuration']['history_limit']
    predictions, baselines, models, selections, curves, targets = [], [], [], [], [], {}
    for fold in folds:
        mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
        if policy != fold['vocabulary_policy']:
            raise ValueError('Recomputed fold policy differs from manifest')
        train = [r for r in mapped if r['split'] == 'train' and r['comparison_eligible']]
        validation = [r for r in mapped if r['split'] == 'validation' and r['comparison_eligible']]
        test = [r for r in mapped if r['split'] == 'test' and r['comparison_eligible']]
        if not validation:
            raise ValueError('Fold has no validation comparison targets for hyperparameter selection')
        if {r['coda_id'] for r in validation} & {r['coda_id'] for r in test}:
            raise ValueError('Validation and test rows overlap')
        vocab, prior = policy['vocabulary'], policy['label_prior']
        rate = sum(r['Ct'] == r['A_history'][-1] for r in train) / len(train)
        majority = max(sorted(prior), key=prior.get)
        for row in test:
            targets[row['coda_id']] = row['Ct']
            p = repeat_distribution(row['A_history'][-1], rate, prior)
            baselines.append(dict(coda_id=row['coda_id'], fold_id=fold['fold_id'], recording_date=row['recording_date'],
                                  target=row['Ct'], majority=majority, repeat_probabilities=p,
                                  repeat_prediction=max(sorted(p), key=p.get),
                                  repeat_log_likelihood_bits=log_likelihood_bits(p[row['Ct']]), training_repeat_rate=rate))
        for condition, fields in conditions.items():
            if condition == 'none':
                p = np.array([[prior[t] for t in vocab]] * len(test))
                for seed in config['seeds']:
                    for row, probs in zip(test, p):
                        predictions.append(dict(coda_id=row['coda_id'], fold_id=fold['fold_id'], condition=condition,
                                                seed=seed, probabilities=dict(zip(vocab, map(float, probs)))))
                continue
            chosen, chosen_fits, grid = select_lstm_grid_and_fit(train, validation, vocab, fields, config,
                                                                 config['seeds'], max_len, curves, device,
                                                                 fold_id=fold['fold_id'], condition=condition,
                                                                 workers=config.get('workers', 1))
            for seed, (model, timing_encoder, conv) in zip(config['seeds'], chosen_fits):
                p = predict_v2(model, timing_encoder, test, vocab, fields, max_len, device)
                models.append(dict(fold_id=fold['fold_id'], condition=condition, seed=seed,
                                   dropout=chosen['dropout'], hidden=chosen['hidden'],
                                   weight_decay_history=chosen['weight_decay_history'], weight_decay_s=chosen['weight_decay_s'],
                                   beta=model.beta.item() if hasattr(model, 'beta') else None,
                                   gamma=model.gamma.item() if hasattr(model, 'gamma') else None,
                                   epochs_trained=conv['epochs_trained'], early_stopped=conv['early_stopped']))
                for row, probs in zip(test, p):
                    predictions.append(dict(coda_id=row['coda_id'], fold_id=fold['fold_id'], condition=condition,
                                            seed=seed, probabilities=dict(zip(vocab, map(float, probs)))))
            selections.append(dict(fold_id=fold['fold_id'], condition=condition, chosen={k: v for k, v in chosen.items() if k != 'convergence'},
                                   grid=[{k: v for k, v in g.items()} for g in grid]))
        print(f"Finished {fold['fold_id']}: {len(test)} test targets", flush=True)
    return predictions, baselines, models, selections, curves, targets


def _write_lstm_v2_artifacts(root, config, source, manifest_path, path, predictions, baselines, models, selections, curves, scored, summary):
    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'predictions.jsonl', predictions)
    write_jsonl(output / 'baseline_predictions.jsonl', baselines)
    write_jsonl(output / 'models.jsonl', models)
    write_jsonl(output / 'lstm_selection.jsonl', selections)
    write_jsonl(output / 'training_curves.jsonl', curves)
    eval_output = output / 'evaluation'
    eval_output.mkdir(exist_ok=True)
    write_jsonl(eval_output / 'scored_predictions.jsonl', scored)
    summary.update(configuration=config,
                   versions={'numpy': np.__version__, 'torch': torch.__version__},
                   input_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, manifest_path, path)})
    write_text(eval_output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return eval_output


def _load_lstm_v2_config(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    manifest_path = root / config['manifest']
    manifest = read_json(manifest_path)
    source = root / manifest['configuration']['input']
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Stale fold manifest')
    evaluation_config = read_json(root / config['evaluation_config'])
    if evaluation_config['seeds'] != config['seeds']:
        raise ValueError('Evaluation seed configuration mismatch')
    return path, root, config, manifest, manifest_path, source, evaluation_config


def run_lstm_v2(config_path):
    """LSTM v2 (DEC-17, DEC-20), full study: all 8 conditions, all 14 folds.
    Requires the complete condition lattice, since evaluate() and
    check_stop_conditions both assume it; use run_lstm_v2_pilot for a
    restricted subset. Keeps LSTM v1 (run(), above) entirely untouched."""
    path, root, config, manifest, manifest_path, source, evaluation_config = _load_lstm_v2_config(config_path)
    if config.get('test_days') or config.get('conditions'):
        raise ValueError('run_lstm_v2 requires the full lattice; use run_lstm_v2_pilot for a restricted subset')
    device = config.get('device', 'cpu')
    rows = read_jsonl(source)
    conditions = manifest['input_conditions']
    predictions, baselines, models, selections, curves, _ = _fit_lstm_v2(rows, manifest, manifest['folds'], conditions, config, device)
    summary, scored, paired = evaluate(predictions, manifest, rows, config['seeds'], evaluation_config['bootstrap_replicates'],
                                       evaluation_config['bootstrap_seed'], evaluation_config['confidence'])
    baseline_metrics = {'majority': classification_metrics([r['target'] for r in baselines], [r['majority'] for r in baselines]),
                        'repeat_own_last': classification_metrics([r['target'] for r in baselines], [r['repeat_prediction'] for r in baselines])}
    baseline_metrics['repeat_own_last']['mean_log_likelihood_bits'] = sum(r['repeat_log_likelihood_bits'] for r in baselines) / len(baselines)
    indexed = {r['coda_id']: r for r in baselines}
    baseline_checks = {}
    for condition in ('A', 'A_B', 'A_S', 'A_B_S'):
        days = {}
        for r in scored:
            if r['condition'] == condition:
                days.setdefault(r['recording_date'], []).append(r['log_likelihood_bits'] - indexed[r['coda_id']]['repeat_log_likelihood_bits'])
        baseline_checks[condition] = day_bootstrap_bits(days, evaluation_config['bootstrap_replicates'], evaluation_config['bootstrap_seed'], evaluation_config['confidence'])

    summary.update(baseline_metrics=baseline_metrics, repeat_baseline_checks=baseline_checks)
    eval_output = _write_lstm_v2_artifacts(root, config, source, manifest_path, path, predictions, baselines, models, selections, curves, scored, summary)
    write_jsonl(eval_output / 'paired_contrasts.jsonl', paired)

    stop_reasons = check_stop_conditions(summary, baseline_metrics, baseline_checks)
    logistic_summary = read_json(root / config['compare_against_logistic']) if config.get('compare_against_logistic') else None
    underperform = check_against_logistic(scored, logistic_summary, root, evaluation_config) if logistic_summary else []
    write_lstm_v2_report(root, config, summary, baseline_metrics, baseline_checks, models, stop_reasons, underperform, logistic_summary)
    return summary, stop_reasons, underperform


def run_lstm_v2_pilot(config_path):
    """LSTM v2 pilot (DEC-17): conditions and test_days are required in
    config and restrict the fitting loop (e.g. A and A_B_S on the 3 largest
    test days plus one small one). Scores directly rather than through
    evaluate(), since evaluate()'s contrasts hardcode the full 8-condition
    lattice, which the pilot deliberately does not have; there is no
    add-one-input lattice or repeat/switch split here, only per-condition
    metrics, the A-vs-repeat-own-last check for whichever A-conditions are
    present, and the [U1] comparison to logistic regression."""
    path, root, config, manifest, manifest_path, source, evaluation_config = _load_lstm_v2_config(config_path)
    if not config.get('test_days') or not config.get('conditions'):
        raise ValueError("run_lstm_v2_pilot requires 'test_days' and 'conditions' in config")
    device = config.get('device', 'cpu')
    rows = read_jsonl(source)
    folds = [f for f in manifest['folds'] if f['test_day'] in config['test_days']]
    if len(folds) != len(config['test_days']):
        raise ValueError('Not every requested pilot test_day exists in the fold manifest')
    conditions = {c: manifest['input_conditions'][c] for c in config['conditions']}
    predictions, baselines, models, selections, curves, targets = _fit_lstm_v2(rows, manifest, folds, conditions, config, device)

    scored = []
    for p in predictions:
        target = targets[p['coda_id']]
        probs = p['probabilities']
        scored.append(dict(coda_id=p['coda_id'], fold_id=p['fold_id'], condition=p['condition'], seed=p['seed'],
                           target=target, predicted=max(sorted(probs), key=probs.get),
                           log_likelihood_bits=log_likelihood_bits(probs[target]),
                           recording_date=next(f['test_day'] for f in folds if f['fold_id'] == p['fold_id'])))
    metrics = {}
    for condition in conditions:
        per_seed = []
        for seed in config['seeds']:
            rs = [r for r in scored if r['condition'] == condition and r['seed'] == seed]
            per_seed.append(dict(classification_metrics([r['target'] for r in rs], [r['predicted'] for r in rs]),
                                 mean_log_likelihood_bits=sum(r['log_likelihood_bits'] for r in rs) / len(rs)))
        metrics[condition] = {f: sum(m[f] for m in per_seed) / len(per_seed)
                              for f in ('accuracy', 'macro_f1', 'mean_log_likelihood_bits')}
    baseline_metrics = {'majority': classification_metrics([r['target'] for r in baselines], [r['majority'] for r in baselines]),
                        'repeat_own_last': classification_metrics([r['target'] for r in baselines], [r['repeat_prediction'] for r in baselines])}
    baseline_metrics['repeat_own_last']['mean_log_likelihood_bits'] = sum(r['repeat_log_likelihood_bits'] for r in baselines) / len(baselines)
    indexed = {r['coda_id']: r for r in baselines}
    baseline_checks = {}
    for condition in {'A', 'A_B', 'A_S', 'A_B_S'} & set(conditions):
        days = {}
        for r in scored:
            if r['condition'] == condition:
                days.setdefault(r['recording_date'], []).append(r['log_likelihood_bits'] - indexed[r['coda_id']]['repeat_log_likelihood_bits'])
        baseline_checks[condition] = day_bootstrap_bits(days, evaluation_config['bootstrap_replicates'], evaluation_config['bootstrap_seed'], evaluation_config['confidence'])

    summary = dict(units='bits per coda', seeds=config['seeds'], metrics=metrics,
                   baseline_metrics=baseline_metrics, repeat_baseline_checks=baseline_checks)
    _write_lstm_v2_artifacts(root, config, source, manifest_path, path, predictions, baselines, models, selections, curves, scored, summary)

    stop_reasons = [f"{c}: reliably below repeat-own-last in log-likelihood (CI {format_ci(check['ci_bits_per_coda'])})"
                   for c, check in baseline_checks.items() if check['ci_bits_per_coda'][1] < 0]
    logistic_summary = read_json(root / config['compare_against_logistic']) if config.get('compare_against_logistic') else None
    underperform = check_against_logistic(scored, logistic_summary, root, evaluation_config) if logistic_summary else []
    write_lstm_v2_pilot_report(root, config, summary, baseline_metrics, baseline_checks, models, curves, stop_reasons, underperform, logistic_summary)
    return summary, stop_reasons, underperform


def write_lstm_v2_report(root, config, summary, baseline_metrics, baseline_checks, models, stop_reasons, underperform, logistic_summary):
    conditions = list(summary['metrics'])
    lines = ['# RQ1 LSTM v2 (copy weights, separate history/S regularisation)', '',
             f"One LSTM encoder per enabled history stream (embedding {config['embedding_dim']}) and a "
             "feed-forward encoder for S, each with dropout on its output, summed through separate linear "
             "projections into a shared vocabulary-sized output (mathematically identical to one projection "
             "over the concatenation, but split for weight decay). Copy weights beta/gamma as in logistic v4: a "
             "shared scalar boost to the logit of the class matching the own/partner last coda. Adam, learning "
             f"rate {config['learning_rate']}, batch {config['batch_size']}, up to {config['max_epochs']} epochs, "
             f"patience {config['patience']}, restoring the best validation epoch. Per fold and condition, dropout "
             f"{config['dropout_grid']} x hidden size {config['hidden_grid']} x history weight decay "
             f"{config['weight_decay_history_grid']}, and where S is present, a separate S weight decay "
             f"{config['weight_decay_s_grid']}, are chosen jointly on the validation day by mean validation "
             "log-likelihood across seeds; ties go to the smaller, more regularised setting. Device: "
             f"{config.get('device', 'cpu')}.", '']

    lines += ['## Metrics', '', '| Condition | LL bits/coda | Accuracy | Macro-F1 |', '|---|---:|---:|---:|']
    for c, m in summary['metrics'].items():
        lines.append(f"| {c} | {m['mean_log_likelihood_bits']:.5f} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} |")
    for c, m in baseline_metrics.items():
        ll = f"{m['mean_log_likelihood_bits']:.5f}" if 'mean_log_likelihood_bits' in m else 'n/a'
        lines.append(f"| {c} | {ll} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} |")

    if baseline_checks:
        lines += ['', '## A-conditions vs repeat-own-last', '',
                 '| Condition | LL contrast | CI | Positive days | Accuracy | Macro-F1 |', '|---|---:|---|---:|---:|---:|']
        for c, check in baseline_checks.items():
            lines.append(f"| {c} | {check['mean_bits_per_coda']:.5f} | {format_ci(check['ci_bits_per_coda'])} | "
                         f"{check['positive_days']}/{check['days']} | {summary['metrics'][c]['accuracy']:.3%} | "
                         f"{summary['metrics'][c]['macro_f1']:.4f} |")

    if any(k.startswith('add_input:') for k in summary['contrasts']):
        lines += ['', '## Add-one-input contrasts', '',
                 '| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |', '|---|---:|---:|---|---:|']
        for d in classify_add_input_contrasts(summary):
            lines.append(f"| adding {d['added']} to {describe(d['base'])} | {d['mean']:.5f} | {d['unweighted_mean']:.5f} | "
                         f"{format_ci([d['low'], d['high']])} | {d['positive_days']}/{d['days']} |")

    if logistic_summary is not None:
        lines += ['', '## LSTM v2 vs logistic regression, by condition [U1]', '',
                 '| Condition | Logistic LL bits/coda | LSTM v2 LL bits/coda | Logistic Accuracy | LSTM v2 Accuracy |',
                 '|---|---:|---:|---:|---:|']
        for c in conditions:
            if c == 'none':
                continue
            lm, nm = logistic_summary['metrics'][c], summary['metrics'][c]
            lines.append(f"| {c} | {lm['mean_log_likelihood_bits']:.5f} | {nm['mean_log_likelihood_bits']:.5f} | "
                         f"{lm['accuracy']:.3%} | {nm['accuracy']:.3%} |")

    lines += ['', '## Chosen setting and fitted beta/gamma per condition', '',
             '| Condition | Dropout | Hidden | WD history | WD S | Beta range | Gamma range | Epochs trained | Hit epoch cap |',
             '|---|---:|---:|---:|---:|---|---|---|---|']
    for c in conditions:
        if c == 'none':
            continue
        rows = [m for m in models if m['condition'] == c]
        dropouts, hiddens = sorted({r['dropout'] for r in rows}), sorted({r['hidden'] for r in rows})
        wdh, wds = sorted({r['weight_decay_history'] for r in rows}), sorted({r['weight_decay_s'] for r in rows if r['weight_decay_s'] is not None})
        betas = [r['beta'] for r in rows if r['beta'] is not None]
        gammas = [r['gamma'] for r in rows if r['gamma'] is not None]
        beta_range = f"{min(betas):.4f} to {max(betas):.4f}" if betas else 'n/a'
        gamma_range = f"{min(gammas):.4f} to {max(gammas):.4f}" if gammas else 'n/a'
        epochs = [r['epochs_trained'] for r in rows]
        hit_cap = sum(e == config['max_epochs'] for e in epochs)
        lines.append(f"| {c} | {dropouts} | {hiddens} | {wdh} | {wds or 'n/a'} | {beta_range} | {gamma_range} | "
                     f"{min(epochs)}-{max(epochs)} | {hit_cap}/{len(epochs)} |")

    all_reasons = list(stop_reasons) + underperform
    if all_reasons:
        lines += ['', '## Stop: results need review before interpretation', '',
                 'The following stop conditions were met. No interpretation or conclusions are written below; '
                 'the tables above are the complete result.', '']
        lines += [f"- {reason}" for reason in all_reasons]
        lines += ['']
    else:
        lines += ['', '## Result', '',
                 'No A-condition falls below repeat-own-last in log-likelihood, no add-one-input contrast has a '
                 'confidence interval entirely below zero, and the LSTM does not fall reliably below logistic '
                 'regression on any condition.', '']

    lines += ['Per-epoch training and validation log-likelihood for every grid-search fit are in '
             'training_curves.jsonl; the full grid (every setting tried, not only the chosen one) is in '
             'lstm_selection.jsonl. Model weights are not saved. Source, fold-manifest and configuration '
             'SHA-256 hashes are recorded in evaluation/summary.json.', '']

    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(lines))


def write_lstm_v2_pilot_report(root, config, summary, baseline_metrics, baseline_checks, models, curves, stop_reasons, underperform, logistic_summary):
    """DEC-17 gate 2: conditions A and A_B_S only, on the pilot's named test
    days, scored directly (no add-one-input lattice, no repeat/switch split,
    since the pilot deliberately has neither the full condition set nor
    enough days for either to be meaningful)."""
    conditions = list(summary['metrics'])
    lines = ['# RQ1 LSTM v2 pilot (DEC-17)', '',
             f"Conditions {list(config['conditions'])} on test days {config['test_days']}, seed(s) "
             f"{config['seeds']}, full grid: dropout {config['dropout_grid']} x hidden {config['hidden_grid']} x "
             f"history weight decay {config['weight_decay_history_grid']}, and where S is present, S weight decay "
             f"{config['weight_decay_s_grid']}. Same architecture and optimiser as the eventual full run (Adam, "
             f"lr {config['learning_rate']}, batch {config['batch_size']}, up to {config['max_epochs']} epochs, "
             f"patience {config['patience']}). Device: {config.get('device', 'cpu')}. This checks the mechanism "
             "works before spending the full study's compute; it is not evaluated against repeat-own-last or "
             "logistic regression as a final result, only as a gate.", '']

    lines += ['## Metrics', '', '| Condition | LL bits/coda | Accuracy | Macro-F1 |', '|---|---:|---:|---:|']
    for c, m in summary['metrics'].items():
        lines.append(f"| {c} | {m['mean_log_likelihood_bits']:.5f} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} |")
    for c, m in baseline_metrics.items():
        ll = f"{m['mean_log_likelihood_bits']:.5f}" if 'mean_log_likelihood_bits' in m else 'n/a'
        lines.append(f"| {c} | {ll} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} |")

    lines += ['', '## vs repeat-own-last, over the pilot folds only', '',
             '| Condition | LL contrast | CI | Positive days | Accuracy | Macro-F1 |', '|---|---:|---|---:|---:|---:|']
    for c, check in baseline_checks.items():
        lines.append(f"| {c} | {check['mean_bits_per_coda']:.5f} | {format_ci(check['ci_bits_per_coda'])} | "
                     f"{check['positive_days']}/{check['days']} | {summary['metrics'][c]['accuracy']:.3%} | "
                     f"{summary['metrics'][c]['macro_f1']:.4f} |")

    if logistic_summary is not None:
        lines += ['', '## LSTM v2 pilot vs logistic regression, by condition [U1]', '',
                 '| Condition | Logistic LL bits/coda | LSTM v2 LL bits/coda | Logistic Accuracy | LSTM v2 Accuracy |',
                 '|---|---:|---:|---:|---:|']
        for c in conditions:
            lm, nm = logistic_summary['metrics'][c], summary['metrics'][c]
            lines.append(f"| {c} | {lm['mean_log_likelihood_bits']:.5f} | {nm['mean_log_likelihood_bits']:.5f} | "
                         f"{lm['accuracy']:.3%} | {nm['accuracy']:.3%} |")

    lines += ['', '## Chosen setting and fitted beta/gamma per condition', '',
             '| Condition | Dropout | Hidden | WD history | WD S | Beta range | Gamma range | Epochs trained | Hit epoch cap |',
             '|---|---:|---:|---:|---:|---|---|---|---|']
    for c in conditions:
        rows = [m for m in models if m['condition'] == c]
        dropouts, hiddens = sorted({r['dropout'] for r in rows}), sorted({r['hidden'] for r in rows})
        wdh = sorted({r['weight_decay_history'] for r in rows})
        wds = sorted({r['weight_decay_s'] for r in rows if r['weight_decay_s'] is not None})
        betas = [r['beta'] for r in rows if r['beta'] is not None]
        gammas = [r['gamma'] for r in rows if r['gamma'] is not None]
        beta_range = f"{min(betas):.4f} to {max(betas):.4f}" if betas else 'n/a'
        gamma_range = f"{min(gammas):.4f} to {max(gammas):.4f}" if gammas else 'n/a'
        epochs = [r['epochs_trained'] for r in rows]
        hit_cap = sum(e == config['max_epochs'] for e in epochs)
        lines.append(f"| {c} | {dropouts} | {hiddens} | {wdh} | {wds or 'n/a'} | {beta_range} | {gamma_range} | "
                     f"{min(epochs)}-{max(epochs)} | {hit_cap}/{len(epochs)} |")

    lines += ['', '## Learning curve, one fold/condition/setting (train vs validation log-likelihood)', '']
    if curves and models:
        sample_condition, sample_fold_id, sample_seed = models[0]['condition'], models[0]['fold_id'], models[0]['seed']
        sample_setting = next((m for m in models if m['fold_id'] == sample_fold_id and m['condition'] == sample_condition), None)
        matching = [c for c in curves if c['fold_id'] == sample_fold_id and c['condition'] == sample_condition
                   and c['dropout'] == sample_setting['dropout'] and c['hidden'] == sample_setting['hidden']
                   and c['weight_decay_history'] == sample_setting['weight_decay_history']
                   and c['weight_decay_s'] == sample_setting['weight_decay_s'] and c['seed'] == sample_seed]
        lines += [f"Fold {sample_fold_id}, condition {sample_condition}, seed {sample_seed}, the chosen setting "
                 f"(dropout {sample_setting['dropout']}, hidden {sample_setting['hidden']}, weight decay history "
                 f"{sample_setting['weight_decay_history']}, weight decay S {sample_setting['weight_decay_s']}):", '',
                 '| Epoch | Train LL bits/coda | Validation LL bits/coda |', '|---:|---:|---:|']
        step = max(1, len(matching) // 20)
        for row in matching[::step]:
            lines.append(f"| {row['epoch']} | {row['train_log_likelihood_bits']:.5f} | {row['validation_log_likelihood_bits']:.5f} |")
    else:
        lines.append('No fits recorded.')

    all_reasons = list(stop_reasons) + underperform
    if all_reasons:
        lines += ['', '## Gate 2: does not hold', '',
                 'The following stop conditions were met. No interpretation or conclusions are written below; '
                 'the tables above are the complete result. Per the plan, settings are not retuned in response '
                 'to this: the pilot folds are test days.', '']
        lines += [f"- {reason}" for reason in all_reasons]
        lines += ['']
    else:
        lines += ['', '## Gate 2: holds so far', '',
                 'Over the pilot folds, no A-condition falls below repeat-own-last in log-likelihood, and the '
                 'LSTM does not fall reliably below logistic regression on either condition. See the plan for '
                 'the remaining gate 2 checks (no fit at the epoch cap, GPU determinism) not computed here.', '']

    lines += ['Per-epoch training and validation log-likelihood for every grid-search fit (not only the chosen '
             'setting) are in training_curves.jsonl; the full grid is in lstm_selection.jsonl. Model weights are '
             'not saved. Source, fold-manifest and configuration SHA-256 hashes are recorded in '
             'evaluation/summary.json.', '']

    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(lines))


def write_report(root, config, summary, baseline_metrics, baseline_checks, convergence, stop_reasons, underperform, logistic_summary):
    lines = ['# RQ1 LSTM main study', '',
             f"One LSTM encoder per enabled history stream (embedding {config['embedding_dim']}, hidden "
             f"{config['hidden_dim']}) and a feed-forward encoder for S (hidden {config['hidden_dim']}), "
             "concatenated into a per-condition output layer; an absent input has no encoder at all. Adam, "
             f"learning rate {config['learning_rate']}, weight decay {config['weight_decay']}, batch "
             f"{config['batch_size']}, early stopping on validation-day log-likelihood with patience "
             f"{config['patience']} and a cap of {config['max_epochs']} epochs. Identical settings across all "
             f"conditions and folds; only which encoders exist differs. Seeds {config['seeds']} are averaged "
             "per coda before contrasts, by the shared evaluation module.", '']

    lines += ['## Metrics', '', '| Condition | LL bits/coda | Accuracy | Macro-F1 |', '|---|---:|---:|---:|']
    for c, m in summary['metrics'].items():
        lines.append(f"| {c} | {m['mean_log_likelihood_bits']:.5f} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} |")
    for c, m in baseline_metrics.items():
        ll = f"{m['mean_log_likelihood_bits']:.5f}" if 'mean_log_likelihood_bits' in m else 'n/a'
        lines.append(f"| {c} | {ll} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} |")

    lines += ['', '## A-conditions vs repeat-own-last', '',
             '| Condition | LL contrast | CI | Positive days | Accuracy | Macro-F1 |', '|---|---:|---|---:|---:|---:|']
    for c in ('A', 'A_B', 'A_S', 'A_B_S'):
        check = baseline_checks[c]
        lines.append(f"| {c} | {check['mean_bits_per_coda']:.5f} | {check['ci_bits_per_coda']} | "
                     f"{check['positive_days']}/{check['days']} | {summary['metrics'][c]['accuracy']:.3%} | "
                     f"{summary['metrics'][c]['macro_f1']:.4f} |")

    lines += ['', '## All twelve add-one-input contrasts', '',
             '| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |', '|---|---:|---:|---|---:|']
    for d in classify_add_input_contrasts(summary):
        lines.append(f"| adding {d['added']} to {describe(d['base'])} | {d['mean']:.5f} | {d['unweighted_mean']:.5f} | "
                     f"[{d['low']:.5f}, {d['high']:.5f}] | {d['positive_days']}/{d['days']} |")

    if logistic_summary is not None:
        lines += ['', '## LSTM vs logistic regression (v2), by condition [U1]', '',
                 '| Condition | Logistic LL bits/coda | LSTM LL bits/coda | Logistic Accuracy | LSTM Accuracy |',
                 '|---|---:|---:|---:|---:|']
        for c in summary['metrics']:
            if c == 'none':
                continue
            lm, nm = logistic_summary['metrics'][c], summary['metrics'][c]
            lines.append(f"| {c} | {lm['mean_log_likelihood_bits']:.5f} | {nm['mean_log_likelihood_bits']:.5f} | "
                         f"{lm['accuracy']:.3%} | {nm['accuracy']:.3%} |")

    epochs = [c['epochs_trained'] for c in convergence]
    lines += ['', f"{len(convergence)} model fits (14 folds times 7 non-unigram conditions times "
             f"{len(config['seeds'])} seeds); epochs trained ranged {min(epochs)} to {max(epochs)}, "
             f"{sum(c['early_stopped'] for c in convergence)} stopped early on patience, the rest hit the "
             f"{config['max_epochs']}-epoch cap. Full per-run convergence is in convergence.jsonl.", '']

    all_reasons = list(stop_reasons) + underperform
    if all_reasons:
        lines += ['## Stop: results need review before interpretation', '',
                 'The following stop conditions were met. No interpretation or conclusions are written below; '
                 'the tables above are the complete result.', '']
        lines += [f"- {reason}" for reason in all_reasons]
        lines += ['']
    else:
        lines += ['## Result', '',
                 'No A-condition falls below repeat-own-last in log-likelihood or accuracy, no add-one-input '
                 'contrast has a confidence interval entirely below zero, and the LSTM does not fall reliably '
                 'below logistic regression on any condition.', '']

    lines += ['Statistical uncertainty and restricted training data limit inference from this model; this report '
             'does not establish grammar or causal mechanisms. Day-bootstrap intervals use '
             f"{config.get('bootstrap_replicates', 10000)} replicates and are approximate with only 14 days. The "
             'switch-choice component of every contrast rests on about 270 non-repeat codas and is exploratory. '
             'Model weights are not saved (parameter count makes this impractical to keep in version control); '
             'convergence.jsonl records per-run epoch counts and best validation log-likelihood, sufficient to '
             'audit training without re-fitting. Source, fold-manifest and configuration SHA-256 hashes are '
             'recorded in evaluation/summary.json.', '']

    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/rq1_lstm.json')
    args = parser.parse_args()
    config = read_json(Path(args.config).resolve())
    if 'devices' in config:
        run_timing_determinism_check(args.config)
        return
    if 'weight_decay_s' in config and 'fields_with_s' in config:
        run_suppression_diagnostic(args.config)
        return
    if config.get('test_days') or config.get('conditions'):
        _, stop_reasons, underperform = run_lstm_v2_pilot(args.config)
    elif 'dropout_grid' in config:
        _, stop_reasons, underperform = run_lstm_v2(args.config)
    else:
        _, stop_reasons, underperform = run(args.config)
    total = len(stop_reasons) + len(underperform)
    if total:
        print(f"STOP: {total} condition(s) need review before interpretation. See the report.")


if __name__ == '__main__':
    main()
