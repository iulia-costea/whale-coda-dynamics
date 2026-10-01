"""Residual LSTM on top of a frozen logistic v4 (DEC-21, PLAN [M4]).

Per fold and condition c the network's logits are

    z(x) = z_LR_c(x) + sum over the condition's blocks s of U_s h_s(x).

z_LR_c are logistic v4's logits for that fold and condition, recomputed from its
committed fits with logistic.py's own feature encoder, and added as a fixed
offset that never trains. h_s is block s's encoder (one LSTM per coda history,
neural.HistoryEncoder; a feed-forward encoder for S, neural.TimingEncoder) and
U_s its bias-free output projection, initialised at zero. There is no output
bias and there are no copy weights outside the offset. A block is either on or
absent; an absent block is not built, so with every block absent the model is
logistic v4 exactly.

Selection uses the validation day only. Every candidate (hidden size x dropout x
which blocks are on, plus logistic v4 itself) is trained with every seed, the
per-target validation log-likelihoods are averaged over the seeds, and the
simplest candidate within one standard error of the best is chosen. Test rows
are never passed to anything that trains or selects; only the chosen
candidate's restored models are scored on them.

Three entry points, chosen by the config (see main()):
- the checks before the full run on training and validation days (PLAN [M4]
  checks 1-4: wiring, learnability, planted signal);
- the device and throughput measurement;
- the full run over all conditions, folds and seeds, with its report.
"""
import argparse
import concurrent.futures
import contextlib
import copy
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.special import logsumexp
from torch import nn

from .evaluation import evaluate, classification_metrics, log_likelihood_bits, day_bootstrap_bits
from .logistic import (Encoder, _copy_logits, last_label_index, repeat_distribution, select_l2_and_fit,
                       check_stop_conditions, classify_add_input_contrasts, describe, format_ci)
from .neural import encode_history, HistoryEncoder, TimingEncoder, LOG2E
from .neural_reproduction import TARGET_PAIRED_CODA_TYPE_ACCURACY
from .preprocessing.folds import prepare_fold
from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text

BLOCKS = ('A', 'B', 'S')
HISTORY_FIELDS = {'A': 'A_history', 'B': 'B_history'}
LR_ONLY = 'LR'
MODEL_KEYS = ('seeds', 'embedding_dim', 'learning_rate', 'weight_decay', 'batch_size', 'max_epochs', 'patience',
              'hidden_grid', 'dropout_grid')
# nn.LSTM's cuDNN determinism note allows :16:8 and :4096:2; the reproducibility page allows :16:8 and :4096:8.
CUBLAS_SETTINGS = (':16:8', ':4096:2', ':4096:8')

# Assayag et al. (2026), Bioacoustics 35(4):472-500, Table 3 (p. 494), LSTM "Coda Type (Accuracy)":
# single-whale self-prediction, columns t0..t-8 (t0 = current coda only). Random 70/15/15 split, no baseline.
TARGET_SINGLE_WHALE_CODA_TYPE_ACCURACY = [0.800, 0.824, 0.821, 0.889, 0.859, 0.886, 0.922, 0.746, 0.845]


# ---------------------------------------------------------------- offsets

def softmax_rows(logits):
    """Exactly logistic.probabilities' normalisation, so the offset alone reproduces v4 bit for bit."""
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def load_v4_models(path):
    models = {}
    for record in read_jsonl(path):
        key = (record['fold_id'], record['condition'])
        if key in models:
            raise ValueError(f'Duplicate logistic v4 model {key}')
        models[key] = record
    return models


def v4_logits(record, train, rows, vocab):
    """Logistic v4's logits for `rows`, from its stored fit and logistic.py's own feature encoder.

    The encoder is refitted on the fold's training comparison rows, exactly as v4 fitted it. Its
    feature names and categories must equal the stored ones, and its standardisation must agree with
    the stored one to 1e-12 (numeric libraries differ in the last bit across machines); the stored
    standardisation is then used, so the offsets are computed exactly as v4 computed its logits."""
    if record['vocabulary'] != vocab:
        raise ValueError('Logistic v4 vocabulary differs from the fold vocabulary')
    fields = record['fields']
    encoder = Encoder(train, vocab, fields, record['order'])
    mean, scale = np.array(record['mean'], dtype=float), np.array(record['scale'], dtype=float)
    if (encoder.names + ['intercept'] != record['feature_names'] or encoder.categories != record['categories']
            or encoder.mean.shape != mean.shape or encoder.scale.shape != scale.shape
            or not np.allclose(encoder.mean, mean, rtol=0, atol=1e-12)
            or not np.allclose(encoder.scale, scale, rtol=0, atol=1e-12)):
        raise ValueError(f"Recomputed v4 encoder differs from the stored one ({record['fold_id']}, {record['condition']})")
    encoder.mean, encoder.scale = mean, scale
    if (record['beta'] is None) == ('A_history' in fields) or (record['gamma'] is None) == ('B_history' in fields):
        raise ValueError('Stored copy weights do not match the condition fields')
    own = last_label_index(rows, 'A_history', vocab) if 'A_history' in fields else None
    partner = last_label_index(rows, 'B_history', vocab) if 'B_history' in fields else None
    weights = np.asarray(record['weights'], dtype=float)
    return _copy_logits(encoder.transform(rows) @ weights, record['beta'], own, record['gamma'], partner)


def prior_logits(prior, vocab, n):
    return np.tile(np.log(np.array([prior[t] for t in vocab])), (n, 1))


def stored_v4_probabilities(path):
    return {(r['coda_id'], r['condition']): r['probabilities'] for r in read_jsonl(path)}


def v4_validation_log_likelihoods(path):
    """The validation log-likelihood v4 recorded for its chosen penalties, per fold and condition."""
    out = {}
    for row in read_jsonl(path):
        chosen = [g for g in row['grid'] if g['l2'] == row['chosen_l2'] and g['l2_s'] == row['chosen_l2_s']]
        if len(chosen) != 1:
            raise ValueError('Chosen v4 penalty not found exactly once in its grid')
        out[row['fold_id'], row['condition']] = chosen[0]['validation_log_likelihood_bits']
    return out


# ---------------------------------------------------------------- inputs

def timing_fields(fields):
    return [f for f in fields if not f.endswith('_history')]


def condition_blocks(fields):
    blocks = [b for b in ('A', 'B') if HISTORY_FIELDS[b] in fields]
    return tuple(blocks + (['S'] if timing_fields(fields) else []))


def encode_split(rows, vocab, fields, max_len, timing_encoder, offsets):
    """Every input of one split as arrays, built once and indexed per minibatch."""
    if len(offsets) != len(rows):
        raise ValueError('One offset row per target is required')
    data = dict(offset=np.ascontiguousarray(offsets, dtype=np.float64),
                y=np.array([vocab.index(r['Ct']) for r in rows], dtype=np.int64))
    for block, field in HISTORY_FIELDS.items():
        if field in fields:
            data[block.lower() + '_seq'], data[block.lower() + '_len'] = encode_history(rows, field, vocab, max_len)
    if timing_encoder is not None:
        data['s'] = timing_encoder.transform(rows)[:, :-1].astype(np.float32)
    return data


def to_tensors(data, device):
    return {k: torch.from_numpy(v).to(device) for k, v in data.items()}


def condition_data(prepared, condition, fields, logits_fn, max_len, splits=('train', 'validation', 'test')):
    """Encoded inputs and the offset-only (logistic) probabilities of one fold and condition.

    logits_fn(rows) gives the offset logits; None means condition `none`, whose offset is the
    log of the fold's smoothed prior and whose offset-only prediction is that prior itself."""
    policy = prepared['policy']
    vocab, prior = policy['vocabulary'], policy['label_prior']
    s_fields = timing_fields(fields)
    timing_encoder = Encoder(prepared['train'], vocab, s_fields) if s_fields else None
    data, lr = {}, {}
    for split in splits:
        rows = prepared[split]
        if logits_fn is None:
            offsets = prior_logits(prior, vocab, len(rows))
            lr[split] = np.array([[prior[t] for t in vocab]] * len(rows))
        else:
            offsets = logits_fn(rows)
            lr[split] = softmax_rows(offsets)
        data[split] = encode_split(rows, vocab, fields, max_len, timing_encoder, offsets)
    return dict(fold_id=prepared['fold']['fold_id'], condition=condition, blocks=condition_blocks(fields),
                timing_dim=len(timing_encoder.names) if timing_encoder is not None else None,
                vocab=vocab, data=data, lr_probabilities=lr)


def per_target_bits(probabilities, y):
    return np.log2(probabilities[np.arange(len(y)), y])


# ---------------------------------------------------------------- model and training

class ResidualModel(nn.Module):
    """Offset plus one zero-initialised, bias-free projection per block that is on.

    Absent blocks are not built and own no parameters. The residual is computed in float32 and
    added to the float64 offset, so with U = 0 the logits are the offset exactly."""
    def __init__(self, vocab_size, blocks, hidden, dropout, embedding_dim, timing_dim=None):
        super().__init__()
        if any(b not in BLOCKS for b in blocks) or len(set(blocks)) != len(blocks):
            raise ValueError(f'Invalid blocks {blocks}')
        self.blocks = tuple(b for b in BLOCKS if b in blocks)
        for block in self.blocks:
            key = block.lower()
            if block == 'S':
                if timing_dim is None:
                    raise ValueError('The S block needs its input size')
                self.s_encoder = TimingEncoder(timing_dim, hidden, dropout)
            else:
                setattr(self, key + '_encoder', HistoryEncoder(vocab_size, embedding_dim, hidden, vocab_size, dropout))
            output = nn.Linear(hidden, vocab_size, bias=False)
            nn.init.zeros_(output.weight)
            setattr(self, key + '_output', output)

    def forward(self, batch):
        logits = batch['offset']
        for block in self.blocks:
            key = block.lower()
            if block == 'S':
                h = self.s_encoder(batch['s'])
            else:
                h = getattr(self, key + '_encoder')(batch[key + '_seq'], batch[key + '_len'])
            logits = logits + getattr(self, key + '_output')(h).double()
        return logits


def log_likelihood_bits_per_target(logits, y):
    """Through a one-hot product rather than an indexed loss: elementwise only, deterministic on every device."""
    log_probabilities = nn.functional.log_softmax(logits, dim=1)
    return (log_probabilities * nn.functional.one_hot(y, logits.shape[1]).to(logits.dtype)).sum(dim=1) * LOG2E


def fit_residual(train, validation, blocks, hidden, dropout, config, seed, device='cpu', timing_dim=None,
                 batches=None, full_train_eval=False):
    """Train one candidate with one seed (AdamW, decoupled weight decay over all parameters).

    Validation log-likelihood is evaluated at epoch 0, before any step, and after every epoch; the
    best epoch is restored, and epoch 0 means the offset alone. config['patience'] None trains for
    all config['max_epochs'] epochs. `batches(idx)` returns one minibatch; by default it indexes the
    pre-encoded training tensors."""
    if not blocks:
        raise ValueError('With every block absent the model is the offset alone; there is nothing to train')
    torch.manual_seed(seed)
    model = ResidualModel(train['offset'].shape[1], blocks, hidden, dropout, config['embedding_dim'], timing_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    train_t, val_t = to_tensors(train, device), to_tensors(validation, device)
    if batches is None:
        def batches(idx):
            idx = idx.to(device)
            return {k: v[idx] for k, v in train_t.items()}
    generator = torch.Generator().manual_seed(seed)
    n, patience = len(train['y']), config['patience']

    def split_bits(data):
        model.eval()
        with torch.no_grad():
            return log_likelihood_bits_per_target(model(data), data['y'])

    val_ll = split_bits(val_t).mean().item()
    train_ll = split_bits(train_t).mean().item()
    curve = [dict(epoch=0, train_log_likelihood_bits=train_ll, validation_log_likelihood_bits=val_ll)]
    if full_train_eval:
        curve[0]['train_eval_log_likelihood_bits'] = train_ll
    best_ll, best_epoch, best_state, since_best, epoch = val_ll, 0, copy.deepcopy(model.state_dict()), 0, 0
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = torch.randperm(n, generator=generator)
        total = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, n, config['batch_size']):
            batch = batches(order[start:start + config['batch_size']])
            ll = log_likelihood_bits_per_target(model(batch), batch['y'])
            loss = -ll.mean() / LOG2E  # mean cross-entropy in nats
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += ll.detach().sum()
        val_ll = split_bits(val_t).mean().item()
        row = dict(epoch=epoch, train_log_likelihood_bits=total.item() / n, validation_log_likelihood_bits=val_ll)
        if full_train_eval:
            row['train_eval_log_likelihood_bits'] = split_bits(train_t).mean().item()
        curve.append(row)
        if val_ll > best_ll:
            best_ll, best_epoch, best_state, since_best = val_ll, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if patience is not None and since_best >= patience:
                break
    model.load_state_dict(best_state)
    trained = best_epoch > 0
    return dict(best_epoch=best_epoch, epochs_trained=epoch,
                early_stopped=patience is not None and since_best >= patience,
                epoch0_validation_log_likelihood_bits=curve[0]['validation_log_likelihood_bits'],
                best_validation_log_likelihood_bits=best_ll,
                validation_log_likelihood_bits_per_target=split_bits(val_t).cpu().numpy() if trained else None,
                curve=curve,
                state_dict={k: v.detach().cpu() for k, v in best_state.items()} if trained else None)


def residual_probabilities(state_dict, blocks, hidden, dropout, data, config, timing_dim):
    model = ResidualModel(data['offset'].shape[1], blocks, hidden, dropout, config['embedding_dim'], timing_dim)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(to_tensors(data, 'cpu')).numpy()
    return softmax_rows(logits)


# ---------------------------------------------------------------- candidates and selection

def candidate_list(blocks, hidden_grid, dropout_grid):
    """Logistic v4 alone, then every non-empty set of blocks x hidden size x dropout."""
    candidates = [dict(name=LR_ONLY, blocks=(), hidden=None, dropout=None)]
    for size in range(1, len(blocks) + 1):
        for subset in itertools.combinations(blocks, size):
            for hidden in hidden_grid:
                for dropout in dropout_grid:
                    candidates.append(dict(name=f"{'+'.join(subset)}/h{hidden}/d{dropout}", blocks=subset,
                                           hidden=hidden, dropout=dropout))
    return candidates


def simplicity_key(candidate):
    """DEC-21: fewer blocks on, then smaller hidden size, then more dropout."""
    return (len(candidate['blocks']), candidate['hidden'] or 0, -(candidate['dropout'] or 0.0))


def select_candidate(candidates, per_target):
    """DEC-21's one-SE rule on seed-averaged per-target validation log-likelihoods (bits).

    The best candidate has the highest mean (ties go to the simplest). A candidate is within one SE
    when its mean paired difference to the best is at most sd(differences, ddof=1)/sqrt(n). The
    chosen candidate is the simplest within; candidates equally simple by DEC-21's order (same
    number of blocks, hidden size and dropout, different blocks) go to the higher mean, then to the
    block order A, B, S."""
    n = len(per_target[LR_ONLY])
    if n < 2:
        raise ValueError('The one-SE rule needs at least two validation targets')
    means = {c['name']: float(np.mean(per_target[c['name']])) for c in candidates}
    ordered = sorted(candidates, key=lambda c: (simplicity_key(c), [BLOCKS.index(b) for b in c['blocks']]))
    best = ordered[0]
    for c in ordered[1:]:
        if means[c['name']] > means[best['name']]:
            best = c
    table = []
    for c in candidates:
        d = per_target[best['name']] - per_target[c['name']]
        mean_d, se = float(np.mean(d)), float(np.std(d, ddof=1) / math.sqrt(n))
        table.append(dict(candidate=c['name'], blocks=list(c['blocks']), hidden=c['hidden'], dropout=c['dropout'],
                          mean_validation_log_likelihood_bits=means[c['name']],
                          mean_difference_to_best_bits=mean_d, standard_error_bits=se, within_one_se=mean_d <= se))
    rows = {row['candidate']: row for row in table}
    chosen = min((c for c in candidates if rows[c['name']]['within_one_se']),
                 key=lambda c: (simplicity_key(c), -means[c['name']], [BLOCKS.index(b) for b in c['blocks']]))
    return dict(chosen=chosen['name'], best=best['name'], validation_targets=n,
                lr_only_validation_log_likelihood_bits=means[LR_ONLY],
                chosen_gain_over_lr_bits=means[chosen['name']] - means[LR_ONLY],
                chosen_mean_difference_to_best_bits=rows[chosen['name']]['mean_difference_to_best_bits'],
                chosen_standard_error_bits=rows[chosen['name']]['standard_error_bits'], candidates=table)


def seed_averaged_validation(candidates, fits, seeds, lr_validation_bits):
    """Per candidate, the per-target validation log-likelihood averaged over seeds. A seed whose best
    epoch is 0 is the offset alone, so it contributes the offset's own per-target values."""
    per_target = {LR_ONLY: lr_validation_bits}
    for c in candidates[1:]:
        seed_fits = [fits[c['name'], s] for s in seeds]
        if all(f['best_epoch'] == 0 for f in seed_fits):
            per_target[c['name']] = lr_validation_bits
        else:
            per_target[c['name']] = np.mean([lr_validation_bits if f['best_epoch'] == 0
                                             else f['validation_log_likelihood_bits_per_target'] for f in seed_fits], axis=0)
    return per_target


def select_condition(candidates, fits, seeds, lr_validation_bits):
    """Validation-day selection for one fold and condition. Receives no test data at all."""
    per_target = seed_averaged_validation(candidates, fits, seeds, lr_validation_bits)
    return select_candidate(candidates, per_target), per_target


def predict_condition(chosen, fits, seeds, test_data, lr_test_probabilities, config, timing_dim):
    """Test probabilities of the chosen candidate, one array per seed."""
    out = []
    for seed in seeds:
        fit = None if chosen['name'] == LR_ONLY else fits[chosen['name'], seed]
        if fit is None or fit['best_epoch'] == 0:
            out.append(lr_test_probabilities)
        else:
            out.append(residual_probabilities(fit['state_dict'], chosen['blocks'], chosen['hidden'], chosen['dropout'],
                                              test_data, config, timing_dim))
    return out


def model_settings(config, **overrides):
    settings = {k: config[k] for k in ('embedding_dim', 'learning_rate', 'weight_decay', 'batch_size', 'max_epochs', 'patience')}
    settings.update(overrides)
    return settings


def condition_tasks(job, config, seeds, device, settings=None, **extra):
    """One task per (candidate, seed) that needs training; train and validation inputs only."""
    settings = settings or model_settings(config)
    tasks = []
    for c in candidate_list(job['blocks'], config['hidden_grid'], config['dropout_grid'])[1:]:
        for seed in seeds:
            tasks.append(dict(key=(job['fold_id'], job['condition'], c['name'], seed),
                              train=job['data']['train'], validation=job['data']['validation'],
                              blocks=c['blocks'], hidden=c['hidden'], dropout=c['dropout'], config=settings,
                              seed=seed, device=device, timing_dim=job['timing_dim'], **extra))
    return tasks


# ---------------------------------------------------------------- running fits

@contextlib.contextmanager
def deterministic_single_thread():
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def _init_worker():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _fit_task(task):
    return task['key'], fit_residual(task['train'], task['validation'], task['blocks'], task['hidden'], task['dropout'],
                                     task['config'], task['seed'], task['device'], task['timing_dim'],
                                     full_train_eval=task.get('full_train_eval', False))


def _warm_worker(device):
    if device == 'cuda':
        torch.zeros(1, device='cuda')
    time.sleep(1.0)
    return os.getpid()


class _Deferred:
    def __init__(self, task):
        self.task = task

    def result(self):
        with deterministic_single_thread():
            return _fit_task(self.task)


class FitRunner:
    """Fits tasks in this process (workers=0) or in a spawn-context process pool. Every fit runs on
    one torch thread with deterministic algorithms, so results don't depend on the worker count.
    Results come back in submission order."""
    def __init__(self, workers, device='cpu'):
        self.workers, self.device, self.pool = workers, device, None

    def __enter__(self):
        if self.workers > 0:
            self.pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init_worker)
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)

    def submit(self, tasks):
        if self.pool is None:
            return [_Deferred(t) for t in tasks]
        return [self.pool.submit(_fit_task, t) for t in tasks]

    def run(self, tasks):
        return [f.result() for f in self.submit(tasks)]


def require_cuda_determinism():
    if not torch.cuda.is_available():
        raise ValueError('CUDA requested but not available')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in CUBLAS_SETTINGS:
        raise ValueError(f'Set CUBLAS_WORKSPACE_CONFIG to one of {CUBLAS_SETTINGS} before starting this process '
                         '(nn.LSTM cuDNN determinism note); it cannot be set at runtime.')


# ---------------------------------------------------------------- shared loading

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    manifest_path = root / config['manifest']
    manifest = read_json(manifest_path)
    source = root / manifest['configuration']['input']
    if _sha256(source) != manifest['input_sha256']:
        raise ValueError('Stale fold manifest')
    return path, root, config, manifest, manifest_path, source


def _hashes(root, paths):
    return {p.relative_to(root).as_posix(): _sha256(p) for p in paths}


def prepare_folds(rows, manifest, test_days=None):
    prepared = []
    for fold in manifest['folds']:
        if test_days is not None and fold['test_day'] not in test_days:
            continue
        mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
        if policy != fold['vocabulary_policy']:
            raise ValueError('Recomputed fold policy differs from manifest')
        splits = {s: [r for r in mapped if r['split'] == s and r['comparison_eligible']] for s in ('train', 'validation', 'test')}
        for split, rows_ in splits.items():
            if sorted(r['coda_id'] for r in rows_) != fold['comparison_ids'][split]:
                raise ValueError(f"{fold['fold_id']}: {split} targets differ from the manifest")
        if not splits['validation']:
            raise ValueError('Fold has no validation comparison targets for selection')
        prepared.append(dict(fold=fold, policy=policy, **splits))
    if test_days is not None and len(prepared) != len(set(test_days)):
        raise ValueError('Not every requested test day exists in the fold manifest')
    return prepared


def v4_job(prepared, condition, fields, v4_models, max_len, splits=('train', 'validation', 'test')):
    if condition == 'none':
        logits_fn = None
    else:
        record = v4_models[prepared['fold']['fold_id'], condition]
        logits_fn = lambda rows: v4_logits(record, prepared['train'], rows, prepared['policy']['vocabulary'])  # noqa: E731
    return condition_data(prepared, condition, fields, logits_fn, max_len, splits)


def check_model_settings(config, other, label):
    for key in MODEL_KEYS:
        if other.get(key) != config[key]:
            raise ValueError(f'{label} used {key}={other.get(key)!r}, the full run uses {config[key]!r}')


# ---------------------------------------------------------------- checks before the full run

def check_wiring(prepared_folds, conditions, v4_models, stored, v4_validation, config, max_len):
    """PLAN [M4] checks 1 and 2 on every fold and condition.

    The test-day part compares two computations of logistic v4 with each other (recomputed offsets
    vs v4's stored probabilities); it uses no residual output and no labels."""
    spec = config['checks']['wiring']
    settings = model_settings(config, max_epochs=0)
    rows = []
    for prepared in prepared_folds:
        fold_id = prepared['fold']['fold_id']
        for condition, fields in conditions.items():
            job = v4_job(prepared, condition, fields, v4_models, max_len)
            vocab = job['vocab']
            stored_test = np.array([[stored[r['coda_id'], condition][t] for t in vocab] for r in prepared['test']])
            lr_test = job['lr_probabilities']['test']
            row = dict(fold_id=fold_id, condition=condition,
                       offset_test_max_abs_difference=float(np.max(np.abs(lr_test - stored_test))),
                       offset_test_bitwise=bool(np.array_equal(lr_test, stored_test)))
            row['offset_test_passed'] = row['offset_test_max_abs_difference'] <= config['offset_tolerance']
            if condition != 'none':
                val = job['data']['validation']
                lr_val = float(np.mean(per_target_bits(job['lr_probabilities']['validation'], val['y'])))
                recorded = v4_validation[fold_id, condition]
                row.update(lr_only_validation_difference_bits=lr_val - recorded,
                           lr_only_validation_passed=abs(lr_val - recorded) <= spec['epoch0_tolerance_bits'])
                with deterministic_single_thread():
                    fit = fit_residual(job['data']['train'], val, job['blocks'], spec['hidden'], spec['dropout'],
                                       settings, spec['seed'], timing_dim=job['timing_dim'])
                    torch.manual_seed(spec['seed'])
                    model = ResidualModel(len(vocab), job['blocks'], spec['hidden'], spec['dropout'],
                                          config['embedding_dim'], job['timing_dim'])
                    model.eval()
                    with torch.no_grad():
                        val_t = to_tensors(val, 'cpu')
                        epoch0_logits_identical = bool(torch.equal(model(val_t), val_t['offset']))
                epoch0 = fit['curve'][0]['validation_log_likelihood_bits']
                row.update(v4_validation_log_likelihood_bits=recorded, lr_only_validation_log_likelihood_bits=lr_val,
                           epoch0_validation_log_likelihood_bits=epoch0,
                           epoch0_validation_difference_bits=epoch0 - recorded,
                           epoch0_logits_identical_to_offset=epoch0_logits_identical,
                           epoch0_passed=abs(epoch0 - recorded) <= spec['epoch0_tolerance_bits'])
            row['passed'] = (row['offset_test_passed'] and row.get('lr_only_validation_passed', True)
                             and row.get('epoch0_passed', True) and row.get('epoch0_logits_identical_to_offset', True))
            rows.append(row)
    return dict(rows=rows, passed=all(r['passed'] for r in rows))


def check_learnability(prepared_folds, conditions, v4_models, config, max_len, runner):
    """PLAN [M4] check 3: every block on, no early stopping; training log-likelihood (eval mode, full
    training set) must end well above the offset's."""
    spec = config['checks']['learnability']
    settings = model_settings(config, max_epochs=spec['max_epochs'], patience=None)
    tasks = []
    for prepared in prepared_folds:
        if prepared['fold']['test_day'] not in spec['test_days']:
            continue
        for condition in spec['conditions']:
            job = v4_job(prepared, condition, conditions[condition], v4_models, max_len, splits=('train', 'validation'))
            for seed in spec['seeds']:
                tasks.append(dict(key=(job['fold_id'], condition, seed), train=job['data']['train'],
                                  validation=job['data']['validation'], blocks=job['blocks'], hidden=spec['hidden'],
                                  dropout=spec['dropout'], config=settings, seed=seed, device='cpu',
                                  timing_dim=job['timing_dim'], full_train_eval=True))
    runs, curves = [], []
    for (fold_id, condition, seed), fit in runner.run(tasks):
        curve = fit['curve']
        offset, final = curve[0]['train_eval_log_likelihood_bits'], curve[-1]['train_eval_log_likelihood_bits']
        runs.append(dict(fold_id=fold_id, condition=condition, seed=seed, epochs=curve[-1]['epoch'],
                         offset_train_log_likelihood_bits=offset, final_train_log_likelihood_bits=final,
                         best_train_log_likelihood_bits=max(r['train_eval_log_likelihood_bits'] for r in curve),
                         train_gain_bits=final - offset,
                         offset_validation_log_likelihood_bits=curve[0]['validation_log_likelihood_bits'],
                         final_validation_log_likelihood_bits=curve[-1]['validation_log_likelihood_bits'],
                         best_validation_epoch=fit['best_epoch'], passed=final - offset >= spec['minimum_gain_bits']))
        curves += [dict(row, fold_id=fold_id, condition=condition, seed=seed) for row in curve]
    return dict(runs=runs, passed=bool(runs) and all(r['passed'] for r in runs)), curves


def planted_eligible(row, lag):
    history = row['A_history']
    return len(history) >= lag and history[-lag] not in history[-3:]


def check_planted(prepared_folds, logistic_config, config, max_len, runner):
    """PLAN [M4] check 4 on one fold, training and validation days only: relabel a fraction of the
    eligible targets as the own coda at the given lag (labels only; histories unchanged), refit
    logistic regression with v4's procedure, and run condition A's full residual selection."""
    spec = config['checks']['planted']
    lag, fields, condition = spec['lag'], ['A_history'], 'A'
    prepared = next(p for p in prepared_folds if p['fold']['test_day'] == spec['test_day'])
    vocab = prepared['policy']['vocabulary']
    rng = np.random.RandomState(spec['sampling_seed'])
    eligible = {s: [i for i, r in enumerate(prepared[s]) if planted_eligible(r, lag)] for s in ('train', 'validation')}
    order = {s: rng.permutation(len(eligible[s])) for s in ('train', 'validation')}
    candidates = candidate_list(('A',), config['hidden_grid'], config['dropout_grid'])
    results = []
    for fraction in spec['fractions']:
        planted, relabelled = {}, {}
        for s in ('train', 'validation'):
            k = int(math.floor(fraction * len(eligible[s]) + 0.5))
            planted[s] = sorted(eligible[s][j] for j in order[s][:k])
            chosen_rows = set(planted[s])
            relabelled[s] = [dict(r, Ct=r['A_history'][-lag]) if i in chosen_rows else r for i, r in enumerate(prepared[s])]
        encoder, refit, _ = select_l2_and_fit(relabelled['train'], relabelled['validation'], vocab, fields, logistic_config)
        logits_fn = lambda rows: _copy_logits(encoder.transform(rows) @ refit['w'], refit['beta'],  # noqa: E731
                                              last_label_index(rows, 'A_history', vocab), None, None)
        job = condition_data(dict(prepared, train=relabelled['train'], validation=relabelled['validation']),
                             condition, fields, logits_fn, max_len, splits=('train', 'validation'))
        fits = {(k[2], k[3]): fit for k, fit in runner.run(condition_tasks(job, config, config['seeds'], 'cpu'))}
        y_val = job['data']['validation']['y']
        lr_val = per_target_bits(job['lr_probabilities']['validation'], y_val)
        selection, per_target = select_condition(candidates, fits, config['seeds'], lr_val)
        best_a = max(candidates[1:], key=lambda c: (float(np.mean(per_target[c['name']])), -candidates.index(c)))
        idx = planted['validation']
        planted_mean = lambda values: float(np.mean(values[idx])) if idx else None  # noqa: E731  f = 0: nothing planted
        results.append(dict(fraction=fraction, eligible_train=len(eligible['train']), eligible_validation=len(eligible['validation']),
                            planted_train=len(planted['train']), planted_validation=len(idx),
                            refit_l2=refit['l2'], refit_beta=refit['beta'],
                            refit_validation_log_likelihood_bits=refit['validation_log_likelihood_bits'],
                            chosen=selection['chosen'], best=selection['best'],
                            a_selected=selection['chosen'] != LR_ONLY,
                            chosen_gain_over_lr_bits=selection['chosen_gain_over_lr_bits'],
                            chosen_standard_error_bits=selection['chosen_standard_error_bits'],
                            best_a_candidate=best_a['name'],
                            best_a_gain_over_lr_bits=float(np.mean(per_target[best_a['name']])) - selection['lr_only_validation_log_likelihood_bits'],
                            planted_mean_bits_offset=planted_mean(lr_val),
                            planted_mean_bits_chosen=planted_mean(per_target[selection['chosen']]),
                            planted_mean_bits_best_a=planted_mean(per_target[best_a['name']]),
                            best_epochs_best_a=[fits[best_a['name'], s]['best_epoch'] for s in config['seeds']],
                            selection=selection))
    top = max(results, key=lambda r: r['fraction'])
    return dict(fold_id=prepared['fold']['fold_id'], lag=lag, results=results, passed=top['a_selected'])


def run_checks(config_path):
    path, root, config, manifest, manifest_path, source = _load(config_path)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = read_jsonl(source)
    max_len = manifest['preparation_configuration']['history_limit']
    inputs = [root / config[k] for k in ('logistic_models', 'logistic_predictions', 'logistic_selection', 'logistic_config')]
    v4_models = load_v4_models(inputs[0])
    logistic_config = read_json(inputs[3])
    prepared = prepare_folds(rows, manifest)
    conditions = manifest['input_conditions']
    wiring = check_wiring(prepared, conditions, v4_models, stored_v4_probabilities(inputs[1]),
                          v4_validation_log_likelihoods(inputs[2]), config, max_len)
    print(f"Wiring: {'passed' if wiring['passed'] else 'FAILED'}", flush=True)
    with FitRunner(config['workers']) as runner:
        learnability, curves = check_learnability(prepared, conditions, v4_models, config, max_len, runner)
        print(f"Learnability: {'passed' if learnability['passed'] else 'FAILED'}", flush=True)
        planted = check_planted(prepared, logistic_config, config, max_len, runner)
        print(f"Planted signal: {'A selected at the largest fraction' if planted['passed'] else 'A NEVER SELECTED'}", flush=True)
    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'learnability_curves.jsonl', curves)
    summary = dict(configuration=config, wiring=wiring, learnability=learnability, planted=planted,
                   gates=dict(wiring=wiring['passed'], learnability=learnability['passed'], planted=planted['passed']),
                   versions={'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__},
                   input_sha256=_hashes(root, [source, manifest_path, path] + inputs))
    write_text(output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(['# RQ1 residual LSTM: checks before the full run (DEC-21, PLAN [M4])', '',
                                  'Training and validation days only; no residual model is scored on a test day. '
                                  'The full run reads the gates from this run\'s summary.json.', '']
                                 + checks_lines(summary, curves) + ['']))
    return summary


# ---------------------------------------------------------------- device and throughput

def measure_throughput(task, seeds, workers, device):
    """Each worker fits every seed once, so a run is len(seeds) x workers fits; workers start first."""
    with FitRunner(workers, device) as runner:
        list(runner.pool.map(_warm_worker, [device] * workers))
        tasks = [dict(task, key=(w, s), seed=s, device=device) for w in range(workers) for s in seeds]
        start = time.perf_counter()
        results = runner.run(tasks)
        wall = time.perf_counter() - start
    epochs = [fit['epochs_trained'] for _, fit in results]
    return dict(device=device, workers=workers, fits=len(tasks), wall_seconds=wall,
                fits_per_hour=len(tasks) / wall * 3600, mean_epochs=sum(epochs) / len(epochs))


def fits_identical(a, b):
    if a['curve'] != b['curve'] or a['best_epoch'] != b['best_epoch']:
        return False
    if (a['state_dict'] is None) != (b['state_dict'] is None):
        return False
    if a['state_dict'] is not None and (a['state_dict'].keys() != b['state_dict'].keys()
                                        or not all(torch.equal(a['state_dict'][k], b['state_dict'][k]) for k in a['state_dict'])):
        return False
    return (a['validation_log_likelihood_bits_per_target'] is None
            or np.array_equal(a['validation_log_likelihood_bits_per_target'], b['validation_log_likelihood_bits_per_target']))


def determinism_check(task, seed, device):
    """Two fits with the same seed, in two separate worker processes."""
    with FitRunner(2, device) as runner:
        (_, a), (_, b) = runner.run([dict(task, key=0, seed=seed, device=device), dict(task, key=1, seed=seed, device=device)])
    return dict(device=device, seed=seed, identical=fits_identical(a, b), best_epochs=[a['best_epoch'], b['best_epoch']])


def choose_device(options, tie_tolerance):
    """Highest throughput among eligible options; a GPU within tie_tolerance of the CPU loses the tie."""
    eligible = [o for o in options if o['eligible']]
    cpu = next(o for o in eligible if o['device'] == 'cpu')
    best = max(eligible, key=lambda o: o['fits_per_hour'])
    if best['device'] != 'cpu' and cpu['fits_per_hour'] >= (1 - tie_tolerance) * best['fits_per_hour']:
        best = cpu
    return dict(device=best['device'], workers=best['workers'], fits_per_hour=best['fits_per_hour'])


def full_study_fits(conditions, config):
    per_fold = sum(len(candidate_list(condition_blocks(f), config['hidden_grid'], config['dropout_grid'])) - 1
                   for f in conditions.values())
    return per_fold * len(config['seeds'])


def run_throughput(config_path):
    path, root, config, manifest, manifest_path, source = _load(config_path)
    spec = config['throughput']
    rows = read_jsonl(source)
    max_len = manifest['preparation_configuration']['history_limit']
    models_path = root / config['logistic_models']
    prepared = prepare_folds(rows, manifest, test_days=[spec['test_day']])[0]
    job = v4_job(prepared, spec['condition'], manifest['input_conditions'][spec['condition']], load_v4_models(models_path),
                 max_len, splits=('train', 'validation'))
    task = dict(train=job['data']['train'], validation=job['data']['validation'], blocks=job['blocks'],
                hidden=spec['hidden'], dropout=spec['dropout'], config=model_settings(config), timing_dim=job['timing_dim'])
    options = [dict(measure_throughput(task, spec['seeds'], spec['cpu_workers'], 'cpu'), eligible=True, reason=None)]
    print(f"cpu x{spec['cpu_workers']}: {options[0]['fits_per_hour']:.1f} fits/hour", flush=True)
    determinism = [determinism_check(task, spec['seeds'][0], 'cpu')]
    gpu_reason = None
    if spec['gpu_processes']:
        if not torch.cuda.is_available():
            gpu_reason = 'CUDA not available'
        elif os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in CUBLAS_SETTINGS:
            gpu_reason = f'CUBLAS_WORKSPACE_CONFIG not set to one of {CUBLAS_SETTINGS} at launch'
        else:
            try:
                determinism.append(determinism_check(task, spec['seeds'][0], 'cuda'))
            except Exception as error:  # a deterministic-mode error makes the GPU ineligible, not the run fail
                determinism.append(dict(device='cuda', seed=spec['seeds'][0], identical=False, error=repr(error)))
            if 'error' in determinism[-1]:
                gpu_reason = f"fit failed under deterministic algorithms: {determinism[-1]['error']}"
            else:
                gpu_identical = determinism[-1]['identical']
                for k in spec['gpu_processes']:
                    result = measure_throughput(task, spec['seeds'], k, 'cuda')
                    options.append(dict(result, eligible=gpu_identical,
                                        reason=None if gpu_identical else 'two same-seed fits differ'))
                    print(f"cuda x{k}: {result['fits_per_hour']:.1f} fits/hour", flush=True)
    if gpu_reason:
        options += [dict(device='cuda', workers=k, eligible=False, reason=gpu_reason) for k in spec['gpu_processes']]
    if not determinism[0]['identical']:
        raise ValueError('Two CPU fits with the same seed differ; the CPU path is not deterministic')
    choice = choose_device(options, spec['tie_tolerance'])
    total = full_study_fits(manifest['input_conditions'], config) * len(manifest['folds'])
    choice.update(full_study_fits=total, estimated_hours=total / choice['fits_per_hour'])
    summary = dict(configuration=config, options=options, determinism=determinism, choice=choice,
                   versions={'numpy': np.__version__, 'torch': torch.__version__,
                             'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
                   cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                   input_sha256=_hashes(root, [source, manifest_path, path, models_path]))
    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    write_text(output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(['# RQ1 residual LSTM: device and throughput (step 6d)', ''] + throughput_lines(summary) + ['']))
    return summary


# ---------------------------------------------------------------- full run

def run(config_path):
    path, root, config, manifest, manifest_path, source = _load(config_path)
    evaluation_config = read_json(root / config['evaluation_config'])
    if evaluation_config['seeds'] != config['seeds']:
        raise ValueError('Evaluation seed configuration mismatch')
    checks_path = root / config['checks_summary']
    curves_path = checks_path.parent / 'learnability_curves.jsonl'
    checks, learnability_curves = read_json(checks_path), read_jsonl(curves_path)
    throughput = read_json(root / config['throughput_summary'])
    check_model_settings(config, checks['configuration'], 'The checks')
    check_model_settings(config, throughput['configuration'], 'The throughput measurement')
    if config.get('enforce_check_gates', True) and not all(checks['gates'].values()):
        raise ValueError(f"The checks before the full run did not pass: {checks['gates']}")
    device, workers = throughput['choice']['device'], throughput['choice']['workers']
    if device == 'cuda':
        require_cuda_determinism()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seeds = config['seeds']
    rows = read_jsonl(source)
    max_len = manifest['preparation_configuration']['history_limit']
    v4_models = load_v4_models(root / config['logistic_models'])
    stored = stored_v4_probabilities(root / config['logistic_predictions'])
    conditions = manifest['input_conditions']
    started = time.perf_counter()

    jobs, baselines = [], []
    for prepared in prepare_folds(rows, manifest):
        fold_id, train, test = prepared['fold']['fold_id'], prepared['train'], prepared['test']
        vocab, prior = prepared['policy']['vocabulary'], prepared['policy']['label_prior']
        rate = sum(r['Ct'] == r['A_history'][-1] for r in train) / len(train)
        majority = max(sorted(prior), key=prior.get)
        for row in test:
            p = repeat_distribution(row['A_history'][-1], rate, prior)
            baselines.append(dict(coda_id=row['coda_id'], fold_id=fold_id, recording_date=row['recording_date'],
                                  target=row['Ct'], majority=majority, repeat_probabilities=p,
                                  repeat_prediction=max(sorted(p), key=p.get),
                                  repeat_log_likelihood_bits=log_likelihood_bits(p[row['Ct']]), training_repeat_rate=rate))
        for condition, fields in conditions.items():
            job = v4_job(prepared, condition, fields, v4_models, max_len)
            stored_test = np.array([[stored[r['coda_id'], condition][t] for t in vocab] for r in test])
            if np.max(np.abs(job['lr_probabilities']['test'] - stored_test)) > config['offset_tolerance']:
                raise ValueError(f'{fold_id} {condition}: offsets do not reproduce logistic v4')
            # The offset alone on a test day is logistic v4: use its stored probabilities, which the
            # recomputed offsets reproduce up to the last bits (v4 was fitted on another machine).
            job['lr_probabilities']['test'] = stored_test
            job['test_rows'] = test
            jobs.append(job)
    print(f'Prepared {len(jobs)} fold-condition jobs on {device} x{workers}', flush=True)

    predictions, selections, fits_out, scores, curves = [], [], [], [], []
    with FitRunner(workers, device) as runner:
        futures = [runner.submit(condition_tasks(job, config, seeds, device)) for job in jobs]
        for job, job_futures in zip(jobs, futures):
            fits = {}
            for future in job_futures:
                (fold_id, condition, name, seed), fit = future.result()
                fits[name, seed] = fit
                fits_out.append(dict(fold_id=fold_id, condition=condition, candidate=name, seed=seed,
                                     best_epoch=fit['best_epoch'], epochs_trained=fit['epochs_trained'],
                                     early_stopped=fit['early_stopped'],
                                     epoch0_validation_log_likelihood_bits=fit['epoch0_validation_log_likelihood_bits'],
                                     best_validation_log_likelihood_bits=fit['best_validation_log_likelihood_bits']))
                curves.append(dict(fold_id=fold_id, condition=condition, candidate=name, seed=seed,
                                   train_log_likelihood_bits=[r['train_log_likelihood_bits'] for r in fit['curve']],
                                   validation_log_likelihood_bits=[r['validation_log_likelihood_bits'] for r in fit['curve']]))
            candidates = candidate_list(job['blocks'], config['hidden_grid'], config['dropout_grid'])
            y_val = job['data']['validation']['y']
            lr_val = per_target_bits(job['lr_probabilities']['validation'], y_val)
            selection, per_target = select_condition(candidates, fits, seeds, lr_val)
            chosen = next(c for c in candidates if c['name'] == selection['chosen'])
            probabilities = predict_condition(chosen, fits, seeds, job['data']['test'], job['lr_probabilities']['test'],
                                              model_settings(config), job['timing_dim'])
            selections.append(dict(fold_id=job['fold_id'], condition=job['condition'], blocks_available=list(job['blocks']),
                                   chosen_best_epochs=[None if chosen['name'] == LR_ONLY else fits[chosen['name'], s]['best_epoch'] for s in seeds],
                                   **selection))
            for c in candidates:
                scores.append(dict(fold_id=job['fold_id'], condition=job['condition'], candidate=c['name'],
                                   validation_log_likelihood_bits_per_target=per_target[c['name']].tolist()))
            for seed, probs in zip(seeds, probabilities):
                for row, p in zip(job['test_rows'], probs):
                    predictions.append(dict(coda_id=row['coda_id'], fold_id=job['fold_id'], condition=job['condition'],
                                            seed=seed, probabilities=dict(zip(job['vocab'], map(float, p)))))
            del fits
            job_futures.clear()  # release this job's fits (weights and curves) once they are recorded
            if job['condition'] == list(conditions)[-1]:
                print(f"Finished {job['fold_id']} ({time.perf_counter() - started:.0f} s)", flush=True)

    summary, scored, paired = evaluate(predictions, manifest, rows, seeds, evaluation_config['bootstrap_replicates'],
                                       evaluation_config['bootstrap_seed'], evaluation_config['confidence'])
    baseline_metrics = {'majority': classification_metrics([r['target'] for r in baselines], [r['majority'] for r in baselines]),
                        'repeat_own_last': classification_metrics([r['target'] for r in baselines], [r['repeat_prediction'] for r in baselines])}
    baseline_metrics['repeat_own_last']['mean_log_likelihood_bits'] = math.fsum(r['repeat_log_likelihood_bits'] for r in baselines) / len(baselines)
    indexed = {r['coda_id']: r for r in baselines}
    seed_mean = seed_averaged_scores(scored)
    baseline_checks = {}
    for condition in ('A', 'A_B', 'A_S', 'A_B_S'):
        days = defaultdict(list)
        for (c, coda_id), (ll, day) in seed_mean.items():
            if c == condition:
                days[day].append(ll - indexed[coda_id]['repeat_log_likelihood_bits'])
        baseline_checks[condition] = day_bootstrap_bits(days, evaluation_config['bootstrap_replicates'],
                                                        evaluation_config['bootstrap_seed'], evaluation_config['confidence'])
    v4_scored_path = root / config['logistic_scored']
    gains = residual_gains(seed_mean, {r['coda_id']: r['target'] for r in scored}, stored, conditions, evaluation_config)
    stop_reasons = check_stop_conditions(summary, baseline_metrics, baseline_checks)
    stop_reasons += [f"{c}: residual gain over logistic v4 reliably below zero (CI {format_ci(g['ci_bits_per_coda'])})"
                     for c, g in gains.items() if g['ci_bits_per_coda'][1] < 0]
    v4_summary_path = root / config['logistic_summary']
    v4_summary = read_json(v4_summary_path)
    context = assayag_context(scored, read_jsonl(v4_scored_path), rows, baselines, seeds,
                              manifest['preparation_configuration']['history_limit'])

    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'predictions.jsonl', predictions)
    write_jsonl(output / 'baseline_predictions.jsonl', baselines)
    write_jsonl(output / 'selection.jsonl', selections)
    write_jsonl(output / 'fits.jsonl', fits_out)
    write_jsonl(output / 'validation_scores.jsonl', scores)
    write_jsonl(output / 'training_curves.jsonl', curves)
    eval_output = output / 'evaluation'
    eval_output.mkdir(exist_ok=True)
    write_jsonl(eval_output / 'scored_predictions.jsonl', scored)
    write_jsonl(eval_output / 'paired_contrasts.jsonl', paired)
    inputs = [source, manifest_path, path, root / config['evaluation_config'], checks_path, curves_path,
              root / config['throughput_summary'], root / config['logistic_models'], root / config['logistic_predictions'],
              v4_scored_path, v4_summary_path]
    summary.update(configuration=config, device=dict(device=device, workers=workers), baseline_metrics=baseline_metrics,
                   repeat_baseline_checks=baseline_checks, residual_gain=gains, stop_reasons=stop_reasons,
                   chosen_candidates=chosen_distribution(selections, conditions), assayag_context=context,
                   versions={'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__},
                   input_sha256=_hashes(root, inputs))
    write_text(eval_output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    write_report(root, config, summary, v4_summary, selections, checks, learnability_curves, throughput)
    return summary, stop_reasons


def seed_averaged_scores(scored):
    """Per (condition, coda): the seed-averaged log-likelihood, exactly as evaluation.evaluate averages it."""
    grouped = defaultdict(list)
    for r in scored:
        grouped[r['condition'], r['coda_id']].append(r)
    return {key: (math.fsum(r['log_likelihood_bits'] for r in rs) / len(rs), rs[0]['recording_date'])
            for key, rs in grouped.items()}


def residual_gains(seed_mean, targets, stored, conditions, evaluation_config):
    """[U1] in DEC-21's form: residual network minus logistic v4, paired per coda, day bootstrap.

    v4's per-coda log-likelihood is recomputed here from its stored probabilities with the shared
    evaluation module, rather than read from its scored file: that file was written on another
    machine, whose log2 can differ in the last bit, which would turn folds where v4 itself was
    chosen (gain exactly zero) into noise-level positive or negative days."""
    gains = {}
    for condition in conditions:
        days = defaultdict(list)
        for (c, coda_id), (ll, day) in seed_mean.items():
            if c == condition:
                days[day].append(ll - log_likelihood_bits(stored[coda_id, condition][targets[coda_id]]))
        stats = day_bootstrap_bits(days, evaluation_config['bootstrap_replicates'], evaluation_config['bootstrap_seed'],
                                   evaluation_config['confidence'])
        stats['zero_days'] = sum(all(v == 0 for v in values) for values in days.values())
        gains[condition] = stats
    return gains


def chosen_distribution(selections, conditions):
    out = {}
    for condition in conditions:
        rows = [s for s in selections if s['condition'] == condition]
        chosen = [next(c for c in s['candidates'] if c['candidate'] == s['chosen']) for s in rows]
        trained = [c for c in chosen if c['candidate'] != LR_ONLY]
        epochs = [e for s in rows for e in s['chosen_best_epochs'] if e is not None]
        gains = sorted(s['chosen_gain_over_lr_bits'] for s in rows)
        out[condition] = dict(folds=len(rows), lr_only=len(rows) - len(trained),
                              blocks=dict(Counter('+'.join(c['blocks']) for c in trained)),
                              hidden=dict(Counter(str(c['hidden']) for c in trained)),
                              dropout=dict(Counter(str(c['dropout']) for c in trained)),
                              best_epochs=sorted(epochs), best_epoch_zero=sum(e == 0 for e in epochs),
                              chosen_gain_over_lr_bits=gains,
                              best_not_lr=sum(s['best'] != LR_ONLY for s in rows))
    return out


def assayag_context(scored, v4_scored, rows, baselines, seeds, history_limit):
    """PLAN [M4] context check, without retraining: condition B on switch/overlap targets against the
    majority class on that subset, and condition A's accuracy by own-history length."""
    targets = {r['coda_id']: r for r in rows if r['comparison_eligible']}
    majority = {b['coda_id']: b['majority'] for b in baselines}
    subset = {c for c, r in targets.items() if r['turn_category'] in ('switch', 'overlap')}
    length = {c: min(len(r['A_history_ids']), history_limit) for c, r in targets.items()}

    def accuracy(rs, keep):
        rs = [r for r in rs if keep(r['coda_id'])]
        return (sum(r['predicted'] == r['target'] for r in rs) / len(rs), len(rs)) if rs else (None, 0)

    def residual_accuracy(condition, keep):
        values = [accuracy([r for r in scored if r['condition'] == condition and r['seed'] == s], keep) for s in seeds]
        return sum(v for v, _ in values) / len(values) if values[0][0] is not None else None, values[0][1]

    b_res, n_b = residual_accuracy('B', lambda c: c in subset)
    b_v4, _ = accuracy([r for r in v4_scored if r['condition'] == 'B'], lambda c: c in subset)
    target_of = {r['coda_id']: r['target'] for r in scored if r['condition'] == 'B' and r['seed'] == seeds[0]}
    majority_accuracy = sum(majority[c] == target_of[c] for c in subset) / len(subset)
    by_length = []
    for k in range(1, history_limit + 1):
        res, n = residual_accuracy('A', lambda c: length[c] == k)
        v4_acc, _ = accuracy([r for r in v4_scored if r['condition'] == 'A'], lambda c: length[c] == k)
        by_length.append(dict(own_history_length=k, targets=n, residual_accuracy=res, v4_accuracy=v4_acc,
                              assayag_column=f't-{k - 1}' if k - 1 < len(TARGET_SINGLE_WHALE_CODA_TYPE_ACCURACY) else None,
                              assayag_accuracy=TARGET_SINGLE_WHALE_CODA_TYPE_ACCURACY[k - 1]
                              if k - 1 < len(TARGET_SINGLE_WHALE_CODA_TYPE_ACCURACY) else None))
    a_res, n_a = residual_accuracy('A', lambda c: True)
    a_v4, _ = accuracy([r for r in v4_scored if r['condition'] == 'A'], lambda c: True)
    return dict(partner_switch_overlap=dict(targets=n_b, residual_accuracy=b_res, v4_accuracy=b_v4,
                                            majority_accuracy=majority_accuracy,
                                            assayag_paired_range=[min(TARGET_PAIRED_CODA_TYPE_ACCURACY.values()),
                                                                  max(TARGET_PAIRED_CODA_TYPE_ACCURACY.values())]),
                own_history=dict(targets=n_a, residual_accuracy=a_res, v4_accuracy=a_v4, by_length=by_length,
                                 assayag_single_whale=TARGET_SINGLE_WHALE_CODA_TYPE_ACCURACY))


# ---------------------------------------------------------------- reports

def _f(value, digits=5):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def _pct(value):
    return 'n/a' if value is None else f'{value:.3%}'


def checks_lines(checks, curves=None, level='##'):
    lines = []
    wiring = checks['wiring']
    lines += [f'{level} a. Wiring (PLAN [M4] checks 1 and 2), all folds and conditions', '',
              'Per condition, over the 14 folds: the largest difference between the softmax of the recomputed offsets '
              '(the network with every block absent) and logistic v4\'s stored test probabilities (v4 against v4; no '
              'residual output, no labels), and how many folds agree bit for bit; the largest difference between the '
              'offset-only validation log-likelihood and the one v4 recorded when it chose its penalties; the largest '
              'difference between the untrained network\'s epoch-0 validation log-likelihood (every block on) and v4\'s '
              'recorded one; and whether the untrained network\'s logits equal the offsets exactly. Logistic v4 was '
              'fitted on another machine, so recomputed values can differ from its stored ones in the last bits; '
              'wherever the full run uses the offset alone on a test day (logistic v4 chosen, or a seed whose best '
              'epoch is 0), it uses v4\'s stored test probabilities, which these offsets reproduce.', '',
              '| Condition | Max offset diff (test) | Bitwise folds | Max offset-only validation diff (bits) | Max epoch-0 diff (bits) | Epoch-0 logits = offset | Passed |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for condition in dict.fromkeys(r['condition'] for r in wiring['rows']):
        rs = [r for r in wiring['rows'] if r['condition'] == condition]
        fitted = [r for r in rs if 'epoch0_passed' in r]
        epoch0 = f"{max(abs(r['epoch0_validation_difference_bits']) for r in fitted):.2e}" if fitted else 'n/a (prior)'
        lr_val = f"{max(abs(r['lr_only_validation_difference_bits']) for r in fitted):.2e}" if fitted else 'n/a (prior)'
        identical = f"{sum(r['epoch0_logits_identical_to_offset'] for r in fitted)}/{len(fitted)}" if fitted else 'n/a'
        lines.append(f"| {condition} | {max(r['offset_test_max_abs_difference'] for r in rs):.2e} | "
                     f"{sum(r['offset_test_bitwise'] for r in rs)}/{len(rs)} | {lr_val} | {epoch0} | {identical} | "
                     f"{sum(r['passed'] for r in rs)}/{len(rs)} |")
    spec = checks['configuration']['checks']
    lines += ['', f"Tolerances: offsets {checks['configuration']['offset_tolerance']}, epoch 0 "
              f"{spec['wiring']['epoch0_tolerance_bits']} bits. Gate: {'passed' if wiring['passed'] else 'FAILED'}.", '']

    learn = checks['learnability']
    ls = spec['learnability']
    lines += [f'{level} b. Learnability (PLAN [M4] check 3)', '',
              f"Every block on, hidden {ls['hidden']}, dropout {ls['dropout']}, {ls['max_epochs']} epochs without "
              f"early stopping, seeds {ls['seeds']}. Training log-likelihood is measured in eval mode on the full "
              f"training set after each epoch; epoch 0 is the offset (logistic v4). Pass: the final training "
              f"log-likelihood is at least {ls['minimum_gain_bits']} bits/coda above the offset's in every run.", '',
              '| Fold | Condition | Seed | Offset train LL | Final train LL | Gain | Offset validation LL | Final validation LL | Best validation epoch | Passed |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in learn['runs']:
        lines.append(f"| {r['fold_id']} | {r['condition']} | {r['seed']} | {_f(r['offset_train_log_likelihood_bits'])} | "
                     f"{_f(r['final_train_log_likelihood_bits'])} | {_f(r['train_gain_bits'])} | "
                     f"{_f(r['offset_validation_log_likelihood_bits'])} | {_f(r['final_validation_log_likelihood_bits'])} | "
                     f"{r['best_validation_epoch']} | {r['passed']} |")
    if curves:
        epochs = sorted({0, 1, 5, 10, 25, 50, 100, 150, 200, 250, 300} & {c['epoch'] for c in curves})
        lines += ['', 'Curves, mean over seeds (train LL in eval mode / validation LL, bits per coda):', '',
                  '| Fold | Condition | ' + ' | '.join(f'Epoch {e}' for e in epochs) + ' |',
                  '|---|---|' + '---:|' * len(epochs)]
        for fold_id, condition in dict.fromkeys((c['fold_id'], c['condition']) for c in curves):
            cells = []
            for e in epochs:
                rs = [c for c in curves if c['fold_id'] == fold_id and c['condition'] == condition and c['epoch'] == e]
                cells.append(f"{sum(c['train_eval_log_likelihood_bits'] for c in rs) / len(rs):.3f} / "
                             f"{sum(c['validation_log_likelihood_bits'] for c in rs) / len(rs):.3f}")
            lines.append(f"| {fold_id} | {condition} | " + ' | '.join(cells) + ' |')
    lines += ['', f"Gate: {'passed' if learn['passed'] else 'FAILED'}.", '']

    planted = checks['planted']
    ps = spec['planted']
    lines += [f'{level} c. Planted signal (PLAN [M4] check 4)', '',
              f"Fold {planted['fold_id']}, condition A, training and validation days only. Eligible targets have at "
              f"least {ps['lag']} own codas, and the own coda at lag {ps['lag']} differs from lags 1-3. A fraction f "
              f"of them (nested subsets from one RandomState({ps['sampling_seed']}) permutation per split) is relabelled "
              f"as the own coda at lag {ps['lag']}; histories are unchanged. Logistic regression is refitted on the "
              "planted data with v4's procedure, and condition A's full residual selection runs on its offsets. "
              "Mean log2-probabilities are over the planted validation targets, seed-averaged. f = 0 plants nothing: "
              "it is a reference for what condition A's selection does on this fold without a planted signal, and "
              "is not part of the gate. Whether A is selected therefore only shows detection where it differs from "
              "f = 0; the planted targets' log-probabilities under the residual network and the offset show whether "
              "the planted signal itself was picked up.", '',
              '| f | Planted train/validation | Refit L2 | A selected | Chosen | Chosen gain over LR (SE) | Best A candidate | Its gain over LR | Planted: offset | Planted: chosen | Planted: best A |',
              '|---:|---|---:|---|---|---|---|---:|---:|---:|---:|']
    for r in planted['results']:
        lines.append(f"| {r['fraction']} | {r['planted_train']}/{r['eligible_train']}, {r['planted_validation']}/{r['eligible_validation']} | "
                     f"{r['refit_l2']} | {'yes' if r['a_selected'] else 'no'} | {r['chosen']} | "
                     f"{_f(r['chosen_gain_over_lr_bits'])} ({_f(r['chosen_standard_error_bits'])}) | {r['best_a_candidate']} | "
                     f"{_f(r['best_a_gain_over_lr_bits'])} | {_f(r['planted_mean_bits_offset'], 3)} | "
                     f"{_f(r['planted_mean_bits_chosen'], 3)} | {_f(r['planted_mean_bits_best_a'], 3)} |")
    lines += ['', f"Gate (A's residual selected at the largest fraction): {'passed' if planted['passed'] else 'FAILED'}.", '']
    return lines


def throughput_lines(throughput):
    config = throughput['configuration']
    spec = config['throughput']
    lines = [f"Fold test-{spec['test_day']}, condition {spec['condition']}, every block on, hidden {spec['hidden']}, "
             f"dropout {spec['dropout']}, normal early stopping (patience {config['patience']}). Each worker fits seeds "
             f"{spec['seeds']} once; throughput = fits / wall time, workers started beforehand. CPU workers use one "
             f"torch thread each. The GPU is eligible only if two fits with the same seed, in two processes under "
             f"torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG (here "
             f"{throughput['cublas_workspace_config']!r}), give identical curves and weights. Ties within "
             f"{spec['tie_tolerance']:.0%} go to the CPU.", '',
             '| Device | Processes | Fits | Wall seconds | Fits/hour | Mean epochs/fit | Eligible |', '|---|---:|---:|---:|---:|---:|---|']
    for o in throughput['options']:
        if 'fits_per_hour' in o:
            lines.append(f"| {o['device']} | {o['workers']} | {o['fits']} | {o['wall_seconds']:.1f} | {o['fits_per_hour']:.1f} | "
                         f"{o['mean_epochs']:.1f} | {'yes' if o['eligible'] else 'no: ' + o['reason']} |")
        else:
            lines.append(f"| {o['device']} | {o['workers']} | - | - | - | - | no: {o['reason']} |")
    lines += ['', '| Determinism check | Seed | Identical | Note |', '|---|---:|---|---|']
    for d in throughput['determinism']:
        lines.append(f"| {d['device']} | {d['seed']} | {d['identical']} | {d.get('error', '')} |")
    c = throughput['choice']
    lines += ['', f"Chosen: {c['device']} with {c['workers']} processes ({c['fits_per_hour']:.1f} fits/hour on the largest "
              f"candidate). Full run: {c['full_study_fits']} fits; estimated wall time {c['estimated_hours']:.1f} hours "
              "(the largest candidate is the slowest per epoch, so this is conservative per epoch; epochs per fit vary).", '']
    return lines


def write_report(root, config, summary, v4_summary, selections, checks, learnability_curves, throughput):
    conditions = list(summary['metrics'])
    gains = summary['residual_gain']
    device = summary['device']
    lines = ['# RQ1 residual LSTM on top of logistic v4 (DEC-21)', '',
             "Logistic v4 (reports/rq1_logistic_v4.md) gives the headline lattice; this network is the "
             "enlarged-family check. Per fold and condition the logits are logistic v4's logits (a fixed offset, "
             "recomputed from its committed fits) plus one zero-initialised, bias-free projection per input block "
             f"that is on: an LSTM (embedding {config['embedding_dim']}) per coda history over up to "
             "10 codas, and a feed-forward encoder for S over turn category and the two onset-available timing values "
             "(standardised on training targets), each with dropout on its output. Absent blocks are not built; with "
             "every block absent the model is logistic v4. "
             f"AdamW (learning rate {config['learning_rate']}, decoupled weight decay {config['weight_decay']} on all "
             f"parameters), batch {config['batch_size']}, up to {config['max_epochs']} epochs, patience "
             f"{config['patience']}, restoring the best validation epoch; epoch 0 (the offset alone) is a candidate.", '',
             f"Candidates per fold and condition: hidden {config['hidden_grid']} x dropout {config['dropout_grid']} x "
             "each block on or absent (at least one on), plus logistic v4 alone. Selection on the validation day: "
             f"per-target validation log-likelihood averaged over seeds {config['seeds']}; the simplest candidate whose "
             "mean paired difference to the best is within one standard error (SD of the per-target differences, "
             "ddof 1, over sqrt(n)); simplicity order: fewer blocks on, smaller hidden size, more dropout; equally "
             "simple candidates go to the higher validation log-likelihood. The SE is a selection rule, not a "
             "reported interval; reported intervals are day-bootstrap intervals "
             f"({summary['bootstrap']['replicates']} replicates). Device: {device['device']} with {device['workers']} "
             "processes (step 6d). Test days are scored once, with the chosen candidate only.", '']

    lines += ['## Per condition: residual network vs logistic v4', '',
              'Residual gain = residual network minus logistic v4, paired per coda (seeds averaged per coda first), '
              'day-bootstrap CI. Zero days: days whose gain is exactly zero (logistic v4 chosen on that fold, or '
              'every seed at epoch 0).', '',
              '| Condition | v4 LL | Residual LL | v4 accuracy | Residual accuracy | v4 macro-F1 | Residual macro-F1 | Gain | Gain unweighted | 95% CI | Positive days | Zero days |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for c in conditions:
        v, m, g = v4_summary['metrics'][c], summary['metrics'][c], gains[c]
        lines.append(f"| {c} | {_f(v['mean_log_likelihood_bits'])} | {_f(m['mean_log_likelihood_bits'])} | "
                     f"{_pct(v['accuracy'])} | {_pct(m['accuracy'])} | {v['macro_f1']:.4f} | {m['macro_f1']:.4f} | "
                     f"{_f(g['mean_bits_per_coda'])} | {_f(g['unweighted_mean_bits_per_coda'])} | "
                     f"{format_ci(g['ci_bits_per_coda'])} | {g['positive_days']}/{g['days']} | {g['zero_days']} |")
    for name, m in summary['baseline_metrics'].items():
        ll = _f(m.get('mean_log_likelihood_bits'))
        lines.append(f"| {name} | {ll} | {ll} | {_pct(m['accuracy'])} | {_pct(m['accuracy'])} | {m['macro_f1']:.4f} | "
                     f"{m['macro_f1']:.4f} | | | | | |")
    lines += ['', f"Repetition: {summary['raw_repeats']} of {summary['comparison_targets']} comparison targets "
              f"({summary['raw_repeats'] / summary['comparison_targets']:.1%}) repeat the whale's own previous coda "
              f"({summary['mapped_repeats']} after UNK mapping).", '']

    lines += ['## Lattices side by side', '', '### [M2] contrasts', '',
              '| Contrast | Model | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part | Switch-choice part |',
              '|---|---|---:|---:|---|---:|---:|---:|']
    for name in ('own_history', 'partner_content', 'timing'):
        for label, s in (('logistic v4', v4_summary), ('residual', summary)):
            comp = s['contrasts'][name]['components']
            a = comp['log_likelihood_bits']
            lines.append(f"| {name} ({s['contrasts'][name]['full']} - {s['contrasts'][name]['reduced']}) | {label} | "
                         f"{_f(a['mean_bits_per_coda'])} | {_f(a['unweighted_mean_bits_per_coda'])} | {format_ci(a['ci_bits_per_coda'])} | "
                         f"{a['positive_days']}/{a['days']} | {_f(comp['repeat_bits']['mean_bits_per_coda'])} | "
                         f"{_f(comp['switch_choice_bits']['mean_bits_per_coda'])} |")
    lines += ['', '### All twelve add-one-input pairs', '',
              '| Pair | v4 bits/coda | v4 unweighted | v4 CI | v4 positive days | Residual bits/coda | Residual unweighted | Residual CI | Residual positive days |',
              '|---|---:|---:|---|---:|---:|---:|---|---:|']
    v4_pairs = {(d['added'], d['base']): d for d in classify_add_input_contrasts(v4_summary)}
    for d in classify_add_input_contrasts(summary):
        v = v4_pairs[d['added'], d['base']]
        lines.append(f"| adding {d['added']} to {describe(d['base'])} | {_f(v['mean'])} | {_f(v['unweighted_mean'])} | "
                     f"{format_ci([v['low'], v['high']])} | {v['positive_days']}/{v['days']} | {_f(d['mean'])} | "
                     f"{_f(d['unweighted_mean'])} | {format_ci([d['low'], d['high']])} | {d['positive_days']}/{d['days']} |")

    lines += ['', '## Chosen candidates', '',
              'Per condition over the 14 folds. Best epochs are those of the chosen candidate\'s seeds (folds where '
              'logistic v4 alone was chosen have none). "Best not LR" counts folds where the best candidate, before '
              'the one-SE rule, was a trained network.', '',
              '| Condition | LR-only chosen | Best not LR | Blocks on | Hidden | Dropout | Best epochs (min / median / max) | Seeds at epoch 0 | Validation gain over LR (min / median / max) |',
              '|---|---:|---:|---|---|---|---|---:|---|']
    for c in conditions:
        d = summary['chosen_candidates'][c]
        fmt = lambda counts: ', '.join(f'{k}: {v}' for k, v in sorted(counts.items())) or '-'  # noqa: E731
        epochs = d['best_epochs']
        epoch_text = f"{epochs[0]} / {epochs[len(epochs) // 2]} / {epochs[-1]}" if epochs else '-'
        g = d['chosen_gain_over_lr_bits']
        lines.append(f"| {c} | {d['lr_only']}/{d['folds']} | {d['best_not_lr']}/{d['folds']} | {fmt(d['blocks'])} | "
                     f"{fmt(d['hidden'])} | {fmt(d['dropout'])} | {epoch_text} | {d['best_epoch_zero']} | "
                     f"{g[0]:.4f} / {g[len(g) // 2]:.4f} / {g[-1]:.4f} |")
    lines += ['', 'Chosen candidate per fold (LR = logistic v4 alone; otherwise blocks/hidden/dropout):', '',
              '| Fold | ' + ' | '.join(c for c in conditions if c != 'none') + ' |',
              '|---|' + '---|' * (len(conditions) - 1)]
    by_fold = defaultdict(dict)
    for s in selections:
        by_fold[s['fold_id']][s['condition']] = s['chosen']
    for fold_id, chosen in by_fold.items():
        lines.append(f"| {fold_id} | " + ' | '.join(chosen[c] for c in conditions if c != 'none') + ' |')

    lines += ['', '## A-conditions vs repeat-own-last (residual network)', '',
              '| Condition | LL contrast | Unweighted day mean | 95% CI | Positive days |', '|---|---:|---:|---|---:|']
    for c, check in summary['repeat_baseline_checks'].items():
        lines.append(f"| {c} | {_f(check['mean_bits_per_coda'])} | {_f(check['unweighted_mean_bits_per_coda'])} | "
                     f"{format_ci(check['ci_bits_per_coda'])} | {check['positive_days']}/{check['days']} |")

    lines += ['', '## Stop rules (PLAN [M4])', '',
              'Stop and investigate before writing conclusions if any of these has a day-bootstrap CI entirely below '
              'zero: an A-condition against repeat-own-last; an add-one-input contrast; the residual gain over logistic '
              'v4 on a condition (which now signals selection overfitting to validation days, DEC-21).', '']
    lines += [f"- {r}" for r in summary['stop_reasons']] or ['None fires.']

    lines += ['', '## Checks before the full run (step 6)', ''] + checks_lines(checks, learnability_curves, level='###')
    lines += ['### d. Device and throughput', ''] + throughput_lines(throughput)

    ctx = summary['assayag_context']
    p, o = ctx['partner_switch_overlap'], ctx['own_history']
    lines += ['## Context check against Assayag et al. (PLAN [M4])', '',
              'Their split is random (70/15/15), ours holds out whole days, so their accuracies are optimistic next to '
              'ours; no model is retrained for this comparison.', '',
              f"Condition B (partner history) on the {p['targets']} switch and overlap targets: residual network "
              f"{_pct(p['residual_accuracy'])}, logistic v4 {_pct(p['v4_accuracy'])}, majority class (each fold's "
              f"training majority) {_pct(p['majority_accuracy'])}. Assayag et al. Table 3, LSTM, paired coda type: "
              f"{p['assayag_paired_range'][0]:.3f} to {p['assayag_paired_range'][1]:.3f} across 0-8 preceding codas.", '',
              f"Condition A (own history), all {o['targets']} targets: residual network {_pct(o['residual_accuracy'])}, "
              f"logistic v4 {_pct(o['v4_accuracy'])}. By own-history length, next to Assayag et al. Table 3, LSTM, "
              "single-whale coda type (column t-k uses the current coda and k before it, i.e. k+1 own codas; their "
              "window is fixed, ours is whatever each target has, capped at 10, so the columns are not paired one to "
              "one):", '',
              '| Own codas available | Targets | Residual accuracy | v4 accuracy | Assayag column | Assayag accuracy |',
              '|---:|---:|---:|---:|---|---:|']
    for r in o['by_length']:
        lines.append(f"| {r['own_history_length']} | {r['targets']} | {_pct(r['residual_accuracy'])} | {_pct(r['v4_accuracy'])} | "
                     f"{r['assayag_column'] or '-'} | {_f(r['assayag_accuracy'], 3) if r['assayag_accuracy'] is not None else '-'} |")

    lines += ['', '## Reading (PLAN §9)', '']
    if summary['stop_reasons']:
        lines += ['A stop rule fires (above). No conclusions are written; the tables are the complete result.', '']
    else:
        lines += ["- Headline numbers come from logistic v4; the residual network's lattice and residual gains are the "
                  "enlarged-family check.",
                  "- A contrast is v4's contrast plus the difference between the two conditions' residual gains, so the "
                  "contrasts are not monotone by construction.",
                  "- On validation days the residual network and logistic regression agree by construction, so their "
                  "agreement is not evidence against an optimisation failure; that evidence is the learnability checks "
                  "above.",
                  "- Logistic regression is a softmax network without hidden layers; the residual LSTM's family contains "
                  "v4 exactly, as the member whose residual is switched off.",
                  "- Limited data and capacity push contrasts toward zero: a positive contrast is robust, a null might "
                  "just mean too little data.", '',
                  '| Condition | Residual gain CI | Reading |', '|---|---|---|']
        for c in conditions:
            if c == 'none':
                continue
            g = gains[c]
            if g['ci_bits_per_coda'][0] > 0:
                reading = ("positive: structure in this condition's blocks that logistic regression can't express "
                           "(for A and B: lags 4-10 or non-additive patterns within lags 1-3; for S: nonlinear timing "
                           "effects; these are not separated, and not called long-range)")
            else:
                reading = ("includes zero: no further structure detectable in the enlarged family at this data size, "
                           "not that none exists")
            lines.append(f"| {c} | {format_ci(g['ci_bits_per_coda'])} | {reading} |")
        lines.append('')
    lines += ['Not run here: the old-delay (retrospective) sensitivity run of the residual network; the logistic v4 '
              'old-delay run is reports/rq1_retrospective_sensitivity_v4.md.', '',
              'This report does not establish grammar, syntax or causal mechanisms. Claims are about two-whale '
              'exchanges only. The switch-choice component of every contrast rests on about 270 non-repeat codas and '
              'is exploratory; 14 days give rough intervals.', '',
              'Files (data/results/rq1_lstm_residual/): predictions.jsonl (test probabilities, every seed), '
              'selection.jsonl (every candidate\'s seed-averaged validation log-likelihood, paired difference to the '
              'best and SE, per fold and condition), validation_scores.jsonl (per-target validation log-likelihoods '
              'behind the selection), fits.jsonl (best epoch, epochs trained and validation log-likelihoods of every '
              'fit), evaluation/ (shared-evaluator output; summary.json records SHA-256 of every input). '
              'training_curves.jsonl (per-epoch training and validation log-likelihood of every fit) is regenerated by '
              'the run and not committed (size). Model weights are not saved.', '']
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/rq1_lstm_residual.json')
    args = parser.parse_args()
    config = read_json(Path(args.config).resolve())
    if 'throughput' in config:
        choice = run_throughput(args.config)['choice']
        print(f"Chosen device: {choice['device']} x{choice['workers']}; estimated {choice['estimated_hours']:.1f} h "
              f"for {choice['full_study_fits']} fits")
    elif 'checks' in config:
        gates = run_checks(args.config)['gates']
        if not all(gates.values()):
            print(f'STOP: a check before the full run failed: {gates}')
    else:
        _, stop_reasons = run(args.config)
        if stop_reasons:
            print(f'STOP: {len(stop_reasons)} stop rule(s) fired. See the report.')


if __name__ == '__main__':
    main()
