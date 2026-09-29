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
import copy
import hashlib
import json
import itertools
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluation import evaluate, classification_metrics, log_likelihood_bits, day_bootstrap_bits
from .logistic import Encoder, repeat_distribution, check_stop_conditions, classify_add_input_contrasts, describe, format_ci
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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, seqs, lengths):
        out, _ = self.lstm(self.embedding(seqs))
        # Padding after each row's true length cannot influence earlier steps
        # (the LSTM is causal), so indexing the true last step needs no packing.
        idx = (lengths - 1).clamp(min=0)
        return out[torch.arange(seqs.shape[0], device=seqs.device), idx]


class TimingEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

    def forward(self, x):
        return self.net(x)


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


def build_batch(rows, vocab, fields, max_len, timing_encoder):
    batch = {}
    if 'A_history' in fields:
        seqs, lengths = encode_history(rows, 'A_history', vocab, max_len)
        batch['a_seq'], batch['a_len'] = torch.from_numpy(seqs), torch.from_numpy(lengths)
    if 'B_history' in fields:
        seqs, lengths = encode_history(rows, 'B_history', vocab, max_len)
        batch['b_seq'], batch['b_len'] = torch.from_numpy(seqs), torch.from_numpy(lengths)
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
        lines.append(f"| {c} | {check['mean_bits_per_coda']:.5f} | {format_ci(check['ci_bits_per_coda'])} | "
                     f"{check['positive_days']}/{check['days']} | {summary['metrics'][c]['accuracy']:.3%} | "
                     f"{summary['metrics'][c]['macro_f1']:.4f} |")

    lines += ['', '## All twelve add-one-input contrasts', '',
             '| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |', '|---|---:|---:|---|---:|']
    for d in classify_add_input_contrasts(summary):
        lines.append(f"| adding {d['added']} to {describe(d['base'])} | {d['mean']:.5f} | {d['unweighted_mean']:.5f} | "
                     f"{format_ci([d['low'], d['high']])} | {d['positive_days']}/{d['days']} |")

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
                 'No A-condition falls below repeat-own-last in log-likelihood, no add-one-input '
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


class ConditionModelV2(nn.Module):
    """The same additive linear head, partitioned by source for Adam decay.

    Each head has no bias; one shared output bias belongs to the base group.
    Creating history parameters before S keeps their initial values independent
    of whether S is enabled, useful for the validation suppression diagnostic.
    """
    def __init__(self, vocab_size, fields, config, timing_dim):
        super().__init__()
        self.fields = fields
        hidden = config['hidden_dim']
        for letter in ('a', 'b'):
            if letter.upper() + '_history' in fields:
                setattr(self, letter + '_encoder', HistoryEncoder(vocab_size, config['embedding_dim'], hidden, vocab_size))
                setattr(self, letter + '_output', nn.Linear(hidden, vocab_size, bias=False))
                setattr(self, 'beta' if letter == 'a' else 'gamma', nn.Parameter(torch.zeros(())))
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        if timing_dim is not None:
            self.s_encoder = TimingEncoder(timing_dim, hidden)
            self.s_output = nn.Linear(hidden, vocab_size, bias=False)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, batch):
        count = next(iter(batch.values())).shape[0]
        logits = self.output_bias.unsqueeze(0).expand(count, -1)
        for letter in ('a', 'b'):
            if not hasattr(self, letter + '_encoder'):
                continue
            seq, lengths = batch[letter + '_seq'], batch[letter + '_len']
            encoded = getattr(self, letter + '_encoder')(seq, lengths)
            logits = logits + getattr(self, letter + '_output')(self.dropout(encoded))
            last = seq.gather(1, (lengths - 1).unsqueeze(1))
            weight = self.beta if letter == 'a' else self.gamma
            logits = logits.scatter_add(1, last, weight.expand(count, 1))
        if hasattr(self, 's_encoder'):
            logits = logits + self.s_output(self.dropout(self.s_encoder(batch['s_features'])))
        return logits

    def optimizer_groups(self, config):
        base, timing = [], []
        for name, parameter in self.named_parameters():
            (timing if name.startswith(('s_encoder.', 's_output.')) else base).append(parameter)
        groups = [{'params': base, 'weight_decay': config['weight_decay']}]
        if timing:
            groups.append({'params': timing, 'weight_decay': config['s_weight_decay']})
        return groups


def v2_probabilities(model, batch):
    model.eval()
    with torch.no_grad():
        # Double precision softmax prevents avoidable underflow in evaluation.
        return torch.softmax(model(batch).double(), dim=1).cpu().numpy()


def mean_ll_bits(probabilities, rows, vocab):
    index = {label: i for i, label in enumerate(vocab)}
    return sum(log_likelihood_bits(float(p[index[r['Ct']]])) for p, r in zip(probabilities, rows)) / len(rows)


def fit_condition_v2(train, validation, vocab, fields, config, seed, max_len):
    """Train only on train; choose the epoch only on validation. Never sees test."""
    if not train or any(r['split'] != 'train' for r in train):
        raise ValueError('Training requires nonempty training rows')
    if not validation or any(r['split'] != 'validation' for r in validation):
        raise ValueError('Early stopping requires nonempty validation rows')
    device = config.get('device', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(config.get('torch_threads', 1))
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    timing_fields = [f for f in fields if not f.endswith('_history')]
    timing_encoder = Encoder(train, vocab, timing_fields) if timing_fields else None
    model = ConditionModelV2(len(vocab), fields, config,
                             len(timing_encoder.names) if timing_encoder else None).to(device)
    optimizer = torch.optim.Adam(model.optimizer_groups(config), lr=config['learning_rate'])
    batches = [{k: v.to(device) for k, v in build_batch(rs, vocab, fields, max_len, timing_encoder).items()}
               for rs in (train, validation)]
    y = torch.tensor([vocab.index(r['Ct']) for r in train], device=device)
    generator = torch.Generator().manual_seed(seed)
    best_ll, best_state, best_epoch, stale = -float('inf'), None, 0, 0
    epochs = []
    started = time.perf_counter()
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).to(device)
        for start in range(0, len(train), config['batch_size']):
            indices = order[start:start + config['batch_size']]
            batch = {k: v[indices] for k, v in batches[0].items()}
            loss = nn.functional.cross_entropy(model(batch), y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_ll = mean_ll_bits(v2_probabilities(model, batches[0]), train, vocab)
        val_ll = mean_ll_bits(v2_probabilities(model, batches[1]), validation, vocab)
        epochs.append(dict(epoch=epoch, training_log_likelihood_bits=train_ll,
                           validation_log_likelihood_bits=val_ll))
        if val_ll > best_ll:
            best_ll, best_state, best_epoch, stale = val_ll, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= config['patience']:
                break
    if best_state is None:
        raise ValueError('No valid validation epoch')
    model.load_state_dict(best_state)
    model.eval()
    if device == 'cuda':
        torch.cuda.synchronize()
    convergence = dict(epochs_trained=len(epochs), best_epoch=best_epoch,
                       best_validation_log_likelihood_bits=best_ll,
                       early_stopped=stale >= config['patience'],
                       hit_epoch_cap=len(epochs) == config['max_epochs'],
                       seconds=time.perf_counter() - started, device=device,
                       beta=float(model.beta.detach().cpu()) if hasattr(model, 'beta') else None,
                       gamma=float(model.gamma.detach().cpu()) if hasattr(model, 'gamma') else None)
    return model, timing_encoder, convergence, epochs


def v2_settings(fields, config):
    timing = any(not f.endswith('_history') for f in fields)
    s_axis = config['s_weight_decay_grid'] if timing else [None]
    return [dict(hidden_dim=h, dropout=d, weight_decay=w, **({'s_weight_decay': s} if s is not None else {}))
            for h, d, w, s in itertools.product(config['hidden_grid'], config['dropout_grid'],
                                               config['weight_decay_grid'], s_axis)]


def _v2_candidate_fit(job):
    train, validation, vocab, fields, config, seed, max_len, setting_id, setting = job
    model, encoder, convergence, epochs = fit_condition_v2(train, validation, vocab, fields,
                                                          dict(config, **setting), seed, max_len)
    return dict(setting_id=setting_id, setting=setting, seed=seed, convergence=convergence, epochs=epochs,
                state={k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()})


def select_neural_setting(train, validation, vocab, fields, config, max_len):
    """Averaged-seed validation selection. Test rows cannot enter this interface."""
    settings = v2_settings(fields, config)
    jobs = [(train, validation, vocab, fields, config, seed, max_len, i, setting)
            for i, setting in enumerate(settings) for seed in config['seeds']]
    workers = config.get('workers', 1)
    if workers > 1:
        if config.get('device', 'cpu') != 'cpu':
            raise ValueError('Concurrent CPU fits require device=cpu')
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            fits = list(pool.map(_v2_candidate_fit, jobs))
    else:
        fits = [_v2_candidate_fit(job) for job in jobs]
    candidates = []
    for i, setting in enumerate(settings):
        values = [f['convergence']['best_validation_log_likelihood_bits'] for f in fits if f['setting_id'] == i]
        candidates.append(dict(setting_id=i, setting=setting, validation_log_likelihood_bits=sum(values) / len(values)))
    def key(candidate):
        s = candidate['setting']
        return (candidate['validation_log_likelihood_bits'], -s['hidden_dim'], s['dropout'],
                s['weight_decay'], s.get('s_weight_decay', 0))
    chosen = max(candidates, key=key)
    return chosen, candidates, fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/rq1_lstm.json')
    args = parser.parse_args()
    if read_json(args.config).get('version') == 'v2':
        from .neural_v2 import run_v2
        run_v2(args.config)
        return
    _, stop_reasons, underperform = run(args.config)
    total = len(stop_reasons) + len(underperform)
    if total:
        print(f"STOP: {total} condition(s) need review before interpretation. See the report.")


if __name__ == '__main__':
    main()
