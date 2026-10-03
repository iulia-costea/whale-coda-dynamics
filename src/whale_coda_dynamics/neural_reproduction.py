"""PLAN [M4] partner-only Assayag reproduction check.

Condition B (partner history only), refit with the LSTM architecture and
hyperparameters from Assayag et al. 2026 (Bioacoustics 35(4)) Table 1 and
Figure 4 (hidden dimension 64, two stacked LSTMs funnelling to hidden // 2,
dropout 0.05, Adam lr 0.001, batch 16), run on our leave-one-day-out folds
rather than their single random 70/15/15 split. This is a standalone
check, never mixed into the primary eight-condition lattice in neural.py,
which uses PLAN [M4]'s own fixed settings instead.

The target figure is Table 3's LSTM "Paired Coda Type (Accuracy)" row:
0.593, 0.766, 0.626, 0.633, 0.606, 0.706, 0.373, 0.720, 0.766 for zero to
eight preceding codas of the other whale, predicting the second whale's
next coda type. That is exactly this condition. The codaspred.ipynb
notebook's own LSTMModel_CODA cell uses different hyperparameters (hidden
128, embedding 32, dropout 0.15, lr 0.01, batch 4) than the published
Table 1; this check follows the published paper, which is citable and
carries the reported target, not the notebook.
"""
import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluation import classification_metrics, log_likelihood_bits
from .logistic import repeat_distribution
from .neural import encode_history, LOG2E
from .preprocessing.folds import prepare_fold
from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text


class AssayagHistoryEncoder(nn.Module):
    """Table 1 / Figure 4 architecture: two stacked multilayer LSTMs (hidden_dim
    then hidden_dim // 2, each with num_layers layers) with dropout before the
    head. The paper's Figure 4 also shows an intermediate hidden_dim -> hidden_dim
    // 2 fully connected layer between the two LSTMs; that bridge is omitted here
    to match the actual runnable LSTMModel_CODA implementation instead, which
    feeds lstm1's hidden_dim output directly into lstm2 with no FC in between."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=pad_idx)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seqs, lengths):
        out, _ = self.lstm1(self.embedding(seqs))
        out, _ = self.lstm2(out)
        idx = (lengths - 1).clamp(min=0)
        return self.dropout(out[torch.arange(seqs.shape[0]), idx])


class AssayagReproductionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, pad_idx):
        super().__init__()
        self.encoder = AssayagHistoryEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, pad_idx)
        self.output = nn.Linear(hidden_dim // 2, vocab_size)

    def forward(self, seqs, lengths):
        return self.output(self.encoder(seqs, lengths))


def fit_reproduction(train, validation, vocab, config, seed, max_len):
    torch.manual_seed(seed)
    pad_idx = len(vocab)
    model = AssayagReproductionModel(len(vocab), config['embedding_dim'], config['hidden_dim'],
                                     config['num_layers'], config['dropout_rate'], pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    y_train = torch.tensor([vocab.index(r['Ct']) for r in train], dtype=torch.long)
    y_val = torch.tensor([vocab.index(r['Ct']) for r in validation], dtype=torch.long)
    val_seq, val_len = encode_history(validation, 'B_history', vocab, max_len)
    val_seq, val_len = torch.from_numpy(val_seq), torch.from_numpy(val_len)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_val_ll, epochs_since_best, epoch = None, -float('inf'), 0, 0
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(train), config['batch_size']):
            idx = order[start:start + config['batch_size']]
            batch_rows = [train[i] for i in idx.tolist()]
            seqs, lengths = encode_history(batch_rows, 'B_history', vocab, max_len)
            logits = model(torch.from_numpy(seqs), torch.from_numpy(lengths))
            loss = nn.functional.cross_entropy(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_ll = (-nn.functional.cross_entropy(model(val_seq, val_len), y_val) * LOG2E).item()
        if val_ll > best_val_ll:
            best_val_ll, best_state, epochs_since_best = val_ll, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= config['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(epochs_trained=epoch, best_validation_log_likelihood_bits=best_val_ll,
                       early_stopped=epochs_since_best >= config['patience'])


def predict_reproduction(model, rows, vocab, max_len):
    """Softmax runs in float32; renormalize in float64 for exact-sum-to-one
    consistency with neural.py's predict(), even though this module scores
    directly rather than through the shared evaluate()."""
    model.eval()
    with torch.no_grad():
        seqs, lengths = encode_history(rows, 'B_history', vocab, max_len)
        logits = model(torch.from_numpy(seqs), torch.from_numpy(lengths))
        probs = nn.functional.softmax(logits, dim=1).numpy().astype(np.float64)
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
    predictions, baselines, convergence, targets = [], [], [], {}
    for fold in manifest['folds']:
        mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
        if policy != fold['vocabulary_policy']:
            raise ValueError('Recomputed fold policy differs from manifest')
        train = [r for r in mapped if r['split'] == 'train' and r['comparison_eligible']]
        validation = [r for r in mapped if r['split'] == 'validation' and r['comparison_eligible']]
        test = [r for r in mapped if r['split'] == 'test' and r['comparison_eligible']]
        vocab, prior = policy['vocabulary'], policy['label_prior']
        rate = sum(r['Ct'] == r['A_history'][-1] for r in train) / len(train)
        majority = max(sorted(prior), key=prior.get)
        for row in test:
            p = repeat_distribution(row['A_history'][-1], rate, prior)
            baselines.append(dict(coda_id=row['coda_id'], fold_id=fold['fold_id'], recording_date=row['recording_date'],
                                  target=row['Ct'], majority=majority, repeat_probabilities=p,
                                  repeat_prediction=max(sorted(p), key=p.get),
                                  repeat_log_likelihood_bits=log_likelihood_bits(p[row['Ct']]), training_repeat_rate=rate))
        for row in test:
            targets[row['coda_id']] = row['Ct']
        p_none = np.array([[prior[t] for t in vocab]] * len(test))
        for seed in config['seeds']:
            for row, probs in zip(test, p_none):
                predictions.append(dict(coda_id=row['coda_id'], fold_id=fold['fold_id'], condition='none',
                                        seed=seed, probabilities=dict(zip(vocab, map(float, probs)))))
            model, conv = fit_reproduction(train, validation, vocab, config, seed, max_len)
            p = predict_reproduction(model, test, vocab, max_len)
            convergence.append(dict(fold_id=fold['fold_id'], condition='B', seed=seed, **conv))
            for row, probs in zip(test, p):
                predictions.append(dict(coda_id=row['coda_id'], fold_id=fold['fold_id'], condition='B',
                                        seed=seed, probabilities=dict(zip(vocab, map(float, probs)))))
        print(f"Finished {fold['fold_id']}: {len(test)} test targets", flush=True)

    scored = []
    for p in predictions:
        target = targets[p['coda_id']]
        probs = p['probabilities']
        scored.append(dict(coda_id=p['coda_id'], condition=p['condition'], seed=p['seed'], target=target,
                           predicted=max(sorted(probs), key=probs.get), log_likelihood_bits=log_likelihood_bits(probs[target])))
    metrics = {}
    for condition in ('none', 'B'):
        per_seed = []
        for seed in config['seeds']:
            rs = [r for r in scored if r['condition'] == condition and r['seed'] == seed]
            per_seed.append(dict(classification_metrics([r['target'] for r in rs], [r['predicted'] for r in rs]),
                                 mean_log_likelihood_bits=math.fsum(r['log_likelihood_bits'] for r in rs) / len(rs)))
        metrics[condition] = {f: math.fsum(m[f] for m in per_seed) / len(per_seed)
                              for f in ('accuracy', 'macro_f1', 'mean_log_likelihood_bits')}
    baseline_metrics = {'majority': classification_metrics([r['target'] for r in baselines], [r['majority'] for r in baselines]),
                        'repeat_own_last': classification_metrics([r['target'] for r in baselines], [r['repeat_prediction'] for r in baselines])}
    baseline_metrics['repeat_own_last']['mean_log_likelihood_bits'] = sum(r['repeat_log_likelihood_bits'] for r in baselines) / len(baselines)

    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'predictions.jsonl', predictions)
    write_jsonl(output / 'baseline_predictions.jsonl', baselines)
    write_jsonl(output / 'convergence.jsonl', convergence)
    eval_output = output / 'evaluation'
    eval_output.mkdir(exist_ok=True)
    write_jsonl(eval_output / 'scored_predictions.jsonl', scored)
    summary = dict(units='bits per coda', seeds=config['seeds'], metrics=metrics, baseline_metrics=baseline_metrics,
                   configuration=config, versions={'numpy': np.__version__, 'torch': torch.__version__},
                   input_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, manifest_path, path)})
    write_text(eval_output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    write_report(root, config, summary, baseline_metrics, convergence)
    return summary


TARGET_PAIRED_CODA_TYPE_ACCURACY = {
    0: 0.593, 1: 0.766, 2: 0.626, 3: 0.633, 4: 0.606, 5: 0.706, 6: 0.373, 7: 0.720, 8: 0.766,
}


def write_report(root, config, summary, baseline_metrics, convergence):
    b = summary['metrics']['B']
    repeat = baseline_metrics['repeat_own_last']
    target_values = list(TARGET_PAIRED_CODA_TYPE_ACCURACY.values())
    lines = ['# RQ1 partner-only Assayag reproduction check', '',
             "Condition B (partner history only) refit with the architecture and hyperparameters in Assayag "
             f"et al. 2026 (Bioacoustics 35(4)), Table 1 and Figure 4: embedding {config['embedding_dim']} "
             f"(not stated in the paper for this task), two stacked LSTMs with {config['num_layers']} layers "
             f"each at hidden {config['hidden_dim']} then {config['hidden_dim'] // 2}, dropout "
             f"{config['dropout_rate']}, Adam learning rate {config['learning_rate']}, batch "
             f"{config['batch_size']}. Run on our fourteen leave-one-day-out folds, not their single random "
             "70/15/15 split. This is a standalone check, not part of the primary eight-condition lattice in "
             "reports/rq1_lstm.md.", '',
             "The target is Table 3's LSTM \"Paired Coda Type (Accuracy)\" row: predicting the second whale's "
             "next coda type from the first whale's preceding codas, which is exactly this condition. The paper "
             "reports one accuracy per preceding-coda count (0 through 8), not a single figure, so the "
             f"comparison below is against that range ({min(target_values):.3f} to {max(target_values):.3f}). "
             "codaspred.ipynb's own LSTMModel_CODA cell uses different hyperparameters (hidden 128, embedding "
             "32, dropout 0.15, lr 0.01, batch 4) than this published table; this check follows the paper.", '']

    lines += ['## Result', '', '| Metric | Value |', '|---|---:|',
             f"| LL bits/coda | {b['mean_log_likelihood_bits']:.5f} |",
             f"| Accuracy | {b['accuracy']:.3%} |", f"| Macro-F1 | {b['macro_f1']:.4f} |",
             f"| repeat-own-last accuracy (for scale) | {repeat['accuracy']:.3%} |",
             f"| majority-class accuracy (for scale) | {baseline_metrics['majority']['accuracy']:.3%} |", '']

    lines += ['## Target (Assayag et al. 2026, Table 3, LSTM, Paired Coda Type)', '',
             '| Preceding codas | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
             '| Accuracy | ' + ' | '.join(f"{v:.3f}" for v in target_values) + ' |', '',
             f"Our accuracy ({b['accuracy']:.3%}) uses up to {config.get('history_limit', 10)} preceding partner "
             "codas per row (whatever each row actually has), not a fixed count, so it is not paired one to one "
             "with a single column above; it is reported here for comparison against the range as a whole, not "
             "interpreted further.", '']

    epochs = [c['epochs_trained'] for c in convergence]
    lines += [f"{len(convergence)} fits (14 folds times {len(config['seeds'])} seeds); epochs trained ranged "
             f"{min(epochs)} to {max(epochs)}, {sum(c['early_stopped'] for c in convergence)} stopped early on "
             f"patience. Source, fold-manifest and configuration SHA-256 hashes are recorded in "
             "evaluation/summary.json.", '']

    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/rq1_lstm_reproduction.json')
    run(parser.parse_args().config)


if __name__ == '__main__':
    main()
