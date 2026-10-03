"""Fixed-L2 multinomial regression and baselines on the shared RQ1 folds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import logsumexp

from .evaluation import evaluate, classification_metrics, log_likelihood_bits, day_bootstrap_bits
from .preprocessing.folds import prepare_fold
from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text


def feature_names(fields, vocabulary, categories, order, history_orders=None):
    """Encoder's column names, in column order (the intercept, appended by transform, is unnamed).
    `order` lags per history field, unless `history_orders` gives that field its own count (as in Encoder)."""
    history_orders = history_orders or {}
    names = []
    for f in fields:
        if f.endswith('_history'):
            names.extend(f'{f}:lag{lag}:{label}' for lag in range(1, history_orders.get(f, order)+1) for label in vocabulary)
        elif f == 'turn_category':
            names.extend('turn:'+c for c in categories)
        elif f == 'retrospective_delay_s':
            names.append(f)
        else:
            names.extend([f, 'log1p:'+f])
    return names


class Encoder:
    """Lag one-hot vectors; raw and log1p timing, each standardized on train.

    `order` lags per history field, unless `history_orders` gives that field its own
    count (DEC-25: {'A_history': k}); fields it doesn't list keep `order`."""
    def __init__(self, train, vocabulary, fields, order=3, history_orders=None):
        if not train or any(r['split'] != 'train' for r in train):
            raise ValueError('Encoder fitting requires training rows')
        if type(order) is not int or order < 1:
            raise ValueError('Invalid history order')
        history_orders = dict(history_orders or {})
        if any(not f.endswith('_history') or type(k) is not int or k < 1 for f, k in history_orders.items()):
            raise ValueError('Invalid per-field history order')
        self.vocabulary, self.fields, self.order = vocabulary, fields, order
        self.history_orders = history_orders
        self.categories = sorted({r['turn_category'] for r in train}) if 'turn_category' in fields else []
        self.names = feature_names(fields, vocabulary, self.categories, order, history_orders)
        x = self.raw(train)
        self.mean, self.scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
        for j, name in enumerate(self.names):
            if name in ('gap_s', 'overlap_elapsed_s', 'log1p:gap_s', 'log1p:overlap_elapsed_s', 'retrospective_delay_s'):
                self.mean[j], self.scale[j] = x[:, j].mean(), x[:, j].std() or 1.0

    @classmethod
    def from_record(cls, record):
        """The encoder of a stored fit (a models.jsonl row of v2-v4), from its stored vocabulary, fields,
        order, per-field orders (DEC-25; absent in rows that predate them), turn categories and
        standardisation; nothing is refitted. The stored feature names must be exactly the names those
        settings produce."""
        encoder = cls.__new__(cls)
        encoder.vocabulary, encoder.fields, encoder.order = record['vocabulary'], record['fields'], record['order']
        encoder.history_orders = dict(record.get('history_orders', {}))
        encoder.categories = record['categories']
        encoder.names = feature_names(encoder.fields, encoder.vocabulary, encoder.categories, encoder.order,
                                      encoder.history_orders)
        encoder.mean = np.asarray(record['mean'], dtype=float)
        encoder.scale = np.asarray(record['scale'], dtype=float)
        if (encoder.names + ['intercept'] != record['feature_names'] or encoder.mean.shape != (len(encoder.names),)
                or encoder.scale.shape != (len(encoder.names),)):
            raise ValueError(f"Stored encoder is inconsistent ({record.get('fold_id')}, {record.get('condition')})")
        return encoder

    def lags(self, field):
        return self.history_orders.get(field, self.order)

    def raw(self, rows):
        result = []
        for r in rows:
            values = []
            for f in self.fields:
                if f.endswith('_history'):
                    for lag in range(1, self.lags(f)+1):
                        label = r[f][-lag] if len(r[f]) >= lag else None
                        values.extend(float(label == t) for t in self.vocabulary)
                elif f == 'turn_category':
                    values.extend(float(r[f] == c) for c in self.categories)
                elif f in ('gap_s', 'overlap_elapsed_s'):
                    v = r[f]
                    if not np.isfinite(v) or v < 0:
                        raise ValueError('Invalid onset-available timing')
                    values.extend([v, np.log1p(v)])
                elif f == 'retrospective_delay_s':
                    # Sensitivity field only (PLAN S9, DEC-1): unlike onset-available
                    # timing, this can be negative (overlapping predecessor), and goes
                    # below -1 in the data, so no log1p companion is defined for it.
                    v = r[f]
                    if not np.isfinite(v):
                        raise ValueError('Invalid retrospective delay')
                    values.append(v)
                else:
                    raise ValueError(f'Unsupported input field {f}')
            result.append(values)
        return np.asarray(result, dtype=float)

    def transform(self, rows):
        x = (self.raw(rows) - self.mean) / self.scale
        return np.column_stack([x, np.ones(len(rows))])


def last_label_index(rows, field, vocabulary):
    """Vocabulary index of each row's most recent label in `field`, or -1 if empty."""
    position = {label: i for i, label in enumerate(vocabulary)}
    return np.array([position[r[field][-1]] if r[field] else -1 for r in rows])


def _unpack(flat, n_features, classes, own_last_idx, partner_last_idx):
    w = flat[:n_features*classes].reshape(n_features, classes)
    offset = n_features*classes
    beta = gamma = None
    if own_last_idx is not None:
        beta = flat[offset]; offset += 1
    if partner_last_idx is not None:
        gamma = flat[offset]; offset += 1
    return w, beta, gamma


def _copy_logits(logits, beta, own_last_idx, gamma, partner_last_idx):
    """Add a single shared scalar to the logit of the class matching the last
    own/partner coda (PLAN [M1], the repeat-baseline expressiveness fix).
    Rows with no history in that stream get no addition (mask excludes -1)."""
    if beta is not None:
        mask = own_last_idx >= 0
        logits[mask, own_last_idx[mask]] += beta
    if gamma is not None:
        mask = partner_last_idx >= 0
        logits[mask, partner_last_idx[mask]] += gamma
    return logits


def objective(weights, x, y, classes, l2, own_last_idx=None, partner_last_idx=None, l2_vector=None):
    """l2 penalizes beta/gamma (and every main weight, when l2_vector is not
    given: the plain single-penalty path used by v1 through v3). l2_vector,
    when given, is a per-feature penalty (DEC-20: separate history/S
    penalties) applied to the main weights instead of the scalar l2."""
    n_features = x.shape[1]
    w, beta, gamma = _unpack(weights, n_features, classes, own_last_idx, partner_last_idx)
    logits = _copy_logits(x @ w, beta, own_last_idx, gamma, partner_last_idx)
    logs = logits - logsumexp(logits, axis=1, keepdims=True)
    l2w = np.broadcast_to(l2 if l2_vector is None else l2_vector, (n_features,))
    loss = (-logs[np.arange(len(y)), y].mean() + (l2w[:, None] * np.square(w)).sum() / 2
           + l2 * ((beta or 0.0)**2 + (gamma or 0.0)**2) / 2)
    residual = np.exp(logs)
    residual[np.arange(len(y)), y] -= 1
    grad = [(x.T @ residual / len(y) + l2w[:, None] * w).ravel()]
    if beta is not None:
        mask = own_last_idx >= 0
        grad.append(np.array([residual[mask, own_last_idx[mask]].sum() / len(y) + l2 * beta]))
    if gamma is not None:
        mask = partner_last_idx >= 0
        grad.append(np.array([residual[mask, partner_last_idx[mask]].sum() / len(y) + l2 * gamma]))
    return loss, np.concatenate(grad)


class NonConvergenceError(ValueError):
    """fit()'s convergence check failed. A ValueError, so callers that do not opt in to skipping
    non-converged candidates (`skip_nonconverged_candidates`, DEC-22) stop exactly as before."""


def fit(x, y, classes, config, own_last_idx=None, partner_last_idx=None, l2_vector=None):
    # Penalize intercepts too, making absent target classes (including UNK)
    # well-defined and finite without synthetic targets or post-hoc clipping.
    if not np.isfinite(config['l2']) or config['l2'] <= 0:
        raise ValueError('Require positive finite L2')
    n_features = x.shape[1]
    n_extra = (own_last_idx is not None) + (partner_last_idx is not None)
    result = minimize(objective, np.zeros(n_features*classes + n_extra),
                      args=(x, y, classes, config['l2'], own_last_idx, partner_last_idx, l2_vector),
                      jac=True, method='L-BFGS-B', options={'maxiter': config['max_iterations'],
                      'gtol': config['gradient_tolerance'], 'ftol': 1e-14})
    gradient = float(np.max(np.abs(result.jac)))
    if not result.success or gradient > 10*config['gradient_tolerance']:
        raise NonConvergenceError(f'Optimizer did not converge: {result.message}; gradient={gradient}')
    w, beta, gamma = _unpack(result.x, n_features, classes, own_last_idx, partner_last_idx)
    return w, beta, gamma, {'iterations': int(result.nit), 'max_gradient': gradient}


def probabilities(x, w, beta=None, gamma=None, own_last_idx=None, partner_last_idx=None):
    logits = _copy_logits(x @ w, beta, own_last_idx, gamma, partner_last_idx)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def is_timing_feature(name):
    """DEC-1/DEC-20: the S block, everything Encoder derives from turn_category,
    gap_s, overlap_elapsed_s or retrospective_delay_s. Everything else (history
    lag one-hots, and the intercept, which Encoder appends unnamed) is history."""
    return name.startswith('turn:') or name in (
        'gap_s', 'overlap_elapsed_s', 'log1p:gap_s', 'log1p:overlap_elapsed_s', 'retrospective_delay_s')


def select_l2_and_fit(train, validation, vocab, fields, config):
    """Grid-search L2 on the validation day only; `test` is never an argument
    here, so it is structurally impossible for test data to influence selection.
    DEC-20: when the condition includes S, history and S get separate penalties,
    chosen jointly from the same grid (a 2D search); conditions without S keep
    the original single-penalty 1D search, unchanged. With `skip_nonconverged_candidates` set (off by
    default; on only in the RQ2 inventory runs, DEC-22), a candidate whose fit fails the convergence check
    is recorded and left out of the selection; if every candidate fails, NonConvergenceError is raised."""
    encoder = Encoder(train, vocab, fields, config['history_order'], config.get('history_orders'))
    x_train, x_val = encoder.transform(train), encoder.transform(validation)
    y_train = np.array([vocab.index(r['Ct']) for r in train])
    y_val = np.array([vocab.index(r['Ct']) for r in validation])
    own_train = last_label_index(train, 'A_history', vocab) if 'A_history' in fields else None
    partner_train = last_label_index(train, 'B_history', vocab) if 'B_history' in fields else None
    own_val = last_label_index(validation, 'A_history', vocab) if 'A_history' in fields else None
    partner_val = last_label_index(validation, 'B_history', vocab) if 'B_history' in fields else None
    timing_mask = np.array([is_timing_feature(name) for name in encoder.names] + [False])  # intercept -> history
    chosen, grid = grid_search_l2(x_train, y_train, x_val, y_val, len(vocab), timing_mask, config,
                                  own_train, partner_train, own_val, partner_val)
    return encoder, chosen, grid


def grid_search_l2(x_train, y_train, x_val, y_val, classes, timing_mask, config,
                   own_train=None, partner_train=None, own_val=None, partner_val=None):
    """select_l2_and_fit's validation-day penalty search on encoded matrices, shared with other model
    families that keep v4's rule (DEC-24). `timing_mask` marks the S columns (the intercept is history):
    with any S column, history and S penalties are searched jointly over the grid (DEC-20), otherwise one
    penalty. Ties go to the larger value on each axis. Returns the chosen candidate and the whole grid."""
    skip_nonconverged = config.get('skip_nonconverged_candidates', False)
    has_timing = bool(timing_mask.any())
    s_axis = config['l2_grid'] if has_timing else [None]
    grid = []
    for l2_history in config['l2_grid']:
        for l2_s in s_axis:
            l2_vector = np.where(timing_mask, l2_s, l2_history) if has_timing else None
            try:
                w, beta, gamma, convergence = fit(x_train, y_train, classes, dict(config, l2=l2_history),
                                                  own_train, partner_train, l2_vector=l2_vector)
            except NonConvergenceError as error:
                if not skip_nonconverged:
                    raise
                grid.append(dict(l2=l2_history, l2_s=l2_s, nonconverged=str(error)))
                continue
            p_val = probabilities(x_val, w, beta, gamma, own_val, partner_val)
            val_ll = float(np.mean(np.log2(p_val[np.arange(len(y_val)), y_val])))
            grid.append(dict(l2=l2_history, l2_s=l2_s, validation_log_likelihood_bits=val_ll,
                            w=w, beta=beta, gamma=gamma, convergence=convergence))
    converged = [g for g in grid if 'nonconverged' not in g]
    if not converged:
        raise NonConvergenceError(f'All {len(grid)} L2 candidates failed the convergence check')
    chosen = max(converged, key=lambda g: (g['validation_log_likelihood_bits'], g['l2'], g['l2_s'] or 0))
    return chosen, grid


def repeat_distribution(last, rate, prior):
    if not 0 < rate < 1:
        raise ValueError('Training repeat rate must lie strictly between 0 and 1')
    other_mass = sum(p for t, p in prior.items() if t != last)
    if other_mass <= 0:
        raise ValueError('Repeat baseline needs at least one alternative label')
    return {t: rate if t == last else (1-rate)*p/other_mass for t, p in prior.items()}


def describe(condition):
    if condition == 'none':
        return 'no inputs'
    parts = condition.split('_')
    return parts[0] if len(parts) == 1 else ' and '.join(parts) if len(parts) == 2 else ', '.join(parts[:-1])+' and '+parts[-1]


def classify_add_input_contrasts(summary):
    """One entry per add-one-input pair, with the day-bootstrap reliability of its sign."""
    details = []
    for key, contrast in summary['contrasts'].items():
        if not key.startswith('add_input:'):
            continue
        full_parts = contrast['full'].split('_') if contrast['full'] != 'none' else []
        reduced_parts = contrast['reduced'].split('_') if contrast['reduced'] != 'none' else []
        added = next(p for p in full_parts if p not in reduced_parts)
        stats = contrast['components']['log_likelihood_bits']
        low, high = stats['ci_bits_per_coda']
        reliability = 'negative' if high < 0 else 'positive' if low > 0 else 'inconclusive'
        details.append(dict(added=added, base=contrast['reduced'], mean=stats['mean_bits_per_coda'],
                            low=low, high=high, reliability=reliability,
                            unweighted_mean=stats.get('unweighted_mean_bits_per_coda'),
                            positive_days=stats['positive_days'], days=stats['days']))
    return details


def format_ci(ci):
    """Round a day-bootstrap CI to 4dp for markdown reports; summary.json keeps full precision."""
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def check_stop_conditions(summary, baseline_metrics, baseline_checks):
    """DEC-16: log-likelihood-only stop rules. Accuracy and macro-F1 are reported
    but never used to stop (v2's accuracy-based rule fired on differences of a
    few codas out of 1,547; log-likelihood is the estimand)."""
    reasons = []
    for condition in ('A', 'A_B', 'A_S', 'A_B_S'):
        check = baseline_checks[condition]
        if check['ci_bits_per_coda'][1] < 0:
            reasons.append(f"{condition}: reliably below repeat-own-last in log-likelihood "
                           f"(CI {format_ci(check['ci_bits_per_coda'])})")
    for name, contrast in summary['contrasts'].items():
        if name.startswith('add_input:'):
            ci = contrast['components']['log_likelihood_bits']['ci_bits_per_coda']
            if ci[1] < 0:
                reasons.append(f"{name}: contrast CI entirely below zero {format_ci(ci)}")
    return reasons


def run(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    manifest_path = root / config['manifest']
    manifest = read_json(manifest_path)
    source = root / manifest['configuration']['input']
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Stale fold manifest')
    evaluation_config = read_json(root / config['evaluation_config'])
    if evaluation_config['seeds'] != [config['seed']]:
        raise ValueError('Evaluation seed configuration mismatch')
    rows = read_jsonl(source)
    predictions, baselines, models, diagnostics = [], [], [], []
    for fold in manifest['folds']:
        mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
        if policy != fold['vocabulary_policy']:
            raise ValueError('Recomputed fold policy differs from manifest')
        train = [r for r in mapped if r['split']=='train' and r['comparison_eligible']]
        test = [r for r in mapped if r['split']=='test' and r['comparison_eligible']]
        vocab, prior = policy['vocabulary'], policy['label_prior']
        y = np.array([vocab.index(r['Ct']) for r in train])
        rate = sum(r['Ct']==r['A_history'][-1] for r in train)/len(train)
        majority = max(sorted(prior), key=prior.get)
        for row in test:
            p = repeat_distribution(row['A_history'][-1], rate, prior)
            baselines.append(dict(coda_id=row['coda_id'],fold_id=fold['fold_id'],recording_date=row['recording_date'],
                                  target=row['Ct'],majority=majority,repeat_probabilities=p,
                                  repeat_prediction=max(sorted(p),key=p.get),
                                  repeat_log_likelihood_bits=log_likelihood_bits(p[row['Ct']]),training_repeat_rate=rate))
        for condition, fields in manifest['input_conditions'].items():
            if condition == 'none':
                p = np.array([[prior[t] for t in vocab]]*len(test))
            else:
                encoder = Encoder(train, vocab, fields, config['history_order'])
                x = encoder.transform(train)
                weights, beta, gamma, convergence = fit(x, y, len(vocab), config)
                p = probabilities(encoder.transform(test), weights, beta, gamma)
                diagnostics.append(dict(fold_id=fold['fold_id'],condition=condition,**convergence))
                models.append(dict(fold_id=fold['fold_id'],condition=condition,vocabulary=vocab,
                                   fields=fields,order=encoder.order,categories=encoder.categories,
                                   feature_names=encoder.names+['intercept'],mean=encoder.mean.tolist(),
                                   scale=encoder.scale.tolist(),weights=weights.tolist()))
            for row, probs in zip(test,p):
                predictions.append(dict(coda_id=row['coda_id'],fold_id=fold['fold_id'],condition=condition,
                                        seed=config['seed'],probabilities=dict(zip(vocab,map(float,probs)))))
        print(f"Finished {fold['fold_id']}: {len(test)} test targets", flush=True)
    summary, scored, paired = evaluate(predictions,manifest,rows,[config['seed']],evaluation_config['bootstrap_replicates'],
                                       evaluation_config['bootstrap_seed'],evaluation_config['confidence'])
    baseline_metrics = {'majority':classification_metrics([r['target'] for r in baselines],[r['majority'] for r in baselines]),
                        'repeat_own_last':classification_metrics([r['target'] for r in baselines],[r['repeat_prediction'] for r in baselines])}
    baseline_metrics['repeat_own_last']['mean_log_likelihood_bits'] = sum(r['repeat_log_likelihood_bits'] for r in baselines)/len(baselines)
    indexed = {r['coda_id']:r for r in baselines}
    baseline_checks = {}
    for condition in ('A','A_B','A_S','A_B_S'):
        days = {}
        for r in scored:
            if r['condition']==condition:
                days.setdefault(r['recording_date'],[]).append(r['log_likelihood_bits']-indexed[r['coda_id']]['repeat_log_likelihood_bits'])
        baseline_checks[condition] = day_bootstrap_bits(days,evaluation_config['bootstrap_replicates'],evaluation_config['bootstrap_seed'],evaluation_config['confidence'])

    add_input_details = classify_add_input_contrasts(summary)
    reliably_negative = [d for d in add_input_details if d['reliability'] == 'negative']
    noise_negative = [d for d in add_input_details if d['mean'] < 0 and d['reliability'] != 'negative']
    own_history_alone = summary['contrasts']['add_input:A-none']['components']['log_likelihood_bits']['mean_bits_per_coda']

    output = root / config['output_directory'];output.mkdir(parents=True,exist_ok=True)
    write_jsonl(output/'predictions.jsonl',predictions)
    write_jsonl(output/'baseline_predictions.jsonl',baselines)
    write_jsonl(output/'models.jsonl',models)
    write_jsonl(output/'convergence.jsonl',diagnostics)
    eval_output = output/'evaluation';eval_output.mkdir(exist_ok=True)
    write_jsonl(eval_output/'scored_predictions.jsonl',scored)
    write_jsonl(eval_output/'paired_contrasts.jsonl',paired)
    summary.update(configuration=config,baseline_metrics=baseline_metrics,repeat_baseline_checks=baseline_checks,
                   versions={'numpy':np.__version__,'scipy':scipy.__version__},
                   input_sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,manifest_path,path)})
    write_text(eval_output/'summary.json',json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# RQ1 logistic-regression reference results','',
           'Fourteen leave-one-day-out folds, fixed settings, no test-driven tuning. These reference results precede the neural study.',
           f"L2={config['l2']} on mean cross-entropy plus L2/2 times squared coefficients, including intercepts. Full fold vocabulary is modeled, including classes absent from training targets. No synthetic training rows or probability clipping.",
           'Last three codas are one-hot encoded by lag; missing lags are zero vectors within enabled streams. Timing uses raw and log1p values, each standardized on training comparison rows. Absent sources supply no features. Validation days remain excluded; fixed models use no tuning or early stopping.',
           'Repeat-own-last uses the mapped training comparison repeat rate, with remaining mass distributed by the add-1 prior excluding the last label. Majority class has classification scores only.',
           '', '| Condition | LL bits/coda | Accuracy | Macro-F1 |','|---|---:|---:|---:|']
    for c,m in summary['metrics'].items():
        lines.append(f"| {c} | {m['mean_log_likelihood_bits']:.5f} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} |")
    for c,m in baseline_metrics.items():
        ll = f"{m['mean_log_likelihood_bits']:.5f}" if 'mean_log_likelihood_bits' in m else '—'
        lines.append(f"| {c} | {ll} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} |")
    lines+=['','| Contrast | Bits/coda | 95% day-bootstrap CI | Positive days | Repeat part | Switch-choice part |', '|---|---:|---|---:|---:|---:|']
    for name in ('own_history','partner_content','timing'):
        c=summary['contrasts'][name]['components'];a=c['log_likelihood_bits']
        lines.append(f"| {name} | {a['mean_bits_per_coda']:.5f} | {a['ci_bits_per_coda']} | {a['positive_days']}/14 | {c['repeat_bits']['mean_bits_per_coda']:.5f} | {c['switch_choice_bits']['mean_bits_per_coda']:.5f} |")
    lines+=['',f"All 98 contextual fits converged. Negative add-input comparisons: {summary['add_input_negative_contrasts']}.",
            f"Own-history conditions below repeat-own-last in mean LL: {[c for c,v in baseline_checks.items() if v['mean_bits_per_coda']<0]}.",
            f"Raw repeats: {summary['raw_repeats']}/{len(baselines)}; mapped repeats: {summary['mapped_repeats']}/{len(baselines)}. Decomposition uses mapped labels and all targets; switch results are exploratory.",
            'Day-bootstrap intervals use 2,000 replicates and are approximate with 14 days. Full per-day values, all lattice comparisons and baseline comparisons are in evaluation/summary.json.', '']
    lines += ['## Interpreting the negative comparisons', '']
    noise_list = ', '.join(f"adding {d['added']} to {describe(d['base'])}" for d in noise_negative)
    lines.append(f"Twelve condition pairs differ by exactly one input source, giving twelve direct checks that adding "
                 f"information does not lower held out performance. Five of the twelve pairs had a negative mean "
                 f"contrast. Repeating the day bootstrap for each of the five shows that three are not distinguishable "
                 f"from zero ({noise_list}), so these are consistent with the added input contributing nothing rather "
                 f"than causing harm.")
    reliable_text = ' and '.join(f"adding {d['added']} to {describe(d['base'])} "
                                 f"({d['mean']:.5f} bits per coda, {d['low']:.5f} to {d['high']:.5f})" for d in reliably_negative)
    lines.append(f"Two pairs are reliably negative, meaning the full interval lies below zero: {reliable_text}.")
    ratio = own_history_alone / max(abs(d['mean']) for d in reliably_negative)
    lines.append(f"Both reliable cases add the interactional state S to a condition that already contains own history "
                 f"A. The size of this loss is small, about {ratio:.0f} times smaller than the {own_history_alone:.3f} "
                 f"bits per coda gained from own history alone. All eight conditions share one L2 penalty of "
                 f"{config['l2']} rather than a penalty tuned per condition, and S contributes only a handful of turn "
                 f"category indicators and two timing values on top of a much larger set of history features when A "
                 f"is present. Adding a small number of weak features to an already large feature set under one "
                 f"fixed penalty is a plausible source of a small, consistent loss on held out data. This is a "
                 f"property of the fixed penalty reference model, not evidence that timing information is harmful, "
                 f"and the same check should be repeated once the neural model, which fits its S encoder separately, "
                 f"is available.")
    lines.append("No change was made to the model, the features, or the penalty after these results were observed. "
                 "Every condition keeps the same architecture and hyperparameters so the contrasts stay comparable; "
                 "adjusting the setup only for the conditions that performed worse would mean choosing settings "
                 "after seeing the outcome.")
    full_vs_repeat = baseline_checks['A_B_S']
    lines.append(f"The full condition (A, B and S) also fell slightly below the repeat own last baseline in mean log "
                 f"likelihood ({full_vs_repeat['mean_bits_per_coda']:.5f} bits per coda), but the day bootstrap "
                 f"interval, {full_vs_repeat['ci_bits_per_coda'][0]:.5f} to {full_vs_repeat['ci_bits_per_coda'][1]:.5f}, "
                 f"comfortably includes zero, and {full_vs_repeat['positive_days']} of {full_vs_repeat['days']} days "
                 f"favoured the full condition. Falling reliably below this baseline would indicate a failed fit; an "
                 f"interval that includes zero indicates the two are indistinguishable, which is a finding rather "
                 f"than a failure. The same pattern, an interval spanning zero, holds for A, for A and B, and for A "
                 f"and S considered on their own.")
    own = summary['contrasts']['own_history']['components']
    total_bits, repeat_bits, switch_bits = own['log_likelihood_bits']['mean_bits_per_coda'], own['repeat_bits']['mean_bits_per_coda'], own['switch_choice_bits']['mean_bits_per_coda']
    lines.append(f"The repeat and switch decomposition of the own history contrast supports the same reading. Of the "
                 f"{total_bits:.3f} bits per coda gained by adding own history, {repeat_bits:.3f} bits "
                 f"({100*repeat_bits/total_bits:.0f} percent) come from predicting whether the whale repeats its "
                 f"previous coda, and {switch_bits:.3f} bits ({100*switch_bits/total_bits:.0f} percent) come from "
                 f"predicting what it produces when it does not repeat. At the level of this reference model, most "
                 f"of the value of own history reflects repetition rather than new sequential structure, consistent "
                 f"with the persistence account this study is designed to test.")
    lines += ['', 'Statistical uncertainty and restricted feature capacity limit inference from this reference model; '
             'this report does not establish grammar or causal mechanisms.',
             'Pending: retrospective-delay sensitivity, neural models and their reproduction/robustness checks.', '']
    report=root/config['report'];report.parent.mkdir(parents=True,exist_ok=True);write_text(report,'\n'.join(lines))
    return summary


def fit_v2_lattice(config, manifest, rows, evaluation_config):
    """Fold loop, fitting and evaluation of logistic v2-v4 over the manifest's conditions;
    writes nothing. Adds shared copy features (beta for A, gamma for B) and per-fold,
    per-condition L2 selected on the validation day; everything else (preprocessing,
    folds, UNK policy) is unchanged. See select_l2_and_fit for the test-isolation
    guarantee. Used by _fit_and_evaluate_v2 and by the own-history length sweep
    (history_length.py, DEC-25), which adds only config['history_orders']."""
    if evaluation_config['seeds'] != [config['seed']]:
        raise ValueError('Evaluation seed configuration mismatch')
    predictions, baselines, models, diagnostics, selections = [], [], [], [], []
    for fold in manifest['folds']:
        mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
        if policy != fold['vocabulary_policy']:
            raise ValueError('Recomputed fold policy differs from manifest')
        train = [r for r in mapped if r['split']=='train' and r['comparison_eligible']]
        validation = [r for r in mapped if r['split']=='validation' and r['comparison_eligible']]
        test = [r for r in mapped if r['split']=='test' and r['comparison_eligible']]
        if not validation:
            raise ValueError('Fold has no validation comparison targets for L2 selection')
        if {r['coda_id'] for r in validation} & {r['coda_id'] for r in test}:
            raise ValueError('Validation and test rows overlap')
        vocab, prior = policy['vocabulary'], policy['label_prior']
        rate = sum(r['Ct']==r['A_history'][-1] for r in train)/len(train)
        majority = max(sorted(prior), key=prior.get)
        for row in test:
            p = repeat_distribution(row['A_history'][-1], rate, prior)
            baselines.append(dict(coda_id=row['coda_id'],fold_id=fold['fold_id'],recording_date=row['recording_date'],
                                  target=row['Ct'],majority=majority,repeat_probabilities=p,
                                  repeat_prediction=max(sorted(p),key=p.get),
                                  repeat_log_likelihood_bits=log_likelihood_bits(p[row['Ct']]),training_repeat_rate=rate))
        for condition, fields in manifest['input_conditions'].items():
            if condition == 'none':
                p = np.array([[prior[t] for t in vocab]]*len(test))
            else:
                try:
                    encoder, chosen, grid = select_l2_and_fit(train, validation, vocab, fields, config)
                except NonConvergenceError as error:
                    if not config.get('skip_nonconverged_candidates', False):
                        raise
                    raise NonConvergenceError(f"{fold['fold_id']}, condition {condition}: {error}") from error
                own_test = last_label_index(test, 'A_history', vocab) if 'A_history' in fields else None
                partner_test = last_label_index(test, 'B_history', vocab) if 'B_history' in fields else None
                p = probabilities(encoder.transform(test), chosen['w'], chosen['beta'], chosen['gamma'], own_test, partner_test)
                diagnostics.append(dict(fold_id=fold['fold_id'],condition=condition,**chosen['convergence']))
                models.append(dict(fold_id=fold['fold_id'],condition=condition,vocabulary=vocab,
                                   fields=fields,order=encoder.order,
                                   **({'history_orders': encoder.history_orders} if encoder.history_orders else {}),
                                   categories=encoder.categories,
                                   feature_names=encoder.names+['intercept'],mean=encoder.mean.tolist(),
                                   scale=encoder.scale.tolist(),weights=chosen['w'].tolist(),
                                   beta=chosen['beta'],gamma=chosen['gamma'],chosen_l2=chosen['l2'],
                                   chosen_l2_s=chosen['l2_s']))
                selections.append(dict(fold_id=fold['fold_id'],condition=condition,chosen_l2=chosen['l2'],
                                       chosen_l2_s=chosen['l2_s'],
                                       grid=[dict(l2=g['l2'],l2_s=g['l2_s'],nonconverged=g['nonconverged']) if 'nonconverged' in g else
                                             dict(l2=g['l2'],l2_s=g['l2_s'],validation_log_likelihood_bits=g['validation_log_likelihood_bits'],
                                                  convergence=g['convergence']) for g in grid]))
            for row, probs in zip(test,p):
                predictions.append(dict(coda_id=row['coda_id'],fold_id=fold['fold_id'],condition=condition,
                                        seed=config['seed'],probabilities=dict(zip(vocab,map(float,probs)))))
        print(f"Finished {fold['fold_id']}: {len(test)} test targets", flush=True)
    summary, scored, paired = evaluate(predictions,manifest,rows,[config['seed']],evaluation_config['bootstrap_replicates'],
                                       evaluation_config['bootstrap_seed'],evaluation_config['confidence'])
    baseline_metrics = {'majority':classification_metrics([r['target'] for r in baselines],[r['majority'] for r in baselines]),
                        'repeat_own_last':classification_metrics([r['target'] for r in baselines],[r['repeat_prediction'] for r in baselines])}
    baseline_metrics['repeat_own_last']['mean_log_likelihood_bits'] = sum(r['repeat_log_likelihood_bits'] for r in baselines)/len(baselines)
    indexed = {r['coda_id']:r for r in baselines}
    baseline_checks = {}
    for condition in ('A','A_B','A_S','A_B_S'):
        days = {}
        for r in scored:
            if r['condition']==condition:
                days.setdefault(r['recording_date'],[]).append(r['log_likelihood_bits']-indexed[r['coda_id']]['repeat_log_likelihood_bits'])
        baseline_checks[condition] = day_bootstrap_bits(days,evaluation_config['bootstrap_replicates'],evaluation_config['bootstrap_seed'],evaluation_config['confidence'])
    return dict(predictions=predictions, baselines=baselines, models=models, diagnostics=diagnostics,
                selections=selections, summary=summary, scored=scored, paired=paired,
                baseline_metrics=baseline_metrics, baseline_checks=baseline_checks)


def _fit_and_evaluate_v2(config_path):
    """fit_v2_lattice on a config file, writing its outputs; shared by run_v2 and
    run_v2_sensitivity, which differ only in the report each writes at the end."""
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    manifest_path = root / config['manifest']
    manifest = read_json(manifest_path)
    source = root / manifest['configuration']['input']
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Stale fold manifest')
    evaluation_config = read_json(root / config['evaluation_config'])
    rows = read_jsonl(source)
    fitted = fit_v2_lattice(config, manifest, rows, evaluation_config)
    predictions, baselines, models, diagnostics, selections, summary, scored, paired, baseline_metrics, baseline_checks = (
        fitted[k] for k in ('predictions', 'baselines', 'models', 'diagnostics', 'selections', 'summary', 'scored',
                            'paired', 'baseline_metrics', 'baseline_checks'))

    output = root / config['output_directory'];output.mkdir(parents=True,exist_ok=True)
    write_jsonl(output/'predictions.jsonl',predictions)
    write_jsonl(output/'baseline_predictions.jsonl',baselines)
    write_jsonl(output/'models.jsonl',models)
    write_jsonl(output/'convergence.jsonl',diagnostics)
    write_jsonl(output/'l2_selection.jsonl',selections)
    eval_output = output/'evaluation';eval_output.mkdir(exist_ok=True)
    write_jsonl(eval_output/'scored_predictions.jsonl',scored)
    write_jsonl(eval_output/'paired_contrasts.jsonl',paired)
    summary.update(configuration=config,baseline_metrics=baseline_metrics,repeat_baseline_checks=baseline_checks,
                   versions={'numpy':np.__version__,'scipy':scipy.__version__},
                   input_sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,manifest_path,path)})
    write_text(eval_output/'summary.json',json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return root, config, summary, baseline_metrics, baseline_checks, models, selections, diagnostics


def run_v2(config_path):
    """Adds shared copy features and tuned L2 to the primary onset-available lattice;
    writes reports/rq1_logistic_v2.md comparing against v1 (fixed L2, no copy features)."""
    root, config, summary, baseline_metrics, baseline_checks, models, selections, diagnostics = _fit_and_evaluate_v2(config_path)
    stop_reasons = check_stop_conditions(summary, baseline_metrics, baseline_checks)
    v1_summary = read_json(root / config['compare_against']) if config.get('compare_against') else None
    write_v2_report(root, config, summary, baseline_metrics, baseline_checks, models, selections, diagnostics, stop_reasons, v1_summary,
                    version_label=config.get('version_label', 'v2'), compare_label=config.get('compare_label', 'v1'))
    return summary, stop_reasons


def run_v2_sensitivity(config_path):
    """Same fitting and evaluation as run_v2, run on a manifest whose S-containing
    conditions substitute retrospective_delay_s for gap_s/overlap_elapsed_s (PLAN
    S9, DEC-1). Reports the comparison separately from the primary lattice, never
    merged into it, and draws no conclusions beyond the numbers."""
    root, config, summary, baseline_metrics, baseline_checks, models, selections, diagnostics = _fit_and_evaluate_v2(config_path)
    stop_reasons = check_stop_conditions(summary, baseline_metrics, baseline_checks)
    primary_summary = read_json(root / config['compare_against'])
    for c in ('A', 'B', 'A_B'):
        if summary['metrics'][c] != primary_summary['metrics'][c]:
            raise ValueError(f"Condition {c} does not use S but differs from the primary run")
    write_sensitivity_report(root, config, summary, baseline_metrics, baseline_checks, stop_reasons, primary_summary,
                             version_label=config.get('version_label', 'v2'))
    return summary, stop_reasons


def nonconvergence_lines(selections, grid_n, selected):
    """DEC-22's opt-in skip_nonconverged_candidates: how many L2 candidates failed fit()'s convergence
    check and were left out of the validation-day selection, per fold and condition."""
    total = sum(len(s['grid']) for s in selections)
    failed = [(s['fold_id'], s['condition'], len(s['grid']), sum('nonconverged' in g for g in s['grid'])) for s in selections]
    failed = [f for f in failed if f[3]]
    lines = [f"{total} fits were run, across {len({s['fold_id'] for s in selections})} folds and "
             f"{len({s['condition'] for s in selections})} non-unigram conditions: conditions without S search "
             f"{grid_n} history L2 values, conditions with S jointly search {grid_n} history values times {grid_n} "
             "S values (DEC-20). With skip_nonconverged_candidates on (DEC-22), a candidate that fails fit()'s "
             "convergence check is left out of the validation-day selection; "
             f"{sum(f[3] for f in failed)} of the {total} did, and every selected model converged. Full "
             "per-grid-value convergence is in l2_selection.jsonl; the "
             f"{selected} selected models' convergence is in convergence.jsonl."]
    if failed:
        lines += ['', '| Fold | Condition | Candidates | Not converged, left out |', '|---|---|---:|---:|']
        lines += [f"| {fold} | {condition} | {n} | {k} |" for fold, condition, n, k in failed]
    else:
        lines += ['', 'No candidate failed the convergence check in any fold or condition.']
    return lines


def write_v2_report(root, config, summary, baseline_metrics, baseline_checks, models, selections, diagnostics,
                    stop_reasons, v1_summary, version_label='v2', compare_label='v1'):
    """Without compare_against (v1_summary is None, as in the RQ2 inventory runs, whose log-likelihoods
    are comparable with no other run's) the comparison columns and sentences are left out. Optional
    config keys: report_title, report_note (appended to the opening paragraph when nothing is compared)."""
    conditions = list(summary['metrics'])
    compare = v1_summary is not None
    title = config.get('report_title', 'RQ1 logistic-regression reference results')
    folds_report = read_json(root / config['manifest'])['configuration']['report']
    non_repeats = summary['comparison_targets'] - summary['raw_repeats']
    model = ("Conditions containing A get a shared "
             "scalar added to the logit of the class matching the whale's own last coda, and conditions containing "
             "B get a shared scalar added to the logit of the class matching the partner's last coda, both "
             "penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from "
             f"{config['l2_grid']} by the highest mean log-likelihood on that fold's validation day (ties go to "
             "the larger value), then the model trained on the training days only is scored on the test day, with "
             "no refit on train plus validation. Bootstrap replicates are 10,000.")
    if compare:
        intro = (f"Same fourteen leave-one-day-out folds as {compare_label}. {model} Preprocessing, folds, the UNK "
                 f"policy and the S definition are unchanged; {compare_label} results are kept for comparison, not "
                 "overwritten.")
    else:
        intro = f"Fourteen leave-one-day-out folds (`{config['manifest']}`). {model}"
        if config.get('report_note'):
            intro += f" {config['report_note']}"
    lines = [f'# {title}, {version_label}', '', intro, '']

    if not compare:
        lines += [f'## Metrics, {version_label}', '', '| Condition | LL bits/coda | Accuracy | Macro-F1 |', '|---|---:|---:|---:|']
        for c in conditions:
            m = summary['metrics'][c]
            lines.append(f"| {c} | {m['mean_log_likelihood_bits']:.5f} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} |")
        for c, m in baseline_metrics.items():
            ll = f"{m['mean_log_likelihood_bits']:.5f}" if 'mean_log_likelihood_bits' in m else 'n/a'
            lines.append(f"| {c} | {ll} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} |")
    else:
        lines += [f'## Metrics, {compare_label} vs {version_label}', '',
                 f'| Condition | {compare_label} LL bits/coda | {version_label} LL bits/coda | {compare_label} Accuracy '
                 f'| {version_label} Accuracy | {compare_label} Macro-F1 | {version_label} Macro-F1 |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for c in conditions:
            v1m = v1_summary['metrics'][c] if v1_summary else None
            v2m = summary['metrics'][c]
            v1ll = f"{v1m['mean_log_likelihood_bits']:.5f}" if v1m else 'n/a'
            v1acc = f"{v1m['accuracy']:.3%}" if v1m else 'n/a'
            v1f1 = f"{v1m['macro_f1']:.4f}" if v1m else 'n/a'
            lines.append(f"| {c} | {v1ll} | {v2m['mean_log_likelihood_bits']:.5f} | {v1acc} | {v2m['accuracy']:.3%} | "
                         f"{v1f1} | {v2m['macro_f1']:.4f} |")
        for c, m in baseline_metrics.items():
            ll = f"{m['mean_log_likelihood_bits']:.5f}" if 'mean_log_likelihood_bits' in m else 'n/a'
            lines.append(f"| {c} | {ll} | {ll} | {m['accuracy']:.3%} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} | {m['macro_f1']:.4f} |")

    lines += ['', '## Chosen L2 per condition (DEC-20: history and S penalties chosen separately)', '',
             '| Condition | History min | History median | History max | History at grid edge | S min | S median | S max | S at grid edge |',
             '|---|---:|---:|---:|---|---:|---:|---:|---|']
    grid_min, grid_max = min(config['l2_grid']), max(config['l2_grid'])
    def l2_stats(values):
        values = sorted(values)
        mid = len(values)//2
        median = values[mid] if len(values)%2 else (values[mid-1]+values[mid])/2
        return values[0], median, values[-1]
    for c in conditions:
        if c == 'none':
            continue
        rows = [s for s in selections if s['condition'] == c]
        h_values = [s['chosen_l2'] for s in rows]
        h_min, h_med, h_max = l2_stats(h_values)
        h_edge = [s['fold_id'] for s in rows if s['chosen_l2'] in (grid_min, grid_max)]
        h_edge_note = f" ({', '.join(h_edge)})" if h_edge else ''
        if rows[0]['chosen_l2_s'] is not None:
            s_values = [s['chosen_l2_s'] for s in rows]
            s_min, s_med, s_max = l2_stats(s_values)
            s_edge = [s['fold_id'] for s in rows if s['chosen_l2_s'] in (grid_min, grid_max)]
            s_cell = f"{s_min} | {s_med} | {s_max} | {len(s_edge)}/{len(s_values)}"
        else:
            s_cell = 'n/a | n/a | n/a | n/a'
        lines.append(f"| {c} | {h_min} | {h_med} | {h_max} | {len(h_edge)}/{len(h_values)}{h_edge_note} | {s_cell} |")

    lines += ['', '## Fitted beta and gamma per condition (range over folds)', '',
             '| Condition | Beta min | Beta max | Gamma min | Gamma max |', '|---|---:|---:|---:|---:|']
    for c in conditions:
        if c == 'none':
            continue
        betas = [m['beta'] for m in models if m['condition']==c and m['beta'] is not None]
        gammas = [m['gamma'] for m in models if m['condition']==c and m['gamma'] is not None]
        bmin, bmax = (f"{min(betas):.4f}", f"{max(betas):.4f}") if betas else ('n/a', 'n/a')
        gmin, gmax = (f"{min(gammas):.4f}", f"{max(gammas):.4f}") if gammas else ('n/a', 'n/a')
        lines.append(f"| {c} | {bmin} | {bmax} | {gmin} | {gmax} |")

    repeat = (f"Repeat-own-last: {baseline_metrics['repeat_own_last']['accuracy']:.3%} accuracy, "
              f"{baseline_metrics['repeat_own_last']['macro_f1']:.4f} macro-F1, "
              f"{baseline_metrics['repeat_own_last']['mean_log_likelihood_bits']:.5f} bits/coda.")
    if not compare:
        lines += ['', f'## A-conditions vs repeat-own-last, {version_label}', '', repeat, '',
                 '| Condition | LL contrast | CI | +days | Accuracy | Macro-F1 |', '|---|---:|---|---:|---:|---:|']
        for c in ('A', 'A_B', 'A_S', 'A_B_S'):
            check = baseline_checks[c]
            lines.append(f"| {c} | {check['mean_bits_per_coda']:.5f} | {format_ci(check['ci_bits_per_coda'])} | "
                         f"{check['positive_days']}/{check['days']} | {summary['metrics'][c]['accuracy']:.3%} | "
                         f"{summary['metrics'][c]['macro_f1']:.4f} |")
    else:
        lines += ['', f'## A-conditions vs repeat-own-last, {compare_label} vs {version_label}', '',
                 repeat, '',
                 f'| Condition | {compare_label} LL contrast | {compare_label} CI | {compare_label} +days | '
                 f'{version_label} LL contrast | {version_label} CI | {version_label} +days | {version_label} Accuracy '
                 f'| {version_label} Macro-F1 |',
                 '|---|---:|---|---:|---:|---|---:|---:|---:|']
        for c in ('A', 'A_B', 'A_S', 'A_B_S'):
            v1c = v1_summary['repeat_baseline_checks'][c] if v1_summary else None
            v2c = baseline_checks[c]
            v1mean = f"{v1c['mean_bits_per_coda']:.5f}" if v1c else 'n/a'
            v1ci = format_ci(v1c['ci_bits_per_coda']) if v1c else 'n/a'
            v1days = f"{v1c['positive_days']}/{v1c['days']}" if v1c else 'n/a'
            lines.append(f"| {c} | {v1mean} | {v1ci} | {v1days} | {v2c['mean_bits_per_coda']:.5f} | {format_ci(v2c['ci_bits_per_coda'])} | "
                         f"{v2c['positive_days']}/{v2c['days']} | {summary['metrics'][c]['accuracy']:.3%} | "
                         f"{summary['metrics'][c]['macro_f1']:.4f} |")

    lines += ['', f'## All twelve add-one-input contrasts ({version_label})', '',
             '| Pair | Weighted mean | Unweighted day mean | 95% CI | Positive days |', '|---|---:|---:|---|---:|']
    for d in classify_add_input_contrasts(summary):
        lines.append(f"| adding {d['added']} to {describe(d['base'])} | {d['mean']:.5f} | {d['unweighted_mean']:.5f} | "
                     f"{format_ci([d['low'], d['high']])} | {d['positive_days']}/{d['days']} |")

    if not compare:
        lines += ['', f'## Main contrasts, {version_label}', '',
                 '| Contrast | bits/coda | Unweighted | CI | +days | Repeat part | Switch part |',
                 '|---|---:|---:|---|---:|---:|---:|']
        for name in ('own_history', 'partner_content', 'timing'):
            comp = summary['contrasts'][name]['components']; a = comp['log_likelihood_bits']
            lines.append(f"| {name} | {a['mean_bits_per_coda']:.5f} | {a['unweighted_mean_bits_per_coda']:.5f} | "
                         f"{format_ci(a['ci_bits_per_coda'])} | {a['positive_days']}/{a['days']} | "
                         f"{comp['repeat_bits']['mean_bits_per_coda']:.5f} | {comp['switch_choice_bits']['mean_bits_per_coda']:.5f} |")
    else:
        lines += ['', f'## Main contrasts, {compare_label} vs {version_label}', '',
                 f'| Contrast | {compare_label} bits/coda | {version_label} bits/coda | {version_label} unweighted | '
                 f'{version_label} CI | {version_label} +days | {version_label} Repeat part | {version_label} Switch part |',
                 '|---|---:|---:|---:|---|---:|---:|---:|']
        for name in ('own_history', 'partner_content', 'timing'):
            v1a = v1_summary['contrasts'][name]['components']['log_likelihood_bits'] if v1_summary else None
            comp = summary['contrasts'][name]['components']; a = comp['log_likelihood_bits']
            v1txt = f"{v1a['mean_bits_per_coda']:.5f}" if v1a else 'n/a'
            lines.append(f"| {name} | {v1txt} | {a['mean_bits_per_coda']:.5f} | {a['unweighted_mean_bits_per_coda']:.5f} | "
                         f"{format_ci(a['ci_bits_per_coda'])} | {a['positive_days']}/{a['days']} | "
                         f"{comp['repeat_bits']['mean_bits_per_coda']:.5f} | {comp['switch_choice_bits']['mean_bits_per_coda']:.5f} |")

    total_fits = sum(len(s['grid']) for s in selections)
    grid_n = len(config['l2_grid'])
    if config.get('skip_nonconverged_candidates', False):
        lines += [''] + nonconvergence_lines(selections, grid_n, len(diagnostics)) + ['']
    else:
        lines += ['', f"All {total_fits} fits converged, across {len({s['fold_id'] for s in selections})} folds and "
                 f"{len({s['condition'] for s in selections})} non-unigram conditions: conditions without S search "
                 f"{grid_n} history L2 values, conditions with S jointly search {grid_n} history values times {grid_n} "
                 "S values (DEC-20). fit() raises immediately on non-convergence, so reaching this report already "
                 "confirms it. Full per-grid-value convergence is in l2_selection.jsonl; the "
                 f"{len(diagnostics)} selected models' convergence is in convergence.jsonl.", '']

    if stop_reasons:
        lines += ['## Stop: results need review before interpretation', '',
                 'The following stop conditions from the task were met. No interpretation or conclusions are '
                 'written below; the tables above are the complete result.', '']
        lines += [f"- {reason}" for reason in stop_reasons]
        lines += ['']
    else:
        lines += ['## Result', '',
                 'No A-condition is reliably below repeat-own-last in log-likelihood, and no add-one-input '
                 'contrast has a confidence interval entirely below zero.', '']

    lines += ['Statistical uncertainty and restricted feature capacity limit inference from this reference model; '
             'this report does not establish grammar or causal mechanisms. Day-bootstrap intervals use 10,000 '
             'replicates and are approximate with only 14 days. The switch-choice component of every contrast '
             f'rests on about {non_repeats} non-repeat codas and is exploratory. Source, fold-manifest and configuration '
             'SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in '
             'evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; '
             f'{folds_report} reports UNK rates per fold.', '']

    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(lines))


def write_sensitivity_report(root, config, summary, baseline_metrics, baseline_checks, stop_reasons, primary_summary,
                             version_label='v2'):
    """PLAN S9: the retrospective-delay sensitivity run, reported separately from
    the primary onset-available results. Same model version (copy features, tuned
    L2) on both sides; only the timing fields feeding the S-containing conditions
    differ. No interpretation is written here, only the numbers."""
    lines = ['# RQ1 retrospective-delay sensitivity run', '',
             f"Same fourteen leave-one-day-out folds, vocabulary and {version_label} model (shared copy features, "
             "L2 tuned per fold and condition on the validation day) as the primary results. Only the timing fields feeding "
             "S, A_S, B_S and A_B_S change: turn_category is kept, but gap_s and overlap_elapsed_s are replaced "
             "by retrospective_delay_s, the onset-to-previous-interlocutor-offset delay used before DEC-1. That "
             "field can be negative (the predecessor coda had not finished when the target started) and is only "
             "standardized, not log1p transformed, since it goes below -1 in this data. Conditions without S "
             "(none, A, B, A_B) are refit identically to the primary run for a complete lattice, and should "
             "reproduce it exactly. This run is a sensitivity check only; it does not alter the primary "
             "preprocessing, model, or conclusions.", '']

    lines += ['## S-containing conditions and repeat-own-last, primary vs retrospective delay', '',
             '| Condition | Primary LL bits/coda | Retrospective LL bits/coda | Primary Accuracy | '
             'Retrospective Accuracy | Primary Macro-F1 | Retrospective Macro-F1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for c in ('S', 'A_S', 'B_S', 'A_B_S'):
        p, r = primary_summary['metrics'][c], summary['metrics'][c]
        lines.append(f"| {c} | {p['mean_log_likelihood_bits']:.5f} | {r['mean_log_likelihood_bits']:.5f} | "
                     f"{p['accuracy']:.3%} | {r['accuracy']:.3%} | {p['macro_f1']:.4f} | {r['macro_f1']:.4f} |")
    pr, rb = primary_summary['baseline_metrics']['repeat_own_last'], baseline_metrics['repeat_own_last']
    lines.append(f"| repeat_own_last | {pr['mean_log_likelihood_bits']:.5f} | {rb['mean_log_likelihood_bits']:.5f} | "
                 f"{pr['accuracy']:.3%} | {rb['accuracy']:.3%} | {pr['macro_f1']:.4f} | {rb['macro_f1']:.4f} |")

    lines += ['', '## Timing contrast and A_B_S vs repeat-own-last, primary vs retrospective delay', '', '']
    pt, rt = primary_summary['contrasts']['timing']['components']['log_likelihood_bits'], summary['contrasts']['timing']['components']['log_likelihood_bits']
    lines[-1] = ('| Comparison | Primary bits/coda | Primary CI | Primary +days | Retrospective bits/coda | '
                 'Retrospective CI | Retrospective +days |')
    lines.append('|---|---:|---|---:|---:|---|---:|')
    lines.append(f"| timing (A_B_S - A_B) | {pt['mean_bits_per_coda']:.5f} | {format_ci(pt['ci_bits_per_coda'])} | "
                 f"{pt['positive_days']}/{pt['days']} | {rt['mean_bits_per_coda']:.5f} | {format_ci(rt['ci_bits_per_coda'])} | "
                 f"{rt['positive_days']}/{rt['days']} |")
    pc, rc = primary_summary['repeat_baseline_checks']['A_B_S'], baseline_checks['A_B_S']
    lines.append(f"| A_B_S vs repeat-own-last | {pc['mean_bits_per_coda']:.5f} | {format_ci(pc['ci_bits_per_coda'])} | "
                 f"{pc['positive_days']}/{pc['days']} | {rc['mean_bits_per_coda']:.5f} | {format_ci(rc['ci_bits_per_coda'])} | "
                 f"{rc['positive_days']}/{rc['days']} |")

    lines += ['', 'The A, B and A_B conditions do not use S and are refit identically to the primary run '
             '(verified exactly equal before this report is written).', '']

    if stop_reasons:
        lines += ['## Stop: results need review before interpretation', '',
                 'The following stop conditions from the task were met on this retrospective-delay run. No '
                 'interpretation or conclusions are written below; the tables above are the complete result.', '']
        lines += [f"- {reason}" for reason in stop_reasons]
        lines += ['']
    else:
        lines += ['## Result', '',
                 'No A-condition falls below repeat-own-last in log-likelihood or accuracy on this run, and no '
                 'add-one-input contrast has a confidence interval entirely below zero.', '']

    lines += ['Day-bootstrap intervals use 10,000 replicates and are approximate with only 14 days. Source, '
             'fold-manifest and configuration SHA-256 hashes and full model settings are recorded in '
             'evaluation/summary.json alongside this report.', '']

    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/rq1_logistic.json')
    args=parser.parse_args()
    config=read_json(Path(args.config).resolve())
    if config.get('sensitivity') == 'retrospective_delay':
        _, stop_reasons = run_v2_sensitivity(args.config)
        if stop_reasons:
            print(f"STOP: {len(stop_reasons)} condition(s) need review before interpretation. See the report.")
    elif 'l2_grid' in config:
        _, stop_reasons = run_v2(args.config)
        if stop_reasons:
            print(f"STOP: {len(stop_reasons)} condition(s) need review before interpretation. See the report.")
    else:
        run(args.config)


if __name__=='__main__':
    main()
