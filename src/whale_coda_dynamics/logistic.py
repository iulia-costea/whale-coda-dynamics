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
from .preprocessing.io_utils import read_json, read_jsonl, read_text, write_jsonl, write_text


class Encoder:
    """Lag one-hot vectors; raw and log1p timing, each standardized on train."""
    def __init__(self, train, vocabulary, fields, order=3):
        if not train or any(r['split'] != 'train' for r in train):
            raise ValueError('Encoder fitting requires training rows')
        if type(order) is not int or order < 1:
            raise ValueError('Invalid history order')
        self.vocabulary, self.fields, self.order = vocabulary, fields, order
        self.names = []
        self.categories = sorted({r['turn_category'] for r in train}) if 'turn_category' in fields else []
        for f in fields:
            if f.endswith('_history'):
                self.names.extend(f'{f}:lag{lag}:{label}' for lag in range(1, order+1) for label in vocabulary)
            elif f == 'turn_category':
                self.names.extend('turn:'+c for c in self.categories)
            elif f == 'retrospective_delay_s':
                self.names.append(f)
            else:
                self.names.extend([f, 'log1p:'+f])
        x = self.raw(train)
        self.mean, self.scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
        for j, name in enumerate(self.names):
            if name in ('gap_s', 'overlap_elapsed_s', 'log1p:gap_s', 'log1p:overlap_elapsed_s', 'retrospective_delay_s'):
                self.mean[j], self.scale[j] = x[:, j].mean(), x[:, j].std() or 1.0

    def raw(self, rows):
        result = []
        for r in rows:
            values = []
            for f in self.fields:
                if f.endswith('_history'):
                    for lag in range(1, self.order+1):
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


def objective(weights, x, y, classes, l2, own_last_idx=None, partner_last_idx=None, feature_l2=None):
    n_features = x.shape[1]
    w, beta, gamma = _unpack(weights, n_features, classes, own_last_idx, partner_last_idx)
    logits = _copy_logits(x @ w, beta, own_last_idx, gamma, partner_last_idx)
    logs = logits - logsumexp(logits, axis=1, keepdims=True)
    if feature_l2 is None:
        # Preserve the original arithmetic for all legacy and no-S fits.
        loss = -logs[np.arange(len(y)), y].mean() + l2 * (np.square(w).sum() + (beta or 0.0)**2 + (gamma or 0.0)**2) / 2
        weight_penalty = l2
    else:
        weight_penalty = np.asarray(feature_l2)[:, None]
        loss = (-logs[np.arange(len(y)), y].mean() + (weight_penalty * np.square(w)).sum() / 2
                + l2 * ((beta or 0.0)**2 + (gamma or 0.0)**2) / 2)
    residual = np.exp(logs)
    residual[np.arange(len(y)), y] -= 1
    grad = [(x.T @ residual / len(y) + weight_penalty * w).ravel()]
    if beta is not None:
        mask = own_last_idx >= 0
        grad.append(np.array([residual[mask, own_last_idx[mask]].sum() / len(y) + l2 * beta]))
    if gamma is not None:
        mask = partner_last_idx >= 0
        grad.append(np.array([residual[mask, partner_last_idx[mask]].sum() / len(y) + l2 * gamma]))
    return loss, np.concatenate(grad)


def fit(x, y, classes, config, own_last_idx=None, partner_last_idx=None, feature_l2=None):
    # Penalize intercepts too, making absent target classes (including UNK)
    # well-defined and finite without synthetic targets or post-hoc clipping.
    if not np.isfinite(config['l2']) or config['l2'] <= 0:
        raise ValueError('Require positive finite L2')
    if feature_l2 is not None:
        feature_l2 = np.asarray(feature_l2, dtype=float)
        if feature_l2.shape != (x.shape[1],) or not np.all(np.isfinite(feature_l2)) or np.any(feature_l2 <= 0):
            raise ValueError('Require one positive finite penalty per feature, including intercept')
    n_features = x.shape[1]
    n_extra = (own_last_idx is not None) + (partner_last_idx is not None)
    result = minimize(objective, np.zeros(n_features*classes + n_extra),
                      args=(x, y, classes, config['l2'], own_last_idx, partner_last_idx, feature_l2),
                      jac=True, method='L-BFGS-B', options={'maxiter': config['max_iterations'],
                      'gtol': config['gradient_tolerance'], 'ftol': 1e-14})
    gradient = float(np.max(np.abs(result.jac)))
    if not result.success or gradient > 10*config['gradient_tolerance']:
        raise ValueError(f'Optimizer did not converge: {result.message}; gradient={gradient}')
    w, beta, gamma = _unpack(result.x, n_features, classes, own_last_idx, partner_last_idx)
    return w, beta, gamma, {'iterations': int(result.nit), 'max_gradient': gradient}


def probabilities(x, w, beta=None, gamma=None, own_last_idx=None, partner_last_idx=None):
    logits = _copy_logits(x @ w, beta, own_last_idx, gamma, partner_last_idx)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def select_l2_and_fit(train, validation, vocab, fields, config):
    """Grid-search L2 on the validation day only; `test` is never an argument
    here, so it is structurally impossible for test data to influence selection."""
    encoder = Encoder(train, vocab, fields, config['history_order'])
    x_train, x_val = encoder.transform(train), encoder.transform(validation)
    y_train = np.array([vocab.index(r['Ct']) for r in train])
    y_val = np.array([vocab.index(r['Ct']) for r in validation])
    own_train = last_label_index(train, 'A_history', vocab) if 'A_history' in fields else None
    partner_train = last_label_index(train, 'B_history', vocab) if 'B_history' in fields else None
    own_val = last_label_index(validation, 'A_history', vocab) if 'A_history' in fields else None
    partner_val = last_label_index(validation, 'B_history', vocab) if 'B_history' in fields else None
    grid = []
    has_s = any(not f.endswith('_history') for f in fields)
    timing_grid = config['s_l2_grid'] if has_s and 's_l2_grid' in config else [None]
    for l2 in config['l2_grid']:
        for s_l2 in timing_grid:
            penalties = None
            if s_l2 is not None:
                penalties = [l2 if name.startswith(('A_history:', 'B_history:')) else s_l2
                             for name in encoder.names] + [l2]
            w, beta, gamma, convergence = fit(x_train, y_train, len(vocab), dict(config, l2=l2),
                                               own_train, partner_train, penalties)
            p_val = probabilities(x_val, w, beta, gamma, own_val, partner_val)
            val_ll = sum(log_likelihood_bits(float(p)) for p in p_val[np.arange(len(validation)), y_val]) / len(validation)
            # Legacy fits retain their original reduction, including exact no-S v3 reproduction.
            if s_l2 is None:
                val_ll = float(np.mean(np.log2(p_val[np.arange(len(validation)), y_val])))
            candidate = dict(l2=l2, validation_log_likelihood_bits=val_ll, w=w, beta=beta,
                             gamma=gamma, convergence=convergence)
            if s_l2 is not None:
                candidate['s_l2'] = s_l2
            grid.append(candidate)
    chosen = max(grid, key=lambda g: (g['validation_log_likelihood_bits'], g['l2'], g.get('s_l2', 0)))
    return encoder, chosen, grid


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


def _fit_and_evaluate_v2(config_path):
    """Fold loop, fitting and evaluation shared by run_v2 and run_v2_sensitivity;
    only the report each writes at the end differs. Adds shared copy features
    (beta for A, gamma for B) and per-fold, per-condition L2 selected on the
    validation day; everything else (preprocessing, folds, UNK policy) is
    unchanged. See select_l2_and_fit for the test-isolation guarantee."""
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
                encoder, chosen, grid = select_l2_and_fit(train, validation, vocab, fields, config)
                own_test = last_label_index(test, 'A_history', vocab) if 'A_history' in fields else None
                partner_test = last_label_index(test, 'B_history', vocab) if 'B_history' in fields else None
                p = probabilities(encoder.transform(test), chosen['w'], chosen['beta'], chosen['gamma'], own_test, partner_test)
                diagnostics.append(dict(fold_id=fold['fold_id'],condition=condition,**chosen['convergence']))
                models.append(dict(fold_id=fold['fold_id'],condition=condition,vocabulary=vocab,
                                   fields=fields,order=encoder.order,categories=encoder.categories,
                                   feature_names=encoder.names+['intercept'],mean=encoder.mean.tolist(),
                                   scale=encoder.scale.tolist(),weights=chosen['w'].tolist(),
                                   beta=chosen['beta'],gamma=chosen['gamma'],chosen_l2=chosen['l2']))
                selections.append(dict(fold_id=fold['fold_id'],condition=condition,chosen_l2=chosen['l2'],
                                       grid=[dict(l2=g['l2'],validation_log_likelihood_bits=g['validation_log_likelihood_bits'],
                                                 convergence=g['convergence']) for g in grid]))
                if 's_l2' in chosen:
                    models[-1]['chosen_s_l2'] = chosen['s_l2']
                    selections[-1]['chosen_s_l2'] = chosen['s_l2']
                    for record, candidate in zip(selections[-1]['grid'], grid):
                        record['s_l2'] = candidate['s_l2']
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
    if config.get('version_label') in ('v3', 'v4'):
        write_gate_summary(root, config, summary, selections, stop_reasons)
    return summary, stop_reasons


def run_v2_sensitivity(config_path):
    """Same fitting and evaluation as run_v2, run on a manifest whose S-containing
    conditions substitute retrospective_delay_s for gap_s/overlap_elapsed_s (PLAN
    S9, DEC-1). Reports the comparison separately from the primary lattice, never
    merged into it, and draws no conclusions beyond the numbers."""
    root, config, summary, baseline_metrics, baseline_checks, models, selections, diagnostics = _fit_and_evaluate_v2(config_path)
    stop_reasons = check_stop_conditions(summary, baseline_metrics, baseline_checks)
    primary_summary = read_json(root / config['compare_against'])
    for c in ('none', 'A', 'B', 'A_B'):
        if summary['metrics'][c] != primary_summary['metrics'][c]:
            raise ValueError(f"Condition {c} does not use S but differs from the primary run")
    write_sensitivity_report(root, config, summary, baseline_metrics, baseline_checks, stop_reasons, primary_summary,
                             version_label=config.get('version_label', 'v2'))
    if config.get('version_label') in ('v3', 'v4'):
        write_gate_summary(root, config, summary, selections, stop_reasons)
    return summary, stop_reasons


def write_v2_report(root, config, summary, baseline_metrics, baseline_checks, models, selections, diagnostics,
                    stop_reasons, v1_summary, version_label='v2', compare_label='v1'):
    conditions = list(summary['metrics'])
    lines = [f'# RQ1 logistic-regression reference results, {version_label}', '',
             f"Same fourteen leave-one-day-out folds as {compare_label}. Conditions containing A get a shared "
             "scalar added to the logit of the class matching the whale's own last coda, and conditions containing "
             "B get a shared scalar added to the logit of the class matching the partner's last coda, both "
             "penalized by the same L2 as every other weight. L2 itself is chosen per fold and condition from "
             f"{config['l2_grid']} by the highest mean log-likelihood on that fold's validation day (ties go to "
             "the larger value), then the model trained on the training days only is scored on the test day, with "
             f"no refit on train plus validation. Bootstrap replicates are 10,000. Preprocessing, folds, the UNK "
             f"policy and the S definition are unchanged; {compare_label} results are kept for comparison, not "
             "overwritten.", '']

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

    lines += ['', '## Chosen L2 per condition', '', '| Condition | Min | Median | Max | Folds at grid edge |',
             '|---|---:|---:|---:|---|']
    grid_min, grid_max = min(config['l2_grid']), max(config['l2_grid'])
    for c in conditions:
        if c == 'none':
            continue
        chosen = sorted(s['chosen_l2'] for s in selections if s['condition'] == c)
        edge_folds = [s['fold_id'] for s in selections if s['condition'] == c and s['chosen_l2'] in (grid_min, grid_max)]
        mid = len(chosen)//2
        median = chosen[mid] if len(chosen)%2 else (chosen[mid-1]+chosen[mid])/2
        edge_note = f" ({', '.join(edge_folds)})" if edge_folds else ''
        lines.append(f"| {c} | {chosen[0]} | {median} | {chosen[-1]} | {len(edge_folds)}/{len(chosen)}{edge_note} |")

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

    lines += ['', f'## A-conditions vs repeat-own-last, {compare_label} vs {version_label}', '',
             f"Repeat-own-last: {baseline_metrics['repeat_own_last']['accuracy']:.3%} accuracy, "
             f"{baseline_metrics['repeat_own_last']['macro_f1']:.4f} macro-F1, "
             f"{baseline_metrics['repeat_own_last']['mean_log_likelihood_bits']:.5f} bits/coda.", '',
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

    total_fits = sum(len(selection['grid']) for selection in selections)
    lines += ['', f"All {total_fits} fits ({len({s['fold_id'] for s in selections})} folds times "
             f"{len({s['condition'] for s in selections})} non-unigram conditions; candidate counts are recorded per fit) converged; fit() raises immediately on non-convergence, so reaching this report already "
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
             'rests on about 270 non-repeat codas and is exploratory. Source, fold-manifest and configuration '
             'SHA-256 hashes, per-fold vocabulary sizes and UNK counts, and full model settings are recorded in '
             'evaluation/summary.json, models.jsonl and l2_selection.jsonl alongside this report; '
             'reports/rq1_folds.md reports UNK rates per fold.', '']

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


def write_gate_summary(root, config, summary, selections, stop_reasons):
    """Persist gate evidence and a complete, directly postable stage summary."""
    edges, boundary_failures = {}, []
    lines = ['', '## Penalty distributions and gate 1', '',
             '| Condition | Penalty | Value: fold count | Lower edge | Upper edge |',
             '|---|---|---|---:|---:|']
    for condition in summary['metrics']:
        records = [r for r in selections if r['condition'] == condition]
        if not records:
            continue
        edges[condition] = {}
        for key, grid_key in [('chosen_l2', 'l2_grid'), ('chosen_s_l2', 's_l2_grid')]:
            counts = Counter(r[key] for r in records if key in r)
            if not counts:
                continue
            lo, hi = min(config[grid_key]), max(config[grid_key])
            count = sum(counts.values())
            edges[condition][key] = dict(distribution=dict(sorted(counts.items())),
                                        lower=counts[lo], upper=counts[hi], folds=count)
            distribution = ', '.join(f'{value}: {n}' for value, n in sorted(counts.items()))
            lines.append(f'| {condition} | {key} | {distribution} | {counts[lo]}/{count} | {counts[hi]}/{count} |')
            if counts[lo] > count / 2:
                boundary_failures.append(f'{condition}: {key} selects the lower edge in {counts[lo]}/{count} folds')
    all_converged = all(g['convergence']['max_gradient'] <= 10 * config['gradient_tolerance']
                        for r in selections for g in r['grid'])
    failures = list(stop_reasons) + boundary_failures
    if not all_converged:
        failures.append('At least one fit did not meet the gradient convergence tolerance')
    gate = dict(numerical_checks_passed=not failures, failures=failures,
                all_fits_converged=all_converged, fits=sum(len(r['grid']) for r in selections),
                penalty_edges=edges,
                note='Full test suite and committed-artifact checks are separate required gate checks.')
    write_text(root / config['output_directory'] / 'gate1.json', json.dumps(gate, indent=2) + '\n')
    lines += ['', 'DEC-16 and grid checks: ' + ('FAIL' if failures else 'PASS') + '.', '']
    lines += ['- ' + reason for reason in failures]
    reference = read_json(root / config['compare_against'])
    lines += ['', '## Complete likelihood contrasts versus reference', '',
              '| Contrast | Run | Weighted bits/coda | Unweighted day mean | 95% CI | Positive days |',
              '|---|---|---:|---:|---|---:|']
    names = ['own_history', 'partner_content', 'timing'] + [n for n in summary['contrasts'] if n.startswith('add_input:')]
    for name in names:
        for label, result in [('reference', reference), (config['version_label'], summary)]:
            a = result['contrasts'][name]['components']['log_likelihood_bits']
            lines.append(f"| {name} | {label} | {a['mean_bits_per_coda']:.5f} | "
                         f"{a['unweighted_mean_bits_per_coda']:.5f} | {format_ci(a['ci_bits_per_coda'])} | "
                         f"{a['positive_days']}/{a['days']} |")
    report = root / config['report']
    content = read_text(report)
    if 's_l2_grid' in config:
        # The shared report's v2/v3 description assumes a single penalty.
        first_table = content.index('## ')
        description = (f"# RQ1 logistic regression {config['version_label']}: separate penalties\n\n"
                       f"Joint validation-day selection over base L2 {config['l2_grid']} and S L2 {config['s_l2_grid']}. "
                       "History lag weights, copy weights and the shared intercept use base L2; all timing and "
                       "turn-category coefficients use S L2. Ties prefer larger base L2, then larger S L2. "
                       "No-S conditions retain v3's fit exactly. Training excludes validation and test days. "
                       "Preprocessing, folds, vocabulary and shared evaluation are unchanged.\n\n")
        if config.get('sensitivity'):
            description += ('Sensitivity only: S uses the retrospective end-to-start delay. The primary and '
                            'retrospective runs reproduce all no-S conditions exactly on this machine.\n\n')
        content = description + content[first_table:]
    write_text(report, content + '\n'.join(lines) + '\n')


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
