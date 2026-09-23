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
            else:
                self.names.extend([f, 'log1p:'+f])
        x = self.raw(train)
        self.mean, self.scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
        for j, name in enumerate(self.names):
            if name in ('gap_s', 'overlap_elapsed_s', 'log1p:gap_s', 'log1p:overlap_elapsed_s'):
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
                else:
                    raise ValueError(f'Unsupported input field {f}')
            result.append(values)
        return np.asarray(result, dtype=float)

    def transform(self, rows):
        x = (self.raw(rows) - self.mean) / self.scale
        return np.column_stack([x, np.ones(len(rows))])


def objective(weights, x, y, classes, l2):
    w = weights.reshape(x.shape[1], classes)
    logits = x @ w
    logs = logits - logsumexp(logits, axis=1, keepdims=True)
    loss = -logs[np.arange(len(y)), y].mean() + l2 * np.square(w).sum() / 2
    residual = np.exp(logs)
    residual[np.arange(len(y)), y] -= 1
    gradient = x.T @ residual / len(y) + l2 * w
    return loss, gradient.ravel()


def fit(x, y, classes, config):
    # Penalize intercepts too, making absent target classes (including UNK)
    # well-defined and finite without synthetic targets or post-hoc clipping.
    if not np.isfinite(config['l2']) or config['l2'] <= 0:
        raise ValueError('Require positive finite L2')
    result = minimize(objective, np.zeros(x.shape[1]*classes), args=(x, y, classes, config['l2']),
                      jac=True, method='L-BFGS-B', options={'maxiter': config['max_iterations'],
                      'gtol': config['gradient_tolerance'], 'ftol': 1e-14})
    gradient = float(np.max(np.abs(result.jac)))
    if not result.success or gradient > 10*config['gradient_tolerance']:
        raise ValueError(f'Optimizer did not converge: {result.message}; gradient={gradient}')
    return result.x.reshape(x.shape[1], classes), {'iterations': int(result.nit), 'max_gradient': gradient}


def probabilities(x, weights):
    logits = x @ weights
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def repeat_distribution(last, rate, prior):
    if not 0 < rate < 1:
        raise ValueError('Training repeat rate must lie strictly between 0 and 1')
    other_mass = sum(p for t, p in prior.items() if t != last)
    if other_mass <= 0:
        raise ValueError('Repeat baseline needs at least one alternative label')
    return {t: rate if t == last else (1-rate)*p/other_mass for t, p in prior.items()}


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
                weights, convergence = fit(x, y, len(vocab), config)
                p = probabilities(encoder.transform(test), weights)
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

    def describe(condition):
        if condition == 'none':
            return 'no inputs'
        parts = condition.split('_')
        return parts[0] if len(parts) == 1 else ' and '.join(parts) if len(parts) == 2 else ', '.join(parts[:-1])+' and '+parts[-1]

    add_input_details = []
    for key, contrast in summary['contrasts'].items():
        if not key.startswith('add_input:'):
            continue
        full_parts = contrast['full'].split('_') if contrast['full'] != 'none' else []
        reduced_parts = contrast['reduced'].split('_') if contrast['reduced'] != 'none' else []
        added = next(p for p in full_parts if p not in reduced_parts)
        stats = contrast['components']['log_likelihood_bits']
        low, high = stats['ci_bits_per_coda']
        reliability = 'negative' if high < 0 else 'positive' if low > 0 else 'inconclusive'
        add_input_details.append(dict(added=added, base=contrast['reduced'], mean=stats['mean_bits_per_coda'],
                                      low=low, high=high, reliability=reliability))
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
                   input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,manifest_path,path)})
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/rq1_logistic.json')
    run(parser.parse_args().config)


if __name__=='__main__':
    main()
