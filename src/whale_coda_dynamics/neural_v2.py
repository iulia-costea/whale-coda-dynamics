"""Gated LSTM v2 orchestration; model and validation selection live in neural.py."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .evaluation import classification_metrics, day_bootstrap_bits, evaluate, log_likelihood_bits
from .logistic import check_stop_conditions, format_ci, repeat_distribution
from .neural import (ConditionModelV2, Encoder, build_batch, check_against_logistic,
                     fit_condition_v2, select_neural_setting, v2_probabilities)
from .preprocessing.folds import prepare_fold
from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text


def load_inputs(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    manifest_path = root / config['manifest']
    manifest = read_json(manifest_path)
    source = root / manifest['configuration']['input']
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Stale fold manifest')
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (path, manifest_path, source)}
    return root, config, manifest, read_jsonl(source), hashes


def fold_rows(rows, fold, manifest):
    mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
    if policy != fold['vocabulary_policy']:
        raise ValueError('Recomputed vocabulary differs from manifest')
    groups = {split: [r for r in mapped if r['split'] == split and r['comparison_eligible']]
              for split in ('train', 'validation', 'test')}
    return groups, policy


def metrics(scored):
    values = classification_metrics([r['target'] for r in scored], [r['prediction'] for r in scored])
    values['mean_log_likelihood_bits'] = sum(r['log_likelihood_bits'] for r in scored) / len(scored)
    return values


def score_rows(probabilities, rows, vocab, condition, fold_id, seed):
    predictions, scored = [], []
    for row, p in zip(rows, probabilities):
        distribution = dict(zip(vocab, map(float, p)))
        predictions.append(dict(coda_id=row['coda_id'], fold_id=fold_id, condition=condition,
                                seed=seed, probabilities=distribution))
        scored.append(dict(coda_id=row['coda_id'], fold_id=fold_id, condition=condition, seed=seed,
                           recording_date=row['recording_date'], target=row['Ct'],
                           prediction=max(sorted(distribution), key=distribution.get),
                           log_likelihood_bits=log_likelihood_bits(distribution[row['Ct']])))
    return predictions, scored


def run_suppression(root, config, manifest, rows, hashes):
    diagnostic = config['suppression']
    fold = next(f for f in manifest['folds'] if f['fold_id'] == diagnostic['fold_id'])
    groups, policy = fold_rows(rows, fold, manifest)
    vocab = policy['vocabulary']
    setting = dict(hidden_dim=diagnostic['hidden_dim'], dropout=diagnostic['dropout'],
                   weight_decay=diagnostic['weight_decay'], s_weight_decay=max(config['s_weight_decay_grid']))
    fits, curves = [], []
    for condition in ('A_B', 'A_B_S'):
        for seed in diagnostic['seeds']:
            _, _, convergence, epochs = fit_condition_v2(
                groups['train'], groups['validation'], vocab, manifest['input_conditions'][condition],
                dict(config, **setting), seed, manifest['preparation_configuration']['history_limit'])
            fits.append(dict(condition=condition, seed=seed, setting=setting, **convergence))
            curves.extend(dict(condition=condition, seed=seed, **e) for e in epochs)
            print(f"Suppression {condition} seed {seed}: validation {convergence['best_validation_log_likelihood_bits']:.6f}, "
                  f"{convergence['seconds']:.1f}s", flush=True)
    means = {c: np.mean([f['best_validation_log_likelihood_bits'] for f in fits if f['condition'] == c]).item()
             for c in ('A_B', 'A_B_S')}
    difference = means['A_B_S'] - means['A_B']
    summary = dict(configuration=config, input_sha256=hashes, device=config.get('device', 'cpu'),
                   versions={'numpy': np.__version__, 'torch': torch.__version__},
                   fold_id=fold['fold_id'], fits=fits, validation_means_bits=means,
                   difference_bits=difference, tolerance_bits=diagnostic['tolerance_bits'],
                   passed=abs(difference) <= diagnostic['tolerance_bits'])
    output = root / config['output_directory'] / 'suppression'
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'epochs.jsonl', curves)
    write_text(output / 'summary.json', json.dumps(summary, indent=2) + '\n')
    lines = ['# LSTM v2 validation-only S suppression check', '',
             f"Fold: {fold['fold_id']}; device: {summary['device']}; setting: {setting}.", '',
             '| Condition | Seed | Validation bits/coda | Best epoch | Epochs | Seconds |',
             '|---|---:|---:|---:|---:|---:|']
    for f in fits:
        lines.append(f"| {f['condition']} | {f['seed']} | {f['best_validation_log_likelihood_bits']:.6f} | "
                     f"{f['best_epoch']} | {f['epochs_trained']} | {f['seconds']:.2f} |")
    lines += ['', f"Mean A_B_S minus A_B: {difference:.6f} bits/coda; predeclared absolute tolerance: "
              f"{diagnostic['tolerance_bits']}. Check: {'PASS' if summary['passed'] else 'FAIL — stop before pilot'}.", '',
              'This diagnostic uses training and validation rows only. Test outcomes do not select or alter the grid.', '']
    write_text(root / config['suppression_report'], '\n'.join(lines))
    print('Suppression check: ' + ('PASS' if summary['passed'] else 'STOP'), flush=True)
    return summary


def restore_fit(fit, train, vocab, fields, config):
    timing_fields = [f for f in fields if not f.endswith('_history')]
    encoder = Encoder(train, vocab, timing_fields) if timing_fields else None
    model = ConditionModelV2(len(vocab), fields, dict(config, **fit['setting']),
                             len(encoder.names) if encoder else None)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in fit['state'].items()})
    return model.to(config.get('device', 'cpu')), encoder


def run_v2(config_path):
    root, config, manifest, rows, hashes = load_inputs(config_path)
    if config.get('mode') == 'suppression':
        return run_suppression(root, config, manifest, rows, hashes)
    if config.get('require_suppression'):
        diagnostic_path = root / config['output_directory'] / 'suppression' / 'summary.json'
        diagnostic = read_json(diagnostic_path)
        if not diagnostic['passed'] or diagnostic['configuration']['s_weight_decay_grid'] != config['s_weight_decay_grid']:
            raise ValueError('A passing suppression check with this S grid is required')
        for key in ('learning_rate', 'batch_size', 'max_epochs', 'patience', 'embedding_dim', 'device', 'suppression'):
            if diagnostic['configuration'][key] != config[key]:
                raise ValueError(f'Suppression diagnostic settings differ: {key}')
        for source, checksum in diagnostic['input_sha256'].items():
            if hashlib.sha256((root / source).read_bytes()).hexdigest() != checksum:
                raise ValueError('Suppression diagnostic inputs are stale')
    evaluation_config = read_json(root / config['evaluation_config'])
    if evaluation_config['seeds'] != config['seeds']:
        raise ValueError('Evaluation seeds differ from training seeds')
    hashes[config['evaluation_config']] = hashlib.sha256((root / config['evaluation_config']).read_bytes()).hexdigest()
    pilot = config.get('mode') == 'pilot'
    folds = [f for f in manifest['folds'] if not pilot or f['fold_id'] in config['pilot_folds']]
    conditions = {c: fs for c, fs in manifest['input_conditions'].items()
                  if not pilot or c in config['pilot_conditions']}
    if pilot and (len(folds) != len(config['pilot_folds']) or set(conditions) != set(config['pilot_conditions'])):
        raise ValueError('Unknown pilot fold or condition')
    reference = read_json(root / config['compare_against_logistic'])
    reference_predictions = read_jsonl(root / reference['configuration']['output_directory'] / 'predictions.jsonl')
    reference_index = {(r['fold_id'], r['condition'], r['coda_id']): r['probabilities'] for r in reference_predictions}
    predictions, scored, baselines, selections, convergence, curves, fold_metrics = [], [], [], [], [], [], []
    all_fit_diagnostics = []
    max_len = manifest['preparation_configuration']['history_limit']
    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timing_context = None
    for fold in folds:
        groups, policy = fold_rows(rows, fold, manifest)
        train, validation, test = groups['train'], groups['validation'], groups['test']
        vocab, prior = policy['vocabulary'], policy['label_prior']
        repeat_rate = sum(r['Ct'] == r['A_history'][-1] for r in train) / len(train)
        baseline_probs = [repeat_distribution(r['A_history'][-1], repeat_rate, prior) for r in test]
        _, baseline_scored = score_rows([[p[t] for t in vocab] for p in baseline_probs], test, vocab,
                                        'repeat_own_last', fold['fold_id'], 0)
        baselines.extend(baseline_scored)
        for condition, fields in conditions.items():
            if condition == 'none':
                for seed in config['seeds']:
                    ps, ss = score_rows([[prior[t] for t in vocab]] * len(test), test, vocab, condition, fold['fold_id'], seed)
                    predictions.extend(ps); scored.extend(ss)
                continue
            chosen, candidates, fits = select_neural_setting(train, validation, vocab, fields, config, max_len)
            selections.append(dict(fold_id=fold['fold_id'], condition=condition, chosen=chosen, candidates=candidates))
            selected_scored = []
            for fit in fits:
                metadata = dict(fold_id=fold['fold_id'], condition=condition, seed=fit['seed'],
                                setting_id=fit['setting_id'], setting=fit['setting'])
                curves.extend(dict(**metadata, **e) for e in fit['epochs'])
                all_fit_diagnostics.append(dict(**metadata, **fit['convergence']))
                if fit['setting_id'] != chosen['setting_id']:
                    continue
                model, encoder = restore_fit(fit, train, vocab, fields, config)
                batch = {k: v.to(config.get('device', 'cpu')) for k, v in build_batch(test, vocab, fields, max_len, encoder).items()}
                ps, ss = score_rows(v2_probabilities(model, batch), test, vocab, condition, fold['fold_id'], fit['seed'])
                predictions.extend(ps); scored.extend(ss); selected_scored.extend(ss)
                convergence.append(dict(**metadata, **fit['convergence']))
            reference_probs = [reference_index[(fold['fold_id'], condition, r['coda_id'])] for r in test]
            _, reference_scored = score_rows([[p[t] for t in vocab] for p in reference_probs], test, vocab, condition, fold['fold_id'], 0)
            fold_metrics.append(dict(fold_id=fold['fold_id'], condition=condition, lstm=seed_metrics(selected_scored),
                                     logistic=metrics(reference_scored), repeat_own_last=metrics(baseline_scored)))
            if timing_context is None:
                timing_context = (train, validation, vocab, fields, chosen['setting'], fold['fold_id'], condition)
            print(f"Finished {fold['fold_id']} {condition}: {len(fits)} fits; chosen {chosen['setting']}", flush=True)
            # Persist full curves after each completed search for auditability.
            write_jsonl(output / 'epochs.jsonl', curves)
    failures = []
    if pilot:
        for condition in conditions:
            values = [r for r in fold_metrics if r['condition'] == condition]
            if sum(r['lstm']['mean_log_likelihood_bits'] - r['repeat_own_last']['mean_log_likelihood_bits'] for r in values) < 0:
                failures.append(f'{condition}: mean over pilot folds below repeat-own-last')
            if all(r['lstm']['mean_log_likelihood_bits'] < r['logistic']['mean_log_likelihood_bits'] for r in values):
                failures.append(f'{condition}: below logistic v4 on every pilot fold')
        if any(r['hit_epoch_cap'] for r in convergence):
            failures.append('At least one selected-setting fit hit the epoch cap')
        summary = dict(fold_metrics=fold_metrics)
    else:
        summary, shared_scored, paired = evaluate(predictions, manifest, rows, config['seeds'],
                                                  evaluation_config['bootstrap_replicates'], evaluation_config['bootstrap_seed'],
                                                  evaluation_config['confidence'])
        checks = {}
        baseline_index = {r['coda_id']: r['log_likelihood_bits'] for r in baselines}
        for condition in ('A', 'A_B', 'A_S', 'A_B_S'):
            days = defaultdict(list)
            grouped = defaultdict(list)
            for r in scored:
                if r['condition'] == condition:
                    grouped[(r['recording_date'], r['coda_id'])].append(r['log_likelihood_bits'])
            for (day, coda_id), values in grouped.items():
                days[day].append(sum(values) / len(values) - baseline_index[coda_id])
            checks[condition] = day_bootstrap_bits(days, evaluation_config['bootstrap_replicates'],
                                                  evaluation_config['bootstrap_seed'], evaluation_config['confidence'])
        summary['repeat_baseline_checks'] = checks
        failures.extend(check_stop_conditions(summary, {}, checks))
        failures.extend(check_against_logistic(shared_scored, reference, root, evaluation_config))
        write_jsonl(output / 'paired_contrasts.jsonl', paired)
    if config.get('benchmark'):
        summary['benchmark'] = benchmark(timing_context, config, max_len)
        benchmark_curves = []
        for fit in summary['benchmark']['runs']:
            benchmark_curves.extend(dict(seed=fit['seed'], device=fit['device'], **epoch) for epoch in fit.pop('epochs'))
        write_jsonl(output / 'benchmark_epochs.jsonl', benchmark_curves)
    summary.update(configuration=config, input_sha256=hashes, versions={'numpy': np.__version__, 'torch': torch.__version__},
                   device=config.get('device', 'cpu'), failures=failures, gate_passed=not failures,
                   elapsed_seconds=time.perf_counter() - started, fold_metrics=fold_metrics,
                   total_candidate_fits=len(all_fit_diagnostics),
                   candidate_fits_at_epoch_cap=sum(f['hit_epoch_cap'] for f in all_fit_diagnostics))
    write_jsonl(output / 'predictions.jsonl', predictions)
    write_jsonl(output / 'scored_predictions.jsonl', scored)
    write_jsonl(output / 'baseline_predictions.jsonl', baselines)
    write_jsonl(output / 'selection.jsonl', selections)
    write_jsonl(output / 'convergence.jsonl', convergence)
    write_jsonl(output / 'all_fit_diagnostics.jsonl', all_fit_diagnostics)
    write_text(output / 'summary.json', json.dumps(summary, indent=2) + '\n')
    write_report(root, config, summary, convergence, curves)
    print('Gate: ' + ('PASS' if not failures else 'STOP: ' + '; '.join(failures)), flush=True)
    return summary


def seed_metrics(scored):
    by_seed = defaultdict(list)
    for r in scored:
        by_seed[r['seed']].append(r)
    per_seed = [metrics(rs) for rs in by_seed.values()]
    return {key: sum(m[key] for m in per_seed) / len(per_seed)
            for key in ('mean_log_likelihood_bits', 'accuracy', 'macro_f1')}


def benchmark(context, config, max_len):
    train, validation, vocab, fields, setting, fold_id, condition = context
    runs = []
    for seed in (42, 43, 44):
        _, _, convergence, epochs = fit_condition_v2(train, validation, vocab, fields,
                                               dict(config, **setting, device='cpu'), seed, max_len)
        runs.append(dict(seed=seed, epochs=epochs, **convergence))
    # Three no-S conditions have 18 settings; four S conditions have 90 settings.
    fits = 14 * 3 * (3 * len(config['hidden_grid']) * len(config['dropout_grid']) * len(config['weight_decay_grid'])
                     + 4 * len(config['hidden_grid']) * len(config['dropout_grid']) * len(config['weight_decay_grid']) * len(config['s_weight_decay_grid']))
    return dict(fold_id=fold_id, condition=condition, setting=setting, runs=runs,
                full_study_fits=fits, cpu_workers=config.get('workers', 1),
                estimated_cpu_seconds=sum(r['seconds'] for r in runs) / len(runs) * fits / config.get('workers', 1),
                estimate_note='Ideal parallel scaling from one fold/condition; excludes worker startup and variation across settings.',
                cuda_available=torch.cuda.is_available())


def write_report(root, config, summary, convergence, curves):
    lines = ['# RQ1 LSTM v2 ' + config.get('mode', 'full'), '',
             f"Device: {summary['device']}. Validation selection uses seeds {config['seeds']}. "
             'Test rows do not enter hyperparameter or epoch selection.', '',
             '| Fold | Condition | Model | LL bits/coda | Accuracy | Macro-F1 |',
             '|---|---|---|---:|---:|---:|']
    for r in summary['fold_metrics']:
        for model in ('lstm', 'logistic', 'repeat_own_last'):
            m = r[model]
            lines.append(f"| {r['fold_id']} | {r['condition']} | {model} | {m['mean_log_likelihood_bits']:.5f} | "
                         f"{m['accuracy']:.3%} | {m['macro_f1']:.4f} |")
    lines += ['', '## Selected fits', '', '| Fold | Condition | Seed | Setting | Beta | Gamma | Best epoch | Epochs | Hit cap |',
              '|---|---|---:|---|---:|---:|---:|---:|---|']
    for c in convergence:
        lines.append(f"| {c['fold_id']} | {c['condition']} | {c['seed']} | {c['setting']} | "
                     f"{c['beta']} | {c['gamma']} | {c['best_epoch']} | {c['epochs_trained']} | {c['hit_epoch_cap']} |")
    if curves and convergence:
        chosen = convergence[0]
        selected = [e for e in curves if all(e[k] == chosen[k] for k in ('fold_id', 'condition', 'seed', 'setting_id'))]
        lines += ['', '## Example learning curve', '', f"{chosen['fold_id']}, {chosen['condition']}, seed {chosen['seed']}.", '',
                  '| Epoch | Training LL | Validation LL |', '|---:|---:|---:|']
        for e in selected:
            if e['epoch'] in (1, chosen['best_epoch'], chosen['epochs_trained']) or e['epoch'] % 20 == 0:
                lines.append(f"| {e['epoch']} | {e['training_log_likelihood_bits']:.5f} | {e['validation_log_likelihood_bits']:.5f} |")
    if 'benchmark' in summary:
        b = summary['benchmark']
        lines += ['', '## CPU timing', '', '| Seed | Seconds | Validation LL |', '|---:|---:|---:|']
        for r in b['runs']:
            lines.append(f"| {r['seed']} | {r['seconds']:.2f} | {r['best_validation_log_likelihood_bits']:.5f} |")
        lines += ['', f"Full search: {b['full_study_fits']} fits; estimated {b['estimated_cpu_seconds']/3600:.2f} CPU wall hours "
                  f"with {b['cpu_workers']} concurrent fits. {b['estimate_note']}", '',
                  'CUDA available: ' + str(b['cuda_available']) + '.']
    lines += ['', '## Gate', '', 'PASS' if summary['gate_passed'] else 'STOP — no conclusions; no retuning on pilot test days.']
    lines += ['', f"{summary['candidate_fits_at_epoch_cap']} of {summary['total_candidate_fits']} candidate fits hit the epoch cap. "
              'The selected-setting cap check is reported above.']
    lines += ['- ' + reason for reason in summary['failures']]
    write_text(root / config['report'], '\n'.join(lines) + '\n')
