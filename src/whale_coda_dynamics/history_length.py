"""Own-history length sweep, PLAN [P2] (DEC-25): logistic v4 with k own lags.

Logistic v4 unchanged except the number of own-history lags: config['history_orders'] = {field: k} for
every k in the config; the partner keeps v4's history_order. Every k fits all of the manifest's conditions
with logistic.fit_v2_lattice: the shared evaluate() needs every condition, and DEC-16's stop rules use the
add-one-input contrasts. Conditions without own history must come out identical at every k.

The k = 3 run, fitted in the same environment as every other k, is the reference for every contrast. It is
first compared with the committed logistic v4 run (the gate): chosen penalties per fold and condition, and
predicted probabilities. Contrasts are LL(k) - LL(3) per condition, paired per coda, with the [P1] repeat
and switch parts from evaluate()'s scored rows and intervals from evaluation.day_bootstrap_bits.

Calibration: residual_planted's planting (lag-6 own-coda copies, its fractions, eligibility and sampling
seed), condition A, logistic regression refitted on the planted data at k = 9 and at k = 3 with v4's
procedure. Recovery ratio r = (LR(9) - LR(3)) / (oracle - LR(3)), the oracle mixing the planted rule into
the k = 3 refit; its interval and bound follow residual_planted.

Three stages (--stage): fit (one process per k), calibrate (one process per fold and fraction), report
(reads only the outputs of the other two and regenerates byte-identically without refitting). Every
fitting process runs with one BLAS thread.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
import platform
import time
from collections import defaultdict

import numpy as np
import scipy

from .evaluation import day_bootstrap_bits, log_likelihood_bits
from .logistic import (check_stop_conditions, fit_v2_lattice, format_ci, last_label_index, probabilities,
                       select_l2_and_fit)
from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text
from .residual import _hashes, _load, prepare_folds
from .residual_planted import (SPLITS, oracle_probabilities, paired_ratio_bootstrap, plant, ratio_status,
                               recovery_bound, relabel)

BLAS_THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')
COMPONENTS = ('log_likelihood_bits', 'repeat_bits', 'switch_choice_bits')
PLANTED_KEYS = ('lag', 'fractions', 'sampling_seed', 'bootstrap_replicates', 'bootstrap_seed', 'confidence',
                'ratio_handful')


# ---------------------------------------------------------------- shared

def require_one_blas_thread():
    wrong = {v: os.environ.get(v) for v in BLAS_THREAD_VARIABLES if os.environ.get(v) != '1'}
    if wrong:
        raise ValueError(f'Logistic fits run with one BLAS thread: set {", ".join(wrong)} to 1 (now {wrong})')


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                blas_threads={v: os.environ.get(v) for v in BLAS_THREAD_VARIABLES})


def order_config(logistic, field, order):
    """Logistic v4's configuration with the swept field's lag count set to `order`."""
    return dict(logistic, history_orders={field: order})


def load(config_path):
    path, root, config, manifest, manifest_path, source = _load(config_path)
    paths = {k: root / config[k] for k in ('logistic_config', 'evaluation_config', 'planted_config')}
    loaded = dict(path=path, root=root, config=config, manifest=manifest, manifest_path=manifest_path, source=source,
                  **{k + '_path': p for k, p in paths.items()}, **{k: read_json(p) for k, p in paths.items()})
    check_config(config, manifest, loaded['logistic_config'], loaded['planted_config'])
    return loaded


def check_config(config, manifest, logistic, planted):
    orders, reference, field = config['orders'], config['reference_order'], config['field']
    limit = manifest['preparation_configuration']['history_limit']
    if len(set(orders)) != len(orders) or any(type(k) is not int or not 1 <= k <= limit for k in orders):
        raise ValueError(f'Orders must be distinct integers from 1 to the history limit ({limit})')
    if reference not in orders or logistic['history_order'] != reference or 'history_orders' in logistic:
        raise ValueError("The reference order must be one of the orders and logistic v4's own history_order")
    named = [config['primary']['order'], config['secondary_order'], config['calibration']['order']]
    if any(k not in orders or k == reference for k in named):
        raise ValueError('Primary, secondary and calibration orders must be swept orders other than the reference')
    conditions = manifest['input_conditions']
    if config['primary']['condition'] not in config['conditions'] or any(
            c not in conditions or field not in conditions[c] for c in config['conditions'] + [config['calibration']['condition']]):
        raise ValueError(f'Swept and calibration conditions must be manifest conditions that use {field}')
    if any(k not in planted for k in PLANTED_KEYS):
        raise ValueError('The planted-signal configuration lacks a setting the calibration reuses')


def order_directory(loaded, order):
    return loaded['root'] / loaded['config']['output_directory'] / f'k{order}'


def calibration_directory(loaded):
    return loaded['root'] / loaded['config']['output_directory'] / 'calibration'


def calibration_configuration(loaded):
    config, planted = loaded['config'], loaded['planted_config']
    return dict(condition=config['calibration']['condition'], field=config['field'],
                orders=[config['reference_order'], config['calibration']['order']],
                **{k: planted[k] for k in ('lag', 'fractions', 'sampling_seed')})


def process_pool(workers):
    return concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))


# ---------------------------------------------------------------- fit stage

def _fit_order(task):
    """One k: every manifest condition, logistic v4 with the swept field's lag count set to k. Module level
    so that spawned processes can import it."""
    config_path, order = task
    require_one_blas_thread()
    start = time.perf_counter()
    loaded = load(config_path)
    config = loaded['config']
    fit_config = order_config(loaded['logistic_config'], config['field'], order)
    fitted = fit_v2_lattice(fit_config, loaded['manifest'], read_jsonl(loaded['source']), loaded['evaluation_config'])
    summary = fitted['summary']
    summary.update(order=order, configuration=fit_config, baseline_metrics=fitted['baseline_metrics'],
                   repeat_baseline_checks=fitted['baseline_checks'],
                   stop_reasons=check_stop_conditions(summary, fitted['baseline_metrics'], fitted['baseline_checks']),
                   environment=environment())
    inputs = [loaded[k] for k in ('source', 'manifest_path', 'path', 'logistic_config_path', 'evaluation_config_path')]
    if order == config['reference_order']:
        summary['gate'], gate_inputs = reference_gate(loaded['root'], config['gate'], fitted)
        inputs += gate_inputs
    summary['input_sha256'] = _hashes(loaded['root'], inputs)
    output = order_directory(loaded, order)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'l2_selection.jsonl', fitted['selections'])
    write_jsonl(output / 'scored.jsonl', [r for r in fitted['scored'] if r['condition'] in config['conditions']])
    write_text(output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return dict(order=order, seconds=time.perf_counter() - start, stop_reasons=summary['stop_reasons'],
                gate=summary.get('gate'))


def reference_gate(root, spec, fitted):
    """The reference fit against the committed logistic v4 run. It passes when at most
    spec['maximum_penalty_differences'] fold and condition fits chose a different penalty and every
    condition's mean log-likelihood is within spec['log_likelihood_tolerance_bits'] of v4's (DEC-25 addendum).
    Predicted probabilities are reported, not enforced: v4's fits reproduce only to ~1e-4 in probability at
    the smallest penalties (optimizer tolerance; the BLAS thread count alone moves them that much). Fits
    differing by more than spec['probability_listing_threshold'] are listed; a fit whose penalty differs is
    listed with the validation log-likelihood of both penalties under this fit and under v4's."""
    directory = root / spec['results_directory']
    paths = [directory / 'predictions.jsonl', directory / 'l2_selection.jsonl', directory / 'evaluation' / 'summary.json']
    stored = {(r['coda_id'], r['condition']): r['probabilities'] for r in read_jsonl(paths[0])}
    stored_selection = {(r['fold_id'], r['condition']): r for r in read_jsonl(paths[1])}
    stored_metrics = read_json(paths[2])['metrics']
    selection = {(r['fold_id'], r['condition']): r for r in fitted['selections']}
    if {(p['coda_id'], p['condition']) for p in fitted['predictions']} != set(stored) or set(selection) != set(stored_selection):
        raise ValueError('The reference fit and logistic v4 cover different targets, folds or conditions')
    differences = {}
    for p in fitted['predictions']:
        theirs = stored[p['coda_id'], p['condition']]
        if set(theirs) != set(p['probabilities']):
            raise ValueError('The reference fit and logistic v4 use different vocabularies')
        key = (p['fold_id'], p['condition'])
        differences[key] = max(differences.get(key, 0.0), max(abs(v - theirs[t]) for t, v in p['probabilities'].items()))
    pairs = []
    for (fold_id, condition), difference in differences.items():
        row = dict(fold_id=fold_id, condition=condition, max_abs_probability_difference=difference)
        if condition != 'none':
            mine, theirs = selection[fold_id, condition], stored_selection[fold_id, condition]
            chosen, v4_chosen = (mine['chosen_l2'], mine['chosen_l2_s']), (theirs['chosen_l2'], theirs['chosen_l2_s'])
            row.update(chosen_l2=chosen[0], chosen_l2_s=chosen[1], v4_chosen_l2=v4_chosen[0], v4_chosen_l2_s=v4_chosen[1],
                       same_penalty=chosen == v4_chosen)
            if chosen != v4_chosen:
                grid = {(g['l2'], g['l2_s']): g['validation_log_likelihood_bits'] for g in mine['grid']}
                v4_grid = {(g['l2'], g['l2_s']): g['validation_log_likelihood_bits'] for g in theirs['grid']}
                # None where a penalty is not on the other run's grid (only when the grids differ)
                row.update(validation_bits_own_choice=grid[chosen], validation_bits_v4_choice=grid.get(v4_chosen),
                           v4_run_validation_bits_own_choice=v4_grid.get(chosen),
                           v4_run_validation_bits_v4_choice=v4_grid[v4_chosen])
        pairs.append(row)
    agreeing = [r for r in pairs if r.get('same_penalty', True)]
    differing = [r for r in pairs if not r.get('same_penalty', True)]
    by_condition = {}
    for condition in dict.fromkeys(r['condition'] for r in pairs):
        rows = [r for r in pairs if r['condition'] == condition]
        by_condition[condition] = dict(
            fits=len(rows), penalties_differing=sum(not r.get('same_penalty', True) for r in rows),
            max_abs_probability_difference=max(r['max_abs_probability_difference'] for r in rows),
            mean_log_likelihood_bits=fitted['summary']['metrics'][condition]['mean_log_likelihood_bits'],
            v4_mean_log_likelihood_bits=stored_metrics[condition]['mean_log_likelihood_bits'])
        by_condition[condition]['log_likelihood_difference_bits'] = (
            by_condition[condition]['mean_log_likelihood_bits'] - by_condition[condition]['v4_mean_log_likelihood_bits'])
    ll_max = max(abs(c['log_likelihood_difference_bits']) for c in by_condition.values())
    threshold = spec['probability_listing_threshold']
    gate = dict(maximum_penalty_differences=spec['maximum_penalty_differences'],
                log_likelihood_tolerance_bits=spec['log_likelihood_tolerance_bits'],
                probability_listing_threshold=threshold,
                fits_compared=sum('same_penalty' in r for r in pairs),
                max_abs_log_likelihood_difference_bits=ll_max,
                max_abs_probability_difference=max(r['max_abs_probability_difference'] for r in pairs),
                max_abs_probability_difference_same_penalty=max(r['max_abs_probability_difference'] for r in agreeing),
                fits_above_probability_threshold=sorted((r for r in pairs if r['max_abs_probability_difference'] > threshold),
                                                        key=lambda r: -r['max_abs_probability_difference']),
                penalty_differences=differing, by_condition=by_condition,
                passed=len(differing) <= spec['maximum_penalty_differences'] and ll_max <= spec['log_likelihood_tolerance_bits'],
                pairs=pairs)
    return gate, paths


def run_fit(config_path, orders=None, workers=None):
    require_one_blas_thread()
    loaded = load(config_path)
    orders = sorted(orders or loaded['config']['orders'], reverse=True)  # most lags, the slowest, first
    if set(orders) - set(loaded['config']['orders']):
        raise ValueError('Every order to fit must be one of the configured orders')
    results = []
    with process_pool(min(workers or os.cpu_count(), len(orders))) as pool:
        futures = [pool.submit(_fit_order, (str(loaded['path']), k)) for k in orders]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            gate = result['gate']
            gate_text = '' if gate is None else (
                f"; gate {'passed' if gate['passed'] else 'FAILED'} ({len(gate['penalty_differences'])} of "
                f"{gate['fits_compared']} penalties differ, mean LL within {gate['max_abs_log_likelihood_difference_bits']:.2g} "
                f"bits/coda; max abs probability difference {gate['max_abs_probability_difference']:.3g}, reported only)")
            print(f"k = {result['order']}: {result['seconds']:.0f} s, {len(result['stop_reasons'])} stop rule(s) fired"
                  f"{gate_text}", flush=True)
            results.append(result)
    return results


# ---------------------------------------------------------------- calibration stage

def _calibrate(task):
    """One fold and fraction: residual_planted's planting on every split, then condition A refitted on the
    planted training and validation days at the reference order and at the calibration order with v4's
    procedure, and the test day scored on its planted labels. Module level for spawned processes."""
    config_path, fold_id, fraction = task
    require_one_blas_thread()
    loaded = load(config_path)
    config, manifest, planted_config = loaded['config'], loaded['manifest'], loaded['planted_config']
    fold = next(f for f in manifest['folds'] if f['fold_id'] == fold_id)
    prepared = prepare_folds(read_jsonl(loaded['source']), manifest, test_days=[fold['test_day']])[0]
    vocab, lag = prepared['policy']['vocabulary'], planted_config['lag']
    eligible, planted = {}, {}
    for split in SPLITS:
        eligible[split], planted[split] = plant(prepared[split], lag, planted_config['fractions'],
                                                planted_config['sampling_seed'], fold_id, split)
    relabelled = {s: relabel(prepared[s], planted[s][fraction], lag) for s in SPLITS}
    fields = manifest['input_conditions'][config['calibration']['condition']]
    test = relabelled['test']
    y = [vocab.index(r['Ct']) for r in test]
    own = last_label_index(test, 'A_history', vocab) if 'A_history' in fields else None
    partner = last_label_index(test, 'B_history', vocab) if 'B_history' in fields else None
    reference = config['reference_order']
    lr, fits = {}, []
    for order in (reference, config['calibration']['order']):
        encoder, chosen, _ = select_l2_and_fit(relabelled['train'], relabelled['validation'], vocab, fields,
                                               order_config(loaded['logistic_config'], config['field'], order))
        lr[order] = probabilities(encoder.transform(test), chosen['w'], chosen['beta'], chosen['gamma'], own, partner)
        fits.append(dict(order=order, chosen_l2=chosen['l2'], chosen_l2_s=chosen['l2_s'],
                         beta=None if chosen['beta'] is None else float(chosen['beta']),
                         gamma=None if chosen['gamma'] is None else float(chosen['gamma']),
                         validation_log_likelihood_bits=chosen['validation_log_likelihood_bits'],
                         convergence=chosen['convergence']))
    oracle = oracle_probabilities(lr[reference], test, eligible['test'], fraction, lag, vocab)
    eligible_test, planted_test = set(eligible['test']), set(planted['test'][fraction])
    scores = [dict(fold_id=fold_id, fraction=fraction, coda_id=row['coda_id'], recording_date=row['recording_date'],
                   target=row['Ct'], eligible=i in eligible_test, planted=i in planted_test,
                   lr_bits={str(k): log_likelihood_bits(float(p[i, y[i]])) for k, p in lr.items()},
                   oracle_bits=log_likelihood_bits(float(oracle[i, y[i]])))
              for i, row in enumerate(test)]
    selection = dict(fold_id=fold_id, fraction=fraction, test_day=fold['test_day'],
                     targets={s: len(prepared[s]) for s in SPLITS}, eligible={s: len(eligible[s]) for s in SPLITS},
                     planted={s: len(planted[s][fraction]) for s in SPLITS}, fits=fits)
    return scores, selection


def run_calibration(config_path, workers=None):
    require_one_blas_thread()
    loaded = load(config_path)
    tasks = [(str(loaded['path']), fold['fold_id'], f)
             for fold in loaded['manifest']['folds'] for f in loaded['planted_config']['fractions']]
    start = time.perf_counter()
    with process_pool(min(workers or os.cpu_count(), len(tasks))) as pool:
        results = [future.result() for future in [pool.submit(_calibrate, t) for t in tasks]]
    output = calibration_directory(loaded)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'scores.jsonl', [s for scores, _ in results for s in scores])
    write_jsonl(output / 'selection.jsonl', [selection for _, selection in results])
    summary = dict(configuration=calibration_configuration(loaded), environment=environment(),
                   input_sha256=_hashes(loaded['root'], [loaded[k] for k in (
                       'source', 'manifest_path', 'path', 'logistic_config_path', 'planted_config_path')]))
    write_text(output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f'Calibration: {len(tasks)} fold and fraction tasks in {time.perf_counter() - start:.0f} s', flush=True)


# ---------------------------------------------------------------- report stage

def require_current(loaded, summary_path):
    """An output's recorded inputs must be the files as they are now; otherwise it is stale."""
    recorded = read_json(summary_path)['input_sha256']
    if recorded != _hashes(loaded['root'], [loaded['root'] / p for p in recorded]):
        raise ValueError(f'{summary_path} was produced from inputs that have changed since; refit it')


def by_coda(rows, condition):
    out = {r['coda_id']: r for r in rows if r['condition'] == condition}
    if len(out) != sum(r['condition'] == condition for r in rows):
        raise ValueError('More than one scored row per coda and condition')
    return out


def order_contrast(rows, reference_rows, condition, boot):
    """LL(k) - LL(reference) per coda, total and [P1] parts, with the shared day bootstrap."""
    mine, base = by_coda(rows, condition), by_coda(reference_rows, condition)
    if set(mine) != set(base):
        raise ValueError('The two orders scored different targets')
    days = {f: defaultdict(list) for f in COMPONENTS}
    for coda_id in sorted(base):
        a, b = mine[coda_id], base[coda_id]
        if (a['recording_date'], a['target'], a['own_last']) != (b['recording_date'], b['target'], b['own_last']):
            raise ValueError('The two orders disagree on a target')
        for f in COMPONENTS:
            if a[f] is not None and b[f] is not None:
                days[f][a['recording_date']].append(a[f] - b[f])
    return {f: day_bootstrap_bits(dict(values), *boot) for f, values in days.items()}


def penalty_summary(selections, condition, grid):
    values = sorted(s['chosen_l2'] for s in selections if s['condition'] == condition)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return dict(median=median, at_grid_edge=sum(v in (min(grid), max(grid)) for v in values), folds=len(values))


def calibration_summary(scores, config, planted_config, a_upper):
    reference, order = str(config['reference_order']), str(config['calibration']['order'])
    boot = (planted_config['bootstrap_replicates'], planted_config['bootstrap_seed'], planted_config['confidence'])
    per_fraction = {}
    for f in planted_config['fractions']:
        rs = [r for r in scores if r['fraction'] == f]
        gain, oracle_gain = defaultdict(list), defaultdict(list)
        for r in rs:
            gain[r['recording_date']].append(r['lr_bits'][order] - r['lr_bits'][reference])
            oracle_gain[r['recording_date']].append(r['oracle_bits'] - r['lr_bits'][reference])
        gain_ci, oracle_ci = day_bootstrap_bits(dict(gain), *boot), day_bootstrap_bits(dict(oracle_gain), *boot)
        ratio = paired_ratio_bootstrap(dict(gain), dict(oracle_gain), *boot)
        if ratio['numerator_ci'] != gain_ci['ci_bits_per_coda'] or ratio['denominator_ci'] != oracle_ci['ci_bits_per_coda']:
            raise ValueError('The ratio bootstrap did not resample the same days as the shared day bootstrap')
        ratio['status'] = ratio_status(ratio, planted_config['ratio_handful'])
        planted = [r for r in rs if r['planted']]
        mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
        per_fraction[str(f)] = dict(
            fraction=f, test_targets=len(rs), eligible_test_targets=sum(r['eligible'] for r in rs),
            planted_test_targets=len(planted), planted_test_share=len(planted) / len(rs),
            gain=gain_ci, oracle_gain=oracle_ci, recovery=ratio,
            planted=dict(targets=len(planted), oracle_bits=mean([r['oracle_bits'] for r in planted]),
                         **{f'lr_bits_k{k}': mean([r['lr_bits'][k] for r in planted]) for k in (reference, order)}),
            # A bound needs a positive upper bound to divide; with a real-data interval at or below zero there is none.
            bound_bits_per_coda=recovery_bound(a_upper, ratio) if a_upper > 0 else None)
    return dict(lag=planted_config['lag'], real_data_contrast_upper_bits=a_upper, per_fraction=per_fraction)


def planting_check(scores, residual_scores, reference):
    """The calibration's planting is residual_planted's: on the same fold, fraction and target, the same
    eligibility, planted flag and planted label (required), and the reference-order refit should give the
    residual run's refitted logistic regression up to cross-machine noise (reported)."""
    theirs = {(r['fold_id'], r['fraction'], r['coda_id']): r for r in residual_scores}
    differences = []
    for r in scores:
        other = theirs.get((r['fold_id'], r['fraction'], r['coda_id']))
        if other is None or (r['eligible'], r['planted'], r['target']) != (other['eligible'], other['planted'], other['target']):
            raise ValueError(f"Planting differs from the residual planted run at {r['fold_id']}, f={r['fraction']}, {r['coda_id']}")
        differences.append(abs(r['lr_bits'][str(reference)] - other['lr_bits']))
    return dict(targets=len(scores), flags_and_labels_identical=True,
                max_abs_refit_bits_difference=max(differences),
                targets_differing_beyond_tolerance=sum(d > 1e-5 for d in differences))


def run_report(config_path):
    loaded = load(config_path)
    root, config, manifest = loaded['root'], loaded['config'], loaded['manifest']
    orders, reference, field = config['orders'], config['reference_order'], config['field']
    evaluation = loaded['evaluation_config']
    boot = (evaluation['bootstrap_replicates'], evaluation['bootstrap_seed'], evaluation['confidence'])
    inputs = [loaded[k] for k in ('source', 'manifest_path', 'path', 'logistic_config_path', 'evaluation_config_path',
                                  'planted_config_path')]
    summaries, selections, scored = {}, {}, {}
    for k in orders:
        directory = order_directory(loaded, k)
        paths = [directory / name for name in ('summary.json', 'l2_selection.jsonl', 'scored.jsonl')]
        if not all(p.is_file() for p in paths):
            raise ValueError(f'k = {k} has not been fitted (run --stage fit)')
        require_current(loaded, paths[0])
        summaries[k], selections[k], scored[k] = read_json(paths[0]), read_jsonl(paths[1]), read_jsonl(paths[2])
        if summaries[k]['configuration'] != order_config(loaded['logistic_config'], field, k):
            raise ValueError(f'k = {k} was fitted with another configuration')
        inputs += paths
    gate = summaries[reference]['gate']
    if config['gate']['enforce'] and not gate['passed']:
        raise ValueError('The reference fit does not reproduce logistic v4 (see k3/summary.json, gate); stop here')
    # Conditions without the swept field are the same model at every k, so they must be identical.
    constant = [c for c, fields in manifest['input_conditions'].items() if field not in fields]
    for k in orders:
        for c in constant:
            if summaries[k]['metrics'][c] != summaries[reference]['metrics'][c]:
                raise ValueError(f'Condition {c} does not use {field} but differs between k = {k} and k = {reference}')

    grid = loaded['logistic_config']['l2_grid']
    curve = {c: {str(k): dict(mean_log_likelihood_bits=summaries[k]['metrics'][c]['mean_log_likelihood_bits'],
                              accuracy=summaries[k]['metrics'][c]['accuracy'], macro_f1=summaries[k]['metrics'][c]['macro_f1'],
                              versus_repeat_own_last=summaries[k]['repeat_baseline_checks'].get(c),
                              history_l2=penalty_summary(selections[k], c, grid))
                 for k in orders} for c in config['conditions']}
    primary = config['primary']
    contrasts = {}
    for c in config['conditions']:
        contrasts[c] = {}
        for k in orders:
            if k == reference:
                continue
            components = order_contrast(scored[k], scored[reference], c, boot)
            role = ('primary' if (c, k) == (primary['condition'], primary['order']) else
                    'secondary' if k == config['secondary_order'] else 'descriptive')
            high = components['log_likelihood_bits']['ci_bits_per_coda'][1]
            contrasts[c][str(k)] = dict(role=role, components=components, fitting_flag=k > reference and high < 0)
    stop_rules = {str(k): summaries[k]['stop_reasons'] for k in orders}

    calibration_dir = calibration_directory(loaded)
    calibration_paths = [calibration_dir / name for name in ('summary.json', 'scores.jsonl', 'selection.jsonl')]
    if not all(p.is_file() for p in calibration_paths):
        raise ValueError('The calibration has not been run (run --stage calibrate)')
    require_current(loaded, calibration_paths[0])
    if read_json(calibration_paths[0])['configuration'] != calibration_configuration(loaded):
        raise ValueError('The calibration was run with another configuration')
    calibration_scores, calibration_selection = read_jsonl(calibration_paths[1]), read_jsonl(calibration_paths[2])
    a_contrast = contrasts.get(config['calibration']['condition'], {}).get(str(config['calibration']['order']))
    if a_contrast is None:
        raise ValueError("The calibration's bound needs the real-data contrast of its condition and order")
    calibration = calibration_summary(calibration_scores, config, loaded['planted_config'],
                                      a_contrast['components']['log_likelihood_bits']['ci_bits_per_coda'][1])
    calibration['selection'] = calibration_selection
    residual_scores_path = root / config['calibration']['residual_planted_scores']
    calibration['planting_check'] = planting_check(calibration_scores, read_jsonl(residual_scores_path), reference)
    inputs += calibration_paths + [residual_scores_path]

    result = dict(units='bits per coda', field=field, orders=orders, reference_order=reference,
                  conditions=config['conditions'], primary=primary, secondary_order=config['secondary_order'],
                  comparison_targets=summaries[reference]['comparison_targets'],
                  repeat_own_last_bits=summaries[reference]['baseline_metrics']['repeat_own_last']['mean_log_likelihood_bits'],
                  bootstrap=dict(zip(('replicates', 'seed', 'confidence'), boot)),
                  gate={k: v for k, v in gate.items() if k != 'pairs'}, constant_conditions=constant,
                  curve=curve, contrasts=contrasts, stop_rules=stop_rules, calibration=calibration,
                  environment={str(k): summaries[k]['environment'] for k in orders},
                  fits=sum(len(s['grid']) for k in orders for s in selections[k]))
    result['reading'] = reading_lines(result, config)
    result['input_sha256'] = _hashes(root, inputs)
    output = root / config['output_directory']
    write_text(output / 'summary.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(report_lines(result, config, loaded)))
    return result


# ---------------------------------------------------------------- reading and report

def _f(value, digits=5):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def _g(value):
    return 'n/a' if value is None else f'{value:.4g}'


def _stat(stats):
    """Weighted mean, unweighted day mean, CI and positive days of one day-bootstrap result."""
    return (f"{stats['mean_bits_per_coda']:.5f} | {stats['unweighted_mean_bits_per_coda']:.5f} | "
            f"{format_ci(stats['ci_bits_per_coda'])} | {stats['positive_days']}/{stats['days']}")


def _part(stats):
    return f"{stats['mean_bits_per_coda']:.5f} {format_ci(stats['ci_bits_per_coda'])}"


def _contrast_text(contrast, condition, k, reference):
    s = contrast['components']['log_likelihood_bits']
    return (f"LL({k}) - LL({reference}) in {condition}: {s['mean_bits_per_coda']:.5f} bits/coda "
            f"(unweighted day mean {s['unweighted_mean_bits_per_coda']:.5f}, 95% CI {format_ci(s['ci_bits_per_coda'])}, "
            f"{s['positive_days']}/{s['days']} days positive)")


def reading_lines(result, config):
    """DEC-25's Reading, chosen mechanically by the primary contrast's interval; nothing beyond it."""
    reference, primary = result['reference_order'], result['primary']
    condition, k = primary['condition'], primary['order']
    contrast = result['contrasts'][condition][str(k)]
    total = contrast['components']['log_likelihood_bits']
    low, high = total['ci_bits_per_coda']
    lines = [f"Primary, pre-specified: {_contrast_text(contrast, condition, k, reference)}. Repeat part "
             f"{_part(contrast['components']['repeat_bits'])}; switch part {_part(contrast['components']['switch_choice_bits'])}."]
    blocked = [str(o) for o in (reference, k) if result['stop_rules'][str(o)]]
    if not result['gate']['passed']:
        lines.append('No reading: the reference fit does not reproduce logistic v4 (gate above).')
    elif blocked:
        lines.append(f"No reading: DEC-16 stop rules fire on k = {', '.join(blocked)} (see the stop rules above).")
    elif high < 0:
        lines.append(f"No reading: the interval lies entirely below zero, so adding own lags beyond {reference} reliably "
                     "lowered held-out log-likelihood. That signals a fitting problem (optimisation or regularisation) "
                     "to investigate before interpreting anything.")
    elif low > 0:
        parts = [name for key, name in (('repeat_bits', 'repeat'), ('switch_choice_bits', 'switch'))
                 if contrast['components'][key]['ci_bits_per_coda'][0] > 0]
        where = (f"It lives in the {' and the '.join(parts)} part{'s' if len(parts) > 1 else ''} (interval above zero)."
                 if parts else 'Neither the repeat nor the switch part alone has an interval above zero.')
        lines.append(f"Own history beyond {reference} codas carries usable information beyond partner and timing. "
                     "Still not grammar: a stable repertoire or a slowly changing behavioural state also produce it "
                     f"(Gero, Whitehead & Rendell 2016b; Aluma et al. 2026). {where}")
    else:
        bounds = [(f, p) for f, p in result['calibration']['per_fraction'].items() if p['bound_bits_per_coda'] is not None]
        cal = config['calibration']
        bound_text = (f"Calibration (condition {cal['condition']}): " + '; '.join(
                          f"at f = {f}, a lag-{result['calibration']['lag']} copy pattern worth more than "
                          f"{p['bound_bits_per_coda']:.4f} bits/coda to the oracle would have been detected" for f, p in bounds) + '.'
                      if bounds else 'The calibration gives no bound (see the calibration above).')
        lines.append(f"No detectable information beyond {reference} own codas for this model at this data size. "
                     f"{bound_text} A null may also mean too little data: limited data and capacity push contrasts "
                     "toward zero (PLAN §9).")
    secondary = config['secondary_order']
    lines.append('Secondary, pre-specified: ' + '; '.join(
        _contrast_text(result['contrasts'][c][str(secondary)], c, secondary, reference) for c in result['conditions']) + '.')
    return lines


def report_lines(result, config, loaded):
    reference, orders, conditions = result['reference_order'], result['orders'], result['conditions']
    gate, cal = result['gate'], result['calibration']
    planted_config, logistic = loaded['planted_config'], loaded['logistic_config']
    field = result['field']
    env = result['environment'][str(reference)]
    k_cal = config['calibration']['order']
    lines = ['# RQ1 own-history length sweep (PLAN [P2], DEC-25)', '',
             f"Logistic v4 unchanged except the number of own-history lags ({field}): k in {orders}. The partner keeps "
             f"{logistic['history_order']} lags; copy weights, the L2 grid {logistic['l2_grid']}, the joint history and S "
             "penalties (DEC-20), the selection rule (validation day, ties to the larger value), the 14 folds, the "
             "targets, the UNK policy and the labels (`guy_coda_type`) are v4's. Every k fits all eight conditions: the "
             "shared evaluation needs every condition, and DEC-16's stop rules use the add-one-input contrasts. "
             f"Conditions without own history ({', '.join(result['constant_conditions'])}) are refitted at every k and are "
             f"identical to the k = {reference} run (checked exactly). All k, including the reference k = {reference}, "
             f"were fitted in one environment (Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}, one "
             f"BLAS thread per fit): {result['fits']} fits, all converged (fit() raises otherwise).", '',
             f"Contrasts are LL(k) - LL({reference}) per condition, paired per coda against this environment's k = "
             f"{reference} fit (not the committed v4 run), with the [P1] repeat and switch parts. Intervals are "
             f"day-bootstrap ({result['bootstrap']['replicates']:,} replicates, seed {result['bootstrap']['seed']}); the "
             'unweighted day mean is the mean of the 14 day-level contrasts.', '',
             f"## Gate: k = {reference} here against the committed logistic v4 run", '',
             f"Chosen penalties agree on {gate['fits_compared'] - len(gate['penalty_differences'])} of "
             f"{gate['fits_compared']} fold and condition fits, and every condition's mean log-likelihood is within "
             f"{gate['max_abs_log_likelihood_difference_bits']:.2g} bits/coda of v4's. The gate (DEC-25 addendum) requires "
             f"at most {gate['maximum_penalty_differences']} differing penalty and mean log-likelihoods within "
             f"{gate['log_likelihood_tolerance_bits']:g} bits/coda: {'passed' if gate['passed'] else 'FAILED'}. Predicted "
             "probabilities are reported, not enforced, because v4's fits reproduce only to about 1e-4 at the smallest "
             "penalties (optimizer tolerance gtol 1e-6; the BLAS thread count alone moves them that much): largest absolute "
             f"difference {gate['max_abs_probability_difference']:.3g} over all fits "
             f"({gate['max_abs_probability_difference_same_penalty']:.3g} over the fits whose penalties agree); "
             f"{len(gate['fits_above_probability_threshold'])} fits differ by more than "
             f"{gate['probability_listing_threshold']:g} (listed below).", '',
             '| Condition | Fits | Penalties differing | Max abs probability difference | LL here | LL v4 | Difference |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for c, g in gate['by_condition'].items():
        lines.append(f"| {c} | {g['fits']} | {g['penalties_differing']} | {g['max_abs_probability_difference']:.3g} | "
                     f"{g['mean_log_likelihood_bits']:.7f} | {g['v4_mean_log_likelihood_bits']:.7f} | "
                     f"{g['log_likelihood_difference_bits']:.2g} |")
    if gate['fits_above_probability_threshold']:
        lines += ['', f"Fits whose predicted probabilities differ from v4's by more than {gate['probability_listing_threshold']:g}, "
                  'with the chosen penalties (history, S):', '',
                  '| Fold | Condition | History L2 | S L2 | Max abs probability difference |', '|---|---|---:|---:|---:|']
        lines += [f"| {r['fold_id']} | {r['condition']} | {r.get('chosen_l2', 'n/a')} | {r.get('chosen_l2_s') or 'n/a'} | "
                  f"{r['max_abs_probability_difference']:.3g} |" for r in gate['fits_above_probability_threshold']]
    if gate['penalty_differences']:
        lines += ['', 'Fits whose chosen penalty (history, S) differs from v4\'s, with the validation-day log-likelihood '
                  '(bits/coda) of both penalties under this fit and under v4\'s run:', '',
                  '| Fold | Condition | Chosen here | v4 chose | Here: own choice | Here: v4\'s choice | v4 run: own choice | '
                  'v4 run: v4\'s choice | Max abs probability difference |',
                  '|---|---|---|---|---:|---:|---:|---:|---:|']
        for d in gate['penalty_differences']:
            lines.append(f"| {d['fold_id']} | {d['condition']} | {d['chosen_l2']}, {d['chosen_l2_s']} | "
                         f"{d['v4_chosen_l2']}, {d['v4_chosen_l2_s']} | {_f(d['validation_bits_own_choice'], 8)} | "
                         f"{_f(d['validation_bits_v4_choice'], 8)} | {_f(d['v4_run_validation_bits_own_choice'], 8)} | "
                         f"{_f(d['v4_run_validation_bits_v4_choice'], 8)} | {d['max_abs_probability_difference']:.3g} |")

    lines += [line for note in config.get('process_notes', []) for line in ('', note)]
    lines += ['', '## Log-likelihood by number of own lags', '',
              f"Held-out log-likelihood (bits/coda, {result['comparison_targets']:,} targets), accuracy, and the chosen "
              "history L2 (median over folds; folds at a grid edge). Repeat-own-last: "
              f"{result['repeat_own_last_bits']:.5f} bits/coda.", '',
              '| k | ' + ' | '.join(f'{c} LL | {c} accuracy | {c} history L2 (edge)' for c in conditions) + ' |',
              '|---:|' + '---:|---:|---:|' * len(conditions)]
    for k in orders:
        cells = []
        for c in conditions:
            p = result['curve'][c][str(k)]
            cells.append(f"{p['mean_log_likelihood_bits']:.5f} | {p['accuracy']:.3%} | {p['history_l2']['median']:g} "
                         f"({p['history_l2']['at_grid_edge']}/{p['history_l2']['folds']})")
        lines.append(f"| {k}{' *' if result['stop_rules'][str(k)] else ''} | " + ' | '.join(cells) + ' |')
    stopped = [str(k) for k in orders if result['stop_rules'][str(k)]]
    stop_note = (f"Rows marked * are the k at which a DEC-16 stop rule fires (k = {', '.join(stopped)}; see the stop rules "
                 "below); no conclusion is drawn from those k." if stopped else None)
    lines += ['', stop_note] if stop_note else []

    lines += ['', f'## Contrasts LL(k) - LL({reference})', '',
              f"Primary (pre-specified): LL({config['primary']['order']}) - LL({reference}) in {config['primary']['condition']}. "
              f"Secondary (pre-specified): LL({config['secondary_order']}) - LL({reference}), in each condition. The rest "
              'of the curve is descriptive.', '',
              '| Condition | k | Role | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part [CI] | Switch part [CI] |',
              '|---|---:|---|---:|---:|---|---:|---:|---:|']
    for c in conditions:
        for k, contrast in result['contrasts'][c].items():
            comp = contrast['components']
            role = f"**{contrast['role']}**" if contrast['role'] != 'descriptive' else contrast['role']
            role += ' *' if result['stop_rules'][k] else ''
            lines.append(f"| {c} | {k} | {role} | {_stat(comp['log_likelihood_bits'])} | {_part(comp['repeat_bits'])} | "
                         f"{_part(comp['switch_choice_bits'])} |")
    flagged = [f"{c}, k = {k}" for c in conditions for k, v in result['contrasts'][c].items() if v['fitting_flag']]
    lines += ['', stop_note] if stop_note else []
    lines += ['', (f"Fitting flags: more own lags than {reference} with an interval entirely below zero ({'; '.join(flagged)}). "
                   'That signals a fitting problem (optimisation or regularisation), not a finding; investigate before '
                   'interpreting.') if flagged else
              (f"No contrast with more own lags than {reference} has an interval entirely below zero (no fitting flag). "
               f"A negative LL(k) - LL({reference}) for k < {reference} is not a flag: the smaller model drops lags.")]

    lag = cal['lag']
    lines += ['', f'## Calibration: planted lag-{lag} copies, condition {config["calibration"]["condition"]}', '',
              f"The residual LSTM's planted-signal protocol (`residual_planted.py`, reused, not copied): on all 14 folds a "
              f"fraction f in {planted_config['fractions']} of the eligible targets on training, validation and test days is "
              f"relabelled as the whale's own coda at lag {lag} (eligible: at least {lag} own codas, and the coda at lag "
              f"{lag} differs from lags 1-3); labels only, histories unchanged; nested subsets, one permutation per fold "
              f"and split (sampling seed {planted_config['sampling_seed']}). Logistic regression is refitted on the planted "
              f"training and validation days with v4's procedure at k = {k_cal} and at k = {reference}. Scored on the planted "
              f"test labels: (1) LR({k_cal}) - LR({reference}); (2) an oracle minus LR({reference}), the oracle predicting "
              f"f x 1[own coda at lag {lag}] + (1 - f) x P_LR({reference}) on eligible test targets and P_LR({reference}) "
              "elsewhere (it knows the rule and f, not which targets were planted); (3) the recovery ratio r = (1)/(2), a "
              "ratio of token-weighted means within each bootstrap replicate over the same resampled days "
              f"({planted_config['bootstrap_replicates']:,} replicates). Replicates whose oracle difference is at or "
              f"below zero have no ratio and are counted (column Non-positive); more than {planted_config['ratio_handful']} "
              'make r unstable at that f.', '',
              f"| f | Planted test targets (share) | (1) LR({k_cal}) - LR({reference}) | CI | Positive days | "
              f"(2) Oracle - LR({reference}) | CI | (3) r | r CI | Non-positive |",
              '|---:|---|---:|---|---:|---:|---|---:|---|---:|']
    for f, p in cal['per_fraction'].items():
        g, o, r = p['gain'], p['oracle_gain'], p['recovery']
        r_ci = ('unstable' if r['status'] == 'unstable' else 'n/a' if r['ci_ratio'] is None
                else f"[{_g(r['ci_ratio'][0])}, {_g(r['ci_ratio'][1])}]")
        lines.append(f"| {f} | {p['planted_test_targets']}/{p['test_targets']} ({p['planted_test_share']:.1%}) | "
                     f"{_f(g['mean_bits_per_coda'])} | {format_ci(g['ci_bits_per_coda'])} | {g['positive_days']}/{g['days']} | "
                     f"{_f(o['mean_bits_per_coda'])} | {format_ci(o['ci_bits_per_coda'])} | {_g(r['ratio'])} | {r_ci} | "
                     f"{r['degenerate_replicates']}/{r['replicates']} |")
    upper = cal['real_data_contrast_upper_bits']
    lines += ['', f"Bound. The real-data contrast LL({k_cal}) - LL({reference}) in {config['calibration']['condition']} has a "
              f"day-bootstrap upper bound of {upper:.4f} bits/coda (contrast table above). Where r's lower bound is above "
              "zero and the oracle difference stays positive in every replicate, that upper bound divided by r's lower "
              f"bound is the oracle gain above which a pattern of this form (a deterministic copy of the own coda at lag "
              f"{lag}) would have pushed the real-data contrast past its observed upper bound. With an upper bound at or "
              'below zero there is nothing to divide and no bound is stated.', '']
    for f, p in cal['per_fraction'].items():
        r = p['recovery']
        ci, degenerate, n = r['ci_ratio'], r['degenerate_replicates'], r['replicates']
        if r['status'] == 'unstable':
            lines.append(f"- f = {f}: the ratio is unstable at this f ({degenerate} of {n} resamples have an oracle "
                         "difference at or below zero), so no bound is stated.")
        elif degenerate:
            lines.append(f"- f = {f}: {degenerate} of {n} resamples have an oracle difference at or below zero, so the denominator "
                         "does not stay positive and no bound is stated.")
        elif p['bound_bits_per_coda'] is not None:
            lines.append(f"- f = {f}: r >= {_g(ci[0])}, so a lag-{lag} copy pattern worth more than "
                         f"{p['bound_bits_per_coda']:.4f} bits/coda to the oracle would have been detected (here the oracle "
                         f"gains {p['oracle_gain']['mean_bits_per_coda']:.4f}).")
        elif upper <= 0:
            lines.append(f"- f = {f}: r's lower bound is {_g(ci[0])}, but the real-data upper bound is not above zero, "
                         'so no bound is stated.')
        else:
            lines.append(f"- f = {f}: r's lower bound is {_g(ci[0])}, at or below zero, so the run gives no bound.")
    lines += ['', 'Planted test targets: mean log2-probability of the planted label.', '',
              f'| f | Targets | LR({reference}) | LR({k_cal}) | Oracle |', '|---:|---:|---:|---:|---:|']
    for f, p in cal['per_fraction'].items():
        q = p['planted']
        lines.append(f"| {f} | {q['targets']} | {_f(q[f'lr_bits_k{reference}'], 3)} | {_f(q[f'lr_bits_k{k_cal}'], 3)} | "
                     f"{_f(q['oracle_bits'], 3)} |")
    check = cal['planting_check']
    lines += ['', f"Planting check against the residual planted run (`{config['calibration']['residual_planted_scores']}`): "
              f"eligibility, planted flags and planted labels are identical on all {check['targets']:,} fold, fraction and "
              f"target combinations; the k = {reference} refit differs from that run's refitted logistic regression by at "
              f"most {check['max_abs_refit_bits_difference']:.3g} bits ({check['targets_differing_beyond_tolerance']} "
              'targets differ by more than 1e-5).']

    lines += ['', '## Stop rules (DEC-16), per k', '',
              'Stop and investigate if an A-condition is reliably below repeat-own-last in log-likelihood, or an '
              'add-one-input contrast has an interval entirely below zero.', '', '| k | Result |', '|---:|---|']
    for k in orders:
        reasons = result['stop_rules'][str(k)]
        lines.append(f"| {k} | {'; '.join(reasons) if reasons else 'no stop rule fires'} |")

    lines += ['', '## Reading (DEC-25)', ''] + result['reading']
    lines += ['', '## Limits', '',
              f"- The extra lags share one history penalty with lags 1-{reference}, the partner block and the copy weights "
              "(v4's rule, kept by DEC-25), so a null may partly reflect that compromise.",
              '- Day-bootstrap intervals are approximate with only 14 days. The switch part rests on about 270 non-repeat '
              'codas and is exploratory.',
              f"- The calibration covers one pattern form (a copy of the own coda at lag {lag}) in condition "
              f"{config['calibration']['condition']} only; it says nothing about structure of other forms.",
              '- Claims are about dyadic exchanges only; this sweep says nothing about grammar or syntax.', '',
              f"Per-k outputs are in `{config['output_directory']}/k<k>/` (summary.json with the stop rules and input "
              'SHA-256, l2_selection.jsonl, scored.jsonl for the swept conditions), the calibration in '
              f"`{config['output_directory']}/calibration/`; `{config['output_directory']}/summary.json` records SHA-256 of "
              'every input to this report.', '']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='config/rq1_history_length.json')
    parser.add_argument('--stage', choices=('all', 'fit', 'calibrate', 'report'), default='all')
    parser.add_argument('--orders', type=int, nargs='+', help='fit stage: the k to fit now (default: every configured k)')
    # Each process imports torch through residual.py (about 230 MB before any data), so the default stays small.
    parser.add_argument('--workers', type=int, default=4, help='processes for the fit and calibrate stages (default 4)')
    args = parser.parse_args()
    if args.stage in ('all', 'fit'):
        run_fit(args.config, args.orders, args.workers)
    if args.stage in ('all', 'calibrate'):
        run_calibration(args.config, args.workers)
    if args.stage in ('all', 'report'):
        result = run_report(args.config)
        print('\n'.join(result['reading']))


if __name__ == '__main__':
    main()
