"""Switch structure, PLAN [P5] (DEC-28): logistic v4 plus shared class-feature weights from the whale's own
earlier codas in the exchange.

Logistic v4 unchanged (labels `guy_coda_type`, targets, folds, UNK policy, S, penalties, grid, selection rule,
copy weights) plus, in conditions with A only, one shared scalar per feature, added to the logits of the classes
the feature marks and penalized with the history penalty like the copy weights (logistic.py's class-feature
weights):
- run_length: log r on the class of the own last coda, r = the identical own codas ending the stream;
- distinct_1, distinct_2: 1 on d1, the most recent own type different from the current one, and on d2, the
  most recent own type different from both;
- repertoire: log(1 + n_c) on every class c, n_c = the whale's earlier codas of type c in the exchange,
  excluding the current run.
The stream is the whale's codas in the exchange with an earlier onset, all of them (not only the 10 in
A_history), with labels mapped by the fold's vocabulary exactly as A_history is (UNK included). A row without
the needed history gets zeros.

Variants: R (v4), L (R + run length), P (L + repertoire), D (L + distinct order), PD (L + repertoire + distinct
order). Every variant fits all eight conditions with logistic.fit_v2_lattice, in one environment; the four
conditions without A must come out identical in every variant. R is compared with the committed v4 run (the
gate, DEC-25's amended rule) and is the reference for every contrast.

Three stages (--stage): describe (the descriptive table from the stored labels, no models, checked against the
expected values in the config), fit (one process per variant), report (reads only the other two stages'
outputs and regenerates byte-identically without refitting). Every fitting process runs with one BLAS thread.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict

import numpy as np

from .history_length import (dec23_checks, environment, order_contrast, penalty_summary, process_pool,
                             reference_gate, require_current, require_one_blas_thread)
from .logistic import check_stop_conditions, fit_v2_lattice, format_ci
from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text
from .residual import _hashes, _load

FEATURES = ('run_length', 'distinct_1', 'distinct_2', 'repertoire')
SYMBOLS = {'copy_beta': 'β (copy)', 'run_length': 'β_r', 'distinct_1': 'δ1', 'distinct_2': 'δ2', 'repertoire': 'ρ'}
TURN_CATEGORIES = ('self_continuation', 'switch', 'overlap')
RUN_BINS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10))
COMPONENTS = ('log_likelihood_bits', 'repeat_bits', 'switch_choice_bits')


# ---------------------------------------------------------------- the whale's stream and its features

def own_streams(rows, label='Ct', coda_ids=None):
    """coda_id -> the `label`s of the whale's codas in the same exchange with an earlier onset, oldest first, all
    of them (DEC-28), for every row or only `coda_ids`. Each earlier own coda must have ended by the row's onset,
    as every A_history coda has: a same-whale overlap would put a coda whose type is not yet known at the onset
    into the stream, so it raises, as do tied onsets."""
    groups = defaultdict(list)
    for r in rows:
        groups[r['exchange_id'], r['whale_id']].append(r)
    wanted = None if coda_ids is None else set(coda_ids)
    streams = {}
    for group in groups.values():
        group = sorted(group, key=lambda r: r['onset_s'])
        for i, r in enumerate(group):
            if wanted is not None and r['coda_id'] not in wanted:
                continue
            if any(e['onset_s'] == r['onset_s'] for e in group[:i] + group[i + 1:]):
                raise ValueError(f"{r['coda_id']}: tied onsets in one whale's stream")
            if any(e['offset_s'] > r['onset_s'] for e in group[:i]):
                raise ValueError(f"{r['coda_id']}: an earlier own coda is still sounding at its onset")
            streams[r['coda_id']] = [e[label] for e in group[:i]]
    if wanted is not None and set(streams) != wanted:
        raise ValueError('Some requested codas are not in the rows')
    return streams


def stream_structure(stream):
    """The collapsed view of one stream: its last label, the run r of identical labels ending it, d1 (the most
    recent label before that run) and d2 (the most recent label before the run that is neither the last label
    nor d1), and the label counts before the run. None for an empty stream."""
    if not stream:
        return None
    last, run = stream[-1], 1
    while run < len(stream) and stream[-run - 1] == last:
        run += 1
    before = stream[:-run]
    d1 = before[-1] if before else None
    d2 = next((label for label in reversed(before) if label not in (last, d1)), None)
    return dict(last=last, run=run, d1=d1, d2=d2, before=Counter(before))


def class_values(structure, name):
    """One row's value of feature `name` as {class label: value}; every class not listed, and every class of a
    row without a stream, gets zero."""
    if name not in FEATURES:
        raise ValueError(f'Unknown class feature {name!r}')
    if structure is None:
        return {}
    if name == 'run_length':
        return {structure['last']: math.log(structure['run'])}
    if name == 'repertoire':
        return {label: math.log1p(n) for label, n in structure['before'].items()}
    distinct = structure['d1' if name == 'distinct_1' else 'd2']
    return {} if distinct is None else {distinct: 1.0}


def feature_matrices(rows, streams, vocabulary, names):
    """One rows x classes matrix per feature name, in `vocabulary`'s column order."""
    position = {label: j for j, label in enumerate(vocabulary)}
    matrices = [np.zeros((len(rows), len(vocabulary))) for _ in names]
    for i, r in enumerate(rows):
        structure = stream_structure(streams[r['coda_id']])
        for matrix, name in zip(matrices, names):
            for label, value in class_values(structure, name).items():
                matrix[i, position[label]] = value
    return matrices


def check_streams(rows, streams, history_limit):
    """Every row's stream ends with its A_history, which holds its last `history_limit` codas."""
    for r in rows:
        stream = streams[r['coda_id']]
        if stream[len(stream) - min(len(stream), history_limit):] != r['A_history']:
            raise ValueError(f"{r['coda_id']}: the whale's stream does not end with its A_history")


def variant_class_features(names, history_limit):
    """logistic.fit_v2_lattice's class_features hook for one variant. None without features, so R runs v4's
    code path; per fold and condition, None for conditions without A. The streams use the fold's mapped rows
    (every split and target), so their labels are mapped exactly as A_history's."""
    if not names:
        return None
    if any(name not in FEATURES for name in names) or len(set(names)) != len(names):
        raise ValueError(f'Class features must be distinct names from {FEATURES}')

    def hook(mapped, policy, fields):
        if 'A_history' not in fields:
            return None
        streams = own_streams(mapped)
        check_streams(mapped, streams, history_limit)
        vocabulary = policy['vocabulary']
        return list(names), lambda rows: feature_matrices(rows, streams, vocabulary, names)
    return hook


# ---------------------------------------------------------------- shared

def load(config_path):
    path, root, config, manifest, manifest_path, source = _load(config_path)
    paths = {k: root / config[k] for k in ('logistic_config', 'evaluation_config')}
    loaded = dict(path=path, root=root, config=config, manifest=manifest, manifest_path=manifest_path, source=source,
                  **{k + '_path': p for k, p in paths.items()}, **{k: read_json(p) for k, p in paths.items()})
    check_config(config, manifest, loaded['logistic_config'])
    return loaded


def check_config(config, manifest, logistic):
    variants, reference = config['variants'], config['reference_variant']
    if reference not in variants or variants[reference]:
        raise ValueError('The reference variant must be one of the variants and have no class features')
    if 'history_orders' in logistic:
        raise ValueError("The variants keep logistic v4's lags")
    for names in variants.values():
        if any(name not in FEATURES for name in names) or len(set(names)) != len(names):
            raise ValueError(f'Class features must be distinct names from {FEATURES}')
    pairs = [(c['full'], c['reduced']) for c in (config['primary'], config['secondary'], config['repertoire_contrast'])]
    pairs += [tuple(p) for p in config['descriptive_contrasts']]
    if any(v not in variants for pair in pairs for v in pair):
        raise ValueError('Every contrast must compare configured variants')
    if any(not set(variants[small]) < set(variants[big]) for big, small in pairs):
        raise ValueError('Every contrast must add class features to the reduced variant')
    conditions = manifest['input_conditions']
    named = [config['primary']['condition'], config['secondary']['condition'], config['repertoire_contrast']['condition']]
    if any(c not in config['conditions'] for c in named) or any(
            c not in conditions or 'A_history' not in conditions[c] for c in config['conditions']):
        raise ValueError('Contrast conditions must be manifest conditions with A')


def descriptive_directory(loaded):
    return loaded['root'] / loaded['config']['output_directory'] / 'descriptive'


def variant_directory(loaded, variant):
    return loaded['root'] / loaded['config']['output_directory'] / variant


def contrast_pairs(config):
    """(full, reduced) in report order: primary, secondary, then the descriptive ones."""
    pairs = [(config['primary']['full'], config['primary']['reduced']),
             (config['secondary']['full'], config['secondary']['reduced'])]
    return list(dict.fromkeys(pairs + [tuple(p) for p in config['descriptive_contrasts']]))


# ---------------------------------------------------------------- describe stage (no models)

def ratio_interval(items, days, settings):
    """Ratio of sums over (day, numerator, denominator) items, and its day bootstrap: each replicate draws all
    the days with replacement (the draws of evaluation.day_bootstrap_bits) and takes the ratio of sums over the
    drawn days. Replicates whose drawn days hold no denominator have no ratio and are counted."""
    numerators, denominators = defaultdict(list), defaultdict(list)
    for day, numerator, denominator in items:
        numerators[day].append(float(numerator))
        denominators[day].append(float(denominator))
    numerator = {d: math.fsum(numerators[d]) for d in days}
    denominator = {d: math.fsum(denominators[d]) for d in days}
    rng = random.Random(settings['bootstrap_seed'])
    samples, empty = [], 0
    for _ in range(settings['bootstrap_replicates']):
        drawn = rng.choices(days, k=len(days))
        total = math.fsum(denominator[d] for d in drawn)
        if total == 0:
            empty += 1
            continue
        samples.append(math.fsum(numerator[d] for d in drawn) / total)
    samples.sort()

    def quantile(q):
        position = q * (len(samples) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return samples[lo] + (position - lo) * (samples[hi] - samples[lo])
    tail = (1 - settings['confidence']) / 2
    return dict(value=math.fsum(numerator.values()) / math.fsum(denominator.values()),
                ci=[quantile(tail), quantile(1 - tail)], replicates_without_denominator=empty)


def describe(rows, settings, history_limit):
    """DEC-28's descriptive table on the comparison targets, labels as stored in Ct (not fold-mapped): own last
    a = A_history[-1], partner last b = B_history[-1], the whale's whole earlier stream in the exchange (run r,
    d1, d2 as in stream_structure). Day-level chances use that day's comparison targets. Intervals: day bootstrap,
    ratio of sums within each replicate (ratio_interval)."""
    streams = own_streams(rows)
    check_streams(rows, streams, history_limit)
    targets = [r for r in rows if r['comparison_eligible']]
    days = sorted({r['recording_date'] for r in targets})
    n_day, own_day, partner_day, target_day = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for r in targets:
        day = r['recording_date']
        n_day[day] += 1
        own_day[day][r['A_history'][-1]] += 1
        partner_day[day][r['B_history'][-1]] += 1
        target_day[day][r['Ct']] += 1
    agreement = {d: math.fsum(own_day[d][k] * partner_day[d][k] for k in own_day[d]) / n_day[d] ** 2 for d in days}

    def switch_chance(r, label):
        p = lambda k: target_day[r['recording_date']][k] / n_day[r['recording_date']]  # noqa: E731
        return p(label) / (1 - p(r['A_history'][-1]))

    def interval(items):
        return ratio_interval(items, days, settings)

    def rate(subset, predicate):
        out = interval([(r['recording_date'], predicate(r), 1) for r in subset])
        return dict(out, count=sum(bool(predicate(r)) for r in subset), n=len(subset))

    def mean_of(subset, value):
        return dict(interval([(r['recording_date'], value(r), 1) for r in subset]), n=len(subset))

    own = lambda r: r['A_history'][-1]  # noqa: E731
    partner = lambda r: r['B_history'][-1]  # noqa: E731
    q = dict(targets=dict(count=len(targets)), repeats_of_own_last=dict(count=sum(r['Ct'] == own(r) for r in targets)))
    for category in TURN_CATEGORIES:
        q[f'repeat_rate:{category}'] = rate([r for r in targets if r['turn_category'] == category],
                                           lambda r: r['Ct'] == own(r))
    q['own_last_equals_partner_last'] = rate(targets, lambda r: own(r) == partner(r))
    q['agreement_chance'] = mean_of(targets, lambda r: agreement[r['recording_date']])
    differ = [r for r in targets if own(r) != partner(r)]
    q['own_last_differs:repeats_own_last'] = rate(differ, lambda r: r['Ct'] == own(r))
    q['own_last_differs:takes_partner_last'] = rate(differ, lambda r: r['Ct'] == partner(r))
    q['own_last_differs:other'] = rate(differ, lambda r: r['Ct'] not in (own(r), partner(r)))
    switches = [r for r in targets if r['Ct'] != own(r)]
    switch_differ = [r for r in switches if own(r) != partner(r)]
    q['switch_own_last_differs:lands_on_partner_last'] = rate(switch_differ, lambda r: r['Ct'] == partner(r))
    q['switch_own_last_differs:chance'] = mean_of(switch_differ, lambda r: switch_chance(r, partner(r)))
    q['switch_own_last_differs:difference'] = mean_of(
        switch_differ, lambda r: (r['Ct'] == partner(r)) - switch_chance(r, partner(r)))
    q['switches'] = dict(count=len(switches))
    q['switches:own_last_differs_lands_on_partner_last'] = dict(count=sum(r['Ct'] == partner(r) for r in switch_differ))
    q['switches:own_last_equals_partner_last'] = dict(count=sum(own(r) == partner(r) for r in switches))
    q['switches:other'] = dict(count=sum(r['Ct'] != partner(r) for r in switch_differ))

    structure = {r['coda_id']: stream_structure(streams[r['coda_id']]) for r in targets}
    if any(structure[r['coda_id']]['last'] != own(r) for r in targets):
        raise ValueError("A stream's last coda differs from A_history's")
    d1 = lambda r: structure[r['coda_id']]['d1']  # noqa: E731
    d2 = lambda r: structure[r['coda_id']]['d2']  # noqa: E731
    with_d1 = [r for r in switches if d1(r) is not None]
    q['switches_with_d1'] = dict(count=len(with_d1))
    q['returns_to_d1'] = rate(with_d1, lambda r: r['Ct'] == d1(r))
    q['returns_to_d1:chance'] = mean_of(with_d1, lambda r: switch_chance(r, d1(r)))
    q['returns_to_d1:difference'] = mean_of(with_d1, lambda r: (r['Ct'] == d1(r)) - switch_chance(r, d1(r)))
    others = lambda r: len(set(structure[r['coda_id']]['before']) - {own(r)})  # noqa: E731
    for key, keep in (('1', lambda k: k == 1), ('2', lambda k: k == 2), ('3_or_more', lambda k: k >= 3)):
        subset = [r for r in with_d1 if keep(others(r))]
        chance = mean_of(subset, lambda r: switch_chance(r, d1(r)))
        q[f'returns_to_d1:other_types_{key}'] = dict(rate(subset, lambda r: r['Ct'] == d1(r)), chance=chance['value'],
                                                     chance_ci=chance['ci'])
    with_d2 = [r for r in with_d1 if d2(r) is not None]
    q['switches_with_d2'] = dict(count=len(with_d2))
    q['switches_with_d2:returns_to_d2'] = rate(with_d2, lambda r: r['Ct'] == d2(r))
    q['switches_with_d2:returns_to_d1'] = rate(with_d2, lambda r: r['Ct'] == d1(r))

    runs = [structure[r['coda_id']]['run'] for r in targets]
    q['run_length:at_least_3'] = dict(count=sum(r >= 3 for r in runs))
    q['run_length:at_least_10'] = dict(count=sum(r >= 10 for r in runs))
    q['run_length:max'] = dict(count=max(runs))
    history_run = {r['coda_id']: stream_structure(r['A_history'])['run'] for r in targets}
    for lo, hi in RUN_BINS:
        q[f'repeat_rate_by_run_in_A_history:{lo}' + ('' if lo == hi else f'-{hi}')] = rate(
            [r for r in targets if lo <= history_run[r['coda_id']] <= hi], lambda r: r['Ct'] == own(r))
    pairs = defaultdict(list)
    for r in targets:
        pairs[r['exchange_id'], r['whale_id']].append(r)
    differences = []
    for group in pairs.values():
        first = [r['Ct'] == own(r) for r in group if history_run[r['coda_id']] == 1]
        later = [r['Ct'] == own(r) for r in group if history_run[r['coda_id']] >= 2]
        if first and later:
            differences.append((group[0]['recording_date'], sum(later) / len(later) - sum(first) / len(first), 1))
    q['within_whale_exchange:run_2_or_more_minus_run_1'] = dict(interval(differences), n=len(differences))
    return q


def check_descriptive(quantities, expected, tolerance):
    """Each expected field against the computed one: counts and n exactly, other values at 3 decimals, interval
    bounds within `tolerance` (different random streams)."""
    checks = []
    for name, wanted in expected.items():
        got = quantities.get(name)
        if got is None:
            checks.append(dict(name=name, expected=wanted, computed=None, failed_fields=sorted(wanted), passed=False))
            continue
        failed = []
        for field, value in wanted.items():
            if field not in got:
                failed.append(field)
            elif field == 'ci':
                if any(abs(a - b) > tolerance + 1e-12 for a, b in zip(got['ci'], value)):
                    failed.append(field)
            elif field in ('count', 'n'):
                if got[field] != value:
                    failed.append(field)
            elif f'{got[field]:.3f}' != f'{value:.3f}':
                failed.append(field)
        checks.append(dict(name=name, expected=wanted, computed={f: got.get(f) for f in wanted},
                           failed_fields=failed, passed=not failed))
    return checks


def run_describe(config_path):
    loaded = load(config_path)
    settings = loaded['config']['descriptive']
    quantities = describe(read_jsonl(loaded['source']), settings,
                          loaded['manifest']['preparation_configuration']['history_limit'])
    checks = check_descriptive(quantities, settings['expected'], settings['interval_tolerance'])
    passed = all(c['passed'] for c in checks)
    summary = dict(settings={k: v for k, v in settings.items() if k != 'expected'}, labels='as stored in Ct',
                   quantities=quantities, checks=checks, passed=passed,
                   input_sha256=_hashes(loaded['root'], [loaded['source'], loaded['path']]))
    output = descriptive_directory(loaded)
    output.mkdir(parents=True, exist_ok=True)
    write_text(output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    failed = [c for c in checks if not c['passed']]
    print(f"Descriptive table: {len(checks) - len(failed)} of {len(checks)} expected values reproduced", flush=True)
    for c in failed:
        print(f"  DIFFERS {c['name']}: computed {c['computed']}, expected {c['expected']}", flush=True)
    if failed and settings['enforce']:
        raise ValueError(f'The descriptive table differs from the expected values ({output / "summary.json"}); stop here')
    return summary


# ---------------------------------------------------------------- fit stage

def _fit_variant(task):
    """One variant: every manifest condition, logistic v4 with that variant's class features in conditions with
    A. Module level so that spawned processes can import it."""
    config_path, variant = task
    require_one_blas_thread()
    start = time.perf_counter()
    loaded = load(config_path)
    config, manifest = loaded['config'], loaded['manifest']
    names = config['variants'][variant]
    fitted = fit_v2_lattice(loaded['logistic_config'], manifest, read_jsonl(loaded['source']), loaded['evaluation_config'],
                            variant_class_features(names, manifest['preparation_configuration']['history_limit']))
    summary = fitted['summary']
    summary.update(variant=variant, class_features=names, configuration=loaded['logistic_config'],
                   baseline_metrics=fitted['baseline_metrics'], repeat_baseline_checks=fitted['baseline_checks'],
                   stop_reasons=check_stop_conditions(summary, fitted['baseline_metrics'], fitted['baseline_checks']),
                   prediction_sha256={c: hashlib.sha256(json.dumps(
                       [p for p in fitted['predictions'] if p['condition'] == c], allow_nan=False).encode()).hexdigest()
                       for c in manifest['input_conditions']},
                   environment=environment())
    inputs = [loaded[k] for k in ('source', 'manifest_path', 'path', 'logistic_config_path', 'evaluation_config_path')]
    if variant == config['reference_variant']:
        summary['gate'], gate_inputs = reference_gate(loaded['root'], config['gate'], fitted)
        inputs += gate_inputs
    summary['input_sha256'] = _hashes(loaded['root'], inputs)
    coefficients = [dict(fold_id=m['fold_id'], condition=m['condition'], chosen_l2=m['chosen_l2'],
                         chosen_l2_s=m['chosen_l2_s'], beta=m['beta'], gamma=m['gamma'],
                         class_weights=dict(zip(m.get('class_features', []), m.get('class_weights', []))))
                    for m in fitted['models'] if 'A_history' in m['fields']]
    summary['fit_seconds'] = time.perf_counter() - start
    output = variant_directory(loaded, variant)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'l2_selection.jsonl', fitted['selections'])
    write_jsonl(output / 'scored.jsonl', [r for r in fitted['scored'] if r['condition'] in config['conditions']])
    write_jsonl(output / 'coefficients.jsonl', coefficients)
    write_text(output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return dict(variant=variant, seconds=summary['fit_seconds'], stop_reasons=summary['stop_reasons'],
                gate=summary.get('gate'))


def run_fit(config_path, variants=None, workers=None):
    require_one_blas_thread()
    loaded = load(config_path)
    variants = variants or list(loaded['config']['variants'])
    if set(variants) - set(loaded['config']['variants']):
        raise ValueError('Every variant to fit must be one of the configured variants')
    results = []
    with process_pool(min(workers or os.cpu_count(), len(variants))) as pool:
        futures = [pool.submit(_fit_variant, (str(loaded['path']), v)) for v in variants]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            gate = result['gate']
            gate_text = '' if gate is None else (
                f"; gate {'passed' if gate['passed'] else 'FAILED'} ({len(gate['penalty_differences'])} of "
                f"{gate['fits_compared']} penalties differ, mean LL within {gate['max_abs_log_likelihood_difference_bits']:.2g} "
                f"bits/coda; max abs probability difference {gate['max_abs_probability_difference']:.3g}, reported only)")
            print(f"{result['variant']}: {result['seconds']:.0f} s, {len(result['stop_reasons'])} DEC-16 stop rule(s) "
                  f"fired{gate_text}", flush=True)
            results.append(result)
    return results


# ---------------------------------------------------------------- report stage

def coefficient_summary(coefficients, names, condition):
    """Per weight (the copy weight beta and each class feature): min, median and max over folds, and per fold."""
    rows = sorted((r for r in coefficients if r['condition'] == condition), key=lambda r: r['fold_id'])
    out = {}
    for name in ['copy_beta'] + list(names):
        per_fold = {r['fold_id']: r['beta'] if name == 'copy_beta' else r['class_weights'][name] for r in rows}
        values = list(per_fold.values())
        out[name] = dict(min=min(values), median=statistics.median(values), max=max(values), per_fold=per_fold)
    return out


def contrast_role(config, condition, full, reduced):
    for role in ('primary', 'secondary'):
        spec = config[role]
        if (spec['condition'], spec['full'], spec['reduced']) == (condition, full, reduced):
            return role
    return 'descriptive'


def run_report(config_path):
    loaded = load(config_path)
    root, config, manifest = loaded['root'], loaded['config'], loaded['manifest']
    evaluation = loaded['evaluation_config']
    boot = (evaluation['bootstrap_replicates'], evaluation['bootstrap_seed'], evaluation['confidence'])
    inputs = [loaded[k] for k in ('source', 'manifest_path', 'path', 'logistic_config_path', 'evaluation_config_path')]
    descriptive_path = descriptive_directory(loaded) / 'summary.json'
    if not descriptive_path.is_file():
        raise ValueError('The descriptive table has not been computed (run --stage describe)')
    require_current(loaded, descriptive_path)
    descriptive = read_json(descriptive_path)
    if config['descriptive']['enforce'] and not descriptive['passed']:
        raise ValueError('The descriptive table does not reproduce the expected values; stop here')
    inputs.append(descriptive_path)

    variants, reference = list(config['variants']), config['reference_variant']
    summaries, selections, scored, coefficients = {}, {}, {}, {}
    for v in variants:
        directory = variant_directory(loaded, v)
        paths = [directory / name for name in ('summary.json', 'l2_selection.jsonl', 'scored.jsonl', 'coefficients.jsonl')]
        if not all(p.is_file() for p in paths):
            raise ValueError(f'Variant {v} has not been fitted (run --stage fit)')
        require_current(loaded, paths[0])
        summaries[v], selections[v], scored[v], coefficients[v] = (read_json(paths[0]), *map(read_jsonl, paths[1:]))
        if summaries[v]['configuration'] != loaded['logistic_config'] or summaries[v]['class_features'] != config['variants'][v]:
            raise ValueError(f'Variant {v} was fitted with another configuration')
        inputs += paths
    gate = summaries[reference]['gate']
    if config['gate']['enforce'] and not gate['passed']:
        raise ValueError(f'The reference fit does not reproduce logistic v4 (see {reference}/summary.json, gate); stop here')
    # Conditions without A carry no class features, so every variant must give them exactly R's predictions.
    constant = [c for c, fields in manifest['input_conditions'].items() if 'A_history' not in fields]
    constant_selection = {v: [s for s in selections[v] if s['condition'] in constant] for v in variants}
    for v in variants:
        for c in constant:
            if (summaries[v]['prediction_sha256'][c] != summaries[reference]['prediction_sha256'][c]
                    or summaries[v]['metrics'][c] != summaries[reference]['metrics'][c]):
                raise ValueError(f'Condition {c} has no class features but differs between variants {v} and {reference}')
        if constant_selection[v] != constant_selection[reference]:
            raise ValueError(f'The penalty search of a condition without A differs between variants {v} and {reference}')

    grid = loaded['logistic_config']['l2_grid']
    conditions = config['conditions']
    metrics = {v: {c: dict(mean_log_likelihood_bits=summaries[v]['metrics'][c]['mean_log_likelihood_bits'],
                           accuracy=summaries[v]['metrics'][c]['accuracy'], macro_f1=summaries[v]['metrics'][c]['macro_f1'],
                           versus_repeat_own_last=summaries[v]['repeat_baseline_checks'][c],
                           history_l2=penalty_summary(selections[v], c, grid))
                   for c in conditions} for v in variants}
    weights = {v: {c: coefficient_summary(coefficients[v], config['variants'][v], c) for c in conditions}
               for v in variants}
    ordered = [config['primary']['condition']] + [c for c in conditions if c != config['primary']['condition']]
    contrasts = {}
    for c in ordered:
        contrasts[c] = {}
        for full, reduced in contrast_pairs(config):
            role = contrast_role(config, c, full, reduced)
            components = order_contrast(scored[full], scored[reduced], c, boot, sign_flip=role != 'descriptive')
            contrasts[c][f'{full}-{reduced}'] = dict(
                full=full, reduced=reduced, role=role, components=components,
                fitting_flag=components['log_likelihood_bits']['ci_bits_per_coda'][1] < 0)
    stop_rules = {v: summaries[v]['stop_reasons'] for v in variants}
    day_counts = {fold['test_day']: len(fold['comparison_ids']['test']) for fold in manifest['folds']}
    stop_rules_dec23 = {v: dec23_checks(summaries[v], day_counts) for v in variants}

    result = dict(units='bits per coda', variants=config['variants'], reference_variant=reference,
                  conditions=conditions, primary=config['primary'], secondary=config['secondary'],
                  repertoire_contrast=config['repertoire_contrast'],
                  comparison_targets=summaries[reference]['comparison_targets'],
                  repeat_own_last_bits=summaries[reference]['baseline_metrics']['repeat_own_last']['mean_log_likelihood_bits'],
                  bootstrap=dict(zip(('replicates', 'seed', 'confidence'), boot)),
                  descriptive={k: descriptive[k] for k in ('settings', 'labels', 'quantities', 'checks', 'passed')},
                  gate={k: v for k, v in gate.items() if k != 'pairs'}, constant_conditions=constant,
                  metrics=metrics, coefficients=weights, contrasts=contrasts, stop_rules=stop_rules,
                  stop_rules_dec23=stop_rules_dec23,
                  environment={v: summaries[v]['environment'] for v in variants},
                  fit_seconds={v: summaries[v]['fit_seconds'] for v in variants},
                  fits=sum(len(s['grid']) for v in variants for s in selections[v]))
    result['reading'] = reading_lines(result)
    result['input_sha256'] = _hashes(root, inputs)
    output = root / config['output_directory']
    write_text(output / 'summary.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(report_lines(result, config)))
    return result


# ---------------------------------------------------------------- reading and report

def _stat(stats):
    return (f"{stats['mean_bits_per_coda']:.5f} | {stats['unweighted_mean_bits_per_coda']:.5f} | "
            f"{format_ci(stats['ci_bits_per_coda'])} | {stats['positive_days']}/{stats['days']}")


def _part(stats):
    return f"{stats['mean_bits_per_coda']:.5f} {format_ci(stats['ci_bits_per_coda'])}"


def _contrast_text(contrast, condition):
    s = contrast['components']['log_likelihood_bits']
    return (f"LL({contrast['full']}) - LL({contrast['reduced']}) in {condition}: {s['mean_bits_per_coda']:.5f} bits/coda "
            f"(unweighted day mean {s['unweighted_mean_bits_per_coda']:.5f}, 95% CI {format_ci(s['ci_bits_per_coda'])}, "
            f"{s['positive_days']}/{s['days']} days positive"
            + (f", two-sided sign-flip p = {s['sign_flip_p_two_sided']:.4f}" if 'sign_flip_p_two_sided' in s else '')
            + f"); repeat part {_part(contrast['components']['repeat_bits'])}, switch part "
            f"{_part(contrast['components']['switch_choice_bits'])}")


def _lookup(result, spec):
    return result['contrasts'][spec['condition']][f"{spec['full']}-{spec['reduced']}"]


def reading_lines(result):
    """DEC-28's Reading, chosen mechanically by the primary contrast's interval and applied to the primary only;
    its repertoire bullet applies only when the primary is null. Withheld when the gate fails or DEC-23's stop
    rule fires on a variant the reading rests on."""
    primary, secondary, repertoire = result['primary'], result['secondary'], result['repertoire_contrast']
    contrast = _lookup(result, primary)
    low, high = contrast['components']['log_likelihood_bits']['ci_bits_per_coda']
    lines = [f"Primary, pre-specified: {_contrast_text(contrast, primary['condition'])}."]
    fired = [v for v, rule in result['stop_rules_dec23'].items() if rule['fired']]
    blocked = [v for v in fired if v in (result['reference_variant'], primary['full'], primary['reduced'])]
    discriminating = result['descriptive']['quantities']['switches_with_d2']['count']
    if not result['gate']['passed']:
        lines.append('No reading: the reference variant does not reproduce logistic v4 (gate above).')
    elif blocked:
        lines.append(f"No reading: DEC-23's stop rule fires on variant{'s' if len(blocked) > 1 else ''} "
                     f"{', '.join(blocked)} (stop rules above).")
    elif high < 0:
        lines.append('No reading: the interval lies entirely below zero, so adding the distinct order reliably lowered '
                     'held-out log-likelihood. That is a fitting flag (optimisation or regularisation), not a finding.')
    elif low > 0:
        parts = [name for key, name in (('switch_choice_bits', 'switch'), ('repeat_bits', 'repeat'))
                 if contrast['components'][key]['ci_bits_per_coda'][0] > 0]
        where = (f"It lives in the {' and the '.join(parts)} part{'s' if len(parts) > 1 else ''} (interval above zero)."
                 if parts else 'Neither the switch nor the repeat part alone has an interval above zero.')
        lines.append('The order of the whale\'s earlier switches carries information beyond its active repertoire (for '
                     f'example A→B→A alternation). Not grammar. {where}')
    else:
        lines.append(f"Not detectable at this data size. The primary rests on few discriminating switches (only "
                     f"{discriminating} switches have two or more other own types to choose between), so its null is "
                     'weak; no calibration is run. Limited data and capacity push contrasts toward zero (PLAN §9).')
        rep = _lookup(result, repertoire)
        rep_low = rep['components']['log_likelihood_bits']['ci_bits_per_coda'][0]
        rep_blocked = [v for v in fired if v in (repertoire['full'], repertoire['reduced'])]
        if rep_low > 0 and not rep_blocked:
            lines.append(f"Repertoire ({repertoire['full']} - {repertoire['reduced']} in {repertoire['condition']}) "
                         f"positive, CI {format_ci(rep['components']['log_likelihood_bits']['ci_bits_per_coda'])}, with the "
                         "primary null: switch choices follow the whale's active repertoire, consistent with stable "
                         'individual or unit repertoires (Gero, Whitehead & Rendell 2016b).')
        elif rep_low > 0:
            lines.append(f"The repertoire contrast's interval is above zero, but DEC-23's stop rule fires on "
                         f"{', '.join(rep_blocked)}, so DEC-28's repertoire reading is not applied.")
        else:
            lines.append(f"The repertoire contrast ({repertoire['full']} - {repertoire['reduced']} in "
                         f"{repertoire['condition']}) is not positive (interval "
                         f"{format_ci(rep['components']['log_likelihood_bits']['ci_bits_per_coda'])}), so DEC-28's "
                         'repertoire reading does not apply.')
    lines.append(f"Secondary, pre-specified (numbers only; no reading): "
                 f"{_contrast_text(_lookup(result, secondary), secondary['condition'])}.")
    return lines


QUANTITY_LABELS = {
    'targets': 'Comparison targets',
    'repeats_of_own_last': 'Targets that repeat the own last coda (a)',
    'repeat_rate:self_continuation': 'Repeat rate after an own coda (self_continuation)',
    'repeat_rate:switch': 'Repeat rate right after a partner coda (switch)',
    'repeat_rate:overlap': 'Repeat rate overlapping a partner coda (overlap)',
    'own_last_equals_partner_last': 'Own last equals partner last (a = b)',
    'agreement_chance': 'a = b expected from day-level frequencies (sum over types of p(own last) p(partner last))',
    'own_last_differs:repeats_own_last': 'a ≠ b: target = a',
    'own_last_differs:takes_partner_last': 'a ≠ b: target = b',
    'own_last_differs:other': 'a ≠ b: target is neither',
    'switch_own_last_differs:lands_on_partner_last': 'Switches with a ≠ b: lands on b',
    'switch_own_last_differs:chance': 'Switches with a ≠ b: day-level chance p(b) / (1 − p(a))',
    'switch_own_last_differs:difference': 'Switches with a ≠ b: lands on b minus chance',
    'switches': 'Switches (target ≠ a)',
    'switches:own_last_differs_lands_on_partner_last': 'Switches: a ≠ b and target = b',
    'switches:own_last_equals_partner_last': 'Switches: a = b',
    'switches:other': 'Switches: a ≠ b and target ≠ b',
    'switches_with_d1': 'Switches with a d1 (an earlier different own coda)',
    'returns_to_d1': 'Switches with a d1: returns to d1',
    'returns_to_d1:chance': 'Switches with a d1: day-level chance p(d1) / (1 − p(a))',
    'returns_to_d1:difference': 'Switches with a d1: returns to d1 minus chance',
    'returns_to_d1:other_types_1': 'Returns to d1, one other type before the run',
    'returns_to_d1:other_types_2': 'Returns to d1, two other types before the run',
    'returns_to_d1:other_types_3_or_more': 'Returns to d1, three or more other types before the run',
    'switches_with_d2': 'Switches with a d2 (two or more other own types)',
    'switches_with_d2:returns_to_d2': 'Switches with a d2: returns to d2',
    'switches_with_d2:returns_to_d1': 'Switches with a d2: returns to d1',
    'run_length:at_least_3': 'Targets with run length r ≥ 3 (whole stream)',
    'run_length:at_least_10': 'Targets with r ≥ 10 (whole stream)',
    'run_length:max': 'Largest r (whole stream)',
    'repeat_rate_by_run_in_A_history:1': 'Repeat rate, r = 1 within A_history (10 codas)',
    'repeat_rate_by_run_in_A_history:2': 'Repeat rate, r = 2 within A_history',
    'repeat_rate_by_run_in_A_history:3': 'Repeat rate, r = 3 within A_history',
    'repeat_rate_by_run_in_A_history:4-5': 'Repeat rate, r = 4–5 within A_history',
    'repeat_rate_by_run_in_A_history:6-10': 'Repeat rate, r = 6–10 within A_history',
    'within_whale_exchange:run_2_or_more_minus_run_1': 'Within whale and exchange: repeat rate at r ≥ 2 minus at r = 1 '
                                                       '(r within A_history), mean over whale-exchanges with both',
}


def _quantity(q, fields=None):
    fields = fields or q
    parts = []
    if 'value' in fields:
        parts.append(f"{q['value']:.3f}" + (f" [{q['ci'][0]:.3f}, {q['ci'][1]:.3f}]" if 'ci' in fields else ''))
    if 'chance' in fields:
        parts.append(f"chance {q['chance']:.3f}")
    if 'count' in fields:
        parts.append(f"count {q['count']:,}")
    if 'n' in fields:
        parts.append(f"n {q['n']:,}")
    return '; '.join(parts)


def _minmax(stats):
    return f"{stats['min']:.3f} / {stats['median']:.3f} / {stats['max']:.3f}"


def report_lines(result, config):
    reference, variants, conditions = result['reference_variant'], result['variants'], result['conditions']
    gate, descriptive = result['gate'], result['descriptive']
    env = result['environment'][reference]
    settings = descriptive['settings']
    variant_text = '; '.join(f"{v}: {', '.join(names) if names else 'none (logistic v4)'}" for v, names in variants.items())
    lines = ['# RQ1 switch structure (PLAN [P5], DEC-28)', '']
    if config.get('report_note'):
        lines += [config['report_note'], '']
    lines += ['Logistic v4 unchanged (labels `guy_coda_type`, targets, folds, UNK policy, S, the L2 grid and selection rule, '
             'the joint history and S penalties, copy weights) plus shared class-feature weights computed from the '
             "whale's own earlier codas in the exchange. Each is one scalar, penalized with the history penalty like the "
             'copy weights, and added to the logits of the classes it marks: run length (β_r log r on the class of the own '
             'last coda, r = the identical own codas ending the history), distinct order (δ1 on d1, the most recent own '
             'type different from the current one; δ2 on d2, the most recent own type different from both) and repertoire '
             '(ρ log(1 + n_c) on every class c, n_c = the whale\'s earlier codas of type c in the exchange, excluding the '
             "current run). The features use the whale's whole earlier stream in the exchange (its codas with an earlier "
             'onset, not only the 10 in A_history), with labels mapped by the fold\'s vocabulary exactly as A_history is '
             '(UNK included); they exist only in conditions with A, and a row without the needed history gets zeros.', '',
             f"Variants (class features): {variant_text}. Every variant fits all eight conditions in one environment "
             f"(Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}, one BLAS thread per fit): "
             f"{result['fits']:,} fits, all converged (fit() raises otherwise). The four conditions without A "
             f"({', '.join(result['constant_conditions'])}) carry no class features; in every variant their penalty "
             f"searches and test predictions are identical to {reference}'s (checked exactly). {reference} is the "
             'reference for every contrast.', '',
             f"Contrasts are paired per coda between two variants' fits of one condition, with the [P1] repeat and switch "
             f"parts; intervals are day-bootstrap ({result['bootstrap']['replicates']:,} replicates, seed "
             f"{result['bootstrap']['seed']}); the unweighted day mean is the mean of the 14 day-level contrasts.", '']

    lines += ['## Descriptive table (no models)', '',
              f"Comparison targets of `rq1_targets.jsonl`, labels as stored in Ct (not fold-mapped). Own last a = "
              "A_history[-1], partner last b = B_history[-1]. The whale's stream: its codas in the same exchange with an "
              "earlier onset, all of them; r = the identical codas ending it; before = the stream without that run; d1 = "
              "before[-1]; d2 = the most recent label in before that is neither a nor d1. A switch is a target that differs "
              "from a. Day-level chances use that day's comparison targets: p(b) / (1 − p(a)) for landing on b, p(d1) / "
              "(1 − p(a)) for returning to d1, where p is the day's share of targets of that type. Intervals: day bootstrap, "
              f"{settings['bootstrap_replicates']:,} replicates of the 14 days (seed {settings['bootstrap_seed']}), ratio of "
              'sums within each replicate. Run length is computed within A_history only in the two rows that say so.', '',
              f"Check against the expected values ({settings['source']}): counts and point estimates must match at 3 "
              f"decimals, interval bounds within ±{settings['interval_tolerance']} (different random streams); "
              f"{sum(c['passed'] for c in descriptive['checks'])} of {len(descriptive['checks'])} agree"
              + ('.' if descriptive['passed'] else '; the rows marked DIFFERS do not.'), '',
              '| Quantity | Computed | Expected | Check |', '|---|---|---|---|']
    checks = {c['name']: c for c in descriptive['checks']}
    for name, q in descriptive['quantities'].items():
        check = checks.get(name)
        expected = _quantity(check['expected']) if check else 'n/a'
        status = 'n/a' if check is None else 'agrees' if check['passed'] else f"DIFFERS ({', '.join(check['failed_fields'])})"
        lines.append(f"| {QUANTITY_LABELS.get(name, name)} | {_quantity(q)} | {expected} | {status} |")

    lines += ['', f'## Gate: {reference} here against the committed logistic v4 run', '',
              f"Chosen penalties agree on {gate['fits_compared'] - len(gate['penalty_differences'])} of "
              f"{gate['fits_compared']} fold and condition fits, and every condition's mean log-likelihood is within "
              f"{gate['max_abs_log_likelihood_difference_bits']:.2g} bits/coda of v4's. DEC-25's amended gate requires at "
              f"most {gate['maximum_penalty_differences']} differing penalty and mean log-likelihoods within "
              f"{gate['log_likelihood_tolerance_bits']:g} bits/coda: {'passed' if gate['passed'] else 'FAILED'}. Predicted "
              'probabilities are reported, not enforced (optimizer tolerance and BLAS threads move them by up to about '
              f"1e-4 at the smallest penalties): largest absolute difference {gate['max_abs_probability_difference']:.3g} "
              f"over all fits ({gate['max_abs_probability_difference_same_penalty']:.3g} where penalties agree); "
              f"{len(gate['fits_above_probability_threshold'])} fits differ by more than "
              f"{gate['probability_listing_threshold']:g}.", '',
              '| Condition | Fits | Penalties differing | Max abs probability difference | LL here | LL v4 | Difference |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for c, g in gate['by_condition'].items():
        lines.append(f"| {c} | {g['fits']} | {g['penalties_differing']} | {g['max_abs_probability_difference']:.3g} | "
                     f"{g['mean_log_likelihood_bits']:.7f} | {g['v4_mean_log_likelihood_bits']:.7f} | "
                     f"{g['log_likelihood_difference_bits']:.2g} |")
    if gate['penalty_differences']:
        lines += ['', "Fits whose chosen penalty (history, S) differs from v4's:", '',
                  '| Fold | Condition | Chosen here | v4 chose | Max abs probability difference |', '|---|---|---|---|---:|']
        lines += [f"| {d['fold_id']} | {d['condition']} | {d['chosen_l2']}, {d['chosen_l2_s']} | {d['v4_chosen_l2']}, "
                  f"{d['v4_chosen_l2_s']} | {d['max_abs_probability_difference']:.3g} |" for d in gate['penalty_differences']]
    lines += [line for note in config.get('process_notes', []) for line in ('', note)]

    lines += ['', '## Log-likelihood and accuracy by variant', '',
              f"Held-out log-likelihood (bits/coda, {result['comparison_targets']:,} targets), accuracy, the difference to "
              f"repeat-own-last ({result['repeat_own_last_bits']:.5f} bits/coda) with its day-bootstrap CI, and the chosen "
              'history L2 (median over folds; folds at a grid edge).', '',
              '| Variant | ' + ' | '.join(f'{c} LL | {c} accuracy | {c} − repeat-own-last [CI] | {c} history L2 (edge)'
                                       for c in conditions) + ' |',
              '|---|' + '---:|---:|---:|---:|' * len(conditions)]
    for v in variants:
        cells = []
        for c in conditions:
            m = result['metrics'][v][c]
            rep = m['versus_repeat_own_last']
            cells.append(f"{m['mean_log_likelihood_bits']:.5f} | {m['accuracy']:.3%} | {rep['mean_bits_per_coda']:.5f} "
                         f"{format_ci(rep['ci_bits_per_coda'])} | {m['history_l2']['median']:g} "
                         f"({m['history_l2']['at_grid_edge']}/{m['history_l2']['folds']})")
        lines.append(f"| {v} | " + ' | '.join(cells) + ' |')

    lines += ['', '## Fitted weights', '',
              'Min / median / max over the 14 folds of the copy weight β and each class-feature weight, at the chosen '
              'penalties. Per-fold values are in `summary.json` (`coefficients`).', '',
              '| Variant | Condition | ' + ' | '.join(SYMBOLS[k] for k in ('copy_beta',) + FEATURES) + ' |',
              '|---|---|' + '---:|' * (1 + len(FEATURES))]
    for v in variants:
        for c in conditions:
            w = result['coefficients'][v][c]
            lines.append(f"| {v} | {c} | " + ' | '.join(_minmax(w[k]) if k in w else '—' for k in ('copy_beta',) + FEATURES)
                         + ' |')

    lines += ['', '## Contrasts', '',
              f"Primary (pre-specified): LL({config['primary']['full']}) - LL({config['primary']['reduced']}) in "
              f"{config['primary']['condition']}: does the order of the whale's earlier switches add usable information "
              f"beyond which types it has used and how often? Secondary (pre-specified): LL({config['secondary']['full']}) - "
              f"LL({config['secondary']['reduced']}) in {config['secondary']['condition']}: does the collapsed view of own "
              'history add anything to v4? Both get the exact two-sided sign-flip p (DEC-23: all 2^14 sign assignments of the '
              'day sums, token-weighted mean as the statistic). Every other row is descriptive. Every contrast adds class '
              'features, so an interval entirely below zero is a fitting flag (DEC-28) and is not interpreted.', '',
              '| Condition | Contrast | Role | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part [CI] '
              '| Switch part [CI] | Sign-flip p (two-sided) | Fitting flag |',
              '|---|---|---|---:|---:|---|---:|---:|---:|---:|---|']
    for c, by_pair in result['contrasts'].items():
        for contrast in by_pair.values():
            comp = contrast['components']
            role = f"**{contrast['role']}**" if contrast['role'] != 'descriptive' else 'descriptive'
            p_two = comp['log_likelihood_bits'].get('sign_flip_p_two_sided')
            lines.append(f"| {c} | {contrast['full']} − {contrast['reduced']} | {role} | {_stat(comp['log_likelihood_bits'])} | "
                         f"{_part(comp['repeat_bits'])} | {_part(comp['switch_choice_bits'])} | "
                         f"{'' if p_two is None else f'{p_two:.4f}'} | {'yes' if contrast['fitting_flag'] else 'no'} |")
    flagged = [f"{c}, {k}" for c, by_pair in result['contrasts'].items() for k, v in by_pair.items() if v['fitting_flag']]
    lines += ['', (f"Fitting flags (interval entirely below zero; not interpreted): {'; '.join(flagged)}." if flagged else
                   'No contrast has an interval entirely below zero (no fitting flag).')]

    lines += ['', '## Stop rules per variant (DEC-23, with DEC-16 beside it)', '',
              "Each variant is a full run of the 16 checks: the 12 add-one-input contrasts and the 4 A-conditions against "
              "repeat-own-last. DEC-23's rule, which governs (DEC-28), gives each check an exact one-sided sign-flip test "
              "(alternative: mean below zero) on its day sums, with Holm's correction across the 16, and fires when an "
              "adjusted p is below 0.05 (`logistic_v5.dec23_stop_rules`). DEC-16's rule, reported beside it, fires when a "
              "check's day-bootstrap interval lies entirely below zero. Each check's day sums are the variant's stored day "
              "means times the day's target count, as in the [P2] sweep.", '',
              "| Variant | DEC-23: Holm-adjusted p < 0.05 | Smallest one-sided p, Holm-adjusted (check) | DEC-16: interval "
              "entirely below zero |", '|---|---|---|---|']
    for v in variants:
        reasons, dec23 = result['stop_rules'][v], result['stop_rules_dec23'][v]
        fired = [f"{r['check']} (p = {r['sign_flip_p']:.4f}, Holm {r['holm_p']:.4f})" for r in dec23['checks'] if r['fires']]
        smallest = min(dec23['checks'], key=lambda r: r['sign_flip_p'])
        lines.append(f"| {v} | {'; '.join(fired) if fired else 'no stop rule fires'} | {smallest['sign_flip_p']:.4f}, Holm "
                     f"{smallest['holm_p']:.4f} ({smallest['check']}) | {'; '.join(reasons) if reasons else 'no stop rule fires'} |")

    discriminating = descriptive['quantities']['switches_with_d2']['count']
    lines += ['', '## Reading (DEC-28, primary only)', ''] + result['reading']
    lines += ['', '## Limits', '',
              f"- The primary rests on few discriminating switches: only {discriminating} switches have two or more other "
              'own types to choose between, so a null is weak.',
              '- No calibration (planted signal) was run, so no size of order effect can be ruled out.',
              '- Run after the results freeze (2026-10-04) as a final run; the descriptive numbers that motivated it were '
              'seen first (DEC-28). The features, variants, grid and rules were fixed before any fit.',
              '- The class-feature weights share one history penalty with the lag one-hots, the partner block and the copy '
              "weights (v4's rule), so a null may partly reflect that compromise.",
              '- Day-bootstrap intervals are approximate with only 14 days; the switch part rests on about 270 non-repeat '
              'codas and is exploratory.',
              '- Claims are about dyadic exchanges only; nothing here is evidence of grammar or syntax.', '',
              f"Per-variant outputs are in `{config['output_directory']}/<variant>/` (summary.json with both stop-rule inputs "
              'and input SHA-256, l2_selection.jsonl, scored.jsonl for A and A_B_S, coefficients.jsonl), the descriptive '
              f"table in `{config['output_directory']}/descriptive/`; `{config['output_directory']}/summary.json` records "
              'SHA-256 of every input to this report.', '']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='config/rq1_switch_structure.json')
    parser.add_argument('--stage', choices=('all', 'describe', 'fit', 'report'), default='all')
    parser.add_argument('--variants', nargs='+', help='fit stage: the variants to fit now (default: every variant)')
    # Each process imports torch through residual.py (about 230 MB before any data).
    parser.add_argument('--workers', type=int, default=5, help='processes for the fit stage (default 5)')
    args = parser.parse_args()
    if args.stage in ('all', 'describe'):
        run_describe(args.config)
    if args.stage in ('all', 'fit'):
        run_fit(args.config, args.variants, args.workers)
    if args.stage in ('all', 'report'):
        result = run_report(args.config)
        print('\n'.join(result['reading']))


if __name__ == '__main__':
    main()
