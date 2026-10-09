"""Rubato inputs, PLAN [R2-9] (DEC-29): the previous codas' durations as continuous inputs to logistic v4.

Logistic v4 unchanged on inventories A (`guy_coda_type`, RQ1), D (Sharma rhythm x KDE tempo) and E' (`guy_coda_type`
x tempos-dict tempo): each inventory's committed v4 configuration (targets, folds, UNK policy, S, copy weights, grid,
selection rule and convergence handling), plus logistic.Encoder's continuous history fields:
- A_log_duration: for lags 1-3 of A_history, the coda's ln d, d = offset_s - onset_s, one column per lag,
  standardised on the column's non-missing training values and 0 when the lag is missing; per-class weights like the
  lag one-hots; history block, history penalty;
- B_log_duration: the same for lags 1-3 of B_history (the partner's codas completed before the target's onset, [D5]).
Each is a per-row list aligned with its history field, built from the history coda IDs and the durations in
`rq1_targets.jsonl` (all rows); every referenced coda must end at or before the target's onset. In a condition's field
list a duration field follows its history field.

Variants: R (v4), DurA (R + own durations), Dur (R + own and partner durations). DurA fits the conditions with A, Dur
those with A or B; every other condition reuses R's fit after an exact refit check on one fold (`reuse_check_fold`).
Dur's A and A_S have the same inputs as DurA's; both are fitted, and they must agree exactly.

Stages (--stage): checks (DEC-29's data checks against the config's expected values; no models), fit (one task per
inventory, variant and fold in a spawned pool, one BLAS thread each: R on every inventory first, then each inventory's
gate (DEC-25's amended rule against its committed v4 run; a failure stops the run), then DurA and Dur inventory by
inventory in `fit_order`), report (reads only the stage outputs; regenerates byte-identically). Each stage records
the SHA-256 of exactly the files the config lists for it (`input_files`) and raises if it read others.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .duration_lattice import compare_checks, require_clean_tree, stage_inputs, write_summary
from .evaluation import day_bootstrap_bits, holm, sign_flip_p
from .history_length import (dec23_checks, environment, penalty_summary, process_pool, reference_gate, require_current,
                             require_one_blas_thread)
from .logistic import CONTINUOUS_HISTORY_FIELDS, check_stop_conditions, evaluate_v2_lattice, fit_v2_fold, format_ci
from .logistic_v5 import day_sums
from .preprocessing.folds import prepare_fold
from .preprocessing.io_utils import read_csv_rows, read_json, read_jsonl, write_jsonl, write_text
from .residual import _sha256

DURATION_FIELDS = ('A_log_duration', 'B_log_duration')
IDENTITY_FIELDS = ('coda_id', 'onset_s', 'offset_s', 'comparison_eligible', 'A_history_ids', 'B_history_ids')
LABEL_FIELDS = ('Ct', 'Ct_raw', 'A_history', 'B_history', 'inventory')
COMPONENTS = ('log_likelihood_bits', 'repeat_bits', 'switch_choice_bits')
SCORED_FIELDS = ('coda_id', 'condition', 'fold_id', 'recording_date', 'target', 'own_last', 'predicted',
                 'log_likelihood_bits', 'repeat_bits', 'switch_choice_bits')
TURN_CATEGORIES = ('self_continuation', 'switch', 'overlap')
VARIANT_FILES = ('summary.json', 'l2_selection.jsonl', 'scored.jsonl', 'coefficients.jsonl')
# Added-input contrasts (DEC-29's fitting flag): the variant pairs, each in the conditions where they differ.
ADDED_INPUTS = (('DurA', 'R', ('A', 'A_B', 'A_S', 'A_B_S')), ('Dur', 'DurA', ('B', 'A_B', 'B_S', 'A_B_S')),
                ('Dur', 'R', ('A', 'B', 'A_B', 'A_S', 'B_S', 'A_B_S')))
STAGES = ('checks', 'fit', 'report')


# ---------------------------------------------------------------- configuration and provenance

def load(config_path):
    """The config, and per inventory its committed v4 configuration, fold manifest, evaluation configuration and
    targets file (checked against the manifest's SHA-256)."""
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    inventories = {}
    for key, spec in config['inventories'].items():
        logistic_path = root / spec['logistic_config']
        logistic = read_json(logistic_path)
        manifest_path, evaluation_path = root / logistic['manifest'], root / logistic['evaluation_config']
        manifest = read_json(manifest_path)
        source = root / manifest['configuration']['input']
        if _sha256(source) != manifest['input_sha256']:
            raise ValueError(f'{key}: stale fold manifest')
        inventories[key] = dict(spec=spec, logistic_config=logistic, logistic_config_path=logistic_path,
                                manifest=manifest, manifest_path=manifest_path, evaluation_config=read_json(evaluation_path),
                                evaluation_config_path=evaluation_path, source=source,
                                v4_directory=root / spec['v4_results_directory'])
    loaded = dict(path=path, root=root, config=config, inventories=inventories, targets=root / config['targets'],
                  raw_source_path=root / config['raw_source'])
    check_config(loaded)
    return loaded


def check_config(loaded):
    config, inventories = loaded['config'], loaded['inventories']
    variants, reference = config['variants'], config['reference_variant']
    if reference not in variants or variants[reference]:
        raise ValueError('The reference variant must be one of the variants and have no duration inputs')
    if any(f not in DURATION_FIELDS for names in variants.values() for f in names):
        raise ValueError(f'Duration inputs must be among {DURATION_FIELDS}')
    if set(config['fit_order']) != set(inventories) or len(config['fit_order']) != len(inventories):
        raise ValueError('fit_order must list every inventory once')
    first = next(iter(inventories.values()))
    for key, inv in inventories.items():
        logistic, manifest = inv['logistic_config'], inv['manifest']
        if logistic['history_order'] != config['history_lags'] or 'history_orders' in logistic:
            raise ValueError(f"{key}: the duration lags must be logistic v4's own history lags")
        if manifest['input_conditions'] != first['manifest']['input_conditions']:
            raise ValueError(f'{key}: the inventories must share the manifest conditions')
        if [f['fold_id'] for f in manifest['folds']] != [f['fold_id'] for f in first['manifest']['folds']]:
            raise ValueError(f'{key}: the inventories must share the folds')
        evaluation = inv['evaluation_config']
        if [evaluation[k] for k in ('bootstrap_replicates', 'bootstrap_seed', 'confidence')] != [
                first['evaluation_config'][k] for k in ('bootstrap_replicates', 'bootstrap_seed', 'confidence')]:
            raise ValueError(f'{key}: the inventories must share the bootstrap settings')
    if config['reuse_check_fold'] not in [f['fold_id'] for f in first['manifest']['folds']]:
        raise ValueError('reuse_check_fold must be a fold of the manifests')
    conditions = first['manifest']['input_conditions']
    if any(c not in conditions for c in config['scored_conditions']):
        raise ValueError('scored_conditions must be manifest conditions')
    for spec in [config['primary']] + config['secondary']:
        if spec['inventory'] not in inventories or any(
                side[0] not in variants or side[1] not in config['scored_conditions'] for side in (spec['full'], spec['reduced'])):
            raise ValueError('Tested contrasts must name an inventory, a variant and a scored condition on each side')
    if set(config['input_files']) != set(STAGES):
        raise ValueError(f'input_files must list the inputs of every stage: {STAGES}')


def external_inputs(loaded):
    """The files load() reads, whatever the stage: this config, rq1_targets.jsonl (durations) and per inventory its
    v4 configuration, fold manifest, evaluation configuration and targets file."""
    paths = [loaded['path'], loaded['targets']]
    for inv in loaded['inventories'].values():
        paths += [inv['logistic_config_path'], inv['manifest_path'], inv['evaluation_config_path'], inv['source']]
    return list(dict.fromkeys(paths))


def gate_inputs(inv):
    directory = inv['v4_directory']
    return [directory / 'predictions.jsonl', directory / 'l2_selection.jsonl', directory / 'evaluation' / 'summary.json']


def output_path(loaded, *parts):
    return loaded['root'].joinpath(loaded['config']['output_directory'], *parts)


def variant_paths(loaded, inventory, variant):
    return [output_path(loaded, 'fit', inventory, variant, name) for name in VARIANT_FILES]


def require_summary(loaded, path, stage):
    if not path.is_file():
        raise ValueError(f'The {stage} stage has not been run (run --stage {stage})')
    require_current(loaded, path)
    return read_json(path)


def require_checks(loaded):
    path = output_path(loaded, 'checks', 'summary.json')
    summary = require_summary(loaded, path, 'checks')
    if loaded['config']['checks']['enforce'] and not summary['passed']:
        raise ValueError('The data checks do not reproduce the expected values; stop here')
    return path


# ---------------------------------------------------------------- durations

def duration_lookup(rows):
    """coda_id -> onset, offset and ln d (d = offset_s - onset_s) for every row of rq1_targets.jsonl."""
    lookup = {}
    for r in rows:
        duration = r['offset_s'] - r['onset_s']
        if r['coda_id'] in lookup:
            raise ValueError(f"Duplicate coda ID {r['coda_id']}")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{r['coda_id']}: duration must be positive")
        lookup[r['coda_id']] = dict(onset_s=r['onset_s'], offset_s=r['offset_s'], log_duration=math.log(duration))
    return lookup


def with_durations(rows, lookup):
    """Each row with A_log_duration and B_log_duration: the ln d of every coda in A_history_ids and B_history_ids,
    aligned with them. Every referenced coda must be in the lookup and end at or before the row's onset."""
    out = []
    for r in rows:
        values = {}
        for field, ids in (('A_log_duration', r['A_history_ids']), ('B_log_duration', r['B_history_ids'])):
            for coda_id in ids:
                if coda_id not in lookup:
                    raise ValueError(f"{r['coda_id']}: history coda {coda_id} has no duration")
                if lookup[coda_id]['offset_s'] > r['onset_s']:
                    raise ValueError(f"{r['coda_id']}: history coda {coda_id} ends after the target's onset")
            values[field] = [lookup[coda_id]['log_duration'] for coda_id in ids]
        out.append(dict(r, **values))
    return out


def variant_fields(fields, durations):
    """A condition's fields with the variant's duration fields, each right after its history field when that field
    is present (a condition without the history field gets none)."""
    out = []
    for f in fields:
        out.append(f)
        out += [d for d in DURATION_FIELDS if d in durations and CONTINUOUS_HISTORY_FIELDS[d] == f]
    return out


def variant_conditions(conditions, durations):
    """name -> fields for every condition under a variant, and the names whose fields differ from v4's."""
    fields = {c: variant_fields(f, durations) for c, f in conditions.items()}
    return fields, [c for c in conditions if fields[c] != conditions[c]]


# ---------------------------------------------------------------- checks stage (no models)

def _pearson(a, b):
    return float(np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1])


def _centred(values, labels):
    """Each value minus the mean of the values sharing its label."""
    groups = defaultdict(list)
    for v, k in zip(values, labels):
        groups[k].append(v)
    means = {k: math.fsum(v) / len(v) for k, v in groups.items()}
    return [v - means[k] for v, k in zip(values, labels)]


def inventory_alignment(reference_rows, rows):
    """Row for row against rq1_targets.jsonl: rows whose identity fields differ, and which other fields differ
    anywhere, split into label fields (expected to differ) and the rest (expected not to)."""
    if len(rows) != len(reference_rows):
        return dict(rows=len(rows), mismatched_rows=None, other_fields_differing=None, label_fields_differing=None)
    mismatched, other, labels = 0, set(), set()
    for a, b in zip(reference_rows, rows):
        mismatched += any(a[k] != b[k] for k in IDENTITY_FIELDS)
        for k in set(a) | set(b):
            if (k in a) != (k in b) or a[k] != b[k]:
                (labels if k in LABEL_FIELDS else other).add(k)
    return dict(rows=len(rows), mismatched_rows=mismatched, other_fields_differing=sorted(other),
                label_fields_differing=sorted(labels))


def fold_alignment(reference_manifest, manifest):
    """Same folds: fold IDs, test and validation days and comparison IDs per split."""
    key = lambda m: [(f['fold_id'], f['test_day'], f['validation_day'], f['comparison_ids']) for f in m['folds']]  # noqa: E731
    return dict(identical=key(reference_manifest) == key(manifest))


def data_checks(rows, raw_rows, inventory_rows, reference_manifest, manifests, lags):
    """DEC-29's data checks on RQ1's targets (`rows`, all of rq1_targets.jsonl) and the raw rows, plus the alignment of
    every other inventory's targets file (`inventory_rows`) and fold manifest with rq1_targets.jsonl and its manifest
    (`reference_manifest`); no models."""
    by_id = {r['coda_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate coda IDs')
    targets = [r for r in rows if r['comparison_eligible']]
    q = {}
    for key, other in inventory_rows.items():
        q[f'inventory_alignment:{key}'] = inventory_alignment(rows, other)
        q[f'fold_alignment:{key}'] = fold_alignment(reference_manifest, manifests[key])
    q['rows'] = dict(count=len(rows))
    q['comparison_targets'] = dict(count=len(targets))
    references = [(t, i) for t in targets for field in ('A_history_ids', 'B_history_ids') for i in t[field][-lags:]]
    q['history_references'] = dict(count=len(references), unresolved=sum(i not in by_id for _, i in references),
                                   ending_after_target_onset=sum(i in by_id and by_id[i]['offset_s'] > t['onset_s']
                                                                 for t, i in references))
    q['duration_vs_raw'] = dict(max_abs=max(abs(r['offset_s'] - r['onset_s'] - float(raw_rows[r['source_row_index']]['Duration']))
                                            for r in rows))
    for name, field in (('own_lags', 'A_history_ids'), ('partner_lags', 'B_history_ids')):
        q[name] = dict(counts=[sum(len(t[field]) >= k for t in targets) for k in range(1, lags + 1)])
    log_d = {r['coda_id']: math.log(r['offset_s'] - r['onset_s']) for r in rows}
    y = [log_d[t['coda_id']] for t in targets]
    own = [t['A_history_ids'][-1] for t in targets]
    partner = [t['B_history_ids'][-1] for t in targets]
    q['target_log_duration'] = dict(mean=statistics.fmean(y), sd=statistics.pstdev(y))
    q['lag1_r'] = dict(own=_pearson(y, [log_d[i] for i in own]), partner=_pearson(y, [log_d[i] for i in partner]))
    # Each variable minus its mean over the 1,547 rows sharing its coda's label (Ct as stored; a history coda by its own).
    centred_y = _centred(y, [t['Ct'] for t in targets])
    q['lag1_r_within_label'] = {
        name: _pearson(centred_y, _centred([log_d[i] for i in ids], [by_id[i]['Ct'] for i in ids]))
        for name, ids in (('own', own), ('partner', partner))}
    return q


def read_raw(loaded):
    return read_csv_rows(loaded['raw_source_path'], encoding='utf-8-sig')[1]


def run_checks(config_path):
    loaded = load(config_path)
    config, settings = loaded['config'], loaded['config']['checks']
    git = require_clean_tree(loaded)
    rows = read_jsonl(loaded['targets'])
    inventories = loaded['inventories']
    own = [inv for inv in inventories.values() if inv['source'] == loaded['targets']]
    if len(own) != 1:
        raise ValueError("Exactly one inventory must be rq1_targets.jsonl's (inventory A)")
    quantities = data_checks(rows, read_raw(loaded), {k: read_jsonl(inv['source']) for k, inv in inventories.items()
                                                      if inv['source'] != loaded['targets']},
                             own[0]['manifest'], {k: inv['manifest'] for k, inv in inventories.items()},
                             config['history_lags'])
    checks = compare_checks(quantities, settings['expected'], settings['tolerance'])
    passed = all(c['passed'] for c in checks)
    summary = dict(source=settings['source'], tolerance=settings['tolerance'], quantities=quantities, checks=checks,
                   passed=passed, git=git,
                   input_sha256=stage_inputs(loaded, 'checks', external_inputs(loaded) + [loaded['raw_source_path']]))
    write_summary(output_path(loaded, 'checks', 'summary.json'), summary)
    failed = [c for c in checks if not c['passed']]
    print(f'Data checks: {len(checks) - len(failed)} of {len(checks)} reproduced', flush=True)
    for c in failed:
        print(f"  DIFFERS {c['name']}: computed {c['computed']}, expected {c['expected']}", flush=True)
    if failed and settings['enforce']:
        raise ValueError('The data checks differ from the expected values (checks/summary.json); stop here')
    return summary


# ---------------------------------------------------------------- fit stage

_LOADED = {}


def _inventory(config_path, inventory):
    """Per process: the loaded config and the inventory's rows with their duration fields."""
    if (config_path, inventory) not in _LOADED:
        loaded = load(config_path)
        inv = loaded['inventories'][inventory]
        rows = with_durations(read_jsonl(inv['source']), duration_lookup(read_jsonl(loaded['targets'])))
        _LOADED[config_path, inventory] = (loaded, inv, rows)
    return _LOADED[config_path, inventory]


def _fit_task(task):
    """One inventory, variant and fold: fit_v2_fold on the given conditions with the variant's duration fields.
    Module level so that spawned processes can import it."""
    config_path, inventory, variant, fold_id, conditions = task
    require_one_blas_thread()
    start = time.perf_counter()
    loaded, inv, rows = _inventory(config_path, inventory)
    manifest = inv['manifest']
    fold = next(f for f in manifest['folds'] if f['fold_id'] == fold_id)
    mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
    if policy != fold['vocabulary_policy']:
        raise ValueError(f'{inventory}, {fold_id}: recomputed fold policy differs from manifest')
    fields, _ = variant_conditions(manifest['input_conditions'], loaded['config']['variants'][variant])
    fitted = fit_v2_fold(inv['logistic_config'], fold, mapped, policy, {c: fields[c] for c in conditions})
    return dict(inventory=inventory, variant=variant, fold_id=fold_id, conditions=list(conditions),
                seconds=time.perf_counter() - start, **fitted)


def _of(records, condition):
    return [r for r in records if r['condition'] == condition]


def merge_folds(manifest, results, reference=None, reuse=()):
    """One variant's records over the folds, condition by condition in manifest order within each fold (as
    fit_v2_lattice collects them). The conditions in `reuse` come from `reference` (fold_id -> R's fold result), the
    others from the variant's own fits; baselines must agree with the reference's."""
    by_fold = {r['fold_id']: r for r in results}
    merged = dict(predictions=[], baselines=[], models=[], diagnostics=[], selections=[])
    for fold in manifest['folds']:
        mine = by_fold[fold['fold_id']]
        if reference is not None and mine['baselines'] != reference[fold['fold_id']]['baselines']:
            raise ValueError(f"{fold['fold_id']}: baselines differ from the reference variant's")
        merged['baselines'] += mine['baselines']
        for c in manifest['input_conditions']:
            if c in reuse:
                source = reference[fold['fold_id']]
            elif c in mine['conditions']:
                source = mine
            else:
                raise ValueError(f"{fold['fold_id']}: condition {c} was neither fitted nor reused")
            for key in ('predictions', 'models', 'diagnostics', 'selections'):
                merged[key] += _of(source[key], c)
    return merged


def compare_fits(mine, theirs, condition):
    """Exact agreement of two fits of one condition on one fold: penalty searches, models and test predictions."""
    return {key: _of(mine[key], condition) == _of(theirs[key], condition)
            for key in ('selections', 'models', 'diagnostics', 'predictions')}


def coefficient_rows(models):
    """Per fold and condition with A or B: chosen penalties, copy weights and the duration weights (per lag and
    class, on the standardised inputs) with their norms."""
    out = []
    for m in models:
        if not {'A_history', 'B_history'} & set(m['fields']):
            continue
        weights = dict(zip(m['feature_names'], m['weights']))
        durations = {name: dict(zip(m['vocabulary'], w)) for name, w in weights.items() if name.split(':')[0] in DURATION_FIELDS}
        norms = {}
        for field in DURATION_FIELDS:
            rows = [w for name, w in weights.items() if name.split(':')[0] == field]
            if rows:
                norms[field] = math.sqrt(math.fsum(v * v for w in rows for v in w))
        out.append(dict(fold_id=m['fold_id'], condition=m['condition'], chosen_l2=m['chosen_l2'], chosen_l2_s=m['chosen_l2_s'],
                        beta=m['beta'], gamma=m['gamma'], duration_weight_norms=norms, duration_weights=durations))
    return out


def slim_scored(scored, conditions):
    return [{k: r[k] for k in SCORED_FIELDS} for r in scored if r['condition'] in conditions]


def prediction_hashes(predictions, conditions):
    return {c: hashlib.sha256(json.dumps(_of(predictions, c), allow_nan=False).encode()).hexdigest() for c in conditions}


def evaluate_variant(loaded, inventory, variant, merged, reused, fitted_conditions, task_seconds):
    """The variant's lattice: evaluate_v2_lattice on the merged predictions, DEC-16's stop reasons, and the records
    the report reads."""
    inv = loaded['inventories'][inventory]
    manifest, logistic = inv['manifest'], inv['logistic_config']
    rows = read_jsonl(inv['source'])
    evaluated = evaluate_v2_lattice(logistic, manifest, rows, inv['evaluation_config'], merged['predictions'],
                                    merged['baselines'])
    summary = evaluated['summary']
    conditions = list(manifest['input_conditions'])
    summary.update(inventory=inventory, variant=variant, duration_inputs=loaded['config']['variants'][variant],
                   configuration=logistic, baseline_metrics=evaluated['baseline_metrics'],
                   repeat_baseline_checks=evaluated['baseline_checks'],
                   stop_reasons=check_stop_conditions(summary, evaluated['baseline_metrics'], evaluated['baseline_checks']),
                   prediction_sha256=prediction_hashes(merged['predictions'], conditions),
                   fitted_conditions=fitted_conditions, reused_conditions=reused,
                   nonconverged_candidates=sum('nonconverged' in g for s in merged['selections']
                                               if s['condition'] not in reused for g in s['grid']),
                   task_seconds=task_seconds, environment=environment())
    selections = [dict(s, reused_from=loaded['config']['reference_variant']) if s['condition'] in reused else s
                  for s in merged['selections']]
    return dict(summary=summary, selections=selections, scored=slim_scored(evaluated['scored'], loaded['config']['scored_conditions']),
                coefficients=coefficient_rows(merged['models']), predictions=merged['predictions'])


def write_variant(loaded, inventory, variant, result):
    paths = variant_paths(loaded, inventory, variant)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths[1], result['selections'])
    write_jsonl(paths[2], result['scored'])
    write_jsonl(paths[3], result['coefficients'])
    write_summary(paths[0], result['summary'])


def _collect(futures):
    return [f.result() for f in futures]


def run_fit(config_path, workers=None):
    require_one_blas_thread()
    loaded = load(config_path)
    config = loaded['config']
    git = require_clean_tree(loaded)
    checks_path = require_checks(loaded)
    read = external_inputs(loaded) + [checks_path] + [p for inv in loaded['inventories'].values() for p in gate_inputs(inv)]
    input_sha256 = stage_inputs(loaded, 'fit', read)
    reference, order = config['reference_variant'], config['fit_order']
    check_fold = config['reuse_check_fold']
    path = str(loaded['path'])
    start = time.perf_counter()
    wall, gates, reuse_checks, same_inputs, results, estimate = {}, {}, {}, {}, {}, None
    with process_pool(workers or os.cpu_count()) as pool:
        def submit(inventory, variant, conditions_for):
            manifest = loaded['inventories'][inventory]['manifest']
            return [pool.submit(_fit_task, (path, inventory, variant, f['fold_id'], conditions_for(f['fold_id'])))
                    for f in manifest['folds']]

        everything = lambda inventory: (lambda fold_id: list(loaded['inventories'][inventory]['manifest']['input_conditions']))  # noqa: E731
        r_futures = {inv: submit(inv, reference, everything(inv)) for inv in order}
        r_folds = {}
        for inv in order:
            folds = _collect(r_futures[inv])
            r_folds[inv] = {r['fold_id']: r for r in folds}
            merged = merge_folds(loaded['inventories'][inv]['manifest'], folds)
            results[inv, reference] = evaluate_variant(loaded, inv, reference, merged, [],
                                                       list(loaded['inventories'][inv]['manifest']['input_conditions']),
                                                       sum(r['seconds'] for r in folds))
            spec = dict(config['gate'], results_directory=config['inventories'][inv]['v4_results_directory'])
            gate, _ = reference_gate(loaded['root'], spec, dict(selections=merged['selections'],
                                                                 predictions=merged['predictions'],
                                                                 summary=results[inv, reference]['summary']))
            results[inv, reference]['summary']['gate'] = gate
            gates[inv] = {k: v for k, v in gate.items() if k != 'pairs'}
            write_variant(loaded, inv, reference, results[inv, reference])
            print(f"{inv}: R gate {'passed' if gate['passed'] else 'FAILED'} ({len(gate['penalty_differences'])} of "
                  f"{gate['fits_compared']} penalties differ, mean LL within {gate['max_abs_log_likelihood_difference_bits']:.2g} "
                  f"bits/coda; max abs probability difference {gate['max_abs_probability_difference']:.3g}, reported only)",
                  flush=True)
        wall['R'] = time.perf_counter() - start
        failed = [inv for inv in order if not gates[inv]['passed']]
        if failed and config['gate']['enforce']:
            raise ValueError(f"The reference fit does not reproduce logistic v4 on {', '.join(failed)}; stop here")

        for i, inv in enumerate(order):
            begin = time.perf_counter()
            manifest = loaded['inventories'][inv]['manifest']
            conditions = manifest['input_conditions']
            futures = {}
            for variant, durations in config['variants'].items():
                if variant == reference:
                    continue
                _, changed = variant_conditions(conditions, durations)
                futures[variant] = submit(inv, variant, lambda fold_id, changed=changed: (
                    list(conditions) if fold_id == check_fold else changed))
            fitted_folds = {}
            for variant, fs in futures.items():
                folds = _collect(fs)
                fitted_folds[variant] = {r['fold_id']: r for r in folds}
                _, changed = variant_conditions(conditions, config['variants'][variant])
                unchanged = [c for c in conditions if c not in changed]
                check = {c: compare_fits(fitted_folds[variant][check_fold], r_folds[inv][check_fold], c) for c in unchanged}
                reuse_checks[inv, variant] = dict(fold_id=check_fold, conditions=check,
                                                  identical=all(all(v.values()) for v in check.values()))
                if not reuse_checks[inv, variant]['identical']:
                    raise ValueError(f'{inv}, {variant}: a refit of an unchanged condition on {check_fold} differs from R; stop here')
                merged = merge_folds(manifest, folds, r_folds[inv], unchanged)
                results[inv, variant] = evaluate_variant(loaded, inv, variant, merged, unchanged, changed,
                                                         sum(r['seconds'] for r in folds))
            # Conditions whose fields agree between two fitted variants (Dur's A and A_S are DurA's) must agree exactly.
            variants = [v for v in config['variants'] if v != reference]
            for a in variants:
                for b in variants[variants.index(a) + 1:]:
                    fa, ca = variant_conditions(conditions, config['variants'][a])
                    fb, cb = variant_conditions(conditions, config['variants'][b])
                    shared = [c for c in conditions if c in ca and c in cb and fa[c] == fb[c]]
                    agree = {c: all(all(compare_fits(fitted_folds[a][f['fold_id']], fitted_folds[b][f['fold_id']], c).values())
                                    for f in manifest['folds']) for c in shared}
                    same_inputs[inv, f'{b}={a}'] = agree
                    if not all(agree.values()):
                        raise ValueError(f'{inv}: {a} and {b} fit the same inputs differently; stop here')
            for variant in variants:
                write_variant(loaded, inv, variant, results[inv, variant])
            wall[inv] = time.perf_counter() - begin
            print(f"{inv}: DurA and Dur fitted in {wall[inv]:.0f} s", flush=True)
            if i == 0 and len(order) > 1:
                cpu = {v: results[v, reference]['summary']['task_seconds'] for v in order}
                estimate = dict(after=inv, seconds=sum(wall[inv] * cpu[v] / cpu[inv] for v in order[1:]),
                                rule="the first inventory's DurA and Dur wall time, scaled by R's CPU time per inventory")
                print(f"Estimate for {', '.join(order[1:])}: about {estimate['seconds']:.0f} s ({estimate['rule']})", flush=True)

    summary = dict(units='bits per coda', inventories=order, variants=config['variants'], reference_variant=reference,
                   gates=gates, reuse_checks={f'{inv}:{v}': c for (inv, v), c in reuse_checks.items()},
                   same_inputs={f'{inv}:{k}': v for (inv, k), v in same_inputs.items()},
                   task_seconds={f'{inv}:{v}': r['summary']['task_seconds'] for (inv, v), r in results.items()},
                   wall_seconds=wall, estimate=estimate, total_seconds=time.perf_counter() - start,
                   workers=workers or os.cpu_count(),
                   environment=environment(), git=git, input_sha256=input_sha256)
    write_summary(output_path(loaded, 'fit', 'summary.json'), summary)
    print(f"Fit stage: {summary['total_seconds']:.0f} s", flush=True)
    return summary


# ---------------------------------------------------------------- report stage

def by_coda(rows, condition):
    out = {r['coda_id']: r for r in rows if r['condition'] == condition}
    if len(out) != sum(r['condition'] == condition for r in rows):
        raise ValueError('More than one scored row per coda and condition')
    return out


def differences(full, reduced):
    """Per component, coda_id -> full - reduced; both sides must score the same targets with the same day, target
    and own last label."""
    if set(full) != set(reduced):
        raise ValueError('The two fits scored different targets')
    out = {f: {} for f in COMPONENTS}
    for coda_id in sorted(full):
        a, b = full[coda_id], reduced[coda_id]
        if (a['recording_date'], a['target'], a['own_last']) != (b['recording_date'], b['target'], b['own_last']):
            raise ValueError(f'{coda_id}: the two fits disagree on the target')
        for f in COMPONENTS:
            if a[f] is not None and b[f] is not None:
                out[f][coda_id] = a[f] - b[f]
    return out


def contrast_stats(values, dates, boot, sign_flip=False):
    """Per component: day bootstrap (token-weighted mean, unweighted day mean, CI, positive days) and days at exactly
    zero; with `sign_flip`, the total's exact two-sided sign-flip p on its day sums (DEC-23)."""
    out = {}
    for f, by_id in values.items():
        days = defaultdict(list)
        for coda_id, value in by_id.items():
            days[dates[coda_id]].append(value)
        stats = day_bootstrap_bits(dict(days), *boot)
        sums, counts = day_sums(by_id, dates)
        stats['zero_days'] = sum(s == 0 for s in sums.values())
        if sign_flip and f == 'log_likelihood_bits':
            stats['sign_flip_p_two_sided'] = sign_flip_p(sums, counts, 'two-sided')
        out[f] = stats
    return out


def breakdown(values, groups, labels, dates):
    """Per group of targets: token-weighted mean of each component, targets, days and positive days; no tests."""
    out = {}
    for label in labels:
        ids = sorted(c for c in values['log_likelihood_bits'] if groups[c] == label)
        sums, _ = day_sums({c: values['log_likelihood_bits'][c] for c in ids}, dates)
        out[label] = dict(codas=len(ids), days=len(sums), positive_days=sum(s > 0 for s in sums.values()),
                          **{f'mean_{f}': math.fsum(values[f][c] for c in ids) / len(ids) for f in COMPONENTS})
    return out


def detected(stats, p_key, alpha):
    """DEC-29's detection rule as in DEC-30 and DEC-31: interval entirely above zero and p below alpha."""
    return stats['ci_bits_per_coda'][0] > 0 and stats[p_key] < alpha


def norm_summary(coefficients, condition, field):
    values = [r['duration_weight_norms'][field] for r in coefficients
              if r['condition'] == condition and field in r['duration_weight_norms']]
    if not values:
        return None
    return dict(min=min(values), median=statistics.median(values), max=max(values), folds=len(values))


def canonical_fit(config, conditions, variant, condition):
    """The first variant (config order) with the same fields as `variant` in `condition`: fits with the same inputs
    are the same fit (checked exactly in the fit stage)."""
    target = variant_conditions(conditions, config['variants'][variant])[0][condition]
    return next(v for v in config['variants'] if variant_conditions(conditions, config['variants'][v])[0][condition] == target)


def run_report(config_path):
    loaded = load(config_path)
    root, config = loaded['root'], loaded['config']
    inventories, order, reference = loaded['inventories'], config['fit_order'], config['reference_variant']
    variants = list(config['variants'])
    first = next(iter(inventories.values()))
    evaluation = first['evaluation_config']
    boot = (evaluation['bootstrap_replicates'], evaluation['bootstrap_seed'], evaluation['confidence'])
    alpha, grid = config['alpha'], first['logistic_config']['l2_grid']
    require_clean_tree(loaded)
    checks_path = require_checks(loaded)
    fit_path = output_path(loaded, 'fit', 'summary.json')
    fit_summary = require_summary(loaded, fit_path, 'fit')
    read = external_inputs(loaded) + [checks_path, fit_path]
    summaries, selections, scored, coefficients = {}, {}, {}, {}
    for inv in order:
        for v in variants:
            paths = variant_paths(loaded, inv, v)
            if not all(p.is_file() for p in paths):
                raise ValueError(f'{inv}, {v} has not been fitted (run --stage fit)')
            summaries[inv, v], selections[inv, v], scored[inv, v], coefficients[inv, v] = (
                read_json(paths[0]), *map(read_jsonl, paths[1:]))
            if (summaries[inv, v]['configuration'] != inventories[inv]['logistic_config']
                    or summaries[inv, v]['duration_inputs'] != config['variants'][v]):
                raise ValueError(f'{inv}, {v} was fitted with another configuration')
            read += paths
    input_sha256 = stage_inputs(loaded, 'report', read)
    checks = read_json(checks_path)
    gates = {inv: summaries[inv, reference]['gate'] for inv in order}
    if config['gate']['enforce'] and not all(g['passed'] for g in gates.values()):
        raise ValueError('A reference fit does not reproduce logistic v4 (see the gates); stop here')
    conditions = first['manifest']['input_conditions']
    # Reused conditions carry R's fits, so they must equal R's exactly.
    for inv in order:
        for v in variants:
            for c in summaries[inv, v]['reused_conditions']:
                if (summaries[inv, v]['prediction_sha256'][c] != summaries[inv, reference]['prediction_sha256'][c]
                        or summaries[inv, v]['metrics'][c] != summaries[inv, reference]['metrics'][c]):
                    raise ValueError(f'{inv}, {v}: reused condition {c} differs from {reference}')

    rows = read_jsonl(loaded['targets'])
    dates = {r['coda_id']: r['recording_date'] for r in rows if r['comparison_eligible']}
    categories = {r['coda_id']: r['turn_category'] for r in rows if r['comparison_eligible']}
    fits = {(inv, v, c): by_coda(scored[inv, v], c) for inv in order for v in variants for c in config['scored_conditions']}
    for key, by_id in fits.items():
        if set(by_id) != set(dates):
            raise ValueError(f'{key}: scored rows do not cover the comparison targets')

    def contrast(inv, full, reduced, sign_flip=False):
        values = differences(fits[(inv, *full)], fits[(inv, *reduced)])
        return dict(inventory=inv, full=list(full), reduced=list(reduced), values=values,
                    components=contrast_stats(values, dates, boot, sign_flip))

    def public(c):
        return {k: v for k, v in c.items() if k != 'values'}

    tested = [dict(public(contrast(config['primary']['inventory'], config['primary']['full'], config['primary']['reduced'], True)),
                   role='primary', name=config['primary']['name'])]
    secondary = [dict(public(contrast(s['inventory'], s['full'], s['reduced'], True)), role='secondary', name=s['name'])
                 for s in config['secondary']]
    for s, adjusted in zip(secondary, holm([s['components']['log_likelihood_bits']['sign_flip_p_two_sided'] for s in secondary])):
        s['components']['log_likelihood_bits']['holm_p'] = adjusted
    tested += secondary
    for t in tested:
        stats = t['components']['log_likelihood_bits']
        t['detected'] = detected(stats, 'sign_flip_p_two_sided' if t['role'] == 'primary' else 'holm_p', alpha)

    descriptive, partner_change, by_turn, added, weights, metrics = {}, {}, {}, {}, {}, {}
    for inv in order:
        rows_ = []
        for v in ('DurA', 'Dur'):
            for c in ('A', 'A_B_S'):
                rows_.append(dict(public(contrast(inv, (v, c), (reference, c))), name=f'{v} − {reference}'))
        for name, (full, reduced) in (('own history, under Dur', (('Dur', 'A_B_S'), ('Dur', 'B_S'))),
                                      ('timing, under Dur', (('Dur', 'A_B_S'), ('Dur', 'A_B'))),
                                      ('partner, under R', ((reference, 'A_B_S'), (reference, 'A_S')))):
            rows_.append(dict(public(contrast(inv, full, reduced)), name=name))
        descriptive[inv] = rows_
        under_dur = contrast(inv, ('Dur', 'A_B_S'), ('Dur', 'A_S'))
        under_r = contrast(inv, (reference, 'A_B_S'), (reference, 'A_S'))
        change = {f: {c: under_dur['values'][f][c] - under_r['values'][f][c] for c in under_dur['values'][f]
                      if c in under_r['values'][f]} for f in COMPONENTS}
        partner_change[inv] = dict(under_r=under_r['components'], under_dur=under_dur['components'],
                                   change=contrast_stats(change, dates, boot))
        by_turn[inv] = breakdown(under_dur['values'], categories, TURN_CATEGORIES, dates)
        added[inv] = []
        for full, reduced, conds in ADDED_INPUTS:
            for c in conds:
                stats = contrast(inv, (full, c), (reduced, c))
                ci = stats['components']['log_likelihood_bits']['ci_bits_per_coda']
                added[inv].append(dict(public(stats), fitting_flag=ci[1] < 0))
        weights[inv] = {v: {c: {f: norm_summary(coefficients[inv, v], c, f) for f in DURATION_FIELDS}
                            for c in ('A', 'A_B_S')} for v in variants if config['variants'][v]}
        metrics[inv] = {v: {c: dict(mean_log_likelihood_bits=summaries[inv, v]['metrics'][c]['mean_log_likelihood_bits'],
                                    accuracy=summaries[inv, v]['metrics'][c]['accuracy'],
                                    macro_f1=summaries[inv, v]['metrics'][c]['macro_f1'],
                                    versus_repeat_own_last=summaries[inv, v]['repeat_baseline_checks'].get(c),
                                    history_l2=penalty_summary([s for s in selections[inv, v]], c, grid) if c != 'none' else None)
                            for c in conditions} for v in variants}

    day_counts = {fold['test_day']: len(fold['comparison_ids']['test']) for fold in first['manifest']['folds']}
    stop_rules = {inv: {v: dict(dec16=summaries[inv, v]['stop_reasons'], dec23=dec23_checks(summaries[inv, v], day_counts))
                        for v in variants} for inv in order}
    result = dict(
        units='bits per coda', inventories={inv: config['inventories'][inv] for inv in order}, variants=config['variants'],
        reference_variant=reference, conditions=list(conditions), alpha=alpha,
        comparison_targets=summaries[order[0], reference]['comparison_targets'],
        bootstrap=dict(zip(('replicates', 'seed', 'confidence'), boot)),
        checks={k: checks[k] for k in ('source', 'tolerance', 'quantities', 'checks', 'passed')},
        gates={inv: {k: v for k, v in g.items() if k != 'pairs'} for inv, g in gates.items()},
        fit={k: fit_summary[k] for k in ('reuse_checks', 'same_inputs', 'task_seconds', 'wall_seconds', 'estimate',
                                         'total_seconds', 'workers', 'environment', 'git')},
        reused_conditions={f'{inv}:{v}': summaries[inv, v]['reused_conditions'] for inv in order for v in variants},
        nonconverged_candidates={f'{inv}:{v}': summaries[inv, v]['nonconverged_candidates'] for inv in order for v in variants},
        metrics=metrics, tested=tested, descriptive=descriptive, partner_change=partner_change,
        partner_by_turn_category=by_turn, added_inputs=added, duration_weight_norms=weights, stop_rules=stop_rules,
        repeat_own_last_bits={inv: summaries[inv, reference]['baseline_metrics']['repeat_own_last']['mean_log_likelihood_bits']
                              for inv in order})
    result['reading'] = reading_lines(result, config, conditions)
    result['input_sha256'] = input_sha256
    write_summary(root / config['output_directory'] / 'summary.json', result)
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(report_lines(result, config)))
    return result


# ---------------------------------------------------------------- reading and report

def _label(result, inv):
    return result['inventories'][inv]['label']


def _side(side):
    return f"LL_{side[0]}({side[1]})"


def _contrast_name(t):
    return f"{_side(t['full'])} − {_side(t['reduced'])}"


def _p(value):
    return 'n/a' if value is None else f'{value:.4f}'


def _part(stats):
    return f"{stats['mean_bits_per_coda']:.5f} {format_ci(stats['ci_bits_per_coda'])}"


def _tested_text(result, t):
    s = t['components']['log_likelihood_bits']
    holm_text = f", Holm-adjusted p = {s['holm_p']:.4f}" if 'holm_p' in s else ''
    return (f"{_contrast_name(t)} on {_label(result, t['inventory'])}: {s['mean_bits_per_coda']:.5f} bits/coda (unweighted "
            f"day mean {s['unweighted_mean_bits_per_coda']:.5f}, 95% CI {format_ci(s['ci_bits_per_coda'])}, "
            f"{s['positive_days']}/{s['days']} days positive, two-sided sign-flip p = {s['sign_flip_p_two_sided']:.4f}"
            f"{holm_text}); repeat part {_part(t['components']['repeat_bits'])}, switch part "
            f"{_part(t['components']['switch_choice_bits'])}")


def reading_lines(result, config, conditions):
    """DEC-29's Reading, chosen mechanically for the primary; its DurA − R bullet is applied descriptively (no test).
    Withheld when a gate fails, DEC-23's stop rule fires on the primary's inventory and variant, or an added-input
    contrast with one of the primary's fits as its full side carries a fitting flag."""
    primary, reference = result['tested'][0], result['reference_variant']
    inv = primary['inventory']
    s = primary['components']['log_likelihood_bits']
    low, high = s['ci_bits_per_coda']
    lines = [f"Primary, pre-specified: {_tested_text(result, primary)}."]
    rules = result['stop_rules'][inv]
    primary_variants = {primary['full'][0], primary['reduced'][0]}
    fired = [v for v in primary_variants if rules[v]['dec23']['fired']]
    primary_fits = {(canonical_fit(config, conditions, *side), side[1]) for side in (primary['full'], primary['reduced'])}
    flags = [a for a in result['added_inputs'][inv] if a['fitting_flag']
             and (canonical_fit(config, conditions, *a['full']), a['full'][1]) in primary_fits]
    by_name = {(t['inventory'], tuple(t['full']), tuple(t['reduced'])): t for t in result['tested']}
    if not all(g['passed'] for g in result['gates'].values()):
        lines.append('No reading: a reference fit does not reproduce logistic v4 (gates above).')
    elif fired:
        lines.append(f"No reading: DEC-23's stop rule fires on {', '.join(sorted(fired))} for {_label(result, inv)} "
                     '(stop rules above).')
    elif flags:
        lines.append('No reading: an added-input contrast on a fit the primary rests on has an interval entirely below '
                     'zero (' + '; '.join(f"{_side(a['full'])} − {_side(a['reduced'])}" for a in flags) + '), a fitting flag.')
    elif high < 0:
        lines.append('No reading: the interval lies entirely below zero, so adding the partner block reliably lowered '
                     'held-out log-likelihood. That signals a fitting problem (optimisation or regularisation), not a finding.')
    elif primary['detected']:
        durations = by_name.get((inv, ('Dur', primary['full'][1]), ('DurA', primary['full'][1])))
        if durations is None or rules['DurA']['dec23']['fired']:
            where = 'Whether it lives in the partner\'s durations (Dur − DurA) is not stated (that contrast is unavailable or a stop rule fires on DurA).'
        elif durations['detected']:
            where = (f"It lives at least partly in the partner's durations: Dur − DurA in {primary['full'][1]} is detectable "
                     f"({durations['components']['log_likelihood_bits']['mean_bits_per_coda']:.5f} bits/coda, CI "
                     f"{format_ci(durations['components']['log_likelihood_bits']['ci_bits_per_coda'])}, Holm-adjusted p = "
                     f"{durations['components']['log_likelihood_bits']['holm_p']:.4f}).")
        else:
            where = (f"It is not shown to live in the partner's durations: Dur − DurA in {primary['full'][1]} is not "
                     f"detectable ({durations['components']['log_likelihood_bits']['mean_bits_per_coda']:.5f} bits/coda, CI "
                     f"{format_ci(durations['components']['log_likelihood_bits']['ci_bits_per_coda'])}, Holm-adjusted p = "
                     f"{durations['components']['log_likelihood_bits']['holm_p']:.4f}).")
        parts = [name for key, name in (('repeat_bits', 'repeat'), ('switch_choice_bits', 'switch'))
                 if primary['components'][key]['ci_bits_per_coda'][0] > 0]
        part = (f"It lives in the {' and the '.join(parts)} part{'s' if len(parts) > 1 else ''} (interval above zero)."
                if parts else 'Neither the repeat nor the switch part alone has an interval above zero.')
        lines.append("Primary positive: with continuous durations visible, the partner's codas carry usable information "
                     "about the coda's rhythm and tempo class beyond own history and S. " + where + ' ' + part +
                     " Onset-available inputs can't separate coordination from both whales responding to the same context.")
    else:
        why = ('interval above zero but p ≥ 0.05' if low > 0 else 'interval includes zero')
        lines.append(f"Primary null ({why}): the partner null holds with continuous durations visible, at this data size. "
                     "Not detectable at this data size; no calibration is run, and limited data and capacity push contrasts "
                     "toward zero (PLAN §9).")
    lines.append('Secondary, pre-specified (Holm across the five; numbers only, no reading): ' + '; '.join(
        _tested_text(result, t) for t in result['tested'][1:]) + '.')
    rubato = []
    for i in (result['descriptive'] if all(g['passed'] for g in result['gates'].values()) else []):
        if any(result['stop_rules'][i][v]['dec23']['fired'] for v in (reference, 'DurA')):
            continue
        for row in result['descriptive'][i]:
            if row['name'] == f'DurA − {reference}' and row['components']['log_likelihood_bits']['ci_bits_per_coda'][0] > 0:
                rubato.append(f"{_label(result, i)}, {row['full'][1]}")
    lines.append("DurA − R (descriptive, no test or multiplicity control): " + (
        f"interval entirely above zero on {'; '.join(rubato)}. DEC-29's reading: the whale's own tempo drift carries "
        "information beyond its labels (on D and E′, its tempo classes): rubato." if rubato else
        "no interval lies entirely above zero, so DEC-29's rubato reading does not apply (not detectable at this data size)."))
    return lines


CHECK_LABELS = {
    'rows': 'Rows in rq1_targets.jsonl', 'comparison_targets': 'Comparison targets',
    'history_references': 'References in the last 3 lags of A_history_ids and B_history_ids (comparison targets)',
    'duration_vs_raw': 'max |offset_s − onset_s − raw Duration| (all rows)',
    'own_lags': 'Targets with own lag 1 / 2 / 3', 'partner_lags': 'Targets with partner lag 1 / 2 / 3',
    'target_log_duration': 'Target ln d: mean; population SD',
    'lag1_r': 'Pearson r of target ln d with own lag-1 ln d; with partner lag-1 ln d',
    'lag1_r_within_label': 'The same, each variable centred on its mean over the 1,547 rows with its coda\'s label (Ct)',
}


def _value(value):
    if isinstance(value, float):
        return f'{value:.4g}' if abs(value) < 1e-3 and value != 0 else f'{value:.4f}'
    if isinstance(value, list):
        return ' / '.join(_value(v) for v in value) if value else '[]'
    return str(value)


def _fields(d):
    return '; '.join(f'{k} {_value(v)}' for k, v in d.items())


def _stat(stats):
    return (f"{stats['mean_bits_per_coda']:.5f} | {stats['unweighted_mean_bits_per_coda']:.5f} | "
            f"{format_ci(stats['ci_bits_per_coda'])} | {stats['positive_days']}/{stats['days']}")


def _norms(stats):
    return '—' if stats is None else f"{stats['min']:.3f} / {stats['median']:.3f} / {stats['max']:.3f}"


def report_lines(result, config):
    reference, order = result['reference_variant'], list(result['inventories'])
    env, boot = result['fit']['environment'], result['bootstrap']
    labels = {inv: _label(result, inv) for inv in order}
    variant_text = '; '.join(f"{v}: {', '.join(d) if d else 'none (logistic v4)'}" for v, d in result['variants'].items())
    reuse_ok = all(c['identical'] for c in result['fit']['reuse_checks'].values())
    same_ok = all(all(v.values()) for v in result['fit']['same_inputs'].values())
    lines = ['# RQ2 rubato inputs (PLAN [R2-9], DEC-29)', '',
             "Logistic v4 unchanged on inventories " + ', '.join(f"{labels[i]} ({result['inventories'][i]['description']})"
                                                                for i in order) +
             ": each inventory's committed v4 configuration (targets, folds, UNK policy, S, copy weights, the L2 grid and "
             "selection rule, the joint history and S penalties, convergence handling), plus the previous codas' durations "
             "as continuous inputs in the history blocks. Own: for lags 1–3 of A_history, the coda's log duration ln d "
             "(d = offset_s − onset_s), one column per lag, standardised on the column's non-missing training values and 0 "
             "when the lag is missing, with per-class weights like the lag one-hots, in the A block under the history "
             "penalty; no missing indicators (a missing lag already has all-zero one-hots). Partner: the same for lags 1–3 "
             "of B_history (the partner's codas completed before the target's onset, [D5]), in the B block. Durations come "
             "from `rq1_targets.jsonl` by history coda ID; every referenced coda ends at or before the target's onset.", '',
             f"Variants (duration inputs): {variant_text}. All fitted in one environment (Python {env['python']}, NumPy "
             f"{env['numpy']}, SciPy {env['scipy']}, one BLAS thread per fitting process). DurA fits the conditions with A "
             "and Dur those with A or B; the other conditions reuse R's fits, after a refit on fold "
             f"{config['reuse_check_fold']} {'gave identical' if reuse_ok else 'did NOT give identical'} penalty searches, "
             "models and test predictions. Dur's A and A_S have DurA's inputs; both were fitted and "
             f"{'agree exactly on every fold' if same_ok else 'do NOT agree exactly'}. Raw log-likelihoods are not compared "
             "across inventories (PLAN [R2-3]); every contrast pairs two fits of one inventory.", '',
             f"Contrasts are paired per coda, with the [P1] repeat and switch parts; intervals are day-bootstrap "
             f"({boot['replicates']:,} replicates, seed {boot['seed']}); the unweighted day mean is the mean of the 14 "
             "day-level contrasts.", '']

    checks = result['checks']
    lines += ['## Data checks (no models)', '',
              f"Against values computed independently before the run ({checks['source']}): counts exactly, other "
              f"statistics within {checks['tolerance']:g} (3 decimals); {sum(c['passed'] for c in checks['checks'])} of "
              f"{len(checks['checks'])} agree" + ('.' if checks['passed'] else '; the rows marked DIFFERS do not.'), '',
              'Inventory files (data/processed): every inventory has the same rows as `rq1_targets.jsonl` in the same order, '
              'with the same coda_id, onset_s, offset_s, comparison_eligible, A_history_ids and B_history_ids; only label '
              'fields differ. Fold manifests: the same folds, days and comparison IDs.', '',
              '| Check | Computed | Expected | Result |', '|---|---|---|---|']
    for c in checks['checks']:
        status = 'agrees' if c['passed'] else f"DIFFERS ({', '.join(c['failed_fields'])})"
        name = c['name']
        label = CHECK_LABELS.get(name) or (f"Inventory {labels.get(name.split(':')[1], name.split(':')[1])}: "
                                           + ('targets file against rq1_targets.jsonl' if name.startswith('inventory')
                                              else 'fold manifest against inventory A\'s'))
        computed = dict(c['computed'])
        quantity = checks['quantities'].get(name, {})
        if 'label_fields_differing' in quantity:
            computed['label_fields_differing'] = quantity['label_fields_differing']
        lines.append(f"| {label} | {_fields(computed)} | {_fields(c['expected'])} | {status} |")

    lines += ['', f'## Gates: {reference} here against each committed logistic v4 run', '',
              "DEC-25's amended gate: chosen penalties identical (at most one differing) and every condition's mean "
              "log-likelihood within 1e-5 bits/coda of v4's; predicted probabilities are reported, not enforced (optimizer "
              'tolerance and BLAS threads move them by up to about 1e-4 at the smallest penalties).', '',
              '| Inventory | Fits compared | Penalties differing | Max abs LL difference (bits/coda) | Max abs probability '
              'difference | Same penalty: max abs probability difference | Fits above 1e-5 | Gate |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for inv in order:
        g = result['gates'][inv]
        lines.append(f"| {labels[inv]} | {g['fits_compared']} | {len(g['penalty_differences'])} | "
                     f"{g['max_abs_log_likelihood_difference_bits']:.2g} | {g['max_abs_probability_difference']:.3g} | "
                     f"{g['max_abs_probability_difference_same_penalty']:.3g} | {len(g['fits_above_probability_threshold'])} | "
                     f"{'passed' if g['passed'] else 'FAILED'} |")
    differing = [(inv, d) for inv in order for d in result['gates'][inv]['penalty_differences']]
    if differing:
        lines += ['', "Fits whose chosen penalty (history, S) differs from v4's:", '',
                  '| Inventory | Fold | Condition | Chosen here | v4 chose | Max abs probability difference |',
                  '|---|---|---|---|---|---:|']
        lines += [f"| {labels[inv]} | {d['fold_id']} | {d['condition']} | {d['chosen_l2']}, {d['chosen_l2_s']} | "
                  f"{d['v4_chosen_l2']}, {d['v4_chosen_l2_s']} | {d['max_abs_probability_difference']:.3g} |" for inv, d in differing]
    nonconverged = result['nonconverged_candidates']
    lines += ['', 'Penalty candidates that failed the convergence check (left out of the selection where the inventory\'s v4 '
              'configuration skips them, DEC-22): ' + '; '.join(f"{labels[k.split(':')[0]]} {k.split(':')[1]} {n}"
                                                                 for k, n in nonconverged.items()) + '.']
    lines += [line for note in config.get('process_notes', []) for line in ('', note)]

    lines += ['', '## Log-likelihood and accuracy', '',
              'Held-out log-likelihood (bits/coda; comparable within an inventory only), accuracy, macro-F1, and the '
              'difference to repeat-own-last with its day-bootstrap CI, for A and A,B,S. Every condition\'s numbers are in '
              '`summary.json` (`metrics`).', '',
              '| Inventory | Variant | A LL | A accuracy | A macro-F1 | A − repeat-own-last [CI] | A,B,S LL | A,B,S accuracy | '
              'A,B,S macro-F1 | A,B,S − repeat-own-last [CI] |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for inv in order:
        for v in result['variants']:
            cells = []
            for c in ('A', 'A_B_S'):
                m = result['metrics'][inv][v][c]
                rep = m['versus_repeat_own_last']
                cells.append(f"{m['mean_log_likelihood_bits']:.5f} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} | "
                             f"{rep['mean_bits_per_coda']:.5f} {format_ci(rep['ci_bits_per_coda'])}")
            lines.append(f"| {labels[inv]} | {v} | " + ' | '.join(cells) + ' |')

    lines += ['', '## Contrasts', '',
              'Primary (pre-specified): the partner contrast LL(A,B,S) − LL(A,S) under Dur on inventory D, with the exact '
              'two-sided sign-flip p (DEC-23: all 2^14 sign assignments of the day sums, token-weighted mean as the '
              'statistic). Secondary (pre-specified), each with the two-sided sign-flip p and Holm\'s correction across the '
              'five: the partner contrast under Dur on A and E′; the partner\'s durations beyond everything else, '
              'LL_Dur(A,B,S) − LL_DurA(A,B,S), on A, D and E′. Detected (for the Reading): interval entirely above zero '
              'and p < 0.05 (two-sided for the primary, Holm-adjusted for the secondaries), the rule of DEC-30 and DEC-31.', '',
              '| Role | Inventory | Contrast | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part [CI] | '
              'Switch part [CI] | Two-sided p | Holm p | Detected |', '|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|']
    for t in result['tested']:
        s = t['components']['log_likelihood_bits']
        lines.append(f"| **{t['role']}** | {labels[t['inventory']]} | {t['name']}: {_contrast_name(t)} | {_stat(s)} | "
                     f"{_part(t['components']['repeat_bits'])} | {_part(t['components']['switch_choice_bits'])} | "
                     f"{_p(s['sign_flip_p_two_sided'])} | {_p(s.get('holm_p'))} | {'yes' if t['detected'] else 'no'} |")
    lines += ['', 'Descriptive (no tests): DurA − R and Dur − R in A and A,B,S; the own and timing contrasts under Dur; the '
              'partner contrast under R.', '',
              '| Inventory | Contrast | Bits/coda | Unweighted day mean | 95% CI | Positive days | Repeat part [CI] | '
              'Switch part [CI] |', '|---|---|---:|---:|---|---:|---:|---:|']
    for inv in order:
        for row in result['descriptive'][inv]:
            comp = row['components']
            lines.append(f"| {labels[inv]} | {row['name']}: {_side(row['full'])} − {_side(row['reduced'])} | "
                         f"{_stat(comp['log_likelihood_bits'])} | {_part(comp['repeat_bits'])} | "
                         f"{_part(comp['switch_choice_bits'])} |")

    lines += ['', '## The partner contrast under R and under Dur', '',
              'LL(A,B,S) − LL(A,S) within each variant, and its per-coda change from R to Dur, '
              '[LL_Dur(A,B,S) − LL_Dur(A,S)] − [LL_R(A,B,S) − LL_R(A,S)], with its day-bootstrap CI (descriptive).', '',
              '| Inventory | Under R [CI] | +days | Under Dur [CI] | +days | Change R → Dur | Change CI | Change +days |',
              '|---|---:|---:|---:|---:|---:|---|---:|']
    for inv in order:
        pc = result['partner_change'][inv]
        r, d, ch = (pc[k]['log_likelihood_bits'] for k in ('under_r', 'under_dur', 'change'))
        lines.append(f"| {labels[inv]} | {_part(r)} | {r['positive_days']}/{r['days']} | {_part(d)} | "
                     f"{d['positive_days']}/{d['days']} | {ch['mean_bits_per_coda']:.5f} | {format_ci(ch['ci_bits_per_coda'])} | "
                     f"{ch['positive_days']}/{ch['days']} |")

    lines += ['', '## The partner contrast under Dur by turn category (no tests)', '',
              'Token-weighted means within each turn category of the target ([D5]); positive days among the days with '
              'targets in that category.', '',
              '| Inventory | Turn category | Targets | Bits/coda | Repeat part | Switch part | Positive days |',
              '|---|---|---:|---:|---:|---:|---:|']
    for inv in order:
        for category, b in result['partner_by_turn_category'][inv].items():
            lines.append(f"| {labels[inv]} | {category} | {b['codas']} | {b['mean_log_likelihood_bits']:.5f} | "
                         f"{b['mean_repeat_bits']:.5f} | {b['mean_switch_choice_bits']:.5f} | {b['positive_days']}/{b['days']} |")

    lines += ['', '## Fitted duration weights', '',
              'Min / median / max over the 14 folds of the L2 norm of the own (A_log_duration, lags 1–3 × classes) and '
              'partner (B_log_duration) duration weights at the chosen penalties, per standard deviation of the lag\'s '
              'training log durations. Per-fold weights are in each variant\'s `coefficients.jsonl`.', '',
              '| Inventory | Variant | A: own | A,B,S: own | A,B,S: partner |', '|---|---|---:|---:|---:|']
    for inv in order:
        for v, by_condition in result['duration_weight_norms'][inv].items():
            lines.append(f"| {labels[inv]} | {v} | {_norms(by_condition['A']['A_log_duration'])} | "
                         f"{_norms(by_condition['A_B_S']['A_log_duration'])} | {_norms(by_condition['A_B_S']['B_log_duration'])} |")

    lines += ['', '## Stop rules (DEC-23, with DEC-16 beside it) and fitting flags', '',
              "Each inventory and variant is a full run of the 16 checks: the 12 add-one-input contrasts and the 4 "
              "A-conditions against repeat-own-last. DEC-23's rule gives each check an exact one-sided sign-flip test "
              "(alternative: mean below zero) on its day sums, with Holm's correction across the 16, and fires when an "
              "adjusted p is below 0.05 (`logistic_v5.dec23_stop_rules`; the day sums are each run's stored day means times "
              "the day's target count, as in the [P2] sweep). DEC-16's rule, beside it, fires when a check's day-bootstrap "
              'interval lies entirely below zero.', '',
              "| Inventory | Variant | DEC-23: Holm-adjusted p < 0.05 | Smallest one-sided p, Holm-adjusted (check) | DEC-16: "
              "interval entirely below zero |", '|---|---|---|---|---|']
    for inv in order:
        for v, rules in result['stop_rules'][inv].items():
            dec23 = rules['dec23']
            fired = [f"{r['check']} (p = {r['sign_flip_p']:.4f}, Holm {r['holm_p']:.4f})" for r in dec23['checks'] if r['fires']]
            smallest = min(dec23['checks'], key=lambda r: r['sign_flip_p'])
            lines.append(f"| {labels[inv]} | {v} | {'; '.join(fired) if fired else 'no stop rule fires'} | "
                         f"{smallest['sign_flip_p']:.4f}, Holm {smallest['holm_p']:.4f} ({smallest['check']}) | "
                         f"{'; '.join(rules['dec16']) if rules['dec16'] else 'no stop rule fires'} |")
    lines += ['', "Added-input contrasts (DEC-29's fitting flag: an interval entirely below zero is a fitting flag and is not "
              'interpreted), each in the conditions where the two variants differ:', '',
              '| Inventory | Contrast | Bits/coda | 95% CI | Positive days | Fitting flag |', '|---|---|---:|---|---:|---|']
    for inv in order:
        for a in result['added_inputs'][inv]:
            s = a['components']['log_likelihood_bits']
            lines.append(f"| {labels[inv]} | {_side(a['full'])} − {_side(a['reduced'])} | {s['mean_bits_per_coda']:.5f} | "
                         f"{format_ci(s['ci_bits_per_coda'])} | {s['positive_days']}/{s['days']} | "
                         f"{'yes' if a['fitting_flag'] else 'no'} |")
    flagged = [f"{labels[inv]}: {_side(a['full'])} − {_side(a['reduced'])}" for inv in order for a in result['added_inputs'][inv]
               if a['fitting_flag']]
    lines += ['', f"Fitting flags: {'; '.join(flagged)}." if flagged else 'No added-input contrast has an interval entirely '
              'below zero (no fitting flag).']

    lines += ['', '## Reading (DEC-29)', ''] + [f'- {line}' for line in result['reading']]
    seconds = result['fit']
    lines += ['', '## Limits', '',
              "- The partner's coda that overlaps the target is not complete at the target's onset, so it is not a B input "
              '([D5]); duration matching within overlaps (Sharma et al. 2024a; Assayag et al. 2026) is outside this design.',
              '- Added after the results freeze (2026-10-04) as a final run (DEC-29); the descriptive correlations that '
              'motivated it were seen first. The inputs, variants, grid and rules were fixed before any fit.',
              '- No calibration (planted signal) was run, so a null says only "not detectable at this data size".',
              "- The duration columns share one history penalty with the lag one-hots, the other history block and the copy "
              "weights (v4's rule), so a null may partly reflect that compromise.",
              '- Day-bootstrap intervals are approximate with only 14 days; the switch part rests on the non-repeat codas and '
              'is exploratory.',
              '- Claims are about dyadic exchanges only; nothing here is evidence of grammar or syntax.', '',
              f"Fit stage: {seconds['total_seconds'] / 60:.1f} min wall time on {seconds['workers']} processes (R on all "
              f"inventories {seconds['wall_seconds']['R'] / 60:.1f} min; then " + ', '.join(
                  f"{labels[i]} {seconds['wall_seconds'][i] / 60:.1f} min" for i in order) + ')' + (
                  f"; after {labels[seconds['estimate']['after']]}, the estimate for the rest was "
                  f"{seconds['estimate']['seconds'] / 60:.1f} min ({seconds['estimate']['rule']})" if seconds['estimate'] else '')
              + '. Outputs: '
              f"`{config['output_directory']}/checks/summary.json`, `fit/summary.json`, per inventory and variant "
              "`fit/<inventory>/<variant>/` (summary.json, l2_selection.jsonl, scored.jsonl, coefficients.jsonl), and "
              f"`{config['output_directory']}/summary.json`, which records the SHA-256 of every input to this report.", '']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='config/rq2_rubato_inputs.json')
    parser.add_argument('--stage', choices=('all',) + STAGES, default='all')
    # Each process imports torch through residual.py (about 230 MB before any data).
    parser.add_argument('--workers', type=int, default=18, help='processes for the fit stage (default 18)')
    args = parser.parse_args()
    if args.stage in ('all', 'checks'):
        run_checks(args.config)
    if args.stage in ('all', 'fit'):
        run_fit(args.config, args.workers)
    if args.stage in ('all', 'report'):
        result = run_report(args.config)
        print('\n'.join(result['reading']))


if __name__ == '__main__':
    main()
