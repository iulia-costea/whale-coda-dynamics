"""Logistic v5 (DEC-23, PLAN [M4]): every input block can be absent, chosen on the validation day.

For each fold and condition c the candidates are logistic v4's chosen fits of every sub-condition
s of c (each of A, B, S on or absent; all absent = condition none, the fold's smoothed prior). The
candidate with the highest validation-day log-likelihood is chosen (exact ties: fewer blocks, then
the block order A, B, S), and its stored v4 test-day predictions become c's v5 predictions. Each
sub-condition's penalties were already chosen by v4's rule, so nothing is fitted: v5 is computed
from a v4 run's committed models.jsonl, l2_selection.jsonl and predictions.jsonl. Sensitivity:
DEC-21's one-SE rule (the fewest blocks within one standard error of the best).

Stop rules (DEC-23), reported for v4, v5 and the one-SE variant: the 16 checks (12 add-one-input
contrasts, 4 A-conditions against repeat-own-last) each get an exact one-sided sign-flip test on
the day sums of per-coda differences, with Holm's correction across the 16; a check fires when its
adjusted p is below alpha. DEC-16's old rule (day-bootstrap CI entirely below zero) is reported
next to it. The three [M2] contrasts also get an exact two-sided sign-flip p.

Gates, checked before any test-day number is computed; nothing is written if one fails:
1. reproduction: every v4 chosen fit's validation log-likelihood, recomputed from models.jsonl
   (stored encoder parameters, weights, beta, gamma), equals l2_selection.jsonl's recorded value to
   the configured tolerance, and its test-day probabilities equal predictions.jsonl's (the prior's
   too, for condition none);
2. validation monotonicity: under the max rule every add-one-input contrast is >= 0 on every
   validation day (true by construction; checked anyway).

select_max, select_one_se, subconditions and dec23_stop_rules are generic over block names, so
other model families with absent-block candidates (DEC-24) can reuse them.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import scipy

from .evaluation import evaluate, classification_metrics, day_bootstrap_bits, sign_flip_p, holm
from .logistic import Encoder, last_label_index, probabilities, check_stop_conditions, format_ci
from .preprocessing.folds import prepare_fold
from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text

BLOCKS = ('A', 'B', 'S')
A_CONDITIONS = ('A', 'A_B', 'A_S', 'A_B_S')
M2 = {'own_history': ('A_B_S', 'B_S'), 'partner_content': ('A_B_S', 'A_S'), 'timing': ('A_B_S', 'A_B')}
RULES = ('max', 'one_se')
# Recomputing log2 of a stored probability can differ from a stored log-likelihood in the last bit when
# the two were computed on different machines (libm); per-coda log-likelihoods are compared to this.
LOG_LIKELIHOOD_TOLERANCE_BITS = 1e-12
MODELS = (('v4', 'v4'), ('v5', 'v5 (max)'), ('v5_one_se', 'v5 (one-SE)'))


# ---------------------------------------------------------------- generic selection

def condition_blocks(condition):
    return () if condition == 'none' else tuple(condition.split('_'))


def condition_name(blocks, order=BLOCKS):
    return '_'.join(sorted(blocks, key=order.index)) or 'none'


def subconditions(blocks, order=BLOCKS):
    """Every subset of `blocks`, each a tuple in `order`: fewest blocks first, then by `order`."""
    blocks = sorted(blocks, key=order.index)
    return [tuple(c) for k in range(len(blocks) + 1) for c in itertools.combinations(blocks, k)]


def _simplicity(blocks, order):
    return len(blocks), [order.index(b) for b in blocks]


def _validated(candidates):
    lengths = {len(per_target) for _, per_target in candidates.values()}
    if not candidates or len(lengths) != 1 or not lengths.pop():
        raise ValueError('Candidates need per-target validation log-likelihoods on the same nonempty targets')
    if any(not np.all(np.isfinite(per_target)) for _, per_target in candidates.values()):
        raise ValueError('Nonfinite validation log-likelihood')
    return {name: float(np.mean(per_target)) for name, (_, per_target) in candidates.items()}


def select_max(candidates, order=BLOCKS):
    """DEC-23's rule. `candidates` maps a name to (blocks, per-target validation log-likelihoods in
    bits). The highest mean wins; exact ties go to fewer blocks, then to `order`."""
    means = _validated(candidates)
    chosen = min(candidates, key=lambda name: (-means[name], _simplicity(candidates[name][0], order)))
    return dict(chosen=chosen, validation_log_likelihood_bits=means)


def select_one_se(candidates, order=BLOCKS):
    """DEC-21's one-SE rule as DEC-23's sensitivity: the best candidate is select_max's; a candidate is
    within one SE when its mean paired difference to the best is at most sd(differences, ddof=1)/sqrt(n);
    the chosen candidate has the fewest blocks among those within, then the higher mean, then `order`."""
    means = _validated(candidates)
    n = len(next(iter(candidates.values()))[1])
    if n < 2:
        raise ValueError('The one-SE rule needs at least two validation targets')
    best = select_max(candidates, order)['chosen']
    table = {}
    for name, (blocks, per_target) in candidates.items():
        d = candidates[best][1] - per_target
        mean_d, se = float(np.mean(d)), float(np.std(d, ddof=1) / math.sqrt(n))
        table[name] = dict(mean_difference_to_best_bits=mean_d, standard_error_bits=se, within_one_se=mean_d <= se)
    chosen = min((name for name in candidates if table[name]['within_one_se']),
                 key=lambda name: (len(candidates[name][0]), -means[name], _simplicity(candidates[name][0], order)))
    return dict(chosen=chosen, best=best, validation_log_likelihood_bits=means, one_se=table)


def add_input_pairs(conditions):
    """(bigger, smaller) for every pair of conditions that differ by exactly one block."""
    blocks = {c: set(condition_blocks(c)) for c in conditions}
    return [(big, small) for big in conditions for small in conditions
            if blocks[small] < blocks[big] and len(blocks[big] - blocks[small]) == 1]


def day_sums(values, dates):
    """Per-day exactly rounded sums and token counts of per-coda values keyed by coda ID."""
    grouped = defaultdict(list)
    for coda_id, value in values.items():
        grouped[dates[coda_id]].append(value)
    return {d: math.fsum(v) for d, v in grouped.items()}, {d: len(v) for d, v in grouped.items()}


def dec23_stop_rules(checks, dates, alpha=0.05):
    """DEC-23's stop rule on a family of checks. `checks` maps a check name to per-coda differences
    (coda ID -> bits), `dates` maps coda ID to its test day. Each check gets an exact one-sided
    sign-flip p (alternative: token-weighted mean below zero) on its day sums; Holm's correction runs
    across the family; a check fires when its adjusted p is below alpha."""
    if not checks or not 0 < alpha < 1:
        raise ValueError('Need at least one check and alpha in (0, 1)')
    rows = []
    for name, values in checks.items():
        sums, counts = day_sums(values, dates)
        rows.append(dict(check=name, mean_bits_per_coda=math.fsum(sums.values()) / sum(counts.values()),
                         days=len(sums), sign_flip_p=sign_flip_p(sums, counts, 'less')))
    for row, adjusted in zip(rows, holm([r['sign_flip_p'] for r in rows])):
        row.update(holm_p=adjusted, fires=adjusted < alpha)
    return dict(alpha=alpha, test='exact one-sided sign-flip on day sums (mean below zero), Holm across the family',
                checks=rows, fired=[r['check'] for r in rows if r['fires']])


# ---------------------------------------------------------------- v4 fits

def stored_fit_probabilities(record, rows, vocab):
    """Probabilities of a stored v2-v4 fit (one models.jsonl row) for `rows`, from its stored encoder
    parameters, weights and copy weights; nothing is refitted."""
    if record['vocabulary'] != vocab:
        raise ValueError(f"Stored vocabulary differs from the fold's ({record['fold_id']}, {record['condition']})")
    fields = record['fields']
    if (record['beta'] is None) == ('A_history' in fields) or (record['gamma'] is None) == ('B_history' in fields):
        raise ValueError('Stored copy weights do not match the condition fields')
    encoder = Encoder.from_record(record)
    own = last_label_index(rows, 'A_history', vocab) if 'A_history' in fields else None
    partner = last_label_index(rows, 'B_history', vocab) if 'B_history' in fields else None
    return probabilities(encoder.transform(rows), np.asarray(record['weights'], dtype=float),
                         record['beta'], record['gamma'], own, partner)


def recorded_validation_log_likelihood(selection):
    """The validation log-likelihood v4 recorded for its chosen penalties (one l2_selection.jsonl row)."""
    chosen = [g for g in selection['grid'] if g['l2'] == selection['chosen_l2'] and g['l2_s'] == selection['chosen_l2_s']
              and 'nonconverged' not in g]
    if len(chosen) != 1:
        raise ValueError(f"Chosen v4 penalty not found exactly once ({selection['fold_id']}, {selection['condition']})")
    return chosen[0]['validation_log_likelihood_bits']


def per_target_bits(p, y):
    return np.log2(p[np.arange(len(y)), y])


def prepare_folds(rows, manifest):
    """Each fold's mapped comparison rows per split, checked against the manifest."""
    prepared = []
    for fold in manifest['folds']:
        mapped, policy = prepare_fold(rows, fold, manifest['preparation_configuration'])
        if policy != fold['vocabulary_policy']:
            raise ValueError('Recomputed fold policy differs from manifest')
        splits = {s: [r for r in mapped if r['split'] == s and r['comparison_eligible']] for s in ('validation', 'test')}
        for split, rows_ in splits.items():
            if sorted(r['coda_id'] for r in rows_) != fold['comparison_ids'][split]:
                raise ValueError(f"{fold['fold_id']}: {split} targets differ from the manifest")
        prepared.append(dict(fold=fold, policy=policy, **splits))
    return prepared


def reproduce_v4(prepared, conditions, models, selections, stored, tolerance):
    """Gate 1. Returns per-target validation log-likelihoods {(fold_id, condition): array} for every
    fold and condition (the prior for none) and one diagnostic row per fold and condition; raises if
    any recomputed validation log-likelihood or test probability misses the stored one by more than
    `tolerance`."""
    per_target, rows, failures = {}, [], []
    for p in prepared:
        fold_id, vocab = p['fold']['fold_id'], p['policy']['vocabulary']
        y_val = np.array([vocab.index(r['Ct']) for r in p['validation']])
        for condition in conditions:
            stored_test = np.array([[stored[r['coda_id'], condition][t] for t in vocab] for r in p['test']])
            row = dict(fold_id=fold_id, condition=condition)
            if condition == 'none':
                prior = np.array([p['policy']['label_prior'][t] for t in vocab])
                p_val, p_test = np.tile(prior, (len(p['validation']), 1)), np.tile(prior, (len(p['test']), 1))
            else:
                record, selection = models[fold_id, condition], selections[fold_id, condition]
                if (record['chosen_l2'], record['chosen_l2_s']) != (selection['chosen_l2'], selection['chosen_l2_s']):
                    raise ValueError(f'{fold_id}, {condition}: models.jsonl and l2_selection.jsonl chose different penalties')
                p_val = stored_fit_probabilities(record, p['validation'], vocab)
                p_test = stored_fit_probabilities(record, p['test'], vocab)
                recorded = recorded_validation_log_likelihood(selection)
            bits = per_target_bits(p_val, y_val)
            per_target[fold_id, condition] = bits
            row.update(validation_log_likelihood_bits=float(np.mean(bits)),
                       test_probability_max_abs_difference=float(np.max(np.abs(p_test - stored_test))))
            if condition != 'none':
                row.update(recorded_validation_log_likelihood_bits=recorded,
                           validation_difference_bits=row['validation_log_likelihood_bits'] - recorded)
            if abs(row.get('validation_difference_bits', 0.0)) > tolerance or row['test_probability_max_abs_difference'] > tolerance:
                failures.append(row)
            rows.append(row)
    if failures:
        raise ValueError(f'Gate 1 failed: {len(failures)} fold-condition fits do not reproduce v4 to {tolerance}: {failures}')
    return per_target, rows


def select_all(prepared, conditions, per_target):
    """Selection rows for every fold, condition and rule, and gate 2 (validation monotonicity)."""
    selection, chosen = [], {}
    for p in prepared:
        fold_id = p['fold']['fold_id']
        for condition in conditions:
            candidates = {condition_name(s): (s, per_target[fold_id, condition_name(s)])
                          for s in subconditions(condition_blocks(condition))}
            results = {'max': select_max(candidates), 'one_se': select_one_se(candidates)}
            for rule in RULES:
                result = results[rule]
                chosen[fold_id, condition, rule] = result['chosen']
                table = [dict(subcondition=name, blocks=list(blocks),
                              validation_log_likelihood_bits=result['validation_log_likelihood_bits'][name],
                              **(result['one_se'][name] if rule == 'one_se' else {}))
                         for name, (blocks, _) in candidates.items()]
                ranked = sorted(result['validation_log_likelihood_bits'].values(), reverse=True)
                selection.append(dict(fold_id=fold_id, condition=condition, rule=rule, chosen=result['chosen'],
                                      **({'best': result['best']} if rule == 'one_se' else {}),
                                      validation_targets=len(p['validation']),
                                      chosen_validation_log_likelihood_bits=result['validation_log_likelihood_bits'][result['chosen']],
                                      runner_up_gap_bits=ranked[0] - ranked[1] if len(ranked) > 1 else None,
                                      candidates=table))
    by_key = {(s['fold_id'], s['condition'], s['rule']): s for s in selection}
    contrasts = []
    for p in prepared:
        fold_id = p['fold']['fold_id']
        for big, small in add_input_pairs(conditions):
            value = (by_key[fold_id, big, 'max']['chosen_validation_log_likelihood_bits']
                     - by_key[fold_id, small, 'max']['chosen_validation_log_likelihood_bits'])
            contrasts.append(dict(fold_id=fold_id, pair=f'{big}-{small}', validation_contrast_bits=value))
    negative = [c for c in contrasts if c['validation_contrast_bits'] < 0]
    if negative:
        raise ValueError(f'Gate 2 failed: add-one-input contrasts below zero on the validation day: {negative}')
    gaps = [s for s in selection if s['rule'] == 'max' and s['runner_up_gap_bits'] is not None]
    closest = min(gaps, key=lambda s: s['runner_up_gap_bits'])
    monotonicity = dict(pairs_checked=len(contrasts),
                        min_add_input_validation_contrast_bits=min(c['validation_contrast_bits'] for c in contrasts),
                        closest_runner_up=dict(fold_id=closest['fold_id'], condition=closest['condition'],
                                               gap_bits=closest['runner_up_gap_bits']))
    return selection, chosen, monotonicity


def rebuild_predictions(selection, v4_predictions, rule='max'):
    """v5 predictions from selection.jsonl rows and v4's predictions.jsonl rows: each condition gets
    the stored v4 predictions of its chosen sub-condition on that fold, in selection order."""
    index = defaultdict(list)
    for row in v4_predictions:
        index[row['fold_id'], row['condition']].append(row)
    out = []
    for s in selection:
        if s['rule'] != rule:
            continue
        rows = index[s['fold_id'], s['chosen']]
        if not rows:
            raise ValueError(f"No v4 predictions for {s['fold_id']}, {s['chosen']}")
        out.extend(dict(row, condition=s['condition']) for row in rows)
    return out


# ---------------------------------------------------------------- evaluation and stop rules

def per_coda_ll(scored):
    ll = {}
    for r in scored:
        key = (r['condition'], r['coda_id'])
        if key in ll:
            raise ValueError('Expected one seed per condition and coda')
        ll[key] = r['log_likelihood_bits']
    return ll


def repeat_checks(ll, dates, repeat, evaluation_config):
    """A-conditions minus repeat-own-last, as logistic v2-v4 compute them."""
    out = {}
    for condition in A_CONDITIONS:
        days = defaultdict(list)
        for coda_id in sorted(dates):
            days[dates[coda_id]].append(ll[condition, coda_id] - repeat[coda_id])
        out[condition] = day_bootstrap_bits(days, evaluation_config['bootstrap_replicates'],
                                            evaluation_config['bootstrap_seed'], evaluation_config['confidence'])
    return out


def stop_rules(summary, baseline_metrics, baseline_checks, ll, dates, repeat, alpha):
    """Both rules for one model's evaluation: DEC-16's (CI entirely below zero) and DEC-23's."""
    checks = {name: {c: ll[contrast['full'], c] - ll[contrast['reduced'], c] for c in dates}
              for name, contrast in summary['contrasts'].items() if name.startswith('add_input:')}
    checks.update({f'repeat:{a}': {c: ll[a, c] - repeat[c] for c in dates} for a in A_CONDITIONS})
    new = dec23_stop_rules(checks, dates, alpha)
    for row in new['checks']:
        if row['check'].startswith('repeat:'):
            stats = baseline_checks[row['check'].split(':')[1]]
        else:
            stats = summary['contrasts'][row['check']]['components']['log_likelihood_bits']
        if not math.isclose(row['mean_bits_per_coda'], stats['mean_bits_per_coda'], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{row['check']}: sign-flip input differs from the evaluated contrast")
        row.update(ci_bits_per_coda=stats['ci_bits_per_coda'], unweighted_mean_bits_per_coda=stats['unweighted_mean_bits_per_coda'],
                   positive_days=stats['positive_days'], day_means_bits=stats['day_means_bits'],
                   old_rule_fires=stats['ci_bits_per_coda'][1] < 0)
    return dict(old=check_stop_conditions(summary, baseline_metrics, baseline_checks), new=new)


def m2_sign_flip(ll, dates):
    out = {}
    for name, (big, small) in M2.items():
        sums, counts = day_sums({c: ll[big, c] - ll[small, c] for c in dates}, dates)
        out[name] = sign_flip_p(sums, counts, 'two-sided')
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    v4_config_path = root / config['v4_config']
    v4_config = read_json(v4_config_path)
    manifest_path = root / v4_config['manifest']
    manifest = read_json(manifest_path)
    source = root / manifest['configuration']['input']
    if sha256(source) != manifest['input_sha256']:
        raise ValueError('Stale fold manifest')
    evaluation_config_path = root / v4_config['evaluation_config']
    evaluation_config = read_json(evaluation_config_path)
    if evaluation_config['seeds'] != [v4_config['seed']]:
        raise ValueError('Evaluation seed configuration mismatch')
    v4 = root / v4_config['output_directory']
    paths = dict(config=path, v4_config=v4_config_path, evaluation_config=evaluation_config_path, manifest=manifest_path,
                 source=source, models=v4 / 'models.jsonl', selection=v4 / 'l2_selection.jsonl',
                 predictions=v4 / 'predictions.jsonl', baselines=v4 / 'baseline_predictions.jsonl',
                 summary=v4 / 'evaluation' / 'summary.json', scored=v4 / 'evaluation' / 'scored_predictions.jsonl')
    if config.get('primary_v5_summary'):
        paths['primary_v5_summary'] = root / config['primary_v5_summary']
    v4_summary = read_json(paths['summary'])
    for key in ('source', 'manifest', 'v4_config'):
        rel = paths[key].relative_to(root).as_posix()
        if v4_summary['input_sha256'].get(rel) != sha256(paths[key]):
            raise ValueError(f'The v4 run was not produced from {rel} as committed')
    return root, config, v4_config, manifest, evaluation_config, paths, v4_summary


def run(config_path):
    root, config, v4_config, manifest, evaluation_config, paths, v4_summary = load(config_path)
    conditions = list(manifest['input_conditions'])
    if set(conditions) != {condition_name(s) for s in subconditions(BLOCKS)}:
        raise ValueError('v5 needs the full A/B/S lattice')
    rows = read_jsonl(paths['source'])
    models = {(r['fold_id'], r['condition']): r for r in read_jsonl(paths['models'])}
    selections = {(r['fold_id'], r['condition']): r for r in read_jsonl(paths['selection'])}
    v4_predictions = read_jsonl(paths['predictions'])
    stored = {}
    for r in v4_predictions:
        if (r['coda_id'], r['condition']) in stored:
            raise ValueError('Duplicate v4 prediction')
        stored[r['coda_id'], r['condition']] = r['probabilities']
    prepared = prepare_folds(rows, manifest)

    # Gates 1 and 2 use training-day fits, validation days and stored test probabilities only.
    per_target, reproduction = reproduce_v4(prepared, conditions, models, selections, stored,
                                            config['reproduction_tolerance_bits'])
    selection, chosen, monotonicity = select_all(prepared, conditions, per_target)

    seed = v4_config['seed']
    replicates, bootstrap_seed, confidence = (evaluation_config[k] for k in ('bootstrap_replicates', 'bootstrap_seed', 'confidence'))
    v4_scored = read_jsonl(paths['scored'])
    v4_ll = per_coda_ll(v4_scored)
    dates = {r['coda_id']: r['recording_date'] for r in v4_scored}
    targets = {r['coda_id']: r['target'] for r in v4_scored}
    v4_scored_difference = max(abs(math.log2(probs[targets[coda_id]]) - v4_ll[condition, coda_id])
                               for (coda_id, condition), probs in stored.items())
    if v4_scored_difference > LOG_LIKELIHOOD_TOLERANCE_BITS:
        raise ValueError(f'v4 scored predictions do not match predictions.jsonl ({v4_scored_difference} bits)')
    baselines = read_jsonl(paths['baselines'])
    repeat = {r['coda_id']: r['repeat_log_likelihood_bits'] for r in baselines}
    baseline_metrics = {'majority': classification_metrics([r['target'] for r in baselines], [r['majority'] for r in baselines]),
                        'repeat_own_last': classification_metrics([r['target'] for r in baselines], [r['repeat_prediction'] for r in baselines])}
    # Exactly logistic v2-v4's computation (plain sum in file order), so the metrics compare equal.
    baseline_metrics['repeat_own_last']['mean_log_likelihood_bits'] = sum(r['repeat_log_likelihood_bits'] for r in baselines) / len(baselines)
    if baseline_metrics != v4_summary['baseline_metrics']:
        raise ValueError('Baseline metrics differ from the v4 run')
    fold_of = {c: f['fold_id'] for f in manifest['folds'] for c in f['comparison_ids']['test']}

    results = {'v4': dict(summary=v4_summary, ll=v4_ll, baseline_checks=v4_summary['repeat_baseline_checks'])}
    v5_scored_difference = {}
    for model, rule in (('v5', 'max'), ('v5_one_se', 'one_se')):
        predictions = rebuild_predictions(selection, v4_predictions, rule)
        summary, scored, paired = evaluate(predictions, manifest, rows, [seed], replicates, bootstrap_seed, confidence)
        ll = per_coda_ll(scored)
        v5_scored_difference[model] = max(abs(value - v4_ll[chosen[fold_of[coda_id], condition, rule], coda_id])
                                          for (condition, coda_id), value in ll.items())
        if v5_scored_difference[model] > LOG_LIKELIHOOD_TOLERANCE_BITS:
            raise ValueError(f'{model}: scored log-likelihood differs from v4 for the chosen sub-condition')
        checks = repeat_checks(ll, dates, repeat, evaluation_config)
        results[model] = dict(summary=summary, scored=scored, paired=paired, ll=ll, baseline_checks=checks)
    for model, result in results.items():
        result['stop_rules'] = stop_rules(result['summary'], baseline_metrics, result['baseline_checks'], result['ll'],
                                          dates, repeat, config['alpha'])
        result['m2_sign_flip_two_sided_p'] = m2_sign_flip(result['ll'], dates)
    if results['v4']['stop_rules']['old'] != check_stop_conditions(
            v4_summary, v4_summary['baseline_metrics'], v4_summary['repeat_baseline_checks']):
        raise ValueError('v4 old-rule outcome is not reproducible')
    primary = None
    if 'primary_v5_summary' in paths:
        primary = read_json(paths['primary_v5_summary'])
        for c in conditions:
            if 'S' not in condition_blocks(c) and primary['metrics'][c] != results['v5']['summary']['metrics'][c]:
                raise ValueError(f'Condition {c} does not use S but differs from the primary v5 run')

    counts = {rule: {c: dict(Counter(chosen[p['fold']['fold_id'], c, rule] for p in prepared)) for c in conditions}
              for rule in RULES}
    v5 = results['v5']
    one_se = results['v5_one_se']
    summary = dict(v5['summary'])
    summary.update(
        model='logistic v5 (DEC-23): absent-block candidates chosen on the validation day from v4 fits',
        configuration=config, v4_configuration=v4_config, evaluation_configuration=evaluation_config,
        baseline_metrics=baseline_metrics, repeat_baseline_checks=v5['baseline_checks'],
        gates=dict(reproduction=dict(tolerance_bits=config['reproduction_tolerance_bits'], fits=len(reproduction),
                                     max_abs_validation_difference_bits=max(abs(r.get('validation_difference_bits', 0.0)) for r in reproduction),
                                     max_abs_test_probability_difference=max(r['test_probability_max_abs_difference'] for r in reproduction),
                                     rows=reproduction),
                   validation_monotonicity=monotonicity,
                   scored_consistency=dict(tolerance_bits=LOG_LIKELIHOOD_TOLERANCE_BITS,
                                           v4_scored_vs_v4_predictions_max_abs_difference_bits=v4_scored_difference,
                                           v5_scored_vs_v4_chosen_max_abs_difference_bits=v5_scored_difference)),
        selection_counts=counts,
        stop_rules={model: results[model]['stop_rules'] for model, _ in MODELS},
        m2_sign_flip_two_sided_p={model: results[model]['m2_sign_flip_two_sided_p'] for model, _ in MODELS},
        governing_stop_rule=dict(model=config['governing_model'], rule='DEC-23',
                                 fired=results[config['governing_model']]['stop_rules']['new']['fired']),
        one_se=dict(metrics=one_se['summary']['metrics'], contrasts=one_se['summary']['contrasts'],
                    repeat_baseline_checks=one_se['baseline_checks']),
        versions={'numpy': np.__version__, 'scipy': scipy.__version__},
        input_sha256={p.relative_to(root).as_posix(): sha256(p) for p in paths.values()},
    )
    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'selection.jsonl', selection)
    evaluation_output = output / 'evaluation'
    evaluation_output.mkdir(exist_ok=True)
    write_jsonl(evaluation_output / 'scored_predictions.jsonl', v5['scored'])
    write_jsonl(evaluation_output / 'paired_contrasts.jsonl', v5['paired'])
    write_text(evaluation_output / 'summary.json', json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(report_lines(config, summary, v4_summary, selection, prepared, conditions, primary)) + '\n')
    return summary


# ---------------------------------------------------------------- report

def _bits(value):
    return f'{value:.5f}'


def _p(value):
    return f'{value:.4g}'


def _days(stats):
    """Positive days, and the days whose difference is exactly zero (v5 often picks the same
    sub-condition for both conditions of a contrast on a fold)."""
    zero = sum(v == 0 for v in stats['day_means_bits'].values())
    return f"{stats['positive_days']}/{stats['days']}" + (f" ({zero} at 0)" if zero else '')


def _pattern(stats):
    low, high = stats['ci_bits_per_coda']
    sign = '+' if stats['mean_bits_per_coda'] > 0 else '−' if stats['mean_bits_per_coda'] < 0 else '0'
    return f"{sign}, {'CI > 0' if low > 0 else 'CI < 0' if high < 0 else 'CI spans 0'}"


def _check_label(name):
    if name.startswith('repeat:'):
        return f"{name.split(':')[1]} − repeat-own-last"
    big, small = name.split(':')[1].split('-')
    added = next(b for b in condition_blocks(big) if b not in condition_blocks(small))
    return f'adding {added} to {small} ({big} − {small})'


def report_lines(config, summary, v4_summary, selection, prepared, conditions, primary=None):
    gates = summary['gates']
    reproduction, monotonicity = gates['reproduction'], gates['validation_monotonicity']
    models = {'v4': v4_summary, 'v5': summary, 'v5_one_se': summary['one_se']}
    v4_dir = summary['v4_configuration']['output_directory']
    lines = [f"# {config['report_title']}", '']
    if config.get('headline_note'):
        lines += [f"**{config['headline_note']}**", '']
    intro = ("Logistic v5 is logistic v4 with optional ignorance by selection (DEC-23, PLAN `[M4]`). For each fold and "
             "condition, the candidates are v4's chosen fits of every sub-condition (A, B and S each on or absent; all "
             "absent is the fold's smoothed prior, condition none). The candidate with the highest validation-day "
             "log-likelihood is chosen (exact ties: fewer blocks, then the order A, B, S), and its stored v4 test-day "
             "predictions become the condition's. Each sub-condition's penalties were already chosen by v4's rule, so "
             f"nothing is refitted: everything comes from `{v4_dir}/` (models.jsonl, l2_selection.jsonl, "
             "predictions.jsonl). Sensitivity: DEC-21's one-SE rule (the fewest blocks within one standard error of the "
             "best; SE of the per-target paired validation differences). Every log-likelihood below is in bits per "
             "coda, from the shared evaluation module.")
    if config.get('report_note'):
        intro += f" {config['report_note']}"
    lines += [intro, '']

    lines += ['## Checks before any test-day number', '',
              f"- Reproduction (gate 1): the validation log-likelihoods of all {sum(1 for r in reproduction['rows'] if 'recorded_validation_log_likelihood_bits' in r)} "
              f"v4 fits, recomputed from models.jsonl, match l2_selection.jsonl to "
              f"{reproduction['max_abs_validation_difference_bits']:.3g} bits at most (tolerance {reproduction['tolerance_bits']:g}); "
              f"their test-day probabilities, and the prior's, match predictions.jsonl to "
              f"{reproduction['max_abs_test_probability_difference']:.3g} at most.",
              f"- Validation monotonicity (gate 2): under the max rule all {monotonicity['pairs_checked']} add-one-input "
              f"contrasts (12 pairs × {len(prepared)} validation days) are ≥ 0 on the validation day; the smallest is "
              f"{monotonicity['min_add_input_validation_contrast_bits']:.3g} bits.",
              f"- The smallest validation-day gap between a chosen candidate and the runner-up is "
              f"{monotonicity['closest_runner_up']['gap_bits']:.3g} bits ({monotonicity['closest_runner_up']['fold_id']}, "
              f"condition {monotonicity['closest_runner_up']['condition']}).",
              f"- Every v5 test-day log-likelihood equals v4's for the chosen sub-condition on the same coda to "
              f"{max(gates['scored_consistency']['v5_scored_vs_v4_chosen_max_abs_difference_bits'].values()):.3g} bits at most "
              f"(tolerance {gates['scored_consistency']['tolerance_bits']:g}; recomputing log2 on another machine can change "
              "the last bit).", '']

    governing = summary['governing_stop_rule']
    label = dict(MODELS)
    lines += ['## Stop rules', '',
              "DEC-23's rule: each of the 16 checks (12 add-one-input contrasts, 4 A-conditions against "
              "repeat-own-last) gets an exact one-sided sign-flip test on the day sums of per-coda differences "
              f"(alternative: mean below zero; all 2^{len(prepared)} sign assignments; statistic: token-weighted mean), "
              f"with Holm's correction across the 16; a check fires when its adjusted p < {summary['configuration']['alpha']}. "
              "DEC-16's old rule (day-bootstrap CI entirely below zero) is shown next to it.", '',
              '| Model | DEC-23 rule (sign-flip, Holm) | DEC-16 rule (CI below zero) |', '|---|---|---|']
    for model, name in MODELS:
        rules = summary['stop_rules'][model]
        new = '; '.join(f"{r['check']} (Holm p {_p(r['holm_p'])})" for r in rules['new']['checks'] if r['fires']) or 'none fired'
        old = '; '.join(rules['old']) or 'none fired'
        lines.append(f'| {name} | {new} | {old} |')
    status = ', '.join(governing['fired']) or 'none fired'
    lines += ['', f"Governing status (DEC-23's rule on {label[governing['model']]}): {status}.", '']
    if governing['fired']:
        lines += ['**Stop: a governing stop rule fired.** The numbers below are reported as they are; no conclusion is '
                  'drawn from this run and nothing was changed after seeing them.', '']

    lines += ['## Lattice, v4 and v5', '',
              '| Condition | v4 LL | v5 LL | v5 − v4 | v5 one-SE LL | v4 accuracy | v5 accuracy | v4 macro-F1 | v5 macro-F1 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in conditions:
        a, b, o = v4_summary['metrics'][c], summary['metrics'][c], summary['one_se']['metrics'][c]
        lines.append(f"| {c} | {_bits(a['mean_log_likelihood_bits'])} | {_bits(b['mean_log_likelihood_bits'])} | "
                     f"{_bits(b['mean_log_likelihood_bits'] - a['mean_log_likelihood_bits'])} | {_bits(o['mean_log_likelihood_bits'])} | "
                     f"{a['accuracy']:.3%} | {b['accuracy']:.3%} | {a['macro_f1']:.4f} | {b['macro_f1']:.4f} |")
    for c, m in summary['baseline_metrics'].items():
        ll = _bits(m['mean_log_likelihood_bits']) if 'mean_log_likelihood_bits' in m else 'n/a'
        lines.append(f"| {c} | {ll} | {ll} | — | {ll} | {m['accuracy']:.3%} | {m['accuracy']:.3%} | {m['macro_f1']:.4f} | {m['macro_f1']:.4f} |")

    subs = [condition_name(s) for s in subconditions(BLOCKS)]
    counts = summary['selection_counts']
    lines += ['', '## Sub-conditions chosen', '',
              f"Folds (of {len(prepared)}) on which each sub-condition was chosen: max rule, with the one-SE rule in "
              "parentheses. A dash marks a sub-condition that is not a candidate for that condition.", '',
              '| Condition | ' + ' | '.join(subs) + ' |', '|---|' + '---:|' * len(subs)]
    for c in conditions:
        cells = []
        for s in subs:
            if not set(condition_blocks(s)) <= set(condition_blocks(c)):
                cells.append('—')
            else:
                cells.append(f"{counts['max'][c].get(s, 0)} ({counts['one_se'][c].get(s, 0)})")
        lines.append(f'| {c} | ' + ' | '.join(cells) + ' |')
    by_key = {(s['fold_id'], s['condition'], s['rule']): s['chosen'] for s in selection}
    lines += ['', 'Per fold: the chosen sub-condition under the max rule, with the one-SE choice after an arrow where it differs.', '',
              '| Test day | ' + ' | '.join(c for c in conditions if c != 'none') + ' |',
              '|---|' + '---|' * (len(conditions) - 1)]
    for p in sorted(prepared, key=lambda p: p['fold']['test_day']):
        fold_id = p['fold']['fold_id']
        cells = []
        for c in conditions:
            if c == 'none':
                continue
            a, b = by_key[fold_id, c, 'max'], by_key[fold_id, c, 'one_se']
            cells.append(a if a == b else f'{a} → {b}')
        lines.append(f"| {p['fold']['test_day']} | " + ' | '.join(cells) + ' |')

    lines += ['', '## RQ1 contrasts (`[M2]`)', '',
              'Paired per-coda differences; positive days, with the days whose difference is exactly zero; day-bootstrap 95% CI '
              f"({summary['bootstrap']['replicates']:,} replicates); exact two-sided sign-flip p on the day sums (DEC-23).", '',
              '| Contrast | Model | Token-weighted mean | Unweighted day mean | 95% CI | Positive days | Two-sided sign-flip p |',
              '|---|---|---:|---:|---|---:|---:|']
    for name in M2:
        for model, model_label in MODELS:
            stats = models[model]['contrasts'][name]['components']['log_likelihood_bits']
            lines.append(f"| {name} | {model_label} | {_bits(stats['mean_bits_per_coda'])} | "
                         f"{_bits(stats['unweighted_mean_bits_per_coda'])} | {format_ci(stats['ci_bits_per_coda'])} | "
                         f"{_days(stats)} | {_p(summary['m2_sign_flip_two_sided_p'][model][name])} |")
    lines += ['', 'Sign of each token-weighted mean and where its CI lies:', '',
              '| Contrast | v4 | v5 (max) | v5 (one-SE) |', '|---|---|---|---|']
    differs = []
    for name in M2:
        cells = [_pattern(models[m]['contrasts'][name]['components']['log_likelihood_bits']) for m, _ in MODELS]
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        if cells[0] != cells[1]:
            differs.append(name)
    lines += ['', ('The sign and CI position of every `[M2]` contrast are the same under v4 and v5.' if not differs else
                   f"The sign or CI position differs between v4 and v5 for: {', '.join(differs)}. Both are reported.")]

    lines += ['', '## The sixteen stop-rule checks', '',
              'Per-coda differences, token-weighted means, day-bootstrap CI and positive days (with the days whose '
              'difference is exactly zero); DEC-16: CI entirely below '
              'zero; one-sided sign-flip p (alternative: mean below zero) and its Holm adjustment across the 16; DEC-23: '
              'adjusted p below alpha.']
    for model, model_label in MODELS:
        lines += ['', f'### {model_label}', '',
                  '| Check | Token-weighted mean | 95% CI | Positive days | DEC-16 rule | Sign-flip p | Holm p | DEC-23 rule |',
                  '|---|---:|---|---:|---|---:|---:|---|']
        for r in summary['stop_rules'][model]['new']['checks']:
            lines.append(f"| {_check_label(r['check'])} | {_bits(r['mean_bits_per_coda'])} | {format_ci(r['ci_bits_per_coda'])} | "
                         f"{_days(r)} | {'fires' if r['old_rule_fires'] else '—'} | {_p(r['sign_flip_p'])} | "
                         f"{_p(r['holm_p'])} | {'fires' if r['fires'] else '—'} |")

    if primary is not None:
        lines += ['', '## Primary timing vs the old delay, v5', '',
                  'The timing contrast (A_B_S − A_B) in the primary v5 run and in this run. Conditions without S are '
                  'identical in both runs (checked before this report was written).', '',
                  '| Run | Token-weighted mean | 95% CI | Positive days |', '|---|---:|---|---:|']
        for name, s in (('primary (onset-available timing)', primary), ('this run (old delay)', summary)):
            t = s['contrasts']['timing']['components']['log_likelihood_bits']
            lines.append(f"| {name} | {_bits(t['mean_bits_per_coda'])} | {format_ci(t['ci_bits_per_coda'])} | {_days(t)} |")

    lines += ['', '## Notes', '',
              f"- Day-bootstrap CIs are approximate with {len(prepared)} days, and limited data and capacity push contrasts "
              'toward zero: a CI that spans zero may mean too little data (PLAN §9).',
              '- Conditions with S choose among 81 penalty pairs, those without among 9, so comparing validation maxima '
              'slightly favours keeping S; the one-SE rule counters this (DEC-23).',
              '- The sign-flip test weights days by their number of targets, so the largest days dominate it (DEC-23).',
              '- This report establishes no grammar or causal mechanism.',
              f'- Per fold, condition and rule, the candidates and their validation log-likelihoods are in '
              f"`{summary['configuration']['output_directory']}/selection.jsonl`; v5's predictions are rebuilt from it "
              f"and `{v4_dir}/predictions.jsonl` (`logistic_v5.rebuild_predictions`).", '',
              '## Inputs', '', 'SHA-256 of every input:', '']
    lines += [f'- `{p}`: `{h}`' for p, h in summary['input_sha256'].items()]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/rq1_logistic_v5.json')
    summary = run(parser.parse_args().config)
    for model, name in MODELS:
        rules = summary['stop_rules'][model]
        print(f"{name}: DEC-23 rule fired for {rules['new']['fired'] or 'none'}; DEC-16 rule: {rules['old'] or 'none'}")
    governing = summary['governing_stop_rule']
    if governing['fired']:
        print(f"STOP: DEC-23's rule fired on {governing['model']}: {', '.join(governing['fired'])}. See the report.")


if __name__ == '__main__':
    main()
