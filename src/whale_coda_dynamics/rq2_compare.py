"""RQ2: RQ1's contrasts under seven token inventories on logistic v4 (PLAN [R2-3]-[R2-6], DEC-22).

Raw log-likelihoods of different inventories are never compared: they keep different information
about the same codas. Everything reported here is a difference within one inventory (the three
[M2] contrasts, total predictability T = LL(A_B_S) - LL(none), A-conditions vs repeat-own-last), a
ratio of such differences (shares of T), a difference of shares between inventories, or, for the
nested pairs A -> E and C -> D, the coarse model's log-likelihood against the fine model's
probability of the same coarse label (the exploratory tempo decomposition, [R2-4]).

Uncertainty: one set of day draws per replicate, shared by every inventory, so differences
against A are paired. The draws are made exactly as evaluation.day_bootstrap_bits makes them, and
the run checks that it reproduces each inventory's recorded contrast CIs. No model is fitted.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from .evaluation import log_likelihood_bits
from .logistic import check_stop_conditions
from .preprocessing.io_utils import read_json, read_jsonl, write_text

CONTRASTS = {'own_history': ('A_B_S', 'B_S'), 'partner_content': ('A_B_S', 'A_S'), 'timing': ('A_B_S', 'A_B')}
TOTAL = ('A_B_S', 'none')
A_CONDITIONS = ('A', 'A_B', 'A_S', 'A_B_S')
CONDITIONS = ('none', 'S', 'A', 'B', 'A_B', 'A_S', 'B_S', 'A_B_S')


def shared_day_draws(days, replicates, seed):
    """Resampled day lists, drawn exactly as evaluation.day_bootstrap_bits draws them."""
    if type(replicates) is not int or replicates < 1:
        raise ValueError('Invalid bootstrap settings')
    days = sorted(days)
    rng = random.Random(seed)
    return [rng.choices(days, k=len(days)) for _ in range(replicates)]


def percentile_interval(samples, confidence):
    """Percentile interval with the same interpolation as evaluation.day_bootstrap_bits."""
    samples = sorted(samples)
    def quantile(q):
        position = q * (len(samples) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return samples[lo] + (position - lo) * (samples[hi] - samples[lo])
    tail = (1 - confidence) / 2
    return [quantile(tail), quantile(1 - tail)]


def day_totals(values, dates):
    """Per-day (sum, count) of per-coda values; `values` and `dates` are keyed by coda ID."""
    sums, sizes = defaultdict(list), defaultdict(int)
    for coda_id, value in values.items():
        if not math.isfinite(value):
            raise ValueError('Nonfinite per-coda value')
        sums[dates[coda_id]].append(value)
        sizes[dates[coda_id]] += 1
    return {d: (math.fsum(v), sizes[d]) for d, v in sums.items()}


def replicate_means(totals, draws):
    """Token-weighted mean in every replicate (days absent from `totals` contribute nothing)."""
    result = []
    for selected in draws:
        size = sum(totals[d][1] for d in selected if d in totals)
        if not size:
            raise ValueError('A bootstrap replicate drew no tokens')
        result.append(math.fsum(totals[d][0] for d in selected if d in totals) / size)
    return result


def replicate_ratios(numerator, denominator, draws):
    """Ratio of token-weighted means (= ratio of sums over the same tokens) in every replicate."""
    result = []
    for selected in draws:
        bottom = math.fsum(denominator[d][0] for d in selected if d in denominator)
        if not bottom > 0:
            raise ValueError('Total predictability is not positive in a bootstrap replicate')
        result.append(math.fsum(numerator[d][0] for d in selected if d in numerator) / bottom)
    return result


def describe_values(values, dates, draws, confidence):
    """Token-weighted and unweighted day means, CI from the shared draws, positive days."""
    totals = day_totals(values, dates)
    means = {d: s / n for d, (s, n) in totals.items()}
    return {'mean_bits_per_coda': math.fsum(s for s, _ in totals.values()) / sum(n for _, n in totals.values()),
            'unweighted_mean_bits_per_coda': math.fsum(means.values()) / len(means),
            'ci_bits_per_coda': percentile_interval(replicate_means(totals, draws), confidence),
            'positive_days': sum(s > 0 for s, _ in totals.values()), 'days': len(totals), 'codas': len(values)}


def type_and_tempo_bits(probabilities, target, unk_token):
    """[R2-4]: log2 p(type) summed over the in-vocabulary labels sharing the target's part before
    '|', and log2 p(tempo | type) = log2 p(target) - log2 p(type). UNK mass is left out of the
    type marginal, so the type term is a lower bound. Their sum is log2 p(target) by construction."""
    if target == unk_token:
        raise ValueError('The decomposition needs an in-vocabulary target')
    kind = target.split('|')[0]
    type_bits = log_likelihood_bits(math.fsum(p for label, p in probabilities.items()
                                              if label != unk_token and label.split('|')[0] == kind))
    return type_bits, log_likelihood_bits(probabilities[target]) - type_bits


def sign(value):
    return '+' if value > 0 else '−' if value < 0 else '0'


def reliability(ci):
    return 'CI > 0' if ci[0] > 0 else 'CI < 0' if ci[1] < 0 else 'CI spans 0'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Inventory:
    """One inventory's committed logistic v4 outputs, checked against its manifest."""
    def __init__(self, root, spec):
        self.id, self.spec = spec['id'], spec
        self.results = root / spec['results_directory']
        self.paths = {'manifest': root / spec['manifest'], 'summary': self.results / 'evaluation' / 'summary.json',
                      'scored': self.results / 'evaluation' / 'scored_predictions.jsonl',
                      'selection': self.results / 'l2_selection.jsonl'}
        self.manifest = read_json(self.paths['manifest'])
        self.summary = read_json(self.paths['summary'])
        recorded = self.summary['input_sha256'].get(spec['manifest'])
        if recorded != sha256(self.paths['manifest']):
            raise ValueError(f"{self.id}: results were not produced from {spec['manifest']} as committed")
        self.unk = self.manifest['preparation_configuration']['unk_token']
        self.ll, self.target, self.target_raw, self.dates = {}, {}, {}, {}
        for row in read_jsonl(self.paths['scored']):
            key = (row['condition'], row['coda_id'])
            if key in self.ll:
                raise ValueError(f'{self.id}: more than one seed or duplicate scored row; expected one seed')
            self.ll[key] = row['log_likelihood_bits']
            self.target[row['coda_id']], self.target_raw[row['coda_id']] = row['target'], row['target_raw']
            self.dates[row['coda_id']] = row['recording_date']
        if {c for c, _ in self.ll} != set(CONDITIONS) or len(self.ll) != len(CONDITIONS) * len(self.dates):
            raise ValueError(f'{self.id}: scored predictions do not cover the eight conditions for every target')
        self.stop_reasons = check_stop_conditions(self.summary, self.summary['baseline_metrics'],
                                                  self.summary['repeat_baseline_checks'])
        self.nonconverged = sum(1 for s in read_jsonl(self.paths['selection']) for g in s['grid'] if 'nonconverged' in g)

    def differences(self, big, small, codas=None):
        return {c: self.ll[big, c] - self.ll[small, c] for c in (self.dates if codas is None else codas)}

    def unk_by_fold(self):
        rows = []
        for fold in self.manifest['folds']:
            counts = fold['vocabulary_policy']['mapping_counts']['comparison']
            test_ids = fold['comparison_ids']['test']
            scored_unk = sum(self.target[c] == self.unk for c in test_ids)
            if scored_unk != counts['test']['targets']:
                raise ValueError(f"{self.id}: UNK test targets in {fold['fold_id']} differ between manifest and scores")
            rows.append({'fold_id': fold['fold_id'], 'test_day': fold['test_day'], 'test_targets': len(test_ids),
                         'test_unk': counts['test']['targets'], 'train_targets': fold['comparison_counts']['train'],
                         'train_unk': counts['train']['targets'], 'vocabulary': len(fold['vocabulary_policy']['vocabulary'])})
        return rows


def compare(inventories, reference, draws, confidence, days=None):
    """Contrasts, T, shares and share differences against the reference, all on the same draws.
    `days`, when given, restricts every inventory to those days (no refitting)."""
    result = {}
    for ident, inv in inventories.items():
        codas = [c for c in inv.dates if days is None or inv.dates[c] in days]
        total = inv.differences(*TOTAL, codas)
        entry = {'T': describe_values(total, inv.dates, draws, confidence), 'contrasts': {}, 'shares': {}}
        total_days = day_totals(total, inv.dates)
        for name, (big, small) in CONTRASTS.items():
            values = inv.differences(big, small, codas)
            entry['contrasts'][name] = describe_values(values, inv.dates, draws, confidence)
            totals = day_totals(values, inv.dates)
            samples = replicate_ratios(totals, total_days, draws)
            estimate = math.fsum(s for s, _ in totals.values()) / math.fsum(s for s, _ in total_days.values())
            entry['shares'][name] = {'estimate': estimate, 'ci': percentile_interval(samples, confidence), 'samples': samples}
        result[ident] = entry
    for ident, entry in result.items():
        entry['share_differences_vs_reference'] = {
            name: {'estimate': share['estimate'] - result[reference]['shares'][name]['estimate'],
                   'ci': percentile_interval([a - b for a, b in zip(share['samples'], result[reference]['shares'][name]['samples'])],
                                             confidence)}
            for name, share in entry['shares'].items()} if ident != reference else None
    for entry in result.values():
        for share in entry['shares'].values():
            del share['samples']
    return result


def decomposition(coarse, fine, predictions, draws, confidence):
    """[R2-4] for a nested pair on the targets that are UNK in neither inventory."""
    if set(coarse.dates) != set(fine.dates):
        raise ValueError('Nested pair scores different targets')
    excluded = {'fine_only': 0, 'coarse_only': 0, 'both': 0}
    eligible = []
    for c in sorted(fine.dates):
        f_unk, c_unk = fine.target[c] == fine.unk, coarse.target[c] == coarse.unk
        if f_unk or c_unk:
            excluded['both' if f_unk and c_unk else 'fine_only' if f_unk else 'coarse_only'] += 1
            continue
        if fine.target[c].split('|')[0] != coarse.target_raw[c]:
            raise ValueError(f'{c}: the fine label does not nest the coarse label')
        eligible.append(c)
    keep = set(eligible)
    type_bits, tempo_bits = defaultdict(dict), defaultdict(dict)
    for row in predictions:
        if row['coda_id'] not in keep:
            continue
        condition, c = row['condition'], row['coda_id']
        t, m = type_and_tempo_bits(row['probabilities'], fine.target[c], fine.unk)
        if not math.isclose(t + m, fine.ll[condition, c], rel_tol=0, abs_tol=1e-9):
            raise ValueError(f'{c} {condition}: decomposition does not reproduce the scored log-likelihood')
        type_bits[condition][c], tempo_bits[condition][c] = t, m
    if any(set(type_bits[k]) != keep for k in CONDITIONS):
        raise ValueError('Fine predictions do not cover every eligible target and condition')
    return {'coarse': coarse.id, 'fine': fine.id, 'eligible_targets': len(eligible), 'excluded_targets': excluded,
            'type_minus_coarse': {k: describe_values({c: type_bits[k][c] - coarse.ll[k, c] for c in eligible},
                                                     fine.dates, draws, confidence) for k in CONDITIONS},
            'tempo_contrasts': {name: describe_values({c: tempo_bits[big][c] - tempo_bits[small][c] for c in eligible},
                                                      fine.dates, draws, confidence)
                                for name, (big, small) in CONTRASTS.items()}}


def run(config_path):
    path = Path(config_path).resolve()
    root, config = path.parent.parent, read_json(path)
    replicates, seed, confidence = config['bootstrap_replicates'], config['bootstrap_seed'], config['confidence']
    inventories = {spec['id']: Inventory(root, spec) for spec in config['inventories']}
    reference = config['reference']
    first = inventories[reference]
    for inv in inventories.values():
        if inv.dates != first.dates:
            raise ValueError(f'{inv.id}: targets or days differ from {reference}')
    days = sorted(set(first.dates.values()))
    draws = shared_day_draws(days, replicates, seed)
    main = compare(inventories, reference, draws, confidence)
    for ident, inv in inventories.items():  # the shared draws reproduce each run's own day bootstrap
        if inv.summary['bootstrap'] != {'replicates': replicates, 'seed': seed, 'confidence': confidence,
                                        'method': inv.summary['bootstrap']['method']}:
            raise ValueError(f'{ident}: its evaluation used other bootstrap settings')
        for name, ours in main[ident]['contrasts'].items():
            recorded = inv.summary['contrasts'][name]['components']['log_likelihood_bits']
            for field in ('mean_bits_per_coda', 'unweighted_mean_bits_per_coda'):
                if not math.isclose(ours[field], recorded[field], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f'{ident} {name}: {field} differs from the recorded summary')
            if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(ours['ci_bits_per_coda'], recorded['ci_bits_per_coda'])):
                raise ValueError(f'{ident} {name}: shared draws do not reproduce the recorded CI')
    left_out = config['leave_out_day']
    kept = [d for d in days if d != left_out]
    if len(kept) != len(days) - 1:
        raise ValueError(f'{left_out} is not one of the test days')
    without = compare(inventories, reference, shared_day_draws(kept, replicates, seed), confidence, set(kept))
    nested = []
    for pair in config['nested_pairs']:
        fine = inventories[pair['fine']]
        predictions = read_jsonl(fine.results / 'predictions.jsonl')
        fine.paths['predictions'] = fine.results / 'predictions.jsonl'
        nested.append(decomposition(inventories[pair['coarse']], fine, predictions, draws, confidence))
    summary = {
        'units': 'bits per coda; shares are ratios of token-weighted means',
        'reference': reference, 'days': days, 'left_out_day': left_out,
        'bootstrap': {'replicates': replicates, 'seed': seed, 'confidence': confidence,
                      'method': 'percentile; one set of resampled days per replicate shared by every inventory; '
                                'token-weighted means; shares are ratios within each replicate'},
        'inventories': {ident: {'description': inv.spec['description'], 'results_directory': inv.spec['results_directory'],
                                'report': inv.spec['report'], 'stop_reasons': inv.stop_reasons,
                                'nonconverged_candidates': inv.nonconverged,
                                'repeat_baseline_checks': {c: inv.summary['repeat_baseline_checks'][c] for c in A_CONDITIONS},
                                'unk_by_fold': inv.unk_by_fold(), **main[ident]}
                        for ident, inv in inventories.items()},
        'without_left_out_day': without,
        'pattern': {ident: {name: {'sign': sign(c['mean_bits_per_coda']), 'ci': reliability(c['ci_bits_per_coda'])}
                            for name, c in main[ident]['contrasts'].items()} for ident in inventories},
        'tempo_decomposition': nested,
        'configuration': config,
        'input_sha256': {p.relative_to(root).as_posix(): sha256(p)
                         for p in [path] + [p for inv in inventories.values() for p in inv.paths.values()]},
    }
    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    write_text(output / 'summary.json', json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(report_lines(summary, config)) + '\n')
    return summary


def bits(value):
    return f"{value:.5f}"


def interval(ci, digits=4):
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def report_lines(summary, config):
    inventories = summary['inventories']
    reference = summary['reference']
    stopped = {i for i, inv in inventories.items() if inv['stop_reasons']}
    def mark(ident):
        return f"{ident}‡" if ident in stopped else ident
    any_codas = next(iter(inventories.values()))['T']['codas']
    lines = ['# RQ2: RQ1\'s contrasts under seven token inventories, logistic v4 (DEC-22)', '',
             f"Generated by `rq2_compare.py` (`{config['output_directory']}/summary.json`) from the committed logistic v4 "
             f"runs; inventory {reference} is RQ1's own run. All inventories label the same {any_codas:,} comparison targets, "
             "and targets, folds, S, the UNK policy, the model and its selection rule are RQ1's (PLAN §7). Raw "
             "log-likelihoods are not compared across inventories (PLAN `[R2-3]`): every number below is a difference "
             "within one inventory, a ratio of such differences, or a difference of those ratios. Each inventory's "
             "lattice is in its own report.", '',
             f"Day-bootstrap CIs: {summary['bootstrap']['replicates']:,} replicates (seed {summary['bootstrap']['seed']}); "
             "each replicate's resampled days are shared by every inventory, so differences against "
             f"{reference} are paired. The draws reproduce each run's recorded contrast CIs. With "
             f"{len(summary['days'])} days the CIs are approximate, and limited data and capacity push contrasts toward "
             "zero: a CI that spans zero may mean too little data (PLAN §9).", '',
             '## Inventories and stop rules', '',
             '| ID | Inventory | Report | Stop rules (`[R2-7]`) | Non-converged L2 candidates |', '|---|---|---|---|---:|']
    for ident, inv in inventories.items():
        status = '; '.join(inv['stop_reasons']) or 'none fired'
        lines.append(f"| {mark(ident)} | {inv['description']} | `{inv['report']}` | {status} | {inv['nonconverged_candidates']} |")
    if stopped:
        lines += ['', '‡ A stop rule fired for this inventory: its numbers are shown and no conclusion is drawn from '
                  'them (PLAN `[R2-7]`).']
    lines += ['', '## RQ1 contrasts per inventory (`[M2]`)', '',
              'Bits per coda within each inventory. Their size depends on the inventory, so inventories are compared '
              'through shares of total predictability and the pattern of signs below.', '',
              '| ID | Contrast | Token-weighted mean | Unweighted day mean | 95% CI | Positive days |', '|---|---|---:|---:|---|---:|']
    for ident, inv in inventories.items():
        for name, c in inv['contrasts'].items():
            lines.append(f"| {mark(ident)} | {name} | {bits(c['mean_bits_per_coda'])} | {bits(c['unweighted_mean_bits_per_coda'])} | "
                         f"{interval(c['ci_bits_per_coda'])} | {c['positive_days']}/{c['days']} |")
    lines += ['', '## Total predictability and A-conditions against repeat-own-last', '',
              'T = LL(A_B_S) − LL(none) per coda (condition viii minus the unigram), and each A-condition minus the '
              'repeat-own-last baseline (from each run\'s evaluation), within each inventory.', '',
              '| ID | T, token-weighted | T, unweighted | T, 95% CI | T, positive days | '
              + ' | '.join(f'{c} − repeat' for c in A_CONDITIONS) + ' |', '|---|---:|---:|---|---:|' + '---|' * len(A_CONDITIONS)]
    for ident, inv in inventories.items():
        t = inv['T']
        checks = ' | '.join(f"{bits(inv['repeat_baseline_checks'][c]['mean_bits_per_coda'])} "
                            f"{interval(inv['repeat_baseline_checks'][c]['ci_bits_per_coda'])}" for c in A_CONDITIONS)
        lines.append(f"| {mark(ident)} | {bits(t['mean_bits_per_coda'])} | {bits(t['unweighted_mean_bits_per_coda'])} | "
                     f"{interval(t['ci_bits_per_coda'])} | {t['positive_days']}/{t['days']} | {checks} |")
    lines += share_lines(inventories, reference, mark, '## Shares of total predictability (`[R2-3]`)')
    without = summary['without_left_out_day']
    lines += ['', f"## Without day {summary['left_out_day']} (`[R2-6]`)", '',
              f"The same quantities on the other {len(summary['days']) - 1} days; the models are not refitted. CIs resample "
              'those days with the same seed, again shared by every inventory.', '',
              '| ID | Contrast | Token-weighted mean | Unweighted day mean | 95% CI | Positive days |', '|---|---|---:|---:|---|---:|']
    for ident, inv in without.items():
        for name, c in [('T', inv['T'])] + list(inv['contrasts'].items()):
            lines.append(f"| {mark(ident)} | {name} | {bits(c['mean_bits_per_coda'])} | {bits(c['unweighted_mean_bits_per_coda'])} | "
                         f"{interval(c['ci_bits_per_coda'])} | {c['positive_days']}/{c['days']} |")
    lines += share_lines(without, reference, mark, f"### Shares without {summary['left_out_day']}")
    lines += ['', '## Pattern (`[R2-3]`)', '',
              'Sign of each contrast\'s token-weighted mean and where its 95% CI lies, all 14 days.', '',
              '| ID | ' + ' | '.join(CONTRASTS) + ' |', '|---|' + '---|' * len(CONTRASTS)]
    for ident, pattern in summary['pattern'].items():
        lines.append(f"| {mark(ident)} | " + ' | '.join(f"{p['sign']}, {p['ci']}" for p in pattern.values()) + ' |')
    lines += unk_lines(inventories, mark)
    for pair in summary['tempo_decomposition']:
        lines += decomposition_lines(pair, mark)
    lines += ['', '## Inputs', '', 'SHA-256 of every input:', '']
    lines += [f"- `{p}`: `{h}`" for p, h in summary['input_sha256'].items()]
    return lines


def share_lines(inventories, reference, mark, heading):
    lines = ['', heading, '',
             'Share = contrast / T, the ratio of token-weighted means, with its CI from the ratio within each replicate. '
             f'Differences against {reference} use the same replicates for both inventories.', '',
             '| ID | ' + ' | '.join(f'{n} share' for n in CONTRASTS) + ' | ' + ' | '.join(f'{n} − {reference}' for n in CONTRASTS) + ' |',
             '|---|' + '---:|' * (2 * len(CONTRASTS))]
    for ident, inv in inventories.items():
        shares = ' | '.join(f"{s['estimate']:.3f} {interval(s['ci'], 3)}" for s in inv['shares'].values())
        diffs = inv['share_differences_vs_reference']
        diff_cells = ' | '.join(f"{d['estimate']:.3f} {interval(d['ci'], 3)}" for d in diffs.values()) if diffs else \
            ' | '.join('—' for _ in CONTRASTS)
        lines.append(f"| {mark(ident)} | {shares} | {diff_cells} |")
    return lines


def unk_lines(inventories, mark):
    ids = list(inventories)
    lines = ['', '## UNK (`[R2-5]`)', '',
             'Comparison targets mapped to UNK under each fold\'s vocabulary (types seen fewer than twice in that fold\'s '
             'training data, or unseen). A closed vocabulary can look easier by predicting UNK (Mielke et al. 2019).', '',
             '| ID | Test targets mapped to UNK | Share of test targets | Training UNK targets, summed over folds | Vocabulary incl. UNK |',
             '|---|---:|---:|---:|---|']
    for ident, inv in inventories.items():
        folds = inv['unk_by_fold']
        test, unk = sum(f['test_targets'] for f in folds), sum(f['test_unk'] for f in folds)
        sizes = [f['vocabulary'] for f in folds]
        lines.append(f"| {mark(ident)} | {unk} | {unk / test:.1%} | {sum(f['train_unk'] for f in folds)} | {min(sizes)}–{max(sizes)} |")
    lines += ['', 'Per fold: test targets mapped to UNK (training comparison targets mapped to UNK).', '',
              '| Test day | Test targets | ' + ' | '.join(mark(i) for i in ids) + ' |', '|---|---:|' + '---:|' * len(ids)]
    by_day = {i: {f['test_day']: f for f in inventories[i]['unk_by_fold']} for i in ids}
    for day in sorted(by_day[ids[0]]):
        cells = ' | '.join(f"{by_day[i][day]['test_unk']} ({by_day[i][day]['train_unk']})" for i in ids)
        lines.append(f"| {day} | {by_day[ids[0]][day]['test_targets']} | {cells} |")
    return lines


def decomposition_lines(pair, mark):
    coarse, fine, excluded = pair['coarse'], pair['fine'], pair['excluded_targets']
    lines = ['', f"## Tempo decomposition, {mark(coarse)} → {mark(fine)} (`[R2-4]`, exploratory)", '',
             f"log2 p({fine} label) = type term + tempo term. The type term is log2 of the summed probability of {fine}'s "
             f"in-vocabulary labels whose part before `|` is the target's {coarse} label; UNK mass is left out, so it is a "
             f"lower bound. It is compared with {coarse}'s log-likelihood of the same label under the same condition "
             f"(paired, per coda). Targets mapped to UNK in either inventory are left out: {pair['eligible_targets']:,} "
             f"remain; {excluded['fine_only']} are UNK in {fine} only, {excluded['coarse_only']} in {coarse} only and "
             f"{excluded['both']} in both.", '',
             f"| Condition | Type term − {coarse} LL, token-weighted | Unweighted day mean | 95% CI | Positive days |",
             '|---|---:|---:|---|---:|']
    for condition, d in pair['type_minus_coarse'].items():
        lines.append(f"| {condition} | {bits(d['mean_bits_per_coda'])} | {bits(d['unweighted_mean_bits_per_coda'])} | "
                     f"{interval(d['ci_bits_per_coda'])} | {d['positive_days']}/{d['days']} |")
    lines += ['', f"The three `[M2]` contrasts on the tempo term (does a source predict {fine}'s tempo class once its "
              f"{coarse} label is known):", '',
              '| Contrast | Token-weighted mean | Unweighted day mean | 95% CI | Positive days |', '|---|---:|---:|---|---:|']
    for name, d in pair['tempo_contrasts'].items():
        lines.append(f"| {name} | {bits(d['mean_bits_per_coda'])} | {bits(d['unweighted_mean_bits_per_coda'])} | "
                     f"{interval(d['ci_bits_per_coda'])} | {d['positive_days']}/{d['days']} |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/rq2_compare.json')
    summary = run(parser.parse_args().config)
    stopped = [i for i, inv in summary['inventories'].items() if inv['stop_reasons']]
    print(f"Compared {len(summary['inventories'])} inventories. Stop rules fired for: {', '.join(stopped) or 'none'}.")


if __name__ == '__main__':
    main()
