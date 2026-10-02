"""Planted-signal sensitivity run for condition A of the residual LSTM (synthetic labels only).

Protocol adapted from Bastings et al. (2022), "'Will You Find These Shortcuts?' A Protocol for
Evaluating the Faithfulness of Input Salience Methods for Text Classification" (EMNLP): partially
synthetic data with a known planted rule. They add examples; here existing targets are relabelled.

On every fold, a fraction f of the eligible targets on training, validation and test days is
relabelled as the whale's own coda at `lag` (eligible: at least `lag` own codas, and the coda at
that lag differs from lags 1-3; as in check 6c). Labels only; histories are unchanged. The planted
subsets are nested across f: one permutation per fold and split. Logistic regression is refitted on
the planted training and validation days with v4's procedure, and condition A's residual selection
runs exactly as in the full run (same grid, seeds, one-SE rule). The test days are then scored on
the planted labels against the refitted logistic regression and against an oracle that knows the
rule and f, but not which targets were planted. Nothing here changes a result on real data.

Per-coda log-likelihoods use evaluation.log_likelihood_bits and intervals evaluation.day_bootstrap_bits;
the shared evaluate() scores real labels over the full condition lattice, so it does not apply here.
"""
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import torch

from .evaluation import day_bootstrap_bits, log_likelihood_bits
from .logistic import _copy_logits, last_label_index, select_l2_and_fit
from .preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text
from .residual import (FitRunner, LR_ONLY, _hashes, _load, candidate_list, check_model_settings, condition_data,
                       condition_tasks, model_settings, per_target_bits, planted_eligible, predict_condition,
                       prepare_folds, require_cuda_determinism, select_condition)

SPLITS = ('train', 'validation', 'test')


def split_seed(seed, fold_id, split):
    """One fixed RandomState seed per fold and split."""
    return int(hashlib.sha256(f'{seed}:{fold_id}:{split}'.encode()).hexdigest()[:8], 16)


def plant(rows, lag, fractions, seed, fold_id, split):
    """Eligible row indices, and for each f the planted ones: the first round(f * n) of one fixed
    permutation of the eligible rows, so the planted subsets are nested across f."""
    eligible = [i for i, r in enumerate(rows) if planted_eligible(r, lag)]
    order = np.random.RandomState(split_seed(seed, fold_id, split)).permutation(len(eligible))
    return eligible, {f: sorted(eligible[j] for j in order[:int(math.floor(f * len(eligible) + 0.5))])
                      for f in fractions}


def relabel(rows, planted, lag):
    """Labels only: the planted rows get their own coda at `lag` as target; histories are untouched."""
    chosen = set(planted)
    return [dict(r, Ct=r['A_history'][-lag]) if i in chosen else r for i, r in enumerate(rows)]


def oracle_probabilities(lr_probabilities, rows, eligible, fraction, lag, vocab):
    """f * 1[own coda at lag] + (1 - f) * P_LR on eligible rows, P_LR elsewhere. The oracle knows the
    rule and f, not which eligible rows were planted."""
    index = {t: i for i, t in enumerate(vocab)}
    p = lr_probabilities.copy()
    for i in eligible:
        p[i] = (1 - fraction) * lr_probabilities[i]
        p[i, index[rows[i]['A_history'][-lag]]] += fraction
    return p


def _quantile(samples, q):
    """evaluation.day_bootstrap_bits' interpolation rule, on sorted samples."""
    position = q * (len(samples) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return samples[lo] + (position - lo) * (samples[hi] - samples[lo])


def paired_ratio_bootstrap(numerator_days, denominator_days, replicates, seed, confidence):
    """Ratio of means, numerator / denominator, with a percentile CI from the same resampled days for
    both: per replicate, the token-weighted mean of the numerator over the resampled days divided by
    the token-weighted mean of the denominator over the same days (never a mean of per-day ratios,
    which blow up on days with few targets). Resampling (random.Random(seed), choices over the sorted
    days), weighting and quantiles are evaluation.day_bootstrap_bits', so numerator and denominator
    intervals here equal that function's.

    The denominator (an oracle's gain) can be at or below zero in a replicate: the oracle loses
    log2(1 - f) on eligible targets that were not planted, so days with eligible but no planted targets
    pull it down. Such replicates have no ratio; they are counted, not silently dropped, and the
    interval is over the remaining replicates. The caller decides what a non-zero count means."""
    days = sorted(numerator_days)
    if days != sorted(denominator_days) or any(len(numerator_days[d]) != len(denominator_days[d]) for d in days):
        raise ValueError('Numerator and denominator need the same days and targets')
    num = {d: math.fsum(numerator_days[d]) for d in days}
    den = {d: math.fsum(denominator_days[d]) for d in days}
    sizes = {d: len(numerator_days[d]) for d in days}
    rng = random.Random(seed)
    numerators, denominators, ratios, degenerate = [], [], [], 0
    for _ in range(replicates):
        selected = rng.choices(days, k=len(days))
        tokens = sum(sizes[d] for d in selected)
        a, b = math.fsum(num[d] for d in selected) / tokens, math.fsum(den[d] for d in selected) / tokens
        numerators.append(a)
        denominators.append(b)
        if b > 0:
            ratios.append(a / b)
        else:
            degenerate += 1
    tail = (1 - confidence) / 2
    interval = lambda xs: [_quantile(sorted(xs), tail), _quantile(sorted(xs), 1 - tail)]  # noqa: E731
    total = sum(sizes.values())
    mean_num, mean_den = math.fsum(num.values()) / total, math.fsum(den.values()) / total
    return dict(ratio=mean_num / mean_den if mean_den > 0 else None,
                ci_ratio=interval(ratios) if ratios else None, replicates=replicates, degenerate_replicates=degenerate,
                numerator_ci=interval(numerators), denominator_ci=interval(denominators))


def run(config_path):
    path, root, config, manifest, manifest_path, source = _load(config_path)
    real_path, real_scored_path = root / config['real_run_summary'], root / config['real_run_scored']
    real = read_json(real_path)
    check_model_settings(config, real['configuration'], 'The full run')
    throughput_path = root / config['throughput_summary']
    choice = read_json(throughput_path)['choice']
    device, workers = choice['device'], choice['workers']
    if device == 'cuda':
        require_cuda_determinism()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    logistic_path = root / config['logistic_config']
    logistic_config = read_json(logistic_path)
    rows = read_jsonl(source)
    max_len = manifest['preparation_configuration']['history_limit']
    seeds, fractions, lag = config['seeds'], config['fractions'], config['lag']
    real_scored = {(r['coda_id'], r['seed']): r for r in read_jsonl(real_scored_path) if r['condition'] == 'A'}
    candidates = candidate_list(('A',), config['hidden_grid'], config['dropout_grid'])
    settings = model_settings(config)

    jobs, counts = [], []
    with FitRunner(workers, device) as runner:
        for prepared in prepare_folds(rows, manifest):
            fold_id, vocab = prepared['fold']['fold_id'], prepared['policy']['vocabulary']
            eligible, planted = {}, {}
            for split in SPLITS:
                eligible[split], planted[split] = plant(prepared[split], lag, fractions, config['sampling_seed'], fold_id, split)
            counts.append(dict(fold_id=fold_id, test_day=prepared['fold']['test_day'],
                               targets={s: len(prepared[s]) for s in SPLITS},
                               eligible={s: len(eligible[s]) for s in SPLITS},
                               planted={str(f): {s: len(planted[s][f]) for s in SPLITS} for f in fractions}))
            for f in fractions:
                relabelled = {s: relabel(prepared[s], planted[s][f], lag) for s in SPLITS}
                encoder, refit, _ = select_l2_and_fit(relabelled['train'], relabelled['validation'], vocab,
                                                      ['A_history'], logistic_config)
                logits_fn = lambda rs, encoder=encoder, refit=refit: _copy_logits(  # noqa: E731
                    encoder.transform(rs) @ refit['w'], refit['beta'], last_label_index(rs, 'A_history', vocab), None, None)
                job = condition_data(dict(prepared, **relabelled), 'A', ['A_history'], logits_fn, max_len)
                job.update(fraction=f, test_rows=relabelled['test'], eligible_test=eligible['test'],
                           planted_test=planted['test'][f], refit_l2=refit['l2'], refit_beta=float(refit['beta']),
                           refit_validation_log_likelihood_bits=refit['validation_log_likelihood_bits'])
                job['futures'] = runner.submit(condition_tasks(job, config, seeds, device))
                jobs.append(job)
            print(f'Submitted {fold_id}', flush=True)

        selections, fits_out, scores = [], [], []
        for job in jobs:
            fits = {}
            for future in job['futures']:
                (fold_id, _, name, seed), fit = future.result()
                fits[name, seed] = fit
                fits_out.append(dict(fold_id=fold_id, fraction=job['fraction'], candidate=name, seed=seed,
                                     best_epoch=fit['best_epoch'], epochs_trained=fit['epochs_trained'],
                                     epoch0_validation_log_likelihood_bits=fit['epoch0_validation_log_likelihood_bits'],
                                     best_validation_log_likelihood_bits=fit['best_validation_log_likelihood_bits']))
            job['futures'].clear()
            lr_val = per_target_bits(job['lr_probabilities']['validation'], job['data']['validation']['y'])
            selection, _ = select_condition(candidates, fits, seeds, lr_val)
            chosen = next(c for c in candidates if c['name'] == selection['chosen'])
            probabilities = predict_condition(chosen, fits, seeds, job['data']['test'], job['lr_probabilities']['test'],
                                              settings, None)
            selections.append(dict(fold_id=job['fold_id'], fraction=job['fraction'], refit_l2=job['refit_l2'],
                                   refit_beta=job['refit_beta'],
                                   refit_validation_log_likelihood_bits=job['refit_validation_log_likelihood_bits'],
                                   chosen_best_epochs=[None if chosen['name'] == LR_ONLY else fits[chosen['name'], s]['best_epoch']
                                                       for s in seeds], **selection))
            scores += score_test(job, probabilities, seeds, lag, real_scored)
            print(f"Finished {job['fold_id']} f={job['fraction']}", flush=True)

    summary = summarise(scores, selections, counts, config, real)
    summary.update(configuration=config, device=dict(device=device, workers=workers),
                   versions={'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__},
                   input_sha256=_hashes(root, [source, manifest_path, path, logistic_path, real_path, real_scored_path,
                                               throughput_path]))
    output = root / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'selection.jsonl', selections)
    write_jsonl(output / 'fits.jsonl', fits_out)
    write_jsonl(output / 'scores.jsonl', scores)
    write_text(output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    report = root / config['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, '\n'.join(report_lines(summary, config)))
    return summary


def score_test(job, probabilities, seeds, lag, real_scored):
    """Per test target, on the planted labels: every seed's residual log-likelihood, the refitted logistic
    regression's and the oracle's; for targets left unplanted, the real-data run's per-seed values too."""
    rows, vocab, y = job['test_rows'], job['vocab'], job['data']['test']['y']
    lr = job['lr_probabilities']['test']
    oracle = oracle_probabilities(lr, rows, job['eligible_test'], job['fraction'], lag, vocab)
    eligible, planted = set(job['eligible_test']), set(job['planted_test'])
    out = []
    for i, row in enumerate(rows):
        t = y[i]
        record = dict(fold_id=job['fold_id'], fraction=job['fraction'], coda_id=row['coda_id'],
                      recording_date=row['recording_date'], target=row['Ct'], eligible=i in eligible, planted=i in planted,
                      residual_bits=[log_likelihood_bits(float(p[i, t])) for p in probabilities],
                      lr_bits=log_likelihood_bits(float(lr[i, t])), oracle_bits=log_likelihood_bits(float(oracle[i, t])))
        if i not in planted:
            real = [real_scored[row['coda_id'], s] for s in seeds]
            if any(r['target'] != row['Ct'] for r in real):
                raise ValueError('An unplanted target differs from the real-data run')
            record['real_residual_bits'] = [r['log_likelihood_bits'] for r in real]
        out.append(record)
    return out


def summarise(scores, selections, counts, config, real):
    boot = dict(replicates=config['bootstrap_replicates'], seed=config['bootstrap_seed'], confidence=config['confidence'])
    a_upper = real['residual_gain']['A']['ci_bits_per_coda'][1]
    per_fraction = {}
    for f in config['fractions']:
        rs = [r for r in scores if r['fraction'] == f]
        residual_gain, oracle_gain, unplanted_diff = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in rs:
            # per-seed differences, so folds where logistic regression alone was chosen give exactly zero
            residual_gain[r['recording_date']].append(math.fsum(x - r['lr_bits'] for x in r['residual_bits']) / len(r['residual_bits']))
            oracle_gain[r['recording_date']].append(r['oracle_bits'] - r['lr_bits'])
            if not r['planted']:
                unplanted_diff[r['recording_date']].append(
                    math.fsum(a - b for a, b in zip(r['residual_bits'], r['real_residual_bits'])) / len(r['residual_bits']))
        gain = day_bootstrap_bits(residual_gain, boot['replicates'], boot['seed'], boot['confidence'])
        oracle = day_bootstrap_bits(oracle_gain, boot['replicates'], boot['seed'], boot['confidence'])
        ratio = paired_ratio_bootstrap(residual_gain, oracle_gain, boot['replicates'], boot['seed'], boot['confidence'])
        if ratio['numerator_ci'] != gain['ci_bits_per_coda'] or ratio['denominator_ci'] != oracle['ci_bits_per_coda']:
            raise ValueError('The ratio bootstrap did not resample the same days as the shared day bootstrap')
        seed_mean = lambda xs: math.fsum(xs) / len(xs)  # noqa: E731
        mean = lambda xs: math.fsum(xs) / len(xs) if xs else None  # noqa: E731
        unplanted = [r for r in rs if not r['planted']]
        planted = [r for r in rs if r['planted']]
        # The bound needs a denominator that stays positive in every replicate; more than a handful of
        # replicates without one make the ratio unstable at this f.
        degenerate = ratio['degenerate_replicates']
        ratio['status'] = ('stable' if degenerate == 0 else
                           'unstable' if degenerate > config['ratio_handful'] else 'denominator not always positive')
        r_low = ratio['ci_ratio'][0] if degenerate == 0 and ratio['ci_ratio'] else None
        per_fraction[str(f)] = dict(
            fraction=f, test_targets=len(rs), planted_test_targets=len(planted), eligible_test_targets=sum(r['eligible'] for r in rs),
            planted_test_share=len(planted) / len(rs),
            folds_with_a_selected=sum(s['chosen'] != LR_ONLY for s in selections if s['fraction'] == f),
            residual_minus_lr=gain, oracle_minus_lr=oracle, recovery=ratio,
            unplanted=dict(targets=len(unplanted), residual_bits=mean([seed_mean(r['residual_bits']) for r in unplanted]),
                           real_run_residual_bits=mean([seed_mean(r['real_residual_bits']) for r in unplanted]),
                           refitted_lr_bits=mean([r['lr_bits'] for r in unplanted]),
                           difference=day_bootstrap_bits(unplanted_diff, boot['replicates'], boot['seed'], boot['confidence'])),
            planted=dict(targets=len(planted), residual_bits=mean([seed_mean(r['residual_bits']) for r in planted]),
                         refitted_lr_bits=mean([r['lr_bits'] for r in planted]), oracle_bits=mean([r['oracle_bits'] for r in planted])),
            bound_bits_per_coda=a_upper / r_low if r_low is not None and r_low > 0 else None)
    return dict(units='bits per coda, on the planted labels', real_run_a_gain_upper_bits=a_upper,
                per_fraction=per_fraction, counts=counts)


def _f(value, digits=5):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def _ci(ci, digits=4):
    return 'n/a' if ci is None else f'[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]'


def _g(value):
    """Four significant digits: the recovery ratio can be small."""
    return 'n/a' if value is None else f'{value:.4g}'


def report_lines(summary, config):
    fractions = [str(f) for f in config['fractions']]
    per = summary['per_fraction']
    lines = ['# RQ1 residual LSTM: planted-signal sensitivity, condition A', '',
             "Synthetic labels only; no result on real data changes. Protocol adapted from Bastings et al. (2022), "
             "\"'Will You Find These Shortcuts?' A Protocol for Evaluating the Faithfulness of Input Salience Methods "
             "for Text Classification\" (EMNLP): partially synthetic data with a known planted rule. They add examples; "
             "here existing targets are relabelled.", '',
             f"On all 14 folds, a fraction f in {config['fractions']} of the eligible targets on training, validation "
             f"and test days is relabelled as the whale's own coda at lag {config['lag']}. Eligible: at least "
             f"{config['lag']} own codas, and the coda at lag {config['lag']} differs from lags 1-3 (as in check 6c). "
             "Labels only; histories are unchanged. Planted subsets are nested across f: one fixed permutation per "
             f"fold and split (RandomState seeded from SHA-256 of '{config['sampling_seed']}:fold:split'). Logistic "
             "regression is refitted per fold and f on the planted training and validation days with v4's procedure "
             "(L2 chosen on the validation day), and condition A's residual selection runs exactly as in the full run: "
             f"hidden {config['hidden_grid']} x dropout {config['dropout_grid']} plus logistic regression alone, seeds "
             f"{config['seeds']}, the one-SE rule. Device: {summary['device']['device']} with "
             f"{summary['device']['workers']} processes.", '',
             'Scored on the planted test labels, over all test targets of the 14 days. (1) residual network minus the '
             'refitted logistic regression; (2) an oracle minus the refitted logistic regression, where the oracle '
             'predicts f x 1[own coda at the lag] + (1 - f) x P_LR on eligible test targets and P_LR elsewhere (it knows '
             'the rule and f, not which targets were planted); (3) the recovery ratio r = (1)/(2). Log-likelihoods use '
             'evaluation.log_likelihood_bits; intervals are evaluation.day_bootstrap_bits '
             f"({config['bootstrap_replicates']} replicates). r is a ratio of means: in each replicate, (1)'s "
             "token-weighted mean over the resampled days divided by (2)'s over the same days. The oracle loses "
             'log2(1 - f) on eligible targets that were not planted, so a replicate can have an oracle difference at or '
             'below zero; those replicates have no ratio and are counted (column "Non-positive"); r\'s interval is over '
             f"the rest. More than {config['ratio_handful']} such replicates make r unstable at that f. Per-coda residual "
             'values are means over seeds of per-seed differences.', '',
             '## Detection', '',
             '| f | Planted test targets (share) | Folds with A selected | (1) Residual - LR | CI | Positive days | (2) Oracle - LR | CI | (3) r | r CI | Non-positive |',
             '|---:|---|---:|---:|---|---:|---:|---|---:|---|---:|']
    for f in fractions:
        p = per[f]
        g, o, r = p['residual_minus_lr'], p['oracle_minus_lr'], p['recovery']
        r_ci = ('unstable' if r['status'] == 'unstable' else 'n/a' if r['ci_ratio'] is None
                else f"[{_g(r['ci_ratio'][0])}, {_g(r['ci_ratio'][1])}]")
        lines.append(f"| {f} | {p['planted_test_targets']}/{p['test_targets']} ({p['planted_test_share']:.1%}) | "
                     f"{p['folds_with_a_selected']}/14 | {_f(g['mean_bits_per_coda'])} | {_ci(g['ci_bits_per_coda'])} | "
                     f"{g['positive_days']}/{g['days']} | {_f(o['mean_bits_per_coda'])} | {_ci(o['ci_bits_per_coda'])} | "
                     f"{_g(r['ratio'])} | {r_ci} | {r['degenerate_replicates']}/{r['replicates']} |")
    lines += ['', '## Bound', '',
              f"The real-data residual gain of condition A over logistic v4 has a day-bootstrap upper bound of "
              f"{summary['real_run_a_gain_upper_bits']:.4f} bits/coda (reports/rq1_lstm_residual.md). Where r's lower "
              "bound is above zero, that upper bound divided by r's lower bound is the oracle gain above which a pattern "
              f"of this form (a deterministic copy of the own coda at lag {config['lag']}, which logistic regression's "
              "three lags cannot express) would have pushed the residual gain past the observed upper bound. It is "
              "stated only where the oracle difference stays positive in every resample.", '']
    for f in fractions:
        p = per[f]
        r = p['recovery']
        ci, k, n = r['ci_ratio'], r['degenerate_replicates'], r['replicates']
        if r['status'] == 'unstable':
            lines.append(f"- f = {f}: the ratio is unstable at this f ({k} of {n} resamples have an oracle difference "
                         "at or below zero), so no bound is stated.")
        elif k:
            lines.append(f"- f = {f}: {k} of {n} resamples have an oracle difference at or below zero, so the "
                         "denominator does not stay positive and no bound is stated.")
        elif p['bound_bits_per_coda'] is not None:
            lines.append(f"- f = {f}: r >= {_g(ci[0])}, so structure of this form and size would have been detected: "
                         f"a lag-{config['lag']} copy pattern worth more than {p['bound_bits_per_coda']:.4f} bits/coda to "
                         f"the oracle (here the oracle gains {p['oracle_minus_lr']['mean_bits_per_coda']:.4f}).")
        else:
            lines.append(f"- f = {f}: r's lower bound is {_g(ci[0])}, at or below zero, so the run gives no bound.")
    lines += ['', "## Bastings et al.'s checks", '',
              '(a) Targets left unplanted: the residual network trained on planted data next to the real-data run on the '
              'same targets and labels (planting should not degrade them), with the refitted logistic regression for '
              'context. Difference: planted-data residual minus real-data residual, day bootstrap.', '',
              '| f | Targets | Residual (planted run) | Residual (real-data run) | Difference | CI | Refitted LR |',
              '|---:|---:|---:|---:|---:|---|---:|']
    for f in fractions:
        u = per[f]['unplanted']
        lines.append(f"| {f} | {u['targets']} | {_f(u['residual_bits'])} | {_f(u['real_run_residual_bits'])} | "
                     f"{_f(u['difference']['mean_bits_per_coda'])} | {_ci(u['difference']['ci_bits_per_coda'])} | "
                     f"{_f(u['refitted_lr_bits'])} |")
    lines += ['', '(b) Planted test targets: mean log2-probability of the planted label.', '',
              '| f | Targets | Residual network | Refitted LR | Oracle |', '|---:|---:|---:|---:|---:|']
    for f in fractions:
        p = per[f]['planted']
        lines.append(f"| {f} | {p['targets']} | {_f(p['residual_bits'], 3)} | {_f(p['refitted_lr_bits'], 3)} | "
                     f"{_f(p['oracle_bits'], 3)} |")
    lines += ['', '## Counts per fold', '',
              'Eligible and planted targets per split (training / validation / test), and the planted share of each '
              "fold's targets over all three splits.", '',
              '| Fold | Targets | Eligible | ' + ' | '.join(f'Planted, f = {f}' for f in fractions) + ' |',
              '|---|---|---|' + '---|' * len(fractions)]
    for c in summary['counts']:
        total = sum(c['targets'].values())
        triple = lambda d: '/'.join(str(d[s]) for s in SPLITS)  # noqa: E731
        cells = [f"{triple(c['planted'][f])} ({sum(c['planted'][f].values()) / total:.1%})" for f in fractions]
        lines.append(f"| {c['fold_id']} | {triple(c['targets'])} | {triple(c['eligible'])} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Chosen candidate per fold and f, and the selection behind it, are in '
              'data/results/rq1_lstm_residual_planted/selection.jsonl; per-target scores in scores.jsonl; every fit in '
              'fits.jsonl; summary.json records SHA-256 of every input.', '',
              'This is a sensitivity check of the model and selection rule on one synthetic pattern; it says nothing '
              'about structure of other forms.', '']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/rq1_lstm_residual_planted.json')
    summary = run(parser.parse_args().config)
    for f, p in summary['per_fraction'].items():
        print(f"f={f}: residual-LR {p['residual_minus_lr']['mean_bits_per_coda']:.5f}, "
              f"oracle-LR {p['oracle_minus_lr']['mean_bits_per_coda']:.5f}, r {p['recovery']['ratio']}, "
              f"r CI {p['recovery']['ci_ratio']}")


if __name__ == '__main__':
    main()
