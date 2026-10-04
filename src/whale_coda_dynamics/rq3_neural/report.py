"""The [R3-8] report: contrasts with group-bootstrap CIs, the predictions table and reports/rq3_neural.md.

Reads only the stage outputs and Vlad's committed n-gram scores; fits nothing. Every contrast is
paired on the same codas and aggregated through Vlad's [R3-3] functions (``contrasts.py``);
positive means the first model is better. The verdict rules for the [R3-8] predictions are fixed
here, in code, before any arm was run (see PREDICTION_RULES).
"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from whale_coda_dynamics.preprocessing.io_utils import read_json, write_text
from whale_coda_dynamics.rq3.experiments.data import (
    TRANSFER_SCORE_COLUMNS,
    TRANSFER_SUMMARY_COLUMNS,
    read_csv,
)
from whale_coda_dynamics.rq3.experiments.evaluation import aggregate_transfer_loss

from .contrasts import ci_above_zero, ci_below_zero, grouped_contrast, paired_contrast
from .runner import (
    GATE1_COLUMNS,
    GROUP_SUM_COLUMNS,
    SELECTION_COLUMNS,
    Stage,
    arm1_source,
    output_dir,
    pooled_model_name,
    read_scores,
    transfer_model_name,
)

CONTRAST_COLUMNS = ['contrast', 'setting', 'clan', 'n_codas', 'n_groups', 'n_symbols', 'first_bits_per_symbol',
                    'second_bits_per_symbol', 'contrast_bits_per_symbol', 'ci95_lower_bits_per_symbol',
                    'ci95_upper_bits_per_symbol', 'contrast_bits_per_coda', 'ci95_lower_bits_per_coda',
                    'ci95_upper_bits_per_coda', 'bootstrap_draws']
MATRIX_COLUMNS = ['source_clan', 'target_clan', 'source_training_codas', 'lstm_own_bits_per_symbol',
                  'lstm_transfer_bits_per_symbol', 'lstm_transfer_loss_bits_per_symbol', 'ci95_lower_bits_per_symbol',
                  'ci95_upper_bits_per_symbol', 'ngram_transfer_loss_bits_per_symbol']
EFFICIENCY_COLUMNS = ['clan', 'share', 'version', 'bits_per_symbol', 'n_symbols']
PREDICTION_COLUMNS = ['prediction', 'rule', 'result', 'verdict']

PREDICTION_RULES = {
    'P1': ('Pooling gain is largest for the smallest clans (EC2, SI) and about zero for the largest (EC1, REG)',
           "At arm 2's primary alpha: (a) the two smallest clans have the two largest pooling gains; (b) the two "
           "largest clans' 95% CIs contain zero. Both: supported; one: partly supported; neither: not supported."),
    'P2': ('Lower alpha helps small clans and costs large ones',
           'Arm 2 at the lowest alpha minus arm 2 at alpha = 1, per clan. Supported: CI above zero for the two '
           'smallest clans and below zero for the two largest; direction only: all four point estimates have the '
           'predicted sign; partly: one to three do; not supported: none.'),
    'P3': ('Tag value is clearly positive for every clan',
           "Tagged minus untagged at arm 2's primary alpha. Supported: CI above zero for all nine clans; partly: "
           'for some; not supported: for none.'),
    'P4': ('The leave-one-out model beats the mean single source',
           'Leave-one-out minus the mean of the eight single-source transfer scores (arm 1), per target clan. '
           'Supported: CI above zero for all nine; partly: for some; not supported: for none.'),
    'P5': ('Fine-tuning helps most at small f, and the gap closes as f grows',
           'Fine-tuned minus from scratch per clan and share. A clan agrees when (a) the CI at the smallest share '
           'is above zero and (b) the largest gain is at one of the two smallest shares and the gain at the '
           'largest share is below the gain at the smallest. Supported: all nine clans; partly: some; '
           'not supported: none.'),
}

FIGURES = [
    ('pooling_gain_vs_size.png', 'Pooling gain against clan size, one line per alpha'),
    ('tag_value.png', 'Tag value per clan, one marker per alpha'),
    ('zero_shot.png', 'Zero-shot loss beside the single-source losses per target'),
    ('data_efficiency.png', 'Data-efficiency curves, one panel per clan'),
    ('matrix_vs_ngram.png', 'Neural against n-gram transfer matrix'),
]

INTERPRETATIONS = [
    'Score rows use PLAN `[R3-3]`\'s columns. `ll_bits` is the coda\'s log2 probability, the same quantity and '
    "sign as Vlad's `bits_per_coda` (negative, higher is better); `n_symbols` is his `token_count` (intervals + "
    '`EOS`, checked equal for every coda and model). A row is the mean over the refit seeds; matrix rows are the '
    'mean over the source\'s fold models of their seed means (as `[R3-4]`). `group_id` for the bootstrap is '
    'joined from the codas file.',
    'Selection fits use one seed (0); the chosen setting is refit with seeds 0, 1, 2 for the selected fit\'s best '
    'epoch (the number of epochs is kept, the number of draws per epoch follows the larger refit training set).',
    'Training draws are independent and with replacement: clan c with probability proportional to n_c^alpha, then '
    'a coda uniformly within the clan. Single-clan models (arms 1 and 5) use the same sampler with one clan.',
    'Early stopping and the choice of hidden size and dropout use the inner-validation bits per symbol over all '
    'inner-validation codas (sum of log2 probabilities over sum of symbols). Only alpha is chosen on the '
    'clan-averaged score, as `[R3-8]` says.',
    "Primary alpha: for each run, the chosen setting's selection fit at its best epoch; a clan's "
    'inner-validation codas are pooled over the runs in which the clan has a held-out fold, so each coda counts '
    'once; clans are then weighted equally. Ties go to the alpha listed first. Each pooled arm has its own '
    "primary alpha; arm 4 uses arm 3's.",
    'Pooled run 5 holds out EC2 fold 5 only; the other clans hold nothing out, validate on fold 1 '
    "((5 + 1) mod 5) and are refit on all five folds. Run 5's scores are EC2 fold 5 only.",
    '"Same settings" for tagged and untagged models means the same fixed settings, grid, seeds and training draws '
    'for every candidate (checked per candidate); each model then chooses its own hidden size and dropout on '
    'inner validation.',
    'Every model has the same input vocabulary (20 interval symbols, `BOS`, nine `BOS_<clan>`, padding), so '
    'tagged and untagged models start from identical weights; symbols a model never sees get no gradient. '
    'Own-clan and data-efficiency models use the shared `BOS`.',
    "Tag value is reported at every alpha; the headline (and P3) uses arm 2's primary alpha.",
    'Dropout is applied to the embeddings and to the LSTM outputs (one layer, so the LSTM\'s own dropout does '
    'nothing).',
    'Selection ties go to the smaller model: smaller hidden size, then smaller dropout, then the earlier '
    'candidate.',
    'Capacity check: run at every alpha at which some clan\'s pooling-gain CI lies entirely below zero; the rerun '
    "doubles each run's chosen hidden size, keeps its dropout, early-stops one selection fit (seed 0) and refits "
    'with the three seeds.',
    'Arm 5: the share is drawn at coda level from the clan\'s training folds without its inner-validation fold, '
    'independently for each share and draw, m = max(1, round(f x pool)). Draw d uses seed d for data order and '
    "initialisation, and the fine-tuned version starts from arm 4's refit with seed d. Epoch 0 (the starting "
    'model) is a candidate for early stopping for both versions; learning-rate ties go to 0.001.',
    'Neural matrix: transfer loss = own-clan LSTM - transfer score, per symbol; the rank correlation is Spearman\'s '
    'over the 72 off-diagonal cells, against the n-gram losses rebuilt from Vlad\'s `scores.csv` with his '
    'aggregation (checked against his `summary.csv`).',
    'Gate 1 and every contrast: {draws:,} group-bootstrap draws with seeds from Vlad\'s `pair_seed({seed}, labels)`; '
    'a CI "entirely below zero" means its upper end is below zero.',
    'The pilot fits score no held-out coda; their epochs feed the time projection only.',
]


def fmt(value, digits=3):
    if value is None or value == '':
        return ''
    return f'{value:+.{digits}f}'


def ci(row, digits=3):
    return (f"{fmt(row['contrast_bits_per_symbol'], digits)} [{fmt(row['ci95_lower_bits_per_symbol'], digits)}, "
            f"{fmt(row['ci95_upper_bits_per_symbol'], digits)}]")


def table(header, rows, align=None):
    align = align or ['---'] * len(header)
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def subset(scores, ids):
    return {i: scores[i] for i in ids}


class Report:
    def __init__(self, ctx, git):
        self.ctx, self.config = ctx, ctx.config
        self.stage = Stage(ctx, 'report', git)
        self.group_of = dict(zip(ctx.ids, ctx.group))
        self.clan_ids = {clan: [ctx.ids[i] for i in ctx.clan_indices(clan)] for clan in ctx.clans}
        self.sizes = {clan: len(ids) for clan, ids in self.clan_ids.items()}
        self.contrast_rows = []

    def path(self, stage, name):
        path = output_dir(self.ctx, stage) / name
        self.stage.add_inputs(path)
        return path

    def summary(self, stage):
        return read_json(self.path(stage, 'summary.json'))

    def contrast(self, name, setting, clan, first, second):
        ids = self.clan_ids[clan]
        row = paired_contrast(subset(first, ids), subset(second, ids), self.group_of, self.config['bootstrap_draws'],
                              self.config['bootstrap_seed'], (name, setting, clan))
        row = dict(contrast=name, setting=setting, clan=clan, **row)
        self.contrast_rows.append(row)
        return row


def ngram_transfer_losses(scores_path, summary_path):
    """Vlad's n-gram transfer loss per off-diagonal cell (bits per symbol), rebuilt from his per-coda
    matrix scores with his aggregation and required to equal his summary.csv."""
    cells = defaultdict(list)
    for row in read_csv(scores_path, TRANSFER_SCORE_COLUMNS):
        if row['source_clan'] != row['target_clan']:
            cells[row['source_clan'], row['target_clan']].append({
                'n_symbols': int(row['token_count']), 'own_bits_per_coda': float(row['own_bits_per_coda']),
                'transfer_bits_per_coda': float(row['transfer_bits_per_coda']),
                'transfer_loss_bits_per_coda': float(row['transfer_loss_bits_per_coda'])})
    losses = {cell: aggregate_transfer_loss(rows)['mean_transfer_loss_bits_per_symbol'] for cell, rows in cells.items()}
    recorded = {(row['source_clan'], row['target_clan']): float(row['mean_transfer_loss_bits_per_symbol'])
                for row in read_csv(summary_path, TRANSFER_SUMMARY_COLUMNS) if row['source_clan'] != row['target_clan']}
    if set(recorded) != set(losses):
        raise ValueError("Vlad's matrix scores and summary cover different cells")
    for cell, value in losses.items():
        if abs(value - recorded[cell]) > 1e-9:
            raise ValueError(f"The n-gram matrix rebuilt from Vlad's scores differs from his summary at {cell}")
    return losses


def stage_report(ctx, git):
    r = Report(ctx, git)
    config, clans = ctx.config, ctx.clans
    alphas = config['pooled_alphas']
    pilot = r.summary('pilot')
    gate0 = read_csv(r.path('pilot', 'gate0.csv'), None)
    timing = read_csv(r.path('pilot', 'timing.csv'), None)
    determinism = read_csv(r.path('pilot', 'determinism.csv'), None)
    source = arm1_source(ctx)
    arm1_runs = ['arm1'] + (['arm1_rerun'] if (output_dir(ctx, 'arm1_rerun') / 'summary.json').is_file() else [])
    gates = {name: read_csv(r.path(name, 'gate1.csv'), GATE1_COLUMNS) for name in arm1_runs}
    summaries = {name: r.summary(name) for name in arm1_runs + ['arm23', 'capacity', 'arm4', 'arm5']}
    own = read_scores(ctx, r.path(source, 'own_scores.csv'))['lstm_own']
    matrix = read_scores(ctx, r.path(source, 'matrix_scores.csv'))
    transfer = {s: matrix[transfer_model_name(s)] for s in clans}
    for s in clans:
        if set(transfer[s]) != {i for t in clans if t != s for i in r.clan_ids[t]}:
            raise ValueError(f'The {s} fold models must score every coda of the other clans once')
    arm2 = read_scores(ctx, r.path('arm23', 'arm2_scores.csv'))
    arm3 = read_scores(ctx, r.path('arm23', 'arm3_scores.csv'))
    loo = read_scores(ctx, r.path('arm4', 'scores.csv'))['lstm_leave_one_out']
    primary = summaries['arm23']['primary_alpha']
    selections = {name: read_csv(r.path(name, 'selection.csv'), SELECTION_COLUMNS)
                  for name in [source, 'arm23', 'arm4']}

    # ------------------------------------------------ neural matrix and the n-gram matrix
    r.stage.add_inputs(ctx.path('ngram_matrix_scores_input'), ctx.path('ngram_matrix_summary_input'))
    ngram_g = ngram_transfer_losses(ctx.path('ngram_matrix_scores_input'), ctx.path('ngram_matrix_summary_input'))
    if len(ngram_g) != len(clans) * (len(clans) - 1):
        raise ValueError('The n-gram matrix must have every off-diagonal cell')
    training_codas = {clan: int(np.mean([int(row['n_train']) for row in read_csv(r.path(source, 'refits.csv'), None)
                                         if row['run'].split(':')[0] == clan])) for clan in clans}
    matrix_out = []
    for s in clans:
        for t in clans:
            if s == t:
                continue
            ids = r.clan_ids[t]
            c = paired_contrast(subset(own, ids), subset(transfer[s], ids), r.group_of, config['bootstrap_draws'],
                                config['bootstrap_seed'], ('matrix', s, t))
            matrix_out.append(dict(source_clan=s, target_clan=t, source_training_codas=training_codas[s],
                                   lstm_own_bits_per_symbol=c['first_bits_per_symbol'],
                                   lstm_transfer_bits_per_symbol=c['second_bits_per_symbol'],
                                   lstm_transfer_loss_bits_per_symbol=c['contrast_bits_per_symbol'],
                                   ci95_lower_bits_per_symbol=c['ci95_lower_bits_per_symbol'],
                                   ci95_upper_bits_per_symbol=c['ci95_upper_bits_per_symbol'],
                                   ngram_transfer_loss_bits_per_symbol=ngram_g[s, t]))
    rho, rho_p = spearmanr([m['lstm_transfer_loss_bits_per_symbol'] for m in matrix_out],
                           [m['ngram_transfer_loss_bits_per_symbol'] for m in matrix_out])
    g_lstm = {(m['source_clan'], m['target_clan']): m for m in matrix_out}

    # ------------------------------------------------ pooled contrasts
    gain, tag, alpha_effect = {}, {}, {}
    for alpha in alphas:
        tagged, untagged = arm2[pooled_model_name('arm2', alpha)], arm3[pooled_model_name('arm3', alpha)]
        for clan in clans:
            gain[alpha, clan] = r.contrast('pooling_gain', alpha, clan, tagged, own)
            tag[alpha, clan] = r.contrast('tag_value', alpha, clan, tagged, untagged)
    low, high = min(alphas), max(alphas)
    for clan in clans:
        alpha_effect[clan] = r.contrast('alpha_effect', f'{low} vs {high}', clan,
                                        arm2[pooled_model_name('arm2', low)], arm2[pooled_model_name('arm2', high)])
    capacity = summaries['capacity']
    capacity_gain = {}
    if capacity['triggered_alphas']:
        doubled = read_scores(ctx, r.path('capacity', 'scores.csv'))
        for alpha in capacity['triggered_alphas']:
            scores = doubled[pooled_model_name('arm2', alpha, capacity['hidden_multiplier'])]
            for clan in clans:
                capacity_gain[alpha, clan] = r.contrast('pooling_gain_hidden_x2', alpha, clan, scores, own)

    # ------------------------------------------------ zero-shot (arm 4)
    zero_shot, mean_single, loo_vs_mean = {}, {}, {}
    for t in clans:
        sources = [s for s in clans if s != t]
        mean_source = {i: (float(np.mean([transfer[s][i][0] for s in sources])), own[i][1]) for i in r.clan_ids[t]}
        zero_shot[t] = r.contrast('zero_shot_loss', 'arm4', t, own, loo)
        mean_single[t] = r.contrast('mean_single_source_loss', 'arm1', t, own, mean_source)
        loo_vs_mean[t] = r.contrast('loo_minus_mean_single_source', 'arm4', t, loo, mean_source)

    # ------------------------------------------------ data efficiency (arm 5)
    arm5 = summaries['arm5']
    shares = arm5['shares']
    group_rows = read_csv(r.path('arm5', 'group_sums.csv'), GROUP_SUM_COLUMNS)
    sums = defaultdict(lambda: [0.0, 0])
    draws_seen = defaultdict(set)
    for row in group_rows:
        key = (row['target_clan'], float(row['share']), row['version'], row['group_id'])
        sums[key][0] += float(row['ll_bits_sum'])
        sums[key][1] = int(row['n_symbols'])
        draws_seen[key].add(int(row['draw']))
    efficiency, curves = {}, []
    for clan in clans:
        groups = sorted({ctx.group[i] for i in ctx.clan_indices(clan)})
        for share in shares:
            per_version = {}
            for version in ('finetuned', 'scratch'):
                g = {}
                for group in groups:
                    key = (clan, share, version, group)
                    if len(draws_seen[key]) != arm5['draws']:
                        raise ValueError(f'arm5 {key}: expected {arm5["draws"]} draws')
                    g[group] = (sums[key][0] / arm5['draws'], sums[key][1])
                per_version[version] = g
                total = sum(n for _, n in g.values())
                curves.append(dict(clan=clan, share=share, version=version,
                                   bits_per_symbol=sum(ll for ll, _ in g.values()) / total, n_symbols=total))
            c = grouped_contrast(per_version['finetuned'], per_version['scratch'], config['bootstrap_draws'],
                                 config['bootstrap_seed'], ('data_efficiency', share, clan))
            efficiency[share, clan] = dict(contrast='data_efficiency', setting=share, clan=clan, **c)
            r.contrast_rows.append(efficiency[share, clan])
        for version, scores in (('zero_shot', loo), ('own_clan_full', own)):
            ids = r.clan_ids[clan]
            curves.append(dict(clan=clan, share='', version=version,
                               bits_per_symbol=sum(scores[i][0] for i in ids) / sum(scores[i][1] for i in ids),
                               n_symbols=sum(scores[i][1] for i in ids)))

    predictions = judge(r, clans, primary, gain, tag, alpha_effect, loo_vs_mean, efficiency, shares, low, high)

    # ------------------------------------------------ outputs
    out = r.stage
    out.csv('contrasts.csv', CONTRAST_COLUMNS, r.contrast_rows)
    out.csv('matrix.csv', MATRIX_COLUMNS, matrix_out)
    out.csv('data_efficiency.csv', EFFICIENCY_COLUMNS, curves)
    out.csv('predictions.csv', PREDICTION_COLUMNS, predictions)
    out.csv('clan_sizes.csv', ['clan', 'n_codas'], [dict(clan=c, n_codas=r.sizes[c]) for c in clans])
    files = file_sizes(ctx)
    summary = out.finish(arm1_source=source, primary_alpha=primary, spearman_rho=rho, spearman_p=rho_p,
                         predictions={row['prediction']: row['verdict'] for row in predictions},
                         gate1_passed={name: summaries[name]['gate_passed'] for name in arm1_runs})
    text = render(r, ctx, pilot, gate0, timing, determinism, arm1_runs, gates, summaries, selections, source,
                  matrix_out, rho, rho_p, primary, gain, tag, alpha_effect, capacity, capacity_gain, zero_shot,
                  mean_single, loo_vs_mean, g_lstm, efficiency, curves, shares, predictions, files, low, high)
    path = ctx.root / config['report_output']
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, text)
    return summary


def file_sizes(ctx):
    base = ctx.root / ctx.config['output_directory']
    checkpoints = (ctx.root / ctx.config['checkpoint_directory']).resolve()
    own_summary = (base / 'report' / 'summary.json').resolve()
    rows = []
    for path in sorted(base.rglob('*')):
        if path.is_file() and checkpoints not in path.resolve().parents and path.resolve() != own_summary:
            rows.append((path.relative_to(ctx.root).as_posix(), path.stat().st_size))
    return rows


def judge(r, clans, primary, gain, tag, alpha_effect, loo_vs_mean, efficiency, shares, low, high):
    by_size = sorted(clans, key=lambda c: r.sizes[c])
    small, large = by_size[:2], by_size[-2:]
    a2 = primary['arm2']
    rows = []

    # P1
    gains = {c: gain[a2, c]['contrast_bits_per_symbol'] for c in clans}
    top2 = sorted(clans, key=lambda c: -gains[c])[:2]
    part_a = set(top2) == set(small)
    part_b = all(not ci_below_zero(gain[a2, c]) and not ci_above_zero(gain[a2, c]) for c in large)
    rho, _ = spearmanr([r.sizes[c] for c in clans], [gains[c] for c in clans])
    verdict = 'supported' if part_a and part_b else 'partly supported' if part_a or part_b else 'not supported'
    rows.append(dict(prediction='P1', rule=PREDICTION_RULES['P1'][1], verdict=verdict, result=(
        f"alpha {a2}: largest gains {top2[0]}, {top2[1]} (smallest clans {small[0]}, {small[1]}); "
        + '; '.join(f'{c} {ci(gain[a2, c])}' for c in small + large)
        + f'; Spearman(size, gain) {rho:+.2f}')))

    # P2
    expected = {c: +1 for c in small} | {c: -1 for c in large}
    right = [c for c in small + large if np.sign(alpha_effect[c]['contrast_bits_per_symbol']) == expected[c]]
    reliable = [c for c in small if ci_above_zero(alpha_effect[c])] + [c for c in large if ci_below_zero(alpha_effect[c])]
    verdict = ('supported' if len(reliable) == 4 else 'direction only' if len(right) == 4
               else 'partly supported' if right else 'not supported')
    rows.append(dict(prediction='P2', rule=PREDICTION_RULES['P2'][1], verdict=verdict, result=(
        f'alpha {low} - alpha {high}: ' + '; '.join(f'{c} {ci(alpha_effect[c])}' for c in small + large)
        + f'; {len(right)}/4 with the predicted sign, {len(reliable)}/4 reliably')))

    # P3
    positive = [c for c in clans if ci_above_zero(tag[a2, c])]
    verdict = 'supported' if len(positive) == len(clans) else 'partly supported' if positive else 'not supported'
    lowest = min(clans, key=lambda c: tag[a2, c]['contrast_bits_per_symbol'])
    rows.append(dict(prediction='P3', rule=PREDICTION_RULES['P3'][1], verdict=verdict, result=(
        f'alpha {a2}: CI above zero for {len(positive)}/{len(clans)} clans; smallest {lowest} {ci(tag[a2, lowest])}; '
        f"largest {max(clans, key=lambda c: tag[a2, c]['contrast_bits_per_symbol'])} "
        f"{ci(tag[a2, max(clans, key=lambda c: tag[a2, c]['contrast_bits_per_symbol'])])}")))

    # P4
    beats = [c for c in clans if ci_above_zero(loo_vs_mean[c])]
    pointwise = [c for c in clans if loo_vs_mean[c]['contrast_bits_per_symbol'] > 0]
    verdict = 'supported' if len(beats) == len(clans) else 'partly supported' if beats else 'not supported'
    rows.append(dict(prediction='P4', rule=PREDICTION_RULES['P4'][1], verdict=verdict, result=(
        f'CI above zero for {len(beats)}/{len(clans)} clans, point estimate above zero for '
        f'{len(pointwise)}/{len(clans)}; ' + '; '.join(f'{c} {ci(loo_vs_mean[c])}' for c in clans))))

    # P5
    agree = []
    for c in clans:
        values = [efficiency[f, c]['contrast_bits_per_symbol'] for f in shares]
        a = ci_above_zero(efficiency[shares[0], c])
        b = int(np.argmax(values)) <= 1 and values[-1] < values[0]
        if a and b:
            agree.append(c)
    verdict = 'supported' if len(agree) == len(clans) else 'partly supported' if agree else 'not supported'
    rows.append(dict(prediction='P5', rule=PREDICTION_RULES['P5'][1], verdict=verdict, result=(
        f'{len(agree)}/{len(clans)} clans agree ({", ".join(agree) or "none"}); gain at f = {shares[0]}: '
        + '; '.join(f"{c} {fmt(efficiency[shares[0], c]['contrast_bits_per_symbol'])}" for c in clans)
        + f'; at f = {shares[-1]}: '
        + '; '.join(f"{c} {fmt(efficiency[shares[-1], c]['contrast_bits_per_symbol'])}" for c in clans))))
    for row in rows:
        row['prediction'] = f"{row['prediction']}: {PREDICTION_RULES[row['prediction']][0]}"
    return rows


def chosen_settings(rows):
    picked = [row for row in rows if row['selected'] == 'True']
    counts = defaultdict(int)
    for row in picked:
        counts[f"h{row['hidden']}/d{row['dropout']}" + ('' if row['learning_rate'] == '0.003' else
                                                         f"/lr{row['learning_rate']}")] += 1
    epochs = [int(row['best_epoch']) for row in picked]
    return picked, ', '.join(f'{k} x{v}' for k, v in sorted(counts.items())), epochs


def contrast_table(clans, rows_by_setting, settings, label):
    header = ['Clan'] + [f'{label} {s}' for s in settings]
    body = [[c] + [ci(rows_by_setting[s, c]) for s in settings] for c in clans]
    return table(header, body, ['---'] + ['---:'] * len(settings))


def render(r, ctx, pilot, gate0, timing, determinism, arm1_runs, gates, summaries, selections, source, matrix_out,
           rho, rho_p, primary, gain, tag, alpha_effect, capacity, capacity_gain, zero_shot, mean_single,
           loo_vs_mean, g_lstm, efficiency, curves, shares, predictions, files, low, high):
    config, clans = ctx.config, ctx.clans
    alphas = config['pooled_alphas']
    smoke = '\n**SMOKE TEST: tiny settings, the numbers are meaningless.**\n' if config['smoke_test'] else ''
    lines = [
        '# RQ3 pooled multi-clan LSTM (`[R3-8]`, DEC-27)',
        smoke,
        'Generated by `python -m whale_coda_dynamics.rq3_neural.runner --stage report` from the stage outputs in '
        f"`{config['output_directory']}/`; do not edit. Every score uses the `[R3-1]` tokeniser "
        f"(`{config['tokenizer_id']}`), the `[R3-2]` folds and Vlad's `[R3-3]` aggregation and group bootstrap "
        f"({config['bootstrap_draws']:,} draws over the target clan's recording groups). Contrasts are in bits per "
        'symbol, paired on the same codas; positive means the first model is better. Claims are limited to how '
        'much other clans\' codas help model a clan and how much knowing the clan is worth, in bits; nothing about '
        'shared ancestry, learning between clans, or meaning.',
        '',
        '## Runs',
        '',
    ]
    stage_names = ['pilot'] + arm1_runs + ['arm23', 'capacity', 'arm4', 'arm5']
    stage_rows, total_wall = [], 0.0
    for name in stage_names:
        s = pilot if name == 'pilot' else summaries[name]
        total_wall += s['wall_seconds']
        stage_rows.append([name, (s['commit'] or '')[:10], s.get('device', 'cpu+gpu' if name == 'pilot' else ''),
                           s.get('workers', ''), f"{s['fits']:,}", f"{s['wall_seconds'] / 60:.1f}",
                           f"{s['fit_seconds'] / 3600:.2f}"])
    lines += [table(['Stage', 'Commit', 'Device', 'Workers', 'Fits', 'Wall minutes', 'Fit hours (sum over workers)'],
                    stage_rows, ['---', '---', '---', '---:', '---:', '---:', '---:']), '',
              f'Total wall time of the fitting stages: {total_wall / 3600:.2f} hours.', '']

    # Gate 0
    lines += ['## Gate 0: coda counts', '',
              'Coda counts per clan and fold from the codas file, from `rq3_folds.csv` and in the encoded data. All '
              'match (the runner raises otherwise).', '']
    body = []
    for clan in clans:
        folds = [row for row in gate0 if row['clan'] == clan and row['fold'] != 'all']
        total = next(row for row in gate0 if row['clan'] == clan and row['fold'] == 'all')
        body.append([clan, len(folds), ' / '.join(row['codas_file'] for row in folds), total['codas_file'],
                     'yes' if all(row['match'] == 'True' for row in folds + [total]) else 'NO'])
    lines += [table(['Clan', 'Folds', 'Codas per fold', 'Codas', 'Match'], body,
                    ['---', '---:', '---', '---:', '---']), '']

    # Pilot
    decision = pilot['decision']
    lines += ['## Pilot, device and compute', '',
              f"Pilot fits: {config['pilot']['own_clan']} fold {config['pilot']['own_fold']} (own-clan) and pooled run "
              f"{config['pilot']['pooled_run']} (alpha {config['pilot']['pooled_alpha']}, tagged) over the full grid, "
              'then the refits; no held-out coda was scored. Throughput: '
              f"{config['pilot']['timing_epochs']}-epoch fits of the pooled run, one per worker, all running at once.",
              '']
    lines += [table(['Device', 'Workers', 'Hidden', 'Draws per second'],
                    [[t['device'], t['workers'], t['hidden'], f"{float(t['draws_per_second']):,.0f}"] for t in timing],
                    ['---', '---:', '---:', '---:']), '']
    lines += [table(['Determinism check', 'Identical'], [[d['device'], d['identical']] for d in determinism]), '']
    proj = decision['projection']
    lines += [f"Chosen: **{decision['device']} with {decision['workers']} workers**. Projected wall time "
              f"{proj['total_hours']:.2f} h (" + ', '.join(f'{k} {v:.2f} h' for k, v in proj['hours'].items())
              + '; the gate rerun and the capacity check are not included). '
              + (f"Over the {config['pilot']['budget_hours']} h budget, so arm 5 was reduced to f in "
                 f"{decision['arm5']['shares']} with {decision['arm5']['draws']} draw per f."
                 if decision['arm5']['reduced'] else
                 f"Within the {config['pilot']['budget_hours']} h budget: arm 5 runs in full (f in "
                 f"{decision['arm5']['shares']}, {decision['arm5']['draws']} draws)."), '']

    # Gate 1
    lines += ['## Gate 1: own-clan LSTM against the own-clan n-gram', '',
              'Per clan, own-clan LSTM minus own-clan n-gram (`[R3-4]` diagonal, Vlad\'s `self_scores.csv`) on the '
              'same codas. A clan fails when the CI lies entirely below zero.', '']
    for name in arm1_runs:
        title = 'First run' if name == 'arm1' else 'Rerun (grid extended by hidden 256, lr 0.001, patience 10)'
        body = [[g['clan'], g['n_codas'], g['n_groups'], f"{float(g['ngram_bits_per_symbol']):.3f}",
                 f"{float(g['lstm_bits_per_symbol']):.3f}",
                 f"{float(g['difference_bits_per_symbol']):+.3f} [{float(g['ci95_lower_bits_per_symbol']):+.3f}, "
                 f"{float(g['ci95_upper_bits_per_symbol']):+.3f}]", 'pass' if g['passed'] == 'True' else '**fail**']
                for g in gates[name]]
        lines += [f'**{title}** ({"passed" if summaries[name]["gate_passed"] else "failed"}):', '',
                  table(['Clan', 'Codas', 'Groups', 'n-gram bits/symbol', 'LSTM bits/symbol', 'LSTM - n-gram [95% CI]',
                         'Gate'], body, ['---', '---:', '---:', '---:', '---:', '---:', '---']), '']
    lines += [f'Later arms use `{source}`.', '']

    # Settings chosen
    lines += ['## Settings chosen on inner validation', '']
    body = []
    for label, name, arm in (('Arm 1 (own-clan)', source, None), ('Arm 2 (pooled, tagged)', 'arm23', 'arm2'),
                             ('Arm 3 (pooled, untagged)', 'arm23', 'arm3'), ('Arm 4 (leave one out)', 'arm4', None)):
        rows = [row for row in selections[name] if arm is None or row['arm'] == arm]
        picked, counts, epochs = chosen_settings(rows)
        body.append([label, len(picked), counts, f'{np.mean(epochs):.1f}', f'{min(epochs)}-{max(epochs)}'])
    lines += [table(['Arm', 'Models', 'Hidden/dropout chosen', 'Mean best epoch', 'Range'], body,
                    ['---', '---:', '---', '---:', '---']), '']

    # Matrix
    lines += ['## Neural 9 x 9 transfer matrix (arm 1)', '',
              'Transfer loss G(s, t) = own-clan LSTM - transfer score from s (mean of s\'s fold models), bits per '
              'symbol, rows = source, columns = target. Its rank correlation with the n-gram matrix (`[R3-4]`, the '
              f'72 off-diagonal cells): **Spearman rho = {rho:+.3f}** (p = {rho_p:.2g}).', '']
    header = ['Source \\ target'] + clans + ['Training codas']
    body = []
    for s in clans:
        row = [s]
        for t in clans:
            row.append('0' if s == t else f"{g_lstm[s, t]['lstm_transfer_loss_bits_per_symbol']:.2f}")
        row.append(next(m['source_training_codas'] for m in matrix_out if m['source_clan'] == s))
        body.append(row)
    lines += [table(header, body, ['---'] + ['---:'] * (len(clans) + 1)), '',
              'n-gram losses for the same cells, from Vlad\'s scores:', '']
    body = [[s] + ['0' if s == t else f"{g_lstm[s, t]['ngram_transfer_loss_bits_per_symbol']:.2f}" for t in clans]
            for s in clans]
    lines += [table(['Source \\ target'] + clans, body, ['---'] + ['---:'] * len(clans)), '']
    lowest = min(matrix_out, key=lambda m: m['ci95_lower_bits_per_symbol'])
    lines += [f"Neural losses with a CI that includes or lies below zero: "
              f"{sum(1 for m in matrix_out if m['ci95_lower_bits_per_symbol'] <= 0)} of 72 "
              f"(smallest lower bound: {lowest['source_clan']} -> {lowest['target_clan']}, "
              f"{lowest['lstm_transfer_loss_bits_per_symbol']:+.3f} [{lowest['ci95_lower_bits_per_symbol']:+.3f}, "
              f"{lowest['ci95_upper_bits_per_symbol']:+.3f}]). CIs for every cell are in "
              f"`{config['output_directory']}/report/matrix.csv`.", '']

    # Alpha
    alpha_scores = read_csv(output_dir(ctx, 'arm23') / 'alpha_scores.csv', None)
    lines += ['## Pooled arms: sampling exponent alpha', '',
              'Clan-averaged inner-validation bits per symbol (clans weighted equally) of each run\'s chosen fit. '
              f"Primary alpha: **arm 2 {primary['arm2']}**, **arm 3 {primary['arm3']}** (arm 4 uses arm 3's).", '']
    body = []
    for arm in ('arm2', 'arm3'):
        body.append([arm] + [f"{float(next(a['bits_per_symbol'] for a in alpha_scores if a['arm'] == arm and float(a['alpha']) == alpha and a['clan'] == 'clan_average')):.4f}"
                             for alpha in alphas])
    lines += [table(['Arm'] + [f'alpha {a}' for a in alphas], body, ['---'] + ['---:'] * len(alphas)), '']

    # Contrasts
    lines += ['## Pooling gain: pooled tagged - own-clan', '',
              'Per target clan, every alpha (bits per symbol, 95% CI). Clan sizes: '
              + ', '.join(f'{c} {r.sizes[c]:,}' for c in sorted(clans, key=lambda c: r.sizes[c])) + '.', '',
              contrast_table(clans, gain, alphas, 'alpha'), '']
    lines += ['## Tag value: pooled tagged - pooled untagged', '',
              'The conditional V-information of the clan given the coda so far, in bits per symbol.', '',
              contrast_table(clans, tag, alphas, 'alpha'), '']
    lines += [f'## Lower alpha: arm 2 at alpha {low} - arm 2 at alpha {high}', '',
              table(['Clan', f'alpha {low} - alpha {high}'], [[c, ci(alpha_effect[c])] for c in clans],
                    ['---', '---:']), '']
    lines += ['## Capacity check', '']
    if capacity['triggered_alphas']:
        triggered = capacity['triggered_alphas']
        lines += [f"Pooled tagged was reliably below own-clan for some clan at alpha {triggered}; arm 2 was rerun "
                  f"there with {capacity['hidden_multiplier']}x the chosen hidden size. Both are reported:", '']
        body = [[c] + [f'{ci(gain[a, c])} / {ci(capacity_gain[a, c])}' for a in triggered] for c in clans]
        lines += [table(['Clan'] + [f'alpha {a}: chosen / doubled' for a in triggered], body,
                        ['---'] + ['---:'] * len(triggered)), '']
    else:
        lines += ['Not triggered: at no alpha was pooled tagged reliably below own-clan for any clan.', '']

    # Zero-shot
    lines += ['## Zero-shot: leave one clan out (arm 4)', '',
              f"Untagged, trained on the other eight clans at arm 3's primary alpha ({summaries['arm4']['alpha']}). "
              'Zero-shot loss = own-clan - leave-one-out; beside it the single-source losses of arm 1 for the same '
              'target (the mean over the eight sources, and the best source, which is an oracle: it is chosen on '
              'the target\'s own scores).', '']
    body = []
    for t in clans:
        best = min((s for s in clans if s != t), key=lambda s: g_lstm[s, t]['lstm_transfer_loss_bits_per_symbol'])
        b = g_lstm[best, t]
        body.append([t, ci(zero_shot[t]), ci(mean_single[t]),
                     f"{best} {b['lstm_transfer_loss_bits_per_symbol']:+.3f} [{b['ci95_lower_bits_per_symbol']:+.3f}, "
                     f"{b['ci95_upper_bits_per_symbol']:+.3f}]", ci(loo_vs_mean[t])])
    lines += [table(['Target', 'Zero-shot loss', 'Mean single-source loss', 'Best single source (oracle)',
                     'Leave-one-out - mean single source'], body, ['---', '---:', '---:', '---', '---:']), '']

    # Data efficiency
    arm5 = summaries['arm5']
    lines += ['## Data efficiency (arm 5)', '',
              f"Arm 4's model fine-tuned on a share f of the target clan's training folds, against the same "
              'architecture from scratch on the same codas; learning rate and early stopping on the full '
              f"inner-validation fold, scored as stopped. {arm5['draws']} draw(s) per f"
              + (' (reduced by the pilot).' if arm5['reduced'] else '.')
              + f" Models stopped at epoch 0 (the starting weights): {arm5['chosen_at_epoch0']}.", '',
              '**Fine-tuned - from scratch:**', '', contrast_table(clans, efficiency, shares, 'f'), '',
              '**Bits per symbol** (fine-tuned / from scratch; zero-shot = arm 4, own-clan = arm 1 trained on all its '
              'training folds):', '']
    curve = {(c['clan'], c['share'], c['version']): float(c['bits_per_symbol']) for c in curves}
    body = [[c] + [f"{curve[c, f, 'finetuned']:.3f} / {curve[c, f, 'scratch']:.3f}" for f in shares]
            + [f"{curve[c, '', 'zero_shot']:.3f}", f"{curve[c, '', 'own_clan_full']:.3f}"] for c in clans]
    lines += [table(['Clan'] + [f'f {f}' for f in shares] + ['zero-shot', 'own-clan'], body,
                    ['---'] + ['---:'] * (len(shares) + 2)), '']

    # Predictions
    lines += ['## Predictions against results', '',
              'One row per `[R3-8]` prediction (written before running). The rule column gives the test, fixed in '
              '`rq3_neural/report.py` before any arm ran.', '',
              table(['Prediction', 'Rule', 'Result', 'Verdict'],
                    [[p['prediction'], p['rule'], p['result'], f"**{p['verdict']}**"] for p in predictions]), '']

    # Figures
    figures = os.path.relpath(ctx.root / config['figure_directory'], (ctx.root / config['report_output']).parent)
    lines += ['## Figures', '',
              f"Written by `python -m whale_coda_dynamics.rq3_neural.plots` from `{config['output_directory']}/report/`.",
              '']
    for name, caption in FIGURES:
        lines += [f'![{caption}]({Path(figures, name).as_posix()})', '']

    # Interpretations
    lines += ['## Interpretations', '',
              'Where `[R3-8]` leaves a choice open, this run takes the reading closest to PLAN:', '']
    lines += [f'{i}. ' + text.format(draws=config['bootstrap_draws'], seed=config['bootstrap_seed'])
              for i, text in enumerate(INTERPRETATIONS, 1)]
    lines += ['', '## Checks built into the runner', '',
              '- Every model\'s training and validation codas are disjoint from the codas it scores and from its '
              'plan\'s held-out codas (fine-tuned models: arm 4\'s training data too).',
              '- Leave-one-out models contain no coda of the left-out clan.',
              '- Tagged and untagged models differ only in the start symbol: same inputs after it, same training '
              'draws, seeds, settings and initial weights (every candidate pair, and the refits where both chose '
              'the same setting; `arm23/tag_pairs.csv`).',
              '- Every scored coda\'s n_symbols equals Vlad\'s `token_count` in every model.',
              '- In every stage one refit is fitted a second time and must give identical scores and weights.', '']

    # Files
    lines += ['## Files', '',
              f"Stage outputs in `{config['output_directory']}/` (checkpoints of arm 4's models are not committed). "
              'Figures: `python -m whale_coda_dynamics.rq3_neural.plots` (matplotlib) writes them to '
              f"`{config['figure_directory']}/`.", '',
              table(['File', 'MB'], [[name, f'{size / 1e6:.2f}'] for name, size in files], ['---', '---:']), '']
    return '\n'.join(lines)
