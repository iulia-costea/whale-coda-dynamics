"""Figures for reports/rq3_neural.md, from the report stage's CSVs (matplotlib; not needed by the tests).

    PYTHONPATH=src python -m whale_coda_dynamics.rq3_neural.plots --config config/rq3_neural.json

Writes five PNGs to the config's figure_directory:
- pooling_gain_vs_size.png: pooling gain against clan size, one line per alpha;
- tag_value.png: tag value per clan, one marker per alpha;
- zero_shot.png: zero-shot loss beside the eight single-source losses, their mean and the best source;
- data_efficiency.png: bits per symbol against the share f, one panel per clan;
- matrix_vs_ngram.png: neural against n-gram transfer loss, the 72 off-diagonal cells.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from whale_coda_dynamics.rq3.experiments.data import read_csv  # noqa: E402

from .data import load_config  # noqa: E402

SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']           # categorical slots 1-3
ORDINAL = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']  # blue 250 / 400 / 550 / 700
NEUTRAL = '#a3a29c'


def style(ax, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def zero_line(ax):
    ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)


def figure(width, height, **kwargs):
    fig, axes = plt.subplots(figsize=(width, height), facecolor=SURFACE, **kwargs)
    return fig, axes


def save(fig, path, title):
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def err(rows):
    return [[r['value'] - r['lo'] for r in rows], [r['hi'] - r['value'] for r in rows]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default='config/rq3_neural.json')
    args = parser.parse_args(argv)
    _, root, config = load_config(args.config)
    report = root / config['output_directory'] / 'report'
    out = root / config['figure_directory']
    out.mkdir(parents=True, exist_ok=True)
    contrasts = defaultdict(dict)
    for row in read_csv(report / 'contrasts.csv', None):
        contrasts[row['contrast'], row['setting']][row['clan']] = dict(
            value=float(row['contrast_bits_per_symbol']), lo=float(row['ci95_lower_bits_per_symbol']),
            hi=float(row['ci95_upper_bits_per_symbol']))
    sizes = {row['clan']: int(row['n_codas']) for row in read_csv(report / 'clan_sizes.csv', None)}
    clans = sorted(sizes, key=lambda c: sizes[c])
    alphas = config['pooled_alphas']

    # Pooling gain against clan size, one line per alpha.
    fig, ax = figure(6.4, 4.0)
    for i, alpha in enumerate(alphas):
        rows = [contrasts['pooling_gain', str(alpha)][c] for c in clans]
        ax.errorbar([sizes[c] for c in clans], [r['value'] for r in rows], yerr=err(rows), color=ORDINAL[i % 4],
                    linewidth=2, marker='o', markersize=5, capsize=0, elinewidth=1, label=f'alpha {alpha}')
    for c in clans:
        ax.annotate(c, (sizes[c], contrasts['pooling_gain', str(alphas[0])][c]['value']), textcoords='offset points',
                    xytext=(4, 6), fontsize=7, color=INK_2)
    ax.set_xscale('log')
    ax.set_xticks([1000, 2000, 4000, 8000], ['1,000', '2,000', '4,000', '8,000'])
    ax.minorticks_off()
    zero_line(ax)
    style(ax, 'Clan size (codas, log scale)', 'Pooled tagged - own-clan (bits/symbol)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, out / 'pooling_gain_vs_size.png', 'Pooling gain against clan size (95% CI)')

    # Tag value per clan, one marker per alpha.
    fig, ax = figure(6.4, 3.6)
    offsets = [(i - (len(alphas) - 1) / 2) * 0.16 for i in range(len(alphas))]
    for i, alpha in enumerate(alphas):
        rows = [contrasts['tag_value', str(alpha)][c] for c in clans]
        ax.errorbar([j + offsets[i] for j in range(len(clans))], [r['value'] for r in rows], yerr=err(rows),
                    color=ORDINAL[i % 4], linestyle='none', marker='o', markersize=5, elinewidth=1,
                    label=f'alpha {alpha}')
    ax.set_xticks(range(len(clans)), clans)
    zero_line(ax)
    style(ax, 'Clan (smallest to largest)', 'Tagged - untagged (bits/symbol)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=len(alphas))
    save(fig, out / 'tag_value.png', 'Tag value per clan (95% CI)')

    # Zero-shot loss beside the single-source losses.
    matrix = read_csv(report / 'matrix.csv', None)
    fig, ax = figure(6.4, 3.8)
    for j, t in enumerate(clans):
        singles = [float(m['lstm_transfer_loss_bits_per_symbol']) for m in matrix if m['target_clan'] == t]
        ax.scatter([j - 0.12] * len(singles), singles, s=14, color=NEUTRAL, zorder=2,
                   label='single source (arm 1)' if j == 0 else None)
        mean = contrasts['mean_single_source_loss', 'arm1'][t]
        ax.scatter([j - 0.12], [mean['value']], s=40, marker='_', color=INK, linewidths=2, zorder=3,
                   label='mean single source' if j == 0 else None)
        z = contrasts['zero_shot_loss', 'arm4'][t]
        ax.errorbar([j + 0.12], [z['value']], yerr=[[z['value'] - z['lo']], [z['hi'] - z['value']]],
                    color=SERIES[0], marker='o', markersize=6, elinewidth=1,
                    label='leave one out (arm 4)' if j == 0 else None)
    ax.set_xticks(range(len(clans)), clans)
    zero_line(ax)
    style(ax, 'Target clan (smallest to largest)', 'Own-clan - model (bits/symbol)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, out / 'zero_shot.png', 'Zero-shot loss beside the single-source losses (lower is better)')

    # Data-efficiency curves, one panel per clan.
    curves = defaultdict(dict)
    for row in read_csv(report / 'data_efficiency.csv', None):
        curves[row['clan']][row['version'], row['share']] = float(row['bits_per_symbol'])
    shares = sorted({float(s) for c in curves.values() for (_, s) in c if s})
    fig, axes = figure(9.6, 8.4, nrows=3, ncols=3, sharex=True)
    for ax, clan in zip(axes.flat, clans):
        c = curves[clan]
        for i, version in enumerate(('finetuned', 'scratch')):
            ax.plot(shares, [c[version, str(f)] for f in shares], color=SERIES[i], linewidth=2, marker='o',
                    markersize=4, label='fine-tuned from arm 4' if version == 'finetuned' else 'from scratch')
        ax.axhline(c['zero_shot', ''], color=SERIES[0], linewidth=1, linestyle='--', label='zero-shot (arm 4)')
        ax.axhline(c['own_clan_full', ''], color=INK_2, linewidth=1, linestyle=':', label='own-clan (arm 1)')
        ax.set_xscale('log')
        ax.set_xticks(shares, [f'{f:g}' for f in shares])
        ax.minorticks_off()
        ax.set_title(f'{clan} ({sizes[clan]:,} codas)', color=INK, fontsize=9, loc='left')
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel('Share f of the training folds (log scale)', color=INK_2, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('Bits per symbol', color=INK_2, fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8, labelcolor=INK_2)
    fig.subplots_adjust(bottom=0.1)
    fig.suptitle('Data efficiency: held-out bits per symbol (higher is better)', color=INK, fontsize=11, x=0.01,
                 ha='left')
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out / 'data_efficiency.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Neural against n-gram matrix.
    fig, ax = figure(4.8, 4.4)
    x = [float(m['ngram_transfer_loss_bits_per_symbol']) for m in matrix]
    y = [float(m['lstm_transfer_loss_bits_per_symbol']) for m in matrix]
    ax.scatter(x, y, s=22, color=SERIES[0], edgecolors=SURFACE, linewidths=1, zorder=3, label='source -> target cell')
    low, high = min(x + y + [0]), max(x + y)
    ax.plot([low, high], [low, high], color=NEUTRAL, linewidth=1, linestyle='--', label='equal loss', zorder=2)
    zero_line(ax)
    style(ax, 'n-gram transfer loss (bits/symbol)', 'LSTM transfer loss (bits/symbol)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, out / 'matrix_vs_ngram.png', 'Neural against n-gram transfer matrix (72 cells)')
    print(f'Figures written to {out}')


if __name__ == '__main__':
    main()
