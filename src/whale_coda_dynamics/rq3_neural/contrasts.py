"""Paired [R3-8] contrasts through Vlad's [R3-3] aggregation and group bootstrap.

Thin wrappers only: they put scores into the row format of ``rq3.experiments.evaluation`` and call
``mean_bits_per_symbol``, ``mean_bits_per_coda``, ``paired_transfer_loss``,
``aggregate_transfer_loss`` and ``bootstrap_group_mean`` (seeded with ``uncertainty.pair_seed``,
CIs by ``uncertainty.percentile``). Vlad's functions compute own - transfer; a contrast
"first - second" passes the first model as "own", so positive means the first model is better.

Score convention (Vlad's): a coda's score is its log2 probability (``bits_per_coda``, negative,
higher is better) and ``token_count`` is its number of predicted symbols (intervals + EOS). In this
package's score rows they are ``ll_bits`` and ``n_symbols`` (PLAN [R3-3]).
"""
from whale_coda_dynamics.rq3.experiments.evaluation import (
    aggregate_transfer_loss,
    bootstrap_group_mean,
    mean_bits_per_coda,
    mean_bits_per_symbol,
    paired_transfer_loss,
)
from whale_coda_dynamics.rq3.experiments.uncertainty import pair_seed, percentile


def vlad_rows(scores):
    """{coda_id: (ll_bits, n_symbols)} -> Vlad's score-row dictionaries."""
    return [{'rq3_coda_id': coda_id, 'bits_per_coda': ll, 'token_count': n, 'bits_per_symbol': ll / n}
            for coda_id, (ll, n) in scores.items()]


def bits_per_symbol(scores):
    return mean_bits_per_symbol(vlad_rows(scores))


def bits_per_coda(scores):
    return mean_bits_per_coda(vlad_rows(scores))


def _interval(samples, key):
    values = [sample[key] for sample in samples]
    return percentile(values, 0.025), percentile(values, 0.975)


def paired_contrast(first, second, group_of, draws, seed, labels):
    """first - second on the same codas, per symbol and per coda, with group-bootstrap 95% CIs.

    first, second: {coda_id: (ll_bits, n_symbols)}; group_of: {coda_id: group_id}."""
    if set(first) != set(second):
        raise ValueError(f'{labels}: the two models scored different codas')
    for coda_id, (_, n) in first.items():
        if second[coda_id][1] != n:
            raise ValueError(f'{labels}: {coda_id} has {n} symbols in one model and {second[coda_id][1]} in the other')
    loss_rows = paired_transfer_loss(vlad_rows(first), vlad_rows(second))
    for row in loss_rows:
        row['group_id'] = group_of[row['rq3_coda_id']]
    point = aggregate_transfer_loss(loss_rows)
    samples = bootstrap_group_mean(loss_rows, draws, pair_seed(seed, *map(str, labels)))
    lo_s, hi_s = _interval(samples, 'mean_transfer_loss_bits_per_symbol')
    lo_c, hi_c = _interval(samples, 'mean_transfer_loss_bits_per_coda')
    return dict(n_codas=len(loss_rows), n_groups=len({row['group_id'] for row in loss_rows}),
                n_symbols=sum(row['n_symbols'] for row in loss_rows),
                first_bits_per_symbol=bits_per_symbol(first), second_bits_per_symbol=bits_per_symbol(second),
                contrast_bits_per_symbol=point['mean_transfer_loss_bits_per_symbol'],
                ci95_lower_bits_per_symbol=lo_s, ci95_upper_bits_per_symbol=hi_s,
                contrast_bits_per_coda=point['mean_transfer_loss_bits_per_coda'],
                ci95_lower_bits_per_coda=lo_c, ci95_upper_bits_per_coda=hi_c,
                bootstrap_draws=len(samples))


def grouped_contrast(first, second, draws, seed, labels):
    """first - second from per-group sums (arm 5): {group_id: (ll_bits_sum, n_symbols)} each.

    One row per group in Vlad's functions, so the per-symbol estimate and its bootstrap are exact;
    the per-coda numbers would weight groups, not codas, and are not reported."""
    if set(first) != set(second):
        raise ValueError(f'{labels}: the two models scored different groups')
    rows = []
    for group in sorted(first):
        (ll1, n1), (ll2, n2) = first[group], second[group]
        if n1 != n2:
            raise ValueError(f'{labels}: group {group} has different symbol counts')
        rows.append({'group_id': group, 'n_symbols': n1, 'own_bits_per_coda': ll1, 'transfer_bits_per_coda': ll2,
                     'transfer_loss_bits_per_coda': ll1 - ll2, 'transfer_loss_bits_per_symbol': (ll1 - ll2) / n1})
    point = aggregate_transfer_loss(rows)
    samples = bootstrap_group_mean(rows, draws, pair_seed(seed, *map(str, labels)))
    lo, hi = _interval(samples, 'mean_transfer_loss_bits_per_symbol')
    total = sum(n for _, n in first.values())
    return dict(n_groups=len(rows), n_symbols=total,
                first_bits_per_symbol=sum(ll for ll, _ in first.values()) / total,
                second_bits_per_symbol=sum(ll for ll, _ in second.values()) / total,
                contrast_bits_per_symbol=point['mean_transfer_loss_bits_per_symbol'],
                ci95_lower_bits_per_symbol=lo, ci95_upper_bits_per_symbol=hi, bootstrap_draws=len(samples))


def ci_below_zero(row):
    return row['ci95_upper_bits_per_symbol'] < 0


def ci_above_zero(row):
    return row['ci95_lower_bits_per_symbol'] > 0
