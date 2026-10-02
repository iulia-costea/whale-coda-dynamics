"""RQ2 comparison arithmetic on synthetic inputs: shared day draws, shares, paired share
differences, the tempo decomposition and UNK exclusion (PLAN [R2-3]-[R2-5])."""
import math
import random
import unittest

from whale_coda_dynamics.evaluation import day_bootstrap_bits
from whale_coda_dynamics.rq2_compare import (
    CONDITIONS, Inventory, compare, decomposition, describe_values, shared_day_draws, type_and_tempo_bits,
)

DAYS = [f'2020-01-{d:02d}' for d in range(1, 8)]


def synthetic(ident, ll, dates, targets=None, unk='UNK'):
    """An Inventory with only what compare() and decomposition() read; no files."""
    inv = Inventory.__new__(Inventory)
    inv.id, inv.ll, inv.dates, inv.unk = ident, ll, dates, unk
    inv.target = targets or {c: 'x' for c in dates}
    inv.target_raw = dict(inv.target)
    return inv


def lattice(dates, rng, total=None, own=None):
    """Per-coda LL for the eight conditions with chosen T and own-history contrast per coda."""
    ll = {}
    for c in dates:
        base = -3.0 + rng.random()
        t = total[c] if total else 1.0 + rng.random()
        o = own[c] if own else rng.uniform(-0.2, 0.8) * t
        ll['none', c], ll['A_B_S', c] = base, base + t
        ll['B_S', c] = base + t - o
        ll['A_S', c], ll['A_B', c] = base + t - rng.uniform(-0.1, 0.1), base + t - rng.uniform(-0.1, 0.1)
        for k in ('S', 'A', 'B'):
            ll[k, c] = base + rng.random()
    return ll


class SharedDrawTests(unittest.TestCase):
    def test_describe_matches_the_shared_evaluation_bootstrap(self):
        rng = random.Random(1)
        dates = {f'c{i}': DAYS[i % len(DAYS)] for i in range(60)}
        values = {c: rng.gauss(0.1, 1.0) for c in dates}
        ours = describe_values(values, dates, shared_day_draws(DAYS, 500, 42), 0.95)
        by_day = {d: [values[c] for c in sorted(dates) if dates[c] == d] for d in DAYS}
        theirs = day_bootstrap_bits(by_day, 500, 42, 0.95)
        for field in ('mean_bits_per_coda', 'unweighted_mean_bits_per_coda'):
            self.assertAlmostEqual(ours[field], theirs[field], places=12)
        self.assertEqual(ours['ci_bits_per_coda'], theirs['ci_bits_per_coda'])
        self.assertEqual(ours['positive_days'], theirs['positive_days'])


class ShareTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(7)
        self.dates = {f'c{i}': DAYS[i % len(DAYS)] for i in range(70)}
        self.draws = shared_day_draws(DAYS, 300, 42)

    def test_share_is_a_ratio_of_token_weighted_means(self):
        ll = lattice(self.dates, self.rng)
        result = compare({'A': synthetic('A', ll, self.dates)}, 'A', self.draws, 0.95)['A']
        own = math.fsum(ll['A_B_S', c] - ll['B_S', c] for c in self.dates)
        total = math.fsum(ll['A_B_S', c] - ll['none', c] for c in self.dates)
        self.assertAlmostEqual(result['shares']['own_history']['estimate'], own / total, places=12)
        self.assertAlmostEqual(result['contrasts']['own_history']['mean_bits_per_coda'] / result['T']['mean_bits_per_coda'],
                               own / total, places=12)
        lo, hi = result['shares']['own_history']['ci']
        self.assertLessEqual(lo, own / total)
        self.assertLessEqual(own / total, hi)

    def test_constant_share_has_a_degenerate_interval(self):
        total = {c: 1.0 + self.rng.random() for c in self.dates}
        ll = lattice(self.dates, self.rng, total, own={c: 0.4 * total[c] for c in self.dates})
        share = compare({'A': synthetic('A', ll, self.dates)}, 'A', self.draws, 0.95)['A']['shares']['own_history']
        self.assertAlmostEqual(share['estimate'], 0.4, places=12)
        self.assertAlmostEqual(share['ci'][0], 0.4, places=12)
        self.assertAlmostEqual(share['ci'][1], 0.4, places=12)

    def test_share_differences_are_paired_on_identical_day_draws(self):
        # B's own-history contrast is A's plus 0.1 T on every coda, with the same T: the share difference is
        # exactly 0.1 in every replicate only if both inventories use the same resampled days.
        total = {c: 1.0 + self.rng.random() for c in self.dates}
        own = {c: self.rng.uniform(-0.5, 1.0) * total[c] for c in self.dates}
        a = lattice(self.dates, random.Random(3), total, own)
        b = dict(a)
        for c in self.dates:
            b['B_S', c] = a['B_S', c] - 0.1 * total[c]
        result = compare({'A': synthetic('A', a, self.dates), 'B': synthetic('B', b, self.dates)}, 'A', self.draws, 0.95)
        difference = result['B']['share_differences_vs_reference']['own_history']
        self.assertAlmostEqual(difference['estimate'], 0.1, places=12)
        self.assertAlmostEqual(difference['ci'][0], 0.1, places=10)
        self.assertAlmostEqual(difference['ci'][1], 0.1, places=10)
        self.assertGreater(result['A']['shares']['own_history']['ci'][1] - result['A']['shares']['own_history']['ci'][0], 0.01)
        self.assertIsNone(result['A']['share_differences_vs_reference'])

    def test_leaving_out_a_day_drops_its_codas(self):
        ll = lattice(self.dates, self.rng)
        kept = DAYS[1:]
        result = compare({'A': synthetic('A', ll, self.dates)}, 'A', shared_day_draws(kept, 100, 42), 0.95, set(kept))['A']
        self.assertEqual(result['T']['days'], len(kept))
        self.assertEqual(result['T']['codas'], sum(d != DAYS[0] for d in self.dates.values()))


class DecompositionTests(unittest.TestCase):
    def test_type_plus_tempo_is_the_label_log_likelihood(self):
        probs = {'a|t0': 0.1, 'a|t1': 0.3, 'b|t1': 0.2, 'UNK': 0.4}
        type_bits, tempo_bits = type_and_tempo_bits(probs, 'a|t1', 'UNK')
        self.assertAlmostEqual(type_bits, math.log2(0.4), places=12)
        self.assertAlmostEqual(tempo_bits, math.log2(0.3 / 0.4), places=12)
        self.assertAlmostEqual(type_bits + tempo_bits, math.log2(0.3), places=12)
        with self.assertRaises(ValueError):
            type_and_tempo_bits(probs, 'UNK', 'UNK')

    def test_unk_targets_are_left_out_and_counted(self):
        codas = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
        dates = {c: DAYS[i % 3] for i, c in enumerate(codas)}
        coarse_targets = {'c0': 'a', 'c1': 'b', 'c2': 'UNK', 'c3': 'UNK', 'c4': 'a', 'c5': 'b'}
        fine_targets = {'c0': 'a|t0', 'c1': 'UNK', 'c2': 'b|t1', 'c3': 'UNK', 'c4': 'a|t1', 'c5': 'b|t0'}
        probs = {'a|t0': 0.3, 'a|t1': 0.2, 'b|t0': 0.25, 'b|t1': 0.15, 'UNK': 0.1}
        coarse_probs = {'a': 0.6, 'b': 0.3, 'UNK': 0.1}
        fine_ll = {(k, c): math.log2(probs[fine_targets[c]] if fine_targets[c] != 'UNK' else 0.1) for k in CONDITIONS for c in codas}
        coarse_ll = {(k, c): math.log2(coarse_probs[coarse_targets[c]]) for k in CONDITIONS for c in codas}
        fine = synthetic('E', fine_ll, dates, fine_targets)
        coarse = synthetic('A', coarse_ll, dates, coarse_targets)
        predictions = [dict(coda_id=c, condition=k, probabilities=probs) for k in CONDITIONS for c in codas]
        result = decomposition(coarse, fine, predictions, shared_day_draws(DAYS[:3], 50, 42), 0.95)
        self.assertEqual(result['eligible_targets'], 3)
        self.assertEqual(result['excluded_targets'], {'fine_only': 1, 'coarse_only': 1, 'both': 1})
        # Type term = log2 p(a) under E's probabilities (0.5) against A's log2 0.6, etc.; tempo contrasts are zero
        # because every condition shares one probability vector.
        expected = (math.log2(0.5) - math.log2(0.6) + math.log2(0.5) - math.log2(0.6) + math.log2(0.4) - math.log2(0.3)) / 3
        self.assertAlmostEqual(result['type_minus_coarse']['A_B_S']['mean_bits_per_coda'], expected, places=12)
        self.assertEqual(result['tempo_contrasts']['own_history']['mean_bits_per_coda'], 0.0)
        broken = synthetic('E', fine_ll, dates, dict(fine_targets, c0='b|t0'))
        with self.assertRaises(ValueError):
            decomposition(coarse, broken, predictions, shared_day_draws(DAYS[:3], 50, 42), 0.95)


if __name__ == '__main__':
    unittest.main()
