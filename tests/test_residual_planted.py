import math
import unittest

import numpy as np

from whale_coda_dynamics.evaluation import day_bootstrap_bits
from whale_coda_dynamics.residual_planted import (oracle_probabilities, paired_ratio_bootstrap, plant, relabel,
                                                  summarise)

VOCAB = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'UNK']


def history_rows(n, seed):
    rng = np.random.RandomState(seed)
    return [dict(coda_id=f'r{i}', Ct=VOCAB[rng.randint(7)], A_history=[VOCAB[rng.randint(7)] for _ in range(1 + rng.randint(10))])
            for i in range(n)]


class PlantingTests(unittest.TestCase):
    def test_nested_eligible_and_fixed_per_fold_and_split(self):
        rows = history_rows(400, 0)
        eligible, planted = plant(rows, 6, [0.1, 0.2, 0.4], 42, 'test-x', 'train')
        self.assertTrue(eligible)
        for i in eligible:
            history = rows[i]['A_history']
            self.assertTrue(len(history) >= 6 and history[-6] not in history[-3:])
        self.assertEqual([i for i, r in enumerate(rows) if len(r['A_history']) >= 6 and r['A_history'][-6] not in r['A_history'][-3:]],
                         eligible)
        for f in (0.1, 0.2, 0.4):
            self.assertEqual(len(planted[f]), int(math.floor(f * len(eligible) + 0.5)))
            self.assertTrue(set(planted[f]) <= set(eligible))
        self.assertTrue(set(planted[0.1]) <= set(planted[0.2]) <= set(planted[0.4]))
        self.assertEqual(plant(rows, 6, [0.4], 42, 'test-x', 'train')[1][0.4], planted[0.4])
        self.assertNotEqual(plant(rows, 6, [0.4], 42, 'test-x', 'test')[1][0.4], planted[0.4])
        self.assertNotEqual(plant(rows, 6, [0.4], 42, 'test-y', 'train')[1][0.4], planted[0.4])

    def test_relabel_changes_labels_only(self):
        rows = history_rows(50, 1)
        eligible, planted = plant(rows, 6, [0.4], 42, 'f', 'test')
        out = relabel(rows, planted[0.4], 6)
        for i, (before, after) in enumerate(zip(rows, out)):
            self.assertEqual(before['A_history'], after['A_history'])
            self.assertEqual(after['Ct'], before['A_history'][-6] if i in planted[0.4] else before['Ct'])

    def test_oracle_mixes_rule_into_eligible_rows_only(self):
        rows = history_rows(60, 2)
        eligible, _ = plant(rows, 6, [0.2], 42, 'f', 'test')
        lr = np.random.RandomState(3).dirichlet(np.ones(len(VOCAB)), size=len(rows))
        p = oracle_probabilities(lr, rows, eligible, 0.2, 6, VOCAB)
        np.testing.assert_allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
        for i, row in enumerate(rows):
            if i in eligible:
                expected = 0.8 * lr[i]
                expected[VOCAB.index(row['A_history'][-6])] += 0.2
                np.testing.assert_allclose(p[i], expected, rtol=0, atol=1e-15)
            else:
                self.assertTrue(np.array_equal(p[i], lr[i]))


def days_of(values_by_day):
    return {d: list(v) for d, v in values_by_day.items()}


class RatioBootstrapTests(unittest.TestCase):
    def test_same_resampling_as_day_bootstrap_and_ratio_of_means(self):
        rng = np.random.RandomState(4)
        den = {f'2020-01-{d:02d}': list(rng.uniform(0.1, 2.0, size=1 + d)) for d in range(1, 9)}
        num = {d: [0.5 * v + rng.normal(scale=0.01) for v in values] for d, values in den.items()}
        out = paired_ratio_bootstrap(num, den, 300, 7, 0.95)
        self.assertEqual(out['numerator_ci'], day_bootstrap_bits(num, 300, 7, 0.95)['ci_bits_per_coda'])
        self.assertEqual(out['denominator_ci'], day_bootstrap_bits(den, 300, 7, 0.95)['ci_bits_per_coda'])
        total = lambda d: math.fsum(v for values in d.values() for v in values)  # noqa: E731
        self.assertAlmostEqual(out['ratio'], total(num) / total(den), places=12)  # ratio of means, not mean of day ratios
        self.assertEqual(out['degenerate_replicates'], 0)
        self.assertLess(out['ci_ratio'][0], 0.5)
        self.assertGreater(out['ci_ratio'][1], 0.5)

    def test_non_positive_denominators_are_counted(self):
        # One day with planted targets (oracle gains), four with eligible but unplanted targets (oracle loses log2(1-f))
        den = {'d1': [3.0, 3.0], 'd2': [-0.5] * 4, 'd3': [-0.5] * 4, 'd4': [-0.5] * 4, 'd5': [-0.5] * 4}
        num = {d: [0.5 * v for v in values] for d, values in den.items()}
        out = paired_ratio_bootstrap(num, den, 500, 1, 0.95)
        self.assertGreater(out['degenerate_replicates'], 0)
        self.assertLess(out['degenerate_replicates'], 500)
        self.assertEqual(out['replicates'], 500)
        self.assertAlmostEqual(out['ci_ratio'][0], 0.5)


def synthetic_scores(oracle_by_day, residual_share=0.5):
    """One planted and one unplanted target per day; the oracle gains `oracle` bits on the planted one."""
    scores = []
    for day, oracle in oracle_by_day.items():
        scores.append(dict(fraction=0.4, recording_date=day, eligible=True, planted=True, lr_bits=-3.0,
                           residual_bits=[-3.0 + residual_share * oracle] * 2, oracle_bits=-3.0 + oracle))
        scores.append(dict(fraction=0.4, recording_date=day, eligible=False, planted=False, lr_bits=-1.0,
                           residual_bits=[-1.0, -1.0], oracle_bits=-1.0, real_residual_bits=[-1.0, -1.1]))
    return scores


class SummaryTests(unittest.TestCase):
    config = dict(fractions=[0.4], bootstrap_replicates=400, bootstrap_seed=42, confidence=0.95, ratio_handful=5)
    real = dict(residual_gain=dict(A=dict(ci_bits_per_coda=[-0.05, 0.0071])))
    selections = [dict(fraction=0.4, chosen='A/h16/d0.4')]

    def test_bound_only_where_the_denominator_stays_positive(self):
        stable = summarise(synthetic_scores({f'2020-01-0{d}': 2.0 + 0.1 * d for d in range(1, 8)}), self.selections, [],
                           self.config, self.real)['per_fraction']['0.4']
        self.assertEqual(stable['recovery']['status'], 'stable')
        self.assertAlmostEqual(stable['bound_bits_per_coda'], 0.0071 / stable['recovery']['ci_ratio'][0])
        unstable = summarise(synthetic_scores({'2020-01-01': 4.0, **{f'2020-01-0{d}': -0.6 for d in range(2, 8)}}),
                             self.selections, [], self.config, self.real)['per_fraction']['0.4']
        self.assertEqual(unstable['recovery']['status'], 'unstable')
        self.assertGreater(unstable['recovery']['degenerate_replicates'], 5)
        self.assertIsNone(unstable['bound_bits_per_coda'])


if __name__ == '__main__':
    unittest.main()
