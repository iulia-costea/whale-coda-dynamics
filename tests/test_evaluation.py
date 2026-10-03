from copy import deepcopy
import math
import unittest

from whale_coda_dynamics.evaluation import (
    evaluate, log_likelihood_bits, repeat_switch_bits, day_bootstrap_bits, classification_metrics, sign_flip_p, holm,
)


def fixture(seeds=(1,)):
    conditions = ['none', 'A', 'B', 'S', 'A_B', 'A_S', 'B_S', 'A_B_S']
    targets, folds, predictions = [], [], []
    for i in range(2):
        day, cid = f'2020-01-0{i+1}', str(i)
        targets.append(dict(coda_id=cid, Ct_raw='a', A_history_ids=[cid],
                            comparison_eligible=True, recording_date=day))
        folds.append(dict(fold_id=cid, test_day=day, comparison_ids={'test': [cid]},
                          vocabulary_policy={'vocabulary': ['a', 'UNK'], 'unk_token': 'UNK'}))
        for c in conditions:
            for seed in seeds:
                predictions.append(dict(coda_id=cid, fold_id=cid, condition=c, seed=seed,
                                        probabilities={'a': 0.5, 'UNK': 0.5}))
    return predictions, {'folds': folds, 'input_conditions': dict.fromkeys(conditions)}, targets


class EvaluationTests(unittest.TestCase):
    def test_bits_and_decomposition(self):
        self.assertEqual(log_likelihood_bits(0.25), -2)
        p = {'a': .6, 'b': .3, 'c': .1}
        for target in p:
            r, s = repeat_switch_bits(p, target, 'a')
            self.assertAlmostEqual(r+s, math.log2(p[target]))
        self.assertEqual(repeat_switch_bits(p, 'a', None), (None, None))
        self.assertEqual(repeat_switch_bits({'a': 1}, 'a', 'a'), (0, 0))
        for bad in (0, -1, float('nan'), 1.1):
            with self.assertRaises(ValueError):
                log_likelihood_bits(bad)

    def test_day_weighting_and_repeatability(self):
        a = day_bootstrap_bits({'a': [1], 'b': [-1, -1, -1]}, 100, 7)
        self.assertEqual(a['mean_bits_per_coda'], -.5)
        self.assertEqual(a['positive_days'], 1)
        self.assertEqual(a, day_bootstrap_bits({'b': [-1]*3, 'a': [1]}, 100, 7))
        self.assertEqual(day_bootstrap_bits({'a': [2], 'b': [2]*3}, 100)['ci_bits_per_coda'], [2, 2])

    def test_full_pairing_seed_log_average_and_lattice(self):
        p, m, t = fixture((1, 2))
        for r in p:
            if r['condition'] == 'A_B_S' and r['seed'] == 2:
                r['probabilities'] = {'a': .25, 'UNK': .75}
        summary, scored, pairs = evaluate(p, m, t, [1, 2], 50)
        self.assertEqual(summary['metrics']['A_B_S']['mean_log_likelihood_bits'], -1.5)
        self.assertEqual(summary['contrasts']['own_history']['components']['log_likelihood_bits']['mean_bits_per_coda'], -.5)
        self.assertEqual(sum(k.startswith('add_input:') for k in summary['contrasts']), 12)
        for row in pairs:
            self.assertAlmostEqual(row['log_likelihood_bits'], row['repeat_bits'] + row['switch_choice_bits'])
        self.assertEqual(len(scored), 32)

    def test_reject_missing_duplicate_and_invalid_probabilities(self):
        p, m, t = fixture()
        for bad in (p[:-1], p + [p[0]]):
            with self.assertRaises(ValueError):
                evaluate(bad, m, t, [1], 10)
        for probs in ({'a': .5}, {'a': .5, 'UNK': .6}, {'a': 0, 'UNK': 1}):
            bad = deepcopy(p)
            bad[0]['probabilities'] = probs
            with self.assertRaises(ValueError):
                evaluate(bad, m, t, [1], 10)

    def test_classification(self):
        result = classification_metrics(['a', 'a', 'b'], ['a', 'b', 'b'])
        self.assertAlmostEqual(result['macro_f1'], 2/3)
        self.assertAlmostEqual(result['accuracy'], 2/3)


class SignFlipTests(unittest.TestCase):
    """Exact sign-flip p-values (DEC-23), every case enumerated by hand in the comments."""

    def test_two_days_each_alternative(self):
        # Sums a: +1 (1 token), b: -3 (3 tokens); T = (s_a*1 + s_b*(-3)) / 4.
        # (+,+) -0.5 observed; (+,-) +1.0; (-,+) -1.0; (-,-) +0.5.
        sums, n = {'a': 1.0, 'b': -3.0}, {'a': 1, 'b': 3}
        self.assertEqual(sign_flip_p(sums, n, 'less'), 2 / 4)       # -0.5, -1.0
        self.assertEqual(sign_flip_p(sums, n, 'greater'), 3 / 4)    # -0.5, +1.0, +0.5
        self.assertEqual(sign_flip_p(sums, n, 'two-sided'), 4 / 4)  # every |T| >= 0.5
        self.assertEqual(sign_flip_p(sums, n), sign_flip_p(sums, n, 'less'))

    def test_ties_at_the_observed_value_count_as_extreme(self):
        # c's sum is exactly zero, so flipping c reproduces every statistic: each value occurs twice.
        # Values: (+,+) 4/N twice (observed), (+,-) and (-,+) 0 four times, (-,-) -4/N twice.
        sums, n = {'a': 2.0, 'b': 2.0, 'c': 0.0}, {'a': 2, 'b': 3, 'c': 5}
        self.assertEqual(sign_flip_p(sums, n, 'less'), 8 / 8)
        self.assertEqual(sign_flip_p(sums, n, 'greater'), 2 / 8)
        self.assertEqual(sign_flip_p(sums, n, 'two-sided'), 4 / 8)  # +4/N twice and -4/N twice
        # Two equal day sums: (+,+) and (-,-) tie in absolute value with the observed +2/N.
        self.assertEqual(sign_flip_p({'a': 1.0, 'b': 1.0}, {'a': 1, 'b': 1}, 'two-sided'), 2 / 4)
        # A contrast that is exactly zero on every day: every assignment ties, p = 1 both ways.
        zero = {d: 0.0 for d in 'abcd'}
        self.assertEqual(sign_flip_p(zero, dict.fromkeys('abcd', 3), 'less'), 1.0)
        self.assertEqual(sign_flip_p(zero, dict.fromkeys('abcd', 3), 'two-sided'), 1.0)

    def test_ties_survive_floating_point_summation_order(self):
        # Sums 0.1, 0.2, -0.3, 0.3. The observed (+,+,+,+) and (+,+,-,-) both equal 0.1 + 0.2 exactly, but
        # left-to-right float addition gives 0.30000000000000004 and 0.3000000000000001. fsum ties them.
        # Exact values of the doubles: c,d contribute 0 (two ways), -0.6 or +0.6; a,b contribute 0.3, -0.1, 0.1 or
        # -0.3. At or above the observed 0.1+0.2: the two ties, then 0.6+0.3, 0.6-0.1 and 0.6+0.1 (0.6-0.3 is
        # 0.2999999999999999611 in exact double arithmetic, below the observed 0.3000000000000000166).
        sums, n = {'a': 0.1, 'b': 0.2, 'c': -0.3, 'd': 0.3}, dict.fromkeys('abcd', 1)
        self.assertNotEqual(((0.1 + 0.2) - 0.3) + 0.3, ((0.1 + 0.2) + 0.3) - 0.3)
        self.assertEqual(sign_flip_p(sums, n, 'greater'), 5 / 16)
        self.assertEqual(sign_flip_p(sums, n, 'less'), 13 / 16)  # 16 - 5 + the 2 ties

    def test_matches_exact_rational_enumeration(self):
        # Independent check: enumerate with Fractions (the exact values of the doubles), with zero days and
        # repeated magnitudes so that exact ties occur.
        import itertools
        from fractions import Fraction
        cases = [{'a': 0.1, 'b': -0.1, 'c': 0.0, 'd': 0.7, 'e': -0.3, 'f': 0.3},
                 {'a': -1e-17, 'b': 1e-17, 'c': 0.25, 'd': -0.25, 'e': 0.0},
                 {'a': 0.123, 'b': -0.456, 'c': 0.789, 'd': -0.333, 'e': 0.333, 'f': -0.123, 'g': 0.0}]
        for sums in cases:
            n = {d: 1 + i for i, d in enumerate(sorted(sums))}
            days = sorted(sums)
            exact = [Fraction(sums[d]) for d in days]
            observed = sum(exact)
            values = [sum(s * v for s, v in zip(signs, exact)) for signs in itertools.product((1, -1), repeat=len(days))]
            expected = {'less': sum(v <= observed for v in values), 'greater': sum(v >= observed for v in values),
                        'two-sided': sum(abs(v) >= abs(observed) for v in values)}
            for alternative, count in expected.items():
                self.assertEqual(sign_flip_p(sums, n, alternative), count / len(values), (sums, alternative))

    def test_all_days_negative_gives_the_smallest_p(self):
        days = [f'd{i:02d}' for i in range(14)]
        sums = {d: -1.0 - i for i, d in enumerate(days)}
        n = {d: 10 + i for i, d in enumerate(days)}
        self.assertEqual(sign_flip_p(sums, n, 'less'), 1 / 2 ** 14)
        self.assertEqual(sign_flip_p(sums, n, 'two-sided'), 2 / 2 ** 14)
        self.assertEqual(sign_flip_p(sums, n, 'greater'), 1.0)

    def test_token_counts_only_scale_the_statistic(self):
        sums = {'a': 0.5, 'b': -1.5, 'c': 0.7, 'd': -0.2}
        p = sign_flip_p(sums, {'a': 1, 'b': 1, 'c': 1, 'd': 1}, 'less')
        self.assertEqual(p, sign_flip_p(sums, {'a': 50, 'b': 3, 'c': 9, 'd': 200}, 'less'))
        # Observed -0.5: assignments with sum <= -0.5 among the 16 (enumerated): 7.
        self.assertEqual(p, 7 / 16)

    def test_invalid_input(self):
        for sums, n, alternative in (({}, {}, 'less'), ({'a': 1.0}, {'b': 1}, 'less'), ({'a': 1.0}, {'a': 0}, 'less'),
                                     ({'a': float('nan')}, {'a': 1}, 'less'), ({'a': 1.0}, {'a': 1.5}, 'less'),
                                     ({'a': 1.0}, {'a': 1}, 'smaller'),
                                     ({str(i): 1.0 for i in range(21)}, {str(i): 1 for i in range(21)}, 'less')):
            with self.assertRaises(ValueError):
                sign_flip_p(sums, n, alternative)


class HolmTests(unittest.TestCase):
    def test_step_down_in_input_order(self):
        # Sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 = 0.02, 0.03, 0.06, 0.04 -> running max 0.02, 0.03, 0.06, 0.06.
        adjusted = holm([0.01, 0.04, 0.03, 0.005])
        for got, expected in zip(adjusted, [0.03, 0.06, 0.06, 0.02]):
            self.assertAlmostEqual(got, expected, places=15)

    def test_cap_at_one_and_ties(self):
        self.assertEqual(holm([0.6, 0.5]), [1.0, 1.0])  # 0.5*2 = 1.0, then max(1.0, 0.6)
        tied = holm([0.01, 0.2, 0.01])                   # 0.03, max(0.03, 0.02), max(0.03, 0.2)
        for got, expected in zip(tied, [0.03, 0.2, 0.03]):
            self.assertAlmostEqual(got, expected, places=15)
        self.assertEqual(holm([]), [])
        self.assertEqual(holm([0.04]), [0.04])
        with self.assertRaises(ValueError):
            holm([0.5, 1.2])
