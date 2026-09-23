from copy import deepcopy
import math
import unittest

from whale_coda_dynamics.evaluation import (
    evaluate, log_likelihood_bits, repeat_switch_bits, day_bootstrap_bits, classification_metrics,
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
