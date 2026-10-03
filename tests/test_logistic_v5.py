"""Logistic v5 (DEC-23): sub-condition selection, validation monotonicity, the one-SE sensitivity,
DEC-23's stop rule and the reproduction of v4's stored fits (gate 1)."""
import json
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from whale_coda_dynamics import logistic_v5
from whale_coda_dynamics.logistic import Encoder, probabilities, select_l2_and_fit, last_label_index
from whale_coda_dynamics.logistic_v5 import (
    BLOCKS, add_input_pairs, condition_blocks, condition_name, dec23_stop_rules, rebuild_predictions,
    reproduce_v4, select_all, select_max, select_one_se, stored_fit_probabilities, subconditions,
)
from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ['none', 'S', 'A', 'B', 'A_B', 'A_S', 'B_S', 'A_B_S']


def candidates_for(condition, per_target):
    return {condition_name(s): (s, per_target[condition_name(s)]) for s in subconditions(condition_blocks(condition))}


class SubconditionTests(unittest.TestCase):
    def test_every_subset_fewest_blocks_first(self):
        self.assertEqual(subconditions(BLOCKS), [(), ('A',), ('B',), ('S',), ('A', 'B'), ('A', 'S'), ('B', 'S'), ('A', 'B', 'S')])
        self.assertEqual(subconditions(('S', 'A')), [(), ('A',), ('S',), ('A', 'S')])
        self.assertEqual([condition_name(s) for s in subconditions(BLOCKS)], ['none', 'A', 'B', 'S', 'A_B', 'A_S', 'B_S', 'A_B_S'])
        for c in CONDITIONS:
            self.assertEqual(condition_name(condition_blocks(c)), c)
        self.assertEqual(len(add_input_pairs(CONDITIONS)), 12)
        # Generic block names (DEC-24 reuses the selection with its own blocks).
        self.assertEqual(subconditions(('tempo', 'rhythm'), order=('tempo', 'rhythm')), [(), ('tempo',), ('rhythm',), ('tempo', 'rhythm')])


class MaxRuleTests(unittest.TestCase):
    def test_highest_mean_wins_and_exact_ties_go_to_fewer_blocks(self):
        none, a, s = np.array([-2.0, -2.0]), np.array([-1.0, -1.5]), np.array([-1.25, -1.25])
        # A and S tie exactly at -1.25 with one block each: the block order A, B, S decides.
        self.assertEqual(select_max({'none': ((), none), 'A': (('A',), a), 'S': (('S',), s)})['chosen'], 'A')
        tie = {'none': ((), none), 'A': (('A',), a), 'S': (('S',), s), 'A_S': (('A', 'S'), np.array([-1.0, -1.5]))}
        self.assertEqual(select_max(tie)['chosen'], 'A')  # A_S ties A exactly: fewer blocks
        better = dict(tie, A_S=(('A', 'S'), np.array([-1.0, -1.4])))
        self.assertEqual(select_max(better)['chosen'], 'A_S')
        with self.assertRaises(ValueError):
            select_max({'none': ((), none), 'A': (('A',), np.array([-1.0]))})

    def test_validation_contrasts_are_monotone_by_construction(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            per_target = {c: rng.normal(-1.5, 0.5, size=7) for c in CONDITIONS}
            chosen = {c: select_max(candidates_for(c, per_target)) for c in CONDITIONS}
            for big, small in add_input_pairs(CONDITIONS):
                gain = (chosen[big]['validation_log_likelihood_bits'][chosen[big]['chosen']]
                        - chosen[small]['validation_log_likelihood_bits'][chosen[small]['chosen']])
                self.assertGreaterEqual(gain, 0.0)

    def test_gate_two_and_selection_rows(self):
        rng = np.random.default_rng(5)
        prepared = [dict(fold=dict(fold_id=f'f{i}'), validation=[None] * 6) for i in range(3)]
        per_target = {(p['fold']['fold_id'], c): rng.normal(-1.5, 0.5, size=6) for p in prepared for c in CONDITIONS}
        selection, chosen, monotonicity = select_all(prepared, CONDITIONS, per_target)
        self.assertEqual(len(selection), 3 * len(CONDITIONS) * 2)
        self.assertEqual(monotonicity['pairs_checked'], 3 * 12)
        self.assertGreaterEqual(monotonicity['min_add_input_validation_contrast_bits'], 0.0)
        for row in selection:
            means = {c['subcondition']: c['validation_log_likelihood_bits'] for c in row['candidates']}
            self.assertEqual(set(means), {condition_name(s) for s in subconditions(condition_blocks(row['condition']))})
            if row['rule'] == 'max':
                self.assertEqual(means[row['chosen']], max(means.values()))
        # A selection that is not the maximum breaks monotonicity; gate 2 must catch it.
        original = logistic_v5.select_max
        def broken(candidates, order=BLOCKS):
            result = original(candidates, order)
            return dict(result, chosen='none') if 'A_B_S' in candidates else result
        with mock.patch.object(logistic_v5, 'select_max', broken):
            with self.assertRaisesRegex(ValueError, 'Gate 2'):
                select_all(prepared, CONDITIONS, per_target)


class OneSERuleTests(unittest.TestCase):
    def test_simplest_candidate_within_one_standard_error(self):
        none = np.full(4, -3.0)
        a = np.array([-1.0, -1.2, -0.8, -1.0])
        # A_S beats A by 0.05 on average, d = (0.1, 0, 0.1, 0): sd 0.0577, SE 0.0289 < 0.05, so A is not within.
        clear = {'none': ((), none), 'A': (('A',), a), 'S': (('S',), none + 0.5), 'A_S': (('A', 'S'), np.array([-0.9, -1.2, -0.7, -1.0]))}
        result = select_one_se(clear)
        self.assertEqual((result['best'], result['chosen']), ('A_S', 'A_S'))
        self.assertFalse(result['one_se']['A']['within_one_se'])
        # d = (0.1, -0.1, 0.1, 0): mean 0.025, sd 0.0957, SE 0.0479 >= 0.025, so A is within and has fewer blocks.
        noisy = dict(clear, A_S=(('A', 'S'), np.array([-0.9, -1.3, -0.7, -1.0])))
        result = select_one_se(noisy)
        self.assertEqual((result['best'], result['chosen']), ('A_S', 'A'))
        self.assertAlmostEqual(result['one_se']['A']['mean_difference_to_best_bits'], 0.025, places=12)
        self.assertAlmostEqual(result['one_se']['A']['standard_error_bits'], math.sqrt(0.0275 / 3) / 2, places=12)
        self.assertTrue(result['one_se']['A_S']['within_one_se'])  # the best is always within
        self.assertFalse(result['one_se']['none']['within_one_se'])
        with self.assertRaises(ValueError):
            select_one_se({'none': ((), np.array([-1.0])), 'A': (('A',), np.array([-0.5]))})


class StopRuleTests(unittest.TestCase):
    def test_sign_flip_with_holm_across_the_family(self):
        dates = {f'c{i}': f'2020-01-0{i // 2 + 1}' for i in range(6)}  # three days, two codas each
        checks = {'negative': {c: -1.0 for c in dates}, 'positive': {c: 1.0 for c in dates},
                  'zero': {c: 0.0 for c in dates}}
        # One-sided p: all three days negative -> 1/8; all positive -> 8/8; all zero -> every assignment ties -> 1.
        # Holm: 0.125 * 3 = 0.375, then 1, 1.
        result = dec23_stop_rules(checks, dates, alpha=0.4)
        rows = {r['check']: r for r in result['checks']}
        self.assertEqual([rows[k]['sign_flip_p'] for k in ('negative', 'positive', 'zero')], [0.125, 1.0, 1.0])
        self.assertEqual([rows[k]['holm_p'] for k in ('negative', 'positive', 'zero')], [0.375, 1.0, 1.0])
        self.assertEqual(result['fired'], ['negative'])
        self.assertEqual(rows['negative']['mean_bits_per_coda'], -1.0)
        self.assertEqual(dec23_stop_rules(checks, dates, alpha=0.05)['fired'], [])


class RebuildTests(unittest.TestCase):
    def test_each_condition_takes_its_chosen_subconditions_predictions(self):
        v4 = [dict(coda_id=f'{f}-{k}', fold_id=f, condition=c, seed=42, probabilities={'x': 1.0, 'tag': f'{f}/{c}'})
              for f in ('f1', 'f2') for c in ('none', 'A', 'S', 'A_S') for k in range(2)]
        selection = [dict(fold_id='f1', condition='A_S', rule='max', chosen='A'),
                     dict(fold_id='f1', condition='A_S', rule='one_se', chosen='none'),
                     dict(fold_id='f2', condition='A_S', rule='max', chosen='A_S')]
        rebuilt = rebuild_predictions(selection, v4)
        self.assertEqual([(r['coda_id'], r['condition'], r['probabilities']['tag']) for r in rebuilt],
                         [('f1-0', 'A_S', 'f1/A'), ('f1-1', 'A_S', 'f1/A'), ('f2-0', 'A_S', 'f2/A_S'), ('f2-1', 'A_S', 'f2/A_S')])
        self.assertEqual([r['probabilities']['tag'] for r in rebuild_predictions(selection, v4, 'one_se')], ['f1/none'] * 2)
        with self.assertRaises(ValueError):
            rebuild_predictions([dict(fold_id='f3', condition='A', rule='max', chosen='A')], v4)


def synthetic_rows(n, seed, split, vocab):
    rng = np.random.RandomState(seed)
    return [dict(coda_id=f'{split}-{seed}-{i}', Ct=vocab[(i + 1) % 2], A_history=[vocab[i % 2], vocab[rng.randint(2)]],
                 B_history=[vocab[rng.randint(2)]], turn_category=['switch', 'overlap'][i % 2], gap_s=float(rng.rand()),
                 overlap_elapsed_s=float(rng.rand()) * (i % 2), split=split) for i in range(n)]


class ReproductionTests(unittest.TestCase):
    """Gate 1 on a small fit stored the way logistic v4 stores it."""

    def setUp(self):
        self.vocab = ['a', 'b', 'UNK']
        train = synthetic_rows(30, 1, 'train', self.vocab)
        self.validation = synthetic_rows(8, 2, 'validation', self.vocab)
        self.test = synthetic_rows(6, 3, 'test', self.vocab)
        fields = ['A_history', 'turn_category', 'gap_s', 'overlap_elapsed_s']
        config = dict(history_order=3, l2_grid=[0.01, 0.1], max_iterations=2000, gradient_tolerance=1e-6)
        encoder, chosen, grid = select_l2_and_fit(train, self.validation, self.vocab, fields, config)
        own_test = last_label_index(self.test, 'A_history', self.vocab)
        self.test_probabilities = probabilities(encoder.transform(self.test), chosen['w'], chosen['beta'], None, own_test, None)
        # JSON round trip, exactly as models.jsonl and l2_selection.jsonl are written and read.
        self.record = json.loads(json.dumps(dict(
            fold_id='f', condition='A_S', vocabulary=self.vocab, fields=fields, order=encoder.order,
            categories=encoder.categories, feature_names=encoder.names + ['intercept'], mean=encoder.mean.tolist(),
            scale=encoder.scale.tolist(), weights=chosen['w'].tolist(), beta=chosen['beta'], gamma=chosen['gamma'],
            chosen_l2=chosen['l2'], chosen_l2_s=chosen['l2_s'])))
        self.selection = json.loads(json.dumps(dict(
            fold_id='f', condition='A_S', chosen_l2=chosen['l2'], chosen_l2_s=chosen['l2_s'],
            grid=[dict(l2=g['l2'], l2_s=g['l2_s'], validation_log_likelihood_bits=g['validation_log_likelihood_bits']) for g in grid])))
        self.prior = {'a': 0.45, 'b': 0.45, 'UNK': 0.1}
        self.stored = {}
        for row, probs in zip(self.test, self.test_probabilities):
            self.stored[row['coda_id'], 'A_S'] = dict(zip(self.vocab, map(float, probs)))
            self.stored[row['coda_id'], 'none'] = dict(self.prior)
        self.prepared = [dict(fold=dict(fold_id='f'), policy=dict(vocabulary=self.vocab, label_prior=self.prior),
                              validation=self.validation, test=self.test)]

    def gate(self, selection=None, stored=None):
        return reproduce_v4(self.prepared, ['none', 'A_S'], {('f', 'A_S'): self.record},
                            {('f', 'A_S'): selection or self.selection}, stored or self.stored, 1e-9)

    def test_encoder_from_record_reproduces_the_fit(self):
        encoder = Encoder.from_record(self.record)
        self.assertEqual(encoder.names + ['intercept'], self.record['feature_names'])
        np.testing.assert_array_equal(stored_fit_probabilities(self.record, self.test, self.vocab), self.test_probabilities)
        with self.assertRaises(ValueError):
            Encoder.from_record(dict(self.record, categories=['switch']))

    def test_gate_passes_and_returns_per_target_validation_bits(self):
        per_target, rows = self.gate()
        self.assertEqual(set(per_target), {('f', 'none'), ('f', 'A_S')})
        y = [self.vocab.index(r['Ct']) for r in self.validation]
        np.testing.assert_allclose(per_target['f', 'none'], [math.log2(self.prior[self.vocab[i]]) for i in y], rtol=0, atol=1e-15)
        fitted = [r for r in rows if r['condition'] == 'A_S'][0]
        self.assertLessEqual(abs(fitted['validation_difference_bits']), 1e-12)
        self.assertEqual(fitted['test_probability_max_abs_difference'], 0.0)

    def test_gate_fails_beyond_tolerance(self):
        chosen = [g for g in self.selection['grid'] if g['l2'] == self.selection['chosen_l2']][0]
        shifted = json.loads(json.dumps(self.selection))
        for g in shifted['grid']:
            if g['l2'] == chosen['l2']:
                g['validation_log_likelihood_bits'] += 2e-9
        with self.assertRaisesRegex(ValueError, 'Gate 1'):
            self.gate(selection=shifted)
        stored = dict(self.stored)
        key = (self.test[0]['coda_id'], 'A_S')
        stored[key] = dict(stored[key], a=stored[key]['a'] + 2e-9)
        with self.assertRaisesRegex(ValueError, 'Gate 1'):
            self.gate(stored=stored)
        stored = dict(self.stored)
        key = (self.test[0]['coda_id'], 'none')
        stored[key] = dict(stored[key], a=stored[key]['a'] + 2e-9)
        with self.assertRaisesRegex(ValueError, 'Gate 1'):
            self.gate(stored=stored)
        with self.assertRaisesRegex(ValueError, 'different penalties'):
            self.gate(selection=dict(self.selection, chosen_l2=0.1 if self.selection['chosen_l2'] == 0.01 else 0.01))


V4 = ROOT / 'data' / 'results' / 'rq1_logistic_v4'


class CommittedV4Tests(unittest.TestCase):
    def test_one_committed_rq1_fold_reproduces(self):
        """Gate 1 on the committed RQ1 v4 run, for the fold with the fewest validation targets."""
        manifest = read_json(ROOT / 'data' / 'processed' / 'rq1_folds' / 'manifest.json')
        fold = min(manifest['folds'], key=lambda f: len(f['comparison_ids']['validation']))
        rows = read_jsonl(ROOT / manifest['configuration']['input'])
        prepared = logistic_v5.prepare_folds(rows, dict(manifest, folds=[fold]))
        fold_id = fold['fold_id']
        models = {(r['fold_id'], r['condition']): r for r in read_jsonl(V4 / 'models.jsonl') if r['fold_id'] == fold_id}
        selections = {(r['fold_id'], r['condition']): r for r in read_jsonl(V4 / 'l2_selection.jsonl') if r['fold_id'] == fold_id}
        stored = {(r['coda_id'], r['condition']): r['probabilities'] for r in read_jsonl(V4 / 'predictions.jsonl')
                  if r['fold_id'] == fold_id}
        per_target, rows_ = reproduce_v4(prepared, CONDITIONS, models, selections, stored, 1e-9)
        self.assertEqual(len(rows_), len(CONDITIONS))
        self.assertEqual({len(v) for v in per_target.values()}, {len(fold['comparison_ids']['validation'])})


if __name__ == '__main__':
    unittest.main()
