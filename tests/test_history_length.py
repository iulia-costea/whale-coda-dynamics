import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from whale_coda_dynamics.evaluation import day_bootstrap_bits
from whale_coda_dynamics.history_length import order_contrast, reading_lines, reference_gate
from whale_coda_dynamics.logistic import Encoder
from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl, write_jsonl, write_text
from whale_coda_dynamics.residual import prepare_folds

ROOT = Path(__file__).resolve().parents[1]


class ReferenceOrderTests(unittest.TestCase):
    def test_reference_order_reproduces_v4_encoding_on_a_real_fold(self):
        """k = 3 through history_orders encodes every condition exactly as logistic v4 did (its stored names)."""
        manifest = read_json(ROOT / 'data/processed/rq1_folds/manifest.json')
        rows = read_jsonl(ROOT / manifest['configuration']['input'])
        prepared = prepare_folds(rows, manifest, test_days=[manifest['folds'][0]['test_day']])[0]
        fold_id, vocab = prepared['fold']['fold_id'], prepared['policy']['vocabulary']
        stored = {r['condition']: r for r in read_jsonl(ROOT / 'data/results/rq1_logistic_v4/models.jsonl') if r['fold_id'] == fold_id}
        for condition, fields in manifest['input_conditions'].items():
            if condition == 'none':
                continue
            default = Encoder(prepared['train'], vocab, fields, 3)
            swept = Encoder(prepared['train'], vocab, fields, 3, {'A_history': 3})
            self.assertEqual(swept.names + ['intercept'], stored[condition]['feature_names'])
            self.assertEqual(swept.names, default.names)
            np.testing.assert_array_equal(swept.mean, default.mean)
            np.testing.assert_array_equal(swept.scale, default.scale)
            for split in ('train', 'validation', 'test'):
                np.testing.assert_array_equal(swept.transform(prepared[split]), default.transform(prepared[split]))


def fake_run(directory, shift, penalties, validation_bits, a_bits=-1.0):
    """Predictions for two folds x conditions none and A, their selection records and metrics."""
    predictions, selections = [], []
    for fold, coda_ids in (('f1', ['c1', 'c2']), ('f2', ['c3'])):
        for condition in ('none', 'A'):
            for coda_id in coda_ids:
                d = shift.get((fold, condition), 0.0)
                predictions.append(dict(coda_id=coda_id, fold_id=fold, condition=condition, seed=42,
                                        probabilities={'a': 0.6 + d, 'b': 0.4 - d}))
            if condition != 'none':
                l2 = penalties[fold]
                selections.append(dict(fold_id=fold, condition=condition, chosen_l2=l2, chosen_l2_s=None,
                                       grid=[dict(l2=v, l2_s=None, validation_log_likelihood_bits=validation_bits[v])
                                             for v in (0.01, 0.1)]))
    metrics = dict(none=dict(mean_log_likelihood_bits=-1.0), A=dict(mean_log_likelihood_bits=a_bits))
    if directory is not None:
        (directory / 'evaluation').mkdir(parents=True)
        write_jsonl(directory / 'predictions.jsonl', predictions)
        write_jsonl(directory / 'l2_selection.jsonl', selections)
        write_text(directory / 'evaluation' / 'summary.json', json.dumps(dict(metrics=metrics)))
    return dict(predictions=predictions, selections=selections, summary=dict(metrics=metrics))


class GateTests(unittest.TestCase):
    """DEC-25 addendum: penalties and mean log-likelihoods are enforced; probabilities are reported."""
    spec = dict(results_directory='v4', maximum_penalty_differences=1, log_likelihood_tolerance_bits=1e-5,
                probability_listing_threshold=1e-5)

    def gate(self, shift, penalties, a_bits=-1.0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_run(root / 'v4', {}, {'f1': 0.01, 'f2': 0.01}, {0.01: -1.0, 0.1: -1.1})
            fitted = fake_run(None, shift, penalties, {0.01: -1.2, 0.1: -1.19}, a_bits)
            return reference_gate(root, self.spec, fitted)[0]

    def test_noise_within_tolerance_passes(self):
        gate = self.gate({('f1', 'A'): 3e-7}, {'f1': 0.01, 'f2': 0.01})
        self.assertTrue(gate['passed'])
        self.assertAlmostEqual(gate['max_abs_probability_difference_same_penalty'], 3e-7)
        self.assertEqual((gate['fits_compared'], gate['penalty_differences'], gate['fits_above_probability_threshold']),
                         (2, [], []))

    def test_probability_differences_are_listed_not_enforced(self):
        gate = self.gate({('f1', 'A'): 1e-4, ('f2', 'A'): 2e-5}, {'f1': 0.01, 'f2': 0.01}, a_bits=-1.0 + 1e-6)
        self.assertTrue(gate['passed'])
        self.assertEqual([(r['fold_id'], r['condition']) for r in gate['fits_above_probability_threshold']],
                         [('f1', 'A'), ('f2', 'A')])  # largest first
        self.assertAlmostEqual(gate['max_abs_log_likelihood_difference_bits'], 1e-6)

    def test_one_differing_penalty_is_listed_with_both_validation_values(self):
        gate = self.gate({('f2', 'A'): 0.1}, {'f1': 0.01, 'f2': 0.1})
        self.assertTrue(gate['passed'])  # its probabilities differ, but only agreeing penalties are compared
        (d,) = gate['penalty_differences']
        self.assertEqual((d['fold_id'], d['chosen_l2'], d['v4_chosen_l2']), ('f2', 0.1, 0.01))
        self.assertEqual((d['validation_bits_own_choice'], d['validation_bits_v4_choice']), (-1.19, -1.2))
        self.assertEqual((d['v4_run_validation_bits_own_choice'], d['v4_run_validation_bits_v4_choice']), (-1.1, -1.0))
        self.assertAlmostEqual(gate['max_abs_probability_difference'], 0.1)

    def test_fails_on_mean_log_likelihood_or_two_differing_penalties(self):
        self.assertFalse(self.gate({}, {'f1': 0.01, 'f2': 0.01}, a_bits=-1.0 + 2e-5)['passed'])
        self.assertFalse(self.gate({}, {'f1': 0.1, 'f2': 0.1})['passed'])


def scored(values_by_day, condition='A'):
    return [dict(coda_id=f'{day}-{i}', condition=condition, recording_date=day, target='a', own_last='a',
                 log_likelihood_bits=ll, repeat_bits=rep, switch_choice_bits=ll - rep)
            for day, values in values_by_day.items() for i, (ll, rep) in enumerate(values)]


class ContrastTests(unittest.TestCase):
    def test_paired_per_coda_and_split_into_parts(self):
        base = scored({'d1': [(-1.0, -0.5), (-2.0, -1.0)], 'd2': [(-1.0, -1.0)], 'd3': [(-3.0, -2.0)]})
        other = scored({'d1': [(-0.5, -0.25), (-2.5, -1.0)], 'd2': [(-0.9, -0.9)], 'd3': [(-2.0, -1.5)]})
        out = order_contrast(other, base, 'A', (200, 42, 0.95))
        differences = {}
        for a, b in zip(other, base):
            differences.setdefault(a['recording_date'], []).append(a['log_likelihood_bits'] - b['log_likelihood_bits'])
        self.assertEqual(out['log_likelihood_bits'], day_bootstrap_bits(differences, 200, 42, 0.95))
        self.assertAlmostEqual(out['log_likelihood_bits']['mean_bits_per_coda'], 1.1 / 4)
        self.assertAlmostEqual(out['repeat_bits']['mean_bits_per_coda'], (0.25 + 0 + 0.1 + 0.5) / 4)
        total = out['repeat_bits']['mean_bits_per_coda'] + out['switch_choice_bits']['mean_bits_per_coda']
        self.assertTrue(math.isclose(total, out['log_likelihood_bits']['mean_bits_per_coda'], abs_tol=1e-12))
        with self.assertRaises(ValueError):
            order_contrast(other[:-1], base, 'A', (200, 42, 0.95))


def result_with(ci, gate=True, stops=()):
    stats = lambda low, high: dict(mean_bits_per_coda=(low + high) / 2, unweighted_mean_bits_per_coda=0.0,  # noqa: E731
                                   ci_bits_per_coda=[low, high], positive_days=7, days=14)
    contrast = lambda low, high: dict(components=dict(log_likelihood_bits=stats(low, high), repeat_bits=stats(low, high),  # noqa: E731
                                                      switch_choice_bits=stats(-0.1, 0.1)))
    return dict(reference_order=3, primary=dict(condition='A_B_S', order=9), conditions=['A', 'A_B_S'],
                contrasts={'A': {'1': contrast(-0.3, -0.1), '9': contrast(-0.01, 0.02)},
                           'A_B_S': {'1': contrast(-0.3, -0.1), '9': contrast(*ci)}},
                stop_rules={'1': [], '3': [], '9': list(stops)}, gate=dict(passed=gate),
                calibration=dict(lag=6, per_fraction={'0.4': dict(bound_bits_per_coda=None)}))


class ReadingTests(unittest.TestCase):
    config = dict(secondary_order=1, calibration=dict(condition='A'))

    def test_reading_follows_the_primary_interval(self):
        self.assertIn('carries usable information beyond partner and timing', reading_lines(result_with((0.01, 0.05)), self.config)[1])
        self.assertIn('It lives in the repeat part', reading_lines(result_with((0.01, 0.05)), self.config)[1])
        null = reading_lines(result_with((-0.01, 0.05)), self.config)[1]
        self.assertTrue(null.startswith('No detectable information beyond 3 own codas for this model at this data size.'))
        self.assertIn('The calibration gives no bound', null)
        self.assertIn('fitting problem', reading_lines(result_with((-0.05, -0.01)), self.config)[1])
        self.assertTrue(reading_lines(result_with((0.01, 0.05), stops=['x']), self.config)[1].startswith('No reading: DEC-16'))
        self.assertTrue(reading_lines(result_with((0.01, 0.05), gate=False), self.config)[1].startswith('No reading: the reference'))
        self.assertTrue(reading_lines(result_with((0.01, 0.05)), self.config)[-1].startswith('Secondary, pre-specified: LL(1) - LL(3) in A'))


if __name__ == '__main__':
    unittest.main()
