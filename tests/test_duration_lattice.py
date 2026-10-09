"""DEC-30's duration lattice (PLAN [P6]): the closed-form ridge against a numerical optimiser, training-only
standardisation, recovery of a planted signal, the Gaussian score, invariance of contrasts to the unit of y, the
penalty search, an independent recomputation of the A and B inputs from the raw rows on every fold, and the data
checks against the committed expected values. The onset property of the inputs is tested in
test_onset_information.py."""
import math
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from whale_coda_dynamics import duration_lattice as dl
from whale_coda_dynamics.evaluation import gaussian_log_density_bits
from whale_coda_dynamics.preprocessing.io_utils import read_csv_rows, read_json, read_jsonl
from whale_coda_dynamics.residual import prepare_folds

ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / 'config/rq1_duration_lattice.json')
MANIFEST = read_json(ROOT / CONFIG['manifest'])
ROWS = read_jsonl(ROOT / MANIFEST['configuration']['input'])
RAW = read_csv_rows(ROOT / CONFIG['raw_source'], encoding='utf-8-sig')[1]
S_FIELDS = MANIFEST['input_conditions']['S']
GRID = read_json(ROOT / CONFIG['logistic_config'])['l2_grid']
_PREPARED = {}


def prepared():
    if not _PREPARED:
        _PREPARED['lookup'] = dl.coda_lookup(ROWS, RAW)
        _PREPARED['folds'] = prepare_folds(ROWS, MANIFEST)
    return _PREPARED['lookup'], _PREPARED['folds']


def objective(w, x, z, penalties):
    residual = z - x @ w
    return float(np.mean(residual ** 2) + penalties @ w ** 2), -2 * x.T @ residual / len(z) + 2 * penalties * w


class RidgeTests(unittest.TestCase):
    def test_a_closed_form_agrees_with_a_numerical_optimiser(self):
        rng = np.random.RandomState(0)
        x = np.column_stack([rng.normal(size=(300, 6)), rng.binomial(1, 0.3, size=(300, 2)), np.ones(300)])
        z = x[:, :3] @ [0.5, -0.2, 0.1] + rng.normal(scale=0.7, size=300)
        for penalties in ([1e-4, 1e-4, 0.01, 0.01, 0.3, 0.3, 1.0, 1.0, 0.0], [0.0003, 1.0, 0.1, 0.001, 0.03, 0.3, 1e-4, 0.01, 0.0]):
            penalties = np.array(penalties)
            closed = dl.ridge_weights(x, z, penalties)
            numerical = minimize(objective, np.zeros(x.shape[1]), args=(x, z, penalties), jac=True, method='L-BFGS-B',
                                 options={'gtol': 1e-12, 'ftol': 1e-15, 'maxiter': 10000})
            self.assertTrue(numerical.success)
            np.testing.assert_allclose(closed, numerical.x, atol=1e-6)
            self.assertLessEqual(objective(closed, x, z, penalties)[0], numerical.fun + 1e-12)

    def test_c_a_planted_linear_signal_is_recovered(self):
        rng = np.random.RandomState(1)
        x = np.column_stack([rng.normal(size=(5000, 4)), np.ones(5000)])
        beta = np.array([0.8, -0.5, 0.0, 0.3])
        y = 2.0 + x[:, :4] @ beta + rng.normal(scale=0.4, size=5000)
        fit = dl.ridge_fit(x, y, np.array([1e-4] * 4 + [0.0]))
        np.testing.assert_allclose(fit['y_scale'] * fit['weights'][:4], beta, atol=0.02)
        self.assertAlmostEqual(math.sqrt(fit['variance']), 0.4, delta=0.01)

    def test_c_a_planted_partner_effect_is_recovered_on_a_real_fold(self):
        """A large planted partner effect (f = 1) gives A,B,S a clear held-out gain over A,S and moves the fitted partner
        lag-1 slope by about the planted f * sigma / scale."""
        lookup, folds = prepared()
        p = folds[0]
        y = {c: v['log_duration'] for c, v in lookup.items()}
        real = {c: dl.fit_condition(p, lookup, y, c, False, CONFIG, S_FIELDS, GRID) for c in ('A_S', 'A_B_S')}
        sigma = math.sqrt(real['A_S']['chosen']['variance'])
        planted = dl.planted_targets(p, lookup, y, 1.0, sigma, CONFIG, S_FIELDS)
        fits = {c: dl.fit_condition(p, lookup, planted, c, False, CONFIG, S_FIELDS, GRID) for c in ('A_S', 'A_B_S')}
        gain = float(np.mean(fits['A_B_S']['test_bits'] - fits['A_S']['test_bits']))
        self.assertGreater(gain, 0.2)
        encoder = fits['A_B_S']['encoder']
        j = encoder.names.index('B:lag1:log_duration')
        slope = lambda f: f['chosen']['y_scale'] * f['chosen']['weights'][j]  # noqa: E731  (per unit of z_b1)
        self.assertAlmostEqual(slope(fits['A_B_S']) - slope(real['A_B_S']), sigma, delta=0.15 * sigma)

    def test_b_standardisation_and_fit_use_training_rows_only(self):
        lookup, folds = prepared()
        p = folds[3]
        train_ids = {r['coda_id'] for r in p['train']}
        y = {c: v['log_duration'] for c, v in lookup.items()}
        shifted = {c: (v if c in train_ids else v + 5.0) for c, v in y.items()}
        a, b = (dl.fit_condition(p, lookup, targets, 'A_B_S', True, CONFIG, S_FIELDS, GRID) for targets in (y, shifted))
        for ga, gb in zip(a['grid'], b['grid']):
            np.testing.assert_array_equal(ga['weights'], gb['weights'])
            self.assertEqual((ga['y_mean'], ga['y_scale'], ga['variance']), (gb['y_mean'], gb['y_scale'], gb['variance']))
        # Codas referenced only by validation or test rows (never by a training row, nor training targets themselves):
        # changing their durations and click counts leaves the encoder and the training design unchanged.
        used = train_ids | {i for r in p['train'] for _, field in dl.STREAMS for i in r[field]}
        others = {i for s in ('validation', 'test') for r in p[s] for _, field in dl.STREAMS for i in r[field]} - used
        self.assertTrue(others)
        changed = {c: (dict(v, log_duration=v['log_duration'] + 1.0, clicks=v['clicks'] + 7, log_clicks=math.log(v['clicks'] + 7))
                       if c in others else v) for c, v in lookup.items()}
        blocks, vocabulary = ('A', 'B', 'S'), p['policy']['vocabulary']
        e1 = dl.DurationEncoder(p['train'], lookup, blocks, S_FIELDS, CONFIG, vocabulary)
        e2 = dl.DurationEncoder(p['train'], changed, blocks, S_FIELDS, CONFIG, vocabulary)
        self.assertEqual(e1.record(), e2.record())
        np.testing.assert_array_equal(e1.transform(p['train'], lookup), e2.transform(p['train'], changed))
        # The history standardisation is the mean and population SD of the non-missing training values.
        for c in e1.history_columns:
            if c['kind'] != 'missing':
                values = [lookup[r[c['field']][-c['lag']]][c['kind']] for r in p['train'] if len(r[c['field']]) >= c['lag']]
                self.assertAlmostEqual(c['mean'], float(np.mean(values)), places=12)
                self.assertAlmostEqual(c['scale'], float(np.std(values)), places=12)


class ScoreTests(unittest.TestCase):
    def test_d_gaussian_score_against_a_direct_computation(self):
        for value, mean, variance in ((0.3, -0.1, 0.04), (-2.0, 1.5, 2.5), (0.0, 0.0, 1.0), (1e-3, 1e-3, 1e-6)):
            direct = norm.logpdf(value, loc=mean, scale=math.sqrt(variance)) / math.log(2)
            self.assertAlmostEqual(gaussian_log_density_bits(value, mean, variance), direct, places=10)
            self.assertAlmostEqual(gaussian_log_density_bits(value, mean, variance),
                                   math.log2(math.exp(-(value - mean) ** 2 / (2 * variance)) / math.sqrt(2 * math.pi * variance)), places=8)
        for bad in ((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (math.nan, 0.0, 1.0), (0.0, math.inf, 1.0), (True, 0.0, 1.0)):
            with self.assertRaises(ValueError):
                gaussian_log_density_bits(*bad)

    def test_e_rescaling_y_shifts_every_score_by_one_constant(self):
        """y -> c y: standardised targets, chosen penalties and every residual in sigma units are unchanged, so every
        score shifts by -log2 |c| and contrasts are unchanged; y -> y + k (another unit of d) leaves every score as is."""
        lookup, folds = prepared()
        p = folds[5]
        y = {c: v['log_duration'] for c, v in lookup.items()}
        base = {c: dl.fit_condition(p, lookup, y, c, False, CONFIG, S_FIELDS, GRID) for c in ('A_S', 'A_B_S')}
        for transform, shift in ((lambda v: 1000.0 * v, -math.log2(1000.0)), (lambda v: -0.25 * v, -math.log2(0.25)),
                                 (lambda v: v + math.log(1000.0), 0.0)):
            moved = {c: transform(v) for c, v in y.items()}
            fits = {c: dl.fit_condition(p, lookup, moved, c, False, CONFIG, S_FIELDS, GRID) for c in ('A_S', 'A_B_S')}
            for c in fits:
                self.assertEqual((fits[c]['chosen']['l2'], fits[c]['chosen']['l2_s']), (base[c]['chosen']['l2'], base[c]['chosen']['l2_s']))
                np.testing.assert_allclose(fits[c]['test_bits'], base[c]['test_bits'] + shift, atol=1e-9)
            np.testing.assert_allclose(fits['A_B_S']['test_bits'] - fits['A_S']['test_bits'],
                                       base['A_B_S']['test_bits'] - base['A_S']['test_bits'], atol=1e-9)


class SearchAndEncoderTests(unittest.TestCase):
    def test_search_axes_follow_the_blocks_and_ties_go_to_larger_values(self):
        lookup, folds = prepared()
        p = folds[1]
        y = {c: v['log_duration'] for c, v in lookup.items()}
        sizes = {c: len(dl.fit_condition(p, lookup, y, c, False, CONFIG, S_FIELDS, GRID)['grid']) for c in MANIFEST['input_conditions']}
        self.assertEqual(sizes, dict(none=1, S=9, A=9, B=9, A_B=9, A_S=81, B_S=81, A_B_S=81))

        class Flat:
            has_history, has_s = True, True

            @staticmethod
            def penalties(l2, l2_s, fixed):
                return np.array([1.0, 1.0])
        x = np.column_stack([np.linspace(-1, 1, 20), np.ones(20)])
        chosen, grid = dl.penalty_search(Flat, x, np.linspace(0, 3, 20), x, np.linspace(0, 3, 20), [0.1, 1.0, 0.5], 1e-4)
        self.assertEqual(len({g['validation_log_density_bits'] for g in grid}), 1)
        self.assertEqual((chosen['l2'], chosen['l2_s']), (1.0, 1.0))

    def test_columns_per_condition_and_variant(self):
        lookup, folds = prepared()
        p = folds[2]
        floor = dl.DurationEncoder(p['train'], lookup, (), S_FIELDS, CONFIG)
        self.assertEqual(set(floor.groups), {'always', 'intercept'})
        self.assertEqual(floor.names[0], 'log_clicks')
        self.assertEqual(floor.count_levels, sorted(c for c in {lookup[r['coda_id']]['clicks'] for r in p['train']}
                                                    if sum(lookup[r['coda_id']]['clicks'] == c for r in p['train']) >= 10))
        typed = dl.DurationEncoder(p['train'], lookup, (), S_FIELDS, CONFIG, p['policy']['vocabulary'])
        self.assertEqual(typed.names[len(floor.names) - 1:-1], ['type=' + t for t in p['policy']['vocabulary']])
        full = dl.DurationEncoder(p['train'], lookup, ('A', 'B', 'S'), S_FIELDS, CONFIG)
        # Every comparison target has lag 1 of both streams, so those missing indicators are constant and dropped.
        self.assertIn('A:lag1:missing', full.dropped)
        self.assertIn('B:lag1:missing', full.dropped)
        self.assertEqual([n for n in full.names if n.startswith('A:')],
                         ['A:lag1:log_duration', 'A:lag1:log_clicks', 'A:lag2:log_duration', 'A:lag2:log_clicks',
                          'A:lag2:missing', 'A:lag3:log_duration', 'A:lag3:log_clicks', 'A:lag3:missing'])
        self.assertEqual([n for n in full.names if n.startswith('S:')],
                         ['S:turn:overlap', 'S:turn:self_continuation', 'S:turn:switch', 'S:gap_s', 'S:log1p:gap_s',
                          'S:overlap_elapsed_s', 'S:log1p:overlap_elapsed_s'])
        penalties = full.penalties(0.3, 0.01, 1e-4)
        self.assertEqual(dict(zip(full.groups, penalties)), {'always': 1e-4, 'A': 0.3, 'B': 0.3, 'S': 0.01, 'intercept': 0.0})
        with self.assertRaises(ValueError):
            full.penalties(None, 0.01, 1e-4)
        with self.assertRaises(ValueError):
            floor.penalties(0.3, None, 1e-4)


class RawRecomputationTests(unittest.TestCase):
    def test_g_history_inputs_match_an_independent_recomputation_on_every_fold(self):
        """Histories rebuilt from the raw rows (TsTo, Duration, Whale, nClicks) of the retained codas, grouped by the
        exchange assignment: each whale's and its partner's codas completed by the target's onset, last 3 by onset.
        Their ln Duration and ln nClicks, standardised on each fold's training targets, must equal the design."""
        assignments = [a for a in read_jsonl(ROOT / 'data/processed/exchange_assignments.jsonl') if a['retained']]
        exchange = defaultdict(list)
        for a in assignments:
            r = RAW[a['source_row_index']]
            onset, duration = float(r['TsTo']), float(r['Duration'])
            exchange[a['exchange_id']].append(dict(coda_id=a['coda_id'], whale=r['Whale'], onset=onset, end=onset + duration,
                                                   values=(math.log(duration), math.log(int(float(r['nClicks']))))))
        independent = {}
        for group in exchange.values():
            for c in group:
                done = sorted((o for o in group if o['onset'] < c['onset'] and o['end'] <= c['onset'] + 1e-9), key=lambda o: o['onset'])
                independent[c['coda_id']] = {'A': [o['values'] for o in done if o['whale'] == c['whale']][-3:][::-1],
                                             'B': [o['values'] for o in done if o['whale'] != c['whale']][-3:][::-1]}
        lookup, folds = prepared()
        targets = [r for r in ROWS if r['comparison_eligible']]
        for r in targets:
            got = dl.history_values(r, lookup, 3)
            for block in ('A', 'B'):
                mine = [v for v in got[block] if v is not None]
                self.assertEqual(len(mine), len(independent[r['coda_id']][block]), r['coda_id'])
                np.testing.assert_allclose(mine, independent[r['coda_id']][block], atol=1e-9, rtol=0)
        checked = 0
        for p in folds:
            encoder = dl.DurationEncoder(p['train'], lookup, ('A', 'B'), S_FIELDS, CONFIG)
            for c in encoder.history_columns:
                if c['kind'] == 'missing':
                    continue
                k = dl.KINDS.index(c['kind'])
                train = [independent[r['coda_id']][c['block']] for r in p['train']]
                values = [h[c['lag'] - 1][k] for h in train if len(h) >= c['lag']]
                mean, scale = float(np.mean(values)), float(np.std(values))
                j = encoder.names.index(c['name'])
                for split in dl.SPLITS:
                    x = encoder.transform(p[split], lookup)[:, j]
                    want = [((h[c['lag'] - 1][k] - mean) / scale) if len(h) >= c['lag'] else 0.0
                            for h in (independent[r['coda_id']][c['block']] for r in p[split])]
                    np.testing.assert_allclose(x, want, atol=1e-8, rtol=0)
                    checked += len(want)
        self.assertGreater(checked, 14 * 1547)


class DataCheckTests(unittest.TestCase):
    def test_the_committed_targets_reproduce_the_expected_values(self):
        quantities = dl.data_checks(ROWS, RAW, CONFIG['history_lags'])
        checks = dl.compare_checks(quantities, CONFIG['checks']['expected'], CONFIG['checks']['tolerance'])
        self.assertEqual([c['name'] for c in checks if not c['passed']], [])
        self.assertEqual(len(quantities['nclicks_vs_intervals']['rows']), 2)

    def test_compare_rules(self):
        q = dict(a=dict(count=3, value=0.5, max_abs=1e-12, counts={'1': 2}))
        ok = dl.compare_checks(q, dict(a=dict(count=3, value=0.5004, max_abs_below=1e-9, counts={'1': 2})), 5e-4)
        self.assertTrue(ok[0]['passed'])
        bad = dl.compare_checks(q, dict(a=dict(count=4, value=0.5006, max_abs_below=1e-13, counts={'1': 3}), b=dict(count=1)), 5e-4)
        self.assertEqual(bad[0]['failed_fields'], ['count', 'value', 'max_abs_below', 'counts'])
        self.assertFalse(bad[1]['passed'])


class InputListingTests(unittest.TestCase):
    def test_stage_inputs_must_be_exactly_the_listed_files(self):
        loaded = dl.load(ROOT / 'config/rq1_duration_lattice.json')
        read = dl.external_inputs(loaded)
        self.assertEqual(list(dl.stage_inputs(loaded, 'checks', read)), CONFIG['input_files']['checks'])
        with self.assertRaises(ValueError):
            dl.stage_inputs(loaded, 'checks', read[:-1])
        with self.assertRaises(ValueError):
            dl.stage_inputs(loaded, 'fit', read)


if __name__ == '__main__':
    unittest.main()
