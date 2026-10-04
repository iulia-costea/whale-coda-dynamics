"""DEC-28 switch-structure features (PLAN [P5]): definitions, an independent recomputation from the raw rows,
agreement with A_history where the stream fits in it, and v4's fit when a variant has no class features.
The onset property of the features is tested in test_onset_information.py."""
import itertools
import math
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from whale_coda_dynamics.logistic import last_label_index, probabilities, select_l2_and_fit
from whale_coda_dynamics.preprocessing.folds import prepare_fold
from whale_coda_dynamics.preprocessing.io_utils import read_csv_rows, read_json, read_jsonl
from whale_coda_dynamics.switch_structure import (FEATURES, check_descriptive, class_values, feature_matrices,
                                                  own_streams, ratio_interval, reading_lines, stream_structure,
                                                  variant_class_features)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = read_json(ROOT / 'data/processed/rq1_folds/manifest.json')
ROWS = read_jsonl(ROOT / MANIFEST['configuration']['input'])
LIMIT = MANIFEST['preparation_configuration']['history_limit']


def mapped_fold(index):
    fold = MANIFEST['folds'][index]
    mapped, policy = prepare_fold(ROWS, fold, MANIFEST['preparation_configuration'])
    if policy != fold['vocabulary_policy']:
        raise AssertionError('Recomputed fold policy differs from the manifest')
    return fold, mapped, policy


class DefinitionTests(unittest.TestCase):
    def test_structure_and_feature_values(self):
        s = stream_structure(['a', 'b', 'b', 'c', 'a', 'a', 'a'])
        self.assertEqual((s['last'], s['run'], s['d1'], s['d2'], s['before']), ('a', 3, 'c', 'b', Counter(a=1, b=2, c=1)))
        self.assertEqual(class_values(s, 'run_length'), {'a': math.log(3)})
        self.assertEqual(class_values(s, 'distinct_1'), {'c': 1.0})
        self.assertEqual(class_values(s, 'distinct_2'), {'b': 1.0})
        self.assertEqual(class_values(s, 'repertoire'), {'a': math.log1p(1), 'b': math.log1p(2), 'c': math.log1p(1)})
        alternating = stream_structure(['a', 'b', 'a'])  # d2 must differ from both the current type and d1
        self.assertEqual((alternating['run'], alternating['d1'], alternating['d2']), (1, 'b', None))
        single = stream_structure(['a'])
        self.assertEqual((single['run'], single['d1'], single['d2'], single['before']), (1, None, None, Counter()))
        self.assertEqual([class_values(single, f) for f in FEATURES], [{'a': 0.0}, {}, {}, {}])
        self.assertIsNone(stream_structure([]))
        self.assertEqual([class_values(None, f) for f in FEATURES], [{}, {}, {}, {}])
        with self.assertRaises(ValueError):
            class_values(single, 'lag_6')

    def test_matrices_place_values_in_vocabulary_columns(self):
        rows = [dict(coda_id='x'), dict(coda_id='y')]
        streams = {'x': ['b', 'UNK', 'UNK'], 'y': []}
        run, d1, rep = feature_matrices(rows, streams, ['a', 'b', 'UNK'], ['run_length', 'distinct_1', 'repertoire'])
        np.testing.assert_array_equal(run, [[0, 0, math.log(2)], [0, 0, 0]])
        np.testing.assert_array_equal(d1, [[0, 1, 0], [0, 0, 0]])
        np.testing.assert_array_equal(rep, [[0, math.log1p(1), 0], [0, 0, 0]])

    def test_streams_are_per_whale_and_exchange_and_fail_loudly(self):
        rows = [dict(coda_id=i, exchange_id=x, whale_id=w, onset_s=t, offset_s=t + 0.5, Ct=label)
                for i, x, w, t, label in (('1', 'x', 'w1', 0.0, 'a'), ('2', 'x', 'w2', 1.0, 'b'), ('3', 'x', 'w1', 2.0, 'c'),
                                          ('4', 'y', 'w1', 3.0, 'd'), ('5', 'x', 'w1', 4.0, 'e'))]
        self.assertEqual(own_streams(rows), {'1': [], '2': [], '3': ['a'], '4': [], '5': ['a', 'c']})
        self.assertEqual(own_streams(rows, coda_ids=['5']), {'5': ['a', 'c']})
        overlapping = rows[:2] + [dict(rows[2], offset_s=4.5)] + rows[3:]
        self.assertEqual(own_streams(overlapping, coda_ids=['3']), {'3': ['a']})
        with self.assertRaises(ValueError):  # coda 3 is still sounding when coda 5 starts
            own_streams(overlapping)
        with self.assertRaises(ValueError):
            own_streams(rows[:4] + [dict(rows[4], onset_s=2.0)])

    def test_variants_without_features_or_without_a_get_none(self):
        self.assertIsNone(variant_class_features([], LIMIT))
        hook = variant_class_features(['run_length', 'repertoire'], LIMIT)
        for fields in (['turn_category', 'gap_s', 'overlap_elapsed_s'], ['B_history'],
                       ['B_history', 'turn_category', 'gap_s', 'overlap_elapsed_s']):
            self.assertIsNone(hook([], {}, fields))
        with self.assertRaises(ValueError):
            variant_class_features(['run_length', 'run_length'], LIMIT)


def runs_view(stream):
    """An independent collapse of a stream into runs (itertools.groupby): last label, run, d1, d2, counts."""
    runs = [(label, len(list(group))) for label, group in itertools.groupby(stream)]
    (last, run), earlier = runs[-1], runs[:-1]
    d1 = earlier[-1][0] if earlier else None
    d2 = None
    for label, _ in reversed(earlier):
        if label != last and label != d1:
            d2 = label
            break
    counts = defaultdict(int)
    for label, n in earlier:
        counts[label] += n
    return last, run, d1, d2, counts


class RawRecomputationTests(unittest.TestCase):
    """(a) The features from an independent recomputation from the raw CSV rows: onsets (TsTo), callers (Whale)
    and types (guy_coda_type) of the retained codas, grouped by the exchange assignment, mapped with each fold's
    vocabulary and collapsed into runs by a different algorithm."""

    def test_every_fold_matches_an_independent_recomputation(self):
        raw = read_csv_rows(ROOT / read_json(ROOT / 'config/preprocessing.json')['input'], encoding='utf-8-sig')[1]
        assignments = [a for a in read_jsonl(ROOT / 'data/processed/exchange_assignments.jsonl') if a['retained']]
        by_caller = defaultdict(list)
        for a in assignments:
            r = raw[a['source_row_index']]
            by_caller[a['exchange_id'], r['Whale']].append(
                dict(coda_id=a['coda_id'], onset=float(r['TsTo']), label=r['guy_coda_type']))
        self.assertEqual({c['coda_id'] for group in by_caller.values() for c in group}, {r['coda_id'] for r in ROWS})
        raw_stream = {}
        for group in by_caller.values():
            for c in group:
                earlier = sorted((o for o in group if o['onset'] < c['onset']), key=lambda o: o['onset'])
                raw_stream[c['coda_id']] = [o['label'] for o in earlier]
        names = list(FEATURES)
        hook = variant_class_features(names, LIMIT)
        checked = 0
        for index in range(len(MANIFEST['folds'])):
            fold, mapped, policy = mapped_fold(index)
            vocabulary, unk = policy['vocabulary'], policy['unk_token']
            rows = [r for r in mapped if r['comparison_eligible']]
            got = hook(mapped, policy, ['A_history'])[1](rows)
            position = {label: j for j, label in enumerate(vocabulary)}
            for i, r in enumerate(rows):
                stream = [label if label in position and label != unk else unk for label in raw_stream[r['coda_id']]]
                expected = [np.zeros(len(vocabulary)) for _ in names]
                last, run, d1, d2, counts = runs_view(stream)
                expected[0][position[last]] = math.log(run)
                if d1 is not None:
                    expected[1][position[d1]] = 1.0
                if d2 is not None:
                    expected[2][position[d2]] = 1.0
                for label, n in counts.items():
                    expected[3][position[label]] = math.log1p(n)
                for matrix, row in zip(got, expected):
                    np.testing.assert_array_equal(matrix[i], row, err_msg=f"{fold['fold_id']} {r['coda_id']}")
                checked += 1
        self.assertEqual(checked, len(MANIFEST['folds']) * MANIFEST['comparison_targets'])


class HistoryAgreementTests(unittest.TestCase):
    """(b) Where the whole stream fits in A_history's codas, features computed from A_history agree."""

    def test_features_from_a_history_where_the_stream_fits(self):
        names = list(FEATURES)
        fold, mapped, policy = mapped_fold(0)
        rows = [r for r in mapped if r['comparison_eligible']]
        streams = own_streams(mapped)
        from_stream = feature_matrices(rows, streams, policy['vocabulary'], names)
        from_history = feature_matrices(rows, {r['coda_id']: r['A_history'] for r in rows}, policy['vocabulary'], names)
        fits = np.array([len(streams[r['coda_id']]) <= LIMIT for r in rows])
        for stream_matrix, history_matrix in zip(from_stream, from_history):
            np.testing.assert_array_equal(stream_matrix[fits], history_matrix[fits])
        for r in rows:
            self.assertEqual(streams[r['coda_id']][-LIMIT:], r['A_history'])
        # Not vacuous: longer streams exist and some of their features need more than A_history.
        self.assertTrue(fits.any() and not fits.all())
        self.assertTrue(any((a[~fits] != b[~fits]).any() for a, b in zip(from_stream, from_history)))


class NoClassFeatureFitTests(unittest.TestCase):
    """(d) Without class features a variant's fit is v4's: the hook gives v4's code path, an empty feature list
    changes nothing, and the fit reproduces the committed v4 model on a real fold."""

    def test_fit_without_features_is_v4(self):
        fold, mapped, policy = mapped_fold(0)
        vocab = policy['vocabulary']
        split = {s: [r for r in mapped if r['split'] == s and r['comparison_eligible']] for s in ('train', 'validation', 'test')}
        stored = next(m for m in read_jsonl(ROOT / 'data/results/rq1_logistic_v4/models.jsonl')
                      if m['fold_id'] == fold['fold_id'] and m['condition'] == 'A')
        config = dict(read_json(ROOT / 'config/rq1_logistic_v4.json'), l2_grid=[stored['chosen_l2'], 1.0])
        fields = MANIFEST['input_conditions']['A']
        encoder, plain, grid = select_l2_and_fit(split['train'], split['validation'], vocab, fields, config)
        _, empty, grid_empty = select_l2_and_fit(split['train'], split['validation'], vocab, fields, config, lambda rows: [])
        np.testing.assert_array_equal(plain['w'], empty['w'])
        self.assertEqual((plain['l2'], plain['beta']), (empty['l2'], empty['beta']))
        self.assertEqual([g['validation_log_likelihood_bits'] for g in grid], [g['validation_log_likelihood_bits'] for g in grid_empty])
        self.assertEqual(plain['l2'], stored['chosen_l2'])
        own = last_label_index(split['test'], 'A_history', vocab)
        mine = probabilities(encoder.transform(split['test']), plain['w'], plain['beta'], None, own, None)
        theirs = probabilities(encoder.transform(split['test']), np.asarray(stored['weights']), stored['beta'], None, own, None)
        self.assertLess(np.abs(mine - theirs).max(), 1e-3)  # v4 reproduces to ~1e-4 across machines (DEC-25)
        # With features the same fold fits non-zero class-feature weights.
        names, featurize = variant_class_features(['run_length', 'repertoire'], LIMIT)(mapped, policy, fields)
        _, featured, _ = select_l2_and_fit(split['train'], split['validation'], vocab, fields,
                                           dict(config, l2_grid=[stored['chosen_l2']]), featurize)
        self.assertEqual(len(featured['class_weights']), 2)
        self.assertTrue(np.all(featured['class_weights'] != 0))


class DescriptiveHelperTests(unittest.TestCase):
    def test_ratio_interval_is_a_ratio_of_sums_over_drawn_days(self):
        settings = dict(bootstrap_replicates=200, bootstrap_seed=1, confidence=0.9)
        items = [('d1', 1, 1), ('d1', 0, 1), ('d2', 1, 1), ('d3', 0, 1)]
        out = ratio_interval(items, ['d1', 'd2', 'd3', 'd4'], settings)
        self.assertAlmostEqual(out['value'], 0.5)
        self.assertTrue(0 <= out['ci'][0] <= out['ci'][1] <= 1)
        same = ratio_interval([('d1', 2, 4), ('d2', 0, 0), ('d3', 1, 2)], ['d1', 'd2', 'd3'], settings)
        self.assertEqual(same['ci'], [0.5, 0.5])
        self.assertGreater(same['replicates_without_denominator'], 0)

    def test_check_compares_counts_exactly_values_at_three_decimals_and_intervals_within_tolerance(self):
        computed = dict(x=dict(value=0.82149, n=10, ci=[0.70, 0.90]), y=dict(count=3))
        good = check_descriptive(computed, dict(x=dict(value=0.821, n=10, ci=[0.71, 0.89]), y=dict(count=3)), 0.015)
        self.assertTrue(all(c['passed'] for c in good))
        bad = check_descriptive(computed, dict(x=dict(value=0.822, n=11, ci=[0.72, 0.90]), y=dict(count=4),
                                               z=dict(count=1)), 0.015)
        self.assertEqual([c['failed_fields'] for c in bad], [['value', 'n', 'ci'], ['count'], ['count']])


def fake_result(primary_ci, repertoire_ci=(-0.1, 0.1), fired=(), switch_ci=(-0.1, 0.1)):
    def contrast(full, reduced, ci, switch=(-0.1, 0.1)):
        stat = lambda c: dict(mean_bits_per_coda=sum(c) / 2, unweighted_mean_bits_per_coda=0.0, ci_bits_per_coda=list(c),  # noqa: E731
                              positive_days=7, days=14)
        return dict(full=full, reduced=reduced, components=dict(log_likelihood_bits=dict(stat(ci), sign_flip_p_two_sided=0.5),
                                                                 repeat_bits=stat((-0.1, 0.1)), switch_choice_bits=stat(switch)))
    return dict(primary=dict(condition='A_B_S', full='PD', reduced='P'), secondary=dict(condition='A_B_S', full='PD', reduced='R'),
                repertoire_contrast=dict(condition='A_B_S', full='P', reduced='L'), reference_variant='R',
                contrasts={'A_B_S': {'PD-P': contrast('PD', 'P', primary_ci, switch_ci), 'PD-R': contrast('PD', 'R', (-0.1, 0.2)),
                                     'P-L': contrast('P', 'L', repertoire_ci)}},
                stop_rules_dec23={v: dict(fired=['x'] if v in fired else []) for v in ('R', 'L', 'P', 'D', 'PD')},
                gate=dict(passed=True), descriptive=dict(quantities=dict(switches_with_d2=dict(count=58))))


class ReadingTests(unittest.TestCase):
    """DEC-28's Reading, applied mechanically to the primary; the repertoire bullet only with a null primary."""

    def test_positive_null_flag_and_blocked(self):
        positive = reading_lines(fake_result((0.01, 0.05), switch_ci=(0.01, 0.04)))
        self.assertIn('carries information beyond its active repertoire', positive[1])
        self.assertIn('It lives in the switch part', positive[1])
        self.assertTrue(positive[-1].startswith('Secondary, pre-specified'))
        repertoire_ignored = reading_lines(fake_result((0.01, 0.05), repertoire_ci=(0.02, 0.06)))
        self.assertFalse(any('active repertoire, consistent' in line for line in repertoire_ignored))
        null = reading_lines(fake_result((-0.02, 0.03)))
        self.assertIn('Not detectable at this data size', null[1])
        self.assertIn('only 58 switches', null[1])
        self.assertIn('does not apply', null[2])
        null_repertoire = reading_lines(fake_result((-0.02, 0.03), repertoire_ci=(0.02, 0.06)))
        self.assertIn("switch choices follow the whale's active repertoire", null_repertoire[2])
        self.assertIn('not applied', reading_lines(fake_result((-0.02, 0.03), repertoire_ci=(0.02, 0.06), fired=('L',)))[2])
        self.assertIn('fitting flag', reading_lines(fake_result((-0.05, -0.01)))[1])
        for variant in ('R', 'P', 'PD'):
            self.assertIn("DEC-23's stop rule fires", reading_lines(fake_result((0.01, 0.05), fired=(variant,)))[1])
        self.assertIn('carries information', reading_lines(fake_result((0.01, 0.05), fired=('D',)))[1])


if __name__ == '__main__':
    unittest.main()
