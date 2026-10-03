"""Partner-tempo ablation (DEC-24): conditions, the factor encoder, copy-weight indices, nesting of v5's
absent candidates, and the label and fold gates."""
import json
import math
from pathlib import Path
import unittest

import numpy as np

from whale_coda_dynamics import tempo_factor
from whale_coda_dynamics.logistic import probabilities
from whale_coda_dynamics.logistic_v5 import subconditions
from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl
from whale_coda_dynamics.tempo_factor import (
    BLOCKS, CONDITIONS, CONTRASTS, FLOOR, LATTICE, FactorEncoder, check_folds, condition_blocks, factor_labels,
    fit_condition, fit_fold, model_predictions, prepare_factor_fold, select_fold, split_label, stop_family,
    validation_monotonicity,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPO = ['t0', 't1', 't2', 't3', 't4']
PATTERNS = ['r1', 'r2', 'r3', 'UNK']
S_FIELDS = ['turn_category', 'gap_s', 'overlap_elapsed_s']
CONFIG = dict(history_order=3, l2_grid=[0.01, 0.1], max_iterations=2000, gradient_tolerance=1e-6)


def synthetic_rows(n, seed, split, copy_partner=False):
    """Factor rows as prepare_factor_fold builds them; with copy_partner the tempo is the partner's last."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        category = ('switch', 'overlap', 'self_continuation')[i % 3]
        row = dict(coda_id=f'{split}{i}', split=split, recording_date='2015-03-02', turn_category=category,
                   gap_s=float(rng.rand()) * (category != 'overlap'), overlap_elapsed_s=float(rng.rand()) * (category == 'overlap'),
                   pattern=PATTERNS[rng.randint(len(PATTERNS))],
                   own_tempo=[TEMPO[rng.randint(5)] for _ in range(1 + i % 4)],
                   own_pattern=[PATTERNS[rng.randint(len(PATTERNS))] for _ in range(1 + i % 4)],
                   partner_tempo=[TEMPO[rng.randint(5)] for _ in range(1 + (i + 1) % 3)],
                   partner_pattern=[PATTERNS[rng.randint(len(PATTERNS))] for _ in range(1 + (i + 1) % 3)])
        row['tempo'] = row['partner_tempo'][-1] if copy_partner else TEMPO[rng.randint(5)]
        rows.append(row)
    return rows


def synthetic_fold(copy_partner=False):
    return dict(fold=dict(fold_id='f'), pattern_policy=dict(vocabulary=PATTERNS),
                train=synthetic_rows(60, 1, 'train', copy_partner), validation=synthetic_rows(15, 2, 'validation', copy_partner),
                test=synthetic_rows(10, 3, 'test', copy_partner))


class ConditionTests(unittest.TestCase):
    def test_lattice_floor_and_names(self):
        self.assertEqual(LATTICE, ['own', 'own_T', 'own_P', 'own_S', 'own_T_P', 'own_T_S', 'own_P_S', 'own_T_P_S'])
        self.assertEqual(CONDITIONS, [FLOOR] + LATTICE)
        self.assertIsNone(condition_blocks(FLOOR))
        self.assertEqual(condition_blocks('own_T_S'), ('T', 'S'))
        for bad in ('own_S_T', 'none', 'T_S', 'own_A'):
            with self.assertRaises(ValueError):
                condition_blocks(bad)

    def test_stop_family_is_every_add_one_block_pair(self):
        family = stop_family()
        self.assertEqual(len(family), 13)
        self.assertEqual(family[-1], ('own', FLOOR))
        for big, small in family[:-1]:
            self.assertEqual(len(set(condition_blocks(big)) - set(condition_blocks(small))), 1)
            self.assertLess(set(condition_blocks(small)), set(condition_blocks(big)))
        self.assertEqual(len(set(family)), 13)
        # DEC-24's four contrasts are add-one-block pairs, so they are in the stop family.
        for pair in CONTRASTS.values():
            self.assertIn(pair, family)


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.train = synthetic_rows(30, 1, 'train')

    def encoder(self, condition):
        return FactorEncoder(self.train, TEMPO, PATTERNS, condition, S_FIELDS)

    def test_columns_per_condition(self):
        floor = self.encoder(FLOOR)
        self.assertEqual(floor.names, ['pattern:' + p for p in PATTERNS])
        full = self.encoder('own_T_P_S')
        expected = ['pattern:' + p for p in PATTERNS]
        for stream, vocabulary in (('own_tempo', TEMPO), ('own_pattern', PATTERNS), ('partner_tempo', TEMPO),
                                   ('partner_pattern', PATTERNS)):
            expected += [f'{stream}:lag{lag}:{label}' for lag in (1, 2, 3) for label in vocabulary]
        s_names = full.s_encoder.names
        self.assertEqual(full.names, expected + s_names)
        self.assertEqual(s_names[:3], ['turn:overlap', 'turn:self_continuation', 'turn:switch'])
        # Only S columns get the S penalty; the intercept (last) is history.
        np.testing.assert_array_equal(full.timing_mask, [False] * len(expected) + [True] * len(s_names) + [False])
        self.assertEqual(full.transform(self.train).shape, (30, len(full.names) + 1))
        self.assertFalse(self.encoder('own_T_P').timing_mask.any())

    def test_every_sub_condition_is_a_column_subset_with_equal_values(self):
        rows = synthetic_rows(12, 5, 'validation')
        for condition in LATTICE:
            big = self.encoder(condition)
            x_big = big.transform(rows)
            for sub in subconditions(condition_blocks(condition), BLOCKS):
                small = self.encoder(tempo_factor.condition_name(sub))
                index = [big.names.index(n) for n in small.names]
                np.testing.assert_array_equal(x_big[:, index], small.transform(rows)[:, :-1])
        x_floor = self.encoder(FLOOR).transform(rows)
        np.testing.assert_array_equal(x_floor[:, :-1], self.encoder('own').transform(rows)[:, :len(PATTERNS)])

    def test_one_hot_lags_missing_lags_and_unknown_labels(self):
        encoder = self.encoder('own_T')
        row = dict(self.train[0], pattern='r2', own_tempo=['t1', 't4'], own_pattern=['r3', 'UNK'], partner_tempo=['t2'])
        x = dict(zip(encoder.names, encoder.transform([row])[0]))
        self.assertEqual(x['pattern:r2'], 1.0)
        self.assertEqual((x['own_tempo:lag1:t4'], x['own_tempo:lag2:t1'], x['own_pattern:lag1:UNK'], x['partner_tempo:lag1:t2']),
                         (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(sum(v for k, v in x.items() if k.startswith('own_tempo:lag3')), 0.0)
        self.assertEqual(sum(v for k, v in x.items() if k.startswith('partner_tempo:lag2')), 0.0)
        self.assertEqual(sum(x.values()), 6.0)
        for bad in (dict(row, pattern='r9'), dict(row, own_tempo=['t7']), dict(row, partner_tempo=['t1', 'r1'])):
            with self.assertRaises(ValueError):
                encoder.transform([bad])
        with self.assertRaises(ValueError):
            FactorEncoder(synthetic_rows(3, 1, 'validation'), TEMPO, PATTERNS, 'own', S_FIELDS)


class CopyWeightTests(unittest.TestCase):
    def test_indices_are_the_last_tempo_classes(self):
        rows = [dict(own_tempo=['t0', 't3'], partner_tempo=['t4', 't1']), dict(own_tempo=['t2'], partner_tempo=[])]
        train = synthetic_rows(10, 1, 'train')
        own, partner = FactorEncoder(train, TEMPO, PATTERNS, 'own_T_S', S_FIELDS).copy_indices(rows)
        np.testing.assert_array_equal(own, [3, 2])
        np.testing.assert_array_equal(partner, [1, -1])
        own, partner = FactorEncoder(train, TEMPO, PATTERNS, 'own_P_S', S_FIELDS).copy_indices(rows)
        np.testing.assert_array_equal(own, [3, 2])
        self.assertIsNone(partner)  # gamma belongs to the partner-tempo block
        self.assertEqual(FactorEncoder(train, TEMPO, PATTERNS, FLOOR, S_FIELDS).copy_indices(rows), (None, None))

    def test_copy_weights_add_to_the_last_tempo_classes_logits(self):
        rows = [dict(own_tempo=['t0', 't3'], partner_tempo=['t1'])]
        own, partner = FactorEncoder(synthetic_rows(10, 1, 'train'), TEMPO, PATTERNS, 'own_T', S_FIELDS).copy_indices(rows)
        p = probabilities(np.ones((1, 1)), np.zeros((1, 5)), 2.0, 1.0, own, partner)[0]
        logits = np.array([0, 1.0, 0, 2.0, 0])
        np.testing.assert_allclose(p, np.exp(logits) / np.exp(logits).sum(), rtol=0, atol=1e-15)

    def test_partner_tempo_copy_weight_is_learned(self):
        """When every coda repeats the partner's last tempo, gamma takes it up; without T there is no gamma."""
        prepared = synthetic_fold(copy_partner=True)
        with_t = fit_condition(prepared, 'own_T', TEMPO, S_FIELDS, CONFIG)['chosen']
        without_t = fit_condition(prepared, 'own', TEMPO, S_FIELDS, CONFIG)['chosen']
        self.assertGreater(with_t['gamma'], 1.0)
        self.assertGreater(with_t['gamma'], abs(with_t['beta']))
        self.assertIsNone(without_t['gamma'])
        self.assertIsNotNone(without_t['beta'])
        floor = fit_condition(prepared, FLOOR, TEMPO, S_FIELDS, CONFIG)['chosen']
        self.assertEqual((floor['beta'], floor['gamma'], floor['l2_s']), (None, None, None))


class NestingTests(unittest.TestCase):
    """v5's absent candidates are the smaller conditions' own fits, exactly."""

    def check_nesting(self, prepared):
        fits = fit_fold(prepared, TEMPO, S_FIELDS, CONFIG)
        selection = select_fold(prepared['fold']['fold_id'], fits)
        self.assertEqual(len(selection), len(LATTICE) * 2)
        for s in selection:
            subs = [tempo_factor.condition_name(b) for b in subconditions(condition_blocks(s['condition']), BLOCKS)]
            self.assertEqual([c['subcondition'] for c in s['candidates']], subs)
            for candidate in s['candidates']:
                self.assertEqual(candidate['validation_log_likelihood_bits'],
                                 float(np.mean(fits[candidate['subcondition']]['validation_bits'])))
                self.assertEqual(candidate['validation_log_likelihood_bits'],
                                 fits[candidate['subcondition']]['chosen']['validation_log_likelihood_bits'])
        monotonicity = validation_monotonicity(selection)
        self.assertGreaterEqual(monotonicity['min_add_block_validation_contrast_bits'], 0.0)
        fold_id = prepared['fold']['fold_id']
        own_fit = [dict(coda_id=r['coda_id'], fold_id=fold_id, condition=c, probabilities=dict(zip(TEMPO, map(float, row))))
                   for c in CONDITIONS for r, row in zip(prepared['test'], tempo_factor.test_probabilities(fits[c], prepared['test']))]
        by_key = {(p['condition'], p['coda_id']): p['probabilities'] for p in own_fit}
        chosen = {(s['condition'], s['rule']): s['chosen'] for s in selection}
        for model, rule in (('v5', 'max'), ('v5_one_se', 'one_se')):
            predictions = model_predictions(selection, own_fit)[model]
            self.assertEqual(len(predictions), len(own_fit))
            for p in predictions:
                source = FLOOR if p['condition'] == FLOOR else chosen[p['condition'], rule]
                self.assertEqual(p['probabilities'], by_key[source, p['coda_id']])
        # own has one candidate, itself.
        self.assertEqual({chosen['own', r] for r in ('max', 'one_se')}, {'own'})
        return fits, selection

    def test_synthetic_fold(self):
        self.check_nesting(synthetic_fold())

    def test_committed_fold(self):
        """One real fold of the primary run (the smallest validation day), with the committed gates."""
        manifest = read_json(ROOT / 'data/processed/rq2_d_rhythm_tempo_folds/manifest.json')
        coarse = read_json(ROOT / 'data/processed/rq2_c_rhythm_folds/manifest.json')
        rows = read_jsonl(ROOT / manifest['configuration']['input'])
        labels = factor_labels(rows, read_jsonl(ROOT / coarse['configuration']['input']), TEMPO, S_FIELDS)
        fold = min(manifest['folds'], key=lambda f: len(f['comparison_ids']['validation']))
        coarse_fold = [f for f in coarse['folds'] if f['fold_id'] == fold['fold_id']][0]
        prepared = prepare_factor_fold(rows, labels, fold, coarse_fold, manifest['preparation_configuration'], S_FIELDS)
        self.assertEqual(prepared['pattern_policy']['vocabulary'], coarse_fold['vocabulary_policy']['vocabulary'])
        for split in ('train', 'validation', 'test'):
            self.assertEqual(sorted(r['coda_id'] for r in prepared[split]), fold['comparison_ids'][split])
        # Histories come from the history IDs: each lag's tempo and pattern are that coda's own labels.
        row = next(r for r in rows if r['coda_id'] == prepared['test'][0]['coda_id'])
        self.assertEqual(prepared['test'][0]['partner_tempo'], [labels[i][1] for i in row['B_history_ids']])
        self.check_nesting(prepared)


class GateTests(unittest.TestCase):
    def test_split_label(self):
        self.assertEqual(split_label('r2|t4'), ('r2', 't4'))
        self.assertEqual(split_label('1+1+3|t0'), ('1+1+3', 't0'))
        for bad in ('r2', 'r2|t4|x', '|t4', 'r2|', None):
            with self.assertRaises(ValueError):
                split_label(bad)

    def test_committed_labels_folds_and_tempo_agree(self):
        fine = {}
        for fine_name, coarse_name in (('rq2_d_rhythm_tempo', 'rq2_c_rhythm'), ('rq2_e_type_tempo', 'rq1')):
            manifest = read_json(ROOT / f'data/processed/{fine_name}_folds/manifest.json')
            coarse = read_json(ROOT / f'data/processed/{coarse_name}_folds/manifest.json')
            check_folds(manifest, coarse, read_json(ROOT / 'data/processed/rq1_folds/manifest.json'))
            fine[fine_name] = factor_labels(read_jsonl(ROOT / manifest['configuration']['input']),
                                            read_jsonl(ROOT / coarse['configuration']['input']), TEMPO, S_FIELDS)
        tempos = [{c: t for c, (_, t) in labels.items()} for labels in fine.values()]
        self.assertEqual(tempos[0], tempos[1])
        self.assertEqual(set(tempos[0].values()), set(TEMPO))

    def test_label_gate_fails(self):
        rows = [dict(coda_id='a', Ct_raw='r2|t4', recording_date='d', comparison_eligible=True, A_history_ids=[],
                     B_history_ids=[], turn_category='switch', gap_s=1.0, overlap_elapsed_s=0.0)]
        coarse = [dict(rows[0], Ct_raw='r2')]
        self.assertEqual(factor_labels(rows, coarse, TEMPO, S_FIELDS), {'a': ('r2', 't4')})
        with self.assertRaisesRegex(ValueError, 'coarse label'):
            factor_labels(rows, [dict(coarse[0], Ct_raw='r3')], TEMPO, S_FIELDS)
        with self.assertRaisesRegex(ValueError, 'configured class'):
            factor_labels([dict(rows[0], Ct_raw='r2|t9')], coarse, TEMPO, S_FIELDS)
        with self.assertRaisesRegex(ValueError, 'fields differ'):
            factor_labels(rows, [dict(coarse[0], gap_s=2.0)], TEMPO, S_FIELDS)
        with self.assertRaisesRegex(ValueError, 'different codas'):
            factor_labels(rows, [dict(coarse[0], coda_id='b')], TEMPO, S_FIELDS)

    def test_fold_gate_fails(self):
        manifest = read_json(ROOT / 'data/processed/rq2_d_rhythm_tempo_folds/manifest.json')
        changed = json.loads(json.dumps(manifest))
        changed['folds'][0]['validation_day'] = changed['folds'][1]['validation_day']
        with self.assertRaisesRegex(ValueError, 'Folds differ'):
            check_folds(manifest, changed)
        changed = json.loads(json.dumps(manifest))
        changed['input_conditions']['S'] = ['turn_category']
        with self.assertRaisesRegex(ValueError, 'S fields'):
            check_folds(manifest, changed)


if __name__ == '__main__':
    unittest.main()
