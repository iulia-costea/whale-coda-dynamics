"""DEC-29 (PLAN [R2-9]): rubato.py's duration fields, variants, reuse and its R fit against logistic v4.

The onset property of the duration inputs is in tests/test_onset_information.py; the Encoder's continuous history
fields in tests/test_logistic.py; an end-to-end run in tests/test_pipeline_smoke.py.
"""
import importlib.util
import math
import os
import unittest
from pathlib import Path
from unittest import mock

HAS_MODELING = all(importlib.util.find_spec(m) for m in ('numpy', 'scipy', 'torch'))
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config' / 'rq2_rubato_inputs.json'
ONE_BLAS_THREAD = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


@unittest.skipUnless(HAS_MODELING, 'needs numpy, scipy and torch (pip install ".[modeling]")')
class DurationFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from whale_coda_dynamics import rubato
        from whale_coda_dynamics.preprocessing.io_utils import read_csv_rows, read_jsonl
        cls.rubato = rubato
        cls.loaded = rubato.load(CONFIG)
        cls.targets = read_jsonl(cls.loaded['targets'])
        cls.lookup = rubato.duration_lookup(cls.targets)
        cls.raw = read_csv_rows(ROOT / cls.loaded['config']['raw_source'], encoding='utf-8-sig')[1]
        cls.read_jsonl = staticmethod(read_jsonl)

    def test_durations_match_an_independent_recomputation_from_the_raw_rows(self):
        """Every history coda's ln d equals ln(raw Duration) of its own raw row, in every inventory, aligned with the IDs."""
        by_id = {r['coda_id']: r for r in self.targets}
        raw_log = {i: math.log(float(self.raw[r['source_row_index']]['Duration'])) for i, r in by_id.items()}
        for key, inv in self.loaded['inventories'].items():
            rows = self.rubato.with_durations(self.read_jsonl(inv['source']), self.lookup)
            checked = 0
            for r in rows:
                for ids, values in ((r['A_history_ids'], r['A_log_duration']), (r['B_history_ids'], r['B_log_duration'])):
                    self.assertEqual(len(values), len(ids))
                    for coda_id, value in zip(ids, values):
                        self.assertLess(abs(value - raw_log[coda_id]), 1e-9)
                        self.assertLessEqual(by_id[coda_id]['offset_s'], r['onset_s'])
                        checked += 1
            self.assertGreater(checked, 10000, key)

    def test_durations_stay_aligned_through_every_fold(self):
        """prepare_fold keeps the duration lists, aligned with the fold-mapped histories (inventory A on every fold,
        D and E′ on one)."""
        from whale_coda_dynamics.preprocessing.folds import prepare_fold
        for key, inv in self.loaded['inventories'].items():
            rows = self.rubato.with_durations(self.read_jsonl(inv['source']), self.lookup)
            folds = inv['manifest']['folds'] if key == 'A' else inv['manifest']['folds'][:1]
            for fold in folds:
                mapped, policy = prepare_fold(rows, fold, inv['manifest']['preparation_configuration'])
                self.assertEqual(policy, fold['vocabulary_policy'])
                for r in mapped:
                    self.assertEqual(r['A_log_duration'], [self.lookup[i]['log_duration'] for i in r['A_history_ids']])
                    self.assertEqual(r['B_log_duration'], [self.lookup[i]['log_duration'] for i in r['B_history_ids']])
                    self.assertEqual(len(r['A_log_duration']), len(r['A_history']))
                    self.assertEqual(len(r['B_log_duration']), len(r['B_history']))

    def test_history_codas_must_be_known_and_complete_at_the_onset(self):
        row = next(r for r in self.targets if r['comparison_eligible'])
        late = dict(self.lookup, **{row['A_history_ids'][-1]: dict(self.lookup[row['A_history_ids'][-1]],
                                                                   offset_s=row['onset_s'] + 0.001)})
        with self.assertRaises(ValueError):
            self.rubato.with_durations([row], late)
        with self.assertRaises(ValueError):
            self.rubato.with_durations([dict(row, B_history_ids=row['B_history_ids'] + ['no-such-coda'])], self.lookup)
        exactly = dict(self.lookup, **{row['A_history_ids'][-1]: dict(self.lookup[row['A_history_ids'][-1]],
                                                                      offset_s=row['onset_s'])})
        self.rubato.with_durations([row], exactly)  # ending exactly at the onset is complete

    def test_variant_fields_and_changed_conditions(self):
        conditions = self.loaded['inventories']['A']['manifest']['input_conditions']
        variants = self.loaded['config']['variants']
        fields, changed = self.rubato.variant_conditions(conditions, variants['R'])
        self.assertEqual((fields, changed), (conditions, []))
        fields, changed = self.rubato.variant_conditions(conditions, variants['DurA'])
        self.assertEqual(changed, ['A', 'A_B', 'A_S', 'A_B_S'])
        self.assertEqual(fields['A_B_S'], ['A_history', 'A_log_duration', 'B_history', 'turn_category', 'gap_s',
                                           'overlap_elapsed_s'])
        fields, changed = self.rubato.variant_conditions(conditions, variants['Dur'])
        self.assertEqual(changed, ['A', 'B', 'A_B', 'A_S', 'B_S', 'A_B_S'])
        self.assertEqual(fields['A_B_S'], ['A_history', 'A_log_duration', 'B_history', 'B_log_duration', 'turn_category',
                                           'gap_s', 'overlap_elapsed_s'])
        self.assertEqual(fields['S'], conditions['S'])
        self.assertEqual(self.rubato.canonical_fit(self.loaded['config'], conditions, 'Dur', 'A_S'), 'DurA')
        self.assertEqual(self.rubato.canonical_fit(self.loaded['config'], conditions, 'Dur', 'S'), 'R')
        self.assertEqual(self.rubato.canonical_fit(self.loaded['config'], conditions, 'Dur', 'A_B_S'), 'Dur')


@unittest.skipUnless(HAS_MODELING, 'needs numpy, scipy and torch (pip install ".[modeling]")')
class MergeTests(unittest.TestCase):
    def fold(self, fold_id, tag, conditions):
        return dict(fold_id=fold_id, conditions=conditions, baselines=[dict(coda_id=f'{fold_id}-b')],
                    **{k: [dict(condition=c, source=tag, fold_id=fold_id) for c in conditions]
                       for k in ('predictions', 'models', 'diagnostics', 'selections')})

    def test_reused_conditions_come_from_the_reference_in_fit_order(self):
        from whale_coda_dynamics.rubato import merge_folds
        manifest = dict(folds=[dict(fold_id='f1'), dict(fold_id='f2')], input_conditions={'none': [], 'S': [], 'A': [], 'A_S': []})
        reference = {f: self.fold(f, 'R', ['none', 'S', 'A', 'A_S']) for f in ('f1', 'f2')}
        mine = [self.fold('f1', 'DurA', ['none', 'S', 'A', 'A_S']), self.fold('f2', 'DurA', ['A', 'A_S'])]
        merged = merge_folds(manifest, mine, reference, ['none', 'S'])
        self.assertEqual([(p['fold_id'], p['condition'], p['source']) for p in merged['predictions']],
                         [('f1', 'none', 'R'), ('f1', 'S', 'R'), ('f1', 'A', 'DurA'), ('f1', 'A_S', 'DurA'),
                          ('f2', 'none', 'R'), ('f2', 'S', 'R'), ('f2', 'A', 'DurA'), ('f2', 'A_S', 'DurA')])
        self.assertEqual([b['coda_id'] for b in merged['baselines']], ['f1-b', 'f2-b'])
        with self.assertRaises(ValueError):  # a condition neither fitted nor reused
            merge_folds(manifest, mine, reference, ['none'])
        with self.assertRaises(ValueError):  # baselines must agree with the reference's
            merge_folds(manifest, [dict(mine[0], baselines=[]), mine[1]], reference, ['none', 'S'])


@unittest.skipUnless(HAS_MODELING, 'needs numpy, scipy and torch (pip install ".[modeling]")')
class ReferenceFitTests(unittest.TestCase):
    def test_without_duration_fields_a_fold_fit_is_v4s(self):
        """Variant R through rubato's fold task (rows carrying duration lists) equals logistic v4's own fold fit on the
        plain rows bit for bit, and the committed v4 run's choice, up to the cross-machine tolerance (CLAUDE.md)."""
        from whale_coda_dynamics import rubato
        from whale_coda_dynamics.logistic import fit_v2_fold
        from whale_coda_dynamics.preprocessing.folds import prepare_fold
        from whale_coda_dynamics.preprocessing.io_utils import read_jsonl
        loaded = rubato.load(CONFIG)
        inv = loaded['inventories']['A']
        fold = inv['manifest']['folds'][1]
        with mock.patch.dict(os.environ, ONE_BLAS_THREAD):
            task = rubato._fit_task((str(CONFIG), 'A', 'R', fold['fold_id'], ['none', 'A']))
        mapped, policy = prepare_fold(read_jsonl(inv['source']), fold, inv['manifest']['preparation_configuration'])
        direct = fit_v2_fold(inv['logistic_config'], fold, mapped, policy,
                             {c: inv['manifest']['input_conditions'][c] for c in ('none', 'A')})
        for key in ('predictions', 'baselines', 'models', 'diagnostics', 'selections'):
            self.assertEqual(task[key], direct[key], key)
        committed = {(r['fold_id'], r['condition']): r for r in read_jsonl(inv['v4_directory'] / 'l2_selection.jsonl')}
        self.assertEqual(task['selections'][0]['chosen_l2'], committed[fold['fold_id'], 'A']['chosen_l2'])
        stored = {r['coda_id']: r['probabilities'] for r in read_jsonl(inv['v4_directory'] / 'predictions.jsonl')
                  if r['condition'] == 'A' and r['fold_id'] == fold['fold_id']}
        mine = {p['coda_id']: p['probabilities'] for p in task['predictions'] if p['condition'] == 'A'}
        self.assertEqual(set(mine), set(stored))
        self.assertLess(max(abs(p[t] - stored[c][t]) for c, p in mine.items() for t in p), 1e-4)

    def test_fitting_refuses_more_than_one_blas_thread(self):
        from whale_coda_dynamics import rubato
        with mock.patch.dict(os.environ, dict(ONE_BLAS_THREAD, OPENBLAS_NUM_THREADS='4')), self.assertRaises(ValueError):
            rubato._fit_task((str(CONFIG), 'A', 'R', 'test-2014-04-16', ['A']))


if __name__ == '__main__':
    unittest.main()
