from collections import Counter
from copy import deepcopy
from pathlib import Path
import unittest

from whale_coda_dynamics.preprocessing.folds import day_folds, prepare_fold
from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl


class FoldTests(unittest.TestCase):
    def test_ring_roles_are_balanced_and_order_independent(self):
        days = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']
        folds = list(day_folds(days, 42))
        self.assertEqual(folds, list(day_folds(reversed(days), 42)))
        self.assertEqual(Counter(f['validation_day'] for f in folds), Counter(days))
        self.assertEqual(Counter(f['test_day'] for f in folds), Counter(days))
        for f in folds:
            self.assertEqual(Counter(f['day_splits'].values()), {'train': 2, 'validation': 1, 'test': 1})
        with self.assertRaises(ValueError):
            list(day_folds(days[:2], 42))

    def test_heldout_labels_and_old_mapping_do_not_affect_training(self):
        root = Path(__file__).resolve().parents[1]
        rows = read_jsonl(root / 'data/processed/rq1_targets.jsonl')
        config = read_json(root / 'config/rq1.json')
        fold = next(day_folds([r['recording_date'] for r in rows], 42))
        _, original = prepare_fold(rows, fold, config)
        changed = deepcopy(rows)
        for r in changed:
            r['Ct'] = 'OLD_MAPPING_MUST_BE_IGNORED'
            r['A_history'] = ['OLD_MAPPING_MUST_BE_IGNORED']
            if fold['day_splits'][r['recording_date']] != 'train':
                r['Ct_raw'] = 'HELDOUT_ONLY'
        mapped, updated = prepare_fold(changed, fold, config)
        for field in ('vocabulary', 'training_type_counts', 'label_prior'):
            self.assertEqual(original[field], updated[field])
        self.assertTrue(all(r['Ct'] == 'UNK' for r in mapped if r['split'] != 'train'))
        self.assertNotIn('HELDOUT_ONLY', updated['vocabulary'])
        self.assertEqual(rows[0]['Ct'], read_jsonl(root / 'data/processed/rq1_targets.jsonl')[0]['Ct'])
