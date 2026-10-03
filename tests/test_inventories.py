"""RQ2 token inventories (PLAN §7, DEC-22): the rules, their checks and the committed inventory files.

Standard library only, except the re-extraction of `sharma_inventory.json`, which needs NumPy to unpickle.
"""
from collections import Counter
import importlib.util
import pickle
import unittest
from pathlib import Path

from whale_coda_dynamics.preprocessing.inventories import (
    check_label, check_rhythm_reproduction, kde_cuts, rhythm_label, tempo_class,
)
from whale_coda_dynamics.preprocessing.io_utils import read_csv_rows, read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / 'config' / 'rq2_inventories.json')
SHARMA = read_json(ROOT / CONFIG['sharma_inventory'])
LABEL_FIELDS = {'Ct_raw', 'Ct', 'A_history', 'B_history'}
# PLAN §7 [R2-0]: labels in use, top-label share and own-last repeats on the 1,547 comparison targets.
R2_0 = {'b_dtree': (25, '56.9%', 1280), 'c_rhythm': (16, '61.9%', 1215), 'd_rhythm_tempo': (42, '39.9%', 1142),
        'e_type_tempo': (39, '39.9%', 1204), 'd_rhythm_tempo_tdict': (47, '39.5%', 1147),
        'e_type_tempo_tdict': (41, '39.4%', 1210)}


def inventory_rows(name):
    return read_jsonl(ROOT / CONFIG['output_directory'] / f'rq2_{name}_targets.jsonl')


class RuleTests(unittest.TestCase):
    rule = CONFIG['rhythm']
    templates = [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.2, 1.0], [0.0, 0.2, 0.4, 1.0]]

    def test_tempo_class_counts_cuts_strictly_below(self):
        cuts = [0.45, 0.61, 0.93, 1.08]
        self.assertEqual([tempo_class(d, cuts) for d in (0.2, 0.45, 0.4501, 0.93, 1.5)], [0, 0, 1, 2, 4])
        with self.assertRaises(ValueError):
            tempo_class(0.0, cuts)

    def test_rhythm_rule_out_of_range_ties_and_invalid_intervals(self):
        self.assertEqual(rhythm_label(2, [], self.templates, self.rule), (17, 0))
        self.assertEqual(rhythm_label(11, [], self.templates, self.rule), (17, 0))
        self.assertEqual(rhythm_label(3, [1.0, 1.0], self.templates, self.rule), (0, 1))  # tie: lowest index
        self.assertEqual(rhythm_label(3, [0.1, 0.4, 9.9], self.templates, self.rule), (2, 0))  # only n-1 intervals used
        for intervals in ([0.0, 0.0], [1.0], [1.0, -0.5]):
            with self.assertRaises(ValueError):
                rhythm_label(3, intervals, self.templates, self.rule)
        with self.assertRaises(ValueError):
            rhythm_label(5, [0.1] * 4, self.templates, self.rule)  # no 5-click template

    def test_labels_reject_unk_and_stray_separators(self):
        check_label('1+1+3', 'UNK', combined=False)
        check_label('r2|t4', 'UNK', combined=True)
        for label, combined in (('UNK', False), ('', False), ('a|b', False), ('r2|4', True), ('r2|t4|t1', True)):
            with self.assertRaises(ValueError):
                check_label(label, 'UNK', combined)


class SourceCheckTests(unittest.TestCase):
    def test_kde_reproduces_cuts_and_modes(self):
        _, rows = read_csv_rows(ROOT / CONFIG['tempo_source'], encoding='utf-8-sig')
        result = kde_cuts(rows, CONFIG['kde'])
        self.assertEqual(result['minima_s'], [0.45, 0.61, 0.93, 1.08])
        self.assertEqual(result['maxima_s'], [0.33, 0.51, 0.8, 1.02, 1.26])
        with self.assertRaises(ValueError):
            kde_cuts(rows, dict(CONFIG['kde'], bandwidth_s=0.05))

    def test_rhythm_rule_reproduces_rhythms_p_on_every_d2_row(self):
        _, rows = read_csv_rows(ROOT / CONFIG['rhythm_source'], encoding='utf-8-sig')
        rhythms = pickle.loads((ROOT / CONFIG['rhythm_labels']).read_bytes())
        result = check_rhythm_reproduction(rows, rhythms, SHARMA['rhythm_templates'], CONFIG['rhythm'])
        self.assertEqual((result['reproduced'], result['rows'], result['ties']), (3840, 3840, 0))
        changed = list(rhythms)
        changed[0] = (changed[0] + 1) % 18
        with self.assertRaises(ValueError):
            check_rhythm_reproduction(rows, changed, SHARMA['rhythm_templates'], CONFIG['rhythm'])

    def test_tempos_dict_boundaries(self):
        self.assertEqual(SHARMA['empty_tempo_classes'], [5])
        self.assertEqual([round(b, 3) for b in SHARMA['tempo_boundaries_s']], [0.392, 0.594, 0.907, 1.091])
        self.assertEqual(SHARMA['template_click_counts'], [len(t) for t in SHARMA['rhythm_templates']])
        self.assertEqual(len(SHARMA['rhythm_templates']), 18)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'unpickling the Sharma files needs numpy')
    def test_reextraction_reproduces_committed_sharma_inventory(self):
        from whale_coda_dynamics.sharma_inventory import extract, serialize
        config = read_json(ROOT / 'config' / 'sharma_inventory.json')
        committed = (ROOT / config['output']).read_bytes()
        self.assertEqual(serialize(extract(ROOT, config)).encode('utf-8'), committed)


class InventoryFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rq1 = read_jsonl(ROOT / CONFIG['targets'])
        cls.inventories = {name: inventory_rows(name) for name in CONFIG['inventories']}

    def test_only_label_fields_and_inventory_differ(self):
        for name, rows in self.inventories.items():
            with self.subTest(inventory=name):
                self.assertEqual([r['coda_id'] for r in rows], [r['coda_id'] for r in self.rq1])
                for before, after in zip(self.rq1, rows):
                    self.assertEqual(set(after), set(before) | {'inventory'})
                    self.assertEqual(after['inventory'], name)
                    self.assertEqual({k: v for k, v in after.items() if k not in LABEL_FIELDS | {'inventory'}},
                                     {k: v for k, v in before.items() if k not in LABEL_FIELDS})
                    self.assertEqual(after['Ct'], after['Ct_raw'])

    def test_histories_are_the_labels_of_their_ids(self):
        for name, rows in self.inventories.items():
            labels = {r['coda_id']: r['Ct_raw'] for r in rows}
            with self.subTest(inventory=name):
                self.assertNotIn('UNK', labels.values())
                for row in rows:
                    for prefix in ('A', 'B'):
                        self.assertEqual(row[prefix + '_history'], [labels[i] for i in row[prefix + '_history_ids']])

    def test_combined_labels_nest_their_scheme(self):
        named = {r['coda_id']: r['Ct_raw'] for r in self.rq1}
        rhythm = {r['coda_id']: r['Ct_raw'] for r in self.inventories['c_rhythm']}
        for name, scheme in (('e_type_tempo', named), ('e_type_tempo_tdict', named),
                             ('d_rhythm_tempo', rhythm), ('d_rhythm_tempo_tdict', rhythm)):
            with self.subTest(inventory=name):
                for row in self.inventories[name]:
                    base, tempo = row['Ct_raw'].split('|')
                    self.assertEqual(base, scheme[row['coda_id']])
                    self.assertIn(tempo, {'t0', 't1', 't2', 't3', 't4'})

    def test_counts_match_plan_r2_0(self):
        """Gate A: labels, top-label share and repeats among the comparison targets equal PLAN [R2-0]."""
        for name, expected in R2_0.items():
            rows = self.inventories[name]
            labels = {r['coda_id']: r['Ct_raw'] for r in rows}
            comparison = [r for r in rows if r['comparison_eligible']]
            counts = Counter(r['Ct_raw'] for r in comparison)
            repeats = sum(r['Ct_raw'] == labels[r['A_history_ids'][-1]] for r in comparison)
            with self.subTest(inventory=name):
                self.assertEqual((len(counts), f"{max(counts.values()) / len(comparison):.1%}", repeats), expected)


if __name__ == '__main__':
    unittest.main()
