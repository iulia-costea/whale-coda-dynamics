"""RQ2 token inventories (PLAN §7, DEC-22): Sharma et al.'s extracted templates and tempo classes.

The re-extraction of `sharma_inventory.json` needs NumPy to unpickle.
"""
import importlib.util
import unittest
from pathlib import Path

from whale_coda_dynamics.preprocessing.io_utils import read_json

ROOT = Path(__file__).resolve().parents[1]
SHARMA_CONFIG = read_json(ROOT / 'config' / 'sharma_inventory.json')
SHARMA = read_json(ROOT / SHARMA_CONFIG['output'])


class SourceCheckTests(unittest.TestCase):
    def test_tempos_dict_boundaries(self):
        self.assertEqual(SHARMA['empty_tempo_classes'], [5])
        self.assertEqual([round(b, 3) for b in SHARMA['tempo_boundaries_s']], [0.392, 0.594, 0.907, 1.091])
        self.assertEqual(SHARMA['template_click_counts'], [len(t) for t in SHARMA['rhythm_templates']])
        self.assertEqual(len(SHARMA['rhythm_templates']), 18)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'unpickling the Sharma files needs numpy')
    def test_reextraction_reproduces_committed_sharma_inventory(self):
        from whale_coda_dynamics.sharma_inventory import extract, serialize
        committed = (ROOT / SHARMA_CONFIG['output']).read_bytes()
        self.assertEqual(serialize(extract(ROOT, SHARMA_CONFIG)).encode('utf-8'), committed)


if __name__ == '__main__':
    unittest.main()
