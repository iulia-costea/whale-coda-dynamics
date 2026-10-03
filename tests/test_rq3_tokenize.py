import unittest

from whale_coda_dynamics.rq3.preprocessing.tokenize import ici_to_symbol


class TokenizerTests(unittest.TestCase):
    def test_exact_fixed_width_bins(self):
        expected = {
            0.0499: 0,
            0.05: 1,
            0.1499: 2,
            0.15: 3,
            0.3: 6,
            0.35: 7,
            0.6: 12,
            0.7: 14,
            0.9999: 19,
        }
        for ici_s, symbol in expected.items():
            with self.subTest(ici_s=ici_s):
                self.assertEqual(ici_to_symbol(ici_s), symbol)

    def test_out_of_range_ici_is_rejected(self):
        for ici_s in (0.0049, 1.0):
            with self.subTest(ici_s=ici_s):
                with self.assertRaises(ValueError):
                    ici_to_symbol(ici_s)
