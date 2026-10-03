import unittest

from whale_coda_dynamics.rq3.experiments.evaluation import (
    aggregate_transfer_loss,
    mean_bits_per_symbol,
)


class EvaluationAggregationTests(unittest.TestCase):
    def test_bits_per_symbol_and_transfer_loss_are_symbol_weighted(self):
        scores = [
            {"bits_per_coda": 10.0, "token_count": 2},
            {"bits_per_coda": 4.0, "token_count": 8},
        ]
        self.assertEqual(mean_bits_per_symbol(scores), 1.4)
        losses = [
            {
                "n_symbols": 2,
                "own_bits_per_coda": 10.0,
                "transfer_bits_per_coda": 8.0,
                "transfer_loss_bits_per_coda": 2.0,
            },
            {
                "n_symbols": 8,
                "own_bits_per_coda": 4.0,
                "transfer_bits_per_coda": 1.0,
                "transfer_loss_bits_per_coda": 3.0,
            },
        ]
        aggregate = aggregate_transfer_loss(losses)
        self.assertEqual(aggregate["mean_transfer_loss_bits_per_coda"], 2.5)
        self.assertEqual(aggregate["mean_transfer_loss_bits_per_symbol"], 0.5)
