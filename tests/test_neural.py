import unittest
import numpy as np
import torch

from whale_coda_dynamics.neural import (
    encode_history, ConditionModel, HistoryEncoder, build_batch, fit_condition, predict, check_against_logistic,
)


def make_rows(n, seed, vocab):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        a_len = 1 + i % 3
        b_len = 1 + (i + 1) % 3
        rows.append(dict(coda_id=f'c{seed}-{i}', Ct=vocab[i % len(vocab)],
                         A_history=[vocab[(i + j) % len(vocab)] for j in range(a_len)],
                         B_history=[vocab[(i + j + 1) % len(vocab)] for j in range(b_len)],
                         turn_category='switch' if i % 2 else 'self_continuation',
                         gap_s=float(rng.rand()), overlap_elapsed_s=0.0, split='train',
                         recording_date='2020-01-01'))
    return rows


class NeuralTests(unittest.TestCase):
    def test_encode_history_padding_and_lengths(self):
        vocab = ['a', 'b', 'c']
        rows = [dict(A_history=['a', 'b']), dict(A_history=['c'])]
        seqs, lengths = encode_history(rows, 'A_history', vocab, max_len=4)
        np.testing.assert_array_equal(lengths, [2, 1])
        pad = len(vocab)
        np.testing.assert_array_equal(seqs[0], [0, 1, pad, pad])
        np.testing.assert_array_equal(seqs[1], [2, pad, pad, pad])

    def test_encode_history_rejects_empty(self):
        with self.assertRaises(ValueError):
            encode_history([dict(A_history=[])], 'A_history', ['a'], max_len=3)

    def test_absent_stream_has_no_parameters(self):
        vocab = ['a', 'b', 'UNK']
        config = dict(hidden_dim=8, embedding_dim=4)
        model = ConditionModel(len(vocab), ['A_history'], config, pad_idx=len(vocab), timing_dim=None)
        self.assertTrue(hasattr(model, 'a_encoder'))
        self.assertFalse(hasattr(model, 'b_encoder'))
        self.assertFalse(hasattr(model, 's_encoder'))
        self.assertEqual(model.output.in_features, 8)

    def test_forward_shapes_and_probabilities(self):
        vocab = ['a', 'b', 'c', 'UNK']
        rows = make_rows(6, 1, vocab)
        config = dict(hidden_dim=8, embedding_dim=4, learning_rate=1e-2, weight_decay=0.0,
                      batch_size=3, max_epochs=3, patience=3)
        model, timing_encoder, conv = fit_condition(rows, rows, vocab, ['A_history', 'B_history'], config, seed=0, max_len=10)
        probs = predict(model, timing_encoder, rows, vocab, ['A_history', 'B_history'], max_len=10)
        self.assertEqual(probs.shape, (len(rows), len(vocab)))
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
        self.assertTrue(np.isfinite(conv['best_validation_log_likelihood_bits']))

    def test_timing_only_condition(self):
        vocab = ['a', 'b', 'UNK']
        rows = make_rows(6, 2, vocab)
        config = dict(hidden_dim=6, embedding_dim=4, learning_rate=1e-2, weight_decay=0.0,
                      batch_size=3, max_epochs=2, patience=2)
        fields = ['turn_category', 'gap_s', 'overlap_elapsed_s']
        model, timing_encoder, conv = fit_condition(rows, rows, vocab, fields, config, seed=0, max_len=10)
        self.assertFalse(hasattr(model, 'a_encoder'))
        self.assertTrue(hasattr(model, 's_encoder'))
        probs = predict(model, timing_encoder, rows, vocab, fields, max_len=10)
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)

    def test_seed_determinism(self):
        vocab = ['a', 'b', 'c', 'UNK']
        rows = make_rows(8, 3, vocab)
        config = dict(hidden_dim=8, embedding_dim=4, learning_rate=1e-2, weight_decay=0.0,
                      batch_size=4, max_epochs=3, patience=3)
        m1, t1, c1 = fit_condition(rows, rows, vocab, ['A_history'], config, seed=7, max_len=10)
        m2, t2, c2 = fit_condition(rows, rows, vocab, ['A_history'], config, seed=7, max_len=10)
        p1 = predict(m1, t1, rows, vocab, ['A_history'], max_len=10)
        p2 = predict(m2, t2, rows, vocab, ['A_history'], max_len=10)
        np.testing.assert_allclose(p1, p2)

    def test_check_against_logistic_flags_reliable_underperformance(self):
        days = [f'2020-01-0{i}' for i in range(1, 8)]
        lstm_scored = []
        logistic_scored = []
        for d in days:
            for j in range(5):
                coda_id = f'{d}-{j}'
                lstm_scored.append(dict(coda_id=coda_id, condition='A', seed=0, recording_date=d, log_likelihood_bits=-2.0))
                logistic_scored.append(dict(coda_id=coda_id, condition='A', log_likelihood_bits=-1.0))
        import json, tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            root = __import__('pathlib').Path(tmp)
            out = root / 'data' / 'results' / 'logistic' / 'evaluation'
            out.mkdir(parents=True)
            with open(out / 'scored_predictions.jsonl', 'w', encoding='utf-8') as f:
                for r in logistic_scored:
                    f.write(json.dumps(r) + '\n')
            logistic_summary = dict(configuration=dict(output_directory='data/results/logistic'))
            evaluation_config = dict(bootstrap_replicates=200, bootstrap_seed=1, confidence=0.95)
            reasons = check_against_logistic(lstm_scored, logistic_summary, root, evaluation_config)
        self.assertEqual(len(reasons), 1)
        self.assertIn('A', reasons[0])


if __name__ == '__main__':
    unittest.main()
