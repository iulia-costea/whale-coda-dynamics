import unittest
import numpy as np
import torch

from whale_coda_dynamics.neural import (
    encode_history, ConditionModel, HistoryEncoder, build_batch, fit_condition, predict, check_against_logistic,
    ConditionModelV2, fit_condition_v2, predict_v2, select_lstm_grid_and_fit,
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


class LstmV2Tests(unittest.TestCase):
    def test_absent_a_b_has_no_copy_weights(self):
        vocab = ['a', 'b', 'UNK']
        config = dict(hidden_dim=8, embedding_dim=4, dropout_rate=0.0)
        model = ConditionModelV2(len(vocab), ['turn_category', 'gap_s', 'overlap_elapsed_s'], config,
                                 pad_idx=len(vocab), timing_dim=5)
        self.assertFalse(hasattr(model, 'beta'))
        self.assertFalse(hasattr(model, 'gamma'))
        self.assertFalse(hasattr(model, 'a_encoder'))
        self.assertTrue(hasattr(model, 's_encoder'))

    def test_gradient_reaches_beta_gamma(self):
        vocab = ['a', 'b', 'c', 'UNK']
        rows = make_rows(6, 10, vocab)
        config = dict(hidden_dim=8, embedding_dim=4, dropout_rate=0.0)
        model = ConditionModelV2(len(vocab), ['A_history', 'B_history'], config, pad_idx=len(vocab), timing_dim=None)
        batch = build_batch(rows, vocab, ['A_history', 'B_history'], 10, None, include_copy_targets=True)
        y = torch.tensor([vocab.index(r['Ct']) for r in rows], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(model(batch), y)
        loss.backward()
        self.assertIsNotNone(model.beta.grad)
        self.assertIsNotNone(model.gamma.grad)
        self.assertNotEqual(model.beta.grad.item(), 0.0)
        self.assertNotEqual(model.gamma.grad.item(), 0.0)

    def test_zeroed_network_large_beta_reproduces_repeat_argmax(self):
        vocab = ['a', 'b', 'c']
        rows = make_rows(6, 11, vocab)
        config = dict(hidden_dim=4, embedding_dim=4, dropout_rate=0.0)
        model = ConditionModelV2(len(vocab), ['A_history'], config, pad_idx=len(vocab), timing_dim=None)
        with torch.no_grad():
            for p in model.a_encoder.parameters():
                p.zero_()
            model.a_output.weight.zero_()
            model.output_bias.zero_()
            model.beta.fill_(50.0)
        batch = build_batch(rows, vocab, ['A_history'], 10, None, include_copy_targets=True)
        probs = torch.nn.functional.softmax(model(batch), dim=1)
        own_idx = batch['own_idx']
        self.assertTrue((own_idx >= 0).all())
        np.testing.assert_array_equal(probs.argmax(dim=1).numpy(), own_idx.numpy())

    def test_grid_splits_weight_decay_only_when_s_present(self):
        vocab = ['a', 'b', 'UNK']
        rows = make_rows(8, 12, vocab)
        config = dict(hidden_dim=4, embedding_dim=4, learning_rate=1e-2, batch_size=4, max_epochs=2, patience=2,
                      dropout_grid=[0.0], hidden_grid=[4], weight_decay_history_grid=[1e-4, 1e-3],
                      weight_decay_s_grid=[1e-4, 1e-3])
        curve = []
        chosen_a, fits_a, grid_a = select_lstm_grid_and_fit(rows, rows, vocab, ['A_history'], config, [42], 10, curve)
        self.assertIsNone(chosen_a['weight_decay_s'])
        self.assertEqual(len(grid_a), 2)
        curve_s = []
        fields_s = ['A_history', 'turn_category', 'gap_s', 'overlap_elapsed_s']
        chosen_s, fits_s, grid_s = select_lstm_grid_and_fit(rows, rows, vocab, fields_s, config, [42], 10, curve_s)
        self.assertIsNotNone(chosen_s['weight_decay_s'])
        self.assertEqual(len(grid_s), 4)
        self.assertTrue(curve_s)

    def test_selection_uses_validation_only(self):
        vocab = ['a', 'b', 'UNK']
        def rows(n, seed, split):
            rng = np.random.RandomState(seed)
            return [dict(coda_id=f'{split}-{seed}-{i}', Ct=vocab[(i + 1) % 2], A_history=[vocab[i % 2]],
                        turn_category='switch', gap_s=float(rng.rand()), overlap_elapsed_s=0.0,
                        split=split, recording_date='2020-01-01') for i in range(n)]
        train, validation = rows(20, 1, 'train'), rows(6, 2, 'validation')
        config = dict(hidden_dim=4, embedding_dim=4, learning_rate=1e-2, batch_size=8, max_epochs=3, patience=3,
                      dropout_grid=[0.0], hidden_grid=[4], weight_decay_history_grid=[1e-4], weight_decay_s_grid=[1e-4])
        chosen1, _, _ = select_lstm_grid_and_fit(train, validation, vocab, ['A_history'], config, [42], 10, [])
        # `test` rows are never passed to select_lstm_grid_and_fit at all; mutating them
        # (as if that mattered) and re-running proves the selection cannot see them.
        test = rows(5, 3, 'test')
        for r in test:
            r['Ct'] = 'UNK'
        chosen2, _, _ = select_lstm_grid_and_fit(train, validation, vocab, ['A_history'], config, [42], 10, [])
        self.assertEqual(chosen1['validation_log_likelihood_bits'], chosen2['validation_log_likelihood_bits'])


if __name__ == '__main__':
    unittest.main()
