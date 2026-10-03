import unittest
import numpy as np

from whale_coda_dynamics.neural_reproduction import AssayagReproductionModel, fit_reproduction, predict_reproduction


def make_rows(n, seed, vocab):
    rng = np.random.RandomState(seed)
    return [dict(coda_id=f'c{seed}-{i}', Ct=vocab[i % len(vocab)],
                B_history=[vocab[(i + j) % len(vocab)] for j in range(1 + i % 3)],
                A_history=[vocab[i % len(vocab)]], split='train', recording_date='2020-01-01')
           for i in range(n)]


class NeuralReproductionTests(unittest.TestCase):
    def test_hidden_dim_must_be_even(self):
        model = AssayagReproductionModel(vocab_size=5, embedding_dim=4, hidden_dim=8, num_layers=1,
                                         dropout_rate=0.1, pad_idx=5)
        self.assertEqual(model.output.in_features, 4)

    def test_fit_and_predict(self):
        vocab = ['a', 'b', 'c', 'UNK']
        rows = make_rows(10, 1, vocab)
        config = dict(embedding_dim=4, hidden_dim=8, num_layers=1, dropout_rate=0.1, learning_rate=1e-2,
                      batch_size=4, max_epochs=3, patience=3)
        model, conv = fit_reproduction(rows, rows, vocab, config, seed=0, max_len=10)
        probs = predict_reproduction(model, rows, vocab, max_len=10)
        self.assertEqual(probs.shape, (len(rows), len(vocab)))
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
        self.assertTrue(np.isfinite(conv['best_validation_log_likelihood_bits']))


if __name__ == '__main__':
    unittest.main()
