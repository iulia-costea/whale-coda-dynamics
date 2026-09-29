import unittest

import numpy as np
import torch

from whale_coda_dynamics.neural import (
    ConditionModelV2, build_batch, fit_condition_v2, mean_ll_bits,
    select_neural_setting, v2_probabilities, v2_settings,
)


def rows(split, n=8):
    return [dict(coda_id=f'{split}-{i}', split=split, Ct=['a', 'b'][i % 2],
                 A_history=[['a', 'b'][i % 2]], B_history=[['a', 'b'][(i + 1) % 2]],
                 turn_category='switch', gap_s=float(i % 3), overlap_elapsed_s=0.) for i in range(n)]


CONFIG = dict(hidden_dim=4, embedding_dim=3, dropout=0., weight_decay=1e-4, s_weight_decay=.1,
              learning_rate=.001, batch_size=4, max_epochs=2, patience=2, device='cpu',
              hidden_grid=[4], dropout_grid=[0., .2], weight_decay_grid=[1e-4],
              s_weight_decay_grid=[.1], seeds=[42], workers=1)
VOCAB = ['UNK', 'a', 'b']


class NeuralV2Tests(unittest.TestCase):
    def test_copy_gradients_and_repeat_argmax(self):
        fields = ['A_history', 'B_history']
        model = ConditionModelV2(len(VOCAB), fields, CONFIG, None)
        batch = build_batch(rows('train'), VOCAB, fields, 3, None)
        loss = torch.nn.functional.cross_entropy(model(batch), torch.tensor([1, 2] * 4))
        loss.backward()
        self.assertIsNotNone(model.beta.grad)
        self.assertIsNotNone(model.gamma.grad)
        self.assertNotEqual(model.beta.grad.item(), 0.)
        self.assertNotEqual(model.gamma.grad.item(), 0.)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.beta.fill_(50.)
        np.testing.assert_array_equal(model(batch).argmax(1).detach().numpy(), batch['a_seq'][:, 0].numpy())

    def test_absent_sources_and_optimizer_partition(self):
        for fields, timing in [(['A_history'], None), (['B_history'], None), (['gap_s'], 2),
                               (['A_history', 'B_history', 'gap_s'], 2)]:
            model = ConditionModelV2(len(VOCAB), fields, CONFIG, timing)
            self.assertEqual(hasattr(model, 'beta'), 'A_history' in fields)
            self.assertEqual(hasattr(model, 'gamma'), 'B_history' in fields)
            groups = model.optimizer_groups(CONFIG)
            assigned = [id(p) for g in groups for p in g['params']]
            self.assertEqual(len(assigned), len(set(assigned)))
            self.assertEqual(set(assigned), {id(p) for p in model.parameters()})
            if timing is not None:
                self.assertEqual({id(p) for p in groups[1]['params']},
                                 {id(p) for n, p in model.named_parameters() if n.startswith(('s_encoder.', 's_output.'))})
            self.assertIn(id(model.output_bias), {id(p) for p in groups[0]['params']})

    def test_selection_uses_validation_only(self):
        train, validation, test = rows('train'), rows('validation'), rows('test')
        fields = ['A_history', 'gap_s']
        chosen1, candidates1, fits1 = select_neural_setting(train, validation, VOCAB, fields, CONFIG, 3)
        for r in test:
            r.update(Ct='UNK', A_history=['UNK'] * 3, gap_s=999.)
        chosen2, candidates2, fits2 = select_neural_setting(train, validation, VOCAB, fields, CONFIG, 3)
        self.assertEqual(chosen1, chosen2)
        self.assertEqual(candidates1, candidates2)
        for f1, f2 in zip(fits1, fits2):
            for name in f1['state']:
                np.testing.assert_array_equal(f1['state'][name], f2['state'][name])
        with self.assertRaises(ValueError):
            fit_condition_v2(test, validation, VOCAB, fields, CONFIG, 42, 3)

    def test_best_epoch_restored_and_logs(self):
        train, validation = rows('train'), rows('validation')
        model, encoder, conv, epochs = fit_condition_v2(train, validation, VOCAB, ['A_history'], CONFIG, 42, 3)
        probabilities = v2_probabilities(model, build_batch(validation, VOCAB, ['A_history'], 3, encoder))
        self.assertAlmostEqual(mean_ll_bits(probabilities, validation, VOCAB), conv['best_validation_log_likelihood_bits'])
        self.assertEqual(conv['best_validation_log_likelihood_bits'], max(e['validation_log_likelihood_bits'] for e in epochs))
        self.assertTrue(all(np.isfinite(e['training_log_likelihood_bits']) for e in epochs))
        self.assertEqual(conv['device'], 'cpu')
        with self.assertRaises(ValueError):
            fit_condition_v2(train, validation, VOCAB, ['A_history'], dict(CONFIG, device='mps'), 42, 3)

    def test_worker_scheduling_preserves_fits(self):
        config = dict(CONFIG, dropout_grid=[0.], seeds=[42, 43], max_epochs=1)
        args = (rows('train'), rows('validation'), VOCAB, ['A_history'])
        chosen1, candidates1, fits1 = select_neural_setting(*args, config, 3)
        chosen2, candidates2, fits2 = select_neural_setting(*args, dict(config, workers=2), 3)
        self.assertEqual(chosen1, chosen2)
        self.assertEqual(candidates1, candidates2)
        for a, b in zip(fits1, fits2):
            for name in a['state']:
                np.testing.assert_array_equal(a['state'][name], b['state'][name])

    def test_grid_omits_absent_s_and_shared_initialization(self):
        config = dict(CONFIG, hidden_dim=32, hidden_grid=[32, 64], dropout_grid=[0, .2, .4],
                      weight_decay_grid=[1e-5, 1e-4, 1e-3], s_weight_decay_grid=[1e-5, 1e-3, .1, 1., 10.])
        self.assertEqual(len(v2_settings(['A_history'], config)), 18)
        self.assertEqual(len(v2_settings(['A_history', 'gap_s'], config)), 90)
        torch.manual_seed(42)
        base = ConditionModelV2(3, ['A_history', 'B_history'], config, None)
        torch.manual_seed(42)
        full = ConditionModelV2(3, ['A_history', 'B_history', 'gap_s'], config, 2)
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, full.state_dict()[name]))
