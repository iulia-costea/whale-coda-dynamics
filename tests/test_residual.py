import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from whale_coda_dynamics.logistic import Encoder, last_label_index, probabilities, select_l2_and_fit
from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl
from whale_coda_dynamics.residual import (
    LR_ONLY, FitRunner, ResidualModel, candidate_list, condition_data, condition_tasks, encode_split, fit_residual,
    load_v4_models, log_likelihood_bits_per_target, model_settings, per_target_bits, planted_reading, predict_condition,
    prepare_folds, residual_gains, select_candidate, select_condition, softmax_rows, stored_v4_probabilities, to_tensors,
    v4_logits,
)
from whale_coda_dynamics.evaluation import log_likelihood_bits

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ['a', 'b', 'c', 'UNK']
FIELDS = ['A_history', 'B_history', 'turn_category', 'gap_s', 'overlap_elapsed_s']
CONFIG = dict(embedding_dim=4, learning_rate=1e-2, weight_decay=0.1, batch_size=4, max_epochs=3, patience=None,
              hidden_grid=[4], dropout_grid=[0.0])


def make_rows(n, seed, split):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        a = [VOCAB[rng.randint(3)] for _ in range(1 + rng.randint(8))]
        b = [VOCAB[rng.randint(3)] for _ in range(1 + rng.randint(4))]
        target = a[-1] if rng.rand() < .7 else VOCAB[rng.randint(4)]
        overlap = rng.rand() < .3
        rows.append(dict(coda_id=f'{split}-{seed}-{i}', split=split, recording_date='2020-01-01', Ct=target,
                         A_history=a, B_history=b, turn_category='overlap' if overlap else ('switch' if i % 2 else 'self_continuation'),
                         gap_s=0.0 if overlap else float(rng.rand() * 3), overlap_elapsed_s=float(rng.rand()) if overlap else 0.0))
    return rows


def v4_like_record(train, validation, fields):
    """A fitted logistic model stored exactly as logistic v4 stores it (models.jsonl, through JSON)."""
    config = dict(history_order=3, l2_grid=[0.01, 0.1], max_iterations=2000, gradient_tolerance=1e-6)
    encoder, chosen, _ = select_l2_and_fit(train, validation, VOCAB, fields, config)
    record = dict(fold_id='f', condition='c', vocabulary=VOCAB, fields=fields, order=encoder.order,
                  categories=encoder.categories, feature_names=encoder.names + ['intercept'], mean=encoder.mean.tolist(),
                  scale=encoder.scale.tolist(), weights=chosen['w'].tolist(), beta=chosen['beta'], gamma=chosen['gamma'])
    return json.loads(json.dumps(record)), encoder, chosen


def synthetic_job(fields=FIELDS, test=None):
    train, validation = make_rows(40, 1, 'train'), make_rows(12, 2, 'validation')
    test = test if test is not None else make_rows(10, 3, 'test')
    record, encoder, chosen = v4_like_record(train, validation, fields)
    prior = {t: 1 / len(VOCAB) for t in VOCAB}
    prepared = dict(fold=dict(fold_id='f'), policy=dict(vocabulary=VOCAB, label_prior=prior),
                    train=train, validation=validation, test=test)
    job = condition_data(prepared, 'c', fields, lambda rows: v4_logits(record, train, rows, VOCAB), 10)
    return job, prepared, record, encoder, chosen


class ModelTests(unittest.TestCase):
    def test_candidate_counts(self):
        for blocks, n in ((('A',), 10), (('A', 'B'), 28), (('A', 'B', 'S'), 64), ((), 1)):
            candidates = candidate_list(blocks, [16, 32, 64], [0.0, 0.2, 0.4])
            self.assertEqual(len(candidates), n)
            self.assertEqual(candidates[0]['name'], LR_ONLY)
            self.assertTrue(all(c['blocks'] for c in candidates[1:]))

    def test_absent_blocks_have_no_parameters(self):
        self.assertEqual(list(ResidualModel(4, (), 8, 0.0, 4).parameters()), [])
        model = ResidualModel(4, ('A',), 8, 0.0, 4)
        names = [n for n, _ in model.named_parameters()]
        self.assertTrue(names and all(n.startswith('a_') for n in names))
        model = ResidualModel(4, ('A', 'S'), 8, 0.2, 4, timing_dim=5)
        names = [n for n, _ in model.named_parameters()]
        self.assertFalse(any(n.startswith('b_') for n in names))
        self.assertTrue(any(n.startswith('s_') for n in names))
        for block in ('a', 's'):
            output = getattr(model, block + '_output')
            self.assertIsNone(output.bias)
            self.assertTrue(torch.equal(output.weight, torch.zeros_like(output.weight)))
        self.assertFalse(any('beta' in n or 'gamma' in n or 'bias' in n and 'output' in n for n in names))

    def test_epoch0_predictions_equal_offset_softmax(self):
        job, *_ = synthetic_job()
        train, val = job['data']['train'], job['data']['validation']
        torch.manual_seed(0)
        model = ResidualModel(len(VOCAB), job['blocks'], 4, 0.4, 4, job['timing_dim'])
        model.eval()
        with torch.no_grad():
            val_t = to_tensors(val, 'cpu')
            logits = model(val_t)
        self.assertTrue(torch.equal(logits, val_t['offset']))
        np.testing.assert_allclose(softmax_rows(logits.numpy()), job['lr_probabilities']['validation'], rtol=0, atol=1e-15)
        fit = fit_residual(train, val, job['blocks'], 4, 0.4, dict(CONFIG, max_epochs=0, patience=20), 42,
                           timing_dim=job['timing_dim'])
        self.assertEqual(fit['best_epoch'], 0)
        self.assertIsNone(fit['state_dict'])
        lr_val = float(np.mean(per_target_bits(job['lr_probabilities']['validation'], val['y'])))
        self.assertAlmostEqual(fit['curve'][0]['validation_log_likelihood_bits'], lr_val, delta=1e-12)
        chosen = candidate_list(job['blocks'], [4], [0.4])[-1]
        fits = {(chosen['name'], 42): fit}
        out = predict_condition(chosen, fits, [42], job['data']['test'], job['lr_probabilities']['test'], CONFIG, job['timing_dim'])
        self.assertIs(out[0], job['lr_probabilities']['test'])

    def test_all_absent_equals_logistic_v4_exactly(self):
        job, prepared, record, encoder, chosen = synthetic_job()
        test = prepared['test']
        expected = probabilities(encoder.transform(test), chosen['w'], chosen['beta'], chosen['gamma'],
                                 last_label_index(test, 'A_history', VOCAB), last_label_index(test, 'B_history', VOCAB))
        self.assertTrue(np.array_equal(job['lr_probabilities']['test'], expected))
        lr = candidate_list(job['blocks'], [4], [0.0])[0]
        out = predict_condition(lr, {}, [42, 43], job['data']['test'], job['lr_probabilities']['test'], CONFIG, job['timing_dim'])
        self.assertTrue(all(np.array_equal(p, expected) for p in out))

    def test_stored_encoder_must_match(self):
        train, validation = make_rows(40, 1, 'train'), make_rows(12, 2, 'validation')
        record, *_ = v4_like_record(train, validation, ['A_history', 'gap_s'])
        with self.assertRaises(ValueError):
            v4_logits(dict(record, mean=[m + 1e-9 for m in record['mean']]), train, validation, VOCAB)
        with self.assertRaises(ValueError):
            v4_logits(record, train[:-1], validation, VOCAB)

    def test_zero_initialised_u_still_trains(self):
        job, *_ = synthetic_job()
        train = to_tensors(job['data']['train'], 'cpu')
        torch.manual_seed(0)
        model = ResidualModel(len(VOCAB), job['blocks'], 4, 0.0, 4, job['timing_dim'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
        losses, encoder_grads = [], []
        for _ in range(3):
            loss = -log_likelihood_bits_per_target(model(train), train['y']).mean()
            optimizer.zero_grad()
            loss.backward()
            encoder_grads.append(float(model.a_encoder.lstm.weight_ih_l0.grad.abs().sum()))
            optimizer.step()
            losses.append(loss.item())
        self.assertEqual(encoder_grads[0], 0.0)  # U = 0 blocks the encoders' gradient at the first step...
        self.assertGreater(encoder_grads[1], 0.0)  # ...but U moves, so they train from the second step on
        self.assertGreater(float(model.a_output.weight.detach().abs().sum()), 0.0)
        self.assertNotEqual(losses[0], losses[1])
        self.assertNotEqual(losses[1], losses[2])
        fit = fit_residual(job['data']['train'], job['data']['validation'], job['blocks'], 4, 0.0, CONFIG, 42,
                           timing_dim=job['timing_dim'])
        self.assertNotEqual(fit['curve'][1]['train_log_likelihood_bits'], fit['curve'][0]['train_log_likelihood_bits'])

    def test_pre_encoded_batches_match_rebuilding_from_rows(self):
        job, prepared, record, *_ = synthetic_job()
        train_rows = prepared['train']
        timing_encoder = Encoder(train_rows, VOCAB, ['turn_category', 'gap_s', 'overlap_elapsed_s'])
        offsets = {r['coda_id']: o for r, o in zip(train_rows, job['data']['train']['offset'])}

        def rebuilt(idx):
            subset = [train_rows[i] for i in idx.tolist()]
            data = encode_split(subset, VOCAB, FIELDS, 10, timing_encoder, np.array([offsets[r['coda_id']] for r in subset]))
            return to_tensors(data, 'cpu')

        for dropout in (0.0, 0.4):
            a = fit_residual(job['data']['train'], job['data']['validation'], job['blocks'], 4, dropout, CONFIG, 42,
                             timing_dim=job['timing_dim'])
            b = fit_residual(job['data']['train'], job['data']['validation'], job['blocks'], 4, dropout, CONFIG, 42,
                             timing_dim=job['timing_dim'], batches=rebuilt)
            self.assertEqual(a['curve'], b['curve'])
            self.assertEqual(a['best_epoch'], b['best_epoch'])
            if a['state_dict'] is not None:
                self.assertTrue(all(torch.equal(a['state_dict'][k], b['state_dict'][k]) for k in a['state_dict']))


def synthetic_selection_table(n=60, seed=0):
    """The best candidate is A+B/h32/d0.0; every other candidate is far below it (mean difference 2, SE
    ~0.001) unless the test replaces it with within(): mean difference exactly 0.05, SE exactly 1/sqrt(n)."""
    rng = np.random.RandomState(seed)
    base = rng.normal(size=n)
    candidates = candidate_list(('A', 'B'), [16, 32], [0.0, 0.4])
    per_target = {c['name']: base - 2.0 + rng.normal(scale=.01, size=n) for c in candidates}
    per_target['A+B/h32/d0.0'] = base

    def within():
        z = rng.normal(size=n)
        z = (z - z.mean()) / z.std(ddof=1)
        return base - 0.05 - z

    return candidates, per_target, within


class SelectionTests(unittest.TestCase):
    def test_simplest_within_one_se(self):
        candidates, per_target, within = synthetic_selection_table()
        for name in ('A+B/h16/d0.4', 'A/h32/d0.4', 'A/h16/d0.0', 'A/h16/d0.4'):
            per_target[name] = within()
        selection = select_candidate(candidates, per_target)
        self.assertEqual(selection['best'], 'A+B/h32/d0.0')
        self.assertEqual(selection['chosen'], 'A/h16/d0.4')  # fewer blocks, then smaller hidden, then more dropout
        rows = {r['candidate']: r for r in selection['candidates']}
        d = per_target['A+B/h32/d0.0'] - per_target['A/h16/d0.4']
        self.assertAlmostEqual(rows['A/h16/d0.4']['standard_error_bits'], np.std(d, ddof=1) / math.sqrt(len(d)))
        self.assertTrue(rows['A/h16/d0.4']['within_one_se'])
        self.assertFalse(rows[LR_ONLY]['within_one_se'])
        self.assertAlmostEqual(selection['chosen_gain_over_lr_bits'],
                               np.mean(per_target['A/h16/d0.4']) - np.mean(per_target[LR_ONLY]))

    def test_order_fewer_blocks_before_smaller_hidden_before_more_dropout(self):
        candidates, per_target, within = synthetic_selection_table(seed=1)
        per_target['A+B/h16/d0.4'] = within()
        per_target['B/h32/d0.0'] = within()
        self.assertEqual(select_candidate(candidates, per_target)['chosen'], 'B/h32/d0.0')
        per_target['B/h16/d0.0'] = within()
        self.assertEqual(select_candidate(candidates, per_target)['chosen'], 'B/h16/d0.0')
        per_target['B/h16/d0.4'] = within()
        self.assertEqual(select_candidate(candidates, per_target)['chosen'], 'B/h16/d0.4')
        per_target[LR_ONLY] = within()
        self.assertEqual(select_candidate(candidates, per_target)['chosen'], LR_ONLY)

    def test_equally_simple_candidates_go_to_higher_mean(self):
        candidates, per_target, within = synthetic_selection_table(seed=2)
        per_target['A/h16/d0.4'] = within()
        per_target['B/h16/d0.4'] = per_target['A/h16/d0.4'] + 0.001
        self.assertEqual(select_candidate(candidates, per_target)['chosen'], 'B/h16/d0.4')

    def test_identical_to_lr_is_lr(self):
        """A candidate whose seeds all stopped at epoch 0 carries the offset's own values: LR wins outright."""
        candidates = candidate_list(('A',), [16], [0.0])
        lr = np.random.RandomState(3).normal(size=20)
        selection = select_candidate(candidates, {LR_ONLY: lr, 'A/h16/d0.0': lr})
        self.assertEqual((selection['best'], selection['chosen']), (LR_ONLY, LR_ONLY))

    def test_selection_never_reads_test_rows(self):
        config = dict(CONFIG, patience=2, max_epochs=4, hidden_grid=[4], dropout_grid=[0.0, 0.2])
        seeds = [42, 43]

        def select(test):
            job, *_ = synthetic_job(test=test)
            candidates = candidate_list(job['blocks'], config['hidden_grid'], config['dropout_grid'])
            with FitRunner(0) as runner:
                fits = {(k[2], k[3]): f for k, f in runner.run(condition_tasks(job, config, seeds, 'cpu'))}
            lr_val = per_target_bits(job['lr_probabilities']['validation'], job['data']['validation']['y'])
            selection, per_target = select_condition(candidates, fits, seeds, lr_val)
            return selection, {k: f['curve'] for k, f in fits.items()}

        test = make_rows(10, 3, 'test')
        first = select(test)
        changed = [dict(r, Ct='UNK', A_history=['c'] * 9, B_history=['a'], turn_category='overlap', gap_s=0.0,
                        overlap_elapsed_s=5.0) for r in test]
        second = select(changed)
        self.assertEqual(first, second)

    def test_pool_matches_in_process(self):
        job, *_ = synthetic_job()
        config = dict(CONFIG, patience=2, max_epochs=3)
        tasks = condition_tasks(job, config, [42], 'cpu')
        with FitRunner(0) as runner:
            serial = runner.run(tasks)
        with FitRunner(2) as runner:
            pooled = runner.run(tasks)
        self.assertEqual([k for k, _ in serial], [k for k, _ in pooled])
        self.assertEqual([f['curve'] for _, f in serial], [f['curve'] for _, f in pooled])


class GainTests(unittest.TestCase):
    def test_fold_with_v4_chosen_gives_exactly_zero_gain(self):
        """Every seed equal to v4 must give a gain of exactly 0 for the day, not float noise: pick a
        probability whose log-likelihood, averaged over five identical copies, is not returned exactly."""
        seeds = [42, 43, 44, 45, 46]
        rng = np.random.RandomState(0)
        for _ in range(100000):
            p = float(rng.uniform(0.01, 0.99))
            if math.fsum([log_likelihood_bits(p)] * len(seeds)) / len(seeds) != log_likelihood_bits(p):
                break
        else:
            self.fail('no probability found whose five-copy average is inexact')
        stored = {('v4-day', 'A'): {'a': p, 'b': 1 - p}, ('trained-day', 'A'): {'a': 0.5, 'b': 0.5}}
        scored = []
        for seed in seeds:  # v4 alone chosen on one day; a trained network, different per seed, on the other
            scored.append(dict(coda_id='v4-day', condition='A', seed=seed, recording_date='2020-01-01', target='a',
                               log_likelihood_bits=log_likelihood_bits(p)))
            scored.append(dict(coda_id='trained-day', condition='A', seed=seed, recording_date='2020-01-02', target='a',
                               log_likelihood_bits=log_likelihood_bits(0.5 + 0.01 * (seed - 41))))
        gain = residual_gains(scored, stored, ['A'], dict(bootstrap_replicates=50, bootstrap_seed=42, confidence=0.95))['A']
        self.assertEqual(gain['day_means_bits']['2020-01-01'], 0.0)
        self.assertEqual(gain['zero_days'], 1)
        self.assertEqual(gain['positive_days'], 1)  # only the trained day

    def test_planted_reading(self):
        def row(f, selected, offset, chosen):
            return dict(fraction=f, a_selected=selected, planted_mean_bits_offset=offset, planted_mean_bits_chosen=chosen,
                        planted_train=round(60 * f))
        results = [row(0.0, True, None, None), row(0.05, True, -3.3, -3.6), row(0.2, True, -6.2, -5.1), row(0.4, True, -6.7, -6.1)]
        sentence = planted_reading(results)
        self.assertIn('f >= 0.2', sentence)
        self.assertIn('about 12 planted training targets', sentence)
        self.assertIn('reports/rq1_lstm_residual_sensitivity.md', sentence)
        self.assertIsNone(planted_reading([row(0.0, False, None, None)] + results[1:]))  # f = 0 does not select A
        self.assertIsNone(planted_reading(results[:1] + [row(0.05, True, -3.3, -3.0)] + results[2:]))  # improves at once


class CommittedV4Tests(unittest.TestCase):
    def test_offsets_reproduce_v4_stored_test_probabilities(self):
        manifest = read_json(ROOT / 'data/processed/rq1_folds/manifest.json')
        rows = read_jsonl(ROOT / manifest['configuration']['input'])
        models = load_v4_models(ROOT / 'data/results/rq1_logistic_v4/models.jsonl')
        stored = stored_v4_probabilities(ROOT / 'data/results/rq1_logistic_v4/predictions.jsonl')
        checked = 0
        for prepared in prepare_folds(rows, manifest):
            vocab, test = prepared['policy']['vocabulary'], prepared['test']
            for condition in manifest['input_conditions']:
                if condition == 'none':
                    continue
                record = models[prepared['fold']['fold_id'], condition]
                p = softmax_rows(v4_logits(record, prepared['train'], test, vocab))
                expected = np.array([[stored[r['coda_id'], condition][t] for t in vocab] for r in test])
                self.assertLessEqual(float(np.max(np.abs(p - expected))), 1e-9, (prepared['fold']['fold_id'], condition))
                checked += len(test)
        self.assertEqual(checked, 7 * manifest['comparison_targets'])


if __name__ == '__main__':
    unittest.main()
