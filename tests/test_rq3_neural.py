"""Checks of the pooled RQ3 LSTM (PLAN [R3-8]): aggregation through Vlad's functions, plans, leakage
checks, tag pairing, symbol counts and refit determinism. Uses the committed RQ3 inputs, read only."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config' / 'rq3_neural.json'
HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_TORCH = HAS_NUMPY and importlib.util.find_spec('torch') is not None
HAS_SCIPY = HAS_NUMPY and importlib.util.find_spec('scipy') is not None

_CTX = None


def context():
    global _CTX
    if _CTX is None:
        from whale_coda_dynamics.rq3_neural.data import Context
        _CTX = Context(CONFIG)
    return _CTX


class VladAggregationTests(unittest.TestCase):
    """The wrappers reproduce Vlad's own-clan n-gram numbers and his matrix (standard library only)."""

    def test_self_scores_reproduce_his_calibration(self):
        from whale_coda_dynamics.rq3.experiments.data import CALIBRATION_COLUMNS, SELF_SCORE_COLUMNS, read_csv
        from whale_coda_dynamics.rq3_neural.contrasts import bits_per_coda, bits_per_symbol
        scores = read_csv(ROOT / 'data/results/rq3/calibration/self_scores.csv', SELF_SCORE_COLUMNS)
        selected = [row for row in read_csv(ROOT / 'data/results/rq3/calibration/self_calibration.csv',
                                            CALIBRATION_COLUMNS) if row['selected'] == 'true']
        self.assertEqual(len(selected), 9)
        for row in selected:
            clan_scores = {s['rq3_coda_id']: (float(s['bits_per_coda']), int(s['token_count']))
                           for s in scores if s['clan'] == row['clan']}
            self.assertEqual(len(clan_scores), int(row['validation_coda_count']))
            self.assertAlmostEqual(bits_per_symbol(clan_scores), float(row['mean_bits_per_symbol']), places=12)
            self.assertAlmostEqual(bits_per_coda(clan_scores), float(row['mean_bits_per_coda']), places=12)

    def test_self_scores_reproduce_the_report_table(self):
        """The rq3_self_calibration.md numbers (rounded to 6 decimals there)."""
        from whale_coda_dynamics.rq3.experiments.data import SELF_SCORE_COLUMNS, read_csv
        from whale_coda_dynamics.rq3_neural.contrasts import bits_per_symbol
        expected = {'EC1': -1.658672, 'EC2': -1.630095, 'FP': -2.871254, 'PALI': -2.004227, 'PO': -1.848356,
                    'REG': -1.443874, 'RI': -2.221911, 'SH': -2.355519, 'SI': -1.956451}
        scores = read_csv(ROOT / 'data/results/rq3/calibration/self_scores.csv', SELF_SCORE_COLUMNS)
        for clan, value in expected.items():
            clan_scores = {s['rq3_coda_id']: (float(s['bits_per_coda']), int(s['token_count']))
                           for s in scores if s['clan'] == clan}
            self.assertAlmostEqual(bits_per_symbol(clan_scores), value, places=6)

    def test_paired_contrast_sign_and_value(self):
        from whale_coda_dynamics.rq3_neural.contrasts import paired_contrast
        first = {f'c{i}': (-2.0 * (i % 3 + 3), i % 3 + 3) for i in range(30)}
        second = {k: (ll - 0.5 * n, n) for k, (ll, n) in first.items()}
        groups = {k: f'g{i % 5}' for i, k in enumerate(first)}
        c = paired_contrast(first, second, groups, 200, 42, ('test',))
        self.assertAlmostEqual(c['contrast_bits_per_symbol'], 0.5, places=12)  # first is better: positive
        self.assertLessEqual(c['ci95_lower_bits_per_symbol'], 0.5 + 1e-12)
        self.assertGreaterEqual(c['ci95_upper_bits_per_symbol'], 0.5 - 1e-12)
        self.assertEqual(c['n_groups'], 5)
        with self.assertRaises(ValueError):
            paired_contrast(first, dict(list(second.items())[:-1]), groups, 10, 42, ('test',))
        with self.assertRaises(ValueError):
            paired_contrast(first, {k: (ll, n + 1) for k, (ll, n) in second.items()}, groups, 10, 42, ('test',))

    def test_grouped_contrast_matches_per_coda(self):
        from whale_coda_dynamics.rq3_neural.contrasts import grouped_contrast, paired_contrast
        first = {f'c{i}': (-1.0 - 0.37 * i, 3 + i % 4) for i in range(40)}
        second = {k: (ll - 0.1 * (i % 7), n) for i, (k, (ll, n)) in enumerate(first.items())}
        groups = {k: f'g{i % 6}' for i, k in enumerate(first)}
        per_coda = paired_contrast(first, second, groups, 100, 7, ('x',))
        sums = {}
        for name, scores in (('first', first), ('second', second)):
            for k, (ll, n) in scores.items():
                s = sums.setdefault(name, {}).setdefault(groups[k], [0.0, 0])
                s[0] += ll
                s[1] += n
        grouped = grouped_contrast({g: tuple(v) for g, v in sums['first'].items()},
                                   {g: tuple(v) for g, v in sums['second'].items()}, 100, 7, ('x',))
        self.assertAlmostEqual(grouped['contrast_bits_per_symbol'], per_coda['contrast_bits_per_symbol'], places=12)
        self.assertAlmostEqual(grouped['ci95_lower_bits_per_symbol'], per_coda['ci95_lower_bits_per_symbol'],
                               places=12)


@unittest.skipUnless(HAS_SCIPY, 'needs numpy and scipy')
class NgramMatrixTests(unittest.TestCase):
    def test_matrix_rebuilt_from_his_scores_matches_his_summary(self):
        from whale_coda_dynamics.rq3_neural.report import ngram_transfer_losses
        losses = ngram_transfer_losses(ROOT / 'data/results/rq3/transfer_matrix/scores.csv',
                                       ROOT / 'data/results/rq3/transfer_matrix/summary.csv')
        self.assertEqual(len(losses), 72)


@unittest.skipUnless(HAS_NUMPY, 'needs numpy')
class PlanAndCheckTests(unittest.TestCase):
    def setUp(self):
        self.ctx = context()

    def test_gate0_counts_match_the_fold_file(self):
        from whale_coda_dynamics.rq3_neural.data import gate0_rows
        rows = gate0_rows(self.ctx)
        self.assertTrue(all(row['match'] for row in rows))
        self.assertEqual(sum(row['codas_file'] for row in rows if row['fold'] == 'all'), 31479)
        original = self.ctx.fold_rows
        try:
            self.ctx.fold_rows = [dict(r) for r in original]
            self.ctx.fold_rows[0]['n_codas'] = str(int(self.ctx.fold_rows[0]['n_codas']) + 1)
            with self.assertRaises(ValueError):
                gate0_rows(self.ctx)
        finally:
            self.ctx.fold_rows = original

    def test_n_symbols_equal_vlad_token_count(self):
        import numpy as np
        from whale_coda_dynamics.rq3_neural.data import check_n_symbols
        ctx = self.ctx
        index = np.arange(len(ctx.ids))
        check_n_symbols('all', index, ctx.n_symbols, ctx)
        for coda in ctx.codas[:50]:
            self.assertEqual(len(coda['tokens']) - 1, ctx.vlad_token_count[coda['rq3_coda_id']])
        wrong = ctx.n_symbols.copy()
        wrong[7] += 1
        with self.assertRaises(ValueError):
            check_n_symbols('tampered', index, wrong, ctx)

    def test_tagged_and_untagged_inputs_differ_only_in_the_start_symbol(self):
        import numpy as np
        from whale_coda_dynamics.rq3_neural.data import check_tag_inputs
        enc = self.ctx.encoded
        check_tag_inputs(enc)
        tampered = dict(enc, x_tagged=enc['x_tagged'].copy())
        tampered['x_tagged'][3, 2] = (tampered['x_tagged'][3, 2] + 1) % 20
        with self.assertRaises(ValueError):
            check_tag_inputs(tampered)
        self.assertTrue(np.array_equal(enc['x_untagged'][:, 1:], enc['x_tagged'][:, 1:]))

    def test_own_clan_plans(self):
        import numpy as np
        ctx = self.ctx
        from whale_coda_dynamics.rq3_neural.data import own_clan_plan
        for clan in ctx.clans:
            held_all = []
            for k in range(ctx.n_folds[clan]):
                plan = own_clan_plan(ctx, clan, k)
                self.assertEqual(plan['inner_folds'][clan], (k + 1) % ctx.n_folds[clan])
                self.assertEqual(np.intersect1d(plan['held'], np.concatenate([plan['select_train'], plan['val']])).size, 0)
                self.assertEqual(np.intersect1d(plan['held'], plan['refit_train']).size, 0)
                self.assertEqual(np.intersect1d(plan['select_train'], plan['val']).size, 0)
                self.assertEqual(len(plan['refit_train']) + len(plan['held']), len(ctx.clan_indices(clan)))
                held_all.append(plan['held'])
            self.assertEqual(sorted(np.concatenate(held_all).tolist()), sorted(ctx.clan_indices(clan).tolist()))

    def test_pooled_plans(self):
        import numpy as np
        from whale_coda_dynamics.rq3_neural.data import pooled_plan, pooled_runs
        ctx = self.ctx
        self.assertEqual(pooled_runs(ctx), [0, 1, 2, 3, 4, 5])
        scored = []
        for run in pooled_runs(ctx):
            plan = pooled_plan(ctx, run)
            seen = np.concatenate([plan['select_train'], plan['val']])
            self.assertEqual(np.intersect1d(plan['held'], seen).size, 0)
            self.assertEqual(np.intersect1d(plan['held'], plan['refit_train']).size, 0)
            for clan in ctx.clans:
                self.assertEqual(plan['inner_folds'][clan], (run + 1) % ctx.n_folds[clan])
            scored.append(plan['held'])
        run5 = pooled_plan(ctx, 5)
        self.assertEqual({ctx.clan_of[i] for i in run5['held']}, {'EC2'})
        self.assertEqual(set(ctx.fold_of[run5['held']].tolist()), {5})
        self.assertEqual(run5['inner_folds']['EC1'], 1)
        self.assertEqual(len(run5['refit_train']), len(ctx.ids) - len(run5['held']))
        self.assertEqual(sorted(np.concatenate(scored).tolist()), list(range(len(ctx.ids))))

    def test_leave_one_out_plans_contain_no_coda_of_the_left_out_clan(self):
        import numpy as np
        from whale_coda_dynamics.rq3_neural.data import check_excludes_clan, loo_plan
        ctx = self.ctx
        for clan in ctx.clans:
            plan = loo_plan(ctx, clan)
            code = ctx.clans.index(clan)
            for key in ('select_train', 'val', 'refit_train'):
                self.assertFalse(np.any(ctx.encoded['clan_index'][plan[key]] == code))
            self.assertEqual(len(plan['refit_train']) + len(plan['held']), len(ctx.ids))
            self.assertEqual(set(ctx.fold_of[plan['val']].tolist()), {0})
            with self.assertRaises(ValueError):
                check_excludes_clan('tampered', ctx, clan, np.concatenate([plan['refit_train'], plan['held'][:1]]))

    def test_task_checks_reject_leakage(self):
        import numpy as np
        from whale_coda_dynamics.rq3_neural.data import own_clan_plan
        from whale_coda_dynamics.rq3_neural.runner import candidate_grid, make_task
        ctx = self.ctx
        plan = own_clan_plan(ctx, 'SI', 0)
        candidate = candidate_grid(ctx.config)[0]
        common = dict(tagged=False, alpha=1.0, candidate=candidate, seed=0, device='cpu')
        make_task(ctx, ('ok',), train=plan['refit_train'], val=None, guard=plan['held'], score=plan['held'],
                  epochs=3, **common)
        with self.assertRaises(ValueError):  # scores a training coda
            make_task(ctx, ('leak',), train=plan['refit_train'], val=None, guard=plan['held'],
                      score=np.concatenate([plan['held'], plan['refit_train'][:1]]), epochs=3, **common)
        with self.assertRaises(ValueError):  # validation codas from the held-out fold
            make_task(ctx, ('leak',), train=plan['select_train'], val=plan['held'], guard=plan['held'], epochs=None,
                      **common)
        with self.assertRaises(ValueError):  # training and validation overlap
            make_task(ctx, ('leak',), train=plan['select_train'], val=plan['select_train'][:5], guard=plan['held'],
                      epochs=None, **common)
        with self.assertRaises(ValueError):  # inherited training (a fine-tuned model) contains a scored coda
            make_task(ctx, ('leak',), train=plan['select_train'], val=plan['val'], guard=plan['held'],
                      score=plan['held'], inherited=(plan['held'][:2],), epochs=None, **common)


@unittest.skipUnless(HAS_TORCH, 'needs numpy and torch')
class FitTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from whale_coda_dynamics.rq3_neural.data import own_clan_plan, pooled_plan
        from whale_coda_dynamics.rq3_neural.runner import candidate_grid
        self.ctx = context()
        self.np = np
        rng = np.random.default_rng(0)
        plan = pooled_plan(self.ctx, 0)
        self.train = np.sort(rng.choice(plan['select_train'], 400, replace=False))
        self.val = np.sort(rng.choice(plan['val'], 150, replace=False))
        self.held = plan['held']
        self.score = np.sort(rng.choice(plan['held'], 120, replace=False))
        self.candidate = dict(candidate_grid(self.ctx.config)[0], hidden=8)
        self.own = own_clan_plan(self.ctx, 'SI', 0)

    def task(self, tagged, seed=0, epochs=None, alpha=0.5, max_epochs=3):
        from whale_coda_dynamics.rq3_neural.runner import make_task
        val = self.val if epochs is None else None
        return make_task(self.ctx, ('test', tagged, seed), train=self.train, val=val, guard=self.held,
                         score=self.score, tagged=tagged, alpha=alpha, candidate=self.candidate, seed=seed,
                         epochs=epochs, device='cpu', max_epochs=max_epochs)

    def test_sampler_follows_n_alpha(self):
        import torch
        from whale_coda_dynamics.rq3_neural.model import Draws, sampling_probabilities
        q = sampling_probabilities([100, 400], 0.5)
        self.assertAlmostEqual(q[0], 10 / 30)
        self.assertAlmostEqual(sum(sampling_probabilities([3, 5, 9], 1.0)), 1.0)
        draws = Draws(self.train, self.ctx.encoded['clan_index'], 0.3, seed=1)
        counts = torch.bincount(torch.from_numpy(self.ctx.encoded['clan_index'])[draws.epoch()],
                                minlength=len(self.ctx.clans)).numpy()
        present = counts[counts > 0] / counts.sum()
        self.assertEqual(len(present), len(draws.q))
        self.assertLess(abs(present - draws.q).max(), 0.08)
        a, b = Draws(self.train, self.ctx.encoded['clan_index'], 0.3, seed=1), Draws(
            self.train, self.ctx.encoded['clan_index'], 0.3, seed=1)
        self.assertTrue(torch.equal(a.epoch(), b.epoch()))
        self.assertTrue(set(a.epoch().tolist()) <= set(self.train.tolist()))

    def test_tagged_and_untagged_fits_share_draws_seed_and_initial_weights(self):
        from whale_coda_dynamics.rq3_neural.model import FitPool
        from whale_coda_dynamics.rq3_neural.runner import check_same_draws
        with FitPool(self.ctx.encoded, 0) as pool:
            ta, tb, tc = self.task(True), self.task(False), self.task(False, seed=1)
            a, b, c = pool.run([ta, tb, tc])
        self.assertGreaterEqual(check_same_draws('pair', ta, a, tb, b, {'tagged'}), 1)
        self.assertEqual(a['init_sha256'], b['init_sha256'])
        self.assertNotEqual(a['state_sha256'], b['state_sha256'])  # the start symbol matters
        with self.assertRaises(ValueError):  # a different seed is not a valid pair
            check_same_draws('pair', ta, a, tc, c, {'tagged'})
        with self.assertRaises(ValueError):  # nor is a different alpha
            check_same_draws('pair', ta, a, dict(tb, alpha=1.0), b, {'tagged'})

    def test_refit_with_the_same_seed_gives_identical_scores(self):
        from whale_coda_dynamics.rq3_neural.model import FitPool
        from whale_coda_dynamics.rq3_neural.runner import check_refit_reproducible
        task = self.task(False, epochs=2)
        with FitPool(self.ctx.encoded, 0) as pool:
            result = pool.run([task])[0]
            check_refit_reproducible(pool, task, result, 'test')
            tampered = dict(result, scores=dict(result['scores'], ll_bits=result['scores']['ll_bits'] + 1e-9))
            with self.assertRaises(ValueError):
                check_refit_reproducible(pool, task, tampered, 'test')

    def test_scores_are_log2_probabilities_over_every_predicted_symbol(self):
        import numpy as np
        import torch
        from whale_coda_dynamics.rq3_neural.model import CodaLSTM, FitPool, score_codas
        from whale_coda_dynamics.rq3_neural.runner import check_scored
        with FitPool(self.ctx.encoded, 0) as pool:
            task = self.task(True, epochs=1)
            result = pool.run([task])[0]
            check_scored(self.ctx, 'test', task, result)
            self.assertTrue(np.all(result['scores']['ll_bits'] < 0))
            self.assertTrue(np.all(result['scores']['ll_eos_bits'] <= 0))
            self.assertTrue(np.all(result['scores']['ll_bits'] <= result['scores']['ll_eos_bits']))
            # An untrained model's per-coda bits equal the sum of its per-position log2 probabilities.
            torch.manual_seed(0)
            model = CodaLSTM(self.ctx.encoded['n_inputs'], 32, 8, 0.0)
            ll, _, counts = score_codas(model, self.score[:5], False, 'cpu', 2)
            x = torch.from_numpy(self.ctx.encoded['x_untagged'][self.score[:5]])
            y = torch.from_numpy(self.ctx.encoded['y'][self.score[:5]])
            with torch.no_grad():
                logp = torch.log_softmax(model(x), dim=-1) / np.log(2)
            for i in range(5):
                n = int(counts[i])
                self.assertEqual(n, self.ctx.n_symbols[self.score[i]])
                expected = float(sum(logp[i, t, y[i, t]] for t in range(n)))
                self.assertAlmostEqual(ll[i], expected, places=4)

    def test_fine_tuning_starts_from_the_checkpoint(self):
        import tempfile
        import torch
        from whale_coda_dynamics.rq3_neural.model import FitPool
        from whale_coda_dynamics.rq3_neural.runner import make_task
        with tempfile.TemporaryDirectory() as tmp, FitPool(self.ctx.encoded, 0) as pool:
            base = pool.run([dict(self.task(False, epochs=1), save_checkpoint=f'{tmp}/m.pt')])[0]
            state = torch.load(f'{tmp}/m.pt', weights_only=True)
            common = dict(train=self.own['select_train'], val=self.own['val'], guard=self.own['held'],
                          score=self.own['held'], tagged=False, alpha=1.0, candidate=self.candidate, seed=0,
                          epochs=None, device='cpu', max_epochs=0)
            tuned = pool.run([make_task(self.ctx, ('ft',), init_checkpoint=f'{tmp}/m.pt',
                                        init_checkpoint_sha256=base['state_sha256'], **common)])[0]
            self.assertEqual(tuned['best_epoch'], 0)
            self.assertEqual(tuned['state_sha256'], base['state_sha256'])
            self.assertEqual(len(state), len(torch.load(f'{tmp}/m.pt', weights_only=True)))
            with self.assertRaises(ValueError):
                pool.run([make_task(self.ctx, ('ft',), init_checkpoint=f'{tmp}/m.pt',
                                    init_checkpoint_sha256='0' * 64, **common)])


if __name__ == '__main__':
    unittest.main()
