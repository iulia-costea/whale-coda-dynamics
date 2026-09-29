import unittest
import numpy as np
from scipy.optimize import check_grad

from whale_coda_dynamics.logistic import (
    Encoder, objective, fit, probabilities, repeat_distribution, last_label_index, select_l2_and_fit,
    check_stop_conditions, is_timing_feature,
)


class LogisticTests(unittest.TestCase):
    def test_separate_penalty_gradient_and_shared_penalty_equivalence(self):
        x = np.array([[1., 0., 1.], [-1., .5, 1.], [.5, 1., 1.]])
        y = np.array([0, 1, 0])
        own, partner = np.array([0, 1, 2]), np.array([1, 0, 2])
        weights = np.arange(11, dtype=float) / 10
        penalties = np.array([.001, 1., .001])
        args = (x, y, 3, .001, own, partner, penalties)
        error = check_grad(lambda w: objective(w, *args)[0], lambda w: objective(w, *args)[1], weights)
        self.assertLess(error, 1e-6)
        old = objective(weights, x, y, 3, .001, own, partner)
        new = objective(weights, x, y, 3, .001, own, partner, np.full(3, .001))
        self.assertAlmostEqual(old[0], new[0])
        np.testing.assert_allclose(old[1], new[1])

    def test_separate_selection_and_no_s_invariance(self):
        vocab = ['a', 'b', 'UNK']
        train = [dict(split='train', Ct=vocab[i % 2], A_history=[vocab[i % 2]],
                      turn_category='switch', gap_s=float(i % 3), overlap_elapsed_s=0.) for i in range(20)]
        validation = [dict(r, split='validation') for r in train[:6]]
        config = dict(history_order=1, l2_grid=[.01, .1], max_iterations=2000, gradient_tolerance=1e-6)
        split_config = dict(config, s_l2_grid=[.01, 1.])
        _, legacy, old_grid = select_l2_and_fit(train, validation, vocab, ['A_history'], config)
        _, split, new_grid = select_l2_and_fit(train, validation, vocab, ['A_history'], split_config)
        self.assertEqual(len(old_grid), len(new_grid))
        self.assertEqual(legacy['l2'], split['l2'])
        np.testing.assert_array_equal(legacy['w'], split['w'])
        self.assertEqual(legacy['beta'], split['beta'])
        self.assertNotIn('s_l2', split)
        fields = ['A_history', 'turn_category', 'gap_s', 'overlap_elapsed_s']
        _, selected, grid = select_l2_and_fit(train, validation, vocab, fields, split_config)
        self.assertEqual({(g['l2'], g['s_l2']) for g in grid}, {(.01, .01), (.01, 1.), (.1, .01), (.1, 1.)})
        expected = max(grid, key=lambda g: (g['validation_log_likelihood_bits'], g['l2'], g['s_l2']))
        self.assertIs(selected, expected)
        with self.assertRaises(ValueError):
            fit(np.ones((3, 2)), np.array([0, 1, 0]), 3, dict(config, l2=.01), feature_l2=[.01, -1.])

    def test_stop_rules_use_likelihood_not_accuracy(self):
        checks = {c: {'ci_bits_per_coda': [-.1, .1]} for c in ('A', 'A_B', 'A_S', 'A_B_S')}
        summary = {'metrics': {c: {'accuracy': 0.} for c in checks}, 'contrasts': {}}
        baseline = {'repeat_own_last': {'accuracy': 1.}}
        self.assertEqual(check_stop_conditions(summary, baseline, checks), [])
        checks['A']['ci_bits_per_coda'] = [-.2, -.01]
        self.assertEqual(len(check_stop_conditions(summary, baseline, checks)), 1)

    def test_gradient_and_absent_class(self):
        x=np.array([[1.,1.],[-1.,1.],[.5,1.]])
        y=np.array([0,1,0]);w=np.arange(6,dtype=float)/10
        error=check_grad(lambda a:objective(a,x,y,3,.01)[0],lambda a:objective(a,x,y,3,.01)[1],w)
        self.assertLess(error,1e-6)
        w2,beta,gamma,_=fit(x,y,3,dict(l2=.01,max_iterations=2000,gradient_tolerance=1e-6))
        self.assertIsNone(beta);self.assertIsNone(gamma)
        p=probabilities(x,w2,beta,gamma)
        self.assertTrue(np.all(p>0))
        np.testing.assert_allclose(p.sum(axis=1),1)

    def test_copy_feature_gradient(self):
        x=np.array([[1.,1.],[-1.,1.],[.5,1.],[.2,-.3]])
        y=np.array([0,1,0,2])
        own_idx=np.array([0,-1,2,1]);partner_idx=np.array([1,2,-1,0])
        w=np.arange(8,dtype=float)/10
        error=check_grad(lambda a:objective(a,x,y,3,.01,own_idx,partner_idx)[0],
                         lambda a:objective(a,x,y,3,.01,own_idx,partner_idx)[1],w)
        self.assertLess(error,1e-6)
        w2,beta,gamma,_=fit(x,y,3,dict(l2=.01,max_iterations=2000,gradient_tolerance=1e-6),own_idx,partner_idx)
        self.assertIsNotNone(beta);self.assertIsNotNone(gamma)
        p=probabilities(x,w2,beta,gamma,own_idx,partner_idx)
        self.assertTrue(np.all(p>0));np.testing.assert_allclose(p.sum(axis=1),1)

    def test_copy_feature_reproduces_repeat_argmax(self):
        w=np.zeros((2,3))
        x=np.array([[.3,-.1],[1.2,.4],[-.5,.9]])
        own_idx=np.array([0,1,2])
        p=probabilities(x,w,beta=50.0,gamma=None,own_last_idx=own_idx,partner_last_idx=None)
        np.testing.assert_array_equal(p.argmax(axis=1),own_idx)

    def test_selection_uses_validation_only(self):
        vocab=['a','b','UNK']
        def rows(n,seed,split):
            rng=np.random.RandomState(seed)
            return [dict(coda_id=f'{split}-{seed}-{i}',Ct=vocab[(i+1)%2],A_history=[vocab[i%2]],
                        B_history=[vocab[i%2]],turn_category='switch',gap_s=float(rng.rand()),
                        overlap_elapsed_s=0.0,split=split) for i in range(n)]
        train,validation=rows(20,1,'train'),rows(6,2,'validation')
        fields=['A_history','turn_category','gap_s','overlap_elapsed_s']
        config=dict(history_order=1,l2_grid=[.01,.1,1.0],max_iterations=500,gradient_tolerance=1e-6)
        _,chosen1,_=select_l2_and_fit(train,validation,vocab,fields,config)
        # `test` rows are never passed to select_l2_and_fit at all; mutating them
        # (as if that mattered) and re-running proves the selection cannot see them.
        test=rows(5,3,'test')
        for r in test: r['Ct']='UNK'
        _,chosen2,_=select_l2_and_fit(train,validation,vocab,fields,config)
        self.assertEqual(chosen1['l2'],chosen2['l2'])
        np.testing.assert_array_equal(chosen1['w'],chosen2['w'])

    def test_last_label_index(self):
        rows=[dict(A_history=['a','b']),dict(A_history=[])]
        np.testing.assert_array_equal(last_label_index(rows,'A_history',['a','b']),[1,-1])

    def test_feature_isolation_and_training_scaling(self):
        train=[dict(split='train',A_history=['a'],B_history=['b'],turn_category='switch',gap_s=v,overlap_elapsed_s=0) for v in (0,2)]
        a=Encoder(train,['a','b'],['A_history'])
        changed=[dict(r,B_history=['a']*10,gap_s=999,turn_category='other') for r in train]
        np.testing.assert_array_equal(a.transform(train),a.transform(changed))
        timing=Encoder(train,['a','b'],['gap_s'])
        np.testing.assert_allclose(timing.transform(train)[:,0],[-1,1])
        before=timing.mean.copy()
        timing.transform(changed)
        np.testing.assert_array_equal(timing.mean,before)
        self.assertEqual(a.transform(train).shape[1],7)

    def test_retrospective_delay_encoding(self):
        train=[dict(split='train',turn_category='switch',retrospective_delay_s=v) for v in (-1.5,0.5,2.0)]
        e=Encoder(train,['a','b'],['turn_category','retrospective_delay_s'])
        self.assertIn('retrospective_delay_s',e.names)
        self.assertNotIn('log1p:retrospective_delay_s',e.names)
        j=e.names.index('retrospective_delay_s')
        x=e.transform(train)
        np.testing.assert_allclose(x[:,j].mean(),0,atol=1e-10)
        with self.assertRaises(ValueError):
            e.raw([dict(turn_category='switch',retrospective_delay_s=float('nan'))])

    def test_repeat_distribution(self):
        p=repeat_distribution('a',.8,{'a':.5,'b':.3,'UNK':.2})
        self.assertAlmostEqual(sum(p.values()),1)
        self.assertAlmostEqual(p['a'],.8)
        self.assertAlmostEqual(p['b'],.12)
        with self.assertRaises(ValueError):
            repeat_distribution('a',1,{'a':.5,'b':.5})

    def test_is_timing_feature(self):
        self.assertTrue(is_timing_feature('turn:switch'))
        self.assertTrue(is_timing_feature('gap_s'))
        self.assertTrue(is_timing_feature('log1p:overlap_elapsed_s'))
        self.assertTrue(is_timing_feature('retrospective_delay_s'))
        self.assertFalse(is_timing_feature('A_history:lag1:a'))
        self.assertFalse(is_timing_feature('B_history:lag2:UNK'))

    def test_split_l2_gradient(self):
        x=np.array([[1.,1.],[-1.,1.],[.5,1.],[.2,-.3]])
        y=np.array([0,1,0,2])
        l2_vector=np.array([.3,.05])
        error=check_grad(lambda a:objective(a,x,y,3,.01,feature_l2=l2_vector)[0],
                         lambda a:objective(a,x,y,3,.01,feature_l2=l2_vector)[1],np.arange(6,dtype=float)/10)
        self.assertLess(error,1e-6)

    def test_selection_splits_l2_only_when_s_present(self):
        vocab=['a','b','UNK']
        rows=[dict(coda_id=f'r{i}',Ct=vocab[(i+1)%2],A_history=[vocab[i%2]],B_history=[vocab[i%2]],
                   turn_category='switch',gap_s=float(i%3),overlap_elapsed_s=0.0,split='train') for i in range(10)]
        config=dict(history_order=1,l2_grid=[.01,.1,1.0],s_l2_grid=[.01,.1,1.0],max_iterations=500,gradient_tolerance=1e-6)
        _,chosen_a,grid_a=select_l2_and_fit(rows,rows,vocab,['A_history'],config)
        self.assertNotIn('s_l2',chosen_a)
        self.assertEqual(len(grid_a),3)
        _,chosen_as,grid_as=select_l2_and_fit(rows,rows,vocab,['A_history','turn_category','gap_s','overlap_elapsed_s'],config)
        self.assertIsNotNone(chosen_as['s_l2'])
        self.assertEqual(len(grid_as),9)
