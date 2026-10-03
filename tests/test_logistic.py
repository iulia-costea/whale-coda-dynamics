import unittest
import numpy as np
from scipy.optimize import check_grad

from whale_coda_dynamics.logistic import (
    Encoder, objective, fit, probabilities, repeat_distribution, last_label_index, select_l2_and_fit,
    is_timing_feature,
)


class LogisticTests(unittest.TestCase):
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
        error=check_grad(lambda a:objective(a,x,y,3,.01,l2_vector=l2_vector)[0],
                         lambda a:objective(a,x,y,3,.01,l2_vector=l2_vector)[1],np.arange(6,dtype=float)/10)
        self.assertLess(error,1e-6)

    def test_selection_splits_l2_only_when_s_present(self):
        vocab=['a','b','UNK']
        rows=[dict(coda_id=f'r{i}',Ct=vocab[(i+1)%2],A_history=[vocab[i%2]],B_history=[vocab[i%2]],
                   turn_category='switch',gap_s=float(i%3),overlap_elapsed_s=0.0,split='train') for i in range(10)]
        config=dict(history_order=1,l2_grid=[.01,.1,1.0],max_iterations=500,gradient_tolerance=1e-6)
        _,chosen_a,grid_a=select_l2_and_fit(rows,rows,vocab,['A_history'],config)
        self.assertIsNone(chosen_a['l2_s'])
        self.assertEqual(len(grid_a),3)
        _,chosen_as,grid_as=select_l2_and_fit(rows,rows,vocab,['A_history','turn_category','gap_s','overlap_elapsed_s'],config)
        self.assertIsNotNone(chosen_as['l2_s'])
        self.assertEqual(len(grid_as),9)

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

    def test_per_field_history_order(self):
        """DEC-25: a field listed in history_orders gets exactly its own number of lags; the others keep order."""
        vocab=['a','b','UNK']
        train=[dict(split='train',A_history=['a','b','b','a','UNK','a','b','a','b','a'][:n],B_history=['b','a','a','b'][:n%5],
                    turn_category=['switch','overlap'][n%2],gap_s=float(n),overlap_elapsed_s=0.0) for n in range(1,11)]
        fields=['A_history','B_history','turn_category','gap_s','overlap_elapsed_s']
        default=Encoder(train,vocab,fields,3)
        for orders in (None,{},{'A_history':3},{'B_history':3}):  # unset or equal to the default: unchanged
            e=Encoder(train,vocab,fields,3,orders)
            self.assertEqual(e.names,default.names)
            np.testing.assert_array_equal(e.mean,default.mean);np.testing.assert_array_equal(e.scale,default.scale)
            np.testing.assert_array_equal(e.transform(train),default.transform(train))
        for k in (1,2,5,9):
            e=Encoder(train,vocab,fields,3,{'A_history':k})
            for label in vocab:
                self.assertEqual([n for n in e.names if n.startswith('A_history:') and n.endswith(':'+label)],
                                 [f'A_history:lag{lag}:{label}' for lag in range(1,k+1)])
                self.assertEqual([n for n in e.names if n.startswith('B_history:') and n.endswith(':'+label)],
                                 [f'B_history:lag{lag}:{label}' for lag in range(1,4)])
            self.assertEqual([n for n in e.names if not n.startswith('A_history:')],
                             [n for n in default.names if not n.startswith('A_history:')])
            x=e.raw(train)
            for i,r in enumerate(train):
                for lag in range(1,k+1):
                    columns=[e.names.index(f'A_history:lag{lag}:{t}') for t in vocab]
                    expected=[float(len(r['A_history'])>=lag and r['A_history'][-lag]==t) for t in vocab]
                    self.assertEqual(list(x[i,columns]),expected)
            self.assertEqual(x.shape[1],len(e.names))
        for bad in ({'A_history':0},{'A_history':2.0},{'gap_s':2}):
            with self.assertRaises(ValueError):
                Encoder(train,vocab,fields,3,bad)

    def test_reference_order_through_per_field_path_is_identical(self):
        """DEC-25's k = 3: history_orders={'A_history': 3} selects and fits exactly what the default does."""
        vocab=['a','b','c','UNK']
        def rows(n,seed,split):
            rng=np.random.RandomState(seed)
            return [dict(coda_id=f'{split}-{i}',Ct=vocab[rng.randint(3)],A_history=[vocab[rng.randint(3)] for _ in range(1+rng.randint(6))],
                         B_history=[vocab[rng.randint(3)] for _ in range(1+rng.randint(4))],turn_category=['switch','overlap'][i%2],
                         gap_s=float(rng.rand()),overlap_elapsed_s=0.0,split=split) for i in range(n)]
        train,validation=rows(60,1,'train'),rows(20,2,'validation')
        fields=['A_history','B_history','turn_category','gap_s','overlap_elapsed_s']
        config=dict(history_order=3,l2_grid=[.01,.1],max_iterations=500,gradient_tolerance=1e-6)
        e1,c1,g1=select_l2_and_fit(train,validation,vocab,fields,config)
        e2,c2,g2=select_l2_and_fit(train,validation,vocab,fields,dict(config,history_orders={'A_history':3}))
        self.assertEqual(e1.names,e2.names)
        self.assertEqual((c1['l2'],c1['l2_s'],c1['beta'],c1['gamma']),(c2['l2'],c2['l2_s'],c2['beta'],c2['gamma']))
        np.testing.assert_array_equal(c1['w'],c2['w'])
        self.assertEqual([g['validation_log_likelihood_bits'] for g in g1],[g['validation_log_likelihood_bits'] for g in g2])

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
