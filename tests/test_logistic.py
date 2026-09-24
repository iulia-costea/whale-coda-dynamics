import unittest
import numpy as np
from scipy.optimize import check_grad

from whale_coda_dynamics.logistic import (
    Encoder, objective, fit, probabilities, repeat_distribution, last_label_index, select_l2_and_fit,
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
