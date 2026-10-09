import unittest
from unittest import mock
import numpy as np
from scipy.optimize import check_grad

from whale_coda_dynamics import logistic
from whale_coda_dynamics.logistic import (
    Encoder, objective, fit, probabilities, repeat_distribution, last_label_index, select_l2_and_fit,
    NonConvergenceError, nonconvergence_lines,
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


class ClassFeatureTests(unittest.TestCase):
    """DEC-28: shared class-feature weights. Without any, every path is v4's, bit for bit."""
    x=np.array([[1.,1.],[-1.,1.],[.5,1.],[.2,-.3],[.7,.1]])
    y=np.array([0,1,0,2,1])
    own_idx=np.array([0,-1,2,1,1]);partner_idx=np.array([1,2,-1,0,2])
    features=[np.array([[np.log(3),0,0],[0,0,0],[0,0,np.log(2)],[0,0,0],[0,np.log(5),0]]),
              np.array([[0,1.,0],[1.,0,0],[0,0,0],[0,0,1.],[1.,0,0]]),
              np.log1p(np.array([[2.,0,1],[0,1,0],[3,0,0],[1,1,1],[0,0,4]]))]

    def test_empty_list_is_exactly_no_class_features(self):
        w=np.arange(12,dtype=float)/10-.4
        l2_vector=np.array([.3,.05])
        for args in ((),(self.own_idx,self.partner_idx),(self.own_idx,self.partner_idx,l2_vector)):
            loss,grad=objective(w,self.x,self.y,3,.01,*args)
            loss_empty,grad_empty=objective(w,self.x,self.y,3,.01,*args,class_features=[])
            self.assertEqual(loss,loss_empty)
            np.testing.assert_array_equal(grad,grad_empty)
        config=dict(l2=.01,max_iterations=2000,gradient_tolerance=1e-6)
        w1,b1,g1,c1=fit(self.x,self.y,3,config,self.own_idx,self.partner_idx)
        w2,b2,g2,c2,weights=fit(self.x,self.y,3,config,self.own_idx,self.partner_idx,class_features=[])
        np.testing.assert_array_equal(w1,w2);self.assertEqual((b1,g1,c1),(b2,g2,c2));self.assertEqual(len(weights),0)
        np.testing.assert_array_equal(probabilities(self.x,w1,b1,g1,self.own_idx,self.partner_idx),
                                      probabilities(self.x,w2,b2,g2,self.own_idx,self.partner_idx,weights,[]))

    def test_class_feature_gradient(self):
        """Finite differences for every parameter, the new scalars included, with copy weights and split penalties."""
        rng=np.random.RandomState(3)
        l2_vector=np.array([.3,.05])
        for k in (1,2,3):
            features=self.features[:k]
            w=rng.normal(size=6+2+k)
            f=lambda a:objective(a,self.x,self.y,3,.02,self.own_idx,self.partner_idx,l2_vector,features)  # noqa: E731
            self.assertLess(check_grad(lambda a:f(a)[0],lambda a:f(a)[1],w),1e-6)
            numeric=[(f(w+e)[0]-f(w-e)[0])/2e-6 for e in np.eye(len(w))[-k:]*1e-6]
            np.testing.assert_allclose(f(w)[1][-k:],numeric,rtol=1e-5,atol=1e-7)
        w=rng.normal(size=6+3)
        g=lambda a:objective(a,self.x,self.y,3,.02,class_features=self.features)  # noqa: E731
        self.assertLess(check_grad(lambda a:g(a)[0],lambda a:g(a)[1],w),1e-6)

    def test_class_features_enter_fit_and_probabilities(self):
        config=dict(l2=.01,max_iterations=2000,gradient_tolerance=1e-6)
        w,beta,gamma,_,weights=fit(self.x,self.y,3,config,self.own_idx,self.partner_idx,class_features=self.features)
        self.assertEqual(len(weights),3)
        p=probabilities(self.x,w,beta,gamma,self.own_idx,self.partner_idx,weights,self.features)
        self.assertTrue(np.all(p>0));np.testing.assert_allclose(p.sum(axis=1),1)
        logits=self.x@w
        logits[[0,2,3,4],self.own_idx[[0,2,3,4]]]+=beta
        logits[[0,1,3,4],self.partner_idx[[0,1,3,4]]]+=gamma
        logits+=sum(t*f for t,f in zip(weights,self.features))
        np.testing.assert_allclose(p,np.exp(logits)/np.exp(logits).sum(axis=1,keepdims=True),rtol=1e-12)
        with self.assertRaises(ValueError):
            fit(self.x,self.y,3,config,class_features=[np.zeros((4,3))])

    def test_selection_stores_class_weights_only_when_given(self):
        vocab=['a','b','UNK']
        rows=[dict(coda_id=f'r{i}',Ct=vocab[(i+1)%2],A_history=[vocab[i%2]],split='train') for i in range(10)]
        config=dict(history_order=1,l2_grid=[.01,.1],max_iterations=500,gradient_tolerance=1e-6)
        _,plain,grid=select_l2_and_fit(rows,rows,vocab,['A_history'],config)
        self.assertNotIn('class_weights',plain);self.assertTrue(all('class_weights' not in g for g in grid))
        _,empty,grid_empty=select_l2_and_fit(rows,rows,vocab,['A_history'],config,lambda rs:[])
        np.testing.assert_array_equal(plain['w'],empty['w'])
        self.assertEqual([g['validation_log_likelihood_bits'] for g in grid],[g['validation_log_likelihood_bits'] for g in grid_empty])
        self.assertEqual(len(empty['class_weights']),0)
        ones=lambda rs:[np.array([[float(r['A_history'][-1]==t) for t in vocab] for r in rs])]  # noqa: E731
        _,chosen,grid=select_l2_and_fit(rows,rows,vocab,['A_history'],config,ones)
        self.assertEqual(len(chosen['class_weights']),1)
        self.assertTrue(all(len(g['class_weights'])==1 for g in grid))


class NonConvergedCandidateTests(unittest.TestCase):
    """DEC-22: skip_nonconverged_candidates is opt-in; without it a failed fit stops the run as before."""
    def setUp(self):
        self.vocab=vocab=['a','b','UNK']
        self.rows=[dict(coda_id=f'r{i}',Ct=vocab[(i+1)%2],A_history=[vocab[i%2]],B_history=[vocab[i%2]],
                        turn_category='switch',gap_s=float(i%3),overlap_elapsed_s=0.0,split='train') for i in range(10)]
        self.config=dict(history_order=1,l2_grid=[.01,.1,1.0],max_iterations=500,gradient_tolerance=1e-6)

    def failing_at(self,*failing):
        real=logistic.fit
        def fit_or_fail(x,y,classes,config,*args,**kwargs):
            if config['l2'] in failing:
                raise NonConvergenceError('Optimizer did not converge: forced in test')
            return real(x,y,classes,config,*args,**kwargs)
        return mock.patch.object(logistic,'fit',side_effect=fit_or_fail)

    def test_fit_raises_the_value_error_subclass(self):
        x=np.array([[1.,1.],[-1.,1.],[.5,1.],[.2,-.3]]);y=np.array([0,1,0,2])
        with self.assertRaises(NonConvergenceError):
            fit(x,y,3,dict(l2=1e-4,max_iterations=1,gradient_tolerance=1e-12))
        self.assertTrue(issubclass(NonConvergenceError,ValueError))

    def test_default_stops_on_a_failed_candidate(self):
        with self.failing_at(.1), self.assertRaises(NonConvergenceError):
            select_l2_and_fit(self.rows,self.rows,self.vocab,['A_history'],self.config)

    def test_opt_in_leaves_failed_candidates_out_of_selection(self):
        config=dict(self.config,skip_nonconverged_candidates=True)
        _,expected,_=select_l2_and_fit(self.rows,self.rows,self.vocab,['A_history'],dict(config,l2_grid=[.01,1.0]))
        with self.failing_at(.1):
            _,chosen,grid=select_l2_and_fit(self.rows,self.rows,self.vocab,['A_history'],config)
        self.assertEqual([g['l2'] for g in grid],[.01,.1,1.0])
        self.assertEqual([('nonconverged' in g) for g in grid],[False,True,False])
        self.assertEqual(chosen['l2'],expected['l2'])
        np.testing.assert_array_equal(chosen['w'],expected['w'])
        with self.failing_at(.01,.1,1.0), self.assertRaises(NonConvergenceError):
            select_l2_and_fit(self.rows,self.rows,self.vocab,['A_history'],config)

    def test_report_counts_left_out_candidates_per_fold_and_condition(self):
        ok=dict(l2=1.0,l2_s=None,validation_log_likelihood_bits=-1.0,convergence={})
        bad=dict(l2=.01,l2_s=None,nonconverged='forced')
        selections=[dict(fold_id='test-1',condition='A',grid=[ok,bad,bad]),dict(fold_id='test-2',condition='A',grid=[ok,ok,ok])]
        text='\n'.join(nonconvergence_lines(selections,3,2))
        self.assertIn('2 of the 6 did',text)
        self.assertIn('| test-1 | A | 3 | 2 |',text)
        self.assertNotIn('| test-2 |',text)
        self.assertIn('No candidate failed','\n'.join(nonconvergence_lines(selections[1:],3,1)))


# ---------------------------------------------------------------- DEC-29: continuous history fields

def pre_dec29_feature_names(fields, vocabulary, categories, order, history_orders=None):
    """logistic.feature_names as it was before DEC-29 (commit d3a8c4a), kept verbatim to check that every path without
    a continuous history field is unchanged."""
    history_orders = history_orders or {}
    names = []
    for f in fields:
        if f.endswith('_history'):
            names.extend(f'{f}:lag{lag}:{label}' for lag in range(1, history_orders.get(f, order)+1) for label in vocabulary)
        elif f == 'turn_category':
            names.extend('turn:'+c for c in categories)
        elif f == 'retrospective_delay_s':
            names.append(f)
        else:
            names.extend([f, 'log1p:'+f])
    return names


class PreDec29Encoder:
    """logistic.Encoder as it was before DEC-29 (commit d3a8c4a), verbatim except for the names."""
    def __init__(self, train, vocabulary, fields, order=3, history_orders=None):
        if not train or any(r['split'] != 'train' for r in train):
            raise ValueError('Encoder fitting requires training rows')
        if type(order) is not int or order < 1:
            raise ValueError('Invalid history order')
        history_orders = dict(history_orders or {})
        if any(not f.endswith('_history') or type(k) is not int or k < 1 for f, k in history_orders.items()):
            raise ValueError('Invalid per-field history order')
        self.vocabulary, self.fields, self.order = vocabulary, fields, order
        self.history_orders = history_orders
        self.categories = sorted({r['turn_category'] for r in train}) if 'turn_category' in fields else []
        self.names = pre_dec29_feature_names(fields, vocabulary, self.categories, order, history_orders)
        x = self.raw(train)
        self.mean, self.scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
        for j, name in enumerate(self.names):
            if name in ('gap_s', 'overlap_elapsed_s', 'log1p:gap_s', 'log1p:overlap_elapsed_s', 'retrospective_delay_s'):
                self.mean[j], self.scale[j] = x[:, j].mean(), x[:, j].std() or 1.0

    def lags(self, field):
        return self.history_orders.get(field, self.order)

    def raw(self, rows):
        result = []
        for r in rows:
            values = []
            for f in self.fields:
                if f.endswith('_history'):
                    for lag in range(1, self.lags(f)+1):
                        label = r[f][-lag] if len(r[f]) >= lag else None
                        values.extend(float(label == t) for t in self.vocabulary)
                elif f == 'turn_category':
                    values.extend(float(r[f] == c) for c in self.categories)
                elif f in ('gap_s', 'overlap_elapsed_s'):
                    v = r[f]
                    if not np.isfinite(v) or v < 0:
                        raise ValueError('Invalid onset-available timing')
                    values.extend([v, np.log1p(v)])
                elif f == 'retrospective_delay_s':
                    v = r[f]
                    if not np.isfinite(v):
                        raise ValueError('Invalid retrospective delay')
                    values.append(v)
                else:
                    raise ValueError(f'Unsupported input field {f}')
            result.append(values)
        return np.asarray(result, dtype=float)

    def transform(self, rows):
        x = (self.raw(rows) - self.mean) / self.scale
        return np.column_stack([x, np.ones(len(rows))])


def duration_rows(split, n, seed):
    """Rows with histories of varying length and aligned log-duration lists."""
    rng = np.random.RandomState(seed)
    vocab = ['a', 'b', 'c']
    rows = []
    for i in range(n):
        a, b = 1 + rng.randint(5), rng.randint(4)
        rows.append(dict(coda_id=f'{split}-{i}', split=split, Ct=vocab[rng.randint(3)],
                         A_history=[vocab[rng.randint(3)] for _ in range(a)], B_history=[vocab[rng.randint(3)] for _ in range(b)],
                         A_log_duration=list(rng.normal(-0.2, 0.5, a)), B_log_duration=list(rng.normal(0.1, 0.4, b)),
                         turn_category=['switch', 'overlap'][i % 2], gap_s=float(rng.rand()), overlap_elapsed_s=0.0))
    return rows


class ContinuousHistoryFieldTests(unittest.TestCase):
    """DEC-29: continuous history fields (the previous codas' log durations) in Encoder."""
    vocab = ['a', 'b', 'c', 'UNK']
    fields = ['A_history', 'A_log_duration', 'B_history', 'B_log_duration', 'turn_category', 'gap_s', 'overlap_elapsed_s']

    def test_column_names_and_order(self):
        train = duration_rows('train', 30, 1)
        e = Encoder(train, self.vocab, self.fields)
        one_hots = lambda f: [f'{f}:lag{lag}:{t}' for lag in (1, 2, 3) for t in self.vocab]  # noqa: E731
        self.assertEqual(e.names, one_hots('A_history') + ['A_log_duration:lag1', 'A_log_duration:lag2', 'A_log_duration:lag3']
                         + one_hots('B_history') + ['B_log_duration:lag1', 'B_log_duration:lag2', 'B_log_duration:lag3']
                         + ['turn:overlap', 'turn:switch', 'gap_s', 'log1p:gap_s', 'overlap_elapsed_s', 'log1p:overlap_elapsed_s'])
        self.assertEqual(e.continuous_columns(), [12, 13, 14, 27, 28, 29])
        self.assertFalse(any(is_timing_feature(n) for n in e.names if 'log_duration' in n))  # history block
        self.assertEqual(logistic.feature_names(self.fields, self.vocab, e.categories, 3), e.names)
        e2 = Encoder(train, self.vocab, self.fields, 3, {'A_history': 2})  # lags follow the history field's order
        self.assertEqual([n for n in e2.names if n.startswith('A_log')], ['A_log_duration:lag1', 'A_log_duration:lag2'])
        self.assertEqual([n for n in e2.names if n.startswith('B_log')], [f'B_log_duration:lag{k}' for k in (1, 2, 3)])

    def test_standardisation_uses_non_missing_training_values_only(self):
        train, validation = duration_rows('train', 40, 2), duration_rows('validation', 15, 3)
        e = Encoder(train, self.vocab, self.fields)
        missing = 0
        for field in ('A_log_duration', 'B_log_duration'):
            for lag in (1, 2, 3):
                j = e.names.index(f'{field}:lag{lag}')
                present = np.array([r[field][-lag] for r in train if len(r[field]) >= lag])
                missing += len(train) - len(present)
                self.assertGreater(len(present), 0)
                self.assertEqual(e.mean[j], present.mean())
                self.assertEqual(e.scale[j], present.std())
        self.assertGreater(missing, 20)  # the training rows do have missing lags, and they are left out
        before = (e.mean.copy(), e.scale.copy())
        shifted = [dict(r, A_log_duration=[v + 5 for v in r['A_log_duration']]) for r in validation]
        x = e.transform(shifted)  # validation rows never change the standardisation
        np.testing.assert_array_equal(e.mean, before[0])
        np.testing.assert_array_equal(e.scale, before[1])
        j = e.names.index('A_log_duration:lag1')
        np.testing.assert_allclose(x[:, j], [(r['A_log_duration'][-1] + 5 - e.mean[j]) / e.scale[j] for r in validation])

    def test_missing_lag_gives_zero(self):
        train = duration_rows('train', 40, 4)
        e = Encoder(train, self.vocab, self.fields)
        x = e.transform(train)
        for i, r in enumerate(train):
            for field in ('A_log_duration', 'B_log_duration'):
                for lag in (1, 2, 3):
                    j = e.names.index(f'{field}:lag{lag}')
                    if len(r[field]) < lag:
                        self.assertEqual(x[i, j], 0.0)
                    else:
                        self.assertAlmostEqual(x[i, j], (r[field][-lag] - e.mean[j]) / e.scale[j], places=12)
        self.assertFalse(np.isnan(x).any())
        constant = [dict(r, A_log_duration=[0.3] * len(r['A_history'])) for r in train]
        e_constant = Encoder(constant, self.vocab, self.fields)
        j = e_constant.names.index('A_log_duration:lag1')
        self.assertEqual((e_constant.mean[j], e_constant.scale[j]), (0.3, 1.0))  # SD 0 becomes 1, as for timing

    def test_invalid_continuous_fields_raise(self):
        train = duration_rows('train', 10, 5)
        with self.assertRaises(ValueError):
            Encoder([dict(r, A_log_duration=r['A_log_duration'][:-1]) for r in train], self.vocab, self.fields)
        with self.assertRaises(ValueError):
            Encoder([dict(r, A_log_duration=[float('nan')] * len(r['A_history'])) for r in train], self.vocab, self.fields)
        with self.assertRaises(ValueError):  # no training row has lag 3 of B
            Encoder([dict(r, B_history=r['B_history'][:2], B_log_duration=r['B_log_duration'][:2]) for r in train],
                    self.vocab, self.fields)

    def test_from_record_rebuilds_the_encoder(self):
        train, validation = duration_rows('train', 40, 6), duration_rows('validation', 15, 7)
        e = Encoder(train, self.vocab, self.fields)
        record = dict(vocabulary=self.vocab, fields=self.fields, order=3, categories=e.categories,
                      feature_names=e.names + ['intercept'], mean=e.mean.tolist(), scale=e.scale.tolist())
        rebuilt = Encoder.from_record(record)
        self.assertEqual(rebuilt.names, e.names)
        np.testing.assert_array_equal(rebuilt.transform(validation), e.transform(validation))

    def test_fit_with_duration_inputs(self):
        train, validation = duration_rows('train', 60, 8), duration_rows('validation', 20, 9)
        config = dict(history_order=3, l2_grid=[.01, .1], max_iterations=1000, gradient_tolerance=1e-6)
        encoder, chosen, grid = select_l2_and_fit(train, validation, self.vocab, self.fields, config)
        self.assertEqual(len(grid), 4)  # history and S penalties searched jointly; durations are history
        self.assertEqual(chosen['w'].shape, (len(encoder.names) + 1, len(self.vocab)))
        self.assertIsNotNone(chosen['beta'])
        self.assertIsNotNone(chosen['gamma'])

    def test_without_continuous_fields_every_path_is_unchanged(self):
        """Feature names, standardisation, design matrices and a fit are bitwise those of the encoder before DEC-29,
        on rows with and without duration lists, with and without per-field orders."""
        vocab = ['a', 'b', 'c', 'UNK']
        conditions = [[], ['turn_category', 'gap_s', 'overlap_elapsed_s'], ['A_history'], ['B_history'], ['A_history', 'B_history'],
                      ['A_history', 'turn_category', 'gap_s', 'overlap_elapsed_s'],
                      ['A_history', 'B_history', 'turn_category', 'gap_s', 'overlap_elapsed_s'],
                      ['turn_category', 'retrospective_delay_s']]
        train, test = duration_rows('train', 50, 10), duration_rows('test', 20, 11)
        for rows in (train, test):
            for i, r in enumerate(rows):
                r['retrospective_delay_s'] = float(i % 7) - 2.5
        plain = lambda rows: [{k: v for k, v in r.items() if not k.endswith('log_duration')} for r in rows]  # noqa: E731
        for fields in conditions:
            for orders in (None, {'A_history': 2}):
                for train_rows, test_rows in ((train, test), (plain(train), plain(test))):
                    old = PreDec29Encoder(plain(train), vocab, fields, 3, orders)
                    new = Encoder(train_rows, vocab, fields, 3, orders)
                    self.assertEqual(new.names, old.names)
                    self.assertEqual(new.continuous_columns(), [])
                    self.assertEqual(new.mean.tobytes(), old.mean.tobytes())
                    self.assertEqual(new.scale.tobytes(), old.scale.tobytes())
                    self.assertEqual(new.transform(test_rows).tobytes(), old.transform(plain(test)).tobytes())
        fields = ['A_history', 'B_history', 'turn_category', 'gap_s', 'overlap_elapsed_s']
        validation = duration_rows('validation', 20, 12)
        config = dict(history_order=3, l2_grid=[.01, .1], max_iterations=1000, gradient_tolerance=1e-6)
        _, chosen, grid = select_l2_and_fit(train, validation, vocab, fields, config)
        with mock.patch.object(logistic, 'Encoder', PreDec29Encoder):
            _, chosen_old, grid_old = select_l2_and_fit(plain(train), plain(validation), vocab, fields, config)
        self.assertEqual((chosen['l2'], chosen['l2_s'], chosen['beta'], chosen['gamma']),
                         (chosen_old['l2'], chosen_old['l2_s'], chosen_old['beta'], chosen_old['gamma']))
        self.assertEqual(chosen['w'].tobytes(), chosen_old['w'].tobytes())
        self.assertEqual([g['validation_log_likelihood_bits'] for g in grid], [g['validation_log_likelihood_bits'] for g in grid_old])
