import unittest

from whale_coda_dynamics.preprocessing.vocabulary import (
    fit_vocabulary, apply_vocabulary, smoothed_label_prior, validate_mapping,
)


class VocabularyTests(unittest.TestCase):
    def test_mapping_recovery_prior_and_validation(self):
        events = [dict(coda_id=str(i), coda_type=t, split=s) for i, (t, s) in enumerate(
            [('a', 'train'), ('a', 'train'), ('b', 'train'), ('c', 'test')])]
        rows = [dict(coda_id=e['coda_id'], Ct=e['coda_type'], split=e['split'],
                     comparison_eligible=True, A_history_ids=['2'], A_history=['b'],
                     B_history_ids=[], B_history=[]) for e in events]
        policy = fit_vocabulary(events[:3])
        mapped = apply_vocabulary(rows, policy)
        self.assertEqual([r['Ct'] for r in mapped], ['a', 'a', 'UNK', 'UNK'])
        self.assertEqual(rows[2]['Ct'], 'b')
        self.assertEqual(mapped[2]['Ct_raw'], 'b')
        self.assertTrue(all(r['A_history'] == ['UNK'] for r in mapped))
        self.assertEqual(smoothed_label_prior(mapped[:3], policy), {'a': 3/5, 'UNK': 2/5})
        validate_mapping(mapped, events, policy)
        # A future fold/threshold must recover raw labels, not refit mapped UNK.
        restored = apply_vocabulary(mapped, fit_vocabulary(events[:3], min_count=1))
        self.assertEqual(restored[2]['Ct'], 'b')
        self.assertEqual(restored[0]['A_history'], ['b'])
        mapped[0]['A_history'] = ['a']
        with self.assertRaises(ValueError):
            validate_mapping(mapped, events, policy)

    def test_reject_leakage_duplicates_and_collision(self):
        event = dict(coda_id='0', coda_type='a', split='train')
        for events in ([event, event], [dict(event, split='test')],
                       [dict(event, coda_type='UNK')], []):
            with self.assertRaises(ValueError):
                fit_vocabulary(events)
