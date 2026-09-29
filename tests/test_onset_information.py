"""The primary inputs A, B and S use only what is known at the target coda's onset (CLAUDE.md).

Property test on build_features with random two-caller exchanges: changing anything that is
not yet known when the target starts must leave the target's A_history, B_history and primary
S fields unchanged. Unknown at onset:
- the target's own coda type and end time;
- anything about codas that start later;
- the type and exact end time of a coda still sounding at the target's onset (only
  "still sounding" is known; histories hold completed codas only).

The retrospective delay is excluded on purpose: it is a sensitivity field that can use a
post-onset end point (flagged by delay_uses_post_onset_endpoint).
"""
import random
import unittest

from whale_coda_dynamics.preprocessing.features import build_features

PRIMARY = ('A_history', 'B_history', 'A_history_ids', 'B_history_ids', 'turn_predecessor_id',
           'turn_category', 'gap_s', 'overlap_elapsed_s')
TYPES = ['1+1+3', '5R1', '4R2', '10i', '5R3']


def random_exchange(rng, size):
    events, onset = [], 0.0
    for i in range(size):
        onset += rng.uniform(0.05, 4.0)
        events.append(dict(coda_id=f'c{i}', whale_id=rng.choice(['w1', 'w2']), exchange_id='x',
                           coda_type=rng.choice(TYPES), onset_s=round(onset, 3),
                           offset_s=round(onset + rng.uniform(0.3, 2.5), 3)))
    events[0]['whale_id'], events[1]['whale_id'] = 'w1', 'w2'  # exactly two callers
    return events


def features_of(events, coda_id):
    rows = build_features(events, provisional_assumptions_accepted=True, history_limit=4)
    row = next(r for r in rows if r['coda_id'] == coda_id)
    return {k: row[k] for k in PRIMARY}


class OnsetInformationTests(unittest.TestCase):
    def test_primary_inputs_ignore_information_after_target_onset(self):
        rng = random.Random(42)
        checked = 0
        for _ in range(300):
            events = random_exchange(rng, rng.randint(4, 12))
            target = rng.choice(events)
            onset = target['onset_s']
            expected = features_of(events, target['coda_id'])
            changed = []
            for e in events:
                e = dict(e)
                if e['coda_id'] == target['coda_id']:
                    e['coda_type'] = rng.choice(TYPES)
                    e['offset_s'] = round(e['onset_s'] + rng.uniform(0.3, 5.0), 3)
                elif e['onset_s'] > onset:
                    e['coda_type'] = rng.choice(TYPES)
                    e['whale_id'] = rng.choice(['w1', 'w2'])
                    e['offset_s'] = round(e['onset_s'] + rng.uniform(0.3, 5.0), 3)
                elif e['offset_s'] > onset:  # still sounding at onset: type and end not yet known
                    e['coda_type'] = rng.choice(TYPES)
                    e['offset_s'] = round(onset + rng.uniform(0.001, 5.0), 3)
                changed.append(e)
            if len({e['whale_id'] for e in changed}) != 2:
                continue
            self.assertEqual(features_of(changed, target['coda_id']), expected)
            checked += 1
        self.assertGreater(checked, 250)

    def test_known_information_does_change_inputs(self):
        # Sanity check that the property above is not vacuous: an earlier completed coda's type matters.
        events = [dict(coda_id='a', whale_id='w1', exchange_id='x', coda_type='1+1+3', onset_s=0.0, offset_s=1.0),
                  dict(coda_id='b', whale_id='w2', exchange_id='x', coda_type='5R1', onset_s=1.5, offset_s=2.5),
                  dict(coda_id='c', whale_id='w1', exchange_id='x', coda_type='4R2', onset_s=3.0, offset_s=4.0)]
        before = features_of(events, 'c')
        events[1] = dict(events[1], coda_type='10i')
        self.assertNotEqual(features_of(events, 'c'), before)


if __name__ == '__main__':
    unittest.main()
