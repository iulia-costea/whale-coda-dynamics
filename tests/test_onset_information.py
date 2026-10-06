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

DEC-28's switch-structure features (switch_structure.py) come from the whale's whole earlier stream
in the exchange; the same property is tested for them: changing or deleting any coda at or after the
target's onset leaves every feature unchanged.
"""
import importlib.util
import random
import unittest

from whale_coda_dynamics.preprocessing.features import build_features

HAS_MODELING = all(importlib.util.find_spec(m) for m in ('numpy', 'scipy', 'torch'))

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


def switch_features_of(events, coda_id):
    from whale_coda_dynamics.switch_structure import FEATURES, class_values, own_streams, stream_structure
    stream = own_streams([dict(e, Ct=e['coda_type']) for e in events], coda_ids=[coda_id])[coda_id]
    structure = stream_structure(stream)
    return {f: class_values(structure, f) for f in FEATURES}


@unittest.skipUnless(HAS_MODELING, 'needs numpy, scipy and torch (pip install ".[modeling]")')
class SwitchStructureOnsetTests(unittest.TestCase):
    def test_switch_features_ignore_codas_at_or_after_target_onset(self):
        """Run length, d1, d2 and repertoire counts: every coda at or after the target's onset is changed (type,
        caller, end) or deleted, and a partner coda still sounding at the onset gets another type and end. A
        target whose own earlier coda is still sounding raises (its type is not known at the onset) and is skipped."""
        rng = random.Random(7)
        checked = skipped = deleted = 0
        for _ in range(600):
            events = random_exchange(rng, rng.randint(4, 14))
            target = rng.choice(events)
            onset, whale = target['onset_s'], target['whale_id']
            try:
                expected = switch_features_of(events, target['coda_id'])
            except ValueError:
                skipped += 1
                continue
            changed = []
            for e in events:
                e = dict(e)
                if e['coda_id'] != target['coda_id'] and e['onset_s'] > onset and rng.random() < 0.3:
                    deleted += 1
                    continue
                if e['coda_id'] == target['coda_id']:
                    e['coda_type'] = rng.choice(TYPES)
                    e['offset_s'] = round(e['onset_s'] + rng.uniform(0.3, 5.0), 3)
                elif e['onset_s'] > onset:
                    e['coda_type'] = rng.choice(TYPES)
                    e['whale_id'] = rng.choice(['w1', 'w2'])
                    e['offset_s'] = round(e['onset_s'] + rng.uniform(0.3, 5.0), 3)
                elif e['offset_s'] > onset:
                    self.assertNotEqual(e['whale_id'], whale)  # own codas still sounding raised above
                    e['coda_type'] = rng.choice(TYPES)
                    e['offset_s'] = round(onset + rng.uniform(0.001, 5.0), 3)
                changed.append(e)
            self.assertEqual(switch_features_of(changed, target['coda_id']), expected)
            checked += 1
        self.assertGreater(checked, 300)
        self.assertGreater(deleted, 300)

    def test_switch_features_use_earlier_own_codas(self):
        # Not vacuous: an earlier own coda's type matters; a partner coda's does not.
        events = [dict(coda_id='a', whale_id='w1', exchange_id='x', coda_type='1+1+3', onset_s=0.0, offset_s=1.0),
                  dict(coda_id='b', whale_id='w2', exchange_id='x', coda_type='5R1', onset_s=1.5, offset_s=2.5),
                  dict(coda_id='c', whale_id='w1', exchange_id='x', coda_type='4R2', onset_s=3.0, offset_s=4.0),
                  dict(coda_id='d', whale_id='w1', exchange_id='x', coda_type='4R2', onset_s=4.5, offset_s=5.0)]
        before = switch_features_of(events, 'd')
        self.assertEqual(before['distinct_1'], {'1+1+3': 1.0})
        self.assertNotEqual(switch_features_of([dict(events[0], coda_type='10i')] + events[1:], 'd'), before)
        self.assertEqual(switch_features_of(events[:1] + [dict(events[1], coda_type='10i')] + events[2:], 'd'), before)


if __name__ == '__main__':
    unittest.main()
