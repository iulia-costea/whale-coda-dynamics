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

DEC-30's duration lattice (duration_lattice.py) reads the durations and click counts of the codas in
A_history_ids and B_history_ids: changing or deleting any coda at or after the target's onset (type,
caller, end, click count), and changing a coda still sounding at the onset, leaves every A, B and S
column unchanged. The always-present columns are the target's own click count (and, in variant T, its
type): the conditioning variables of the target by design (DEC-30), not onset-available inputs.

DEC-29's rubato inputs (rubato.py) are the durations of the codas in A_history_ids and B_history_ids, as
continuous history fields of logistic.Encoder: changing or deleting any coda at or after the target's onset,
and changing a coda still sounding at the onset, leaves every duration field and duration column unchanged;
changing one completed coda's duration changes exactly the inputs that reference it.
"""
import importlib.util
import math
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


DURATION_SETTINGS = dict(history_lags=3, count_one_hot_min_training_targets=1)


def with_clicks(rng, events):
    return [dict(e, n_clicks=rng.randint(3, 12)) for e in events]


def duration_lookup(events):
    return {e['coda_id']: dict(log_duration=math.log(e['offset_s'] - e['onset_s']), clicks=e['n_clicks'],
                               log_clicks=math.log(e['n_clicks'])) for e in events}


def duration_rows(events):
    return [dict(r, split='train') for r in build_features(events, provisional_assumptions_accepted=True, history_limit=4)]


def duration_inputs_of(encoder, events, coda_id):
    """The target's A, B and S columns of the full design (A, B, S), from an encoder fitted on the original
    exchange; the always-present columns (the target's own click count) are left out."""
    rows = duration_rows(events)
    row = next(r for r in rows if r['coda_id'] == coda_id)
    x = encoder.transform([row], duration_lookup(events))[0]
    return [float(v) for v, g in zip(x, encoder.groups) if g in ('A', 'B', 'S')]


@unittest.skipUnless(HAS_MODELING, 'needs numpy, scipy and torch (pip install ".[modeling]")')
class DurationLatticeOnsetTests(unittest.TestCase):
    def test_duration_inputs_ignore_codas_at_or_after_target_onset(self):
        from whale_coda_dynamics.duration_lattice import DurationEncoder
        rng = random.Random(11)
        checked = deleted = 0
        for _ in range(400):
            events = with_clicks(rng, random_exchange(rng, rng.randint(4, 12)))
            target = rng.choice(events)
            onset = target['onset_s']
            rows = duration_rows(events)
            encoder = DurationEncoder(rows, duration_lookup(events), ('A', 'B', 'S'),
                                      ['turn_category', 'gap_s', 'overlap_elapsed_s'], DURATION_SETTINGS)
            expected = duration_inputs_of(encoder, events, target['coda_id'])
            changed = []
            for e in events:
                e = dict(e)
                if e['coda_id'] != target['coda_id'] and e['onset_s'] > onset and rng.random() < 0.3:
                    deleted += 1
                    continue
                if e['coda_id'] == target['coda_id']:
                    e.update(coda_type=rng.choice(TYPES), offset_s=round(e['onset_s'] + rng.uniform(0.3, 5.0), 3),
                             n_clicks=rng.randint(3, 12))
                elif e['onset_s'] > onset:
                    e.update(coda_type=rng.choice(TYPES), whale_id=rng.choice(['w1', 'w2']),
                             offset_s=round(e['onset_s'] + rng.uniform(0.3, 5.0), 3), n_clicks=rng.randint(3, 12))
                elif e['offset_s'] > onset:  # still sounding at onset: type, end and click count not yet known
                    e.update(coda_type=rng.choice(TYPES), offset_s=round(onset + rng.uniform(0.001, 5.0), 3),
                             n_clicks=rng.randint(3, 12))
                changed.append(e)
            if len({e['whale_id'] for e in changed}) != 2:
                continue
            self.assertEqual(duration_inputs_of(encoder, changed, target['coda_id']), expected)
            checked += 1
        self.assertGreater(checked, 250)
        self.assertGreater(deleted, 200)

    def test_duration_inputs_use_earlier_completed_codas(self):
        # Not vacuous: an earlier completed partner coda's duration and click count change the inputs.
        from whale_coda_dynamics.duration_lattice import DurationEncoder
        events = [dict(coda_id='a', whale_id='w1', exchange_id='x', coda_type='1+1+3', onset_s=0.0, offset_s=1.0, n_clicks=5),
                  dict(coda_id='b', whale_id='w2', exchange_id='x', coda_type='5R1', onset_s=1.5, offset_s=2.5, n_clicks=5),
                  dict(coda_id='c', whale_id='w1', exchange_id='x', coda_type='4R2', onset_s=3.0, offset_s=4.0, n_clicks=4),
                  dict(coda_id='d', whale_id='w2', exchange_id='x', coda_type='4R2', onset_s=4.5, offset_s=5.5, n_clicks=4),
                  dict(coda_id='e', whale_id='w1', exchange_id='x', coda_type='4R2', onset_s=6.0, offset_s=7.0, n_clicks=4)]
        encoder = DurationEncoder(duration_rows(events), duration_lookup(events), ('A', 'B', 'S'),
                                  ['turn_category', 'gap_s', 'overlap_elapsed_s'], DURATION_SETTINGS)
        before = duration_inputs_of(encoder, events, 'e')
        longer = events[:3] + [dict(events[3], offset_s=5.9, n_clicks=6)] + events[4:]
        self.assertNotEqual(duration_inputs_of(encoder, longer, 'e'), before)



RUBATO_FIELDS = ['A_history', 'A_log_duration', 'B_history', 'B_log_duration']


def rubato_rows(events, durations_from=None):
    """build_features' rows with the duration fields; durations from `durations_from` (default: the same events)."""
    from whale_coda_dynamics.rubato import duration_lookup, with_durations
    onsets = {e['coda_id']: e['onset_s'] for e in events}  # prepare.py adds the target's onset to each row
    rows = [dict(r, onset_s=onsets[r['coda_id']]) for r in build_features(events, provisional_assumptions_accepted=True,
                                                                            history_limit=4)]
    return [dict(r, split='train') for r in with_durations(rows, duration_lookup(durations_from or events))]


def rubato_encoder(rng):
    """logistic.Encoder with the duration fields, fitted once on rows pooled from many random exchanges (so every lag
    has training values) and then held fixed."""
    from whale_coda_dynamics.logistic import Encoder
    pooled = []
    for k in range(40):
        pooled += rubato_rows([dict(e, coda_id=f'p{k}-{e["coda_id"]}', exchange_id=f'p{k}')
                               for e in random_exchange(rng, rng.randint(8, 14))])
    return Encoder(pooled, TYPES, RUBATO_FIELDS)


def rubato_inputs_of(encoder, events, coda_id):
    row = next(r for r in rubato_rows(events) if r['coda_id'] == coda_id)
    columns = encoder.continuous_columns()
    return dict(fields={f: row[f] for f in ('A_log_duration', 'B_log_duration')},
                columns=[float(v) for v in encoder.transform([row])[0][columns]])


@unittest.skipUnless(HAS_MODELING, 'needs numpy, scipy and torch (pip install ".[modeling]")')
class RubatoOnsetTests(unittest.TestCase):
    def test_duration_inputs_ignore_codas_at_or_after_target_onset(self):
        rng = random.Random(29)
        encoder = rubato_encoder(rng)
        checked = deleted = 0
        for _ in range(400):
            events = random_exchange(rng, rng.randint(4, 12))
            target = rng.choice(events)
            onset = target['onset_s']
            expected = rubato_inputs_of(encoder, events, target['coda_id'])
            changed = []
            for e in events:
                e = dict(e)
                if e['coda_id'] != target['coda_id'] and e['onset_s'] > onset and rng.random() < 0.3:
                    deleted += 1
                    continue
                if e['coda_id'] == target['coda_id']:
                    e.update(coda_type=rng.choice(TYPES), offset_s=round(e['onset_s'] + rng.uniform(0.3, 5.0), 3))
                elif e['onset_s'] > onset:
                    e.update(coda_type=rng.choice(TYPES), whale_id=rng.choice(['w1', 'w2']),
                             offset_s=round(e['onset_s'] + rng.uniform(0.3, 5.0), 3))
                elif e['offset_s'] > onset:  # still sounding at onset: type and end (so duration) not yet known
                    e.update(coda_type=rng.choice(TYPES), offset_s=round(onset + rng.uniform(0.001, 5.0), 3))
                changed.append(e)
            if len({e['whale_id'] for e in changed}) != 2:
                continue
            self.assertEqual(rubato_inputs_of(encoder, changed, target['coda_id']), expected)
            checked += 1
        self.assertGreater(checked, 250)
        self.assertGreater(deleted, 200)

    def test_one_codas_duration_changes_exactly_the_inputs_that_reference_it(self):
        """With the histories fixed, shortening one referenced coda (it still ends before every target whose history
        holds it) changes the duration fields at exactly the positions holding its ID, and the encoded duration
        columns at exactly its lags 1-3, in every row."""
        rng = random.Random(30)
        encoder = rubato_encoder(rng)
        columns = encoder.continuous_columns()
        names = [encoder.names[j] for j in columns]
        checked = 0
        for _ in range(150):
            events = random_exchange(rng, rng.randint(6, 14))
            rows = rubato_rows(events)
            referenced = sorted({i for r in rows for i in r['A_history_ids'] + r['B_history_ids']})
            if not referenced:
                continue
            coda_id = rng.choice(referenced)
            shorter = [dict(e, offset_s=round(e['onset_s'] + (e['offset_s'] - e['onset_s']) / 2, 4)) if e['coda_id'] == coda_id
                       else e for e in events]
            after = {r['coda_id']: r for r in rubato_rows(events, durations_from=shorter)}
            x_before, x_after = encoder.transform(rows)[:, columns], encoder.transform([after[r['coda_id']] for r in rows])[:, columns]
            for i, r in enumerate(rows):
                a = after[r['coda_id']]
                for ids, field in ((r['A_history_ids'], 'A_log_duration'), (r['B_history_ids'], 'B_log_duration')):
                    self.assertEqual(a[field.replace('log_duration', 'history_ids')], ids)
                    for k, (i_d, before_v, after_v) in enumerate(zip(ids, r[field], a[field])):
                        self.assertEqual(before_v != after_v, i_d == coda_id)
                for j, name in enumerate(names):
                    field, lag = name.split(':lag')
                    ids = r[field.replace('log_duration', 'history_ids')]
                    references = len(ids) >= int(lag) and ids[-int(lag)] == coda_id
                    self.assertEqual(x_before[i, j] != x_after[i, j], references, (r['coda_id'], name))
                    checked += references
        self.assertGreater(checked, 100)


if __name__ == '__main__':
    unittest.main()
