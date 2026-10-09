"""Ornament merge (PLAN [P7], DEC-31): the [D1]/[D2] match, the neighbour rule, relabelling, and the
committed inventory O file against DEC-31's independently computed counts.

Standard library only.
"""
import unittest
from pathlib import Path

from whale_coda_dynamics.ornament_merge import (check_expected, exchange_streams, find_neighbour,
                                                 match_d1_to_d2, relabel_ornaments)
from whale_coda_dynamics.preprocessing.inventories import check_relabelled, relabel
from whale_coda_dynamics.preprocessing.io_utils import read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / 'config' / 'rq2_o_ornament_merged.json')


def row(coda_id, exchange_id='e1', whale_id='w1', onset_s=0.0):
    return dict(coda_id=coda_id, exchange_id=exchange_id, whale_id=whale_id, onset_s=onset_s)


class MatchTests(unittest.TestCase):
    def test_matches_one_to_one_on_rounded_rec_tsto_duration(self):
        targets = [dict(coda_id='a'), dict(coda_id='b')]
        normalized = {
            # The two releases print the same value differently (inventories.py's own example).
            'a': {'raw_values': {'REC': 'r1', 'TsTo': '0.9277916000000002', 'Duration': '1.0'}},
            'b': {'raw_values': {'REC': 'r1', 'TsTo': '2.0', 'Duration': '1.0'}},  # no matching [D2] row
        }
        d2_rows = [{'REC': 'r1', 'TsTo': '0.9277916', 'Duration': '1.0'}]  # same key at 3 decimals
        match, unmatchable = match_d1_to_d2(targets, normalized, d2_rows, decimals=3)
        self.assertEqual(match, {'a': 0})
        self.assertEqual(unmatchable, 0)

    def test_duplicate_key_on_either_side_raises(self):
        targets = [dict(coda_id='a'), dict(coda_id='b')]
        normalized = {cid: {'raw_values': {'REC': 'r1', 'TsTo': '0.1', 'Duration': '1.0'}} for cid in ('a', 'b')}
        d2_rows = [{'REC': 'r1', 'TsTo': '0.1', 'Duration': '1.0'}]
        with self.assertRaises(ValueError):
            match_d1_to_d2(targets, normalized, d2_rows, decimals=3)

    def test_unparseable_key_counts_as_unmatchable_not_an_error(self):
        targets = [dict(coda_id='a')]
        normalized = {'a': {'raw_values': {'REC': 'r1', 'TsTo': 'n/a', 'Duration': '1.0'}}}
        match, unmatchable = match_d1_to_d2(targets, normalized, [], decimals=3)
        self.assertEqual(match, {})
        self.assertEqual(unmatchable, 1)


class NeighbourRuleTests(unittest.TestCase):
    """find_neighbour: both qualify -> nearer, tie -> preceding; one qualifies -> that one; none -> None."""

    def test_tie_goes_to_preceding(self):
        stream = [row('prev', onset_s=0.0), row('mid', onset_s=5.0), row('next', onset_s=10.0)]
        n_clicks = {'prev': 4, 'mid': 5, 'next': 4}  # both neighbours one fewer click, equally distant (5 s)
        neighbour, side = find_neighbour('mid', 1, stream, n_clicks, window_s=10.0)
        self.assertEqual((neighbour, side), ('prev', 'preceding'))

    def test_nearer_neighbour_wins_when_both_qualify(self):
        stream = [row('prev', onset_s=0.0), row('mid', onset_s=9.0), row('next', onset_s=10.0)]
        n_clicks = {'prev': 4, 'mid': 5, 'next': 4}
        neighbour, side = find_neighbour('mid', 1, stream, n_clicks, window_s=10.0)
        self.assertEqual((neighbour, side), ('next', 'following'))  # 1 s away vs 9 s away

    def test_window_boundary_is_inclusive(self):
        stream = [row('prev', onset_s=0.0), row('mid', onset_s=10.0)]
        n_clicks = {'prev': 4, 'mid': 5}
        self.assertEqual(find_neighbour('mid', 1, stream, n_clicks, window_s=10.0), ('prev', 'preceding'))
        self.assertEqual(find_neighbour('mid', 1, stream, n_clicks, window_s=9.999), (None, None))

    def test_wrong_click_count_does_not_qualify(self):
        stream = [row('prev', onset_s=0.0), row('mid', onset_s=5.0)]
        n_clicks = {'prev': 5, 'mid': 5}  # same click count, not one fewer
        self.assertEqual(find_neighbour('mid', 1, stream, n_clicks, window_s=10.0), (None, None))

    def test_no_neighbour_at_either_end_of_the_stream(self):
        stream = [row('only', onset_s=0.0)]
        self.assertEqual(find_neighbour('only', 0, stream, {'only': 5}, window_s=10.0), (None, None))


class ExchangeStreamsTests(unittest.TestCase):
    def test_groups_by_exchange_and_whale_sorted_by_onset(self):
        rows = [row('b', onset_s=2.0), row('a', onset_s=1.0), row('other_whale', whale_id='w2', onset_s=0.5)]
        streams = exchange_streams(rows)
        self.assertEqual([r['coda_id'] for r in streams[('e1', 'w1')]], ['a', 'b'])
        self.assertEqual([r['coda_id'] for r in streams[('e1', 'w2')]], ['other_whale'])


class RelabelOrnamentsTests(unittest.TestCase):
    def test_flagged_coda_takes_neighbours_reference_label_and_reaches_histories(self):
        targets = [
            dict(coda_id='a', exchange_id='e1', whale_id='w1', onset_s=0.0, Ct_raw='typeA', Ct='typeA',
                A_history=[], B_history=[], A_history_ids=[], B_history_ids=[]),
            dict(coda_id='b', exchange_id='e1', whale_id='w1', onset_s=5.0, Ct_raw='typeB', Ct='typeB',
                A_history=['typeA'], B_history=[], A_history_ids=['a'], B_history_ids=[]),
        ]
        normalized = {'a': {'raw_values': {'nClicks': '5'}}, 'b': {'raw_values': {'nClicks': '6'}}}
        labels, records = relabel_ornaments(targets, normalized, flagged={'b'}, window_s=10.0)
        self.assertEqual(labels, {'a': 'typeA', 'b': 'typeA'})  # b's neighbour a has one fewer click (5 vs 6)
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]['old_label'], records[0]['new_label'], records[0]['neighbour_id'], records[0]['side']),
                         ('typeB', 'typeA', 'a', 'preceding'))
        relabelled = relabel(targets, labels, 'o')
        check_relabelled(targets, relabelled, labels, 'o')  # the new label reaches A_history
        self.assertEqual(relabelled[1]['A_history'], ['typeA'])

    def test_flagged_coda_without_a_qualifying_neighbour_keeps_its_label(self):
        targets = [dict(coda_id='a', exchange_id='e1', whale_id='w1', onset_s=0.0, Ct_raw='typeA', Ct='typeA',
                       A_history=[], B_history=[], A_history_ids=[], B_history_ids=[])]
        normalized = {'a': {'raw_values': {'nClicks': '5'}}}
        labels, records = relabel_ornaments(targets, normalized, flagged={'a'}, window_s=10.0)
        self.assertEqual(labels, {'a': 'typeA'})
        self.assertEqual(records[0]['neighbour_id'], None)
        self.assertEqual(records[0]['side'], None)


class CheckExpectedTests(unittest.TestCase):
    def test_mismatch_is_reported_without_adjustment(self):
        stats = {'matched': 1706, 'flagged': 66, 'top_changes': [[['6i', '5R1'], 18]]}
        self.assertEqual(check_expected(stats, {'matched': 1706}), [])
        mismatches = check_expected(stats, {'flagged': 67})
        self.assertEqual(mismatches, [('flagged', 67, 66)])
        self.assertEqual(check_expected(stats, {'top_changes': [[['6i', '5R1'], 18]]}), [])
        self.assertEqual(check_expected(stats, {'top_changes': [[['6i', '5R1'], 19]]}),
                         [('top_changes', [[['6i', '5R1'], 19]], [[['6i', '5R1'], 18]])])


class CommittedInventoryTests(unittest.TestCase):
    """The committed inventory O file against DEC-31's independently computed counts (the build stage's own
    gate already enforces this when the module runs; this is the regression check that the committed file
    still agrees)."""

    @classmethod
    def setUpClass(cls):
        cls.rq1 = read_jsonl(ROOT / CONFIG['targets'])
        cls.o_rows = read_jsonl(ROOT / CONFIG['output_directory'] / f"rq2_{CONFIG['name']}_targets.jsonl")

    def test_only_label_fields_and_inventory_differ(self):
        self.assertEqual([r['coda_id'] for r in self.o_rows], [r['coda_id'] for r in self.rq1])
        for before, after in zip(self.rq1, self.o_rows):
            self.assertEqual(set(after), set(before) | {'inventory'})
            self.assertEqual(after['inventory'], CONFIG['name'])
            self.assertEqual(after['Ct'], after['Ct_raw'])

    def test_histories_are_the_labels_of_their_ids(self):
        labels = {r['coda_id']: r['Ct_raw'] for r in self.o_rows}
        for row_ in self.o_rows:
            for prefix in ('A', 'B'):
                self.assertEqual(row_[prefix + '_history'], [labels[i] for i in row_[prefix + '_history_ids']])

    def test_counts_match_dec31s_expected_table(self):
        reference = {r['coda_id']: r['Ct_raw'] for r in self.rq1}
        labels = {r['coda_id']: r['Ct_raw'] for r in self.o_rows}
        # Recompute only what build_statistics needs beyond the committed files, from the committed rows.
        from collections import Counter
        comparison = [r for r in self.rq1 if r['comparison_eligible']]
        changed_comparison = sum(labels[r['coda_id']] != reference[r['coda_id']] for r in comparison)
        self.assertEqual(changed_comparison, CONFIG['expected']['comparison_targets_changed'])
        changed_all = sum(labels[c] != reference[c] for c in reference)
        self.assertEqual(changed_all, CONFIG['expected']['rows_changed'])


if __name__ == '__main__':
    unittest.main()
