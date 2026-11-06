import unittest

from scripts.audit_edit_support import group_relation, transition_record


def event(kind='tap', lane=0, end=100_000):
    return dict(onset_us=100_000, end_us=end, type=kind, lane=lane)


class EditSupportTest(unittest.TestCase):
    def test_binding_swap_is_negative_despite_equal_field_marginals(self):
        target = [event(lane=0), event('hold', 1, 9_100_000)]
        source = [event('hold', 0, 9_100_000), event(lane=1)]
        row = group_relation(source, target, 'mania_4k')
        self.assertEqual(row['label'], 'replace')
        self.assertTrue(row['binding_swap_negative'])
        self.assertTrue(row['locally_valid_payload'])
        self.assertEqual(source[0]['end_us'], 9_100_000)

    def test_phira_multiplicity_and_complete_tail_are_part_of_keep(self):
        a = event(lane=-1)
        self.assertEqual(group_relation([a,a], [a,a], 'phira')['label'], 'keep')
        self.assertEqual(group_relation([a,a], [a], 'phira')['label'], 'replace')
        self.assertTrue(group_relation([event('hold',-1,200_000)],
            [event('hold',-1,201_000)],'phira')['content_negative'])

    def test_unresolved_never_becomes_replace_or_a_repaired_group(self):
        row = transition_record(None, [event()], [event()], 'mania_4k', True)
        self.assertEqual(row['label'], 'unresolved')
        self.assertFalse(row['copy_eligible_in_saved_transition'])
        self.assertFalse(row['wrong_group_repaired'])
        row = transition_record([event()], [event()], None, 'mania_4k', True)
        self.assertEqual(row['following_label'], 'unresolved')
        self.assertFalse(row['correct_group_damaged'])

    def test_copy_eligibility_obeys_predicted_count_and_copy_all_blocks_repairs(self):
        target = [event()]
        dropped = transition_record(target, target, None, 'mania_4k', False)
        self.assertTrue(dropped['correct_group_damaged'])
        self.assertFalse(dropped['copy_eligible_in_saved_transition'])
        repaired = transition_record([event(lane=1)], target, target, 'mania_4k', True)
        self.assertTrue(repaired['copy_eligible_in_saved_transition'])
        self.assertTrue(repaired['wrong_group_repaired'])
        # A different hidden target cannot change the source's runtime eligibility.
        changed = transition_record([event(lane=1)], [event(lane=2)], target, 'mania_4k', True)
        self.assertEqual(changed['copy_eligible_in_saved_transition'], repaired['copy_eligible_in_saved_transition'])


if __name__ == '__main__': unittest.main()
