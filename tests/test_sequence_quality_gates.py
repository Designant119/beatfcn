import json
from pathlib import Path
import unittest

from scripts.compare_local_sequence_quality import numerical_gates, type_match_count


class QualityGateTest(unittest.TestCase):
    def setUp(self):
        self.limits=json.loads(Path('bench/rhythm-v2/d87-sequence-candidate-v1.json').read_text())['quality_gates']
        self.control=dict(macro_f1=.1,tap_positive_intervals=100,tap_short_fraction=.3,
                          native_tag_macro_f1=dict(EZ=.2,HD=.2),reference_rest_mean=1)
        self.candidate=dict(macro_f1=.15,tap_positive_intervals=100,tap_short_fraction=.1,
                            native_tag_macro_f1=dict(EZ=.2,HD=.2),reference_rest_mean=1,
                            complete_legal_songs=20,nonempty_songs=20,songs_not_worse=15,
                            type_matches=dict(hold=1,flick=1,drag=1))

    def test_numerically_reachable(self):
        self.assertTrue(all(numerical_gates(self.control,self.candidate,self.limits).values()))

    def test_empty_cannot_pass_spacing(self):
        self.candidate.update(tap_positive_intervals=0,tap_short_fraction=None,nonempty_songs=0)
        result=numerical_gates(self.control,self.candidate,self.limits)
        self.assertFalse(result['tap_spacing']);self.assertFalse(result['nonempty'])

    def test_type_disappearance_fails_despite_aggregate_gain(self):
        self.candidate['type_matches']['hold']=0
        self.assertFalse(numerical_gates(self.control,self.candidate,self.limits)['nontap_types'])

    def test_missing_easy_tier_cannot_pass(self):
        del self.candidate['native_tag_macro_f1']['EZ']
        self.assertFalse(numerical_gates(self.control,self.candidate,self.limits)['ez_hd'])

    def test_hold_requires_both_endpoints(self):
        self.assertEqual(type_match_count('hold',dict(note_20ms=dict(tp=3),hold_joint_20ms=dict(tp=0))),0)


if __name__=='__main__':unittest.main()
