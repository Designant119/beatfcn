"""Missing fourth-round output is not replaced by a prior successful round."""
import unittest
from scripts.review_complete_event_network import summary


class R17CompletenessTest(unittest.TestCase):
    def sample(self, count, complete=True):
        types={kind:dict(true=0,pred=0,matched=0) for kind in ('tap','hold','flick','drag')}
        return dict(cores=[dict(identity=dict(mode='mania_4k'),rounds=[dict(full_note_f1=.99,
            exact_onset_group_recall=.99,complete=complete,per_type=types) for _ in range(count)])])

    def test_first_round_cannot_be_final_or_revision_pass(self):
        result=summary(self.sample(1))['mania_4k']
        self.assertEqual(result['first_f1'],.99)
        self.assertIsNone(result['final_f1'])
        self.assertIsNone(result['final_group_recall'])
        self.assertFalse(result['all_four_rounds'])

    def test_incomplete_fourth_round_is_not_validated(self):
        self.assertIsNone(summary(self.sample(4,False))['mania_4k']['final_f1'])

    def test_four_complete_rounds_are_counted(self):
        result=summary(self.sample(4))['mania_4k']
        self.assertEqual(result['final_f1'],.99)
        self.assertTrue(result['all_four_rounds'])


if __name__=='__main__':unittest.main()
