from collections import Counter
import unittest

from beatfcn.events.controls import MODE_FIELDS
from beatfcn.train.positive_pairs import matched_schedules


class PositivePairScheduleTests(unittest.TestCase):
    def pairs(self):
        return [dict(pair_id=f'{mode}:{i}',mode=mode,audio_sha256=f'{mode}-audio-{i}',
            changed_fields=list(fields)) for mode,fields in MODE_FIELDS.items() for i in range(9)]

    def test_equal_exposure_at_every_stopping_boundary_and_condition_mask(self):
        schedules=matched_schedules(self.pairs(),steps=1000,seed=71)
        key=lambda x:(x['pair_id'],x['side'],x['field'])
        for end in range(100,1001,100):
            counts=[Counter(key(x) for batch in schedules[arm][:end] for x in batch)
                    for arm in ('paired','unpaired')]
            self.assertEqual(*counts)
        for paired,control in zip(schedules['paired'],schedules['unpaired']):
            self.assertEqual(paired[0]['pair_id'],paired[1]['pair_id'])
            self.assertNotEqual(paired[0]['side'],paired[1]['side'])
            self.assertEqual(paired[0]['field'],paired[1]['field'])
            self.assertNotEqual(control[0]['audio_sha256'],control[1]['audio_sha256'])
            self.assertEqual(control[0]['field'],control[1]['field'])
            self.assertEqual(control[0]['mode'],control[1]['mode'])
        self.assertEqual(schedules,matched_schedules(self.pairs(),steps=1000,seed=71))

    def test_missing_support_is_not_silently_oversampled(self):
        with self.assertRaisesRegex(ValueError,'independent audio'):
            matched_schedules(self.pairs()[:3],steps=100)


if __name__=='__main__': unittest.main()
