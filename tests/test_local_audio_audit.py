import numpy as np
import unittest

from beatfcn.events.canonical import Event, LineLoc
from beatfcn.train.local_audio_audit import decoded_bounds, tail_coverage, tensor_health


def event(identifier, head, tail=None):
    return Event(identifier, head, head if tail is None else tail,
                 'tap' if tail is None else 'hold', LineLoc('0', 'above', 0.0))


class LocalAudioAuditTests(unittest.TestCase):
    def test_integer_decoded_bounds_do_not_use_rounded_duration(self):
        # 1001 samples is 41708.333... us, whose rounded metadata is only 41700 us.
        events = [event('real_tail', 41705), event('outside', 41709), event('hold', 100, 41708)]
        assert decoded_bounds(events, 1001)['examples'] == ['outside']
        assert decoded_bounds([event('at_end', 40000)], 960)['count'] == 1
        assert decoded_bounds([event('hold', 0, 40000)], 960)['count'] == 0


    def test_shift_and_missing_cache_tail_are_separate(self):
        events = [event('tail_tap', 41000), event('hold', 1000, 41700)]
        assert decoded_bounds(events, 1001)['count'] == 0
        assert decoded_bounds(events, 1001, 1000)['count'] == 2
        tail = tail_coverage(events, 1001)
        assert tail['heads_in_omitted_tail'] == tail['hold_ends_in_omitted_tail'] == 1
        assert tail['omitted_samples'] == 41
        assert tail['legacy_mel_frames'] == 4
        assert tail['complete_mel_frames'] == 8
        assert tail_coverage([], 960)['omitted_samples'] == 0


    def test_tensor_health_does_not_hide_nonfinite_or_empty_input(self):
        assert tensor_health(np.array([0., 1.], dtype=np.float32))['mean'] == .5
        for bad in (np.array([]), np.array([np.nan]), np.array([np.inf])):
            with self.assertRaises(ValueError):
                tensor_health(bad)
