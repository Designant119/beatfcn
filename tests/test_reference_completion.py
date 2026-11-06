from collections import Counter
import unittest

import numpy as np

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events
from beatfcn.train.reference_completion import TASKS, partition, sparse_reference


def example():
    return [RhythmEvent(1_000_000, 5_000_000, 'hold', -1),
            RhythmEvent(2_000_000, 2_000_000, 'tap', -1),
            RhythmEvent(2_000_000, 2_000_000, 'drag', -1),
            RhythmEvent(5_000_000, 6_000_000, 'hold', -1),
            RhythmEvent(7_000_000, 7_000_000, 'flick', -1)]


def test_masks_preserve_whole_events_groups_and_codec():
    request = EventRequest('phira', 0, 8_000_000, 10_000_000)
    for task in TASKS:
        target, visible, hidden, regions = partition(example(), request, task, 'song', 7)
        assert Counter(visible + hidden) == Counter(example())
        assert not ({e.onset_us for e in visible} & {e.onset_us for e in hidden})
        encode_events(hidden, target)
        if task == 'middle':
            assert target.start_us == 2_000_000
            assert target.known_carry[0].end_us == 5_000_000
        if task == 'random_groups':
            assert not regions


def test_sparse_reference_contains_only_visible_events_and_no_known_silence():
    request = EventRequest('phira', 0, 8_000_000, 10_000_000)
    visible = example()[:1]
    ref = sparse_reference(visible, request)
    assert len(ref['reference']) == 2
    assert ref['reference'][:, 14].sum() == 0
    assert ref['reference'][:, 15].sum() == 1
    assert ref['reference'][1, 3] == 1  # hold including authorized tail
    assert np.isclose(ref['reference_times'][1], 1.)
    assert np.array_equal(ref['reference'], sparse_reference(visible, request)['reference'])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in (
        test_masks_preserve_whole_events_groups_and_codec,
        test_sparse_reference_contains_only_visible_events_and_no_known_silence))
