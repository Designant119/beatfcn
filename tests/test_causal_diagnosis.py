import unittest

import torch

from beatfcn.train.causal_diagnosis import audio_intervention


def sample():
    return dict(local=torch.arange(12.).reshape(1, 1, 12, 1, 1),
                local_valid=torch.tensor([[True] * 10 + [False] * 2]),
                local_times=torch.arange(12.)[None] * .04,
                mel=torch.arange(48.).reshape(1, 48, 1),
                mel_valid=torch.tensor([[True] * 40 + [False] * 8]),
                mel_times=torch.arange(48.)[None] * .01,
                song=torch.arange(20.).reshape(1, 20, 1),
                song_valid=torch.ones(1, 20, dtype=torch.bool),
                song_times=(torch.arange(20.) - 10)[None])


def test_local_shift_preserves_padding_clock_masks_and_song():
    inputs = sample()
    out = audio_intervention(inputs, 'local_plus200')
    assert out['mel'][0, 0] == inputs['mel'][0, 20]
    assert torch.equal(out['mel'][0, 40:], inputs['mel'][0, 40:])
    assert torch.equal(out['local'][0, :, 10:], inputs['local'][0, :, 10:])
    for key in inputs.keys() - {'local', 'mel'}:
        assert torch.equal(out[key], inputs[key])
    assert inputs['mel'][0, 0] == 0


def test_remote_order_preserves_nearby_content_and_multiset():
    inputs = sample()
    out = audio_intervention(inputs, 'remote_reverse')
    nearby = (inputs['song_times'] >= -.5) & (inputs['song_times'] <= .89)
    assert torch.equal(out['song'][nearby], inputs['song'][nearby])
    assert torch.equal(out['song'].sort(dim=1).values, inputs['song'].sort(dim=1).values)
    for key in inputs.keys() - {'song'}:
        assert torch.equal(out[key], inputs[key])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in (
        test_local_shift_preserves_padding_clock_masks_and_song,
        test_remote_order_preserves_nearby_content_and_multiset))
