import unittest

import torch

from beatfcn.model.direct_song_canvas import DirectSongCanvas
from beatfcn.model.local_mamba_canvas import MambaCanvas
from tests.test_local_mamba_canvas import small_config,synthetic_inputs
from tests.test_local_event_package import RejectTargetReads


class DirectSongTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_no_new_parameters_and_disabled_path_exactly_equals_original(self):
        torch.manual_seed(41);baseline=MambaCanvas(small_config(),density_reference=4).eval()
        torch.manual_seed(41);candidate=DirectSongCanvas(small_config(),density_reference=4).eval()
        self.assertEqual(set(baseline.state_dict()),set(candidate.state_dict()))
        for k,v in baseline.state_dict().items():self.assertTrue(torch.equal(v,candidate.state_dict()[k]),k)
        inputs,_=synthetic_inputs();candidate.direct_song=False
        with torch.no_grad():a,b=baseline(inputs),candidate(inputs)
        for k in a:self.assertTrue(torch.equal(a[k],b[k]),k)
        candidate.direct_song=True;candidate.direct_song_strength=0.
        with torch.no_grad():b=candidate(inputs)
        for k in a:self.assertTrue(torch.equal(a[k],b[k]),k)

    def test_song_audio_reaches_queries_without_target_fields(self):
        torch.manual_seed(41);model=DirectSongCanvas(small_config(),density_reference=4).eval()
        inputs,_=synthetic_inputs();out=model(RejectTargetReads(inputs))
        changed={**inputs,'song':torch.zeros_like(inputs['song'])}
        other=model(changed)
        self.assertGreater(float((out['count']-other['count']).abs().max()),1e-5)
        out['count'].square().mean().backward()
        grad=model.song_projection.weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),1e-6)

    def test_invalid_padding_does_not_enter_direct_read(self):
        torch.manual_seed(41);model=DirectSongCanvas(small_config(),density_reference=4).eval()
        inputs,_=synthetic_inputs();changed={k:v.clone() for k,v in inputs.items()}
        changed['song'][~changed['song_valid']]=float('nan')
        changed['song_times'][~changed['song_valid']]=float('nan')
        with torch.no_grad():a,b=model(inputs),model(changed)
        for k in a:torch.testing.assert_close(a[k],b[k],atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
