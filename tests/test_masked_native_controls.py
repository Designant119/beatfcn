import unittest
import torch

from tests.test_masked_native_events import toy_inputs
from scripts.run_masked_native_controls import same_song_local_roll,full_supported
from scripts.run_native_cell_recovery import zero_audio,input_digest


class MatchedControlTests(unittest.TestCase):
    def test_roll_preserves_clocks_validity_song_and_zero_identity_cpu_mps(self):
        for device in ('cpu','mps'):
            if device=='mps' and not torch.backends.mps.is_available():continue
            x=toy_inputs(device);y=same_song_local_roll(x)
            for name in ('local','mel','song'):
                self.assertTrue(torch.equal(x[name+'_times'],y[name+'_times']))
                self.assertTrue(torch.equal(x[name+'_valid'],y[name+'_valid']))
            self.assertTrue(torch.equal(x['song'],y['song']))
            self.assertFalse(torch.equal(x['mel'],y['mel']))
            self.assertEqual(input_digest(zero_audio(x)),input_digest(zero_audio(y)))

    def test_foreign_full_range_eligibility(self):
        x=toy_inputs();self.assertTrue(all(v==1 for v in full_supported(x,x).values()))
        y={k:v.clone() for k,v in x.items()};y['song_valid'][0,4:]=False
        self.assertLess(full_supported(x,y)['song'],1.)


if __name__=='__main__':unittest.main()
