import unittest
import numpy as np
from beatfcn.audio.mert2 import Mert2Spec, windows, pad_tail, fine_mel, validate_local_source, SR


class Mert2ClockTests(unittest.TestCase):
    def test_tail_is_not_cut_to_an_old_40ms_boundary(self):
        wave=np.zeros(961,dtype=np.float32);wave[-1]=1.
        padded=pad_tail(wave)
        self.assertEqual(len(padded),1920)
        self.assertEqual(padded[960],1.)
        mel=fine_mel(wave)
        self.assertEqual(mel['values'].shape,(5,128))
        self.assertAlmostEqual(mel['intervals_s'][-1,1],961/SR)
        self.assertAlmostEqual(mel['times_s'][0],.005)
        self.assertTrue(np.isfinite(mel['values']).all())
        self.assertTrue(mel['values'].flags.c_contiguous)

    def test_full_song_and_long_song_windows_cover_tail(self):
        s=Mert2Spec()
        self.assertEqual(windows(360*SR,s),[(0,360*SR)])
        self.assertEqual(windows(501*SR+1,s),[(0,300*SR),(200*SR,500*SR),(400*SR,501*SR+1)])
        self.assertEqual(windows(500*SR,s),[(0,300*SR),(200*SR,500*SR)])

    def test_invalid_signals_and_specs_fail(self):
        for wave in [[],[float('nan')],[[1.,2.]]]:
            with self.assertRaises(ValueError):pad_tail(wave)
        with self.assertRaises(ValueError):Mert2Spec(layers=(0,23))
        with self.assertRaises(ValueError):Mert2Spec(overlap_s=300)

    def test_pinned_local_source_is_exact_and_missing_sources_fail(self):
        import hashlib
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            expected={}
            for name in ('config.json','model.safetensors','configuration_mert2.py','modeling_mert2.py'):
                payload=(name+' payload').encode();(folder/name).write_bytes(payload)
                expected[name]=hashlib.sha256(payload).hexdigest()
            self.assertTrue(validate_local_source(folder,expected).endswith(str(folder)))
            (folder/'config.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'identity changed'):
                validate_local_source(folder,expected)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'lacks|directory'):validate_local_source(temporary)

if __name__=='__main__':unittest.main()
