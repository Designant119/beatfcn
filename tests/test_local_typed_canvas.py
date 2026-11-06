import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_typed_canvas import LocalTypedCanvas, best_tail, canvas_loss
from beatfcn.train.local_typed_canvas import request_layout, encode_canvas, decode_target


class TypedCanvasTests(unittest.TestCase):
    def test_exact_multiset_with_microsecond_copy_and_cross_window_tail(self):
        r=EventRequest('phira',2_000_000,2_080_000,9_000_000,copy_times=(2_003_001,2_003_002))
        e=[RhythmEvent(2_003_001,2_003_001,'flick'),RhythmEvent(2_003_002,2_003_002,'drag'),
           RhythmEvent(2_004_000,2_004_000,'tap'),RhythmEvent(2_004_000,2_004_000,'tap'),
           RhythmEvent(2_004_000,8_006_000,'hold'),RhythmEvent(2_005_000,2_011_000,'hold')]
        self.assertEqual(decode_target(encode_canvas(e,r),r),sorted(e))
        with self.assertRaisesRegex(ValueError,'capacity'):
            encode_canvas([RhythmEvent(2_000_000,2_000_000,'tap')]*17,r)
        with self.assertRaisesRegex(ValueError,'exact'):
            encode_canvas([RhythmEvent(2_003_003,2_003_003,'tap')],r)

    def test_joint_tail_matches_exhaustive_legality(self):
        rng=np.random.default_rng(7)
        for head,end in [(1_999_001,4_003_000),(6_198_000,6_204_001),(10,6_999_000)]:
            b,c,m=rng.normal(size=601),rng.normal(size=50),rng.normal(size=40)
            values=np.arange(head//1000+1,end//1000+1)
            scores=b[values//2000]+c[(values%2000)//40]+m[values%40]
            self.assertEqual(best_tail(b,c,m,head,end),int(values[scores.argmax()])*1000)

    def test_input_only_paths_gradients_and_density_monotonicity(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        cfg=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=48,encoder_layers=1,decoder_layers=0,
                            tap_dim=16,bands=4,dropout=0.,mel_bins=8)
        model=LocalTypedCanvas(cfg,density_reference=4).eval()
        r=EventRequest('phira',0,80_000,4_000_000)
        x=dict(local=torch.randn(1,3,2,4,16),local_times=torch.tensor([[.02,.06]]),local_valid=torch.ones(1,2,dtype=torch.bool),
               song=torch.randn(1,4,16),song_times=torch.arange(4)[None]*.5+.25,song_valid=torch.ones(1,4,dtype=torch.bool),
               mel=torch.randn(1,8,8),mel_times=torch.arange(8)[None]*.01+.005,mel_valid=torch.ones(1,8,dtype=torch.bool),
               reference=torch.zeros(1,1,16),reference_times=torch.zeros(1,1),reference_valid=torch.ones(1,1,dtype=torch.bool),
               controls=torch.zeros(1,8,6),mode=torch.tensor([1]))
        x.update({k:torch.from_numpy(np.asarray(v))[None] for k,v in request_layout(r).items()})
        target=encode_canvas([RhythmEvent(4_000,2_012_000,'hold'),RhythmEvent(5_000,5_000,'flick')],r)
        t={k:torch.from_numpy(v)[None] for k,v in target.items()}
        y=model(x);loss,_=canvas_loss(y,t,x);loss.backward()
        for n in ('quantity.weight','offset.weight','tail_block.weight','frame_read.weight'):
            g=dict(model.named_parameters())[n].grad
            self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0,n)
        x['controls'][0,0,:3]=torch.tensor([2/64,3/64,1])
        with torch.no_grad():low=model(x)
        x['controls'][0,0,:2]=torch.tensor([6/64,8/64])
        with torch.no_grad():high=model(x)
        torch.testing.assert_close(low['base_count'],high['base_count'])
        self.assertTrue((high['count'].argmax(-1)>=low['count'].argmax(-1)).all())
        for n in ('offset','tail_block','tail_cell','tail_ms'):
            torch.testing.assert_close(low[n],high[n])


if __name__=='__main__':unittest.main()
