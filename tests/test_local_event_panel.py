import random
import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_panel import collate
from scripts.run_local_learning_curve import condition_view
from tests.test_full_event_model import inputs


class LocalPanelTests(unittest.TestCase):
    def test_mixed_length_padding_preserves_logits_and_equal_request_loss(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        config=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=48,encoder_layers=1,
            decoder_layers=1,tap_dim=16,bands=4,dropout=0.)
        model=FullEventModel(config).eval()
        rows=[]
        for count in (1,3):
            request=EventRequest('mania_4k',0,8_000_000,10_000_000)
            events=[RhythmEvent(i*1_000_000,i*1_000_000,'tap',i%4) for i in range(count)]
            row_inputs={k:v[0].numpy() for k,v in inputs(config).items()}
            if count==1:
                for key in ('song','song_times','song_valid'):
                    row_inputs[key]=row_inputs[key][:3]
            rows.append(dict(inputs=row_inputs,sequence=teacher_sequence(encode_events(events,request),request)))
        batched,sequence=collate(rows)
        with torch.no_grad():
            joint=model(batched,sequence)
            losses=[]
            for i,row in enumerate(rows):
                x={k:torch.from_numpy(np.asarray(v))[None] for k,v in row['inputs'].items()}
                y={k:torch.from_numpy(v)[None] for k,v in row['sequence'].items()}
                single=model(x,y)
                torch.testing.assert_close(joint[i,:single.shape[1]],single[0],atol=2e-5,rtol=2e-5)
                losses.append(event_loss(single,y))
            torch.testing.assert_close(event_loss(joint,sequence),torch.stack(losses).mean())
        self.assertFalse(sequence['valid'][:,max(len(r['sequence']['valid']) for r in rows):].any())

    def test_condition_dropout_preserves_applicability_scope_and_visible_values(self):
        controls=np.zeros((8,6),np.float32)
        controls[:7,:]=[.2,.4,1,1,0,.01]
        row={'controls':controls}
        rng=random.Random(9)
        counts={}
        for _ in range(100):
            result,kind=condition_view(row,rng)
            counts[kind]=counts.get(kind,0)+1
            np.testing.assert_array_equal(result[:,3:],controls[:,3:])
            visible=result[:,2].astype(bool)
            np.testing.assert_array_equal(result[visible],controls[visible])
            self.assertTrue((result[~visible,:3]==0).all())
            self.assertEqual(result[7].sum(),0)
        self.assertEqual(set(counts),{'none','single','subset','all'})


if __name__=='__main__':unittest.main()
