import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.train.local_typed_canvas import encode_canvas,decode_target
from beatfcn.train.relative_canvas import relative_targets


class RelativeCanvasTests(unittest.TestCase):
    def test_native_copy_endpoint_and_absolute_origin_translation(self):
        request=EventRequest('phira',100_000_000,108_000_000,400_000_000,copy_times=(100_003_001,))
        events=[RhythmEvent(100_003_001,120_006_000,'hold'),RhythmEvent(100_004_000,100_010_000,'hold')]
        raw=encode_canvas(events,request);target={k:torch.from_numpy(v)[None] for k,v in raw.items()}
        rel=relative_targets(target,[request]);base=100000+torch.arange(200)[None,:,None]*40
        absolute=(rel['tail_block']*2000+rel['tail_cell']*40+rel['tail_ms'])+base
        holds=rel['slot_valid'][:,:,1]
        restored={k:v.clone() for k,v in target.items()}
        restored['tail_block']=torch.where(holds,absolute//2000,0)
        restored['tail_cell']=torch.where(holds,(absolute%2000)//40,0)
        restored['tail_ms']=torch.where(holds,absolute%40,0)
        self.assertEqual(decode_target({k:v[0].numpy() for k,v in restored.items()},request),sorted(events))
        shifted=EventRequest('phira',200_000_000,208_000_000,500_000_000,copy_times=(200_003_001,))
        shifted_events=[RhythmEvent(e.onset_us+100_000_000,e.end_us+100_000_000,e.type) for e in events]
        raw2=encode_canvas(shifted_events,shifted)
        rel2=relative_targets({k:torch.from_numpy(v)[None] for k,v in raw2.items()},[shifted])
        for name in ['tail_block','tail_cell','tail_ms']:torch.testing.assert_close(rel[name],rel2[name])


if __name__=='__main__':unittest.main()
