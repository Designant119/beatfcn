from types import SimpleNamespace
import unittest
import torch
from beatfcn.train.event_transport import pack_event_batch,EventTransport


class EventTransportTests(unittest.TestCase):
    def make(self):
        inp=dict(mode=torch.tensor([0]),controls=torch.zeros(1,8,6),local=torch.zeros(1,1,3,1,4),
                 song=torch.zeros(1,2,4),mel=torch.zeros(1,12,128),local_valid=torch.ones(1,3,dtype=torch.bool))
        seq=dict(tokens=torch.tensor([[1,2]]),targets=torch.tensor([[2,1]]),fields=torch.zeros(1,2,dtype=torch.long),
                 times=torch.zeros(1,2),valid=torch.ones(1,2,dtype=torch.bool),allowed=torch.ones(1,2,3,dtype=torch.bool))
        recipe=dict(keys=['a'],layers=[12],local_slices=[(1,3)],song_slices=[(0,2)],mel_slices=[(4,12)])
        p=pack_event_batch(inp,seq,recipe)
        class Pool:
            def __init__(self):self.pins=0
            def request(self,keys):self.pins+=1
            def release(self,keys):self.pins-=1
            def get(self,key):return SimpleNamespace(taps={12:torch.arange(20).reshape(5,4).float()},
                summary=torch.arange(8).reshape(2,4).float(),mel10=torch.ones(20,128))
            def telemetry(self):return {'pins':self.pins}
        return SimpleNamespace(batches=[(p,{})]*4,skip_batches=0),Pool(),seq

    def test_reuses_banks_and_exact_slices_without_audio_in_metadata(self):
        tape,pool,truth=self.make();p=tape.batches[0][0]
        self.assertNotIn('input:local',[r[0][0] for r in p['_layout']])
        stream=EventTransport(tape,pool,device='cpu',prefetch=2)
        pointers=[]
        for inputs,sequence,_ in stream:
            pointers.append(inputs['local'].data_ptr())
            torch.testing.assert_close(inputs['local'][0,0,:,0],torch.arange(4,16).reshape(3,4).float())
            self.assertTrue(torch.equal(sequence['targets'],truth['targets']))
        self.assertEqual(pointers[0],pointers[2]);self.assertEqual(pointers[1],pointers[3])
        self.assertNotEqual(pointers[0],pointers[1]);self.assertEqual(pool.pins,0)
        self.assertEqual(stream.stats['batches'],4)

    def test_early_close_releases_prefetched_song_references(self):
        tape,pool,_=self.make();stream=EventTransport(tape,pool,device='cpu',prefetch=3)
        next(stream);self.assertEqual(pool.pins,2);stream.close();self.assertEqual(pool.pins,0)

    def test_buffer_budget_is_checked_before_allocating(self):
        tape,pool,_=self.make()
        with self.assertRaises(ValueError):EventTransport(tape,pool,device='cpu',max_audio_buffer_bytes=1)

if __name__=='__main__':unittest.main()
