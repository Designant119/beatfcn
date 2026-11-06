"""CPU data-contract tests; no model forward, training, network or credentials."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.train.modal_event_dataset import (
    MertOnlyFeatureStore, ModalEventDataset, PublicDurationBatchSampler,
    collate_event_batch, event_array,
)
from scripts.prepare_modal_event_data import public_head_layout, validate_native_targets


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / 'artifacts/modal120m-data-v1/prepared-v3/manifest.json'


class SyntheticDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = 'a' * 64
        self.samples = 24100
        frames = (self.samples + 959) // 960 * 4
        cells = (self.samples + 959) // 960
        folder = self.root / 'mert'
        folder.mkdir()
        for layer in (12, 23):
            values = np.arange(cells * 1024, dtype=np.float32).reshape(cells, 1024) / 4096 + layer
            np.save(folder / f'L{layer}.npy', values.astype(np.float16))
        times = (np.arange(cells) * 960 + 360) / 24000
        intervals = np.stack((np.arange(cells) * 960,
                              np.minimum((np.arange(cells) + 1) * 960, self.samples)), -1) / 24000
        np.save(folder / 'times_s.npy', times)
        np.save(folder / 'intervals_s.npy', intervals)
        self.mel = np.arange(frames * 128, dtype=np.float32).reshape(frames, 128).astype(np.float16)
        np.save(self.root / 'mel.npy', self.mel)
        self.store = MertOnlyFeatureStore(self.root, [dict(audio_sha256=self.audio,
            decoded_samples=self.samples, duration_us=1004166,
            mert_path='mert', complete_mel='mel.npy')])

    def tearDown(self):
        self.temp.cleanup()

    def row(self, count):
        x = self.store.window(self.audio, 0, 1_000_000)
        x.update(mode=np.asarray(1, np.int64), controls=np.zeros((8, 6), np.float32),
            reference=np.zeros((1, 16), np.float32), reference_times=np.zeros(1, np.float32),
            reference_valid=np.ones(1, bool), copy_times_us=np.zeros(0, np.int64),
            copy_us=np.zeros(0, np.int64), known_carry_us=np.zeros((0, 4), np.int64),
            known_regions_us=np.zeros((0, 2), np.int64), head_us=np.arange(1000, dtype=np.int64)*1000,
            type_allowed=np.ones((1000, 4), bool), duration_us=np.asarray(1_004_166, np.int64))
        return dict(inputs=x, targets=np.tile(np.array([[1000,1000,0,-1]],np.int64),(count,1)),metadata={})

    def test_fine_frames_partial_tail_and_true_mert_intervals_survive(self):
        x = self.store.window(self.audio, 0, 1_000_000)
        np.testing.assert_array_equal(x['whole_mel'], self.mel.astype(np.float32))
        self.assertEqual(int(x['whole_mel_valid'].sum()), 101)
        self.assertEqual(len(x['whole_mel']), 104)
        self.assertAlmostEqual(float(x['whole_mel_times'][100]), (24000+24100)/48000, places=7)
        self.assertEqual(x['local_intervals'][-1,1],np.float32(self.samples/24000))
        self.assertTrue(np.all(x['whole_mel_intervals'][~x['whole_mel_valid'],0]
                               ==x['whole_mel_intervals'][~x['whole_mel_valid'],1]))

    def test_loader_never_reads_pupu_and_has_native_local_shape(self):
        original = np.load
        opened = []
        def tracked(path,*args,**kwargs):
            opened.append(str(path))
            return original(path,*args,**kwargs)
        with patch('numpy.load',side_effect=tracked):
            x = self.store.window(self.audio, 0, 1_000_000)
        self.assertEqual(x['local'].shape,(2,26,1,1024))
        self.assertEqual(len(opened),5)
        self.assertFalse(any('pca' in path or 'pupu' in path or 'L9' in path for path in opened))

    def test_417_targets_keep_all_512_slots_and_cannot_change_public_inputs(self):
        empty = collate_event_batch([self.row(0)])
        dense = collate_event_batch([self.row(417)])
        self.assertEqual(tuple(dense['targets'].shape),(1,512,4))
        self.assertEqual(int(dense['target_valid'].sum()),417)
        for key in empty['inputs']:
            self.assertTrue(torch.equal(empty['inputs'][key],dense['inputs'][key]),key)
        self.assertEqual(tuple(dense['inputs']['whole_mel'].shape),(1,4096,128))
        self.assertEqual(int(dense['inputs']['head_valid'].sum()),1000)
        with self.assertRaisesRegex(ValueError,'OVERFLOW'):
            collate_event_batch([self.row(513)])

    def test_copy_duplicates_hold_tail_and_public_regions(self):
        request = EventRequest(mode='phira',start_us=0,end_us=4000,duration_us=4100,
            copy_times=(500,4100),known_carry=())
        offsets, allowed = public_head_layout(request,((1000,2000),))
        self.assertEqual(offsets.tolist(),[0,500,2000,3000])
        self.assertTrue(allowed[:,1].all())
        events=[RhythmEvent(500,4100,'hold'),RhythmEvent(500,4100,'hold')]
        validate_native_targets(events,request)
        self.assertEqual(event_array(events).tolist(),[[500,4100,1,-1]]*2)
        with self.assertRaisesRegex(ValueError,'Fractional tail'):
            validate_native_targets(events,replace(request,copy_times=(500,)))
        with self.assertRaisesRegex(ValueError,'OVERFLOW'):
            validate_native_targets(events,replace(request,max_events=1))
        end_request=EventRequest(mode='phira',start_us=0,end_us=4100,duration_us=4100)
        offsets,allowed=public_head_layout(end_request)
        self.assertEqual(offsets[-1],4000)
        self.assertFalse(allowed[-1,1])


class PreparedRealDataTest(unittest.TestCase):
    @unittest.skipUnless(PREPARED.exists(),'Run the explicitly bounded CPU preparation first')
    def test_real_eight_views_public_input_parity_and_native_multisets(self):
        from scripts.run_local_evolution import load_data
        from beatfcn.train.event_density_drag_filter import EventDensityFilteredPanel, filtered_request_row
        from beatfcn.train.local_bundle import batch_rows
        torch.set_num_threads(1)
        data=ModalEventDataset(ROOT,str(PREPARED.relative_to(ROOT)))
        panel,old_row=load_data()
        filtered=EventDensityFilteredPanel(panel)
        selected=[]
        for view in ('draft','single_condition','all_conditions','random_groups','middle','continuation'):
            selected.append(next(i for i,r in enumerate(data.records) if r['split']=='train' and r['view']==view
                                 and panel.records[r['index']]['request']['start_us']>=4_000_000
                                 and r['request']['duration_us']<512_000_000))
        selected.append(next(i for i,r in enumerate(data.records) if r['split']=='dev' and r['view']=='draft'))
        # Dense raw windows include the >256 historical-capacity boundary.
        dense_index=max((i for i,r in enumerate(panel.records) if r['split']=='train' and r['mode']=='phira'),
                        key=lambda i:panel.records[i]['event_count'])
        selected.append(next(i for i,r in enumerate(data.records) if r['index']==dense_index and r['view']=='draft'))
        evidence=[]
        for i in selected:
            new=data[i];rec=new['metadata']
            raw=old_row(rec['index'],rec['original_view'],rec['field_index'])
            old=filtered_request_row(filtered,raw)
            old_x,_=batch_rows([old])
            new_x=collate_event_batch([new])['inputs']
            fields=[]
            for key,value in old_x.items():
                self.assertTrue(torch.equal(value,new_x[key]),f'{rec["index"]}:{rec["view"]}:{key}')
                fields.append(key)
            self.assertEqual(Counter(map(tuple,new['targets'])),Counter(map(tuple,event_array(old['targets']))))
            original_request=as_request_dict(old['request'])
            prepared_request=rec['request']
            changed={k:[original_request[k],prepared_request[k]] for k in original_request
                     if original_request[k]!=prepared_request[k]}
            self.assertEqual(set(changed),{'max_events','max_tokens'})
            self.assertEqual(prepared_request['max_events'],512)
            self.assertEqual(prepared_request['max_tokens'],8192)
            evidence.append(dict(index=rec['index'],view=rec['view'],fields=fields,
                targets=len(new['targets']),raw_targets=len(raw['targets']),public_fields_bitwise_equal=True,
                request_clock_carry_copy_equal=True,explicit_public_resource_changes=changed))
        path=PREPARED.parent.parent/'old-loader-parity-v3.json'
        path.write_text(json.dumps(dict(status='passed',cases=evidence,model_calls=0,optimizer_updates=0,cloud_calls=0),indent=2)+'\n')

    @unittest.skipUnless(PREPARED.exists(),'Run the explicitly bounded CPU preparation first')
    def test_prepared_coverage_support_no_runtime_filter_and_public_buckets(self):
        data=ModalEventDataset(ROOT,str(PREPARED.relative_to(ROOT)),split='train',views=['draft'])
        self.assertEqual(len(data),6699)
        self.assertEqual(len({r['chart_id'] for r in data.records}),324)
        with patch('beatfcn.train.event_density_drag_filter.dense_event_drag_mask',side_effect=AssertionError('Runtime filtering')):
            batch=collate_event_batch([data[0]])
        self.assertEqual(batch['targets'].shape[1],512)
        sampler=PublicDurationBatchSampler(data,16,shuffle=False)
        ids=[i for b in sampler for i in b]
        self.assertEqual(sorted(ids),list(range(len(data))))
        for b in sampler:
            self.assertEqual(len({data.public_audio_bucket(i) for i in b}),1)


def as_request_dict(request):
    from dataclasses import asdict
    return json.loads(json.dumps(asdict(request)))


if __name__=='__main__':
    unittest.main()
