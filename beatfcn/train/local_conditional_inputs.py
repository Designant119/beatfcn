"""Reuse frozen local probe identities for explicit native-prefix requests."""
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.train.charter_timing import ChartTimingTable, window_beat_features
from beatfcn.train.counterfactual_exposure import condition_inputs
from beatfcn.train.history_context import check_no_leak, visible_source
from beatfcn.train.local_request_panel import request_inputs
from scripts.evolve_surgery import file_sha
import scripts.run_benchmark_v2 as benchmark


class FrozenLocalInputs:
    def __init__(self, root, manifest, model):
        self.root, self.model = Path(root), model
        self.spec = json.loads(Path(manifest).read_text())
        self.allowed_splits = set(self.spec.get('allowed_splits',['train']))
        if not self.allowed_splits or not self.allowed_splits <= {'train','val'}:
            raise ValueError('Only explicitly frozen train/validation inputs are supported')
        self.frozen_rows = {r['chart_id']:r for r in self.spec['rows']}
        benchmark.ROOT = self.root
        self.store = benchmark.local_mert_store()
        self.timing = ChartTimingTable.load('/Users/yukoval/beatfcn-backup/data/chart_timing-v2.json',
            sha256='fdbe0e9a47d8e43d7260344c31708f2e54b299af102e2a87f179098e3f759558')
        self.rows = {}
        if 'feature_receipts' in self.spec:
            self.feature_receipts = self.spec['feature_receipts']
        else:
            evidence = self.root/'reports/workflow-pareto-20261105/local-history-probe/results.jsonl'
            self.feature_receipts = {r['chart_id']: r['features_sha256'] for r in map(json.loads,evidence.read_text().splitlines())}

    def build(self, item, band=None):
        cid = item['chart_id']
        if item['split'] not in self.allowed_splits or item != self.frozen_rows.get(cid):
            raise ValueError('Input is outside the frozen split/row identity')
        if cid not in self.rows:
            path = Path(item['chart_path'])
            if file_sha(path) != item['chart_sha256']:
                raise ValueError('Chart identity changed')
            target = path.with_name(path.name.replace('-chart.json','-targets.npz'))
            if file_sha(target) != item['target_sha256']:
                raise ValueError('Target identity changed')
            if not self.feature_receipts.get(cid):
                raise ValueError('Missing explicit feature identities')
            for path, digest in self.feature_receipts[cid].items():
                if file_sha(path) != digest:
                    raise ValueError('Feature identity changed')
            with np.load(target) as source:
                arrays = {k: source[k].copy() for k in source.files}
            native = [RhythmEvent(int(a),int(z),TYPES[int(k)],int(l)) for a,z,k,l in
                      zip(arrays['onset_us'],arrays['end_us'],arrays['kind'],arrays['lane'])]
            t = item['start_us']
            carry = tuple(e for e in native if e.type=='hold' and e.onset_us<t<e.end_us)
            req = EventRequest(item['mode'],t,t+8_000_000,item['duration_us'],known_carry=carry)
            source = visible_source(arrays,mode=item['mode'],audio_sha256=item['audio_sha256'],
                role='model_history',region=(t-8_000_000,t),source_id=cid)
            check_no_leak((source,),req)
            if not self.store.available(item['audio_sha256']):
                raise ValueError('Missing frozen features')
            self.store.ensure(item['audio_sha256'])
            conditions = condition_inputs(req.mode,req)
            inputs, provenance = request_inputs(self.store,item['audio_sha256'],req,conditions['controls'],sources=(source,))
            inputs['beat'] = window_beat_features(inputs,t,self.timing.segments(cid))
            if not all(np.isfinite(v).all() for v in inputs.values()):
                raise ValueError('Nonfinite model input')
            self.rows[cid] = (inputs,req,provenance)
        inputs,req,provenance = self.rows[cid]
        conditions = condition_inputs(req.mode,req,bands={} if band is None else {'hold_head_ratio':band})
        return dict(inputs,**conditions),req,provenance

    def sequence(self, events, request):
        return teacher_sequence(encode_events([RhythmEvent(**e) for e in events],request,self.model.vocab),request,self.model.vocab)
