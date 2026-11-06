"""Plain event-model condition interventions, bound to a completed D87 arm."""
import argparse
import json
from pathlib import Path
import time

import torch

from beatfcn.events.controls import raw_controls
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.event_conditions import encode_controls, in_band
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.audit_local_candidate_conditions import request_legality
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def load_model(plan_path, checkpoint):
    training = json.loads(plan_path.read_text())
    for source, expected in {**training['inputs_sha256'], **training['code_sha256']}.items():
        if sha(ROOT/source) != expected:
            raise ValueError('Frozen training source changed: '+source)
    result = json.loads((ROOT/training['output']).read_text())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (not result.get('complete') or result.get('error') or result['updates'] != 1000
            or saved['updates'] != 1000 or saved['config'] != training['config']
            or saved['plan_sha256'] != sha(plan_path) or result['plan_sha256'] != sha(plan_path)
            or saved.get('objective') != training['loss_mode'] or result['checkpoint_sha256'] != sha(checkpoint)):
        raise ValueError('Need completed, correctly bound 1000-update objective arm')
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    return model, training


def field_summary(rows):
    directed = sum(r['complete'] and not r['identical_token_stream'] and r['direction_delta'] is not None
        and r['direction_delta'] >= r['minimum_delta'] for r in rows)
    hits = sum(sum(r['in_band']) for r in rows)
    legal = all(not any(r['independent_legality']) for r in rows)
    return dict(pairs=len(rows), identical_pairs=sum(r['identical_token_stream'] for r in rows),
        sufficiently_directed_pairs=directed, in_band_outputs=hits,
        sentinel_passed=len(rows)==4 and all(r['complete'] for r in rows) and legal and directed>=3 and hits>=6)


@torch.no_grad()
def run(path):
    plan = json.loads(path.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError(output)
    for source, expected in plan['inputs_sha256'].items():
        if sha(ROOT/source) != expected:
            raise ValueError('Frozen condition input changed: '+source)
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    torch.set_num_threads(protocol['cpu_threads'])
    model, training = load_model(ROOT/plan['training_plan'], ROOT/plan['checkpoint'])
    features = LocalFeatureStore(ROOT, training['features'])
    for song in protocol['songs']:
        for source, expected in features.records[song['audio_sha256']]['sha256'].items():
            if sha(ROOT/source) != expected:
                raise ValueError('Audio feature changed: '+source)
    report = dict(id=plan['id'], plan_sha256=sha(path), complete=False, execution_complete=False,
        checkpoint=plan['checkpoint'], checkpoint_sha256=sha(ROOT/plan['checkpoint']),
        checkpoint_updates=1000, optimizer_updates=0, new_cloud_compute_calls=0, backend='cpu',
        rows=[], by_field={}, quality_passed=False,
        scope='Same four heldout cores; strict D68 per-field3/4 directed and6/8 in band. Stop after a failed field; missing fields never pass.')
    started = time.monotonic()
    try:
        for field in protocol['fields']:
            rows = []
            for song in protocol['songs']:
                a,b = song['scope_us']
                request = EventRequest('phira', a,b,features.records[song['audio_sha256']]['duration_us'])
                views=[]
                for band in (field['low'],field['high']):
                    inp,provenance=request_inputs(features,song['audio_sha256'],request,
                        encode_controls('phira',{field['name']:band},(a,b),origin_us=a))
                    views.append(dict(inputs=inp,sequence=teacher_sequence(encode_events([],request),request)))
                inputs,_=collate(views);inputs=pad_input_shapes(inputs,training['input_shapes'])
                if any(not torch.equal(v[0],v[1]) for k,v in inputs.items() if k!='controls'):
                    raise ValueError('A non-condition input differs between requests')
                predictions=generate_batch(model,inputs,[request,request],protocol['token_cap'],deadline=started+protocol['max_wall_s'])
                events=[[RhythmEvent(**e)for e in p['events']]for p in predictions]
                values=[raw_controls('phira',canonical(es,'phira'),[(a,b)])for es in events]
                realized=[v[field['name']] for v in values]
                complete=all(p['complete']for p in predictions)
                row=dict(chart_id=song['chart_id'],audio_sha256=song['audio_sha256'],scope_us=[a,b],field=field['name'],
                    low=field['low'],high=field['high'],minimum_delta=field['minimum_delta'],predictions=predictions,
                    realized=values,complete=complete,direction_delta=realized[1]-realized[0] if all(v is not None for v in realized) else None,
                    identical_token_stream=complete and predictions[0]['tokens']==predictions[1]['tokens'],
                    in_band=[in_band(field['name'],v,band)for v,band in zip(realized,(field['low'],field['high']))],
                    independent_legality=[request_legality(es,request)for es in events],reference_provenance=provenance)
                rows.append(row);report['rows'].append(row);report['wall_s']=time.monotonic()-started
                write_json(output,report)
            report['by_field'][field['name']]=field_summary(rows)
            if not report['by_field'][field['name']]['sentinel_passed']:
                report['stop_reason']='This required field failed; remaining condition expansion stopped.'
                break
        measured=set(report['by_field'])
        report.update(execution_complete=True,complete=len(measured)==len(protocol['fields']),
            unmeasured_fields=[f['name']for f in protocol['fields']if f['name']not in measured],
            expansion_allowed=len(measured)==len(protocol['fields']) and all(r['sentinel_passed']for r in report['by_field'].values()),
            wall_s=time.monotonic()-started)
        write_json(output,report)
        print(json.dumps({k:report[k]for k in ['complete','execution_complete','by_field','unmeasured_fields','expansion_allowed','wall_s']}))
    except Exception as exc:
        report.update(execution_complete=True,error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started)
        write_json(output,report)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    run(parser.parse_args().plan)
