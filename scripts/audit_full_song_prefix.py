"""Frozen SSD1000: genuine whole-song draft versus a16s human prefix.

CPU, zero updates, two songs fixed by existing dev index order. Eight-second
non-AR blocks cover the entire admitted song, including its partial tail.
The same explicitly visible human prefix conditions every suffix block;
generated history is not relabeled as human input. No hidden suffix enters
controls, forward inputs or decoding. This is a baseline diagnostic, not
selection of a new checkpoint or evidence of spatial playability.
"""
from dataclasses import asdict,replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.local_song_rollout import canonical
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.event_reference import VisibleReference
from beatfcn.train.local_bundle import annotated_sequence
from beatfcn.train.local_request_panel import event_request,request_inputs
from beatfcn.train.relative_canvas import relative_batch,generate_relative
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import sha,write
from scripts.run_local_typed_canvas import score


ROOT=Path(__file__).resolve().parents[1];IDENT='local-full-song-prefix-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_full_song_prefix_v1.json';ART=ROOT/f'artifacts/{IDENT}'


@torch.inference_mode()
def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve whole-song audit')
    torch.set_num_threads(2)
    oldpath=ROOT/'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json';parent=json.loads(oldpath.read_text())
    weight=ROOT/'artifacts/local-portfolio-v1/mamba-quality/mamba/quality-1000.pt'
    plan=dict(id=IDENT,checkpoint_sha256=sha(weight),parent_plan_sha256=sha(oldpath),indices=parent['indices']['dev'][:4:2],
        prefix_end_us=16_000_000,block_us=8_000_000,wall_s=900,optimizer_updates=0,device='cpu',cloud_compute_calls=0,
        selection='First two distinct dev components in the already-frozen order; not chosen by score. FrozenSSD1000 reference, not a newly selected winner.',
        input='Full-song frozenMERT plus commonmel. Draft has no reference. Prefix mode copies only native events with onset<16s; the complete16s visible region is explicitly authorized for this benchmark. Every later block sees that same prefix and only its supplied incoming holds. No generated-history feedback or hidden suffix properties.',
        primary='Score suffix[16s,song_end) for both modes; whole draft also reported. All native repeated events retained and prefix copy exact. Structural completion is distinct from old token-budget serialization.',
        boundaries='Two reused dev songs, blockwise nonAR whole-song inference, no human editing/listening/playtest or proof of whole-song musical coherence.',
        source_sha256={n:sha(ROOT/n) for n in ['scripts/audit_full_song_prefix.py','beatfcn/model/local_mamba_canvas.py','beatfcn/train/event_reference.py','beatfcn/train/relative_canvas.py']})
    write(PLAN,plan);panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    model=MambaCanvas(FullEventConfig(**parent['model']),density_reference=parent['density_reference']).eval()
    saved=torch.load(weight,map_location='cpu',weights_only=False)
    if saved['updates']!=1000 or saved['plan_sha256']!=sha(oldpath):raise ValueError('Exact frozen reference identity')
    model.load_state_dict(saved['model']);del saved
    report=dict(id=IDENT,status='running',songs=[],optimizer_updates=0,cloud_compute_calls=0,confirmation_scored=False);write(REPORT,report);started=time.monotonic()
    try:
        for index in plan['indices']:
            record=panel.records[index];original=event_request(record['request']);duration=original.duration_us
            _,values=panel._arrays(record['chart_index'])
            truth=[RhythmEvent(int(h),int(t),TYPES[int(k)],int(l)) for h,t,k,l in zip(*(values[n] for n in ('onset_us','end_us','kind','lane')))]
            if any(not 0<=e.onset_us<duration or e.end_us>duration for e in truth):raise ValueError('Full native chart not in admitted song domain; do not filter')
            prefix=[e for e in truth if e.onset_us<plan['prefix_end_us']];hidden=[e for e in truth if e.onset_us>=plan['prefix_end_us']]
            summary=dict(index=index,chart_id=record['chart_id'],component=record['component'],audio_sha256=record['audio_sha256'],
                duration_us=duration,target_events=len(truth),prefix_events=len(prefix),modes={})
            for mode in ('draft','human_prefix'):
                events=[] if mode=='draft' else list(prefix);blocks=[]
                beginning=0 if mode=='draft' else plan['prefix_end_us']
                for start in range(beginning,duration,plan['block_us']):
                    if time.monotonic()-started>plan['wall_s']:raise TimeoutError('Whole-song audit wall')
                    end=min(start+plan['block_us'],duration);visible=prefix if mode=='human_prefix' else []
                    carry=tuple(e for e in visible if e.type=='hold' and e.onset_us<start<e.end_us)
                    copies=tuple(sorted({t for e in visible for t in (e.onset_us,e.end_us) if t%1000}))
                    request=replace(original,start_us=start,end_us=end,known_carry=carry,copy_times=copies)
                    controls=encode_controls('phira',{},(start,end),origin_us=start)
                    sources=() if not visible else (VisibleReference(source_id=f'benchmark16s:{record["chart_id"]}',mode='phira',
                        audio_sha256=record['audio_sha256'],role='human_input',regions=((0,plan['prefix_end_us']),),events=tuple(canonical(visible,'phira'))),)
                    inputs,provenance=request_inputs(features,record['audio_sha256'],request,controls,sources=sources)
                    e=dict(index=index,view='whole_song_'+mode,inputs=inputs,request=request,sequence=annotated_sequence([],request),
                        origin_s=start/1e6,targets=[],visible=visible)
                    inp,_=relative_batch([e],'cpu');prediction=generate_relative(model,inp,request,deadline=started+plan['wall_s'])
                    generated=[RhythmEvent(**v) for v in prediction['events']]
                    if not prediction['complete']:raise ValueError('Whole-song block incomplete: '+str(prediction.get('failure')))
                    if any(not start<=v.onset_us<end or v.end_us>duration for v in generated):raise ValueError('Whole-song illegal block output')
                    events.extend(generated);blocks.append(dict(start_us=start,end_us=end,reference_rows=int(inp['reference_valid'].sum()),
                        supplied_carry=len(carry),prediction=prediction,provenance=provenance))
                    write(ART/f'{index}-{mode}-blocks.json',blocks)
                events.sort();suffix=[v for v in events if v.onset_us>=plan['prefix_end_us']]
                if mode=='human_prefix' and [v for v in events if v.onset_us<plan['prefix_end_us']]!=sorted(prefix):raise ValueError('Prefix multiset was rewritten')
                shifts=[match_f1([plan['prefix_end_us']+(v.onset_us-plan['prefix_end_us']+d)%(duration-plan['prefix_end_us']) for v in suffix],
                    [v.onset_us for v in hidden],10000)[3] for d in parent['shifts_us']]
                result=dict(complete=True,events=[asdict(v) for v in events],target=[asdict(v) for v in truth],prefix_copy_exact=True if mode=='human_prefix' else None,
                    suffix_scores=score(suffix,hidden),whole_scores=score(events,truth),suffix_shift_onset10=shifts,generated_blocks=len(blocks),
                    predicted_events=len(events),predicted_suffix_events=len(suffix),target_suffix_events=len(hidden))
                write(ART/f'{index}-{mode}.json',result);summary['modes'][mode]={k:v for k,v in result.items() if k not in ('events','target')}
            report['songs'].append(summary);write(REPORT,report)
        report.update(status='completed',wall_s=time.monotonic()-started,checkpoint_unchanged=sha(weight)==plan['checkpoint_sha256']);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',failure=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':main()
