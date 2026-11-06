"""D44: replay data corruption and saved visibility, never run a model."""
import ast
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import EMPTY
from beatfcn.model.group_canvas import new_group_canvas, masked_group_view, decode_group_targets
from scripts.matched_acoustic_data import make_example
from scripts.r09_local_data import file_sha, verify_files

ROOT = Path(__file__).resolve().parents[1]


def signature(events):
    return Counter((e['onset_us'], e['end_us'], e['type'], e['lane']) for e in events)


def head_time(layout, heads, block, group, copy_times=()):
    offset, copied = map(int, heads[block, group])
    return copy_times[copied] if copied >= 0 else layout.blocks[block][0] + offset*1000


def canvas_payload(canvas):
    payload = {}
    for block, group in zip(*np.nonzero(canvas.multiplicity > 0)):
        onset = head_time(canvas.layout, canvas.heads, block, group, canvas.copy_times)
        payload[(int(block), int(group))] = [dict(onset_us=onset,
            end_us=int(note[1])*1000 if note[0] == 1 else onset,
            type=TYPES[int(note[0])], lane=int(note[2]))
            for note in canvas.notes[block,group,:canvas.multiplicity[block,group]]]
    return payload


def saved_payload(saved, layout):
    sizes = np.asarray(saved['member_counts'], np.int64)
    heads = np.asarray(saved['raw_group_heads'], np.int64)
    by_head, owners = defaultdict(list), defaultdict(list)
    for event in saved['events']:
        by_head[event['onset_us']].append(event)
    for block, group in zip(*np.nonzero(sizes > 0)):
        owners[head_time(layout, heads, block, group)].append((int(block), int(group)))
    payload, unresolved = {}, []
    for head, positions in owners.items():
        if len(positions) != 1 or len(by_head[head]) != int(sizes[positions[0]]):
            unresolved.append(dict(onset_us=head, positions=[list(p) for p in positions],
                                   events=by_head[head], reason='member-to-group ownership not uniquely saved'))
            continue
        payload[positions[0]] = by_head[head]
    if not unresolved and signature([e for events in payload.values() for e in events]) != signature(saved['events']):
        raise ValueError('saved group reconstruction changed event multiset')
    return sizes, heads, payload, owners, unresolved


def draft_stats(sizes, heads, visible, payload, labels):
    true_payload = canvas_payload(labels)
    truth_heads = np.array([v[0]['onset_us'] for v in true_payload.values()], np.int64)
    positive = visible & (sizes > 0)
    true_positive = labels.multiplicity > 0
    available = [tuple(map(int,p)) for p in zip(*np.nonzero(positive))]
    resolved = [p for p in available if p in payload]
    events = [e for p in resolved for e in payload[p]]
    both = positive & true_positive
    near_slot, near_any = 0, 0
    for b,g in available:
        head = head_time(labels.layout, heads, b, g, labels.copy_times)
        if len(truth_heads) and np.min(np.abs(truth_heads-head)) <= 20_000:
            near_any += 1
        if true_positive[b,g] and abs(head-head_time(labels.layout, labels.heads,b,g,labels.copy_times)) <= 20_000:
            near_slot += 1
    known_true = [tuple(map(int,p)) for p in zip(*np.nonzero(visible & true_positive))]
    wrong_resolved = sum(signature(payload.get(p, [])) != signature(true_payload[p])
                         for p in known_true if sizes[p] <= 0 or p in payload)
    durations = [e['end_us']-e['onset_us'] for e in events if e['type']=='hold']
    return dict(true_groups=int(true_positive.sum()), visible_slots=int(visible.sum()),
                visible_positive_groups=int(positive.sum()), visible_notes=int(sizes[positive].sum()),
                resolved_positive_groups=len(resolved), resolved_visible_notes=len(events),
                unresolved_positive_groups=len(available)-len(resolved),
                extra_groups_at_target_empty_slots=int((positive & ~true_positive).sum()),
                visible_true_slots=len(known_true), wrong_resolved_visible_true_slots=wrong_resolved,
                both_predicted_and_true_positive_slots=int(both.sum()), same_slot_head_within20ms=near_slot,
                any_target_head_within20ms=near_any, types=dict(Counter(e['type'] for e in events)),
                hold_duration_us=durations, long_holds_ge8s=sum(d>=8_000_000 for d in durations))


def aggregate(rows):
    durations = [d for r in rows for d in r['stats']['hold_duration_us']]
    keys = ('visible_positive_groups','visible_notes','extra_groups_at_target_empty_slots',
            'resolved_positive_groups','unresolved_positive_groups','visible_true_slots',
            'wrong_resolved_visible_true_slots','long_holds_ge8s')
    total = {k:sum(r['stats'][k] for r in rows) for k in keys}
    npositive = total['visible_positive_groups']
    return dict(views=len(rows), totals=total,
        mean_visible_groups=total['visible_positive_groups']/len(rows) if rows else None,
        mean_visible_notes=total['visible_notes']/len(rows) if rows else None,
        extra_group_fraction=total['extra_groups_at_target_empty_slots']/npositive if npositive else None,
        any_target_head_within20ms_fraction=sum(r['stats']['any_target_head_within20ms'] for r in rows)/npositive if npositive else None,
        visible_type_counts=dict(sum((Counter(r['stats']['types']) for r in rows),Counter())),
        hold_duration_ms=dict(n=len(durations),median=float(np.median(durations))/1000 if durations else None,
                             p90=float(np.percentile(durations,90))/1000 if durations else None))


def lost_holds(core, previous, following, layout, visible, owners):
    truth = signature([e for e in core['target_events'] if e['type']=='hold'])
    before, after = signature(previous['events']) & truth, signature(following['events']) & truth
    records = []
    for event, multiplicity in (before-after).items():
        onset,end,kind,lane=event
        positions=owners[onset]
        known = len(positions)==1
        alternatives=[e for e in following['events'] if e['onset_us']==onset]
        if any(e['type']=='tap' and e['lane']==lane for e in alternatives):
            change='tap_at_same_head_and_lane'
        elif any(e['type']=='hold' and e['lane']==lane for e in alternatives):
            change='hold_tail_changed_at_same_head_and_lane'
        elif not alternatives:
            change='exact_head_absent'
        else:
            change='other_action_at_same_head'
        records.append(dict(identity=core['identity'], from_round=previous['round'], to_round=following['round'],
            exact_correct_hold=dict(onset_us=onset,end_us=end,type=kind,lane=lane), multiplicity=multiplicity,
            source_onset_us=onset+core['identity']['source_clip_start_us'],
            owner_resolved=known, positions=[list(p) for p in positions],
            was_visible=bool(visible[positions[0]]) if known else None,
            replacement_class=change, same_head_next_events=alternatives))
    return records


def main():
    started=time.perf_counter()
    path=ROOT/'bench/rhythm-v2/d44-draft-exposure-v1.json'
    plan=json.loads(path.read_text());verify_files(plan)
    destination=ROOT/plan['result']
    if destination.exists():raise FileExistsError('preserve previous audit')
    tree=ast.parse((ROOT/plan['corruption_source']).read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('blank_for','corruption')]
    if len(functions)!=2:raise ValueError('frozen corruption definitions changed')
    scope=dict(np=np,new_group_canvas=new_group_canvas,masked_group_view=masked_group_view,EMPTY=EMPTY)
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<frozen-corruption>','exec'),scope)
    manifest=json.loads((ROOT/plan['panel_manifest']).read_text());verify_files(manifest)
    panel=json.loads((ROOT/plan['panel_plan']).read_text())
    torch.set_num_threads(2)
    examples=[make_example(item,panel) for item in manifest['items']]
    reports={arm:json.loads((ROOT/name).read_text()) for arm,name in plan['reports'].items()}
    for arm,report in reports.items():
        if report['optimizer_updates']!=100 or not report['complete']:raise ValueError('incomplete source fitting')
        for (_,labels,identity), core in zip(examples,report['generation']['original']['cores']):
            if identity!=core['identity'] or [asdict(e) for e in decode_group_targets(labels)]!=core['target_events']:
                raise ValueError('source target or clock differs')
    training=[];rng=np.random.default_rng(plan['seed'])
    for step in range(100):
        for inputs,labels,identity in examples:
            view,kind=scope['corruption'](labels,inputs,step,rng)
            if any(r['history'][step]['view']!=kind for r in reports.values()):raise ValueError('corruption cycle differs')
            stats=draft_stats(view.multiplicity,view.heads,view.multiplicity>=0,canvas_payload(view),labels)
            training.append(dict(step=step+1,kind=kind,identity=identity,stats=stats))
    generated={};losses={};unresolved={}
    for arm,report in reports.items():
        rows=[];lost=[];uncertain=[]
        for (_,labels,_),core in zip(examples,report['generation']['original']['cores']):
            rng=np.random.default_rng(plan['seed'])
            for index,threshold in enumerate((.5,.25,.125)):
                previous,following=core['rounds'][index:index+2]
                sizes,heads,payload,owners,issues=saved_payload(previous,labels.layout)
                visible=rng.random(sizes.shape)>=threshold
                stats=draft_stats(sizes,heads,visible,payload,labels)
                rows.append(dict(identity=core['identity'],from_round=index+1,to_round=index+2,stats=stats,
                                 visible_positions=np.argwhere(visible).tolist()))
                uncertain.extend(dict(identity=core['identity'],round=index+1,**i) for i in issues)
                lost.extend(lost_holds(core,previous,following,labels.layout,visible,owners))
        generated[arm],losses[arm],unresolved[arm]=rows,lost,uncertain
    summaries={mode:dict(training={kind:aggregate([r for r in training if r['kind']==kind and r['identity']['mode']==mode])
                                  for kind in ('all_mask','reveal_half','synthetic_wrong_draft')},
                        generated={arm:aggregate([r for r in rows if r['identity']['mode']==mode]) for arm,rows in generated.items()})
               for mode in ('mania_4k','phira')}
    lost_summary={arm:dict(total=sum(r['multiplicity'] for r in rows),
        visible=sum(r['multiplicity'] for r in rows if r['was_visible'] is True),
        remasked=sum(r['multiplicity'] for r in rows if r['was_visible'] is False),
        unresolved=sum(r['multiplicity'] for r in rows if r['was_visible'] is None),
        replacements=dict(Counter(r['replacement_class'] for r in rows))) for arm,rows in losses.items()}
    result=dict(id='D44',complete=True,plan_sha256=file_sha(path),input_sha256=plan['input_sha256'],
        scope='data-only exact RNG replay and saved event multisets; no causal or quality-pass claim',
        training_views=training,generated_views=generated,summary=summaries,
        lost_exact_correct_holds=losses,lost_hold_summary=lost_summary,unresolved_group_ownership=unresolved,
        optimizer_updates=0,model_forwards=0,new_paid_compute_usd=0,model_promotion=False,
        wall_s=time.perf_counter()-started,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(summary=summaries,lost=lost_summary,
        unresolved={k:len(v) for k,v in unresolved.items()},wall_s=result['wall_s'],peak_rss_gib=result['peak_rss_gib'])))


if __name__=='__main__':main()
