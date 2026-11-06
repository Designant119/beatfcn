"""Validate pinned local packages and prepare a larger diagnostic song panel."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import time
import zipfile

from beatfcn.events.phira import load_package
from beatfcn.events.osu_mania import parse_osu
from beatfcn.events.canonical import UnsupportedFormat
from beatfcn.model.action_state import encode, State, apply_action
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.ingest.coverage import phira_native_difficulty, native_tier
from scripts.audit_plain_packages import package_members
from scripts.prepare_action_state_batch import profile, score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def event_key(e):
    return e['event_id'], e['onset_us'], e['end_us'], e['type'], json.dumps(e['location'], sort_keys=True)


def raw_chart(item, manifest):
    canon = json.loads((ROOT / item['canonical']).read_text())
    if item['mode'] == 'phira':
        path = ROOT / item['audio_or_archive']
        info, members = package_members(path)
        assert sha256(members['chart']['bytes']).hexdigest() == canon['source_sha256'], 'raw_chart_hash'
        assert sha256(members['music']['bytes']).hexdigest() == item['audio_sha256'], 'package_audio_hash'
        chart = load_package(path)
        declared, reason = phira_native_difficulty({'level': info.get('level')})
        if declared is not None:
            assert abs(declared - item['native_difficulty']) <= 1, 'package_difficulty_conflict'
        evidence = dict(package=str(path.relative_to(ROOT)), package_sha256=digest(path),
            music_member=members['music']['path'], chart_member=members['chart']['path'],
            package_level=info.get('level'), package_format=info.get('format'),
            raw_note_count=sum(len(line.get(side) or []) for line in json.loads(members['chart']['bytes'].decode('utf-8-sig')).get('judgeLineList', [])
                for side in (('notes',) if chart.source_format=='phira_rpe' else ('notesAbove','notesBelow'))),
            native_authorship='public_phira_upload; provenance and operation-domain review not proof of human authorship')
    else:
        path = ROOT / 'inputs/data_expansion_v1/corpus/raw/osu/packs' / (str(manifest['set_id'])+'.zip')
        matches = []
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if not entry.filename.lower().endswith('.osu'):
                    continue
                data = archive.read(entry)
                if sha256(data).hexdigest() == canon['source_sha256']:
                    matches.append((entry.filename, data))
        assert len(matches) == 1, 'raw_osu_source_missing_or_ambiguous'
        chart = parse_osu(matches[0][1], source_name=matches[0][0])
        assert digest(item['audio_or_archive']) == item['audio_sha256'], 'mania_audio_hash'
        evidence = dict(package=str(path.relative_to(ROOT)), package_sha256=digest(path), chart_member=matches[0][0],
            native_difficulty_evidence=manifest['acquisition'])
    assert chart.mode == item['mode'] and chart.source_sha256 == canon['source_sha256']
    assert Counter(event_key(e.to_json()) for e in chart.events) == Counter(event_key(e) for e in canon['events']), 'canonical_events_drift'
    assert Counter(event_key(e.to_json()) for e in chart.fakes) == Counter(event_key(e) for e in canon['fakes']), 'canonical_fakes_drift'
    assert not [i for i in chart.issues if i.severity == 'quarantine'], 'raw_parser_quarantine'
    if 'raw_note_count' in evidence:
        assert evidence['raw_note_count'] == len(chart.events) + len(chart.fakes)
    evidence.update(playable_events=len(chart.events), preserved_fakes=len(chart.fakes),
        type_counts=dict(Counter(e.type for e in chart.events)), raw_source_sha256=canon['source_sha256'])
    return canon, evidence


def choose_core(canon, item, audio_sec):
    shift = item['frozen_clock_shift_us']
    mode = item['mode']
    events = sorted(RhythmEvent(e['onset_us']+shift, e['end_us']+shift, e['type'],
        e['location']['lane'] if mode == 'mania_4k' else -1) for e in canon['events'])
    heads = [e.onset_us for e in events]
    holds = [e for e in events if e.type == 'hold']
    best = None
    rejections = Counter()
    for start in range(8_000_000, round(audio_sec * 1e6) - 24_000_000, 100_000):
        if any(e.onset_us < start <= e.end_us for e in holds):
            continue
        chosen = events[bisect_left(heads, start):bisect_left(heads, start+8_000_000)]
        if not chosen or max(e.end_us for e in chosen) > start+24_000_000:
            continue
        try:
            support = profile(chosen, mode)
        except ValueError as exc:
            rejections[str(exc)] += 1
            continue
        if support['actions'] > 256 or support['maximum_active_hold_start_groups'] > 8 or support['maximum_per_type_head_count'] > 8:
            rejections['model_capacity'] += 1
            continue
        rank = (score(support, mode), -start)
        if best is None or rank > best[0]:
            best = (rank, start, support, chosen)
    if best is None:
        return None, dict(rejections)
    _, start, support, chosen = best
    clip_start = start - 8_000_000
    target = tuple(replace(e, onset_us=e.onset_us-clip_start, end_us=e.end_us-clip_start) for e in chosen)
    actions = encode(target, mode)
    prefix = []
    state = State(mode)
    for action in actions:
        prefix.append(asdict(state))
        state = apply_action(state, action).state
    assert not state.active_holds
    return dict(source_core_start_us=start, source_clip_start_us=clip_start, support=support,
        core_bounds_us=[8_000_000,16_000_000], clip_duration_us=32_000_000, empty_initial_state=True,
        target_events=[asdict(e) for e in target], target_actions=[asdict(a) for a in actions],
        teacher_forcing_prefix_states=prefix), dict(rejections)


def allocate(rows, base, count):
    used_audio = {r['audio_sha256'] for r in base}
    used_groups = {r['song_group'] for r in base}
    choices = defaultdict(dict)
    for r in sorted(rows, key=lambda r: sha256((r['chart_id']+r['audio_sha256']).encode()).hexdigest()):
        if r['audio_sha256'] in used_audio or r['song_group'] in used_groups:
            continue
        choices[r['mode'], r['tier']].setdefault(r['song_group'], r)
    slots = [(mode, tier, k) for mode in ['mania_4k','phira'] for tier in ['low','medium','high'] for k in range(count-1)]
    assigned = {}
    def augment(slot, seen):
        for group in choices[slot[:2]]:
            if group in seen:
                continue
            seen.add(group)
            if group not in assigned or augment(assigned[group], seen):
                assigned[group] = slot
                return True
        return False
    for slot in sorted(slots, key=lambda s:(len(choices[s[:2]]),s)):
        if not augment(slot, set()):
            return None, {'unfilled_slot':slot, 'eligible_groups':{str(k):len(v) for k,v in choices.items()}}
    additions = [choices[slot[:2]][group] for group,slot in sorted(assigned.items(), key=lambda x:x[1])]
    all_rows = base+additions
    assert len({r['audio_sha256'] for r in all_rows}) == len(all_rows), 'cross_group_audio_duplicate'
    return all_rows, None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    for path, expected in plan['input_sha256'].items():
        assert digest(path) == expected, path
    result_path = ROOT / plan['result']; assert not result_path.exists()
    output = ROOT / plan['output']; output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads((ROOT / plan['inventory']).read_text())
    manifests = {r['chart_id']:r for r in (json.loads(line) for line in (ROOT / plan['manifest']).read_text().splitlines())}
    original = json.loads((ROOT / plan['original_panel']).read_text())
    r25 = json.loads((ROOT / plan['original_training_plan']).read_text())
    base_old = [r for r in original['rows'] if r['id'] in r25['train_ids']]
    held = [r for r in original['rows'] if r['id'] in r25['heldout_ids']]
    blocked_audio = {r['audio_sha256'] for r in manifests.values() if r['split'] != 'train'} | {r['audio_sha256'] for r in held}
    blocked_group = {r['song_group'] for r in manifests.values() if r.get('audio_sha256') in blocked_audio}
    report = dict(id='D54', complete=False, passed=False, optimizer_updates=0, new_cloud_spend_usd=0,
        model_promotion=False, formal_training_ready=False, plan_sha256=digest(str(args.plan)), rows=[], failures=[])
    start = time.time()
    def save():
        report['wall_s'] = time.time()-start; write(result_path,report)
    save()
    accepted=[]
    for i,item in enumerate(inventory['rows']):
        rec=dict(chart_id=item['chart_id'],mode=item['mode'],tier=item['tier'],passed=False)
        try:
            old=manifests[item['chart_id']]
            assert old['split']=='train' and item['audio_sha256'] not in blocked_audio and item['song_group'] not in blocked_group
            assert digest(item['canonical']) == item['canonical_sha256']
            canon,evidence=raw_chart(item,old)
            inherited=next((r for r in base_old if r['chart_id']==item['chart_id']),None)
            if inherited is not None:
                core={k:inherited[k] for k in ['source_core_start_us','source_clip_start_us','support','core_bounds_us','clip_duration_us',
                    'empty_initial_state','target_events','target_actions','teacher_forcing_prefix_states']}
                rejected={}
            else:
                core,rejected=choose_core(canon,item,old['audio_sec'])
            assert core is not None,'no_supported_full_core'
            value={**item,**core,'declared_audio_sec':old['audio_sec'],'split':'train',
                'title':old['title'],'raw_evidence':evidence,'candidate_core_rejections':rejected,
                'clock_shift_us':item['frozen_clock_shift_us'],'offset_decision':old['offset_decision']}
            value['whole_song_measured_controls']=old.get('measured_controls')
            accepted.append(value);rec.update(passed=True,source_evidence=evidence,support=core['support'])
        except (AssertionError,ValueError,KeyError,OSError,zipfile.BadZipFile,UnsupportedFormat) as error:
            rec.update(error_type=type(error).__name__,error=str(error));report['failures'].append(rec)
        report['rows'].append(rec)
        if (i+1)%10==0:
            save();print(json.dumps(dict(audited=i+1,accepted=len(accepted))),flush=True)
    base=[]
    for old in base_old:
        item=next((r for r in accepted if r['chart_id']==old['chart_id']),None)
        if item is None:
            report['failures'].append(dict(error='missing_original_training_core',chart_id=old['chart_id']))
        else:base.append(item)
    selected,error=allocate(accepted,base,plan['per_cell']) if len(base)==6 else (None,{'base_incomplete':True})
    report['selection_error']=error
    if selected is not None:
        selected_rows=[]
        for i,item in enumerate(selected):
            old=next((r for r in base_old if r['chart_id']==item['chart_id']),None)
            if old:
                row=dict(old);row['song_group']=item['song_group'];row['origin']='frozen_r25_training_core'
            else:
                row=dict(item,id=f'd54-{i:02d}',origin='new_independent_song')
                if item['mode']=='phira':
                    info,members=package_members(ROOT/item['audio_or_archive'])
                    ext=Path(members['music']['path']).suffix.lower()
                    assert ext in ['.mp3','.ogg','.wav','.flac','.m4a','.opus','.aac'],ext
                    audio=output/'audio'/(item['audio_sha256']+ext);audio.parent.mkdir(exist_ok=True)
                    data=members['music']['bytes'];assert sha256(data).hexdigest()==item['audio_sha256']
                    audio.write_bytes(data);row['audio']=str(audio.relative_to(ROOT))
                else:row['audio']=item['audio_or_archive']
            row['split']='train';selected_rows.append(row)
        assert len(selected_rows)==36
        assert Counter((r['mode'],r['tier']) for r in selected_rows)==Counter({(m,t):6 for m in ['mania_4k','phira'] for t in ['low','medium','high']})
        result_panel=dict(id='D54',rows=selected_rows,heldout_rows=held,conditions='all unknown; no hidden controls',
            training_ready=False,scope='36-song bounded diagnosis only; official quota and authorship admission not established')
        write(output/'panel.json',result_panel)
        audio_inputs=dict(rows=[{k:r[k] for k in ['chart_id','audio','audio_sha256']} for r in selected_rows if r['origin']=='new_independent_song'])
        write(output/'audio-inputs.json',audio_inputs)
        report.update(passed=True,selected_charts=36,selected_unique_audio=36,new_audio_inference_songs=30,
            selected_events=sum(len(r['target_events']) for r in selected_rows),
            panel=str((output/'panel.json').relative_to(ROOT)),panel_sha256=digest(output/'panel.json'),
            audio_inputs=str((output/'audio-inputs.json').relative_to(ROOT)))
    report.update(complete=True,accepted_candidates=len(accepted))
    save();print(json.dumps({k:report.get(k) for k in ['complete','passed','accepted_candidates','selected_charts','selection_error','wall_s']}))


if __name__=='__main__':main()
