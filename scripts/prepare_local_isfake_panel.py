"""Versioned post-D91 event tapes and cache inventory, with no cloud calls."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from beatfcn.events.canonical import build_model_view
from beatfcn.train.data import chart_from_canon, model_shift_us
from beatfcn.train.local_canonical_audit import rhythm_fingerprint
from scripts.prepare_local_request_tape import requests_for_chart

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/d92-isfake-panel-v1.json'
TAPE = 'reports/d92_all_source_request_tape_v1.json'
LOCAL = 'reports/d92_local_ready_request_tape_v1.json'
INVENTORY = 'reports/d92_panel_inventory_v1.json'


def read(path):
    return json.loads((ROOT/path).read_text())


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def save(path, value):
    path = ROOT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def summarize(charts):
    result = {}
    for split in ('train', 'val'):
        rows = [r for r in charts if r['split'] == split]
        requests = [q for r in rows for q in r['requests']]
        result[split] = dict(charts=len(rows), songs=len({r['component'] for r in rows}),
            mode_charts=dict(Counter(r['mode'] for r in rows)),
            strata=dict(Counter(r['stratum'] for r in rows)), requests=len(requests),
            events=sum(q['event_count'] for q in requests), tokens=sum(q['tokens'] for q in requests),
            empty_requests=sum(q['event_count']==0 for q in requests),
            tail_requests=sum(q['request']['end_us']-q['request']['start_us']<8_000_000 for q in requests))
    return result


def freeze():
    if (ROOT/PLAN).exists():
        raise FileExistsError('Plan already frozen; run the exact existing plan.')
    sources = ['reports/d91_isfake_policy_impact_v1.json',
        'bench/rhythm-v2/fullscale-data-contract-v4.json',
        'reports/d66_source_gates_v1.json', 'reports/d67_local_request_tape_v1.json',
        'reports/d67_local_feature_audit_v1.json', 'reports/d66_audio_clock_v1.json',
        'reports/d66_combined_canonical_receipt.json',
        'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl',
        'inputs/d66_local_cohort/evaluation20-candidates-v1.json',
        'reports/d66_source_inventory_v2.json','reports/d66_added_feature_inventory_v1.json']
    code = ['scripts/prepare_local_isfake_panel.py','scripts/prepare_local_request_tape.py',
        'beatfcn/train/data.py','beatfcn/train/local_canonical_audit.py','beatfcn/events/canonical.py',
        'beatfcn/model/full_event_codec.py','beatfcn/train/event_conditions.py','beatfcn/events/controls.py']
    save(PLAN,dict(id='D92-isfake-panel-v1',inputs_sha256={p:sha(p) for p in sources},
        code_sha256={p:sha(p) for p in code},max_wall_s=180,expected_charts=584,
        output=TAPE,local_output=LOCAL,inventory=INVENTORY,
        cache_root='inputs/d92_local_requests/all-source',
        baseline_tape=sources[3], feature_audit=sources[4],
        optimizer_updates=0,new_cloud_compute_calls=0,quality_passed=False,
        meaning='Data preparation only. Old targets and validation20 remain frozen. All missing features are reported; no training or cache downloads.'))
    print(json.dumps(dict(plan=PLAN,sha256=sha(PLAN))))


def inventory(plan, candidates, features):
    available = {r['audio_sha256']:r for r in features['records'] if r['passed']}
    local_files = subprocess.check_output(['rg','--files','inputs','-g','L24.npy'],cwd=ROOT,text=True).splitlines()
    local_by_audio = defaultdict(list)
    for path in local_files:
        local_by_audio[Path(path).parent.name].append(str(Path(path).parent))
    remote = defaultdict(dict)
    for row in read('reports/d66_source_inventory_v2.json')['records']:
        if row['kind']=='taps' and row.get('available'):
            remote[row['audio_sha256']].update({Path(r['path']).name:r for r in row['files']})
    for row in read('reports/d66_added_feature_inventory_v1.json')['records']:
        if row['kind']=='pupu_tap':
            remote[row['audio_sha256']][Path(row['path']).name]=row
    by_audio=defaultdict(list)
    for row in candidates:
        by_audio[row['audio_sha256']].append(row)
    rows=[]
    for audio, charts in sorted(by_audio.items()):
        files=remote[audio]
        keys=['L9.npy','L16.npy','L24.npy']
        rows.append(dict(audio_sha256=audio,charts=[r['chart_id'] for r in charts],
            splits=sorted({r['split'] for r in charts}),
            complete_feature_audited=audio in available,
            local_unadmitted_pupu_dirs=local_by_audio.get(audio,[]) if audio not in available else [],
            historical_remote_files=[files[k] for k in keys if k in files],
            historical_remote_complete=all(k in files for k in keys),
            historical_remote_bytes=sum(files[k]['bytes'] for k in keys if k in files),
            local_download_bytes=0))
    missing=[r for r in rows if not r['complete_feature_audited']]
    return dict(id='D92-local-panel-inventory-v1',complete=True,rows=rows,
        known_missing_feature_bytes=sum(r['historical_remote_bytes'] for r in missing),
        unique_audio=len(rows),audited_audio=len(rows)-len(missing),missing_audited_audio=len(missing),
        missing_without_full_historical_remote_metadata=sum(not r['historical_remote_complete'] for r in missing),
        restored_charts_ready=[r['chart_id'] for r in candidates if r['restored_by_d91'] and r['audio_sha256'] in available],
        optimizer_updates=0,new_cloud_compute_calls=0,
        download_performed=False,remote_inventory_refreshed=False,
        limitation='Historical remote sizes are planning evidence only; no live availability or transfer cost asserted.')


def run():
    plan=read(PLAN)
    for p,h in {**plan['inputs_sha256'],**plan['code_sha256']}.items():
        if sha(p)!=h:raise ValueError('Frozen source changed: '+p)
    previous=read(plan['baseline_tape'])
    features=read(plan['feature_audit'])
    if not previous['complete'] or previous['failures'] or not features['complete'] or features['failures']:
        raise ValueError('Existing evidence is incomplete')
    source=read('reports/d66_source_gates_v1.json')
    metadata={r['chart_id']:r for k in ('eligible','unresolved') for r in source[k]}
    policy=read('reports/d91_isfake_policy_impact_v1.json')
    candidates=[]
    for r in policy['rows']:
        if not r['current_policy_source_pass']:continue
        c=dict(metadata[r['chart_id']],source_screen_passed=True,
            model_rhythm_sha256=r['model_rhythm_sha256'],restored_by_d91=not r['old_source_pass'])
        candidates.append(c)
    assert len(candidates)==plan['expected_charts']
    by_component=defaultdict(set)
    by_audio=defaultdict(set)
    for r in candidates:
        by_component[r['component']].add(r['split']);by_audio[r['audio_sha256']].add(r['split'])
    if any(len(s)>1 for s in [*by_component.values(),*by_audio.values()]):
        raise ValueError('Cross-split song/audio leakage')
    if (ROOT/TAPE).exists():
        report=read(TAPE)
        if report['plan_sha256']!=sha(PLAN) or report['failures']:
            raise ValueError('Unresolved failure or identity mismatch; inspect, do not retry blindly')
        if report['complete']:raise ValueError('Tape already complete')
    else:
        report=dict(id=plan['id'],plan_sha256=sha(PLAN),code_sha256=plan['code_sha256'],
            charts=[],failures=[],attempts=[],complete=False,training_ready=False,
            optimizer_updates=0,cloud_compute_calls=0,
            source_policy='explicit-isfake-only-v1',evaluation20='inputs/d66_local_cohort/evaluation20-candidates-v1.json')
    old={r['chart_id']:r for r in previous['charts']}
    seen={r['chart_id'] for r in report['charts']}
    manifest={r['chart_id']:r for r in map(json.loads,(ROOT/'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl').read_text().splitlines())}
    files={r['path']:r for r in read('reports/d66_combined_canonical_receipt.json')['files']}
    clock={r['audio_sha256']:r for r in read('reports/d66_audio_clock_v1.json')['audio']}
    started=time.monotonic()
    attempt=dict(started_at_unix=time.time(),reused_charts=0,new_charts=0)
    report['attempts'].append(attempt)

    def persist():
        attempt.update(wall_s=time.monotonic()-started,observed_at_unix=time.time())
        report.update(complete=len(report['charts'])==len(candidates) and not report['failures'],
            summary=summarize(report['charts']),prepared_charts=len(report['charts']),expected_charts=len(candidates))
        save(TAPE,report)

    # Existing record order is retained before deterministic new chart IDs.
    order={r['chart_id']:i for i,r in enumerate(previous['charts'])}
    for c in sorted(candidates,key=lambda c:(order.get(c['chart_id'],len(order)),c['chart_id'])):
        cid=c['chart_id']
        if cid in seen:continue
        if time.monotonic()-started>plan['max_wall_s']:break
        try:
            if cid in old:
                record=old[cid]
                if record['model_rhythm_sha256']!=c['model_rhythm_sha256']:
                    raise ValueError('Reused target differs from policy audit')
                for field in ('target','token'):
                    if sha(record[field+'_path'])!=record[field+'_sha256']:
                        raise ValueError('Reused target/token cache drift')
                report['charts'].append(record)
                attempt['reused_charts']+=1
            else:
                row=manifest[cid]
                receipt=files[f"canon/{row['source']}/{row['native_id']}.json"]
                if sha(receipt['local'])!=receipt['sha256']:raise ValueError('Canonical source drift')
                chart=chart_from_canon(read(receipt['local']),model_shift_us(row,True))
                view=build_model_view(row['mode'],(),chart.events)
                events=tuple(e for g in view.target_groups for e in g.events)
                if view.quarantine or len(events)!=len(chart.events) or rhythm_fingerprint(row['mode'],events)!=c['model_rhythm_sha256']:
                    raise ValueError('Target changed or representation failure')
                duration=clock[c['audio_sha256']]['decoded_samples']*1_000_000//24000
                requests,tokens,targets=requests_for_chart(events,row['mode'],duration,c['split'])
                folder=ROOT/plan['cache_root'];folder.mkdir(parents=True,exist_ok=True)
                stem=cid.replace(':','-')
                target_path,token_path=folder/(stem+'-targets.npz'),folder/(stem+'-tokens.npy')
                if target_path.exists() or token_path.exists():raise FileExistsError('Unregistered cache exists: '+cid)
                np.savez(target_path,**targets);np.save(token_path,tokens,allow_pickle=False)
                record=dict(chart_id=cid,component=c['component'],split=c['split'],mode=c['mode'],stratum=c['stratum'],
                    audio_sha256=c['audio_sha256'],canonical_sha256=receipt['sha256'],model_rhythm_sha256=c['model_rhythm_sha256'],
                    duration_us=duration,target_path=str(target_path.relative_to(ROOT)),token_path=str(token_path.relative_to(ROOT)),
                    target_sha256=sha(str(target_path.relative_to(ROOT))),token_sha256=sha(str(token_path.relative_to(ROOT))),requests=requests)
                report['charts'].append(record);attempt['new_charts']+=1
            if len(report['charts'])%25==0:persist()
        except Exception as exc:
            report['failures'].append(dict(chart_id=cid,error=f'{type(exc).__name__}: {exc}'))
            persist();raise
    persist()
    if report['complete']:
        available={r['audio_sha256'] for r in features['records'] if r['passed']}
        local={**report,'id':'D92-local-audited-feature-subset-v1',
            'charts':[r for r in report['charts'] if r['audio_sha256'] in available],
            'all_source_tape':TAPE,'all_source_tape_sha256':sha(TAPE),
            'scope':'Existing feature-admitted audio only; not a337-song training experiment.'}
        local.update(summary=summarize(local['charts']),prepared_charts=len(local['charts']),expected_charts=len(local['charts']))
        save(LOCAL,local)
        inv=inventory(plan,candidates,features)
        inv.update(all_source_summary=report['summary'],local_feature_ready_summary=local['summary'],
            plan_sha256=sha(PLAN),source_sha256=plan['inputs_sha256'],
            requests_all=TAPE,requests_all_sha256=sha(TAPE),requests_local=LOCAL,requests_local_sha256=sha(LOCAL))
        save(INVENTORY,inv)
    print(json.dumps(dict(complete=report['complete'],prepared_charts=len(report['charts']),attempt=attempt,
                         summary=report['summary']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    if parser.parse_args().freeze:freeze()
    else:run()
