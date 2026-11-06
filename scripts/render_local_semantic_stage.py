"""Export frozen D97 versus matched D94 stage events; no model updates or event edits."""
import argparse
from collections import Counter
import csv
import html
import json
import os
from pathlib import Path

import numpy as np

from beatfcn.model.benchmark import audio_features
from scripts.review_local_song_rollouts import write_svg
from scripts.run_local_song_baseline import sha, write_json

ROOT=Path(__file__).resolve().parents[1]


def run(step, output):
    paths={'control':f'reports/d94_stage_{step}_v1.json', 'candidate':f'reports/d97_stage_{step}_v1.json'}
    reports={arm:json.loads((ROOT/path).read_text())for arm,path in paths.items()}
    if not all(r['complete'] and not r.get('error') for r in reports.values()):
        raise ValueError('Both complete stage audits are required')
    if len(reports['control']['rows']) != 8 or len(reports['candidate']['rows']) != 8:
        raise ValueError('Both frozen eight-case panels are required')
    for a,b in zip(reports['control']['rows'],reports['candidate']['rows'],strict=True):
        keys=('record_index','role','mode','chart_id','component','audio_sha256','stratum','scope_us','truth_events')
        if any(a[k]!=b[k] for k in keys):
            raise ValueError('Stage cases or target labels differ')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    feature_path='reports/d67_local_feature_audit_v1.json'
    clock_path='reports/d66_audio_clock_v1.json'
    features={r['audio_sha256']:r for r in json.loads((ROOT/feature_path).read_text())['records']}
    clocks={r['audio_sha256']:r for r in json.loads((ROOT/clock_path).read_text())['audio']}
    page=['<!doctype html><meta charset="utf-8"><title>D97 semantic-objective stage review</title>',
        '<style>body{font:16px system-ui;background:#101720;color:#eee;margin:24px}img{width:100%}a{color:#8cf}audio{width:100%}</style>',
        f'<h1>D97 {step}步（同2.37M/333歌：原CE控制 / 语义监督候选）：原始片段对照</h1><p>8个片段诊断，不是20首整曲质量验收。没有吸附、删音符、重排或补空白。人谱留白不等同音乐静音。</p>']
    manifest=dict(step=step,optimizer_updates=0,records=[],
        source_sha256={p:sha(ROOT/p)for p in [*paths.values(),feature_path,clock_path,
            'scripts/render_local_semantic_stage.py','scripts/review_local_song_rollouts.py']},
        audio_audition_complete=False,visual_review_complete=False)
    for a,b in zip(reports['control']['rows'],reports['candidate']['rows'],strict=True):
        if a['record_index']!=b['record_index'] or a['truth_events']!=b['truth_events']:
            raise ValueError('Paired rows/labels differ')
        audio=a['audio_sha256'];feature=features[audio];raw=ROOT/clocks[audio]['raw_path']
        if sha(raw)!=audio or sha(ROOT/feature['complete_mel'])!=feature['sha256'][feature['complete_mel']]:
            raise ValueError('Original audio/feature changed')
        mel=np.load(ROOT/feature['complete_mel'],mmap_mode='r',allow_pickle=False)[:feature['valid_mel_frames']]
        acoustic=audio_features(mel,len(mel)*10000)
        item=dict(record_index=a['record_index'],chart_id=a['chart_id'],role=a['role'],scope_us=a['scope_us'],
            original_audio=str(raw.relative_to(ROOT)),original_audio_sha256=audio,variants={})
        page.append(f'<h2>{html.escape(a["chart_id"])} · {a["role"]} · {a["scope_us"][0]/1e6:g}–{a["scope_us"][1]/1e6:g}s</h2>')
        relative=os.path.relpath(raw,output)
        page.append(f'<audio controls preload="none" src="{html.escape(relative)}#t={a["scope_us"][0]/1e6},{a["scope_us"][1]/1e6}"></audio>')
        for arm,row in [('control',a),('candidate',b)]:
            name=f'{a["record_index"]}-{arm}'
            svg=output/(name+'.svg')
            write_svg(svg,f'{arm} step{step} {row["chart_id"]} complete={row["prediction"]["complete"]}',
                row['prediction']['events'],row['truth_events'],feature['duration_us'],acoustic['flux'],scope=row['scope_us'])
            csv_path=output/(name+'.csv')
            with csv_path.open('w')as stream:
                writer=csv.DictWriter(stream,fieldnames=['onset_us','end_us','type','lane']);writer.writeheader();writer.writerows(row['prediction']['events'])
            heads=sorted({e['onset_us']for e in row['prediction']['events']if e['type']=='tap'})
            delta=np.diff(heads)
            item['variants'][arm]=dict(svg=str(svg.relative_to(ROOT)),svg_sha256=sha(svg),
                csv=str(csv_path.relative_to(ROOT)),csv_sha256=sha(csv_path),notes=len(row['prediction']['events']),
                prediction_complete=row['prediction']['complete'],token_capped=row['prediction']['token_limit_reached'],
                tap_ioi_median_ms=float(np.median(delta)/1000)if len(delta)else None,
                exactly_1ms_tap_intervals=int((delta==1000).sum()),
                tap_heads_per_second=dict(Counter((t-row['scope_us'][0])//1000000 for t in heads)))
            page.append(f'<h3>{arm}</h3><img src="{name}.svg"><p><a href="{name}.csv">原始时间表</a></p>')
        manifest['records'].append(item)
    (output/'index.html').write_text('\n'.join(page))
    write_json(output/'manifest.json',manifest)
    print(json.dumps(dict(output=str(output),records=len(manifest['records']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.step,args.output.resolve())
