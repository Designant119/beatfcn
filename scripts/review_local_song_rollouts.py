"""Build per-song audio/timeline/native packages and diagnostic evidence, without inference."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.events.controls import raw_controls, union
from beatfcn.events.rhythm_preview import write_phira_preview
from beatfcn.model.benchmark import audio_features, capability_metrics, music_metrics
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.evaluation import score_events
from beatfcn.model.local_song_rollout import canonical

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gaps(events, duration):
    heads = sorted({0, duration, *(e['onset_us'] for e in events)})
    return [(a, b) for a, b in zip(heads, heads[1:]) if b-a >= 4_000_000]


def write_svg(path, label, prediction, human, duration, flux, *, scope=None):
    """Raw onset/tail visual evidence; no resampling or editing of events."""
    start, end = scope or (0, duration)
    width, left = 1800, 165
    x = lambda t: left+(t-start)/(end-start)*width
    colors = dict(tap='#579bff', hold='#46c8a1', flick='#ee9b3b', drag='#b172e1')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="510" viewBox="0 0 2000 510">',
           '<rect width="2000" height="510" fill="#101720"/>',
           '<g font-family="Arial,sans-serif" fill="#e2e8ef" font-size="20">',
           f'<text x="20" y="30">{html.escape(label)} | {start/1e6:.2f}–{end/1e6:.2f} s</text>',
           '<text x="20" y="75">Audio flux</text>']
    first, last = start//10000, min(len(flux), (end+9999)//10000)
    stride = max(1, (last-first)//1800)
    series = [(i*10000, float(np.mean(flux[i:min(i+stride,last)]))) for i in range(first,last,stride)]
    maximum = max((v for _, v in series), default=1.) or 1.
    points = ' '.join(f'{x(t):.2f},{110-v/maximum*60:.2f}' for t,v in series)
    svg.append(f'<polyline points="{points}" fill="none" stroke="#b8c2ce" stroke-width="1.4"/>')
    for name, events, top in [('Model', prediction, 165), ('Human', human, 335)]:
        visible_count = sum(e['onset_us'] < end and (e['onset_us'] >= start or e['end_us'] > start) for e in events)
        svg.append(f'<text x="20" y="{top-22}">{name} ({visible_count})</text>')
        for row, (kind, color) in enumerate(colors.items()):
            y = top+row*30
            svg += [f'<text x="30" y="{y+4}" fill="{color}" font-size="16">{kind}</text>',
                    f'<path d="M{left} {y}h{width}" stroke="#334252"/>']
            for event in events:
                if event['type'] != kind or event['onset_us'] >= end or (event['onset_us'] < start and event['end_us'] <= start):
                    continue
                a, b = max(start,event['onset_us']), min(end,event['end_us'])
                svg.append(f'<rect x="{x(a):.2f}" y="{y-8}" width="{max(1.4,x(b)-x(a)):.2f}" height="10" fill="{color}" opacity=".85"/>')
    for t in np.linspace(start, end, 11):
        svg.append(f'<text x="{x(t):.2f}" y="475" text-anchor="middle" font-size="16">{t/1e6:.1f}</text>')
    svg.append('<text x="20" y="502" font-size="15">Raw timing/types. Human is the frozen model-clock view. Display is not spatial or playability evaluation.</text></g></svg>')
    path.write_text(''.join(svg))


def write_page(path, title, audio_name, prediction, human, duration, flux, complete):
    data = json.dumps(dict(prediction=prediction, human=human, duration=duration/1e6,
        flux=flux, complete=complete), ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><meta charset="utf-8"><title>Rhythm comparison</title>
<style>body{background:#121820;color:#edf1f7;font:16px system-ui;margin:24px}canvas{width:100%;background:#18212c}a{color:#7bcbff}p{max-width:1000px}</style>
<h1>TITLE</h1><p>原始节奏输出与人谱模型视图。点击图定位音频；蓝/绿/橙/紫分别为 tap/hold/flick/drag。布局仅用于时间检查。完成状态：COMPLETE</p>
<audio id="audio" controls src="AUDIO"></audio><p><a href="model.zip">Phira 模型节奏稿（辅助摆位）</a> · <a href="human-original.zip">原作者完整谱包</a> · <a href="model-timeline.csv">模型时间表</a> · <a href="human-timeline.csv">人谱模型视图时间表</a></p>
<canvas id="chart" width="2000" height="500"></canvas><p id="clock"></p><p><a href="comparison.svg">Full-song timing figure</a> · <a href="active-detail.svg">Audio-active 8s detail</a></p>
<script>const D=DATA;const c=document.querySelector('#chart'),ctx=c.getContext('2d'),audio=document.querySelector('#audio');
const colors={tap:'#6daaff',hold:'#65dcab',flick:'#ffba68',drag:'#c08aff'},types=['tap','hold','flick','drag'];
function draw(){ctx.clearRect(0,0,c.width,c.height);ctx.font='18px sans-serif';ctx.fillStyle='#edf1f7';ctx.fillText('Music flux / 音频谱通量',15,23);ctx.fillText('Model / 模型',15,125);ctx.fillText('Human / 人谱',15,315);
const left=180,w=1800;let max=Math.max(1e-9,...D.flux);ctx.strokeStyle='#a5adba';ctx.beginPath();D.flux.forEach((y,i)=>{let x=left+i*w/Math.max(1,D.flux.length-1),v=90-y/max*55;i?ctx.lineTo(x,v):ctx.moveTo(x,v)});ctx.stroke();
for(const [events,base] of [[D.prediction,150],[D.human,340]]){types.forEach((t,k)=>{ctx.fillStyle=colors[t];ctx.fillText(t,25,base+k*32);ctx.fillRect(left,base+k*32,w,1)});for(const e of events){const y=base+types.indexOf(e.type)*32,x=left+e.onset_us/1e6/D.duration*w;ctx.fillStyle=colors[e.type];ctx.fillRect(x,y-9,Math.max(2,(e.end_us-e.onset_us)/1e6/D.duration*w),10)}}
ctx.strokeStyle='#fff';let x=left+audio.currentTime/D.duration*w;ctx.beginPath();ctx.moveTo(x,25);ctx.lineTo(x,470);ctx.stroke();document.querySelector('#clock').textContent=audio.currentTime.toFixed(2)+' / '+D.duration.toFixed(2)+' s';}
c.onclick=e=>{let r=c.getBoundingClientRect();audio.currentTime=Math.max(0,Math.min(D.duration,((e.clientX-r.left)*c.width/r.width-180)/1800*D.duration));draw()};audio.ontimeupdate=draw;draw();</script>'''
    page = page.replace('TITLE', html.escape(title)).replace('COMPLETE', str(complete)).replace('AUDIO', html.escape(audio_name)).replace('DATA', data)
    path.write_text(page)


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    for path, expected in plan['inputs_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Review source changed: '+path)
    generation = json.loads((ROOT/plan['generation']).read_text())
    if len(generation['song_files']) != 20:
        raise ValueError('Need twenty actual attempted whole-song outputs, including failures')
    evaluation = json.loads((ROOT/plan['evaluation20']).read_text())['records']
    expected = {r['chart_id']: r for r in evaluation}
    if len(expected) != 20 or set(expected) != set(generation['song_files']):
        raise ValueError('Generated songs differ from the frozen independent evaluation panel')
    if generation['optimizer_updates'] != plan['expected_optimizer_updates']:
        raise ValueError('Unexpected training exposure for the song review')
    if sha(ROOT/generation['checkpoint']) != generation['checkpoint_sha256']:
        raise ValueError('Generation checkpoint changed')
    tape = json.loads((ROOT/plan['tape']).read_text())
    targets = {r['chart_id']: r for r in tape['charts']}
    features = {r['audio_sha256']: r for r in json.loads((ROOT/plan['features']).read_text())['records']}
    clock = {r['audio_sha256']: r for r in json.loads((ROOT/plan['clock']).read_text())['audio']}
    manifest = {r['chart_id']: r for r in map(json.loads, (ROOT/plan['manifest']).read_text().splitlines())}
    output = ROOT/plan['output_dir']
    if output.exists():
        raise FileExistsError('Review artifacts are immutable; use a new output directory')
    output.mkdir(parents=True)
    records = []
    for chart_id, filename in generation['song_files'].items():
        song = json.loads((ROOT/filename).read_text())
        if song['chart_id'] != chart_id or song['mode'] != 'phira':
            raise ValueError('Expected the frozen Phira song output')
        if filename not in plan['inputs_sha256']:
            raise ValueError('Individual raw generation must be frozen before review')
        target, row = targets[chart_id], manifest[chart_id]
        audio = song['audio_sha256']
        if audio != target['audio_sha256'] or audio != expected[chart_id]['audio_sha256']:
            raise ValueError('Human and model audio identity differ')
        source = ROOT/target['target_path']
        if sha(source) != target['target_sha256']:
            raise ValueError('Human model-view targets changed')
        with np.load(source, allow_pickle=False) as a:
            human = [dict(onset_us=int(h), end_us=int(t), type=TYPES[int(k)], lane=int(l)) for h, t, k, l in
                     zip(*(a[key] for key in ('onset_us', 'end_us', 'kind', 'lane')))]
        pred = song['events']
        p, h = (canonical([RhythmEvent(**e) for e in es], 'phira') for es in (pred, human))
        duration = song['duration_us']
        feature = features[audio]
        if duration != feature['duration_us']:
            raise ValueError('Generation and admitted audio durations differ')
        mel_path = ROOT/feature['complete_mel']
        if sha(mel_path) != feature['sha256'][feature['complete_mel']]:
            raise ValueError('Complete audio feature identity changed')
        mel = np.load(mel_path, mmap_mode='r', allow_pickle=False)[:feature['valid_mel_frames']]
        acoustic = audio_features(mel, len(mel)*10_000)
        acoustic.update(duration_us=duration, coverage_end_us=duration, uncovered_tail_us=0,
            recipe='complete_padded_mel_valid_frames_positive_flux_v1', final_frame_may_be_partial=True)
        request = dict(mode='phira', regions=[dict(start_us=0, end_us=duration, state='editable')], observed_events=[])
        scores = score_events(p, h, 'phira')
        capability = capability_metrics(p, h, request, acoustic)
        tap_hold = {key: music_metrics([e for e in es if e.type in ('tap', 'hold')], [(0, duration)], acoustic)
                    for key, es in [('model', p), ('human', h)]}
        gaps_with_context = []
        holds = union((e['onset_us'], e['end_us']) for e in pred if e['type'] == 'hold')
        for start, end in gaps(pred, duration):
            held_us = sum(max(0, min(end, b)-max(start, a)) for a, b in holds)
            gaps_with_context.append(dict(start_us=start, end_us=end,
                definition='No new note heads; sustained holds are reported separately',
                model_hold_covered_us=held_us, model_unoccupied_us=end-start-held_us,
                human_heads=sum(start < e['onset_us'] < end for e in human),
                audio_flux_mean=float(np.mean(acoustic['flux'][start//10000:min(len(mel), (end+9999)//10000)]))))
        folder = output/chart_id.replace(':', '-')
        folder.mkdir()
        ext = {'mp3': '.mp3', 'vorbis': '.ogg', 'flac': '.flac', 'pcm_s16le': '.wav', 'opus': '.ogg'}.get(row['codec'])
        if ext is None:
            raise ValueError('Unregistered audio container extension')
        raw = ROOT/clock[audio]['raw_path']
        if sha(raw) != audio:
            raise ValueError('Audio bytes changed')
        audio_name = 'audio'+ext
        shutil.copyfile(raw, folder/audio_name)
        title = row.get('title') or chart_id
        artist = row.get('artist') or 'Unknown'
        export = write_phira_preview(folder/'model.zip', pred, raw, title=title,
                                     artist=artist, audio_name=audio_name)
        native = ROOT/f"inputs/d66_local_cohort/source/raw/phira/zips/{row['native_id']}.zip"
        if not native.is_file():
            raise ValueError('Original human package missing')
        shutil.copyfile(native, folder/'human-original.zip')
        for name, es in [('model', pred), ('human', human)]:
            with (folder/(name+'-timeline.csv')).open('w') as file:
                writer = csv.writer(file); writer.writerow(['onset_us', 'end_us', 'type', 'lane'])
                writer.writerows((e[k] for k in ('onset_us','end_us','type','lane')) for e in es)
        flux = np.asarray(acoustic['flux'])
        active = max(range(0, max(1, duration-8_000_000+1), 8_000_000),
                     key=lambda t: float(flux[t//10000:min(len(flux),(t+8_000_000)//10000)].mean()))
        active_scope = (active, min(duration, active+8_000_000))
        write_svg(folder/'comparison.svg', chart_id+' '+target['stratum'], pred, human, duration, flux)
        write_svg(folder/'active-detail.svg', chart_id+' audio-active detail', pred, human, duration, flux, scope=active_scope)
        coarse = [float(flux[i:i+10].mean()) for i in range(0, len(flux), 10)]
        write_page(folder/'index.html', title, audio_name, pred, human, duration, coarse, song['complete'])
        counts = scores['edit_utility']['full_note']
        record = dict(chart_id=chart_id, title=title, stratum=target['stratum'], audio_sha256=audio,
            source_generation=filename, source_generation_sha256=sha(ROOT/filename),
            complete=song['complete'], failure=song['failure'], legality=song['legality'],
            notes=dict(model=len(pred), human=len(human)),
            types={name: dict(Counter(e['type'] for e in es)) for name, es in [('model', pred), ('human', human)]},
            full_note_f1=2*counts['matched']/(counts['pred']+counts['true']) if counts['pred']+counts['true'] else None,
            scores=scores, capability=capability, tap_hold_music=tap_hold,
            controls={name: raw_controls('phira', es, [(0, duration)]) for name, es in [('model', p), ('human', h)]},
            long_gaps=gaps_with_context, export=export, source_offset_decision=row['offset_decision'],
            audio_selected_detail_scope_us=active_scope,
            model_clock='Unedited frozen offset-v2 model view; no new fit or snapping during export',
            preview=str((folder/'index.html').relative_to(ROOT)),
            original_human_package_sha256=sha(native), assistant_review_complete=False, human_playtest=False)
        (folder/'analysis.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        records.append(record)
    summary = dict(id=plan['id'], plan_sha256=sha(plan_path), source_sha256=plan['inputs_sha256'],
        checkpoint=generation['checkpoint'], checkpoint_sha256=generation['checkpoint_sha256'],
        generation_complete=generation['complete'], songs=len(records), complete_songs=sum(r['complete'] for r in records),
        records=records, assistant_review_complete=False, quality_passed=False,
        conditions='not measured in these unspecified requests', human_style_continuation='not trained in this baseline',
        report_scope='Automatic evidence and inspectable artifacts, not human listening or native playtesting')
    (ROOT/plan['report']).write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    links = ''.join(f'<li><a href="{html.escape(r["chart_id"].replace(":", "-"))}/index.html">{html.escape(r["title"])} — {html.escape(r["stratum"])}</a> · {r["notes"]["model"]}/{r["notes"]["human"]} notes · complete={r["complete"]}</li>' for r in records)
    (output/'index.html').write_text('<meta charset="utf-8"><h1>20 首真实整曲输出审阅</h1><p>辅助摆位不代表模型空间能力。各页含原音频、人谱原包、模型稿和原始时间表。</p><ol>'+links+'</ol>')
    print(json.dumps(dict(songs=len(records), complete_songs=summary['complete_songs'], report=plan['report'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
