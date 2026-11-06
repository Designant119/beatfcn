"""Export saved rhythm rollouts with an explicitly synthetic, static Phira layout.

No model inference, fitting, timing edits, type edits, or source artwork reuse.
The fixed 60 BPM export clock encodes seconds, not an estimated musical tempo.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile
import zlib

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.events.phira import parse_phira, load_package

OUT = ROOT / 'artifacts/chart-preview-20261025/phira'
RECORDS = ROOT / 'reports/condition_v1_benchmark/records'
SOURCES = {'latest': '7fc2a6e49cabea618f9e792a.json', 'baseline': '2c597c1de573cea59d557f0f.json'}
LABELS = {'latest': '4500 步 · 原始节奏', 'baseline': '3500 步 · 原始节奏', 'human': '人谱节奏参考 · 统一布局'}


def layout(events, correction_ms):
    """Separate chords/occupied holds; these positions are NOT model predictions."""
    grouped = defaultdict(list)
    for event in events:
        grouped[event['onset_us']].append(event)
    positions = list(range(-450, 451, 90))
    busy = dict.fromkeys(positions, -10**12)
    last = dict.fromkeys(positions, -10**12)
    result = []
    for gi, (time, notes) in enumerate(sorted(grouped.items())):
        free = [x for x in positions if busy[x] <= time]
        notes = sorted(notes, key=lambda e: (e['type'], e['end_us'], e['event_id']))
        if len(free) < len(notes):
            raise ValueError('cannot display all notes without overlapping occupied holds')
        chosen = []
        for ni, event in enumerate(notes):
            target = (-225 if (gi+ni) % 2 == 0 else 225)
            x = min(free, key=lambda x: abs(x-target) + 1000 * (time-last[x] < 120000))
            free.remove(x)
            chosen.append(x)
            busy[x] = event['end_us'] if event['type'] == 'hold' else time
            last[x] = time
            result.append(dict(id=event['event_id'], t_us=time-correction_ms*1000,
                               end_us=event['end_us']-correction_ms*1000, type=event['type'], x=x))
    return result


def beat(us):
    q = Fraction(us, 1000000)
    whole = q.numerator // q.denominator
    rest = q - whole
    return [whole, rest.numerator, rest.denominator]


def rpe(events, label):
    def constant(value):
        return [dict(startTime=[0, 0, 1], endTime=[1000, 0, 1], start=value, end=value,
                     easingType=1, easingLeft=0., easingRight=1., bezier=0,
                     bezierPoints=[0., 0., 0., 0.], linkgroup=0)]
    kinds = dict(tap=1, hold=2, flick=3, drag=4)
    notes = [dict(above=1, alpha=255, startTime=beat(e['t_us']), endTime=beat(e['end_us']),
                  isFake=0, positionX=e['x'], size=1., speed=1., type=kinds[e['type']],
                  visibleTime=999999., yOffset=0.) for e in events]
    return dict(BPMList=[dict(bpm=60., startTime=[0, 0, 1])],
                META=dict(RPEVersion=150, background='background.png', charter='BeatFCN / display layout',
                          composer='Sakuzyo', id='', level=label, name='Altale', offset=0, song='audio.ogg'),
                judgeLineGroup=['Default'], multiLineString='', multiScale=1.,
                judgeLineList=[dict(Group=0, Name='Static display line', Texture='line.png',
                                    bpmfactor=1., father=-1, isCover=1, zOrder=0,
                                    eventLayers=[dict(alphaEvents=constant(255), moveXEvents=constant(0.),
                                                      moveYEvents=constant(-230.), rotateEvents=constant(0.),
                                                      speedEvents=constant(6.))],
                                    numOfNotes=len(notes), notes=notes)])


def plain_background():
    # A plain color file required by the package; not an illustrative image.
    w, h = 16, 9
    raw = b''.join(b'\0' + bytes((13, 20, 31))*w for _ in range(h))
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')


def summary(events):
    starts = sorted({e['t_us'] for e in events})
    gaps = sorted([dict(start_s=a/1e6, end_s=b/1e6, duration_s=(b-a)/1e6)
                   for a,b in zip(starts, starts[1:]) if b-a >= 2000000],
                  key=lambda g: -g['duration_s'])
    per_second = Counter(e['t_us']//1000000 for e in events)
    return dict(notes=len(events), types=dict(Counter(e['type'] for e in events)),
                first_s=starts[0]/1e6, last_s=starts[-1]/1e6, gaps=gaps,
                peak_one_second_notes=max(per_second.values()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    row = next(r for r in json.loads((ROOT/'bench/capability-v1/manifest.json').read_text())['rows']
               if r['chart_id']=='phira:17188')
    audio_hash = hashlib.sha256((OUT/'audio.ogg').read_bytes()).hexdigest()
    assert audio_hash == row['audio_sha256']
    records = {key: json.loads((RECORDS/name).read_text()) for key,name in SOURCES.items()}
    for record in records.values():
        assert record['chart_id']==row['chart_id'] and record['request_kind']=='from_scratch'
    raw_events = {k:r['generated_events'] for k,r in records.items()}
    raw_events['human'] = records['latest']['truth_events']
    correction = row['offset_decision']['c_ms']
    versions = {k:layout(es, correction) for k,es in raw_events.items()}
    manifest = dict(title=row['title'], artist=row['artist'], duration_s=row['audio_sec'],
                    audio_sha256=audio_hash, source_clock_correction_ms=-correction,
                    clock='60 BPM serialization clock only; absolute audio times; not musical tempo',
                    layout='one static judge line; alternating display positions with occupied-hold exclusion; no model space output',
                    timing_type_duration_multiplicity_unchanged=True, new_cloud_compute=False,
                    source_records={k:dict(path=str((RECORDS/n).relative_to(ROOT)),
                        sha256=hashlib.sha256((RECORDS/n).read_bytes()).hexdigest(),
                        checkpoint=records[k]['checkpoint']) for k,n in SOURCES.items()},
                    variants={}, limitations=['display positions do not establish two-hand playability',
                    'human rhythm reference loses the original spatial choreography',
                    'package parser checked; native Phira playtest not yet performed'])
    for key, events in versions.items():
        expected = sorted((e['onset_us']-correction*1000, e['end_us']-correction*1000, e['type']) for e in raw_events[key])
        assert expected == sorted((e['t_us'],e['end_us'],e['type']) for e in events)
        assert min(e['t_us'] for e in events) >= 0
        chart = json.dumps(rpe(events, LABELS[key]), ensure_ascii=False).encode()
        parsed = parse_phira(chart)
        assert not parsed.quarantined, parsed.issues
        assert expected == sorted((e.onset_us,e.end_us,e.type) for e in parsed.events)
        info = dict(name=f'Altale — {LABELS[key]}', level='DRAFT · unrated', difficulty=0.,
                    charter='BeatFCN model + synthetic display layout' if key!='human' else 'DR3AMLAND rhythm / synthetic display layout',
                    composer='Sakuzyo', illustrator='Plain background', chart='chart.json', music='audio.ogg',
                    illustration='background.png', previewStart=60., previewEnd=75., offset=0.,
                    aspectRatio=16/9, backgroundDim=.5, lineLength=6.,
                    intro='Private rhythm inspection. Spatial positions are display-only. Difficulty is unrated.')
        package = OUT/f'altale-{key}.zip'
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('chart.json',chart)
            z.writestr('info.yml',yaml.safe_dump(info,allow_unicode=True))
            z.writestr('background.png',plain_background())
            z.write(OUT/'audio.ogg','audio.ogg')
        parsed_package = load_package(package)
        assert not parsed_package.quarantined, parsed_package.issues
        assert expected == sorted((e.onset_us,e.end_us,e.type) for e in parsed_package.events)
        (OUT/f'{key}.rpe.json').write_bytes(chart)
        with (OUT/f'{key}-timeline.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['event_id','audio_time_ms','type','end_ms','display_x_not_model'])
            writer.writerows((e['id'],e['t_us']/1000,e['type'],e['end_us']/1000,e['x']) for e in events)
        manifest['variants'][key]=dict(label=LABELS[key],**summary(events),export_roundtrip_exact=True,
                                      package_sha256=hashlib.sha256(package.read_bytes()).hexdigest())
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(OUT/'audio.ogg'),
                    '-ac','2','-ar','24000','-c:a','pcm_s16le',str(OUT/'preview.wav')],check=True)
    wave_bytes=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(OUT/'audio.ogg'),
                              '-ac','1','-ar','24000','-f','f32le','pipe:1'],capture_output=True,check=True).stdout
    wave=np.frombuffer(wave_bytes,dtype='<f4')
    rms=np.array([np.sqrt(np.mean(wave[i:i+2400]**2)) for i in range(0,len(wave),2400)])
    rms=(rms/max(float(rms.max()),1e-9)).round(4).tolist()
    content=dict(title=row['title'],artist=row['artist'],duration=row['audio_sec'],
                 labels=LABELS, versions={k:[dict(t=e['t_us']/1e6,end=e['end_us']/1e6,
                    type=e['type'],x=e['x']) for e in es] for k,es in versions.items()},
                 summaries=manifest['variants'],waveform=rms,waveform_hop_s=.1)
    (OUT/'chart-data.js').write_text('const CHART = '+json.dumps(content,ensure_ascii=False)+';\n')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Altale · Phira 节奏试听

4500 步主对照：r02-density-contrastive。3500 步为父检查点。全曲保存的 from_scratch 输出，不运行新推理。
时间、类型、时长、同时音符数量全部保留。导出统一减去数据时钟补偿 10 ms，不拟合听感偏移。

模型没有输出 Phira 空间位置。横向位置和固定判定线只是展示布局，不代表模型能力，也未通过双手实玩验收。
人谱参考采用保存的干净 truth_events，同样换成展示布局；不是原作者的完整空间演出，不应按此评判原作。

每个 zip 含 RPE chart.json、原音频、纯色背景和 info.yml，可交给 Phira 导入；已由项目解析器逐音符回读校验，尚未在原生客户端实玩。
60 BPM 只用于绝对秒数序列化，不能当作音乐 BPM。

音频及原谱仅供私人检查。原曲 Sakuzyo，原人谱 DR3AMLAND。无云 GPU 计算。
格式依据：https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html
''')
    print(json.dumps(manifest['variants'],ensure_ascii=False))


if __name__=='__main__':
    main()
