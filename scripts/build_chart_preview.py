"""Turn one saved rollout into an inspectable 4K draft; never run the neural model.

Timing/type/duration/multiplicity remain fixed. A small human-pattern library
and explicit ergonomic costs assign lanes. This is assisted chart editing,
not a new checkpoint benchmark or a claim of human play-test acceptance.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.train.data import selected_for_training, chart_from_canon, dup_only, check_stacks
from beatfcn.events.osu_mania import parse_osu

OUT = ROOT / 'artifacts/chart-preview-20261025'
SOURCE = ROOT / 'reports/condition_v1_benchmark/records/412740394827aa998f539b5f.json'


def groups(events):
    result = defaultdict(list)
    for e in events:
        result[e['onset_us']].append(e)
    return [(t, sorted(v, key=lambda x: (x['type'], x['end_us'], x['event_id']))) for t, v in sorted(result.items())]


def gap_bucket(us):
    return sum(us >= x for x in (80000, 140000, 250000, 500000))


def mirror(mask):
    return sum(((mask >> i) & 1) << (3-i) for i in range(4))


def human_patterns():
    rows = [json.loads(l) for l in (ROOT/'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl').read_text().splitlines()]
    heldout = {r['audio_sha256'] for r in json.loads((ROOT/'bench/capability-v1/manifest.json').read_text())['rows']}
    unigram, bigram, trigram, sources = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), []
    for row in rows:
        if row.get('mode') != 'mania_4k' or row.get('split') != 'train' or not selected_for_training(row) or not (row.get('kept') or dup_only(row)):
            continue
        path = ROOT/'inputs/training_audit_20261023/canon/osu'/f'{row["native_id"]}.json'
        if not path.exists():
            continue
        if row['audio_sha256'] in heldout:
            raise ValueError('pattern library must exclude all benchmark audio')
        doc = json.loads(path.read_text())
        chart = chart_from_canon(doc)
        if doc['audio_sha256'] != row['audio_sha256']:
            raise ValueError('pattern library audio identity mismatch')
        if dup_only(row):
            check_stacks(chart.events)
        sources.append(dict(chart_id=row['chart_id'], sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        previous, previous2, last = 0, 0, -10**9
        for t, notes in groups(doc['events']):
            mask = sum(1 << e['location']['lane'] for e in notes)
            if len(notes) > 4 or mask.bit_count() != len(notes):
                continue
            n, gap = len(notes), gap_bucket(t-last)
            for a, b, c in ((previous2, previous, mask), (mirror(previous2), mirror(previous), mirror(mask))):
                unigram[n][c] += 1
                bigram[(b, n, gap)][c] += 1
                trigram[(a, b, n, gap)][c] += 1
            previous2, previous, last = previous, mask, t
    return unigram, bigram, trigram, sources


def arrange(events, patterns):
    uni, bi, tri, _ = patterns
    # score, last two masks, per-lane head/end, per-hand counts, chosen lane tuples
    initial = (0., 0, 0, (-10**9,)*4, (-10**9,)*4, (0,0), ())
    beam, original = [initial], initial
    last_time = -10**9
    for t, notes in groups(events):
        n, gap = len(notes), gap_bucket(t-last_time)
        choices = math.comb(4, n)
        def advance(state, lanes):
            cost, p2, prev, heads, ends, hands, path = state
            if len(set(lanes)) != n or any(ends[l] > t for l in lanes):
                return None
            mask = sum(1 << l for l in lanes)
            freq = tri.get((p2, prev, n, gap))
            if not freq or sum(freq.values()) < 20:
                freq = bi.get((prev, n, gap))
            if not freq or sum(freq.values()) < 20:
                freq = uni[n]
            extra = -math.log((freq[mask]+1)/(sum(freq.values())+choices))
            h, e, balance = list(heads), list(ends), list(hands)
            for note, lane in zip(notes, lanes):
                delta = t-heads[lane]
                extra += 12 * max(0., (100000-delta)/100000)**2
                if delta < 50000:
                    extra += 15
                # Release then re-press on the same finger also needs room.
                if ends[lane] > heads[lane] and 0 <= t-ends[lane] < 50000:
                    extra += 3 * (1-(t-ends[lane])/50000)
                h[lane], e[lane] = t, note['end_us']
                balance[lane//2] += 1
            extra += .02 * ((balance[0]-balance[1])**2 - (hands[0]-hands[1])**2)
            if n == 2 and ((mask & 3) == mask or (mask & 12) == mask):
                extra += .15
            return (cost+extra, prev, mask, tuple(h), tuple(e), tuple(balance), path+(tuple(lanes),))
        candidates = {}
        for state in beam:
            free = [l for l in range(4) if state[4][l] <= t]
            # Equal-type, equal-duration notes are interchangeable.
            seen = set()
            for lanes in itertools.permutations(free, n):
                assignment = tuple(sorted((e['type'], e['end_us'], l) for e,l in zip(notes,lanes)))
                if assignment in seen:
                    continue
                seen.add(assignment)
                s = advance(state, lanes)
                if s is not None:
                    key = s[1:6]
                    if key not in candidates or s[0] < candidates[key][0]:
                        candidates[key] = s
        original = advance(original, tuple(e['location']['lane'] for e in notes))
        if original is None:
            raise ValueError('saved original chart has a lane/hold conflict')
        beam = sorted(candidates.values(), key=lambda x: x[0])[:80]
        # Preserve a complete legal baseline path even if the bounded search
        # drops a hold allocation that later becomes necessary.
        if original not in beam:
            beam.append(original)
        last_time = t
    chosen = min(beam, key=lambda x:x[0])[-1]
    result = []
    for (_, notes), lanes in zip(groups(events), chosen, strict=True):
        for note, lane in zip(notes, lanes):
            new = deepcopy(note)
            new['location']['lane'] = lane
            result.append(new)
    return result


def metrics(events):
    heads, ends, counts = [-10**9]*4, [-10**9]*4, [0]*4
    short, overlap, duplicates = 0, 0, 0
    for t, notes in groups(events):
        lanes = [e['location']['lane'] for e in notes]
        duplicates += len(lanes)-len(set(lanes))
        for e in notes:
            l = e['location']['lane']
            short += t-heads[l] < 80000
            overlap += ends[l] > t
            heads[l], ends[l] = t, e['end_us']
            counts[l] += 1
    return dict(notes=len(events), groups=len(groups(events)), holds=sum(e['type']=='hold' for e in events),
                lane_counts=counts, same_lane_under_80ms=short, hold_overlaps=overlap, duplicate_lanes=duplicates)


def osu(row, events, version):
    header = ['osu file format v14','','[General]','AudioFilename: audio.mp3','AudioLeadIn: 0','Mode: 3',
              'SampleSet: Soft','','[Metadata]',f'Title:{row["title"]}',f'TitleUnicode:{row["title"]}',
              f'Artist:{row["artist"]}',f'ArtistUnicode:{row["artist"]}','Creator:BeatFCN + Codex',f'Version:{version}',
              'Tags:private-preview saved-model-timing placeholder-bpm','','[Difficulty]','HPDrainRate:5',
              'CircleSize:4','OverallDifficulty:6','ApproachRate:5','SliderMultiplier:1.4','SliderTickRate:1','',
              '[TimingPoints]','0,500,4,2,0,40,1,0','','[HitObjects]']
    shift = row['offset_decision']['c_ms']
    for e in sorted(events,key=lambda x:(x['onset_us'],x['location']['lane'])):
        t, end, lane = e['onset_us']//1000-shift, e['end_us']//1000-shift, e['location']['lane']
        if t < 0 or end < t:
            raise ValueError('export would require changing event timing')
        extra = f'{end}:0:0:0:0:' if e['type']=='hold' else '0:0:0:0:'
        header.append(f'{64+128*lane},192,{t},{128 if e["type"]=="hold" else 1},0,{extra}')
    return '\r\n'.join(header)+'\r\n'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    record = json.loads(SOURCE.read_text())
    row = next(r for r in json.loads((ROOT/'bench/capability-v1/manifest.json').read_text())['rows'] if r['chart_id']==record['chart_id'])
    if hashlib.sha256((OUT/'audio.mp3').read_bytes()).hexdigest()!=row['audio_sha256']:
        raise ValueError('demo audio differs from generation audio')
    original = record['generated_events']
    patterns = human_patterns()
    arranged = arrange(original,patterns)
    truthless = lambda es: sorted((e['event_id'],e['onset_us'],e['end_us'],e['type']) for e in es)
    assert truthless(original)==truthless(arranged)
    a, b = metrics(original), metrics(arranged)
    assert b['hold_overlaps']==b['duplicate_lanes']==0
    original_by_id={e['event_id']:e for e in original}
    manifest=dict(title=row['title'],artist=row['artist'],duration_s=row['audio_sec'],checkpoint=record['checkpoint'],
                  source_record=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  audio_sha256=row['audio_sha256'],source_clock_correction_ms=-row['offset_decision']['c_ms'],
                  timing_type_duration_multiplicity_unchanged=True,
                  changed_lanes=sum(e['location']!=original_by_id[e['event_id']]['location'] for e in arranged),
                  original=a,arranged=b,pattern_sources=patterns[3],
                  method='human chart n-gram library plus explicit ergonomic beam search; assisted edit, not model output',
                  timing_point='120 BPM placeholder for playback/export; no beat-grid claim; absolute note times retained',
                  limitations=['no human playtest yet','timing errors and missing holds intentionally retained',
                               'human pattern library is train-only and excludes all benchmark audio',
                               'hand-balanced patterns do not establish musical phrasing quality',
                               'cannot promote checkpoint using this edited demonstration'],new_cloud_compute=False)
    for name,events in [('original',original),('arranged',arranged)]:
        text=osu(row,events,'Original model lanes' if name=='original' else 'Human-pattern lane draft')
        path=OUT/f'{name}.osu';path.write_bytes(text.encode())
        parsed=parse_osu(text.encode(),source_name=path.name)
        assert len(parsed.events)==len(events)
        assert not parsed.quarantined, parsed.quarantine_reasons()
    with (OUT/'timeline.csv').open('w') as file:
        writer=csv.writer(file);writer.writerow(['time_ms','type','end_ms','original_lane_1to4','arranged_lane_1to4'])
        for e in arranged:
            writer.writerow([e['onset_us']//1000-14,e['type'],e['end_us']//1000-14,
                             original_by_id[e['event_id']]['location']['lane']+1,e['location']['lane']+1])
    ui=lambda es:[dict(t=e['onset_us']/1e6-.014,end=e['end_us']/1e6-.014,lane=e['location']['lane'],type=e['type']) for e in es]
    content=dict(title=row['title'],artist=row['artist'],duration=row['audio_sec'],original=ui(original),arranged=ui(arranged),metrics=manifest)
    (OUT/'chart-data.js').write_text('const CHART = '+json.dumps(content,ensure_ascii=False)+';\n')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text(f'# {row["title"]} — 私人试听谱\n\n原音频，约 {row["audio_sec"]:.1f} 秒。{len(original)} 个音符，{b["holds"]} 个长按。\n\n打开 index.html 播放预览；拖动时间条或切换原模型排键／参考人谱重排。导入 beatfcn-preview.osz 可在 osu!mania 中试听两份难度。timeline.csv 是时间表。\n\n时间点、类型、长按时长、和弦数量均来自保存的 4500 步模型输出，未人工删改；键道重排参考了 {len(patterns[3])} 份独立训练人谱及显式手感规则，不是模型自行获得的进步。原模型时间点仍可能有多放、漏放和局部错拍。\n\n文件中 120 BPM timing point 仅是播放占位，不能用它推断真实 BPM 或吸附编辑。音符使用绝对时间。此次试听不构成 benchmark 晋升或人工验收。\n\n编排参考：https://osu.ppy.sh/wiki/en/Guides/osu%21mania_mapping_guide\n')
    with zipfile.ZipFile(OUT/'beatfcn-preview.osz','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('audio.mp3','original.osu','arranged.osu'):
            z.write(OUT/name,name)
    print(json.dumps({k:manifest[k] for k in ('title','changed_lanes','original','arranged')},ensure_ascii=False))


if __name__=='__main__':main()
