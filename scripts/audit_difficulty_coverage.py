"""Audit manifest density coverage and propose a wider development listening panel.

Density is only a workload proxy, NOT osu star rating or subjective difficulty.
Manifest availability, actual tape exposure, and cached local controls are separate.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from beatfcn.train.data import selected_for_training, dup_only, phira_tag

BINS = [(0,2),(2,4),(4,6),(6,8),(8,12),(12,float('inf'))]
NAMES = ['0–2','2–4','4–6','6–8','8–12','12+']


def density(row):
    return row['n_events']/row['audio_sec']


def bucket(row):
    d=density(row)
    return next(name for (lo,hi),name in zip(BINS,NAMES) if lo<=d<hi)


def describe(rows):
    values=sorted(density(r) for r in rows)
    return dict(charts=len(rows),unique_audio=len({r['audio_sha256'] for r in rows}),
                density_histogram={name:sum(bucket(r)==name for r in rows) for name in NAMES},
                density_quantiles={str(q):values[round((len(values)-1)*q)] if values else None for q in (.1,.25,.5,.75,.9)},
                hold_ratio_histogram=dict(Counter('0' if not r['types'].get('hold',0) else
                    '<10%' if r['types'].get('hold',0)/r['n_events']<.1 else
                    '10–30%' if r['types'].get('hold',0)/r['n_events']<.3 else '30%+'
                    for r in rows if r['n_events'])),
                named_difficulty_top20=dict(Counter(r.get('difficulty_label','?') for r in rows).most_common(20)))


def main():
    manifest_path=ROOT/'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl'
    rows=[json.loads(line) for line in manifest_path.read_text().splitlines()]
    eligible=[r for r in rows if r.get('mode') in ('mania_4k','phira') and selected_for_training(r)
              and (r.get('kept') or dup_only(r)) and r.get('audio_sec',0)>0 and r.get('n_events',0)>0]
    by_id={r['chart_id']:r for r in eligible}
    benchmark=json.loads((ROOT/'bench/capability-v1/manifest.json').read_text())['rows']
    observed={}
    for path in (ROOT/'reports/condition_v1_benchmark/records').glob('*.json'):
        doc=json.loads(path.read_text())
        if doc['request_kind']=='from_scratch':
            observed[doc['chart_id']]=by_id[doc['chart_id']]
    cells={}
    for mode in ('mania_4k','phira'):
        for split in ('train','val','test'):
            cells[f'{mode}/{split}']=describe([r for r in eligible if r['mode']==mode and r['split']==split])
        cells[f'{mode}/benchmark_manifest']=describe([r for r in benchmark if r['mode']==mode])
        cells[f'{mode}/evaluated_from_scratch']=describe([r for r in observed.values() if r['mode']==mode])
    candidates=[];seen_audio=set()
    for mode in ('mania_4k','phira'):
        for name in NAMES:
            pool=[r for r in eligible if r['mode']==mode and r['split']=='val' and bucket(r)==name and r.get('kept')]
            # Three independent audio songs, each with a different hold burden where available.
            pool.sort(key=lambda r:(r['types'].get('hold',0)/r['n_events'],r['chart_id']))
            selected=[]
            for fraction in (.1,.5,.9):
                available=[r for r in pool if r['audio_sha256'] not in seen_audio]
                if not available:break
                row=available[round((len(available)-1)*fraction)]
                seen_audio.add(row['audio_sha256']);selected.append(row)
            for row in selected:
                candidates.append(dict(chart_id=row['chart_id'],audio_sha256=row['audio_sha256'],mode=mode,
                    title=row['title'],native_difficulty=row.get('difficulty_label'),split='val',
                    density_bucket=name,density_nps=density(row),duration_s=row['audio_sec'],
                    hold_head_ratio=row['types'].get('hold',0)/row['n_events'],
                    candidate_only=True,canonical_and_request_validation_pending=True))
    same_audio=defaultdict(list)
    for row in eligible:
        if row['split']=='train':same_audio[(row['mode'],row['audio_sha256'])].append(row)
    pairing={mode:dict(songs=sum(k[0]==mode for k in same_audio),
                 multiple_density_buckets=sum(k[0]==mode and len({bucket(r) for r in rs})>1 for k,rs in same_audio.items()))
                 for mode in ('mania_4k','phira')}
    # Existing cached scope audit was reconstructed, not observed training exposure.
    coverage=json.loads((ROOT/'reports/condition_coverage_2026-10-25.json').read_text())
    conditions={}
    for mode in ('mania_4k','phira'):
        for variant in ('current_request','density_window'):
            selected=[r for r in coverage['records'] if r['mode']==mode and r['variant']==variant]
            if not selected:continue
            fields={}
            for name in selected[0]['fields']:
                vals=[r['fields'][name] for r in selected]
                fields[name]=dict(present=sum(v['requested_band'] is not None for v in vals),samples=len(vals),
                    label_inconsistent=sum(v['label_inconsistent'] for v in vals),
                    comparable_local=sum(v['local_band_differs'] is not None for v in vals),
                    local_band_differs=sum(v['local_band_differs'] is True for v in vals))
            conditions[f'{mode}/{variant}']=fields
    report=dict(schema='beatfcn.difficulty_coverage.v1',manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                scope='eligible manifest charts; density proxy only; NOT measured training draw distribution',
                cells=cells,same_audio_training_pair_candidates=pairing,
                phira_train_tags=dict(Counter(phira_tag(r) for r in eligible if r['mode']=='phira' and r['split']=='train')),
                reconstructed_condition_coverage=conditions,candidate_panel=candidates,
                limitations=['No numeric osu star rating in this manifest; arbitrary difficulty names are not comparable.',
                    'Audio-duration mean density understates short bursts and long silent intros.',
                    'Manifest metadata precedes full canonical validation; candidates have not yet been accepted.',
                    'Exact 32000-window tape per-chart difficulty exposure has not been recovered.',
                    'Equal density does not imply equal hand strain, jack speed, hold-release coordination or musical complexity.'],
                new_paid_dispatches=0)
    assert report['manifest_sha256']==coverage['manifest_sha256']
    dest=ROOT/'reports/difficulty_coverage_2026-10-25.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    panel=dict(schema='beatfcn.difficulty_panel.candidates.v1',status='prepared_not_dispatched',
               selection='3 validation audio songs per mode x mean-density bucket, spread across hold ratio',
               rows=candidates,missing_cells=[f'{m}/{b}' for m in ('mania_4k','phira') for b in NAMES
                      if sum(r['mode']==m and r['density_bucket']==b for r in candidates)<3],
               acceptance='Validate native/canon timing, peak density, chords, jacks, hold occupancy and applicable types before freezing.',
               no_training_or_gpu_evaluation_started=True)
    (ROOT/'bench/rhythm-v2/difficulty-panel-candidates.json').write_text(json.dumps(panel,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(cells={k:v for k,v in cells.items() if k.startswith('mania_4k/')},pairing=pairing,
                         phira_train_tags=report['phira_train_tags'],candidate_count=len(candidates)),ensure_ascii=False))


if __name__=='__main__':main()
