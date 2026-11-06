"""Freeze nested real-song candidates before source and cache admission.

No network, training, output-label edits, or changes to the production manifest.
"""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

from beatfcn.ingest.coverage import native_tier
from beatfcn.train.data import dup_only,selected_for_training
from beatfcn.train.local_cohort import connected_songs,estimate_feature_bytes,stratified_song_order

ROOT=Path(__file__).resolve().parents[1]
SOURCES=dict(manifest='inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl',
    overlay='reports/d57_phira_strata_audit_v2.corpus-overlay.jsonl',
    mania='inputs/data_scale_v2/catalog_mania_v2/mania/catalog.jsonl',
    policy='bench/rhythm-v2/data-scale-policy-v1.json',diagnostic='inputs/d65_local/panel.json')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def row_sha(row):return hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def read_lines(path):return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    output=ROOT/'inputs/d66_local_cohort';output.mkdir(parents=True,exist_ok=True)
    target=output/'candidates-v1.json'
    if target.exists():raise FileExistsError(target)
    rows=read_lines(ROOT/SOURCES['manifest']);identity,splits=connected_songs(rows)
    overlay={r['chart_id']:r for r in read_lines(ROOT/SOURCES['overlay']) if r['manifest']==SOURCES['manifest']}
    mania={str(b['id']):b for s in read_lines(ROOT/SOURCES['mania']) for b in s['beatmaps']
        if b.get('mode_int')==3 and b.get('cs')==4 and b.get('status') in ('ranked','approved')}
    policy=json.loads((ROOT/SOURCES['policy']).read_text())
    diagnostic=json.loads((ROOT/SOURCES['diagnostic']).read_text())['records']
    excluded={identity[r['chart_id']] for r in diagnostic if r['chart_id'] in identity}
    excluded_reasons=Counter();eligible=[];original={r['chart_id']:r for r in rows}
    for row in rows:
        cid=row['chart_id'];component=identity.get(cid);mode=row.get('mode')
        if component is None or mode not in ('phira','mania_4k'):
            excluded_reasons['missing_identity_or_other_mode']+=1;continue
        if len(splits[component])!=1 or splits[component] not in ({'train'},{'val'}):
            excluded_reasons['split_conflict_or_test']+=1;continue
        if component in excluded:
            excluded_reasons['prior_d65_diagnostic_component']+=1;continue
        if not selected_for_training(row) or not (row.get('kept') or dup_only(row)):
            excluded_reasons['prior_metadata_or_technical_filter']+=1;continue
        duration=row.get('audio_sec')
        if not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<8:
            excluded_reasons['missing_or_short_duration']+=1;continue
        if row.get('offset_decision',{}).get('decision_version')!=2:
            excluded_reasons['missing_offset_v2']+=1;continue
        if mode=='phira':
            joined=overlay.get(cid)
            if not joined or joined['source_row_sha256']!=row_sha(row):
                excluded_reasons['phira_metadata_identity_missing_or_changed']+=1;continue
            if not joined['metadata_candidate']:
                excluded_reasons['phira_metadata_screen_or_stratum']+=1;continue
            meta=joined['current_catalog_audit'];stratum=meta['stratum']['key']
            difficulty=meta['stratum']['native_difficulty'];difficulty_source='D57 metadata ID join; package revision unverified'
        else:
            meta=mania.get(row['native_id'],{});difficulty=meta.get('difficulty_rating')
            tier=native_tier(mode,difficulty,policy)
            if tier is None:excluded_reasons['mania_unknown_native_difficulty']+=1;continue
            stratum=tier;difficulty_source='Frozen osu catalog native ID join; original checksum verification pending'
        eligible.append(dict(chart_id=cid,component=component,mode=mode,split=row['split'],stratum=stratum,
            native_difficulty=difficulty,difficulty_source=difficulty_source,audio_sha256=row['audio_sha256'],
            song_group=row['song_group'],audio_sec=duration,row_sha256=row_sha(row),
            native_content_verified=False,clock_verified=False,feature_verified=False,training_ready=False))
    orders={};used=set()
    for split,counts in [('val',{'phira':24,'mania_4k':8}),('train',{'phira':225,'mania_4k':75})]:
        for mode,n in counts.items():
            available=[r for r in eligible if r['split']==split and r['mode']==mode]
            order=stratified_song_order(available,seed=f'D66:20261025:{split}:{mode}',used=used)
            if len(order)<n:raise ValueError(f'Insufficient independent candidate songs: {split}/{mode} {len(order)}<{n}')
            orders[f'{split}/{mode}']=order[:n];used.update(r['component'] for r in order[:n])
    records=[];by_component=defaultdict(list)
    for row in eligible:by_component[row['component']].append(row)
    for key,order in orders.items():
        for rank,representative in enumerate(order):
            same=[r for r in by_component[representative['component']] if r['mode']==representative['mode']]
            # Use exactly the same audio bytes for initial paired chart candidates.
            same=[r for r in same if r['audio_sha256']==representative['audio_sha256']]
            same.sort(key=lambda r:(r['stratum']==representative['stratum'],row_sha(r)))
            selected=[representative]+[r for r in same if r['chart_id']!=representative['chart_id']][:2]
            records.append(dict(key=key,rank=rank,component=representative['component'],
                representative_chart_id=representative['chart_id'],representative_stratum=representative['stratum'],
                audio_sha256=representative['audio_sha256'],audio_sec=representative['audio_sec'],
                candidate_charts=selected,estimated_feature_bytes=estimate_feature_bytes(representative['audio_sec'])))
    def members(size):
        return [r['component'] for r in records if r['key'].startswith('train/')
            and r['rank']<({'phira':size*3//4,'mania_4k':size//4}[r['key'].split('/')[1]])]
    tiers={str(size):members(size) for size in (100,300)}
    if not set(tiers['100'])<set(tiers['300']):raise AssertionError('Nested train selection failed')
    validation=[r['component'] for r in records if r['key'].startswith('val/')]
    if set(validation)&set(tiers['300']):raise AssertionError('Song split overlap')
    selected={c['chart_id'] for r in records for c in r['candidate_charts']}
    frozen_rows=output/'source-rows-v1.jsonl'
    frozen_rows.write_text(''.join(json.dumps(original[cid],ensure_ascii=False)+'\n' for cid in sorted(selected)))
    report=dict(id='D66-local-nested-candidates-v1',complete=True,training_ready=False,
        optimizer_updates=0,new_cloud_compute_calls=0,source_sha256={p:sha(ROOT/p) for p in SOURCES.values()},
        code_sha256={p:sha(ROOT/p) for p in ['scripts/prepare_local_cohort_candidates.py','beatfcn/train/local_cohort.py']},
        source_rows=str(frozen_rows.relative_to(ROOT)),source_rows_sha256=sha(frozen_rows),
        records=records,train_tiers=tiers,validation=validation,exclusions=dict(excluded_reasons),
        source_component_count=len(splits),source_split_conflict_components=sum(len(s)>1 for s in splits.values()),
        excluded_d65_components=len(excluded),
        per_group={key:dict(songs=len(order),representative_strata=dict(Counter(r['stratum'] for r in order))) for key,order in orders.items()},
        candidate_charts=len(selected),same_audio_multiple_chart_songs=sum(len(r['candidate_charts'])>1 for r in records),
        estimated_feature_bytes=sum(r['estimated_feature_bytes'] for r in records),
        disk_free_bytes=shutil.disk_usage(ROOT).free,disk_minimum_free_reserve_bytes=32*2**30,
        pending=['Native package/source identity and semantic screen','Canonical target and clock audit',
                 'Rhythm target dedup and actual same-scope condition support','Actual feature headers/bytes/finiteness',
                 'Read-only transfer plan and storage bound','Representative playable draft evaluation'],
        scope='Frozen candidate songs only. Metadata strata and cached estimates do not grant admission. No window counts as a song.')
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['complete','per_group','candidate_charts','same_audio_multiple_chart_songs','estimated_feature_bytes','source_split_conflict_components','exclusions']}))


if __name__=='__main__':main()
