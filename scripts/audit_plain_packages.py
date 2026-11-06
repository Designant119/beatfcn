"""Native-package preservation, decoded audio and frozen-split audit."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import yaml

from beatfcn.events.canonical import build_model_view, UnsupportedFormat
from beatfcn.events.phira import load_package
from beatfcn.events.controls import raw_controls
from beatfcn.ingest import pipeline
from beatfcn.ingest.coverage import native_tier, phira_native_difficulty, policy_identity
from beatfcn.ingest.expansion import incremental_splits
from scripts.audit_phigros_resource import sha, write, rhythm_hash, audio_metadata
from beatfcn.train.data import check_stacks, selected_for_training, PHIRA_SELECTION_VERSION

ROOT=Path(__file__).resolve().parents[1]


def package_members(path):
    """Read only; reject ambiguous basenames instead of choosing an arbitrary file."""
    with zipfile.ZipFile(path) as archive:
        names={}
        for e in archive.infolist():
            if e.is_dir():continue
            key=e.filename.lower().rsplit('/',1)[-1]
            if key in names:raise ValueError('ambiguous archive basename: '+key)
            names[key]=e.filename
        info_name=names.get('info.yml') or names.get('info.yaml')
        if info_name is None:raise UnsupportedFormat('missing info.yml')
        info=yaml.safe_load(archive.read(info_name).decode('utf-8-sig'))
        if not isinstance(info,dict):raise ValueError('info must be a mapping')
        found={}
        for key in ('chart','music'):
            value=str(info.get(key) or '').lower().rsplit('/',1)[-1]
            if value not in names:raise ValueError(key+' missing from archive')
            found[key]=dict(path=names[value],bytes=archive.read(names[value]))
        return info,found


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    args=parser.parse_args();plan=json.loads((ROOT/args.plan).read_text());start=time.monotonic()
    for name,expected in plan['input_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('input identity drift: '+name)
    output=ROOT/plan['output_root'];report_path=ROOT/plan['result']
    if report_path.exists():raise FileExistsError('previous audit receipt exists')
    old=[json.loads(line) for name in plan['frozen_manifests'] for line in (ROOT/name).read_text().splitlines()]
    old_ids={r['chart_id'] for r in old}
    policy=json.loads((ROOT/plan['difficulty_policy']).read_text())
    transfer=json.loads((ROOT/plan['transfer']).read_text());assert transfer['complete']
    records={r['id']:r for r in transfer['records']}
    acquisition=json.loads((ROOT/plan['acquisition']).read_text())
    excluded=[dict(id=i['id'],chart_id=f"phira:{i['id']}",reason='already_in_frozen_corpus_identity')
              for i in acquisition['items'] if f"phira:{i['id']}" in old_ids]
    report=dict(id=plan.get('id', 'D34'),complete=False,records=[],existing_identity_exclusions=excluded,
                plan_sha256=sha(ROOT/args.plan),training_ready=False,
                new_cloud_compute_usd=0,optimizer_updates=0,applied_clock_shift_us=0)
    def save():
        report['wall_s']=time.monotonic()-start;write(report_path,report)
    save()
    for item in acquisition['items']:
        if f"phira:{item['id']}" in old_ids:continue
        if time.monotonic()-start>plan['audit_deadline_s']:raise TimeoutError('audit deadline')
        receipt=records[item['id']];meta=json.loads((ROOT/item['metadata']).read_text())
        difficulty, difficulty_evidence = phira_native_difficulty(meta)
        row=dict(chart_id=f"phira:{item['id']}",source='phira',native_id=str(item['id']),title=meta['name'],
            artist=meta.get('composer',''),charter=meta.get('charter',''),mode='phira',metadata_path=item['metadata'],
            metadata_sha256=sha(ROOT/item['metadata']),native_difficulty=difficulty,native_level=meta['level'],
            difficulty_label=meta['level'],difficulty_evidence=difficulty_evidence,
            rating=meta.get('rating'),rating_count=meta.get('ratingCount'),
            selection_policy=PHIRA_SELECTION_VERSION,
            difficulty_tier=native_tier('phira',difficulty,policy),difficulty_policy_sha256=policy_identity(policy),
            difficulty_review_status='declared_native_tier_burden_review_pending',
            structural_audit_pass=False,technical_audit_pass=False,training_ready=False,
            feature_cache_status='not_encoded',clock_calibration='pending_source_recipe_and_alignment',
            native_authorship_status='no_explicit_conversion_in_metadata_not_proof_of_authorship',
            visibility_audit='pending; legal artistic drag retained',quarantined=True)
        if row['chart_id'] in old_ids:raise ValueError('already existing chart identity')
        if receipt['status']!='downloaded_archive_verified':
            row.update(status='not_available',error=receipt.get('error',receipt['status']))
            report['records'].append(row);save();continue
        path=ROOT/receipt['path']
        if sha(path)!=receipt['sha256']:raise ValueError('archive identity drift')
        row.update(archive_path=receipt['path'],archive_sha256=receipt['sha256'])
        try:
            if not selected_for_training(row):
                raise ValueError('metadata does not pass current training selection')
            info,members=package_members(path)
            row['info_format']=str(info.get('format') or '')
            chart=load_package(path)
            raw=json.loads(members['chart']['bytes'].decode('utf-8-sig'))
            raw_count=sum(len(line.get(side) or []) for line in raw.get('judgeLineList',[])
                          for side in (('notes',) if 'META' in raw else ('notesAbove','notesBelow')))
            audio=members['music']['bytes'];audio_sha=pipeline.sha256(audio)
            audio_path=output/'audio'/f'{audio_sha}.audio';audio_path.parent.mkdir(parents=True,exist_ok=True)
            if audio_path.exists():
                if sha(audio_path)!=audio_sha:raise ValueError('cached audio identity drift')
            else:audio_path.write_bytes(audio)
            ameta=audio_metadata(audio_path)
            if ameta['audio_sec']>1200:raise ValueError('audio duration outside bounded diagnostic support')
            run=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(audio_path),
                '-vn','-f','f32le','-ac','1','-ar','24000','pipe:1'],capture_output=True,check=True,timeout=30)
            wave=np.frombuffer(run.stdout,dtype='<f4')
            if not len(wave) or not np.isfinite(wave).all():raise ValueError('decoded audio empty or nonfinite')
            audio_us=round(len(wave)*1e6/24000)
            row.update(audio_sha256=audio_sha,audio_path=str(audio_path.relative_to(ROOT)),
                codec=ameta['codec'],audio_sec=len(wave)/24000,decoded_audio_stats=dict(shape=list(wave.shape),
                min=float(wave.min()),max=float(wave.max()),mean=float(wave.mean()),std=float(wave.std()),nonfinite=0))
            del wave,run,audio
            row['native_issue_audit'] = dict(policy='existing_DATA_RULES_v4',
                maximum_source_stack=check_stacks(chart.events), removed_or_merged_events=0)
            reasons=sorted({i.code for i in chart.issues
                if i.severity=='quarantine' and i.code != 'duplicate_object'})
            reasons+=pipeline.chart_validation(chart,audio_us)
            if raw_count!=len(chart.events)+len(chart.fakes):reasons.append('raw_canonical_count_mismatch')
            view=build_model_view('phira',(),chart.events)
            target=[e for g in view.target_groups for e in g.events]
            if len(target)!=len(chart.events):reasons.append('canonical_model_count_mismatch')
            reasons += [i.code for i in view.quarantine]
            row.update(status='parsed',source_format=chart.source_format,format_version=chart.format_version,
                raw_notes=raw_count,canonical_notes=len(chart.events),fakes=len(chart.fakes),model_notes=len(target),
                structural_reasons=sorted(set(reasons)),structural_audit_pass=not reasons,
                types=dict(Counter(e.type for e in chart.events)),fake_types=dict(Counter(e.type for e in chart.fakes)),
                warnings=sorted({i.code for i in chart.issues if i.severity!='quarantine'}),
                maximum_group_size=max((len(g.events) for g in view.target_groups),default=0),
                burden=raw_controls('phira',target,[(0,audio_us)]),
                pure_rhythm_hash=rhythm_hash(target) if not view.quarantine and len(target)==len(chart.events) else None)
            canonical=pipeline.chart_json(chart,dict(chart_id=row['chart_id'],audio_sha256=audio_sha,
                raw_chart_path=f"{receipt['path']}::{members['chart']['path']}"),
                dict(stage='structural_decoded_audio_only',applied_clock_shift_us=0,clock_calibration='pending',structural_reasons=row['structural_reasons']))
            cp=output/'canon'/f"{item['id']}.json";write(cp,canonical);row['canonical_path']=str(cp.relative_to(ROOT));row['canonical_sha256']=sha(cp)
        except (OSError,ValueError,KeyError,UnsupportedFormat,subprocess.SubprocessError,yaml.YAMLError,zipfile.BadZipFile) as exc:
            row.update(status='error',error=f'{type(exc).__name__}: {exc}')
        report['records'].append(row);save()
        print(json.dumps(dict(chart_id=row['chart_id'],status=row['status'],structural_pass=row['structural_audit_pass'])),flush=True)
    rows=incremental_splits(old,report['records']);groups=defaultdict(list)
    for r in rows:
        if r.get('pure_rhythm_hash'):groups[(r['audio_sha256'],r['pure_rhythm_hash'])].append(r)
    old_hashes=defaultdict(list)
    for r in old:
        if r.get('pure_rhythm_hash'):old_hashes[(r['audio_sha256'],r['pure_rhythm_hash'])].append(r['chart_id'])
    for group in groups.values():
        ids=sorted(r['chart_id'] for r in group)
        for r in group:
            key=(r['audio_sha256'],r['pure_rhythm_hash']);r['within_batch_duplicate_of']=ids[0] if r['chart_id']!=ids[0] else None
            r['exact_known_source_duplicate_ids']=old_hashes.get(key,[])
    for r in rows:
        r['cross_source_content_audit']='pending' if r['split_evidence']['existing_chart_ids'] else 'no_link_in_frozen_name_audio_keys_other_aliases_unverified'
    report['records']=rows
    report['summary']=dict(raw_packages=len(acquisition['items']),new_candidate_packages=len(rows),
        existing_identity_exclusions=len(excluded),downloaded=transfer['downloaded'],parsed=sum(r['status']=='parsed' for r in rows),
        structural_pass=sum(r['structural_audit_pass'] for r in rows),
        structural_unique_train_tiers=dict(Counter(r['difficulty_tier'] for r in rows if r['structural_audit_pass'] and r['split']=='train'
            and not r.get('within_batch_duplicate_of') and not r.get('exact_known_source_duplicate_ids'))),
        raw_tiers=dict(Counter(r['difficulty_tier'] for r in rows)),split_conflicts=sum(r['split_conflict'] for r in rows),
        fakes_preserved=sum(r.get('fakes',0) for r in rows),playable_notes=sum(r.get('canonical_notes',0) for r in rows),
        training_ready_charts=0,errors=[dict(chart_id=r['chart_id'],error=r['error']) for r in rows if r.get('error')])
    manifest=output/'manifest/candidates.jsonl';manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    # The audit never rewrites existing split records or source/canonical files.
    for name in plan['frozen_manifests']:
        if sha(ROOT/name)!=plan['input_sha256'][name]:raise ValueError('old manifest changed during audit')
    report.update(complete=True,manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha(manifest),old_manifests_unchanged=True)
    save();print(json.dumps(report['summary'],ensure_ascii=False))


if __name__=='__main__':main()
