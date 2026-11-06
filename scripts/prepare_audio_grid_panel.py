"""Freeze a reproducible, audio-deduplicated D48 diagnostic panel from local files."""
import hashlib
import json
from pathlib import Path
import time

from beatfcn.ingest.coverage import native_tier, phira_native_difficulty
from scripts.audit_plain_packages import package_members

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('inputs/data_expansion_v1/corpus/manifest/expansion-candidates.jsonl')
POLICY = Path('bench/rhythm-v2/data-scale-policy-v1.json')
OUTPUT = Path('inputs/diagnostic_panels/d48-audio-grid')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    if (ROOT / OUTPUT / 'panel.json').exists():
        raise FileExistsError('D48 panel is already frozen')
    policy = json.loads((ROOT / POLICY).read_text())
    rows = [json.loads(line) for line in (ROOT / MANIFEST).read_text().splitlines()]
    eligible = {}
    for row in rows:
        if (row.get('technical_audit_pass') is not True or row.get('quarantined')
                or row.get('split_conflict') or not 30 <= row.get('audio_sec', 0) <= 300):
            continue
        mode = row.get('mode')
        if mode == 'mania_4k':
            value = row.get('acquisition', {}).get('star_rating')
            evidence = 'per_chart_native_star_rating'
        elif mode == 'phira':
            value, evidence = phira_native_difficulty(row)
        else:
            continue
        tier = native_tier(mode, value, policy)
        if tier is None:
            continue
        eligible.setdefault((mode, tier), []).append((row, value, evidence))
    selected, used_audio = [], set()
    for mode in ('mania_4k', 'phira'):
        for tier in ('low', 'medium', 'high'):
            candidates = sorted(eligible[(mode, tier)], key=lambda item:
                hashlib.sha256(f'D48:20261025:{item[0]["chart_id"]}'.encode()).hexdigest())
            chosen = 0
            for row, value, evidence in candidates:
                audio_sha = row['audio_sha256']
                if audio_sha in used_audio:
                    continue
                source, native_id = row['source'], row['native_id']
                canon = Path(f'inputs/data_expansion_v1/corpus/canon/{source}/{native_id}.json')
                chart = json.loads((ROOT / canon).read_text())
                if chart['chart_id'] != row['chart_id'] or chart['audio_sha256'] != audio_sha:
                    raise ValueError('canonical identity mismatch')
                package = None
                if mode == 'mania_4k':
                    audio = Path(f'inputs/data_expansion_v1/corpus/raw/osu/audio/{audio_sha}.{row["audio_ext"]}')
                else:
                    archive = Path(f'inputs/data_expansion_v1/corpus/raw/phira/zips/{native_id}.zip')
                    _, members = package_members(ROOT / archive)
                    content = members['music']['bytes']
                    if hashlib.sha256(content).hexdigest() != audio_sha:
                        raise ValueError('archive music identity mismatch')
                    audio = OUTPUT / 'audio' / f'{audio_sha}.{row["audio_ext"]}'
                    (ROOT / audio).parent.mkdir(parents=True, exist_ok=True)
                    if (ROOT / audio).exists():
                        if sha(ROOT / audio) != audio_sha:
                            raise ValueError('existing extracted music differs')
                    else:
                        with (ROOT / audio).open('xb') as stream:
                            stream.write(content)
                    package = dict(path=str(archive), sha256=sha(ROOT / archive),
                                   music_member=members['music']['path'])
                if sha(ROOT / audio) != audio_sha:
                    raise ValueError('audio identity mismatch')
                offset = row['offset_decision']
                clock_shift_us = 1000 * (int(offset.get('shift_ms') or 0) + int(offset.get('c_ms') or 0))
                selected.append(dict(chart_id=row['chart_id'], title=row['title'], mode=mode,
                    tier=tier, native_difficulty=value, difficulty_evidence=evidence,
                    split=row['split'], song_group=row['song_group'], audio=str(audio),
                    audio_sha256=audio_sha, declared_audio_sec=row['audio_sec'],
                    canonical=str(canon), canonical_sha256=sha(ROOT / canon), package=package,
                    offset_decision=offset, frozen_model_shift_us=clock_shift_us,
                    measured_controls=row['measured_controls'], types=row['types'],
                    events=len(chart['events']), fakes_excluded=len(chart.get('fakes', [])),
                    feature_cache_status=row['feature_cache_status'], training_ready=row['training_ready']))
                used_audio.add(audio_sha)
                chosen += 1
                if chosen == 2:
                    break
            if chosen != 2:
                raise ValueError(f'insufficient unique audio in {mode}/{tier}')
    panel = dict(id='D48', frozen_at_unix=time.time(), manifest=str(MANIFEST),
        manifest_sha256=sha(ROOT / MANIFEST), policy=str(POLICY), policy_sha256=sha(ROOT / POLICY),
        selection='technical pass, no quarantine/split conflict, 30..300s; SHA256 D48:20261025:chart_id order; two per mode/native tier, globally distinct audio',
        eligible_counts={f'{m}/{t}': len(v) for (m, t), v in eligible.items()},
        difficulty_scope='Frozen native acquisition tiers; not calibrated operational burden.',
        formal_training_admission=False, label_corrections_applied=False,
        selection_uses_audio_predictions=False, rows=selected)
    write_once(ROOT / OUTPUT / 'panel.json', panel)
    # The inference runner reads this label-free manifest. Canonical charts,
    # native BPM, offsets and targets are available only to the later auditor.
    write_once(ROOT / OUTPUT / 'audio-inputs.json', dict(id='D48', rows=[
        {k: row[k] for k in ('chart_id', 'audio', 'audio_sha256')} for row in selected]))
    print(json.dumps(dict(charts=len(selected), unique_audio=len(used_audio),
        audio_minutes=sum(r['declared_audio_sec'] for r in selected) / 60,
        output=str(OUTPUT), new_cloud_spend_usd=0)))


if __name__ == '__main__':
    main()
