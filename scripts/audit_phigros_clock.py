"""D26: measure source clocks without changing labels or using model predictions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from beatfcn.audio.preprocess import CELL_SAMPLES, HOP, SR
from beatfcn.ingest.offset import estimate_offset, flux_envelope
from beatfcn.ingest.pipeline import mel10

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def health(array):
    finite = np.isfinite(array)
    result = dict(shape=list(array.shape), dtype=str(array.dtype),
                  nan=int(np.isnan(array).sum()), inf=int(np.isinf(array).sum()))
    if array.size == 0 or not finite.all():
        raise ValueError(f'invalid tensor: {result}')
    result.update(min=float(array.min()), max=float(array.max()),
                  mean=float(array.mean(dtype=np.float64)), std=float(array.std(dtype=np.float64)))
    return result


def decode(path, timeout_s):
    result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
        '-i', str(path), '-vn', '-f', 'f32le', '-ac', '1', '-ar', str(SR), 'pipe:1'],
        capture_output=True, check=True, timeout=timeout_s)
    return np.frombuffer(result.stdout, dtype='<f4').copy()


def estimate(heads, envelope):
    result = estimate_offset(heads, env=envelope)
    result.pop('agree_counts', None)
    return result


def reliable(result, minimum_confidence):
    return (result.get('status') == 'ok' and result['conf'] >= minimum_confidence
            and not result['edge'])


def codec_control(plan, output):
    """Known event times through actual encoding, decoding, and the production mel recipe."""
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(plan['seed'])
    # Non-periodic attacks prevent a neighbouring beat from being an equally valid alignment.
    heads = np.cumsum(rng.uniform(.35, .7, 64)) + 1
    wave = np.zeros(int((heads[-1] + 1) * SR), dtype=np.float32)
    for head in heads:
        start = int(round(head * SR))
        n = int(.08 * SR)
        burst = rng.normal(size=n) * np.exp(-np.arange(n) / (.013 * SR)) * .12
        wave[start:start+n] += burst.astype(np.float32)
    pcm = output / 'known-attacks.f32'
    pcm.write_bytes(wave.astype('<f4').tobytes())
    records = []
    for name, codec, extension in [('pcm', 'pcm_f32le', 'wav'), ('vorbis', 'vorbis', 'ogg')]:
        path = output / f'known-attacks.{extension}'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-y',
            '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', str(pcm), '-c:a', codec,
            '-strict', 'experimental', '-ac', '2',
            str(path)], check=True, capture_output=True, timeout=plan['decode_timeout_s'])
        decoded = decode(path, plan['decode_timeout_s'])
        mel = mel10(decoded)
        env = flux_envelope(mel)
        estimates = {str(shift): estimate(heads + shift/1000., env)
                     for shift in plan['control_chart_shifts_ms']}
        zero = estimates['0']
        errors = {shift: value.get('delta_ms', 999) - zero.get('delta_ms', 0) + int(shift)
                  for shift, value in estimates.items()}
        passed = (all(reliable(x, plan['min_confidence']) for x in estimates.values())
                  and max(abs(x) for x in errors.values()) <= plan['control_tolerance_ms'])
        records.append(dict(codec=name, encoder=codec, path=str(path.relative_to(ROOT)), sha256=sha(path),
            decoded_samples=len(decoded), decoded_duration_error_ms=(len(decoded)-len(wave))*1000/SR,
            waveform=health(decoded), mel=health(mel), estimates=estimates,
            relative_shift_errors_ms=errors, passed=passed))
    phase_difference = records[1]['estimates']['0']['delta_ms'] - records[0]['estimates']['0']['delta_ms']
    return dict(records=records, known_attacks_sec=heads.tolist(), seed=plan['seed'],
        encoder='local ffmpeg; not proof of the original archive encoder delay',
        phase_difference_ms=phase_difference,
        passed=all(x['passed'] for x in records) and abs(phase_difference) <= plan['control_tolerance_ms'])


def chart_clock(row, canon, envelope, samples, plan):
    events = canon['events']
    heads = {
        'all': np.unique([e['onset_us']/1e6 for e in events]),
        'tap_hold': np.unique([e['onset_us']/1e6 for e in events if e['type'] in ('tap', 'hold')]),
        'drag': np.unique([e['onset_us']/1e6 for e in events if e['type'] == 'drag']),
        'flick': np.unique([e['onset_us']/1e6 for e in events if e['type'] == 'flick'])}
    whole = {name: estimate(values, envelope) for name, values in heads.items()}
    thirds = []
    for i in range(3):
        lo, hi = len(envelope)*i//3, len(envelope)*(i+1)//3
        thirds.append(estimate(heads['tap_hold'] - lo*HOP/SR, envelope[lo:hi]))
    stable = all(reliable(x, plan['min_confidence']) for x in thirds)
    span = max(x['delta_ms'] for x in thirds)-min(x['delta_ms'] for x in thirds) if stable else None
    bounds = [e['event_id'] for e in events if e['onset_us'] < 0 or e['end_us']*SR > samples*1_000_000]
    views_difference = (whole['all']['delta_ms']-whole['tap_hold']['delta_ms']
        if all(reliable(whole[k], plan['min_confidence']) for k in ('all', 'tap_hold')) else None)
    reasons = []
    if row['split'] != 'train':
        reasons.append('not_training_split')
    if not row['structural_audit_pass'] or row.get('source_duplicate_of'):
        reasons.append('structural_or_duplicate_exclusion')
    if bounds:
        reasons.append('decoded_audio_bounds')
    if not reliable(whole['tap_hold'], plan['min_confidence']):
        reasons.append('primary_estimate_unreliable')
    if not stable:
        reasons.append('not_three_reliable_thirds')
    elif span > plan['max_thirds_span_ms']:
        reasons.append('within_song_drift')
    elif abs(whole['tap_hold'].get('delta_ms', 999)-np.median([x['delta_ms'] for x in thirds])) > plan['max_whole_thirds_difference_ms']:
        reasons.append('whole_thirds_disagreement')
    if views_difference is not None and abs(views_difference) > plan['max_view_difference_ms']:
        reasons.append('all_vs_tap_hold_disagreement')
    return dict(chart_id=row['chart_id'], audio_sha256=row['audio_sha256'], name=row['name'],
        source_chart_id=row['source_chart_id'], split=row['split'], tier=row['difficulty_tier'],
        status='ok', notes=len(events), types=dict(Counter(e['type'] for e in events)),
        decoded_bounds_violations=bounds, whole=whole, primary_thirds=thirds,
        reliable_thirds=sum(reliable(x, plan['min_confidence']) for x in thirds),
        thirds_span_ms=span, all_vs_tap_hold_ms=views_difference,
        calibration_eligible=not reasons, calibration_exclusions=reasons,
        interpretation='Unavailable or weak acoustic evidence is not evidence of poor chart quality.')


def calibration(records, plan):
    # Defensive split check: a cached eligibility flag can never admit held-out songs.
    groups = defaultdict(list)
    for r in records:
        if r.get('calibration_eligible') and r.get('split') == 'train' and r.get('status') == 'ok':
            groups[r['audio_sha256']].append(r)
    songs, excluded = [], []
    for audio_sha, rows in sorted(groups.items()):
        deltas = [r['whole']['tap_hold']['delta_ms'] for r in rows]
        if max(deltas)-min(deltas) > plan['max_cross_chart_span_ms']:
            excluded.append(dict(audio_sha256=audio_sha, reason='same_audio_chart_disagreement', deltas_ms=deltas))
        else:
            songs.append(dict(audio_sha256=audio_sha, delta_ms=float(np.median(deltas)),
                chart_ids=[r['chart_id'] for r in rows], tiers=sorted({r['tier'] for r in rows})))
    if not songs:
        return dict(status='insufficient_evidence', qualified_songs=[], excluded=excluded, applied=False)
    deltas = np.array([r['delta_ms'] for r in songs])
    median = float(np.median(deltas))
    abs_p90 = float(np.percentile(np.abs(deltas-median), 90))
    by_tier = {tier: [r['delta_ms'] for r in songs if tier in r['tiers']] for tier in ('low', 'medium')}
    tier_stats = {tier: dict(independent_audio=len(values), median_delta_ms=float(np.median(values)) if values else None)
                  for tier, values in by_tier.items()}
    # A single, pre-registered random half split checks that the estimate is not a few songs.
    order = np.random.default_rng(plan['seed']).permutation(len(deltas))
    half = len(deltas)//2
    half_difference = abs(float(np.median(deltas[order[:half]]))-float(np.median(deltas[order[half:]]))) if half else None
    reasons = []
    if len(songs) < plan['minimum_independent_training_audio']:
        reasons.append('insufficient_independent_training_audio')
    if abs_p90 > plan['maximum_source_abs_deviation_p90_ms']:
        reasons.append('heterogeneous_source_clock')
    if half_difference is None or half_difference > plan['maximum_half_difference_ms']:
        reasons.append('half_split_disagreement')
    for tier, stats in tier_stats.items():
        if stats['independent_audio'] < plan['minimum_audio_per_tier']:
            reasons.append(f'{tier}_insufficient_support')
        elif abs(stats['median_delta_ms']-median) > plan['maximum_tier_difference_ms']:
            reasons.append(f'{tier}_source_disagreement')
    return dict(status='candidate_supported' if not reasons else 'insufficient_evidence',
        raw_estimator_source_constant_ms=int(np.round(median)), absolute_deviation_p90_ms=abs_p90,
        half_split_difference_ms=half_difference, by_tier=tier_stats, reasons=reasons,
        qualified_songs=songs, excluded=excluded, applied=False,
        interpretation='Song-equal estimator reference, not proof of a physical delay; no note shifts applied.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for name, expected in plan['input_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'registered source changed: {name}')
    rows = [json.loads(line) for line in (ROOT/plan['manifest']).read_text().splitlines()]
    if len(rows) != plan['expected_charts'] or len({r['audio_sha256'] for r in rows}) != plan['expected_audio']:
        raise ValueError('source count changed')
    receipt = json.loads((ROOT/plan['receipt']).read_text())
    expected_raw = {r['path']: r['sha256'] for r in receipt['files']}
    cache = ROOT/plan['cache']
    cache.mkdir(parents=True, exist_ok=True)
    report = dict(schema='beatfcn.phigros_decoded_clock.v1', complete=False,
        plan_sha256=sha(args.plan), source_sha256=sha(Path(__file__)), records=[], audio={},
        training_updates=0, new_paid_compute_usd=0, training_ready=False,
        model_quality_promotion=False, labels_changed=False, limitations=plan['limitations'])
    result = ROOT/plan['result']
    try:
        report['control'] = codec_control(plan, cache/'control')
    except subprocess.CalledProcessError as exc:
        report['control'] = dict(passed=False, error=exc.stderr.decode(errors='replace')[-4000:])
        save(result, report)
        raise
    save(result, report)
    if not report['control']['passed']:
        raise ValueError('actual audio codec/feature known-shift control failed; corpus audit not started')
    by_audio = defaultdict(list)
    for row in rows:
        by_audio[row['audio_sha256']].append(row)
    for index, (audio_sha, group) in enumerate(sorted(by_audio.items())):
        if time.monotonic()-started > plan['audit_deadline_s']:
            raise TimeoutError('registered wall time exceeded; completed audio records retained')
        first = group[0]
        raw = ROOT/plan['raw_root']/first['audio_path']
        try:
            if sha(raw) != audio_sha or expected_raw[first['audio_path']] != audio_sha:
                raise ValueError('audio identity mismatch')
            wave = decode(raw, plan['decode_timeout_s'])
            samples = len(wave)
            waveform_health = health(wave)
            mel = mel10(wave)
            del wave
            mel_health = health(mel)
            if mel.shape != (samples//CELL_SAMPLES*CELL_SAMPLES//HOP, 128):
                raise ValueError('mel clock shape mismatch')
            env = flux_envelope(mel)
            mel_path = cache/'mel10'/f'{audio_sha}.npy'
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mel_path, mel, allow_pickle=False)
            del mel
            info = dict(status='ok', audio_sha256=audio_sha, raw_path=first['audio_path'],
                waveform=waveform_health, mel=mel_health, decoded_samples=samples,
                decoded_duration_sec=samples/SR, container_duration_sec=first['audio_sec'],
                duration_difference_ms=(samples/SR-first['audio_sec'])*1000,
                mel_trimmed_samples=samples % CELL_SAMPLES,
                mel_path=str(mel_path.relative_to(ROOT)), mel_sha256=sha(mel_path))
            chart_records = []
            for row in group:
                path = ROOT/plan['canon_root']/(row['native_id']+'.json')
                if sha(path) != plan['canonical_sha256'][row['native_id']]:
                    raise ValueError(f'canonical identity changed: {row["chart_id"]}')
                canon = json.loads(path.read_text())
                chart_records.append(chart_clock(row, canon, env, samples, plan))
            report['audio'][audio_sha] = info
            report['records'].extend(chart_records)
            del env
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
            report['audio'][audio_sha] = dict(status='error', error=f'{type(exc).__name__}: {exc}')
            done = {r['chart_id'] for r in report['records']}
            for row in group:
                if row['chart_id'] not in done:
                    report['records'].append(dict(chart_id=row['chart_id'], audio_sha256=audio_sha,
                        split=row['split'], tier=row['difficulty_tier'], status='error',
                        calibration_eligible=False, error=f'{type(exc).__name__}: {exc}'))
        if (index+1) % 8 == 0:
            report.update(processed_audio=index+1, wall_s=time.monotonic()-started)
            save(result, report)
            print(json.dumps(dict(audio=index+1, charts=len(report['records']),
                errors=sum(a['status'] != 'ok' for a in report['audio'].values()),
                wall_s=round(report['wall_s'], 2))), flush=True)
    report['calibration'] = calibration(report['records'], plan)
    good = [r for r in report['records'] if r['status'] == 'ok']
    report['summary'] = dict(audio=len(report['audio']), charts=len(report['records']),
        decoded_ok=sum(a['status'] == 'ok' for a in report['audio'].values()),
        charts_ok=len(good), decoded_bounds_charts=sum(bool(r['decoded_bounds_violations']) for r in good),
        calibration_eligible_charts=sum(r['calibration_eligible'] for r in good),
        eligible_training_audio=len(report['calibration'].get('qualified_songs', [])),
        exclusions=dict(Counter(x for r in good for x in r['calibration_exclusions'])),
        calibration_status=report['calibration']['status'])
    report.update(complete=True, wall_s=time.monotonic()-started)
    save(result, report)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
    if len(good) != plan['expected_charts'] or report['summary']['decoded_ok'] != plan['expected_audio']:
        raise SystemExit('source audit completed with errors; no admission change')


if __name__ == '__main__':
    main()
