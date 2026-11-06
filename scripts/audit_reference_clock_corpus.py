"""D15: local, prediction-independent audit of reference timing uncertainty."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.ingest.offset import estimate_offset, flux_envelope
from beatfcn.train.data import chart_from_canon, load_manifest, model_shift_us

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_estimate(heads, envelope):
    result = estimate_offset(heads, env=envelope)
    result.pop('agree_counts', None)
    return result


def tensor_health(mel):
    if mel.ndim != 2 or mel.shape[0] == 0 or mel.shape[1] != 128 or not np.isfinite(mel).all():
        raise ValueError('non-finite or invalid mel tensor')
    # Cached inputs can be fp16: a finite tensor's fp16 reduction can overflow.
    return dict(shape=list(mel.shape), dtype=str(mel.dtype), min=float(mel.min()),
        max=float(mel.max()), mean=float(mel.mean(dtype=np.float64)),
        std=float(mel.std(dtype=np.float64)), nonfinite=0)


def metadata_summary(rows, tolerance_ms):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['mode']}/{row['split']}"].append(row)
    output = {}
    for key, group in groups.items():
        usable = [r for r in group if (r.get('offset_decision') or {}).get('r_ms') is not None]
        residual = [r['offset_decision']['r_ms'] - r['offset_decision']['shift_ms'] for r in usable]
        beyond = [r for r, value in zip(usable, residual) if abs(value) > tolerance_ms]
        output[key] = dict(charts=len(group), audio=len({r['audio_sha256'] for r in group}),
            missing_residual=len(group)-len(usable), remaining_residual_beyond_20ms=len(beyond),
            audio_with_residual_beyond_20ms=len({r['audio_sha256'] for r in beyond}),
            residual_abs_quantiles_ms=np.percentile(np.abs(residual), [50, 90, 95, 99, 100]).tolist() if residual else None,
            missing_estimator_confidence=sum(r.get('offset', {}).get('conf') is None for r in usable),
            low_estimator_confidence=sum(r.get('offset', {}).get('conf') is not None and
                r['offset']['conf'] < 3 for r in usable),
            remaining_beyond_20ms_high_confidence=sum(r.get('offset', {}).get('conf', 0) >= 3 for r in beyond))
    return output


def summarize(records, min_confidence, tolerance_ms):
    output = {}
    for mode in ('mania_4k', 'phira'):
        group = [r for r in records if r['mode'] == mode]
        ok = [r for r in group if r.get('status') == 'ok']
        stable = [r for r in ok if len(r['reliable_thirds']) == 3]
        both_views = [r for r in ok if all(r['whole'][name].get('status') == 'ok' and
                      r['whole'][name]['conf'] >= min_confidence and not r['whole'][name]['edge']
                      for name in ('all', 'tap_hold'))]
        output[mode] = dict(selected_charts=len(group), completed_charts=len(ok),
            independent_audio=len({r['audio_sha256'] for r in ok}),
            errors=[dict(chart_id=r['chart_id'], error=r['error']) for r in group if r.get('status') != 'ok'],
            metadata_delta_mismatch_charts=sum(r['metadata_delta_difference_ms'] != 0 for r in ok),
            residual_beyond_20ms=sum(abs(r['unapplied_residual_ms']) > tolerance_ms for r in ok),
            all_heads_low_confidence=sum(r['whole']['all'].get('conf', 0) < min_confidence for r in ok),
            three_reliable_thirds=len(stable),
            thirds_span_beyond_20ms=sum(r['thirds_span_ms'] > tolerance_ms for r in stable),
            reliable_all_and_tap_hold=len(both_views),
            all_vs_tap_hold_difference_beyond_20ms=sum(abs(r['whole']['all']['delta_ms']-
                r['whole']['tap_hold']['delta_ms']) > tolerance_ms for r in both_views),
            maximum_reliable_thirds_span_ms=max((r['thirds_span_ms'] for r in stable), default=None),
            by_stratum=dict(Counter(r['stratum'] for r in group)))
    return output


def main():
    started = time.monotonic()
    plan_path = ROOT/'bench/rhythm-v2/d15-reference-clock-corpus-v1.json'
    plan = json.loads(plan_path.read_text())
    cache = ROOT/plan['cache']
    assert fingerprint(cache/'audit_selection.json') == plan['selection_sha256']
    assert fingerprint(cache/'manifest/charts.jsonl') == plan['manifest_sha256']
    selected = json.loads((cache/'audit_selection.json').read_text())
    assert len(selected) == plan['expected_charts']
    assert len({r['audio_sha256'] for r in selected}) == plan['expected_audio']
    assert not set(plan['excluded_evaluation_audio_sha256']) & {r['audio_sha256'] for r in selected}
    metadata = metadata_summary(load_manifest(str(cache)), plan['tolerance_ms'])
    records, health = [], {}
    for index, row in enumerate(selected):
        if time.monotonic()-started > plan['maximum_wall_s']:
            raise TimeoutError('registered local audit time limit reached')
        item = dict(chart_id=row['chart_id'], mode=row['mode'], split=row['split'],
            audio_sha256=row['audio_sha256'], stratum=row['_audit_stratum'], status='error')
        try:
            paths = dict(canon=cache/'canon'/row['source']/(row['native_id']+'.json'),
                         mel=cache/'mel10'/(row['audio_sha256']+'.npy'))
            for path in paths.values():
                if fingerprint(path) != plan['source_files_sha256'][str(path.relative_to(ROOT))]:
                    raise ValueError('registered source identity changed')
            chart = chart_from_canon(json.loads(paths['canon'].read_text()))
            mel = np.load(paths['mel'], allow_pickle=False)
            health[row['audio_sha256']] = tensor_health(mel)
            envelope = flux_envelope(mel)
            del mel
            heads = {
                'all': np.unique([e.onset_us/1e6 for e in chart.events]),
                'tap_hold': np.unique([e.onset_us/1e6 for e in chart.events if e.type in ('tap', 'hold')]),
                'drag': np.unique([e.onset_us/1e6 for e in chart.events if e.type == 'drag'])}
            whole = {name: compact_estimate(values, envelope) for name, values in heads.items()}
            thirds = []
            for third in range(plan['thirds']):
                lo = len(envelope)*third//plan['thirds']
                hi = len(envelope)*(third+1)//plan['thirds']
                thirds.append(compact_estimate(heads['all']-lo*.01, envelope[lo:hi]))
            reliable = [x for x in thirds if x.get('status') == 'ok' and
                        x['conf'] >= plan['min_confidence'] and not x['edge']]
            del envelope
            decision = row['offset_decision']
            if whole['all'].get('status') != 'ok':
                raise ValueError('whole chart estimate unavailable')
            item.update(status='ok', types=dict(Counter(e.type for e in chart.events)),
                original_offset_decision=decision, whole=whole, thirds=thirds,
                reliable_thirds=reliable,
                thirds_span_ms=max(x['delta_ms'] for x in reliable)-min(x['delta_ms'] for x in reliable) if reliable else None,
                metadata_delta_difference_ms=whole['all']['delta_ms']-row['offset']['delta_ms'],
                unapplied_residual_ms=decision['r_ms']-decision['shift_ms'],
                model_shift_ms=model_shift_us(row, True)//1000)
        except (ValueError, KeyError, OSError) as exc:
            item['error'] = f'{type(exc).__name__}: {exc}'
        records.append(item)
        if (index+1) % 32 == 0:
            print(json.dumps(dict(processed=index+1, errors=sum(r['status'] != 'ok' for r in records),
                                  wall_s=round(time.monotonic()-started, 2))), flush=True)
    report = dict(schema='beatfcn.reference_clock_corpus.v1', complete=True,
        plan_sha256=fingerprint(plan_path), source_sha256=fingerprint(Path(__file__)),
        selected_charts=len(selected), checked_audio=len(health), data_health=health,
        full_manifest_metadata=metadata, summary=summarize(records, plan['min_confidence'], plan['tolerance_ms']),
        records=records, new_paid_dispatches=0, training_updates=0, acceptance_changed=False,
        model_quality_promotion=False, limitations=plan['limitations'], wall_s=time.monotonic()-started)
    path=ROOT/'reports/d15_reference_clock_corpus.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(summary=report['summary'], full_manifest_metadata=metadata), ensure_ascii=False))


if __name__ == '__main__':
    main()
