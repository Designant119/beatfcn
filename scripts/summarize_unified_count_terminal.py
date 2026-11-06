"""Recompute C06 terminal mechanisms from saved events/probabilities, zero updates."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.preflight_c06_candidates import rank_stats
from scripts.review_frontend_canvas_comparison import metrics
from scripts.summarize_evolution_terminals import behavior, difference


ROOT = Path(__file__).resolve().parents[1]
ART = 'artifacts/c06-reset-unified-count-law-v1'
KINDS = ('tap', 'hold', 'flick', 'drag')


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def macro(rows):
    return {field: float(np.mean([r[field] for r in rows]))
            for field in ('full_note_f1', 'onset10', 'onset50')}


def score_view(score):
    return dict(full_note_f1=score['full_note_f1'], onset10=score['onset_10ms']['f1'],
                onset50=score['onset_50ms']['f1'])


def main():
    started = time.monotonic()
    plan_path = 'bench/rhythm-v2/c06-reset-unified-count-law-v1.json'
    report_path = 'reports/local-evolution-v2/unified-count.json'
    plan, report = read(plan_path), read(report_path)
    baseline = read(plan['baseline_report'])
    assert report['quality_updates'] == 1000 and report['health_updates'] == 100
    assert not report['failures'] and not report['health_weights_used']
    assert sha(plan_path) == report['plan_sha256']
    assert sha(f'{ART}/quality-1000.pt') == report['final_weight_sha256']
    for path, expected in plan['sources'].items():
        assert sha(path) == expected, path
    primary = read(f'{ART}/unified/quality-1000-draft.json')
    recomputed = metrics(primary)
    for field in ('full_note_f1', 'onset10', 'onset50', 'density_ratio', 'attempted', 'complete'):
        assert abs(recomputed[field] - report['evaluations']['draft'][field]) < 1e-12, field

    counts = read(f'{ART}/count-audio-diagnostics.json')
    events = read(f'{ART}/diagnostic-events.json')
    assert [r['index'] for r in counts] == [r['index'] for r in events]
    logits = torch.load(ROOT / ART / 'count-logits.pt', map_location='cpu', weights_only=False)
    proper, ranks, expected_counts = {}, {}, {}
    for name in ('old', 'new'):
        proper[name], ranks[name], expected_counts[name] = {}, {}, {}
        for k, kind in enumerate(KINDS):
            proper[name][kind] = {}
            for stratum in ('all', 'positive', 'zero'):
                values = [r['count'][name]['proper_scores'][kind][stratum] for r in counts]
                n = sum(v['n'] for v in values)
                proper[name][kind][stratum] = dict(n=n, **{
                    metric: sum(v['n'] * v[metric] for v in values if v['n']) / n if n else None
                    for metric in ('nll', 'brier')})
            probability, positive = [], []
            expectation = 0.
            true_events = 0
            count_histogram = Counter()
            for row in events:
                truth = np.zeros(200, dtype=np.int64)
                start = row['request']['start_us']
                for event in row['target']:
                    if event['type'] == kind:
                        cell = (event['onset_us'] - start) // 40000
                        assert 0 <= cell < 200
                        truth[cell] += 1
                count_histogram.update(map(int, truth))
                x = logits[row['index']][name]['count'][0, :, k].numpy().astype(np.float64)
                p = np.exp(x - x.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                probability.extend(1 - p[:, 0])
                positive.extend(truth > 0)
                expectation += float((p * np.arange(17)).sum())
                true_events += int(truth.sum())
            ranks[name][kind] = rank_stats(probability, positive)
            expected_counts[name][kind] = dict(expected=expectation, true=true_events,
                                                truth_cell_count_histogram=dict(sorted(count_histogram.items())))
    swaps = {}
    for condition in events[0]['conditions']:
        views = [score_view(r['conditions'][condition]['scores']) for r in events]
        swaps[condition] = dict(component_macro=macro(views),
            predicted_events=sum(len(r['conditions'][condition]['prediction']['events']) for r in events),
            per_component=[dict(index=r['index'], **v) for r, v in zip(events, views)])
    audio = {}
    for name in ('old', 'new'):
        real = [score_view(r['audio'][name]['real']) for r in counts]
        shifted = [score_view(r['audio'][name]['shifted']) for r in counts]
        audio[name] = dict(real=macro(real), shifted=macro(shifted),
            real_minus_shifted={k: macro(real)[k] - macro(shifted)[k] for k in macro(real)},
            positive_components=sum(a['onset10'] > b['onset10'] for a, b in zip(real, shifted)),
            equal_denominator_components=sum(r['audio'][name]['same_member_denominator'] for r in counts),
            excluded_edges_ms=400,
            per_component=[dict(index=r['index'], real=a, shifted=b,
                same_member_denominator=r['audio'][name]['same_member_denominator'])
                for r, a, b in zip(counts, real, shifted)])
    health = {phase: dict(loss=float(np.mean([r['loss'] for r in report[phase + '_health']])),
        parts={k: float(np.mean([r['parts'][k] for r in report[phase + '_health']]))
               for k in report[phase + '_health'][0]['parts']}) for phase in ('initial', 'final')}
    tasks = {task: {k: v for k, v in result.items() if k != 'components'}
             for task, result in report['evaluations'].items()}
    result = dict(id='c06-unified-count-terminal-analysis-v1', optimizer_updates=0, inference_calls=0,
        scope='Saved outputs only; 32-window primary and 16-component mechanism diagnostics kept separate.',
        identities={path: sha(path) for path in (plan_path, report_path, plan['baseline_report'],
            f'{ART}/quality-1000.pt', f'{ART}/count-logits.pt', f'{ART}/diagnostic-events.json',
            f'{ART}/count-audio-diagnostics.json')},
        health=health, tasks=tasks,
        primary_change=difference(report['evaluations']['draft'], baseline['evaluations']['draft']),
        behavior=behavior([primary]), count_multiset=report['mechanism_diagnostics']['count_multiset'],
        proper_scores_cell_weighted=proper, occupancy_rank_stats=ranks,
        expected_type_counts_on_16_windows=expected_counts, count_swaps=swaps, fixed_count_audio=audio,
        reference_full_gain=tasks['completion']['full_note_f1'] - tasks['reference_masked']['full_note_f1'],
        wall_s={k: report[k] for k in ('health_wall_s', 'quality_wall_s', 'evaluation_wall_s', 'total_process_wall_s')},
        limitations=['One seed and reused development components; no capacity or emotion test.',
            'Count swaps freeze all coarse cell/type/multiplicity decisions, not just total N.',
            '400ms encoded-value roll preserves global song inputs; it is not a waveform intervention.',
            'AUROC/AP are occupancy diagnostics; positives are not independent songs.'],
        analysis_wall_s=time.monotonic() - started)
    path = ROOT / 'reports/local-evolution-v2/terminal-analysis-v1.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(path.relative_to(ROOT)), primary_change=result['primary_change'],
        count_multiset=result['count_multiset'], count_swaps={k:v['component_macro'] for k,v in swaps.items()},
        audio={k:{a:b for a,b in v.items() if a!='per_component'} for k,v in audio.items()},
        proper_scores=proper), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
