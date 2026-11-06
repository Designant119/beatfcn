"""Check the post-isFake tape, labels and real reference support without training."""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.events.canonical import Event, LaneLoc, LineLoc, TYPES
from beatfcn.events.controls import MODE_FIELDS
from beatfcn.train.event_conditions import encode_controls, training_labels
from beatfcn.train.event_reference import VisibleReference, encode_reference
from beatfcn.train.local_request_panel import LocalRequestPanel

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/d92-panel-audit-v1.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    h = hashlib.sha256()
    with (ROOT / path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def events_for(chart):
    with np.load(ROOT / chart['target_path'], allow_pickle=False) as a:
        return tuple(Event(f"{chart['chart_id']}:{i}", int(h), int(t), TYPES[int(k)],
            LaneLoc(int(l)) if chart['mode'] == 'mania_4k' else LineLoc('rhythm', 'above', 0.))
            for i, (h, t, k, l) in enumerate(zip(*(a[key] for key in ('onset_us', 'end_us', 'kind', 'lane')))))


def signature(events):
    values = sorted((e.onset_us, e.end_us, e.type,
        e.location.lane if isinstance(e.location, LaneLoc) else -1) for e in events)
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def stats(value):
    a = np.asarray(value)
    f = a.astype(np.float64)
    return dict(shape=list(a.shape), dtype=str(a.dtype), nan=int(np.isnan(f).sum()),
        inf=int(np.isinf(f).sum()), minimum=float(f.min()), maximum=float(f.max()),
        mean=float(f.mean()), std=float(f.std()))


def pair_summary(cases, admitted):
    groups = defaultdict(list)
    for case in cases:
        if case['chart_id'] in admitted:
            groups[(case['split'], case['mode'], case['audio_sha256'],
                    *case['scope_us'], case['request_identity'])].append(case)
    counts, songs = defaultdict(Counter), defaultdict(lambda: defaultdict(set))
    examples = []
    for group in groups.values():
        for a, b in combinations(group, 2):
            key = a['split'] + '/' + a['mode']
            labels = ['same_audio_scope_and_carry']
            if a['prefix_sha256'] != b['prefix_sha256'] and a['target_sha256'] != b['target_sha256']:
                labels.append('distinct_prefix_and_target')
                if a['stratum'] == b['stratum']:
                    labels.append('also_same_fine_stratum')
                    fields = MODE_FIELDS[a['mode']]
                    if (all(f in a['bands'] and f in b['bands'] for f in fields)
                            and a['bands'] == b['bands']):
                        labels.append('also_all_applicable_bands_equal')
                        if len(examples) < 20:
                            examples.append(dict(chart_ids=[a['chart_id'], b['chart_id']],
                                component=a['component'], split=a['split'], mode=a['mode'],
                                scope_us=a['scope_us'], stratum=a['stratum'], bands=a['bands']))
            for label in labels:
                counts[key][label] += 1
                songs[key][label].add(a['component'])
    return dict(by_split_mode={k: {name: dict(window_pairs=n, independent_songs=len(songs[k][name]))
                for name, n in counts[k].items()} for k in sorted(counts)}, examples=examples)


def run():
    plan = read(PLAN)
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen audit source changed: ' + path)
    report = dict(id='D92-panel-audit-v1', complete=False, plan_sha256=sha(PLAN),
        source_sha256=plan['source_sha256'], optimizer_updates=0, model_forwards=0,
        new_cloud_compute_calls=0, training_ready=False, cases=[], tensor_cases=[])
    try:
        tape, local, old = (read(plan[k]) for k in ('all_tape', 'local_tape', 'old_tape'))
        charts = {c['chart_id']: c for c in tape['charts']}
        ready_ids = {c['chart_id'] for c in local['charts']}
        old_ids = {c['chart_id'] for c in old['charts']}
        assert all(charts[c['chart_id']] == c for c in old['charts'])
        assert all(charts[c['chart_id']] == c for c in local['charts'])
        eval_ids = {c['chart_id'] for c in read(plan['evaluation20'])['records']}
        assert len(eval_ids) == 20 and eval_ids <= ready_ids
        assert all(charts[c]['split'] == 'val' for c in eval_ids)
        split_keys = defaultdict(set)
        for c in charts.values():
            split_keys['component:' + c['component']].add(c['split'])
            split_keys['audio:' + c['audio_sha256']].add(c['split'])
        assert all(len(v) == 1 for v in split_keys.values())
        report['integrity'] = dict(old_records_exact=len(old_ids), eval20_retained=len(eval_ids),
            split_leakage=0, chart_count=len(charts), local_chart_count=len(ready_ids))
        coverage, ranges, skipped = defaultdict(Counter), defaultdict(list), Counter()
        total_events = total_tokens = 0
        for c in charts.values():
            if time.monotonic() - started > plan['max_wall_s']:
                raise TimeoutError('Read-only audit time limit reached')
            for field in ('target', 'token'):
                assert sha(c[field + '_path']) == c[field + '_sha256']
            events = events_for(c)
            heads = np.asarray([e.onset_us for e in events], dtype=np.int64)
            tokens = np.load(ROOT / c['token_path'], mmap_mode='r', allow_pickle=False)
            next_head = next_token = previous_end = 0
            for q in c['requests']:
                start, end = q['request']['start_us'], q['request']['end_us']
                lo, hi = np.searchsorted(heads, [start, end])
                assert start == previous_end and q['target_start'] == next_head == lo
                assert q['target_stop'] == hi and hi - lo == q['event_count']
                assert q['token_start'] == next_token and q['token_stop'] - next_token == q['tokens']
                next_head, next_token, previous_end = int(hi), q['token_stop'], end
                bands, raw = training_labels(c['mode'], events, (start, end))
                assert q['bands'] == (bands if c['split'] == 'train' else None)
                expected = encode_controls(c['mode'], bands if c['split'] == 'train' else {},
                                           (start, end), origin_us=start)
                assert np.array_equal(np.asarray(q['controls'], np.float32), expected)
                if c['split'] == 'train':
                    for field in MODE_FIELDS[c['mode']]:
                        for panel in ('all_source', 'local_ready') if c['chart_id'] in ready_ids else ('all_source',):
                            key = panel + '/' + c['mode'] + '/' + field
                            label = 'labelled' if field in bands else 'undefined' if raw[field] is None else 'out_of_table'
                            coverage[key][label] += 1
                            if raw[field] is not None:
                                ranges[key].append(raw[field])
                if start < 16_000_000:
                    skipped['less_than_16s_prefix'] += 1
                    continue
                lower = start - 16_000_000
                if any(e.is_hold and e.onset_us < lower and e.end_us > start for e in events):
                    skipped['unseen_incoming_hold_head'] += 1
                    continue
                visible = tuple(e for e in events if lower <= e.onset_us < start)
                if len(visible) + 2 > 4096:
                    skipped['reference_capacity_exceeded_not_truncated'] += 1
                    continue
                report['cases'].append(dict(chart_id=c['chart_id'], component=c['component'],
                    split=c['split'], mode=c['mode'], stratum=c['stratum'], audio_sha256=c['audio_sha256'],
                    scope_us=[start, end], prefix_sha256=signature(visible), target_sha256=signature(events[lo:hi]),
                    visible_notes=len(visible), bands=bands,
                    request_identity=json.dumps(q['request'], sort_keys=True)))
            assert next_head == len(events) and next_token == len(tokens) and previous_end == c['duration_us']
            total_events += len(events)
            total_tokens += len(tokens)
        report['integrity'].update(events=total_events, tokens=total_tokens,
                                  all_heads_partitioned=True, scoped_labels_exact=True)
        report['condition_coverage'] = {k: dict(v, finite_min=min(ranges[k]) if ranges[k] else None,
            finite_max=max(ranges[k]) if ranges[k] else None) for k, v in sorted(coverage.items())}
        report['reference_skipped'] = dict(skipped)
        report['reference_support'] = {name: pair_summary(report['cases'], ids) for name, ids in
            (('old208', old_ids), ('local213', ready_ids), ('all584', set(charts)))}

        # Only the five new locally feature-ready charts need new input assembly.
        panel = LocalRequestPanel(ROOT, plan['local_tape'], plan['features'])
        rehashed = {}
        for cid in sorted(ready_ids - old_ids):
            c = charts[cid]
            record = panel.features.records[c['audio_sha256']]
            for p, h in record['sha256'].items():
                if p not in rehashed:
                    assert sha(p) == h
                    rehashed[p] = h
            indices = [i for i, r in enumerate(panel.records) if r['chart_id'] == cid]
            active = [i for i in indices if panel.records[i]['event_count'] and
                      panel.records[i]['request']['start_us'] >= 16_000_000]
            selected = sorted({active[len(active) // 2], indices[-1]})
            events = events_for(c)
            for i in selected:
                row = panel.row(i)
                seq = row['sequence']
                assert seq['valid'].all() and seq['allowed'][np.arange(len(seq['targets'])), seq['targets']].all()
                inp = row['inputs']; q = panel.records[i]['request']
                lower, start = q['start_us'] - 16_000_000, q['start_us']
                visible = tuple(e for e in events if lower <= e.onset_us < start)
                source = VisibleReference(cid, c['mode'], c['audio_sha256'], 'human_input', ((lower, start),), visible)
                ref, provenance = encode_reference(mode=c['mode'], audio_sha256=c['audio_sha256'],
                    duration_us=c['duration_us'], origin_us=start,
                    unknown_regions=((start, q['end_us']),), sources=(source,))
                assert all(p['event_id'] is None or lower <= p['interval_us'][0] < start for p in provenance)
                summary = {k: stats(v) for k, v in {**inp, **ref}.items()}
                assert all(not s['nan'] and not s['inf'] for s in summary.values())
                report['tensor_cases'].append(dict(chart_id=cid, index=i, scope_us=[start, q['end_us']],
                    input_stats=summary, sequence_stats={k: stats(v) for k, v in seq.items()},
                    targets_legal=True, hidden_heads_in_reference=0, valid_tokens=len(seq['targets']),
                    partial_tail=q['end_us'] - start < 8_000_000,
                    known_cross_boundary_holds=sum(e.is_hold and e.end_us > start for e in visible)))
        report.update(complete=True, feature_files_rehashed=rehashed,
            limitations=['Data/tensor evidence only; no fitting or free-generation quality claim.',
                'Validation-derived bands exist only in this descriptive audit, never in evaluation inputs.',
                'Equal coarse bands and difficulty do not hold exact load constant or establish causal style learning.',
                'Reference support requires distinct real prefixes and continuations with identical audio/scope/carry.'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_s'] = time.monotonic() - started
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('complete', 'wall_s', 'integrity', 'reference_support')}))


if __name__ == '__main__':
    run()
