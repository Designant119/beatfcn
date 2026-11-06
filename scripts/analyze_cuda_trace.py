"""Bounded-memory Chrome trace analysis; durations include profiler overhead."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


class Events:
    def __init__(self, path, allow_partial=False):
        self.path, self.allow_partial, self.complete = path, allow_partial, False

    def __iter__(self):
        decoder = json.JSONDecoder()
        with open(self.path) as source:
            buf = source.read(8 << 20)
            marker = '"traceEvents"'
            if marker not in buf:
                raise ValueError('traceEvents missing from header')
            position = buf.index('[', buf.index(marker)) + 1
            while True:
                while position < len(buf) and buf[position] in ' \r\n\t,':
                    position += 1
                if position < len(buf) and buf[position] == ']':
                    self.complete = True
                    return
                try:
                    event, end = decoder.raw_decode(buf, position)
                except json.JSONDecodeError:
                    more = source.read(8 << 20)
                    if not more:
                        if self.allow_partial:
                            return
                        raise ValueError('incomplete trace; no final timeline claims permitted')
                    buf = buf[position:] + more
                    position = 0
                    if len(buf) > 16 << 20:
                        raise ValueError('trace event exceeds size bound')
                    continue
                if not isinstance(event, dict):
                    raise ValueError('trace event is not an object')
                yield event
                position = end


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def analyze(path, allow_partial=False):
    reader = Events(path, allow_partial)
    categories, totals = Counter(), Counter()
    gpu, apis, scopes, long_ops = [], [], [], []
    names = defaultdict(lambda: [0, 0., 0.])
    for event in reader:
        category = event.get('cat', '')
        categories[category] += 1
        if event.get('ph') != 'X' or 'dur' not in event:
            continue
        start, duration = event['ts'], event['dur']
        end, name = start + duration, event.get('name', '')
        totals[category] += duration / 1e6
        record = {'start': start, 'end': end, 'duration_ms': duration / 1000,
                  'name': name, 'tid': event.get('tid'), 'cat': category,
                  'correlation': event.get('args', {}).get('correlation')}
        if category in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            gpu.append(record)
        elif category in ('cuda_runtime', 'cuda_driver'):
            apis.append(record)
        elif category == 'user_annotation':
            scopes.append(record)
        elif category == 'cpu_op' and duration >= 20_000:
            long_ops.append(record)
        if category in ('kernel', 'cuda_runtime', 'cuda_driver'):
            entry = names[(category, name)]
            entry[0] += 1
            entry[1] += duration / 1e6
            entry[2] = max(entry[2], duration / 1000)
    result = {'source': str(path), 'complete': reader.complete, 'categories': dict(categories),
              'summed_event_seconds_not_wall_time': dict(totals),
              'warning': 'Profiled window only; nested and overlapping durations cannot be summed as wall time.',
              'top_kernels': [], 'top_cuda_apis': []}
    for (category, name), (count, seconds, maximum) in sorted(names.items(), key=lambda kv: -kv[1][1]):
        target = result['top_kernels' if category == 'kernel' else 'top_cuda_apis']
        if len(target) < 20:
            target.append({'name': name, 'count': count, 'summed_seconds': seconds, 'max_ms': maximum})
    if not gpu or not reader.complete:
        result['timeline'] = None
        return result
    gpu.sort(key=lambda e: e['start'])
    occupied = union((e['start'], e['end']) for e in gpu)
    start, end = occupied[0][0], occupied[-1][1]
    busy = sum(b - a for a, b in occupied)
    kernel_busy = sum(b - a for a, b in union((e['start'], e['end']) for e in gpu if e['cat'] == 'kernel'))
    gaps = sorted(((a[1], b[0]) for a, b in zip(occupied, occupied[1:])), key=lambda p: p[0] - p[1])

    def display(e):
        return {k: v for k, v in e.items() if k not in ('start', 'end')} | {
            'start_s': (e['start'] - start) / 1e6, 'end_s': (e['end'] - start) / 1e6}

    gap_records = []
    for a, b in gaps[:15]:
        overlap = lambda e: e['start'] < b and e['end'] > a
        covering = [e for e in long_ops if overlap(e)]
        # The narrowest covering CPU ops are more useful than parent scopes.
        covering.sort(key=lambda e: e['duration_ms'])
        gap_records.append({'start_s': (a - start) / 1e6, 'duration_ms': (b - a) / 1000,
                            'scopes': [display(e) for e in scopes if overlap(e)],
                            'cpu_ops': [display(e) for e in covering[:8]],
                            'cuda_apis': [display(e) for e in apis if overlap(e) and e['duration_ms'] >= 1][:8]})
    result['timeline'] = {'device_span_s': (end - start) / 1e6,
                          'device_busy_union_s': busy / 1e6, 'kernel_busy_union_s': kernel_busy / 1e6,
                          'device_busy_fraction': busy / (end - start),
                          'gaps_over_20ms': sum(b - a >= 20_000 for a, b in gaps),
                          'gaps_over_20ms_total_s': sum(b - a for a, b in gaps if b - a >= 20_000) / 1e6,
                          'largest_gaps': gap_records,
                          'scopes': [display(e) for e in scopes],
                          'longest_cuda_apis': [display(e) for e in sorted(apis, key=lambda e: -e['duration_ms'])[:20]]}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trace')
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    summary = analyze(args.trace, args.allow_partial)
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('timeline', 'top_kernels', 'top_cuda_apis')}))
    if summary['timeline']:
        print(json.dumps({k: v for k, v in summary['timeline'].items() if k not in ('largest_gaps', 'scopes', 'longest_cuda_apis')}))
