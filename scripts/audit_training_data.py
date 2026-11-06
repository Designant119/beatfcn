"""Local, resumable audit. Downloads existing Volume files; never starts cloud compute.

PYTHONPATH=. .venv/bin/python scripts/audit_training_data.py --root inputs/training_audit_20261023 --migrate --fetch-sample --audit
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time
from beatfcn.train.audit import refresh_offsets, eligible, sample_rows, audit_cores


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def fetch(root, paths):
    import modal
    volume = modal.Volume.from_name('beatfcn-data')
    def one(remote):
        dest = root / remote
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + '.tmp')
        for attempt in range(3):
            try:
                with temp.open('wb') as f:
                    for chunk in volume.read_file(remote):
                        f.write(chunk)
                temp.replace(dest)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    paths = sorted(set(paths))
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(one, p): p for p in paths}
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 100 == 0:
                print('downloaded', i, '/', len(paths), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reports', type=Path, default=Path('reports'))
    p.add_argument('--fetch-diagnostics', action='store_true')
    p.add_argument('--migrate', action='store_true')
    p.add_argument('--fetch-sample', action='store_true')
    p.add_argument('--audit', action='store_true')
    p.add_argument('--per-mode', type=int, default=128)
    p.add_argument('--requests', type=int, default=4)
    args = p.parse_args()
    path = args.root/'manifest/charts.jsonl'
    original = path.with_name('charts.original.jsonl')
    if not original.exists():
        shutil.copyfile(path, original)
    rows = [json.loads(l) for l in original.read_text().splitlines()]
    if args.fetch_diagnostics:
        fetch(args.root, [f"work/rows/osu/{int(r['shard']):06d}.jsonl" if r['source']=='osu'
                         else f"work/rows/phira/{r['native_id']}.json" for r in rows if r['status']=='parsed'])
    if args.migrate:
        diagnostics = {}
        for source, glob in [('osu', '*.jsonl'), ('phira', '*.json')]:
            for file in (args.root/'work/rows'/source).glob(glob):
                for line in file.read_text().splitlines():
                    row = json.loads(line)
                    if 'offset' in row:
                        diagnostics[row['chart_id']] = row['offset']
        rows, report = refresh_offsets(rows, lambda r: diagnostics[r['chart_id']])
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        write_json(args.reports/'offset_migration_2026-10-23.json', report)
        print({k:v for k,v in report.items() if k!='changes'}, flush=True)
    else:
        rows = [json.loads(l) for l in path.read_text().splitlines()]
    have = set(json.loads((args.root/'taps/v4_i8_L9-16-24/_index.json').read_text()))
    candidates = [r for r in rows if r.get('split')=='train' and eligible(r) and r['audio_sha256'] in have]
    selected = sample_rows(candidates, per_mode=args.per_mode)
    write_json(args.root/'audit_selection.json', selected)
    if args.fetch_sample:
        fetch(args.root, [f"canon/{r['source']}/{r['native_id']}.json" for r in selected]
                         + [f"mel10/{r['audio_sha256']}.npy" for r in selected])
    if args.audit:
        report, traces = audit_cores(args.root, selected, requests_per_chart=args.requests)
        report.update(manifest_charts=len(rows), train_candidates=len(candidates),
                      candidate_songs=len({r['audio_sha256'] for r in candidates}))
        write_json(args.reports/'training_data_audit_2026-10-23.json', report)
        (args.root/'core_audit_traces.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in traces))
        print({k:{kk:vv for kk,vv in report[k].items() if kk!='stages'} for k in ('dynamic','static')})

if __name__ == '__main__':
    main()
