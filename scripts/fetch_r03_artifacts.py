"""Bounded reads of existing R03 artifacts; never launches cloud computation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = 'evolution/capability-evolution-20261024/window-conditions-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'preflight', 'train', 'evaluate'], required=True)
    args = parser.parse_args()
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    output = ROOT / 'reports/r03_window_raw'; output.mkdir(exist_ok=True)
    lock = threading.Lock(); transferred = [0]
    def fetch(pair):
        remote, local = pair
        data = bytearray()
        for chunk in volume.read_file(remote):
            with lock:
                transferred[0] += len(chunk)
                if transferred[0] > 32 * 2**20:
                    raise ValueError('artifact transfer bound exceeded')
            data.extend(chunk)
        local.parent.mkdir(parents=True, exist_ok=True)
        temp = local.with_suffix(local.suffix + '.tmp'); temp.write_bytes(data); temp.replace(local)
        return dict(remote=remote, local=str(local.relative_to(ROOT)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    names = dict(prepare=['tape/manifest.json', 'panel.json', 'probe-batch.pt'],
                 preflight=['baseline.json', 'preflight.json'], train=['training.json'],
                 evaluate=['evaluation.json', 'evaluation_observer.json', 'baseline_observed.json', 'r03-window-density.benchmark.json',
                           'r03-window-all.benchmark.json'])[args.phase]
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(fetch, [(f'{REMOTE}/{name}', output / name.replace('tape/', '')) for name in names]))
        if args.phase == 'prepare':
            manifest = json.loads((output / 'manifest.json').read_text())
            for name, key in [('panel.json', 'panel_sha256'), ('probe-batch.pt', 'probe_batch_sha256')]:
                if hashlib.sha256((output / name).read_bytes()).hexdigest() != manifest[key]:
                    raise ValueError('saved artifact fingerprint differs')
            rows = {r['chart_id']: r for r in manifest['probe_rows']}
            records += list(executor.map(fetch, [(f'canon/{r["source"]}/{r["native_id"]}.json',
                output / 'canon' / r['source'] / f'{r["native_id"]}.json') for r in rows.values()]))
    report = dict(phase=args.phase, transferred_bytes=transferred[0], remote_compute_started=False, files=records)
    (output / f'{args.phase}-download.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(phase=args.phase, files=len(records), transferred_bytes=transferred[0], remote_compute_started=False)))


if __name__ == '__main__':
    main()
