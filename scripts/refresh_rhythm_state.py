"""Bounded read-only refresh; failed requests never replace saved evidence."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def billing_start_date(ledger):
    """Keep prior-day campaign charges visible after UTC midnight."""
    cutoff = datetime.fromisoformat(ledger['billing_through_exclusive_utc'])
    if cutoff.tzinfo is None:
        raise ValueError('historical billing cutoff must include a timezone')
    starts = [cutoff.timestamp()]
    for stage in ledger['paid_stages_dispatched']:
        reserved = stage.get('reserved_at')
        if reserved is None:
            continue
        if isinstance(reserved, bool) or not isinstance(reserved, (int, float)) or not math.isfinite(reserved) or reserved <= 0:
            raise ValueError('invalid stage reservation timestamp')
        starts.append(reserved)
    return datetime.fromtimestamp(min(starts), timezone.utc).date().isoformat()


def refresh():
    ledger = json.loads((ROOT/'reports/rhythm_v2_budget.json').read_text())
    start_date = billing_start_date(ledger)
    commands = {
        'apps': ['app', 'list', '--json'],
        'rates': ['billing', 'rates', '--json'],
        'billing': ['billing', 'report', '--start', start_date, '--resolution', 'h', '--tz', 'UTC', '--json'],
    }
    def read(item):
        name, command = item
        try:
            proc = subprocess.run([str(ROOT / '.venv/bin/modal'), *command], cwd=ROOT,
                                  capture_output=True, text=True, timeout=35)
            if proc.returncode:
                return name, {'ok': False, 'reason': 'Modal CLI returned an error', 'exit_code': proc.returncode}
            value = json.loads(proc.stdout)
            if not isinstance(value, dict if name == 'rates' else list):
                raise ValueError('unexpected response shape')
            path = ROOT / f'reports/rhythm_v2_{name}.json'
            temp = path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(value, indent=2) + '\n')
            temp.replace(path)
            return name, {'ok': True, 'path': str(path.relative_to(ROOT)), 'entries': len(value)}
        except subprocess.TimeoutExpired:
            return name, {'ok': False, 'reason': 'read-only request timed out after 35 seconds'}
        except (ValueError, OSError) as exc:
            return name, {'ok': False, 'reason': f'{type(exc).__name__}: {exc}'}
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = dict(executor.map(read, commands.items()))
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'checked_at_unix': time.time(),
              'billing_start_utc': start_date, 'billing_end': 'now; complete hourly intervals only',
              'fresh_check_succeeded': all(r['ok'] for r in results.values()), 'results': results,
              'new_paid_dispatches': 0, 'next_paid_dispatch_requires_fresh_state_and_rates': True}
    path = ROOT / 'reports/rhythm_v2_connection_status.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    refresh()
