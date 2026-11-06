#!/usr/bin/env python3
"""Live monitor for the 0.6B chart-generation training run.

Reads the Modal app log stream (`modal app logs -f`) and the Volume checkpoint
listing, then renders a small terminal dashboard every few seconds.

It never launches or stops training. It also appends every parsed record to
`reports/600m-monitor.jsonl` and rewrites `reports/600m-monitor.md`, so the
progress can be inspected without a terminal.

Usage:
    .venv/bin/python -m scripts.monitor_600m --watch
    .venv/bin/python -m scripts.monitor_600m --once
    .venv/bin/python -m scripts.monitor_600m --watch --app ap-xxxx
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import os
import json
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
JSONL = REPORTS / '600m-monitor.jsonl'
STATUS = REPORTS / '600m-monitor.md'
PYTHON = ROOT / '.venv' / 'bin' / 'python'
MODAL = ROOT / '.venv' / 'bin' / 'modal'
CHECKPOINT_DIR = 'stage0/inputs/600m-native-cf/checkpoints'
WATCH = REPORTS / '600m-learning-watch'

SPARK = '▁▂▃▄▅▆▇█'


def modal_json(*args: str):
    try:
        result = subprocess.run([str(MODAL), *args, '--json'], capture_output=True,
                                text=True, timeout=10, cwd=ROOT)
        if result.returncode:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def app_status(app):
    status = app.get('state', 'unknown')
    if 'tasks' in app:
        status += f'; tasks={app["tasks"]}'
    return status


def active_app(explicit: str | None):
    apps = modal_json('app', 'list')
    if apps is None:
        return explicit, 'discovery unavailable'
    if explicit:
        match = next((a for a in apps if a.get('app_id') == explicit), None)
        return explicit, app_status(match) if match else 'not found'
    live = [a for a in apps
            if a.get('description') == 'stage0-cuda-training'
            and a.get('state') in ('ephemeral', 'deployed')]
    if not live:
        return None, 'idle'
    newest = max(live, key=lambda a: a.get('created_at', ''))
    return newest.get('app_id'), app_status(newest)


def watch_contract():
    try:
        return json.loads((WATCH / 'contract.json').read_text())
    except (OSError, ValueError):
        return {}


def list_checkpoints():
    try:
        directory = watch_contract().get('active_run', {}).get('checkpoint_dir', CHECKPOINT_DIR)
        result = subprocess.run([str(MODAL), 'volume', 'ls', 'beatfcn-data', directory],
                                capture_output=True, text=True, timeout=10, cwd=ROOT)
        names = [line.strip().rsplit('/', 1)[-1] for line in result.stdout.splitlines() if line.strip()]
        return sorted(name for name in names if name.endswith('.pt'))
    except Exception:
        return []


def refresh_evaluation(state):
    contract = watch_contract()
    active = contract.get('active_run', {})
    remote = active.get('evaluation_remote')
    if not remote or active.get('call_id') != state.call_id:
        return
    try:
        result = subprocess.run([str(MODAL), 'volume', 'get', 'beatfcn-data', remote, '-'],
                                capture_output=True, text=True, timeout=15, cwd=ROOT)
        if result.returncode:
            return
        # Modal appends a download confirmation even when destination is stdout.
        value, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
        if value.get('plan_sha256') != active.get('plan_sha256'):
            return
        temporary = WATCH / 'evaluation_latest.json.tmp'
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(WATCH / 'evaluation_latest.json')
        state.evaluation = value
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return


def refresh_runtime(state):
    active = watch_contract().get('active_run', {})
    remote = active.get('report_remote')
    if not remote or active.get('call_id') != state.call_id:
        return
    try:
        result = subprocess.run([str(MODAL), 'volume', 'get', 'beatfcn-data', remote, '-'],
                                capture_output=True, text=True, timeout=15, cwd=ROOT)
        if result.returncode:
            return
        value, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
        if value.get('plan_sha256') != active.get('plan_sha256'):
            return
        state.runtime = value
        temporary = WATCH / 'runtime_latest.json.tmp'
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(WATCH / 'runtime_latest.json')
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return


def sparkline(values, width=40):
    values = [v for v in values if isinstance(v, (int, float))][-width:]
    if len(values) < 2:
        return ''
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return SPARK[0] * len(values)
    return ''.join(SPARK[min(7, int((v - low) / (high - low) * 7.999))] for v in values)


class Tail(threading.Thread):
    """Follow one Modal app log stream without blocking the renderer."""

    def __init__(self, app_id: str, out: queue.Queue, call_id=None):
        super().__init__(daemon=True)
        self.app_id = app_id
        self.out = out
        self.call_id = call_id
        self.proc = None
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                command = [str(MODAL), 'app', 'logs', '-f', self.app_id, '--show-function-call-id']
                if self.call_id:
                    command += ['--function-call', self.call_id]
                self.proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=ROOT,
                    env=dict(os.environ, PYTHONUNBUFFERED='1'))
            except Exception as exc:
                self.out.put(('error', f'{type(exc).__name__}: {exc}'))
                time.sleep(5)
                continue
            for line in self.proc.stdout:
                if self.stop:
                    break
                self.out.put(('line', line.rstrip('\n')))
            self.proc.wait()
            if self.stop:
                break
            self.out.put(('error', 'log stream ended; retrying'))
            time.sleep(3)

    def kill(self):
        self.stop = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def parse(line: str):
    start = line.find('{"phase"')
    if start < 0:
        return None
    try:
        record = json.loads(line[start:])
        prefix = line[:start].strip()
        if prefix.startswith('fc-'):
            record['call_id'] = prefix.split()[0]
        return record
    except ValueError:
        return None


class State:
    def __init__(self):
        self.records = []
        self.total_steps = None
        self.app_id = None
        self.app_state = ''
        self.checkpoints = []
        self.last_checkpoint_poll = 0.
        self.started_wall = None
        self.started_elapsed = None
        self.last_record_wall = None
        self.logged_error = None
        self.failed = False
        self.call_id = None
        self.phase = 'waiting'
        self.evaluation = None
        self.eval_summaries = []
        self.memory_probes = []
        self.expected_call_id = None
        self.runtime = None
        self.restored_updates = None

    def select_app(self, app_id, app_state):
        if app_id != self.app_id:
            total_steps = self.total_steps
            self.__init__()
            self.total_steps = total_steps
        self.app_id, self.app_state = app_id, app_state

    def ingest_line(self, line):
        record = parse(line)
        if record:
            call_id = record.get('call_id')
            if self.expected_call_id and call_id and call_id != self.expected_call_id:
                return None
            if call_id and call_id != self.call_id:
                total, app_id, app_state = self.total_steps, self.app_id, self.app_state
                self.__init__()
                self.total_steps, self.app_id, self.app_state = total, app_id, app_state
                self.call_id = call_id
            if self.apply(record) is False:
                return None
        else:
            error = re.search(r'\b(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)|Error|Exception): .+', line)
            if error:
                self.logged_error = error.group(0)
        return record

    def apply(self, record):
        if (record.get('phase') == 'train' and self.records
                and record.get('step', 0) <= self.records[-1].get('step', 0)):
            return False
        self.phase = record.get('phase', self.phase)
        if self.phase == 'checkpoint_restored':
            self.restored_updates = record.get('optimizer_step_min')
        if self.phase == 'learning_eval_done':
            self.eval_summaries = record.get('results', [])
        if self.phase == 'memory_probe':
            self.memory_probes.append(record)
        if record.get('phase') == 'failed':
            self.failed = True
        if record.get('phase') == 'train':
            if self.records and (record.get('elapsed') or 0) < (self.records[-1].get('elapsed') or 0):
                self.failed = False
                self.logged_error = None
                self.records = []  # a new run restarted the clock
                self.started_elapsed = None
                self.started_wall = None
            self.records.append(record)
            self.last_record_wall = time.time()
            del self.records[:-2000]
            if self.started_elapsed is None:
                self.started_elapsed = record.get('elapsed') or 0.
                self.started_wall = time.time() - self.started_elapsed
        return True

    def latest(self):
        return self.records[-1] if self.records else {}


def restore_monitor_history(state, contract):
    """Restore this call's measured chart history; old log replay cannot rewind it."""
    records = {}
    try:
        with JSONL.open() as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if value.get('call_id') == state.call_id and value.get('phase') == 'train':
                    records[value['step']] = value
        for record in sorted(records.values(), key=lambda r: r['step'])[-2000:]:
            state.apply(record)
    except OSError:
        pass
    expected = contract.get('active_run', {}).get('plan_sha256')
    for attribute, name in [('evaluation', 'evaluation_latest.json'), ('runtime', 'runtime_latest.json')]:
        try:
            value = json.loads((WATCH/name).read_text())
            if expected and value.get('plan_sha256') == expected:
                setattr(state, attribute, value)
        except (OSError, ValueError):
            continue

def dashboard_view(state):
    from scripts.monitor_600m_dashboard import snapshot
    contract = watch_contract()
    history = []
    for entry in contract.get('evaluation_history', []):
        path = ROOT / entry['path']
        try:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() == entry['sha256']:
                history.append(json.loads(raw))
        except (OSError, ValueError):
            continue
    return snapshot(state, contract, history)


def render(state):
    from scripts.monitor_600m_dashboard import render_text
    return render_text(dashboard_view(state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=5.)
    parser.add_argument('--steps', type=int, default=300000)
    parser.add_argument('--checkpoint-every', type=float, default=60.)
    parser.add_argument('--max-hours', type=float, default=0., help='0: run until stopped')
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    REPORTS.mkdir(exist_ok=True)
    lock = (REPORTS / '600m-monitor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('600m monitor is already running', file=sys.stderr)
        return 1
    state = State()
    state.total_steps = args.steps
    tail = None
    lines_out = queue.Queue()
    started = time.time()
    last_app_poll = 0.
    snapshot_thread = None
    try:
        with JSONL.open('a') as journal:
            while True:
                if time.time() - last_app_poll > 10:
                    app_id, app_state = active_app(args.app)
                    if app_state == 'discovery unavailable':
                        state.app_state = app_state
                    else:
                        active = watch_contract().get('active_run', {})
                        expected_call = active.get('call_id') if active.get('app_id') == app_id else None
                        if app_id != state.app_id or (expected_call and expected_call != state.call_id):
                            if tail:
                                tail.kill()
                            lines_out = queue.Queue()
                            total = state.total_steps
                            state = State()
                            state.total_steps = total
                            state.select_app(app_id, app_state)
                            state.call_id = state.expected_call_id = expected_call
                            restore_monitor_history(state, watch_contract())
                            tail = Tail(app_id, lines_out, expected_call) if app_id else None
                            if tail:
                                tail.start()
                        state.app_state = app_state
                    last_app_poll = time.time()
                deadline = time.time() + args.interval
                while time.time() < deadline:
                    try:
                        kind, payload = lines_out.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if kind == 'line':
                        record = state.ingest_line(payload)
                        if record:
                            state.stream_error = None
                            journal.write(json.dumps(dict(record, app_id=state.app_id), ensure_ascii=False) + '\n')
                            journal.flush()
                    else:
                        state.stream_error = payload
                if (state.app_id and time.time() - state.last_checkpoint_poll > args.checkpoint_every
                        and (snapshot_thread is None or not snapshot_thread.is_alive())):
                    def update_snapshots(target=state):
                        target.checkpoints = list_checkpoints()
                        refresh_evaluation(target)
                        refresh_runtime(target)
                    snapshot_thread = threading.Thread(target=update_snapshots, daemon=True)
                    snapshot_thread.start()
                    state.last_checkpoint_poll = time.time()
                from scripts.monitor_600m_dashboard import render_text, render_html
                view = dashboard_view(state)
                text = render_text(view)
                html_path = REPORTS / '600m-monitor.html'
                html_tmp = html_path.with_suffix('.html.tmp')
                html_tmp.write_text(render_html(view))
                html_tmp.replace(html_path)
                temporary = STATUS.with_suffix('.md.tmp')
                temporary.write_text('```text\n'+text+'```\n')
                os.replace(temporary, STATUS)
                if args.watch or args.once:
                    sys.stdout.write('\033[2J\033[H' + text + '\n(ctrl-c to quit)\n')
                    sys.stdout.flush()
                if args.once or (args.max_hours > 0 and time.time() - started > args.max_hours * 3600):
                    return 0
    finally:
        if tail:
            tail.kill()
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
