"""Stop Modal containers that bill without working (local watchdog, never trains).

Every --every-s seconds, exec a tiny probe into each running container of the
app: total CPU seconds of every process in the container + GPU util. A
container whose CPU time grew by < --min-cpu-s across --strikes consecutive
samples is dead (lost input, hung child, idle after cancel) and is stopped
with `modal container stop -y`. Setup/staging/CPU phases keep burning CPU, so
they are never touched. Containers younger than one interval are skipped.

  .venv/bin/python -m scripts.modal_watchdog [--app stage0-cuda-training] [--every-s 300]
"""
import argparse
import base64
import json
import subprocess
import time

MODAL = '.venv/bin/modal'
PROBE = r'''
import os,subprocess,json
hz=os.sysconf('SC_CLK_TCK');t=0.0;cmds=[]
for p in os.listdir('/proc'):
    if not p.isdigit(): continue
    try:
        s=open(f'/proc/{p}/stat').read().rsplit(')',1)[1].split()
        t+=sum(int(x) for x in s[11:15])/hz
        c=open(f'/proc/{p}/cmdline','rb').read().replace(b'\0',b' ').decode(errors='replace')
        if 'scripts.' in c: cmds.append(c[c.find('scripts.'):][:90])
    except Exception: pass
g=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip() if os.path.exists('/usr/bin/nvidia-smi') else ''
print('WD'+json.dumps(dict(cpu_s=round(t,1),gpu=g,cmds=cmds)))
'''


def containers(app):
    out = subprocess.run([MODAL, 'container', 'list', '--json'], capture_output=True, text=True, timeout=120).stdout
    return [c for c in json.loads(out or '[]') if c.get('app_name') == app]


def probe(container_id):
    code = base64.b64encode(PROBE.encode()).decode()
    try:
        out = subprocess.run([MODAL, 'container', 'exec', container_id, '--', 'python', '-c',
                              f"import base64;exec(base64.b64decode('{code}'))"],
                             capture_output=True, text=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        return None
    for line in out.splitlines():
        if line.startswith('WD'):
            return json.loads(line[2:])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app', default='stage0-cuda-training')
    p.add_argument('--every-s', type=int, default=300)
    p.add_argument('--min-cpu-s', type=float, default=2.0)
    p.add_argument('--strikes', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--max-hours', type=float, default=12)
    a = p.parse_args()
    last, strikes, end = {}, {}, time.time() + a.max_hours * 3600
    while time.time() < end:
        live = containers(a.app)
        seen = set()
        for c in live:
            cid = c['container_id']
            seen.add(cid)
            sample = probe(cid)
            stamp = time.strftime('%H:%M:%S')
            if sample is None:  # finished between list and exec, or exec failed: no evidence, no action
                print(json.dumps(dict(t=stamp, id=cid, probe='none')), flush=True)
                continue
            prev = last.get(cid)
            last[cid] = sample
            if prev is None:
                print(json.dumps(dict(t=stamp, id=cid, first=sample)), flush=True)
                continue
            delta = sample['cpu_s'] - prev['cpu_s']
            strikes[cid] = strikes.get(cid, 0) + 1 if delta < a.min_cpu_s else 0
            print(json.dumps(dict(t=stamp, id=cid, cpu_delta_s=round(delta, 1), gpu=sample['gpu'],
                                  strikes=strikes[cid], cmds=sample['cmds'][:1])), flush=True)
            if strikes[cid] >= a.strikes:
                print(json.dumps(dict(t=stamp, id=cid, action='STOP' + (' (dry-run)' if a.dry_run else ''),
                                      reason=f'cpu < {a.min_cpu_s}s per {a.every_s}s x{a.strikes}')), flush=True)
                if not a.dry_run:
                    subprocess.run([MODAL, 'container', 'stop', '-y', cid], timeout=120)
                strikes[cid] = 0
        for cid in list(last):
            if cid not in seen:
                last.pop(cid)
                strikes.pop(cid, None)
        time.sleep(a.every_s)


if __name__ == '__main__':
    main()
