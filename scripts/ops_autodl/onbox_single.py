"""onbox_single.py ARM: single-sample score (sample 0) and playability faults per minute before clean-up, all 8 samples. CPU only."""
import json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
sys.path[:0] = ['/root/beatfcn', '/root/beatfcn/scripts']
import run_workflow_benchmark as W
from beatfcn.events.phira_postprocess import playable
G = sys.argv[1].rstrip('/')+'/'
O = G+'single/'
os.makedirs(O, exist_ok=True)
meta = json.load(open(G+'run-meta.json'))
smp = json.load(open(G+'drafts-samples.json'))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
KEYS = ('fragments_dropped', 'chords_cleaned', 'holds_shortened', 'holds_to_tap', 'dropped', 'shortened', 'to_tap', 'types_cleaned')


def job(s):
    ds = [W.draft_tuples(s, smp[str(i)][s['song_id']]) for i in range(N)]
    faults, notes = {k: 0 for k in KEYS}, 0
    for d in ds:
        notes += len(d)
        if s['mode'] == 'phira':
            _, report = playable([dict(onset_us=a, end_us=b, type=t, lane=l) for a, b, t, l in d], 'standard')
            for k in KEYS:
                faults[k] += report.get(k, 0)
    first = dict(events=[dict(onset_us=a, end_us=b, type=t, lane=l) for a, b, t, l in ds[0]])
    return s['song_id'], s['mode'], s['duration_us'], first, faults, notes


songs = [s for s in meta['panel']['songs'] if all(s['song_id'] in smp[str(i)] for i in range(N))]
rows, total = {}, {}
with ProcessPoolExecutor(8) as ex:
    for sid, mode, dur, first, faults, notes in ex.map(job, songs):
        rows[sid] = first
        t = total.setdefault(mode, dict(minutes=0., notes=0, songs=0, **{k: 0 for k in KEYS}))
        t['minutes'] += dur/60e6*N
        t['notes'] += notes
        t['songs'] += 1
        for k in KEYS:
            t[k] += faults[k]
for mode, t in total.items():
    t['notes_per_min'] = t['notes']/t['minutes']
    t['faults_per_min'] = sum(t[k] for k in KEYS)/t['minutes']
    print(mode, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in t.items()}), flush=True)
json.dump(total, open(O+'faults.json', 'w'), indent=1)
json.dump(rows, open(O+'drafts.json', 'w'))
T = '/data/full-database-v1/native-v2-tape'
subprocess.run(['/root/envs/bf/bin/python', 'scripts/benchmark_v3.py', 'score', '--drafts', O+'drafts.json', '--panel', G+'run-meta.json', '--out', O+'v3',
                '--floor', '/data/evolve/floor.json', '--tape', T, '--timing', '/data/full-database-v1/chart-timing/chart_timing-v2.json', '--features-dir', '/root/data/feat_new/features'],
               cwd='/root/beatfcn', check=True, env=dict(os.environ, OMP_NUM_THREADS='1', BEATFCN_ALLOW_MISSING_MERT='1'))
print('single done', flush=True)
