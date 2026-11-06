"""Sampled onsets vs human onsets on the exact windows R1-peer sampled. CPU only. Controls for the window mix."""
import json, sys
from pathlib import Path
sys.path[:0] = ['/root/beatfcn']
import numpy as np
import torch
from scripts.evolve_finetune import load_recipe
from beatfcn.train.training_panel import build_training_panel
spec = json.load(open('/data/evolve/plans/R1-peer.json'))
saved = torch.load(spec['init'], map_location='cpu', weights_only=False, mmap=True)
recipe = load_recipe(spec, saved['config']); del saved
panel = build_training_panel(Path('/'), recipe)
steps = json.load(open(spec['exposure']['path']))
steps = steps['steps'] if isinstance(steps, dict) else steps
ACCUM, STRIDE, REUSE = 16, 2, 4
def window(u):
    k = u*(ACCUM//STRIDE)+(ACCUM//STRIDE-1)  # last loader item of update u; it holds base steps 2k, 2k+1
    a, b = steps[2*k], steps[2*k+1]
    return (a['indices']+b['indices'])[:8], (a['rates']+b['rates'])[:8]
memo = {}
def human(i):
    if i not in memo:
        memo[i] = len({int(e.onset_us) for e in panel.targets(i)})
    return memo[i]
log = [json.loads(x) for x in open('/root/data/evolve-R1-peer.log') if '"phase": "train"' in x]
rows, done = [], 0
for rec in log:
    u_hi = rec['updates']
    want = want_edit = want_mania = 0.; n = 0
    for u in range(done, u_hi):
        ids, rates = window(u-u % REUSE)
        for i, r in zip(ids, rates):
            want += human(i)*(1. if r is None else float(r)); n += 1
    rows.append((u_hi, rec['rl'].get('onsets', 0)/8, want, rec['rl'].get('windows', 0), n))
    done = u_hi
B = 10
for s in range(0, len(rows), B):
    p = rows[s:s+B]
    got, want, w, n = sum(x[1] for x in p), sum(x[2] for x in p), sum(x[3] for x in p), sum(x[4] for x in p)
    per = [x[1]/max(x[2], 1) for x in p]
    print(p[-1][0], 'model/human onsets %.3f +- %.3f' % (got/max(want, 1), np.std(per, ddof=1)/len(per)**.5), 'model %.1f human %.1f per window' % (got/max(w, 1), want/max(n, 1)), 'logged windows', w, 'reconstructed', n, flush=True)
