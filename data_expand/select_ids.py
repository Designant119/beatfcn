#!/usr/bin/env python3
"""Pick a quality-filtered expansion set from phira_all_charts.jsonl.
Rules: reviewed only; drop troll/visual divisions; ranked/stable first, then rating*log(1+count);
require ratingCount>=20 and rating>=0.8 for unranked; at most 2 charts per song (normalized name),
preferring higher difficulty among the qualified. Existing ids count toward the per-song cap.
Usage: select_ids.py <jsonl> <existing_ids.txt> <target_total> <out_ids.txt>"""
import sys, json, math, re, collections
jl, ex, target, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
R = [json.loads(l) for l in open(jl, encoding='utf-8') if l.strip()]
have = set(open(ex).read().split())
def div(r):
    for t in r.get('tags') or []:
        if t in ('regular', 'plain', 'troll', 'visual'): return t
    return ''
def key(r): return re.sub(r'[\W_]+', '', (r.get('name') or '').lower())
per = collections.Counter(key(r) for r in R if str(r['id']) in have)
seen_file = set(r.get('file') for r in R if str(r['id']) in have)
def ok(r):
    if not r.get('reviewed') or div(r) in ('troll', 'visual') or not r.get('file'): return False
    if r.get('ranked') or r.get('stable'): return True
    return (r.get('ratingCount') or 0) >= 20 and (r.get('rating') or 0) >= 0.8
def q(r):
    return (1 if (r.get('ranked') or r.get('stable')) else 0,
            round((r.get('rating') or 0) * math.log1p(r.get('ratingCount') or 0), 1),
            r.get('difficulty') or 0)
C = sorted((r for r in R if str(r['id']) not in have and ok(r)), key=q, reverse=True)
# group by song: take best-quality first, then within song prefer harder
bysong = collections.defaultdict(list)
for r in C: bysong[key(r)].append(r)
pick = []
for r in C:
    k = key(r)
    if per[k] >= 2 or r['file'] in seen_file or not k: continue
    cands = [c for c in bysong[k] if str(c['id']) not in have and c['file'] not in seen_file]
    top = sorted(cands[:4], key=lambda c: c.get('difficulty') or 0, reverse=True)[0]
    pick.append(top); per[k] += 1; seen_file.add(top['file']); have.add(str(top['id']))
    if len(pick) + 0 >= target - len(set(open(ex).read().split())): break
open(out, 'w').write('\n'.join(str(r['id']) for r in pick) + '\n')
print('candidates', len(C), 'songs', len(bysong), 'picked', len(pick))
print(collections.Counter((bool(r.get('ranked') or r.get('stable')), div(r)) for r in pick))
