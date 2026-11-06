"""Post-hoc diagnostic only: exact simultaneous type multisets and hold10.

No model execution, threshold search, checkpoint selection or primary-score
replacement. Matching is maximum cardinality on the declared valid edges.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.run_local_portfolio_scale import sha,write


ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/local_portfolio_multiset_audit_v1.json'


def matched(left,right,allowed):
    edges=[[j for j,b in enumerate(right) if allowed(a,b)] for a in left]
    owner={}
    def visit(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owner or visit(owner[j],seen):owner[j]=i;return True
        return False
    return sum(visit(i,set()) for i in range(len(left)))


def groups(events):
    result=defaultdict(Counter)
    for e in events:result[e['onset_us']][e['type']]+=1
    return sorted(result.items())


def one(rows):
    totals=Counter();components=defaultdict(Counter)
    for row in rows:
        pred=row['prediction']['events'];truth=row['target']
        pg,tg=groups(pred),groups(truth);counts=Counter()
        for name,condition in [('multi',lambda c:sum(c.values())>=2),
            ('mixed',lambda c:len(c)>=2),('repeated',lambda c:max(c.values())>=2)]:
            targets=[v for v in tg if condition(v[1])]
            correct=matched(targets,pg,lambda a,b:abs(a[0]-b[0])<=10000 and a[1]==b[1])
            counts[name+'_true']+=len(targets);counts[name+'_matched']+=correct
        holds=[v for v in truth if v['type']=='hold'];ph=[v for v in pred if v['type']=='hold']
        counts['hold10_true']+=len(holds);counts['hold10_pred']+=len(ph)
        counts['hold10_matched']+=matched(holds,ph,lambda a,b:abs(a['onset_us']-b['onset_us'])<=10000 and abs(a['end_us']-b['end_us'])<=10000)
        totals.update(counts);components[row['component']].update(counts)
    def summarize(c):
        return {**dict(c),**{name+'_recall':c[name+'_matched']/c[name+'_true'] if c[name+'_true'] else None for name in ('multi','mixed','repeated','hold10')}}
    return dict(pooled=summarize(totals),components={k:summarize(v) for k,v in components.items()})


def main():
    if REPORT.exists():raise FileExistsError('Preserve multiset audit')
    candidates={
        'grouped_ar_v2':[f'artifacts/local-grouped-chart-v2/draft-sample{i}.json' for i in range(4)],
        'whole_song_B':[f'artifacts/local-whole-song-arrangement-v1/draft-sample{i}.json' for i in range(4)],
        'ssd1000':['artifacts/local-mamba-readout-v1/draft-expected_tail_rank.json'],
        'lattice':['artifacts/local-lattice-complete-v1/draft-final.json']}
    report=dict(status='completed',role='Post-hoc disjoint diagnostic, no new gate or primary change',
        tolerance_us=10000,optimizer_updates=0,cloud_compute_calls=0,results={},source_sha256={})
    for name,paths in candidates.items():
        values=[]
        for path in paths:
            full=ROOT/path;report['source_sha256'][path]=sha(full)
            values.append(one(json.loads(full.read_text())))
        report['results'][name]=values
    write(REPORT,report)


if __name__=='__main__':main()
