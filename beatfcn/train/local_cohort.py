"""Song-connected, nested candidate selection; never grants training admission."""
from collections import defaultdict
import hashlib


def connected_songs(rows):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for row in rows:
        if not row.get('audio_sha256') or not row.get('song_group'):
            continue
        a,b=find('a:'+row['audio_sha256']),find('g:'+row['song_group'])
        parent[max(a,b)]=min(a,b)
    identity={};splits=defaultdict(set)
    for row in rows:
        if not row.get('audio_sha256') or not row.get('song_group'):
            continue
        component=find('a:'+row['audio_sha256'])
        identity[row['chart_id']]=component
        splits[component].add(row.get('split','unknown'))
    return identity,dict(splits)


def stratified_song_order(candidates,*,seed,used=()):
    """A deterministic prefix order: round-robin native strata, unique components.

    A song with multiple charts may enter several strata; once chosen, all aliases
    are skipped. The selected representative stratum is retained for auditing.
    """
    buckets=defaultdict(list)
    def key(row):
        return hashlib.sha256(f"{seed}:{row['component']}:{row['chart_id']}".encode()).hexdigest()
    for row in candidates:buckets[row['stratum']].append(row)
    for rows in buckets.values():rows.sort(key=key,reverse=True)
    names=sorted(buckets,key=lambda name:(len({r['component'] for r in buckets[name]}),name))
    seen=set(used);result=[]
    while True:
        changed=False
        for name in names:
            rows=buckets[name]
            while rows and rows[-1]['component'] in seen:rows.pop()
            if rows:
                chosen=rows.pop();seen.add(chosen['component']);result.append(chosen);changed=True
        if not changed:return result


def estimate_feature_bytes(audio_seconds):
    """Legacy int8 3 x 8 x 1024 at 25Hz plus fp16 128-bin mel at 100Hz.

    This is a planning estimate. Remote file sizes and array headers must confirm it.
    No chart-derived summary is included; summaries are built from audio locally.
    """
    cells=int(audio_seconds*25)
    return 3*(128+cells*8*1024)+128+cells*4*128*2
