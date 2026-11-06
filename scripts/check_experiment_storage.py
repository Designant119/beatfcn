"""Local experiment disk admission and read-only checkpoint retention inventory."""
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30


def admission(path, planned_bytes, reserve_bytes):
    usage = shutil.disk_usage(path)
    ok = usage.free - planned_bytes >= reserve_bytes
    return dict(status='admitted' if ok else 'blocked',
        free_bytes=usage.free, planned_peak_new_bytes=planned_bytes,
        reserve_bytes=reserve_bytes, projected_free_bytes=usage.free-planned_bytes,
        deletes_files=False)


def inventory():
    """Candidate list is reviewable, not deletion authority or a reachability proof."""
    from scripts.validate_mechanism_reviews import load_registry
    pinned=set(); references=set()
    for entry in load_registry():
        review=json.loads((ROOT/entry['review']).read_text())
        pinned.update(k for k in review['identity'] if k.endswith('.pt'))
    pattern=re.compile(r'artifacts/[A-Za-z0-9_.%/{}:-]+\.pt')
    for directory in ('reports','bench','scripts','design','beatfcn'):
        for path in (ROOT/directory).rglob('*'):
            if not path.is_file() or path.is_symlink() or path.suffix not in ('.json','.jsonl','.py','.md'): continue
            # Large files remain explicit unknowns; no "unreferenced" proof claimed.
            if path.stat().st_size > 16*1024*1024: continue
            references.update(pattern.findall(path.read_text(errors='replace')))
    by_kind=defaultdict(lambda:dict(bytes=0,files=0)); candidates=[]; protected=[]
    for path in sorted((ROOT/'artifacts').rglob('*.pt')):
        if not path.is_file() or path.is_symlink(): continue
        rel=str(path.relative_to(ROOT)); size=path.stat().st_size
        match=re.fullmatch(r'(health|quality)-(\d+)\.pt',path.name)
        kind='other' if not match else match[1]
        by_kind[kind]['bytes']+=size;by_kind[kind]['files']+=1
        terminals=[p for p in path.parent.glob('quality-*.pt')
            if re.fullmatch(r'quality-(0900|1000|2000)\.pt',p.name)]
        reason=None
        if rel in pinned: reason='Exact path pinned by a completed mechanism review'
        elif rel in references: reason='Literal path found in source/report/plan text'
        elif not match or not rel.startswith('artifacts/local-'): reason='Outside recognized local health/intermediate policy'
        elif match[1]=='quality' and int(match[2]) in (900,1000,2000): reason='Potential preregistered endpoint; retained'
        elif not terminals: reason='No recognized final checkpoint in this directory'
        item=dict(path=rel,bytes=size)
        if reason: protected.append(dict(**item,reason=reason))
        else: candidates.append(dict(**item,kind=match[1],retained_endpoint_paths=[str(p.relative_to(ROOT)) for p in terminals],
            needs_owner_scope_approval=True,needs_final_use_check=True))
    return dict(status='read_only_inventory',recorded_at=datetime.now(timezone.utc).isoformat(),
        disk=shutil.disk_usage(ROOT)._asdict(),by_kind=dict(by_kind),
        candidate_bytes=sum(v['bytes'] for v in candidates),candidate_count=len(candidates),
        candidates=candidates,protected=protected,
        limitations=['Literal references and registered SHA pins are checked;dynamic loads and omitted large text are not fully resolved.',
            'Names alone do not prove a checkpoint dispensable;this report neither deletes nor authorizes deletion.',
            'Original archives,features,raw predictions,probability traces,Council opinions and terminal weights remain retained.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--planned-gib',type=float,default=1)
    parser.add_argument('--reserve-gib',type=float,default=15)
    parser.add_argument('--inventory',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in (args.planned_gib,args.reserve_gib)):
        parser.error('Disk budgets must be finite and nonnegative')
    result=inventory() if args.inventory else admission(ROOT,int(args.planned_gib*GIB),int(args.reserve_gib*GIB))
    result['recorded_at']=datetime.now(timezone.utc).isoformat()
    if args.output:
        if args.output.exists(): raise FileExistsError('Keep prior audit receipts;use a new output version')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidates','protected')},ensure_ascii=False,indent=2))
    if result['status']=='blocked': raise SystemExit(2)


if __name__=='__main__': main()
