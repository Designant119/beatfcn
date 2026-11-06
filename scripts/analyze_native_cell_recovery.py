"""Terminal raw-fact extraction only; never auto-closes a mechanism review."""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/c12-native-cell-recovery-v1'
DEST=ROOT/'reports/local-integration-v1/c12-recovery-terminal-facts.json'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text())


def mean(values):return float(np.mean(values)) if values else None


def p2_statistics(path):
    r=read(path);base=r['arms']['base'];arms={}
    for name,rows in r['arms'].items():
        for b,a in zip(base,rows):
            if (b['cell'],b['rank'],b['head_us'])!=(a['cell'],a['rank'],a['head_us']):raise ValueError('P2 fixed tape identity')
        types=[a['type_nll']-b['type_nll'] for b,a in zip(base,rows) if not b['forced'] and b.get('type_nll') is not None and a.get('type_nll') is not None]
        endpoints=[a['endpoint_true_path']['nll']-b['endpoint_true_path']['nll'] for b,a in zip(base,rows)
                   if b.get('endpoint_true_path',{}).get('nll') is not None and a.get('endpoint_true_path',{}).get('nll') is not None]
        effect=[sum(abs(x-y) for x,y in zip(b['type_probability'],a['type_probability']))/2 for b,a in zip(base,rows) if not b['forced']]
        arms[name]=dict(nonforced_type_paired=len(types),type_nll_minus_base=mean(types),type_total_variation=mean(effect),
                       endpoint_true_path_pairs=len(endpoints),endpoint_nll_minus_base=mean(endpoints),
                       matched_unique_heads=sum(a['matched_unique_head'] for a in rows),forced_members=sum(a['forced'] for a in rows),
                       type_support_failures=sum(a.get('type_support_failure',False) for a in rows),
                       head_support_failures=sum(a.get('head_support_failure',False) for a in rows))
    return dict(index=r.get('index',int(path.stem.rsplit('-',1)[-1])),certificate=r['certificate'],field_identity_max_diff=r['field_identity_max_diff'],arms=arms,
                scope='Positive nll-minus-base favors original conditional prediction on fixed matched subset;not whole-law likelihood')


def main():
    if DEST.exists():raise FileExistsError('Preserve terminal analysis;newversion fornewanalysis')
    main_report=ROOT/'reports/local-integration-v1/native-cell-recovery.json';top=read(main_report)
    if top['status'].endswith('_running') or top['status']=='running':raise ValueError('Not terminal;no interim dev peeking')
    sources={str(main_report.relative_to(ROOT)):digest(main_report)}
    result=dict(id='c12-native-cell-recovery-v1',status='facts_ready_pending_parent_mechanism_review',terminal_status=top['status'],
                health={},quality={},tasks={},field_panels=[],cloud_spend_usd=0,
                scope='Raw terminal diagnostics;human/agent causal judgment stillrequired,not automatic closure')
    for stage in ('preflight','health','quality','evaluation'):
        path=ART/f'{stage}.json'
        if not path.exists():continue
        r=read(path);sources[str(path.relative_to(ROOT))]=digest(path)
        result[stage]={k:r.get(k) for k in ('status','updates','failure','wall_s','checks','research_checks','research_qualified','runtime_prefix','cpu_mps','applied_exposure','seen_types','accumulation_phase','checkpoint_sha256','non_tap_evidence')}
        telemetry=ART/f'{stage}-telemetry.json'
        if telemetry.exists():
            t=read(telemetry);result[stage]['telemetry']=dict(peak_sampled_physical_bytes=t['peak_sampled_physical_bytes'],error=t['error'],samples=len(t['samples']))
            sources[str(telemetry.relative_to(ROOT))]=digest(telemetry)
    for task in ('draft','empty','completion','reference_masked','density_low','density_high','audio_zero','audio_shift','audio_foreign'):
        path=ART/f'{task}.json'
        if not path.exists():continue
        rows=read(path);bad=[];counts=Counter();type_scores={};proper={}
        for r in rows:
            events=r['prediction']['events'];targets=r['target'];counts.update(e['type'] for e in events)
            unique=sorted({e['onset_us'] for e in events});gaps=np.diff(unique)
            duplicate=len(events)-len({(e['onset_us'],e['end_us'],e['type']) for e in events})
            bad.append(dict(index=r['index'],component=r['component'],complete=r['prediction']['complete'],failure=r['prediction']['failure'],
                full50=r['scores']['full_note_f1'],onset10=r['scores']['onset_10ms']['f1'],pred=len(events),target=len(targets),types=dict(Counter(e['type'] for e in events)),
                target_types=dict(Counter(e['type'] for e in targets)),duplicate_extras=duplicate,
                positive_gap_median_ms=float(np.median(gaps)/1000) if len(gaps) else None,
                positive_gap_under10_fraction=float(np.mean(gaps<10000)) if len(gaps) else None,
                legacy_tail_error=r['scores'].get('tail_error_given_head_10ms'),minimum_cost_tail=r['endpoint_min_cost']))
        for kind in ('tap','hold','flick','drag'):
            values=[r['scores']['per_type'][kind] for r in rows];tp=sum(v['tp'] for v in values);n=sum(v['pred'] for v in values);true=sum(v['true'] for v in values)
            type_scores[kind]=dict(tp=tp,pred=n,true=true,micro_f1=2*tp/max(n+true,1),matched_components=len({r['component'] for r in rows if r['scores']['per_type'][kind]['tp']>0}))
            by_component=defaultdict(list)
            for r in rows:by_component[r['component']].extend(r['teacher_scores']['type_binary'][kind])
            proper[kind]={c:dict(decisions=len(v),positive=sum(t['positive'] for t in v),proper_gain=mean([t['prior_nll']-t['nll'] for t in v])) for c,v in by_component.items()}
        result['tasks'][task]=dict(attempted=len(rows),complete=sum(r['prediction']['complete'] for r in rows),predicted_types=dict(counts),
            type_scores=type_scores,teacher_binary_proper_by_component=proper,
            worst_windows=sorted(bad,key=lambda r:r['full50'])[:8],all_windows=bad,
            teacher_forced_types=sum(r['teacher_scores']['forced'] for r in rows))
        sources[str(path.relative_to(ROOT))]=digest(path)
    for path in sorted(ART.glob('*-p2-*.json')):
        result['field_panels'].append(p2_statistics(path));sources[str(path.relative_to(ROOT))]=digest(path)
    result['identity']=sources
    DEST.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],terminal_status=result['terminal_status'],tasks=list(result['tasks']),field_panels=len(result['field_panels'])),ensure_ascii=False))


if __name__=='__main__':main()
