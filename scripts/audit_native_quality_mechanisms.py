"""Zero-model-call decomposition of frozen C13 probabilities and predictions."""
from collections import Counter,defaultdict
import json
import math

import numpy as np

from scripts.run_local_portfolio_scale import ROOT,write,sha


ART=ROOT/'artifacts/c13-native-quality-v1'
DEST=ROOT/'reports/local-integration-v1/c13-quality-mechanism-audit.json'


def mean(values):return float(np.mean(values)) if len(values) else None


def count_facts(rows):
    probabilities=np.concatenate([np.asarray(r['count_probability']) for r in rows]);truth=np.concatenate([r['count_target'] for r in rows])
    mode=probabilities.argmax(-1);nonempty=truth>0;prediction=mode>0;p=1-probabilities[:,0]
    expected=(probabilities[:,:65]*np.arange(65)[None]).sum(-1)
    nll=-np.log(np.maximum(probabilities[np.arange(len(truth)),truth],1e-300))
    bins=[]
    for low,high in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
        at=(p>=low)&(p<high if high<1 else p<=high)
        bins.append(dict(low=float(low),high=float(high),cells=int(at.sum()),average_probability=mean(p[at]),observed_positive_fraction=mean(nonempty[at])))
    return dict(cells=len(truth),target_events=int(truth.sum()),mode_events=int(mode.sum()),nonoverflow_expected_events=float(expected.sum()),
        overflow_mass=float(probabilities[:,65].sum()),positive_cells=int(nonempty.sum()),mode_positive_cells=int(prediction.sum()),
        missed_positive_cells=int((nonempty&~prediction).sum()),false_positive_cells=int((~nonempty&prediction).sum()),
        cell_presence_precision=float((nonempty&prediction).sum()/max(prediction.sum(),1)),cell_presence_recall=float((nonempty&prediction).sum()/max(nonempty.sum(),1)),
        mean_presence_probability_on_positive=mean(p[nonempty]),mean_presence_probability_on_zero=mean(p[~nonempty]),
        expected_at_missed_positive_cells=float(expected[nonempty&~prediction].sum()),
        expected_at_zero_cells=float(expected[~nonempty].sum()),count_nll_positive=mean(nll[nonempty]),count_nll_zero=mean(nll[~nonempty]),
        true_count_histogram=dict(Counter(map(int,truth))),mode_count_histogram=dict(Counter(map(int,mode))),reliability=bins,
        scope='Frozen per-cell distribution;expected counts do not establish that hidden events or heads are correct')


def main():
    if DEST.exists():raise FileExistsError('Preserve mechanism audit')
    rows=json.loads((ART/'draft.json').read_text());baseline=json.loads((ROOT/'artifacts/local-evolution-v1/ssd_full/quality-1000-draft.json').read_text())
    if len(rows)!=len(baseline) or not all(a['index']==b['index'] and a['target']==b['target'] for a,b in zip(rows,baseline)):
        raise ValueError('Frozen baseline target identity differs')
    heads=defaultdict(list);types={k:dict(target=0,teacher_argmax_correct=0,forced=0,positive=[],negative=[]) for k in ('tap','hold','flick','drag')}
    factors=Counter();members=0;forced=0;underflow=[]
    for r in rows:
        ss=r['teacher_scores'];factors.update(ss['exact_factor_nll']);members+=len(r['teacher_steps']);forced+=ss['forced']
        for h in ss['head_decisions']:
            if h['nll'] is not None and h['legal_addresses']>1:heads[h['type']].append(dict(nll=h['nll'],uniform_nll=math.log(h['legal_addresses'])))
        for step in r['teacher_steps']:
            k=step['event']['type'];types[k]['target']+=1;types[k]['teacher_argmax_correct']+=int(np.argmax(step['type_probability']))==('tap','hold','flick','drag').index(k)
            types[k]['forced']+=sum(step['kind_allowed'])==1
        for k in types:
            for d in ss['type_binary'][k]:types[k]['positive' if d['positive'] else 'negative'].append(d['prior_nll']-d['nll'])
        underflow.extend(ss['numerical_or_support_failures'])
    for k,v in types.items():
        for name in ('positive','negative'):
            a=v.pop(name);v[name+'_proper_decisions']=len(a);v[name+'_proper_gain']=mean(a)
    hold_pairs=[p for r in rows for p in r['endpoint_min_cost'].get('pairs',[])]
    density_low=json.loads((ART/'density_low.json').read_text());density_high=json.loads((ART/'density_high.json').read_text())
    completion=json.loads((ART/'completion.json').read_text());masked=json.loads((ART/'reference_masked.json').read_text())
    identity={str((ART/n).relative_to(ROOT)):sha(ART/n) for n in ('draft.json','empty.json','completion.json','reference_masked.json','density_low.json','density_high.json')}
    identity['artifacts/local-evolution-v1/ssd_full/quality-1000-draft.json']=sha(ROOT/'artifacts/local-evolution-v1/ssd_full/quality-1000-draft.json')
    result=dict(status='completed',model_calls=0,optimizer_updates=0,baseline_targets_equal=True,identity=identity,
        count=count_facts(rows),empty_count=count_facts(json.loads((ART/'empty.json').read_text())),
        teacher=dict(events=members,forced_type=forced,exact_factors=dict(factors),numerical_or_support_failures=underflow,
            head={k:dict(decisions=len(v),nll=mean([a['nll'] for a in v]),uniform_nll=mean([a['uniform_nll'] for a in v]),uniform_gain=mean([a['uniform_nll']-a['nll'] for a in v])) for k,v in heads.items()},types=types),
        hold=dict(head_matches50=sum(r['scores']['hold_head_50ms']['tp'] for r in rows),joint_matches50=sum(r['scores']['hold_joint_50ms']['tp'] for r in rows),minimum_cost_pairs=hold_pairs),
        density=[dict(index=a['index'],events_low=len(a['prediction']['events']),events_high=len(b['prediction']['events']),identical=a['prediction']['events']==b['prediction']['events']) for a,b in zip(density_low,density_high)],
        reference=[dict(index=a['index'],full50_delta=a['scores']['full_note_f1']-b['scores']['full_note_f1'],onset10_delta=(a['scores']['onset_10ms']['f1'] or 0)-(b['scores']['onset_10ms']['f1'] or 0),identical=a['prediction']['events']==b['prediction']['events']) for a,b in zip(completion,masked)],
        scope='Read-only diagnosis;teacher is oracle-conditioned;no redecoding,threshold search,new predictions,training or cloud')
    write(DEST,result);print(json.dumps({k:result[k] for k in ('count','empty_count','teacher','density')},ensure_ascii=False))


if __name__=='__main__':main()
