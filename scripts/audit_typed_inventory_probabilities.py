"""Posthoc distribution arithmetic only; no new decode or neural prediction."""
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from beatfcn.model.typed_inventory import masked_scores, count_totals, count_log_marginals, count_modes
from beatfcn.train.exact_cell_support import ExactCellSupport, ExactEndpointTree
from beatfcn.train.native_diagnostic_priors import duration_probability
from scripts.run_local_portfolio_scale import sha, write


ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/c15-typed-inventory-v1'
DEST=ROOT/'reports/local-integration-v1/c15-typed-inventory-probability-audit.json'
KINDS=('tap','hold','flick','drag')


def mean(values):return float(np.mean(values)) if len(values) else None


def main():
    if DEST.exists():raise FileExistsError('Preserve posthoc audit')
    torch.set_num_threads(4)
    prior=json.loads((ROOT/'artifacts/c14-training-prior-audit-v1a/prior.json').read_text())
    result=dict(scope='Frozen probability/trace arithmetic;no model forward,new events,sampling,training or chosen replacement decoder',
        neural_forwards=0,new_free_outcomes=0,optimizer_updates=0,identity={},arms={},probability_controls={})
    for arm in ('A','C'):
        rows=json.loads((ART/f'{arm}-draft.json').read_text());totals=defaultdict(float);type_bins={k:defaultdict(list) for k in KINDS};per_component=defaultdict(list)
        head=tail=uniform_head=uniform_tail=duration=0.;heads=holds=empty_windows=0;count_values=[];old_modes_exact=0
        for r in rows:
            path=ROOT/r['probability_path']
            if sha(path)!=r['probability_sha256']:raise ValueError('Saved probabilities changed')
            result['identity'][r['probability_path']]=r['probability_sha256']
            with np.load(path,allow_pickle=False) as pack:
                scores=torch.tensor(pack['scores'][None],dtype=torch.float64);overflow=torch.tensor(pack['overflow'][None],dtype=torch.float64)
                allowed=torch.tensor(pack['allowed'][None],dtype=torch.bool);truth=pack['target'].copy()
            masked=masked_scores(scores,allowed);total=count_totals(masked);z=total.logsumexp(-1)
            editable=allowed.any(-1);lognon=torch.where(editable,torch.nn.functional.logsigmoid(-overflow),torch.zeros_like(overflow))
            distribution=(total-z[...,None]+lognon[...,None]).exp()[0].numpy()
            marginal=count_log_marginals(scores,overflow,allowed).exp()[0].numpy()
            number=np.arange(65);expectation=(marginal[...,:65]*number).sum(-1);pzero=distribution[:,0]
            modes=np.array(r['prediction']['count_decision'],np.int64)
            rebuilt,failures=count_modes(scores,overflow,allowed)
            if failures or not np.array_equal(rebuilt,modes):raise ValueError('Float64 probability mode disagrees with deployed mode')
            old_modes_exact+=1
            independent=masked[0].argmax(-1).numpy()
            if not np.array_equal(independent,modes):raise ValueError('Capacity-binding case requires separate interpretation')
            positive=truth.sum(-1)>0;zero=~positive;allzero=modes.sum(-1)==0;cmode=distribution.argmax(-1)
            totals['expected_events']+=expectation.sum();totals['expected_at_positive_onset_cells']+=expectation[positive].sum();totals['expected_at_zero_onset_cells']+=expectation[zero].sum()
            totals['expected_at_positive_type_cells']+=expectation[truth>0].sum();totals['expected_at_zero_type_cells']+=expectation[truth==0].sum()
            totals['overflow_probability_sum']+=np.where(editable[0].numpy(),torch.sigmoid(overflow)[0].numpy(),0).sum()
            totals['positive_onset_cells']+=positive.sum();totals['zero_onset_cells']+=zero.sum();totals['empty_vector_cells']+=allzero.sum()
            totals['true_events_under_empty_vector']+=truth[allzero].sum()
            totals['true_events_under_empty_vector_Ptotal0_ge_half']+=truth[allzero&(pzero>=.5)].sum()
            totals['true_events_under_empty_vector_Ptotal0_lt_half']+=truth[allzero&(pzero<.5)].sum()
            totals['cells_empty_vector_but_total_mode_positive']+=int((allzero&(cmode>0)).sum())
            totals['true_events_empty_vector_but_total_mode_positive']+=truth[allzero&(cmode>0)].sum()
            totals['total_marginal_mode_event_sum_diagnostic_only']+=cmode.sum()
            empty_windows+=not len(r['prediction']['events']);count_values.append(dict(index=r['index'],component=r['component'],pred=int(modes.sum()),true=int(truth.sum()),expected=float(expectation.sum()),p0_true_cell_median=float(np.median(pzero[positive])) if positive.any() else None))
            for k,kind in enumerate(KINDS):
                pos=truth[:,k]>0
                pp=1-marginal[:,k,0]-marginal[:,k,65]
                type_bins[kind]['positive_presence_probability'].extend(pp[pos].tolist())
                type_bins[kind]['positive_true_count_probability'].extend(marginal[np.flatnonzero(pos),k,truth[pos,k]].tolist())
                type_bins[kind]['zero_presence_probability'].extend(pp[(~pos)&allowed[0,:,k].numpy()].tolist())
                type_bins[kind]['expected_at_positive_type'].append(float(expectation[pos,k].sum()))
                type_bins[kind]['expected_at_zero_type'].append(float(expectation[~pos,k].sum()))
            for name in ('joint_count_nll','prior_joint_count_nll'):
                per_component[r['component']].append((name,mean(r[name])))
            request=type('Request',(),r['request'])()
            values,valid=ExactCellSupport().addresses(request)
            by=defaultdict(list)
            for event in r['target']:by[((event['onset_us']-request.start_us)//40000,event['type'])].append(event)
            for (cell,kind),events in by.items():
                previous=None
                for e in sorted(events,key=lambda v:(v['onset_us'],v['end_us'])):
                    mask=valid[cell].copy()
                    if previous:mask&=values[cell]>=previous['onset_us']
                    if kind=='hold':mask&=(values[cell]//1000+1)*1000<=request.duration_us
                    uniform_head+=math.log(int(mask.sum()));heads+=1
                    if kind=='hold':
                        minimum=previous['end_us'] if previous and previous['onset_us']==e['onset_us'] else None
                        tree=ExactEndpointTree(request,e['onset_us'],minimum);b,c,leaf=tree.path(e['end_us'])
                        uniform_tail+=math.log(len(tree.blocks()))+math.log(len(tree.cells(b)))+math.log(len(tree.leaves(b,c)))
                        duration-=math.log(duration_probability(prior,e['onset_us'],e['end_us'],request.duration_us,request.copy_times,minimum));holds+=1
                    previous=e
            head+=r['teacher_head_nll'];tail+=r['teacher_tail_nll']
        types={}
        for k,bins in type_bins.items():
            types[k]=dict(positive_cells=len(bins['positive_presence_probability']),zero_cells=len(bins['zero_presence_probability']),
                positive_presence_mean=mean(bins['positive_presence_probability']),positive_presence_median=float(np.median(bins['positive_presence_probability'])),
                positive_presence_ge_half=sum(p>=.5 for p in bins['positive_presence_probability']),
                positive_true_count_probability_mean=mean(bins['positive_true_count_probability']),zero_presence_mean=mean(bins['zero_presence_probability']),
                expected_at_positive_type=sum(bins['expected_at_positive_type']),expected_at_zero_type=sum(bins['expected_at_zero_type']))
        result['arms'][arm]=dict(totals=dict(totals),types=types,empty_draft_windows=empty_windows,requests=count_values,
            deployed_mode_reproduced=old_modes_exact,capacity_constraint_nonbinding_in_mode=True,
            teacher=dict(events=heads,holds=holds,head_nll_per_event=head/heads,uniform_head_nll_per_event=uniform_head/heads,
                tail_nll_per_hold=tail/holds,uniform_tree_tail_nll_per_hold=uniform_tail/holds,duration_prior_nll_per_hold=duration/holds))
        tasks={task:json.loads((ART/f'{arm}-{task}.json').read_text()) for task in ('draft','audio_shift','audio_foreign','audio_zero')}
        def groups_of(task):
            groups=defaultdict(list)
            for r in tasks[task]:groups[r['component']].append(mean(r['joint_count_nll']))
            return {c:mean(v) for c,v in groups.items()}
        base=groups_of('draft');result['probability_controls'][arm]={}
        for task in tasks:
            values=groups_of(task);delta={c:values[c]-base[c] for c in base}
            result['probability_controls'][arm][task]=dict(joint_nll_component_macro=mean(list(values.values())),nll_increase_from_real=mean(list(delta.values())),components_worse_than_real=sum(v>0 for v in delta.values()),per_component_delta=delta)
    write(DEST,result)
    print(json.dumps(dict(status='completed',A=result['arms']['A']['totals'],C=result['arms']['C']['totals'],new_model_calls=0),ensure_ascii=False))


if __name__=='__main__':main()
