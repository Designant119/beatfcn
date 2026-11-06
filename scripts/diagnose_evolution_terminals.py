"""Zero-update factorial/readout diagnostics; never replace terminal scores."""
from dataclasses import asdict
from itertools import groupby
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from beatfcn.model.grouped_chart_process import GroupedChartProcess, generate_grouped_process
from beatfcn.model.local_evolution_controls import AudioInteractionGroupedProcess, reset_mamba_boundaries
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.local_typed_canvas import best_tail, canvas_loss
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_probability import explicit_row, probability_batch
from beatfcn.train.relative_canvas import relative_batch, generate_relative
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory, sha, write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/evolution-terminal-diagnostics-v1'
PLAN=ROOT/'bench/rhythm-v2/evolution-terminal-diagnostics-v1.json'
REPORT=ROOT/'reports/local-evolution-v1/terminal-diagnostics.json'


def read(p):return json.loads((ROOT/p).read_text())


class FrozenOutput(nn.Module):
    def __init__(self,output):super().__init__();self.output=output
    def forward(self,inputs):return self.output


def record(e,pred,panel,shifts):
    events=pred['events'];typed=[__import__('beatfcn.model.event_canvas',fromlist=['RhythmEvent']).RhythmEvent(**v) for v in events]
    s=score(typed,e['targets']);start=e['request'].start_us;width=e['request'].end_us-start
    null=[match_f1([start+(v['onset_us']-start+d)%width for v in events],[v.onset_us for v in e['targets']],10000)[3] for d in shifts]
    return dict(index=e['index'],component=panel.records[e['index']]['component'],request=asdict(e['request']),target=[asdict(v) for v in e['targets']],prediction=pred,scores=s,shifted_onset10=null)


@torch.no_grad()
def ar_teacher(model,inp,e):
    context=model.context(inp,e['request']);hidden=model.chart_start;previous=None;rows=[]
    for head,iterator in groupby(sorted(e['targets'],key=lambda v:v.onset_us),key=lambda v:v.onset_us):
        members=list(iterator);candidate=int((context['head']==head).nonzero().flatten()[0]);frame=int(context['frame'][candidate])
        law=model.head_law(context,hidden,e['request'],previous)
        ids=((context['frame']==frame)&torch.isfinite(law[:-1])).nonzero().flatten()
        sub=law[ids].log_softmax(0);where=int((ids==candidate).nonzero().flatten()[0]);best=int(sub.argmax())
        phase=context['head'][ids]-e['request'].start_us
        seconds=phase.float()/1e6
        clock=torch.stack((seconds/8,torch.sin(seconds*math.pi),torch.cos(seconds*math.pi),(phase%1000).float()/1000),-1)
        audio=model.address_key(context['x'][(frame+37)%len(context['x'])]);clock=model.physical_phase(clock)
        keys=audio+clock
        if isinstance(model,AudioInteractionGroupedProcess):keys=keys+audio*clock
        swapped={**context,'keys':context['keys'].clone()};swapped['keys'][ids]=keys
        displaced=model.head_law(swapped,hidden,e['request'],previous)[ids].log_softmax(0)
        rows.append(dict(nll=float(-sub[where]),uniform_nll=math.log(len(ids)),
            displaced_frame_nll=float(-displaced[where]),cell_probability=float(law[ids].exp().sum()),
            argmax_error_ms=abs(int(context['head'][ids[best]])-head)/1000,
            argmax_at_legal_boundary=best in (0,len(ids)-1),entropy=float(-(sub.exp()*sub).sum()),
            entropy_uniform=math.log(len(ids))))
        counts=torch.tensor([sum(v.type==k for v in members) for k in ('tap','hold','flick','drag')])
        _,hidden,_,_=model.emit_group(context,hidden,candidate,counts,inp,e['request'],events=members);previous=head
    return rows


def means(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


@torch.no_grad()
def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Keep diagnostic attempts')
    torch.set_num_threads(2);p=read('bench/rhythm-v2/local-evolution-v1.json')
    panel,store=setup(read('bench/rhythm-v2/local-frontend-comparison-v2.json'));row=row_factory(panel,store)
    seen=set();indices=[]
    for i in p['ar']['indices']['dev']:
        c=panel.records[i]['component']
        if c not in seen:seen.add(c);indices.append(i)
    weights=dict(ar_old='artifacts/local-grouped-chart-v2/quality-1000.pt',
        ar_new='artifacts/local-evolution-v1/ar_interaction/quality-1000.pt',
        ssd_full='artifacts/local-evolution-v1/ssd_full/quality-1000.pt',ssd_reset='artifacts/local-evolution-v1/ssd_reset/quality-1000.pt')
    plan=dict(id='evolution-terminal-diagnostics-v1',optimizer_updates=0,device='CPU FP32 two threads',wall_s=420,
        indices=indices,empty_indices=p['ar']['indices']['empty_dev'],sample_id=0,weight_sha256={k:sha(ROOT/v) for k,v in weights.items()},
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),**p['sources']},
        ar='Cross trained-old/new weights with additive/bilinear inference; teacher within-true-cell head and displaced encoded-frame score. True histories/cells are diagnostic only.',
        ssd='Four trained-weight/carry combinations, plus count logits held to full model for member-only and all-audio-zero controls. No hidden count is read.',
        primary_unchanged=True,cloud_compute_calls=0)
    write(PLAN,plan);result=dict(status='running',plan_sha256=sha(PLAN),ar={},ssd={},optimizer_updates=0,cloud_compute_calls=0);write(REPORT,result)
    start=time.monotonic();deadline=start+plan['wall_s']
    ar_models={}
    for trained in ('old','new'):
        state=torch.load(ROOT/weights['ar_'+trained],map_location='cpu',weights_only=False)['model']
        for interaction,cls in [('off',GroupedChartProcess),('on',AudioInteractionGroupedProcess)]:
            model=cls(FullEventConfig(**p['ar']['model']),density_reference=p['ar']['density_reference'],history_dropout=0).eval();model.load_state_dict(state)
            ar_models[f'{trained}_{interaction}']=model
    raw={name:[] for name in ar_models};teachers={name:[] for name in ar_models}
    for index in indices:
        e=explicit_row(row(index,'draft'));inp,_=probability_batch([e],'cpu')
        for name,model in ar_models.items():
            if time.monotonic()>deadline:raise TimeoutError('Diagnostic wall')
            events,info=generate_grouped_process(model,inp,e['request'],seed=p['ar']['seed']+17*index,deadline=deadline)
            raw[name].append(record(e,dict(complete=True,events=[asdict(v) for v in events],diagnostics=info),panel,p['ar']['shifts_us']))
            teachers[name].extend(ar_teacher(model,inp,e))
    for name,rows in raw.items():
        write(ART/f'ar-{name}.json',rows)
        result['ar'][name]=dict(free=metrics(rows),teacher_with_true_history_cell=means(teachers[name]),head_groups=len(teachers[name]))
    write(REPORT,result);del ar_models
    models={}
    for trained in ('full','reset'):
        state=torch.load(ROOT/weights['ssd_'+trained],map_location='cpu',weights_only=False)['model']
        for carry in ('on','off'):
            model=MambaCanvas(FullEventConfig(**p['ssd']['model']),density_reference=p['ssd']['density_reference'])
            if carry=='off':reset_mamba_boundaries(model)
            model.load_state_dict(state);models[f'{trained}_{carry}']=model.eval()
    raw={name:[] for name in [*models,'reset_off_full_count','full_off_full_count','full_on_no_audio','reset_off_no_audio']};teacher={name:[] for name in models}
    empty={name:[] for name in models}
    for index in indices+plan['empty_indices']:
        e=row(index,'draft');inp,target=relative_batch([e],'cpu');outputs={name:model(inp) for name,model in models.items()}
        for name,out in outputs.items():
            loss,parts=canvas_loss(out,target,inp);teacher[name].append(dict(loss=float(loss),**{k:float(v) for k,v in parts.items()}))
            if index in plan['empty_indices']:
                prob=out['count'].softmax(-1);expected=float(((prob*torch.arange(17)).sum(-1)*inp['canvas_valid'][...,None]).sum())
                empty[name].append(dict(index=index,expected_events=expected));continue
            pred=generate_relative(FrozenOutput(out),inp,e['request'],deadline=deadline)
            raw[name].append(record(e,pred,panel,p['ssd']['shifts_us']))
        if index in plan['empty_indices']:continue
        for name in ('reset_off','full_off'):
            out={**outputs[name],'count':outputs['full_on']['count']}
            pred=generate_relative(FrozenOutput(out),inp,e['request'],deadline=deadline)
            raw[name+'_full_count'].append(record(e,pred,panel,p['ssd']['shifts_us']))
        zero={**inp,**{k:torch.zeros_like(inp[k]) for k in ('local','song','mel')}}
        for name in ('full_on','reset_off'):
            pred=generate_relative(models[name],zero,e['request'],deadline=deadline)
            raw[name+'_no_audio'].append(record(e,pred,panel,p['ssd']['shifts_us']))
        if time.monotonic()>deadline:raise TimeoutError('Diagnostic wall')
    for name,rows in raw.items():
        write(ART/f'ssd-{name}.json',rows);result['ssd'][name]=dict(free=metrics(rows))
        if name in teacher:result['ssd'][name].update(teacher_fields=means(teacher[name]),empty=empty[name])
    result.update(status='completed',wall_s=time.monotonic()-start,limits='16firstwindows/16reusedcomponents; AR teacher cells/history leak labels and are diagnostic only. Frozen all-audio-zero is OOD. No inference variant replaces original terminal. One seed, not family or product proof.')
    write(REPORT,result);print(json.dumps(dict(status='completed',wall_s=result['wall_s']),ensure_ascii=False))


if __name__=='__main__':main()
