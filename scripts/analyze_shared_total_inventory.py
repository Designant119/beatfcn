"""Terminal evidence extraction, including exact optimizer counters; no inference."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.shared_total_inventory import SharedTotalInventory as TypedInventory
from scripts.run_local_portfolio_scale import sha, write


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/c16-shared-total-inventory-v1'
DEST=ROOT/'reports/local-integration-v1/c16-shared-total-inventory-terminal-facts.json'


def main():
    if DEST.exists():raise FileExistsError('Preserve terminal arithmetic evidence')
    top=json.loads((ROOT/'reports/local-integration-v1/c16-shared-total-inventory-v1.json').read_text())
    if top['status'].endswith('_running'):raise ValueError('Not a terminal experiment')
    plan=json.loads((ROOT/'bench/rhythm-v2/c16-shared-total-inventory-v1.json').read_text())
    expected=json.loads((ROOT/plan['expected_free_metadata']).read_text())
    facts=dict(id='c16-shared-total-inventory-terminal-facts',original_status=top['status'],
        optimizer_updates=top['optimizer_updates'],wall_s=top['wall_s'],cloud_calls=0,
        model_forwards=0,new_predictions=0,new_optimizer_updates=0,
        parameters=plan['parameters'],stages={},free={},identity={},
        scope='Posthoc arithmetic/exact checkpoint counters;does not decide mechanism closure')
    paths=['bench/rhythm-v2/c16-shared-total-inventory-v1.json',
        'reports/local-integration-v1/c16-shared-total-inventory-v1.json',
        'artifacts/c16-shared-total-inventory-v1/admission.json']
    for path in paths:facts['identity'][path]=sha(ROOT/path)
    names=None
    for stage in ('admission','health-A','health-C','quality-A','quality-C','evaluation'):
        path=ART/f'{stage}.json'
        if not path.exists():
            facts['stages'][stage]=dict(status='not_attempted');continue
        value=json.loads(path.read_text());facts['identity'][str(path.relative_to(ROOT))]=sha(path)
        item={k:value[k] for k in ('status','updates','wall_s','checks','before','after','failure','cpu_mps','prefix64','initial_digest','exposure') if k in value}
        if 'telemetry' in value:item['resource']=dict(peak_bytes=value['telemetry']['peak_sampled_physical_bytes'],samples=len(value['telemetry']['samples']),error=value['telemetry']['error'])
        item['worker_optimizer_updates']=value.get('optimizer_updates')
        if value.get('optimizer_updates')!=value.get('updates'):raise ValueError('Applied counter mismatch')
        item['applied_updates']=value.get('updates',0)
        if value.get('checkpoint'):
            checkpoint=ROOT/value['checkpoint']
            if sha(checkpoint)!=value['checkpoint_sha256']:raise ValueError('Checkpoint identity')
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if state['updates']!=400 or state['plan_sha256']!=sha(ROOT/paths[0]):raise ValueError('Actual quality state metadata')
            if names is None:
                torch.set_num_threads(4)
                model=TypedInventory(FullEventConfig(**plan['model']),density_reference=plan['density_reference'])
                names=[n for n,_ in model.named_parameters()];del model
            ids=state['optimizer']['param_groups'][0]['params']
            if len(ids)!=len(names):raise ValueError('Optimizer parameter-order identity')
            counts={name:int(state['optimizer']['state'][pid]['step'].item()) for name,pid in zip(names,ids) if pid in state['optimizer']['state']}
            if counts.get('total_count.weight')!=400 or counts.get('total_count.bias')!=400 or counts.get('typed_count.weight')!=400 or counts.get('typed_count.bias')!=400 or max(counts.values())!=400:
                raise ValueError('Real count-head Adam steps do not confirm400')
            item['optimizer_steps']=dict(total_head_weight=counts['total_count.weight'],total_head_bias=counts['total_count.bias'],parameter_steps=counts,count_head_weight=counts['typed_count.weight'],count_head_bias=counts['typed_count.bias'],histogram=dict(Counter(counts.values())),
                note='Conditional member/endpoint parameters may have fewer Adam steps on batches without that branch;global update count uses always-present count head.')
            facts['identity'][value['checkpoint']]=value['checkpoint_sha256'];del state
        facts['stages'][stage]=item
    for arm in ('A','C'):
        facts['free'][arm]={}
        for task,planned in expected.items():
            path=ART/f'{arm}-{task}.json'
            records=json.loads(path.read_text()) if path.exists() else planned
            if [(r['index'],r['target']) for r in records]!=[(r['index'],r['target']) for r in planned]:raise ValueError('Free target/denominator drift')
            types={k:dict(tp=0,pred=0,true=0,matched_components=set()) for k in ('tap','hold','flick','drag')}
            completed=attempted=typed_slots=total_slots=pred=truth=duplicate_extra=0
            per_component={};probability_files=0
            for r in records:
                attempted+=r['status']!='not_attempted';completed+=bool(r['prediction']['complete'])
                events=r['prediction']['events'];pred+=len(events);truth+=len(r['target'])
                duplicate_extra+=len(events)-len({(e['onset_us'],e['end_us'],e['type'],e.get('lane',-1)) for e in events})
                per_component.setdefault(r['component'],[]).append(dict(full50=r['scores']['full_note_f1'],onset10=r['scores']['onset_10ms']['f1'] or 0))
                for kind,values in types.items():
                    for k in ('tp','pred','true'):values[k]+=r['scores']['per_type'][kind][k]
                    if r['scores']['per_type'][kind]['tp']>0:values['matched_components'].add(r['component'])
                if r['prediction'].get('count_decision') is not None:
                    chosen=np.array(r['prediction']['count_decision'],np.int64);target=np.zeros_like(chosen)
                    for e in r['target']:target[(e['onset_us']-r['request']['start_us'])//40000,('tap','hold','flick','drag').index(e['type'])]+=1
                    typed_slots+=int(np.minimum(chosen,target).sum());total_slots+=int(np.minimum(chosen.sum(-1),target.sum(-1)).sum())
                if r.get('probability_path'):
                    p=ROOT/r['probability_path']
                    if sha(p)!=r['probability_sha256']:raise ValueError('Probability file identity')
                    with np.load(p,allow_pickle=False) as arrays:
                        if arrays['scores'].shape!=(len(r['count_target']),4,65) or not np.isfinite(arrays['scores']).all() or not np.isfinite(arrays['total_scores']).all():raise ValueError('Probability array identity/finite')
                    probability_files+=1
            for values in types.values():
                values['matched_components']=len(values['matched_components']);values['f1']=2*values['tp']/max(values['pred']+values['true'],1)
            facts['free'][arm][task]=dict(expected=len(planned),attempted=attempted,complete=completed,pred=pred,true=truth,types=types,
                full50_component_macro=float(np.mean([np.mean([v['full50'] for v in rows]) for rows in per_component.values()])),
                onset10_component_macro=float(np.mean([np.mean([v['onset10'] for v in rows]) for rows in per_component.values()])),
                mode_typed_slot_coverage=typed_slots,mode_total_slot_coverage=total_slots,duplicate_extras=duplicate_extra,probability_files_checked=probability_files,
                scope='Exact-cell typed/total supply only;not an upper bound on 50ms tolerant matches. All failures retain denominators.')
            if path.exists():facts['identity'][str(path.relative_to(ROOT))]=sha(path)
    if (ART/'evaluation.json').exists():
        facts['research']=json.loads((ART/'evaluation.json').read_text()).get('research')
    facts['artifact_bytes']=sum(p.stat().st_size for p in ART.rglob('*') if p.is_file())
    facts['peak_sampled_physical_bytes']=max(s.get('resource',{}).get('peak_bytes',0) for s in facts['stages'].values())
    facts['identity_checks']=dict(source=0,runtime=0,input_payloads=0)
    for key,label in [('source_sha256','source'),('runtime_records','runtime'),('input_payload_sha256','input_payloads')]:
        for name,expected in plan[key].items():
            if sha(ROOT/name)!=expected:raise ValueError('Post-run identity drift '+name)
            facts['identity_checks'][label]+=1
    write(DEST,facts)
    print(json.dumps(dict(status='extracted',optimizer_updates=facts['optimizer_updates'],original_status=facts['original_status'],draft={a:facts['free'][a]['draft'] for a in ('A','C')}),ensure_ascii=False))


if __name__=='__main__':main()
