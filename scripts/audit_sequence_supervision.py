"""D36 frozen-label/decision audit; contains no neural model or output repair."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from beatfcn.model.event_canvas import ClockLayout, RhythmEvent
from beatfcn.model.event_sequence import Decision, EventSequenceState, SequenceSpec, encode_sequence

ROOT=Path(__file__).resolve().parents[1]


def distribution(counts):
    n=sum(counts.values())
    entropy=-sum(v/n*math.log(v/n) for v in counts.values()) if n else 0.
    return dict(counts=dict(counts),n=n,majority_accuracy=max(counts.values())/n if n else None,
                constant_distribution_nll=entropy)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    args=parser.parse_args();plan_path=ROOT/args.plan;plan=json.loads(plan_path.read_text())
    for name,h in plan['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('changed source: '+name)
    path=ROOT/plan['result']
    if path.exists():raise FileExistsError('preserve previous audit')
    documents={arm:json.loads((ROOT/name).read_text()) for arm,name in plan['reports'].items()}
    fields=defaultdict(Counter);by_members=defaultdict(Counter);target_rows=[]
    for core in documents['A']['generation']['original']['cores']:
        mode=core['identity']['mode'];spec=SequenceSpec(ClockLayout(8_000_000,16_000_000),mode,32_000_000)
        events=[RhythmEvent(**e) for e in core['target_events']]
        tokens=encode_sequence(events,spec);state=EventSequenceState(spec)
        counters=defaultdict(Counter);member_rule_hits=0;member_total=0
        for token in tokens:
            fields[token.field][str(token.value)]+=1;counters[token.field][str(token.value)]+=1
            if token.field=='member':
                by_members[str(state.members)][str(token.value)]+=1
                member_rule_hits+=int(token.value==int(state.members==0));member_total+=1
            state.apply(token)
        target_rows.append(dict(identity=core['identity'],fields={k:distribution(v) for k,v in counters.items()},
                                members_zero_predict_new_else_end_accuracy=member_rule_hits/member_total,
                                diagnostic_rule_not_used_by_generator=True))
    arms={}
    for arm,doc in documents.items():
        generated=[]
        for core in doc['generation']['original']['cores']:
            mode=core['identity']['mode'];spec=SequenceSpec(ClockLayout(8_000_000,16_000_000),mode,32_000_000)
            rounds=[]
            for raw in core['rounds']:
                state=EventSequenceState(spec);ends=[]
                for d in raw['decisions']:
                    token=Decision(**d)
                    if token.field=='group' and token.value==0:
                        ends.append(dict(at_last_onset_us=state.last_onset_us,
                                         domain_forced=state.allowed_values()==(0,)))
                    state.apply(token)
                if [asdict(e) for e in state.events]!=raw['events']:raise ValueError('raw event replay mismatch')
                sizes=Counter(e.onset_us for e in state.events)
                rounds.append(dict(round=raw['round'],events=len(state.events),groups=len(sizes),
                                   group_sizes=distribution(Counter(map(str,sizes.values()))),
                                   maximum_group_size=max(sizes.values(),default=0),ends=ends,complete=state.done))
            generated.append(dict(identity=core['identity'],rounds=rounds))
        heads={}
        for field in fields:
            rows=[c for c in doc['final']['cores'] if field in c['accuracy']]
            constant=[c['fields'][field]['majority_accuracy'] for c in target_rows if field in c['fields']]
            entropy=[c['fields'][field]['constant_distribution_nll'] for c in target_rows if field in c['fields']]
            heads[field]=dict(actual_mean_accuracy=sum(r['accuracy'][field] for r in rows)/len(rows),
                              mean_per_core_majority_accuracy=sum(constant)/len(constant),
                              actual_mean_nll=sum(r['terms'][field] for r in rows)/len(rows),
                              mean_per_core_constant_nll=sum(entropy)/len(entropy))
        all_ends=[end for c in generated for r in c['rounds'] for end in r['ends']]
        arms[arm]=dict(heads=heads,generated=generated,total_generated_ends=len(all_ends),
                       domain_forced_ends=sum(e['domain_forced'] for e in all_ends))
    output=dict(id='D36',complete=True,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                targets={k:distribution(v) for k,v in fields.items()},member_targets_by_completed_members={k:distribution(v) for k,v in by_members.items()},
                target_cores=target_rows,arms=arms,optimizer_updates=0,new_paid_compute_usd=0,
                no_generation_repair=True,model_promotion=False,
                limitations=['A constant-class reference matching aggregate accuracy does not prove identical predictions.',
                             'The count rule is a training-label learnability diagnostic, not a human quality metric.',
                             'No end-class logits/confusion matrix were saved; per-class teacher performance remains unmeasured except kind.'])
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(targets=output['targets'],member_targets_by_completed_members=output['member_targets_by_completed_members'],
                         arms={a:dict(heads=r['heads'],total_generated_ends=r['total_generated_ends'],domain_forced_ends=r['domain_forced_ends']) for a,r in arms.items()}),ensure_ascii=False))


if __name__=='__main__':main()
