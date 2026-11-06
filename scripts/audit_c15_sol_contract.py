"""Small mathematical and trace audits of the proposal; no neural execution."""
import ast
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'reports/council-mechanisms-c15/parent-S-contract-audit.json'


def read(name):
    return json.loads((ROOT/name).read_text())


def digest(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def lse(values):
    values=[v for v in values if v!=-math.inf]
    if not values:
        return -math.inf
    m=max(values)
    return m+math.log(sum(math.exp(v-m) for v in values))


def count_dp(scores, cap):
    table=[0.]+[-math.inf]*cap
    modes=[(0.,())]+[(-math.inf,())]*cap
    for kind in scores:
        table=[lse([table[t-n]+kind[n] for n in range(t+1)]) for t in range(cap+1)]
        options=[[(modes[t-n][0]+kind[n],modes[t-n][1]+(n,)) for n in range(t+1)
                  if modes[t-n][0]!=-math.inf and kind[n]!=-math.inf] for t in range(cap+1)]
        modes=[min(candidates,key=lambda v:(-v[0],v[1])) if candidates else (-math.inf,()) for candidates in options]
    return lse(table),min(modes,key=lambda v:(-v[0],v[1]))


@dataclass(frozen=True)
class Carry:
    onset_us:int
    end_us:int
    type:str
    lane:int=-1


@dataclass(frozen=True)
class Request:
    start_us:int
    known_carry:tuple=()


def main():
    if DEST.exists():
        raise FileExistsError('Preserve independent audit identity')
    rng=random.Random(20261030)
    max_error=0.; mode_errors=[]
    for case in range(60):
        cap=case%6+1
        scores=[[rng.uniform(-4,3) for _ in range(cap+1)] for _ in range(4)]
        if case%3==0:
            scores[case%4][1:]=[-math.inf]*cap
        vectors=[(sum(scores[k][n[k]] for k in range(4)),n)
                 for n in product(range(cap+1),repeat=4) if sum(n)<=cap]
        vectors=[v for v in vectors if v[0]!=-math.inf]
        z,mode=count_dp(scores,cap)
        max_error=max(max_error,abs(z-lse([v[0] for v in vectors])))
        brute=min(vectors,key=lambda v:(-v[0],v[1]))
        if mode[1]!=brute[1]:mode_errors.append(case)
    assert max_error<1e-12 and not mode_errors
    # This is a mathematical counterexample, not a fit to current probabilities.
    scores=[[math.log(.7),math.log(.3)]+[-math.inf]*3 for _ in range(4)]
    z,mode=count_dp(scores,4)
    assert mode[1]==(0,0,0,0) and abs(z)<1e-12
    old=read('artifacts/local-evolution-v1/ssd_full/quality-1000-draft.json')
    new=read('artifacts/c13-native-quality-v1/draft.json')
    before={r['index']:r for r in old}; differences=[]
    source='beatfcn/train/canonical_probability.py'
    tree=ast.parse((ROOT/source).read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='explicit_row')
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    namespace=dict(replace=replace,annotated_sequence=lambda *args:dict(stub='No token generation;request metadata test only'))
    exec(compile(module,str(ROOT/source),'exec'),namespace)
    for r in new:
        previous=before[r['index']]
        assert previous['target']==r['target']
        changed=[k for k in set(previous['request'])|set(r['request']) if previous['request'].get(k)!=r['request'].get(k)]
        if changed:
            assert changed==['known_carry']
            request=Request(previous['request']['start_us'],tuple(Carry(**e) for e in previous['request']['known_carry']))
            visible=[Carry(**e) for e in r['visible']]
            replay=namespace['explicit_row'](dict(request=request,visible=visible))['request']
            assert not replay.known_carry
            differences.append(dict(index=r['index'],old_carry=previous['request']['known_carry'],new_carry=r['request']['known_carry'],replayed_current_function_drops_carry=True))
    assert len(differences)==8
    # Compare both explicit positive/negative scoring denominators, never edit C14 facts.
    allowed={}; all_negatives={}
    for arm in ('B','Z','T'):
        allowed[arm]={}; all_negatives[arm]={}; items=[]
        for record in new:
            name=f'diagnostic-{arm}-TT-draft-{record["index"]}.json'
            p=ROOT/'artifacts/c14-type-diagnostic-completion-v1'/name
            if not p.exists():p=ROOT/'artifacts/c14-type-song-prefix-v1'/name
            items.extend(v for v in json.loads(p.read_text())['items'] if not v['forced'])
        for k,name in enumerate(('tap','hold','flick','drag')):
            positive=[v for v in items if v['true_type']==k]
            negative=[v for v in items if v['true_type']!=k]
            legal_negative=[v for v in negative if v['kind_allowed'][k]]
            def average(rows):
                values=[-math.log(1-v['type_probability'][k]) for v in rows]
                return sum(values)/len(values) if values else None
            allowed[arm][name]=dict(positive=len(positive),negative=len(legal_negative),model_negative_nll=average(legal_negative))
            all_negatives[arm][name]=dict(positive=len(positive),negative=len(negative),mask_forbidden_negative=len(negative)-len(legal_negative),model_negative_nll=average(negative))
    output=dict(status='completed',model_calls=0,optimizer_updates=0,cloud_calls=0,new_predictions=0,
        identities={p:digest(p) for p in [source,'reports/council-mechanisms-c15/sol.md','reports/local-integration-v1/c14-type-effective-facts.json','artifacts/local-evolution-v1/ssd_full/quality-1000-draft.json','artifacts/c13-native-quality-v1/draft.json']},
        probability_contract=dict(small_standard_library_cases=60,max_logZ_error=max_error,mode_errors=mode_errors,
            zero_mode_counterexample=dict(description='Four independent Bernoulli(.3) counts;cap4neverbinds',mode=list(mode[1]),expected_total=1.2,mode_probability=.7**4),
            boundary='No neural/torch/MPS/autograd implementation or actual performance tested. This is only the proposed DP mathematics.',
            lesson='If independent per-type potential maxima satisfy total<=64,the capacity constraint cannot alter their mode. Typed DP alone does not solve all-zero modes;new candidate needs proper/free/empty evidence.'),
        carry_metadata=dict(changed_requests=differences,unchanged_requests=24,source_function='explicit_row replaces known_carry with visible-crossing holds',
            source_replay='AST function replay with dataclass metadata and sequence stub;no project import/tensor/preprocessing',
            boundary='Request metadata differs;actual network information differs only if a consumer uses carry. Existing serialized request equality is not input tensor equality. No old-input tensors reconstructed.',
            proposed_requirement='Preserve authorized incoming carry, expose intended conditioning explicitly, and freeze equal public request metadata across matched candidates. Do not attribute old quality gap to carry or SSD.'),
        proper_denominators=dict(legal_only_negative=allowed,all_nonforced_negative=all_negatives,
            lesson='Both arithmetic denominators are explicit, but forbidden-class zero losses cannot be credited as learned rejection. Use legal-only class-negative strata for learning claims. Drag comparison is unchanged.',
            original_C14_facts_unchanged=True),
        disposition='Proposal mathematically inspectable,not implementation-ready or adopted training;full C15 decision follows independent Grok result. No new candidate or optimizer dispatched.')
    DEST.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'],DP_cases=60,carry_metadata_differences=len(differences),new_training=0)))


if __name__=='__main__':
    main()
