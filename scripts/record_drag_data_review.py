"""Close the actual data pair with identity, mechanisms and live budget state."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/local-integration-v1'
ID='data-drag-density-probe-v1'
ART=ROOT/'artifacts'/ID


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


def main():
    report=read(BASE/f'{ID}.json');plan=read(ROOT/'bench/rhythm-v2'/f'{ID}.json')
    summary=read(BASE/'data-drag-density-probe-summary-v1.json');post=read(BASE/'data-drag-density-posthoc-chain-v1.json')
    if report['status']!='completed_pending_data_mechanism_review' or report['optimizer_updates']!=356 or post['optimizer_updates']:
        raise ValueError('Unexpected actual execution')
    for name,h in plan['source_sha256'].items():
        if sha(ROOT/name)!=h or sha(ART/'source'/name)!=h:raise ValueError('Frozen source drift '+name)
    paths=[ROOT/'bench/rhythm-v2'/f'{ID}.json',BASE/f'{ID}.json',BASE/'data-drag-density-audit-v1.json',
        BASE/'data-drag-density-probe-summary-v1.json',BASE/'data-drag-density-posthoc-chain-v1.json',
        BASE/'data-drag-density-review-v1.md',ART/'pipeline-preflight.json',ART/'numerical-preflight.json',
        ART/'optimizer-counters.json',ART/'posthoc-chain-controls-plan.json',
        ROOT/'artifacts/data-drag-density-v1/removed-event-indices.json',
        ROOT/'bench/rhythm-v2/high-density-drag-policy-v1.json']
    narrative=BASE/'data-drag-density-review-v1.md';receipt=BASE/'data-drag-density-mechanism-review-v1.json'
    if receipt.exists():raise FileExistsError('Preserve terminal analysis')
    ev=lambda n:'reports/local-integration-v1/'+n
    row=dict(candidate=ID,analysis_status='complete',identity={str(p.relative_to(ROOT)):sha(p) for p in paths},
        layers=dict(health='Actual640261parameterCPU/MPSloss/gradientparity passed;bothdiscarded50fitratio.454/.474;global356updates,quality128each',
            quality='Both34free+34allmask+34machineproper complete;same1389devtargets;full50.18139→.17542,onset10.19296→.22262;non-tapTP97→82',
            structure_conditions='Data pipeline removes onlydrag963/20002 across32/416;devremove0;firstNLL1.25102→1.25445vsprior1.17498;freeempty1each;holdtail117→212.5ms;zero-update positive/audio controls completed'),
        mechanism_hypotheses=[
            dict(id='dense_drag_sufficient_cause',observation='Allaffected32trainingwindowsincluded;drag5964→4038 exposures,otherclassesidentical;first/non-tap/tailnotimproved',
                mechanism='If dense drag alone drove failures,removing it should improve fixed dev conditionaltypes/hold and deployment',
                prediction='Typeprior gap and hold/free chains jointly improve under matched data-only intervention',
                evidence=[ev('data-drag-density-audit-v1.json'),ev('data-drag-density-probe-summary-v1.json')],
                judgment='Not supported in this one-seed small budget;data preference retained;not family-wide proof'),
            dict(id='prior_redistribution',observation='Onlytap positiveNLLimproves.09285;hold/flick/dragworsen;tapnegativeNLLworsens;microgain.02028',
                mechanism='Changed class mass favors common taps rather than stronger minority type information',
                prediction='Overall type score can improve while firstmember and minority positive scores remain weak',
                evidence=[ev('data-drag-density-probe-summary-v1.json')],judgment='Consistent quantitative decomposition;gradient/slot/capacity causes not uniquely isolated'),
            dict(id='event_context_weak_use',observation='Trueoddcontext gains typeNLL only.00294/.00029;machinecontext similar;all supports finite',
                mechanism='Current learned event conditioning provides little extra information even when context is correct',
                prediction='Targeted meaningful-versus-mismatched context learning must move type/end scores before free benefit claim',
                evidence=[ev('data-drag-density-posthoc-chain-v1.json'),ev('data-drag-density-probe-summary-v1.json')],
                judgment='Limited frozen-state positive control supports weak use;not proof context intrinsically uninformative'),
            dict(id='new_readout_not_global_empty_mode',observation='MarginalNULLmode1458/1472active,oneemptyeach;expectations1849/1967;actualfree1792/1412',
                mechanism='Current failures cannot be explained solely by old C15 all-empty hard-count mode',
                prediction='Pure count reparameterization need not improve first types or hold tails;canonical wrong-slot expectation may persist',
                evidence=[ev('data-drag-density-probe-summary-v1.json')],judgment='Supported for this law;slot margin is not physical type×cell or chartMAP'),
            dict(id='duration_prior_dependence',observation='TrueheadtailNLLgainvsprioronly.00371/.00343;freeholdTP12→4,tailmedian117→212.5ms',
                mechanism='Endpoint residual has little learned information;counts cannot supply missing duration/context signal',
                prediction='Direct endpoint/context learning required;more occurrence alone does not fix completeholds',
                evidence=[ev('data-drag-density-probe-summary-v1.json'),ev('data-drag-density-posthoc-chain-v1.json')],
                judgment='Present weakness observed;duration information versus optimization and budget remains open')],
        controls=['Same initial tensor/optimizer/audio/stream/mask/law;onlydense-drag data intervention',
            'Same original and filtered dev cross-scoring;0dev removals;no prediction filter',
            'Actual raw/filtered firstmember,perclass positive/negative proper,common/own training priors',
            'Sharedrawmachinecontext and posthoc trueodd positive control;all-audiozero OOD negative;0updates',
            'Native/COPY/duplicate/full-chart boundaries and original partition checks;empty windows;checkpoint Adam128/115 counters'],
        missing_controls=[dict(question='Is weak type learning due to context use or missing intent?',
                comparison='One paired training that changes event/type context learning directly,meaningful versus mismatched context with same occurrence/time path',
                decision_if='If true-context/typeproper improves,connect to free deployment;otherwise change type information/intent supervision,not countlaw'),
            dict(question='How much effect is optimization/slot reassignment versus data semantics?',
                comparison='Matched fixed event anchors and frozen occurrence/time representation for type/context intervention;multiple seed/new-song confirmation only after meaningful gain',
                decision_if='No gain rejects this mechanism under budget;not enough to reject data family or justify widening'),
            dict(question='Can neural duration exceed trainingprior on generatedheads?',
                comparison='Fixed support and matchedhead context/endpoint supervision with true and selfcontexts,separate proper and free matched-tail measures',
                decision_if='Proper-only gain routes to exposure/transfer;no proper gain routes to input/supervision')],
        lessons=['Dense drag is a reproducible Owner data preference,not established dirty labels',
            'Filtering before random partition is a confound;preserve original public partition and non-drag targets',
            'Counts near truth/onset gain do not substitute minority-positive proper,firstmember or holdjoint quality',
            'Truecontext positive control and firstmember audio effect prevent hiding weak mark learning behind pooled acoustic gain'],
        successor_contract=dict(required_changes=['Retain Owner filtered data view and full native support;cloud0',
                'Next complete candidate directly targets eventcontext,typelearning or deployment gap;default firstmember/event-conditioned type+endpoint information',
                'Keep occurrence/count path fixed as control where possible;registered meaningful/mismatched context and perclass proper/free/empty/tail outcomes'],
            forbidden_repeats=['No further equivalent count-only factorization or execution-cost detour',
                'No unchanged longer/wider trial or selected seed/checkpoint/threshold rescue',
                'No declaring density cleanup a type/hold success from totalcount,onset or pooledNLL alone']),
        closure_reason='The requested data rule and full pipeline intervention are complete. A matched640k128update pair removed dense drag without deleting charts or changing dev labels. It improved onset localization and count magnitude but firstmember type still lost its prior,three minority positive scores worsened,non-tapTPfell97to82,and holdtailmedian worsened117to212.5ms. Frozen positive-context controls show little use;this is not a count-only collapse. Close promotion of this data-only recipe,retain the Owner filter and limited findings,and route the next training directly to event/type/endpoint information with explicit causal controls. One seed/small budget does not reject the method family.',
        source_record=str(narrative.relative_to(ROOT)))
    write(receipt,row)
    p=ROOT/'reports/local-evolution-v1/mechanism-reviews/index.json';registry=read(p)
    if any(v['id']==ID for v in registry['candidates']):raise ValueError('Duplicate terminal')
    registry['candidates'].append(dict(id=ID,analysis_status='complete',review=str(receipt.relative_to(ROOT)),review_sha256=sha(receipt),
        markdown=str(narrative.relative_to(ROOT)),narrative_sha256=sha(narrative)));write(p,registry)
    artifact_bytes=sum(p.stat().st_size for p in ART.rglob('*') if p.is_file())
    peak=max(v['runtime']['peak_sampled_physical_bytes'] for v in report['stages'].values())
    p=ROOT/'reports/rhythm_v2_budget.json';budget=read(p);entry=budget['local_cpu_diagnostics'][ID]
    entry.update(status='completed_data_hypothesis_not_supported_mechanism_closed',optimizer_updates=356,health_updates=100,quality_updates=dict(raw=128,filtered=128),
        main_wall_s=report['wall_s'],posthoc_zero_update_wall_s=post['wall_s'],free_completed=68,conditional_completed=136,posthoc_conditionals=128,
        peak_physical_bytes=peak,artifact_bytes=artifact_bytes,review=str(narrative.relative_to(ROOT)),cloud_spend_usd=0)
    write(p,budget)
    p=BASE/'state.json';state=read(p);state['current_experiment'].update(status='data_only_not_promoted_mechanism_closed',health_updates=dict(raw=50,filtered=50),
        quality_updates=dict(raw=128,filtered=128),global_optimizer_updates=356,free_completed=68,conditional_completed=136,
        posthoc_zero_update_conditionals=128,review=str(narrative.relative_to(ROOT)),data_rule_retained=True,research_supported=False)
    state.update(status='data_pair_complete_next_direct_event_type_context',training_process_running=False,diagnostic_process_running=False,
        report=str((BASE/f'{ID}.json').relative_to(ROOT)),plan=str((ROOT/'bench/rhythm-v2'/f'{ID}.json').relative_to(ROOT)),
        terminal_receipt=str(receipt.relative_to(ROOT)),new_neural_updates=356,health_updates=100,quality_updates=256,
        next_entry='Direct event/type/context Council after completed data review;no count-only successor;cloud0',last_update_at=datetime.now(timezone.utc).isoformat())
    write(p,state)
    print(dict(status='complete',updates=356,main_wall_s=report['wall_s'],posthoc_wall_s=post['wall_s'],peak_physical_bytes=peak,artifact_bytes=artifact_bytes,cloud=0))


if __name__=='__main__':main()
