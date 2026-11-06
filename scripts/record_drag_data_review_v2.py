"""Record strict1s terminal separately from v1 and the new120M candidate."""
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts.run_drag_data_probe import sha,read,write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/local-integration-v1'
ID='data-drag-density-probe-v2';ART=ROOT/'artifacts'/ID


def main():
    report=read(BASE/f'{ID}.json');plan=read(ROOT/'bench/rhythm-v2'/f'{ID}.json')
    if report['optimizer_updates']!=356 or report['status']!='completed_pending_data_mechanism_review':raise ValueError('Actual run incomplete')
    for name,h in plan['source_sha256'].items():
        if sha(ROOT/name)!=h or sha(ART/'source'/name)!=h:raise ValueError('Frozen source drift')
    narrative=BASE/'data-drag-density-review-v2.md';path=BASE/'data-drag-density-mechanism-review-v2.json'
    if path.exists():raise FileExistsError('Preserve strict1s mechanism receipt')
    paths=[ROOT/'bench/rhythm-v2'/f'{ID}.json',BASE/f'{ID}.json',BASE/'data-drag-density-audit-v2.json',
        BASE/'data-drag-density-probe-summary-v2.json',narrative,ART/'pipeline-preflight.json',ART/'numerical-preflight.json',
        ART/'optimizer-counters.json',ART/'masked-exposure-audit.json',ROOT/'bench/rhythm-v2/high-density-drag-policy-v2.json',
        ROOT/'artifacts/data-drag-density-v2/removed-event-indices.json']
    ev=['reports/local-integration-v1/data-drag-density-probe-summary-v2.json']
    r=dict(candidate=ID,analysis_status='complete',identity={str(p.relative_to(ROOT)):sha(p) for p in paths},
        layers=dict(health='Strict12/13/alltype/halfopen/duplicates/bruteforce5tests+163realinput/supportchecks;CPU/MPS;bothdiscarded50fit;fresh128each;356updates',
            quality='68free/340conditionalcomplete;same1304primarytargets full50.17443→.16037/onset10.19325→.22858;non-tapTP77→41;raw1389secondaryalsofalls',
            structure_conditions='Rule removes4024drag/133of416;dev85drag/5of32;otherclassesunchanged;firstNLL1.30159→1.32232vsprior1.22533;truecontextgain.00283/.00382;tail117→191ms'),
        mechanism_hypotheses=[
            dict(id='stronger_density_rule_not_sufficient',observation='Data and selecteddrag exposurechange strongly;first/non-tap/tail not jointly improved',
                mechanism='Dense-object supervision contributes activity/prior shifts but is not sufficient explanation of all three chains',
                prediction='Changing only threshold can reduce count without correcting type and duration or firstmember',evidence=ev,
                judgment='Supported within single-seed640k128update pair;retain Owner rule,not a family rejection'),
            dict(id='prior_and_slot_redistribution',observation='Train drag37.27%→12.73%;other positiveclassNLLimproves but drag worsens.48516;selectedholds1416→1428',
                mechanism='Frequency shift and changed canonical slot assignment alter both activity and supervised exposure even with identical mask RNG',
                prediction='Same-target and actual selected-type exposure comparisons differ from naive own-cohort gains',
                evidence=ev+['artifacts/data-drag-density-probe-v2/masked-exposure-audit.json'],judgment='Observed;does not uniquely isolate gradientcompetition or capacity'),
            dict(id='context_and_endpoint_weak_information',observation='Truthcontexttypesgain<.004;tailmodel beatsdurationpriorby<.004;freeholdTP12→6',
                mechanism='Current event information/endpoint residual is weak independently of sorting support failures',
                prediction='Next meaningful/incorrect eventcontext and ownstate training must show direct field information before broader claims',evidence=ev,
                judgment='Limited positive control;not evidence that context can never help'),
            dict(id='deployment_recall_and_relabeling',observation='MarginalNULLactive1458→834 and trueunderhalf467→654;free1792→859;newtargets1304 versusold1389',
                mechanism='More conservative occurrence and shifted mark posterior reduce full-event recall;labeldeletion also changes reportedscore',
                prediction='Both arms evaluated on identical original and filtered cohorts reveal training loss separately from score-only relabeling',evidence=ev,
                judgment='Explicit2x2confirmed;not old whole-chart all-empty mode,not a reason for another equivalent countlaw')],
        controls=['Exactv1initialtensor/128window/stream/Adam/law reused;rawbaselineactuallyrerun',
            'Same1304primary andsame1389secondarycrossscores;rawanchorsNLLalso both;nooutputfilter',
            'Allmask/commonrawmachine/truecontext/audiozero on identical filteredanchors,perclasspositive/negative and firstmember',
            'Fullnative/COPY/duplicates/carry/support;2emptywindows;actualmaskedtypeexposure andAdam128/115'],
        missing_controls=[dict(question='Which event/type information change helps beyond frequency?',comparison='Matched architecture/data/occurrencepath with meaningful versus incorrect eventcontext or own-state training',decision_if='Direct mark/endproper gain connects to free output;otherwise change input/intent/representation'),
            dict(question='Can a larger designed capacity repair rather than merely add parameters?',comparison='Owner120M candidate requires module-purpose/gradient/activeparameter/localnative/independentreview plus own CUDA resource evidence',decision_if='Only separately reviewed newdesign maystart;smallpairnegative is neither admission nor blanket refusal'),
            dict(question='How much is slot supervision versus semantic data?',comparison='Fixed-anchor type/context intervention while preserving full multiset and full NULL supervision',decision_if='Ifonlyslotallocationchanges,fix transfer directly;no count-only reparameterization')],
        lessons=['Strict ALL-event>12 differs from drag-only>=8;both trigger and removal counts must be explicit',
            'Changeddevlabels require common-cohort cross-scoring,not own-target comparison',
            'Maskstreamidentity does not fix surviving-member supervision counts after canonicaldeletion',
            'New Owner120M authorization is distinct from declaring current datarecipe a qualitywinner'],
        successor_contract=dict(required_changes=['Retain strict1s v2 data rule and native support',
            'Next design directly addresses eventcontext/type/endpoint or deployment gap;capacity budget allocated by function,not unusedpadding',
            'New120M independentreview and realprobes before Modal;explicit newfee cap and original GPU guards/250stepreview'],
            forbidden_repeats=['No unchanged longer640k training or equivalent count-only law',
            'No mixing1304and1389 own-cohort scores or scoring filteredpredictions',
            'No spending oldreservedmoney/old6USDprobe grant as new120Mbudget without explicit authorization']),
        closure_reason='The exact strict1s ALL-object>12 intervention is fully implemented and tested. It removed4024drag from133trainingwindows and85from5devwindows while preserving other events. A same-initial640k128update pair and common1304/1389cross-evaluation show localization/probability gains but worsefirstmember,dragpositiveproper,fullnon-taprecall andholdtails. This data-only recipe is not promoted;the Owner filter remains. The separately requested120M design must add useful context/type/endpoint computation and pass independent review and actual admission,not simply widen this recipe or change counts.',
        source_record=str(narrative.relative_to(ROOT)))
    write(path,r);p=ROOT/'reports/local-evolution-v1/mechanism-reviews/index.json';d=read(p)
    d['candidates'].append(dict(id=ID,analysis_status='complete',review=str(path.relative_to(ROOT)),review_sha256=sha(path),markdown=str(narrative.relative_to(ROOT)),narrative_sha256=sha(narrative)));write(p,d)
    p=ROOT/'reports/rhythm_v2_budget.json';budget=read(p);budget['local_cpu_diagnostics'][ID].update(status='completed_not_promoted_mechanism_closed',optimizer_updates=356,
        quality_updates=dict(raw=128,filtered=128),health_updates=100,free_completed=68,conditional_completed=340,wall_s=report['wall_s'],
        peak_physical_bytes=max(v['runtime']['peak_sampled_physical_bytes'] for v in report['stages'].values()),review=str(narrative.relative_to(ROOT)))
    budget.setdefault('conditional_candidate_authorizations',{})['owner-modal120m-20261030']=dict(minimum_trainable_parameters=120000000,deadline_utc='2026-10-30T10:02:09Z',
        conditional_on_independent_review=True,new_usd_cap=None,cost_question_pending=True,paid_dispatched=False,old_reservations_reallocatable=False)
    write(p,budget)
    p=BASE/'state.json';s=read(p);s['local_data_experiment']=dict(id=ID,status='completed_not_promoted_mechanism_closed',review=str(narrative.relative_to(ROOT)),optimizer_updates=356,cloud_spend_usd=0)
    s['current_experiment'].update(status='data_pair_completed_while120m_design_active',review=str(narrative.relative_to(ROOT)),optimizer_updates=356)
    s.update(status='v2_data_closed_120m_design_implementation_active',training_process_running=False,terminal_receipt=str(path.relative_to(ROOT)),last_update_at=datetime.now(timezone.utc).isoformat(),
        deadline_entry='reports/modal-120m-20261030/owner-deadline.json',next_entry='Complete138Mimplementation/data/realprobes→independentreview→Modalonlyafterpassandbudget')
    write(p,s);print(dict(status='recorded',candidate=ID,updates=356,cloud=0,new120M='separate design active;budget pending'))


if __name__=='__main__':main()
