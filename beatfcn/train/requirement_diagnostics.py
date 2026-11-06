"""Separate reference accuracy, rhythmic similarity, control response and failures."""
from collections import Counter
import math

import numpy as np

from beatfcn.model.evaluation import score_events
from beatfcn.train.control_metrics import density_request_metrics
from beatfcn.train.requirement_benchmarks import as_events

TYPES = ('tap', 'hold', 'flick', 'drag')


def _js(left, right):
    a, b = np.asarray(left, float), np.asarray(right, float)
    if not a.sum() or not b.sum():
        return None
    a, b = a/a.sum(), b/b.sum()
    mean = (a+b)/2
    def kl(value):
        mask = value > 0
        return float((value[mask]*np.log(value[mask]/mean[mask])).sum())
    return (kl(a)+kl(b))/2


def _hist(values):
    return np.histogram(values, [0, 1, 10, 30, 60, 120, 250, 500, 1000, 2000, 4000, np.inf])[0]


def _profile(events, start, end, cell_us=500_000):
    edges = np.append(np.arange(start, end, cell_us, dtype=np.int64), end)
    counts = np.histogram([e['onset_us'] for e in events], edges)[0]
    return counts/(np.diff(edges)/1e6)


def _pattern(events, start, end):
    cells = {}
    for event in events:
        cell = (event['onset_us']-start)//1_000_000
        cells.setdefault(cell, []).append((event['onset_us']-start-cell*1_000_000,
            event['end_us']-event['onset_us'], event['type'], event.get('lane', -1)))
    patterns = Counter(tuple(sorted(values)) for values in cells.values())
    ordered=sorted(events,key=lambda e:(e['onset_us'],e['type'],e.get('lane',-1),e['end_us']))
    symbols=[(b['onset_us']-a['onset_us'],b['type'],b['end_us']-b['onset_us'],b.get('lane',-1))
             for a,b in zip(ordered,ordered[1:])]
    repeat_fraction=0.;repeat_period=None
    for period in range(1,min(16,len(symbols)//3)+1):
        streak=0
        for i in range(period,len(symbols)):
            streak=streak+1 if symbols[i]==symbols[i-period]else 0
            if streak>=2*period and (streak+period)/len(symbols)>repeat_fraction:
                repeat_fraction=(streak+period)/len(symbols);repeat_period=period
    span = max((e['onset_us'] for e in events), default=start)-min(
        (e['onset_us'] for e in events), default=start)
    return dict(occupied_head_seconds=len(cells),
        occupied_head_seconds_fraction=len(cells)/max(1, math.ceil((end-start)/1e6)),
        head_span_fraction=span/(end-start),
        largest_repeated_second_pattern_fraction=max(patterns.values(), default=0)/max(1,len(cells)),
        tandem_repeat_fraction=repeat_fraction,tandem_repeat_period_events=repeat_period,
        scope='Exact native-clock event multisets; repeated rhythm is diagnostic, not automatically wrong.')


def diagnose(prediction, truth, request, *, requested_curve=None, reference_applicable=True):
    pred = prediction['events']; mode=request['mode']; start=request['start_us'];end=request['end_us']
    illegal=[]
    for index,e in enumerate(pred):
        good=(e['type'] in (('tap','hold')if mode=='mania_4k' else TYPES)
            and start<=e['onset_us']<end and e['onset_us']<=e['end_us']<=request['duration_us']
            and (e['end_us']>e['onset_us'] if e['type']=='hold' else e['end_us']==e['onset_us'])
            and (request.get('allowed_types') is None or e['type']in request['allowed_types'])
            and (mode!='mania_4k' or e.get('lane') in range(4)))
        if not good:illegal.append(index)
    # Invalid predictions remain in raw output; they do not become silently cleaned predictions.
    scores=score_events(as_events(pred,mode),as_events(truth,mode),mode) if not illegal else None
    types={}
    if scores is not None:
        for kind in (('tap','hold')if mode=='mania_4k'else TYPES):
            types[kind]=score_events(as_events([e for e in pred if e['type']==kind],mode),
                as_events([e for e in truth if e['type']==kind],mode),mode)['note_50ms']
    a,b=Counter(e['type']for e in pred),Counter(e['type']for e in truth)
    def differences(events):
        return np.diff(sorted(e['onset_us']for e in events))/1000
    def durations(events):
        return [(e['end_us']-e['onset_us'])/1000 for e in events if e['type']=='hold']
    p,t=_profile(pred,start,end),_profile(truth,start,end)
    pearson=float(np.corrcoef(p,t)[0,1])if len(p)>1 and p.std()>0 and t.std()>0 else None
    similarity=dict(type_js_nats=_js([a[k]for k in TYPES],[b[k]for k in TYPES]),
        ioi_js_nats=_js(_hist(differences(pred)),_hist(differences(truth))),
        hold_duration_js_nats=_js(_hist(durations(pred)),_hist(durations(truth))),
        density_contour_pearson=pearson,density_contour_mae_heads_per_s=float(np.abs(p-t).mean()),
        scope='Rhythmic statistics against one human reference; not human preference or playability.')
    return dict(accuracy=dict(reference_applicable=reference_applicable,scores=scores,per_type=types,
            complete=prediction['complete'],censored=not prediction['complete']),
        human_statistical_similarity=similarity,
        conditions=density_request_metrics(pred,requested_curve)if requested_curve is not None else None,
        failures=dict(illegal_event_indices=illegal,empty_positive=bool(truth)and not pred,
            generation_failure=prediction.get('failure'),
            complete_nonempty_positive=bool(truth)and bool(pred)and prediction['complete'],
            deadline_reached=prediction.get('deadline_reached',False),token_limit_reached=prediction.get('token_limit_reached',False),
            single_type_on_multitype_reference=len(a)==1 and len(b)>1,
            missing_reference_types=[k for k in b if not a[k]],
            predicted_pattern=_pattern(pred,start,end),reference_pattern=_pattern(truth,start,end)),
        counts=dict(predicted=len(pred),reference=len(truth),predicted_types=dict(a),reference_types=dict(b)))


def summarize_requirements(records, protocol):
    free=[r for r in records if r['condition']=='free']
    positive=[r for r in free if r['case_role']=='positive']
    silence=[r for r in free if r['case_role']=='natural_silence']
    expected=sum(s['case_role']=='positive' for s in protocol['samples'])
    all_complete=len(positive)==expected and all(r['prediction']['complete'] for r in positive)
    def mean(values):return float(np.mean(values)) if values else None
    def score(r,key):
        s=r['diagnostics']['accuracy']['scores'];return s[key]['f1'] if s else None
    f1=[score(r,'note_50ms')for r in positive];hold=[score(r,'hold_joint_50ms')for r in positive]
    matched={}
    for r in positive:
        for kind,s in r['diagnostics']['accuracy']['per_type'].items():
            total=matched.setdefault(kind,Counter());total.update({k:s[k]for k in ('tp','pred','true')})
    for s in matched.values():
        s['precision']=s['tp']/s['pred']if s['pred']else None
        s['recall']=s['tp']/s['true']if s['true']else None
        s['f1']=2*s['tp']/(s['pred']+s['true'])if s['pred']+s['true']else None
    conditions=[];audio=[]
    for anchor in protocol['anchors']:
        group={r['condition']:r for r in records if r['index']==anchor['index']}
        low,high=group.get('density_low'),group.get('density_high')
        complete=bool(low and high and low['prediction']['complete']and high['prediction']['complete'])
        zero=group.get('density_zero');shuffle=group.get('density_shuffle')
        conditions.append(dict(index=anchor['index'],complete=complete,
            high_minus_low_count=len(high['prediction']['events'])-len(low['prediction']['events'])if complete else None,
            low_mae=low['diagnostics']['conditions']['mae_heads_per_s']if low else None,
            high_mae=high['diagnostics']['conditions']['mae_heads_per_s']if high else None,
            explicit_zero_complete=bool(zero and zero['prediction']['complete']),
            explicit_zero_false_events=len(zero['prediction']['events'])if zero else None,
            shape_intervention_identifiable=(bool(high and shuffle)and
                [s['density']for s in high['requested_curve']]!=[s['density']for s in shuffle['requested_curve']]),
            shuffled_mae=shuffle['diagnostics']['conditions']['mae_heads_per_s']if shuffle else None,
            scope='Counts/MAE are censored until generation completes. Low/high curves derive from reference density, not exact event truth.'))
        real=group.get('free')
        for kind in ('audio_zero','local_roll','audio_foreign'):
            other=group.get(kind);done=bool(real and other and real['prediction']['complete']and other['prediction']['complete']
                and real['diagnostics']['accuracy']['scores']and other['diagnostics']['accuracy']['scores'])
            audio.append(dict(index=anchor['index'],intervention=kind,complete=done,
                real_minus_counterfactual_note_f1=(score(real,'note_50ms')-score(other,'note_50ms'))if done else None,
                same_event_multiset=(sorted(tuple(sorted(e.items()))for e in real['prediction']['events'])==
                    sorted(tuple(sorted(e.items()))for e in other['prediction']['events']))if done else None))
    useful=sum(r['diagnostics']['failures']['complete_nonempty_positive'] for r in positive)
    # Undefined per-window similarities are counted, never silently dropped.
    undefined={key:sum(r['diagnostics']['human_statistical_similarity'][key]is None for r in positive)
        for key in ('type_js_nats','ioi_js_nats','hold_duration_js_nats','density_contour_pearson')}
    imputed={}
    for key,kind in (('type_js_nats','type'),('ioi_js_nats','ioi')):
        values=[]
        for r in positive:
            value=r['diagnostics']['human_statistical_similarity'][key]
            if value is None:
                n,m=r['diagnostics']['counts']['predicted'],r['diagnostics']['counts']['reference']
                value=0. if (n<(2 if kind=='ioi' else 1))==(m<(2 if kind=='ioi' else 1)) else JS_MAX_NATS
            values.append(value)
        imputed[key]=mean(values)if all_complete else None
    # Pooled hold counts include windows where only one side has holds; a
    # per-window mean that skips both-empty windows is reported alongside.
    hold_counts=[r['diagnostics']['accuracy']['scores']['hold_joint_50ms']for r in positive
        if r['diagnostics']['accuracy']['scores']]
    pooled_hold=dict(tp=sum(c['tp']for c in hold_counts),pred=sum(c['pred']for c in hold_counts),
        true=sum(c['true']for c in hold_counts),windows=len(hold_counts),
        windows_with_true_holds=sum(c['true']>0 for c in hold_counts),
        windows_with_pred_holds=sum(c['pred']>0 for c in hold_counts),
        windows_without_any_hold=sum(c['true']==0 and c['pred']==0 for c in hold_counts))
    pooled_hold['f1']=(2*pooled_hold['tp']/(pooled_hold['pred']+pooled_hold['true'])
        if pooled_hold['pred']+pooled_hold['true'] else None)
    silence_seconds=sum((r['request']['end_us']-r['request']['start_us'])/1e6 for r in silence if 'request' in r)
    similarity_keys=('type_js_nats','ioi_js_nats','hold_duration_js_nats','density_contour_pearson','density_contour_mae_heads_per_s')
    human={key:mean([r['diagnostics']['human_statistical_similarity'][key]for r in positive
        if r['diagnostics']['human_statistical_similarity'][key]is not None])if all_complete else None for key in similarity_keys}
    return dict(expected_positive=expected,attempted_positive=len(positive),complete_positive=sum(r['prediction']['complete']for r in positive),
        complete_nonempty_positive=useful,all_empty_observed=bool(positive)and all(not r['prediction']['events']for r in positive),
        all_empty_hack=all_complete and all(not r['prediction']['events']for r in positive),
        teacher_positive_ce=mean([r['teacher_joint_ce']for r in positive if r.get('teacher_joint_ce')is not None]),
        teacher_silence_ce=mean([r['teacher_joint_ce']for r in silence if r.get('teacher_joint_ce')is not None]),
        full_note_f1_50ms=mean([x for x in f1 if x is not None])if all_complete else None,
        full_hold_joint_f1_50ms=mean([x for x in hold if x is not None])if all_complete else None,
        prefix_note_f1_diagnostic=mean([x for x in f1 if x is not None]),matched_per_type=matched,matched_per_type_censored=not all_complete,
        silence_false_events_lower_bound=sum(len(r['prediction']['events'])for r in silence),
        silence_false_events_per_s=(sum(len(r['prediction']['events'])for r in silence)/silence_seconds
            if silence_seconds else None),
        silence_rate_is_lower_bound=not all(r['prediction']['complete']for r in silence),
        hold_joint_pooled_50ms=pooled_hold,
        human_similarity_undefined_windows=undefined,
        human_similarity_one_side_empty_imputed=imputed,
        silence_all_complete=bool(silence)and all(r['prediction']['complete']for r in silence),
        conditions_response=conditions,audio_counterfactuals=audio,
        human_statistical_similarity=human,human_similarity_scope='Descriptive single-reference rhythmic similarity; no calibrated human rating.',
        whole_song=dict(status='UNMEASURED',reason='Fixed8second windows do not establish full-song structure, generated history or play/edit quality.'),
        product_requirements_satisfied=None,product_verdict='NOT_ESTABLISHED',
        evidence_limit='No product pass from type counts, low teacher loss, empty outputs, truncated prefixes or marginal F1 gains.')


# --------------------------------------------------------------------------
# Benchmark v2 metrics (design/BENCHMARK_GUIDE_v2.md).  Every function takes raw
# event dicts {onset_us,end_us,type,lane}; nothing here cleans or snaps output.
# Undefined values are returned with an explicit status and are counted by the
# summaries instead of silently dropping out of a mean.
# --------------------------------------------------------------------------
TOLERANCES_MS = (20, 50, 100)
# Full-range gap bins: the first four bins expose sub-40 ms gaps that the legacy
# 30 ms IOI floor (beatfcn/model/evaluation.py IOI_MIN_MS) cannot see.
GAP_EDGES_MS = (0., 10., 20., 30., 40., 60., 80., 120., 160., 250., 375., 500., 750.,
                1000., 2000., 4000., math.inf)
SHORT_GAP_MS = 40.
JS_MAX_NATS = math.log(2.)


def js_with_status(left, right):
    """Jensen-Shannon (nats) with an explicit status.

    both_empty -> (0., 'both_empty'); one side empty -> (ln 2, 'one_side_empty'),
    the maximum JS, because an empty histogram cannot resemble a nonempty one.
    """
    a, b = np.asarray(left, float), np.asarray(right, float)
    if not a.sum() and not b.sum():
        return 0., 'both_empty'
    if not a.sum() or not b.sum():
        return JS_MAX_NATS, 'one_side_empty'
    return _js(a, b), 'defined'


def kinds_for(mode):
    return ('tap', 'hold') if mode == 'mania_4k' else TYPES


def onset_gaps_ms(events):
    """Gaps between consecutive DISTINCT onsets (chord members are not gaps)."""
    onsets = np.unique(np.asarray([e['onset_us'] for e in events], np.int64))
    return np.diff(onsets)/1000.


def gap_histogram(gaps):
    return np.histogram(np.asarray(gaps, float), GAP_EDGES_MS)[0]


def _times(events, kind=None, lane=None):
    return sorted(e['onset_us'] for e in events if (kind is None or e['type'] == kind)
                  and (lane is None or e.get('lane', -1) == lane))


def _note_tp(pred, truth, mode, tol_us, kind=None):
    from beatfcn.model.sample import match_f1
    keys = {(e['type'], e.get('lane', -1) if mode == 'mania_4k' else None)
            for e in list(pred)+list(truth) if kind is None or e['type'] == kind}
    tp = 0
    for k, lane in keys:
        tp += match_f1(_times(pred, k, lane if mode == 'mania_4k' else None),
                       _times(truth, k, lane if mode == 'mania_4k' else None), tol_us)[0]
    return tp


def match_counts(pred, truth, mode):
    """Raw tp/pred/true counts, so pooled (micro) F1 is unbiased across windows."""
    from beatfcn.model.evaluation import hold_matches
    from beatfcn.model.sample import match_f1
    out = {}
    for ms in TOLERANCES_MS:
        tol = ms*1000
        out[f'onset_{ms}ms'] = dict(tp=match_f1(_times(pred), _times(truth), tol)[0],
                                    pred=len(pred), true=len(truth))
        out[f'note_{ms}ms'] = dict(tp=_note_tp(pred, truth, mode, tol), pred=len(pred), true=len(truth))
    for kind in kinds_for(mode):
        p = [e for e in pred if e['type'] == kind]; t = [e for e in truth if e['type'] == kind]
        out[f'type_{kind}_50ms'] = dict(tp=_note_tp(p, t, mode, 50_000, kind), pred=len(p), true=len(t))
    p, t = as_events(pred, mode), as_events(truth, mode)
    for ms in (50, 100):
        joint = hold_matches(p, t, ms*1000, ms*1000, match_lane=mode == 'mania_4k')
        joint.pop('tail_errors_us')
        out[f'hold_joint_{ms}ms'] = joint
    return out


def f1_of(counts):
    total = counts['pred']+counts['true']
    return 2*counts['tp']/total if total else None


def pool_counts(rows):
    pooled = {}
    for row in rows:
        for key, counts in row.items():
            target = pooled.setdefault(key, dict(tp=0, pred=0, true=0))
            for name in ('tp', 'pred', 'true'):
                target[name] += counts[name]
    for counts in pooled.values():
        counts['f1'] = f1_of(counts)
        counts['precision'] = counts['tp']/counts['pred'] if counts['pred'] else None
        counts['recall'] = counts['tp']/counts['true'] if counts['true'] else None
    return pooled


def circular_shift(events, start_us, end_us, shift_us):
    span = end_us-start_us
    shifted = []
    for e in events:
        onset = start_us+(e['onset_us']-start_us+shift_us) % span
        shifted.append(dict(e, onset_us=onset, end_us=onset+e['end_us']-e['onset_us']))
    return shifted


def covered_end(prediction, start_us, end_us):
    """End of the time prefix the decoder actually covered.

    Complete output covers the window. Incomplete output covers only up to its
    last emitted head; nothing after it was decided, so it cannot count as
    missed or as silent.
    """
    if prediction.get('complete'):
        return end_us
    onsets = [e['onset_us'] for e in prediction.get('events', ())]
    return min(end_us, max(onsets)+1) if onsets else start_us


def window_accuracy(prediction, truth, request, *, null_shifts_s=(1, 2, 3)):
    mode, start, end = request['mode'], request['start_us'], request['end_us']
    pred = list(prediction.get('events', ()))
    cover = covered_end(prediction, start, end)
    prefix_pred = [e for e in pred if e['onset_us'] < cover]
    prefix_truth = [e for e in truth if e['onset_us'] < cover]
    null = []
    for seconds in null_shifts_s:
        for sign in (1, -1):
            null.append(match_counts(circular_shift(pred, start, end, sign*int(seconds*1e6)), truth, mode))
    return dict(full=match_counts(pred, truth, mode), prefix=match_counts(prefix_pred, prefix_truth, mode),
        coverage_fraction=(cover-start)/(end-start), covered_end_us=cover,
        null_shift=pool_counts(null), null_shifts_s=list(null_shifts_s),
        has_true_holds=any(e['type'] == 'hold' for e in truth),
        has_pred_holds=any(e['type'] == 'hold' for e in pred))


def relative_similarity(left, right, left_start, right_start, mode, tol_us=50_000):
    """Onset+type(+lane) F1 between two outputs aligned to their own window starts.

    Both empty -> 1.0 (identical), which is exactly what template collapse needs.
    """
    a = [dict(e, onset_us=e['onset_us']-left_start) for e in left]
    b = [dict(e, onset_us=e['onset_us']-right_start) for e in right]
    if not a and not b:
        return 1.
    return 2*_note_tp(a, b, mode, tol_us)/(len(a)+len(b))


def same_relative_multiset(left, right, left_start, right_start):
    key = lambda events, start: sorted((e['onset_us']-start, e['end_us']-e['onset_us'], e['type'],
                                        e.get('lane', -1)) for e in events)
    return key(left, left_start) == key(right, right_start)


def human_likeness(pairs):
    """Pooled rhythm statistics of outputs vs their human references.

    pairs: [(pred_events, truth_events, request)]. Short gaps are counted; the
    reference rate is computed from the very same windows' human charts.
    """
    stats = {}
    for side in ('model', 'human'):
        gaps = [onset_gaps_ms(p if side == 'model' else t) for p, t, _ in pairs]
        gaps = np.concatenate(gaps) if gaps else np.zeros(0)
        stats[side] = dict(gaps=int(len(gaps)),
            short_gap_fraction=float((gaps < SHORT_GAP_MS).mean()) if len(gaps) else None,
            sub30ms_gap_fraction=float((gaps < 30.).mean()) if len(gaps) else None,
            histogram=gap_histogram(gaps).tolist())
    ioi, ioi_status = js_with_status(stats['model']['histogram'], stats['human']['histogram'])
    types_model = Counter(e['type'] for p, _, _ in pairs for e in p)
    types_human = Counter(e['type'] for _, t, _ in pairs for e in t)
    type_js, type_status = js_with_status([types_model[k] for k in TYPES], [types_human[k] for k in TYPES])
    pearson, undefined, repeat = [], 0, dict(model=[], human=[])
    for p, t, request in pairs:
        a, b = _profile(p, request['start_us'], request['end_us']), _profile(t, request['start_us'], request['end_us'])
        if len(a) > 1 and a.std() > 0 and b.std() > 0:
            pearson.append(float(np.corrcoef(a, b)[0, 1]))
        else:
            undefined += 1
        repeat['model'].append(_pattern(p, request['start_us'], request['end_us'])['tandem_repeat_fraction'])
        repeat['human'].append(_pattern(t, request['start_us'], request['end_us'])['tandem_repeat_fraction'])
    mean = lambda values: float(np.mean(values)) if values else None
    return dict(windows=len(pairs), gap_edges_ms=[g if math.isfinite(g) else 'inf' for g in GAP_EDGES_MS],
        pattern_profile=pattern_profile_by_mode(pairs),
        ioi_js_nats=ioi, ioi_js_status=ioi_status,
        short_gap_fraction_model=stats['model']['short_gap_fraction'],
        short_gap_fraction_human=stats['human']['short_gap_fraction'],
        sub30ms_gap_fraction_model=stats['model']['sub30ms_gap_fraction'],
        sub30ms_gap_fraction_human=stats['human']['sub30ms_gap_fraction'],
        gaps_model=stats['model']['gaps'], gaps_human=stats['human']['gaps'],
        gap_histogram_model=stats['model']['histogram'], gap_histogram_human=stats['human']['histogram'],
        type_mix_js_nats=type_js, type_mix_js_status=type_status,
        type_counts_model=dict(types_model), type_counts_human=dict(types_human),
        density_profile_pearson=mean(pearson), density_profile_defined=len(pearson),
        density_profile_undefined=undefined,
        tandem_repeat_fraction_model=mean(repeat['model']), tandem_repeat_fraction_human=mean(repeat['human']))


def pattern_profile_by_mode(pairs):
    """Pattern-level human likeness (workflow_metrics.PROFILE_METRICS) per mode, pooled over windows:
    each window's model and human profiles (window clock, ms IOIs -- no charter timing here) are
    summed, then compared once. Values: model, human, |model - human|, JS (nats)."""
    from beatfcn.train.workflow_metrics import compare_profiles, merge_profiles, pattern_profile
    by_mode = {}
    for p, t, request in pairs:
        span = dict(start_us=request['start_us'], duration_us=request['end_us']-request['start_us'])
        entry = by_mode.setdefault(request['mode'], ([], []))
        entry[0].append(pattern_profile(p, request['mode'], **span))
        entry[1].append(pattern_profile(t, request['mode'], **span))
    out = {}
    for mode, (model, human) in sorted(by_mode.items()):
        cmp = compare_profiles(merge_profiles(model), merge_profiles(human))
        out[mode] = dict(windows=len(model), model_notes=cmp['draft_notes'], human_notes=cmp['reference_notes'],
                         metrics={name: dict(model=m['draft'], human=m['reference'], abs_diff=m['abs_diff'],
                                             **({'js_nats': m['js_nats'], 'js_status': m['js_status']}
                                                if 'js_nats' in m else {}))
                                  for name, m in cmp['metrics'].items()},
                         scope='pooled 8 s windows on the window clock (IOI in ms); see BENCHMARK_GUIDE_v2 1.2b')
    return out


def template_similarity(outputs, mode):
    """Mean pairwise relative similarity across DIFFERENT audio windows.

    outputs: [(events, start_us, audio_sha256)]. Compare against the same
    statistic on the human charts: model >> human means one template is reused.
    """
    values, identical = [], 0
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            a, b = outputs[i], outputs[j]
            if a[2] == b[2]:
                continue
            values.append(relative_similarity(a[0], b[0], a[1], b[1], mode))
            identical += same_relative_multiset(a[0], b[0], a[1], b[1])
    return dict(pairs=len(values), mean=float(np.mean(values)) if values else None,
                identical_pairs=identical)


# ---------------------------------------------------------------- silence split
# Predicted notes where the human chart is empty are split by the LOCAL audio:
# "true silence" (the music is near-silent there: a real hack) vs "uncharted
# music" (audible music the human left empty: a charting choice, not a hack).
SILENCE_PERCENTILE = 2.
SILENCE_MARGIN = 1.
SILENCE_WINDOW_US = 50_000
MEL_FRAME_US = 10_000


def song_energy(mel):
    """Per-frame mean log-mel of a whole song's complete mel ([frames, bins], 10 ms frames,
    frame k centred at k*10 ms + 5 ms) and the near-silence threshold: the song's
    2nd-percentile frame energy + 1.0 (log-mel units)."""
    energy = np.asarray(mel, np.float64).mean(-1)
    if not len(energy):
        return None
    return dict(energy=energy, threshold=float(np.percentile(energy, SILENCE_PERCENTILE) + SILENCE_MARGIN),
                rule=f'near-silent: max frame mean-log-mel within +-{SILENCE_WINDOW_US//1000} ms of the onset '
                     f'< song {SILENCE_PERCENTILE:g}th-percentile frame energy + {SILENCE_MARGIN:g}')


def near_silent(onset_us, song):
    """True when the max frame energy within +-50 ms of ``onset_us`` is below the threshold
    (no frame in range -> treated as silent: beyond the decoded audio)."""
    lo = max(0, math.ceil((onset_us - SILENCE_WINDOW_US - MEL_FRAME_US // 2) / MEL_FRAME_US))
    hi = min(len(song['energy']) - 1, math.floor((onset_us + SILENCE_WINDOW_US - MEL_FRAME_US // 2) / MEL_FRAME_US))
    if hi < lo:
        return True
    return float(song['energy'][lo:hi+1].max()) < song['threshold']


def silence_split(events, song):
    """Counts of predicted events (by onset) that are near-silent vs audible music."""
    silent = sum(near_silent(int(e['onset_us']), song) for e in events)
    return dict(true_silence_events=int(silent), uncharted_music_events=len(events) - int(silent),
                threshold=song['threshold'])


def uncharted_regions(notes, duration_us, *, min_gap_us=4_000_000, margin_us=100_000):
    """Regions where a human chart has no note (head or hold body) for >= ``min_gap_us``
    (song start/end included), shrunk by ``margin_us`` next to the bounding human notes.
    ``notes``: (onset_us, end_us, ...) tuples. Returns [(lo_us, hi_us)]."""
    spans = sorted((int(n[0]), max(int(n[0]), int(n[1]))) for n in notes)
    regions, cursor, first = [], 0, True
    for onset, end in spans + [(int(duration_us), int(duration_us))]:
        if onset - cursor >= min_gap_us:
            lo = cursor if first else cursor + margin_us
            hi = onset if onset >= duration_us else onset - margin_us
            if hi > lo:
                regions.append((lo, hi))
        first = False
        cursor = max(cursor, end)
    return regions


def uncharted_split(draft_onsets_us, regions, song):
    """Whole-song silence split: draft notes inside ``regions`` split by local energy."""
    seconds = sum(hi - lo for lo, hi in regions) / 1e6
    inside = [t for t in draft_onsets_us if any(lo <= t < hi for lo, hi in regions)]
    silent = sum(near_silent(int(t), song) for t in inside)
    return dict(uncharted_s=seconds, regions=len(regions), uncharted_notes=len(inside),
                true_silence_notes=int(silent), uncharted_music_notes=len(inside) - int(silent),
                true_silence_events_per_s=silent / seconds if seconds else None,
                uncharted_music_events_per_s=(len(inside) - silent) / seconds if seconds else None)
