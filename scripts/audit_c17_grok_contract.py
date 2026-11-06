"""Parent C17-G contract pre-audit; stdlib hashes and frozen-data arithmetic.

No neural imports/calls, optimizer updates, free event outputs, or training.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'reports/council-shared-total-c17'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(name, value):
    with (FOLDER / name).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def probability_factorization():
    native_and_failure = [0.46, 0.29, 0.12, 0.08, 0.05]
    # Class0 is absence, classes1..K are positive counts, final class is overflow.
    presence = 1-native_and_failure[0]
    conditional = [v/presence for v in native_and_failure[1:]]
    rebuilt = [1-presence]+[presence*v for v in conditional]
    assert max(abs(a-b) for a,b in zip(rebuilt,native_and_failure)) < 1e-15
    normalized_log_score = -math.log(presence)-math.log(conditional[1])
    assert abs(normalized_log_score+math.log(native_and_failure[2])) < 1e-15
    # Expected natural joint log-score versus separately positive-normalized objective.
    # This is a supplied illustrative batch, not the unobserved training gradients.
    targets = [0, 0, 0, 1]
    bernoulli = [-math.log(1-presence) if n == 0 else -math.log(presence) for n in targets]
    positive_nll = [-math.log(conditional[n-1]) for n in targets if n]
    joint = (sum(bernoulli)+sum(positive_nll))/len(targets)
    split = sum(bernoulli)/len(targets)+sum(positive_nll)/len(positive_nll)
    assert split > joint
    # A mean true-class NLL improvement is not the posterior empty-vs-positive odds.
    prior_zero, prior_one = .813166, .136242
    posterior_one = prior_one*math.exp(.363)
    posterior_zero = .19
    rest = (1-posterior_one-posterior_zero)/64
    assert posterior_one > max(posterior_zero, rest)
    assert .363 < math.log(prior_zero/prior_one)
    return dict(
        fixed_distribution_reconstruction_error=max(abs(a-b) for a,b in zip(rebuilt,native_and_failure)),
        native_presence_excludes_overflow='Must be explicit:1-p0 includes Omega;sum native positive excludes it.',
        illustrative_normalization=dict(cells=len(targets),positive_cells=len(positive_nll),
            old_natural_joint=joint,new_separately_positive_normalized=split,
            conditional_count_reweighting=len(targets)/len(positive_nll)),
        actual_eval_denominator_ratio_illustration=6400/988,
        ratio_scope='Eval6400/988 is illustrative only; actual batch positive fractions vary; not a gradient measurement.',
        odds_counterexample=dict(prior_zero=prior_zero,prior_one=prior_one,
            posterior_zero=posterior_zero,posterior_one=posterior_one,other_class_probability=rest,
            true_class_nll_gain=.363,prior_log_odds=math.log(prior_zero/prior_one),
            posterior_correct_mode=1),
        conclusions=[
            'Hurdle and categorical counts have a fixed-distribution chain-rule identity; no new event support.',
            'Neural parameterization, separately positive-normalized loss, decision, and removed auxiliary gradients all change.',
            'One complete bundle can be tested, but cannot isolate reparameterization versus gradient removal without controls.',
            'Mean true-count NLL gain versus prior log-odds is not a proof of the required per-example mode displacement.',
        ])


def ten_ms_capacity(rows):
    total_events = occupied = extra_events = multi_type = separate_heads = 0
    max_count = 0
    examples = []
    for row in rows:
        bins = defaultdict(list)
        origin = row['request']['start_us']
        for event in row['target']:
            bins[(event['onset_us']-origin)//10000].append(event)
        total_events += len(row['target'])
        occupied += len(bins)
        for frame, events in bins.items():
            extra_events += len(events)-1
            different_types = len({e['type'] for e in events}) > 1
            multi_type += different_types
            separate_heads += len({e['onset_us'] for e in events}) > 1
            max_count = max(max_count, len(events))
            if different_types and len(examples) < 3:
                examples.append(dict(index=row['index'],frame=frame,events=events))
    return dict(requests=len(rows),events=total_events,occupied_frames=occupied,
        event_extras_over_one_event_per_frame=extra_events,multi_type_frames=multi_type,
        distinct_head_frames=separate_heads,max_events_per_frame=max_count,examples=examples,
        scope='Conditional representability audit: a single type or single event per10msframe is insufficient; '
              'R02 has not specified multiset multiplicity/member law. Not a verdict against10msaddressing.')


def duration_mode(prior, head, song_end, copies, minimum=None):
    low = max(head+1, minimum or 0)
    lower = 0
    candidates = []
    for weight, upper in zip(prior['duration_counts'], prior['duration_upper_us']):
        lo, hi = max(low, head+lower+1), min(song_end, head+upper)
        first = ((lo+999)//1000)*1000
        native = max(0, hi//1000-(lo+999)//1000+1)
        fractional = [t for t in set(copies) if t % 1000 and lo <= t <= hi]
        leaves = native+len(fractional)
        if leaves:
            earliest = min(([first] if native else [])+fractional)
            candidates.append((weight/leaves, earliest, leaves))
        lower = upper
    if not candidates:
        raise ValueError('No duration-prior legal endpoint')
    # Shared normalizer over nonempty bins cancels for leaf MAP. Earliest tie.
    score, endpoint, leaves = min(candidates, key=lambda v: (-v[0], v[1]))
    assert endpoint > head and endpoint <= song_end
    assert endpoint % 1000 == 0 or endpoint in copies
    return endpoint


def prior_duration_readout(rows):
    path = ROOT/'artifacts/c14-training-prior-audit-v1a/prior.json'
    prior = read(path)
    durations, errors, first_errors = [], [], []
    boundary_minimum_count = 0
    for row in rows:
        request = row['request']
        groups = defaultdict(list)
        for event in row['target']:
            if event['type'] == 'hold':
                groups[(event['onset_us']-request['start_us'])//40000].append(event)
        for events in groups.values():
            previous = None
            for event in sorted(events, key=lambda e: (e['onset_us'], e['end_us'])):
                minimum = previous['end_us'] if previous and previous['onset_us'] == event['onset_us'] else None
                endpoint = duration_mode(prior,event['onset_us'],request['duration_us'],request['copy_times'],minimum)
                errors.append(abs(endpoint-event['end_us'])/1000)
                durations.append((endpoint-event['onset_us'])/1000)
                if minimum is not None:
                    boundary_minimum_count += 1
                else:
                    first_errors.append(errors[-1])
                previous = event
    def median(values):
        v=sorted(values);n=len(v)
        return (v[(n-1)//2]+v[n//2])/2 if n else None
    return dict(prior_path=str(path.relative_to(ROOT)),prior_sha256=sha(path),holds=len(errors),
        prior_leaf_mode_duration_median_ms=median(durations),prior_leaf_mode_duration_at_most_10ms=sum(v<=10 for v in durations),
        tail_median_error_ms=median(errors),tail_mae_ms=sum(errors)/len(errors),
        tail_within_50ms=sum(v<=50 for v in errors),minimum_from_true_same_head_previous_end=boundary_minimum_count,
        no_forced_previous_end_tail_median_error_ms=median(first_errors),
        scope='Oracle actual true heads and canonical teacher previous ends, exact discrete legal native/COPY leaf MAP; '
              'no free chart outputs or new F1, no comparison to free neural-tail score.',
        conclusion='Better duration proper score does not establish free or teacher leaf-MAP quality; '
                   'bucket mass and uniform within-bucket leaf density differ.')


def main():
    started=time.monotonic()
    for name in ['parent-material-verification-G.json','parent-G-contract-audit.json']:
        assert not (FOLDER/name).exists(), 'Do not overwrite a certificate'
    manifest=read(FOLDER/'material-manifest.json')
    audit=read(FOLDER/'grok-access-audit.json')
    entries={r['path']:r for r in audit['files']}
    assert len(entries)==len(manifest['files'])==1554
    total=0;levels=Counter()
    for row in manifest['files']:
        path=ROOT/row['path'];assert path.resolve().is_relative_to(ROOT.resolve())
        size=path.stat().st_size;digest=sha(path);a=entries[row['path']]
        assert size==row['bytes']==a['bytes'] and digest==row['sha256']==a['sha256']
        total+=size;levels[a['read_level']]+=1
    assert dict(levels)==audit['read_level_counts']
    assert sha(FOLDER/'material-manifest.json')==audit['manifest_sha256']
    terminal=read(FOLDER/'terminal-G.json')['result']['structuredContent']
    assert terminal['status']=='completed' and not terminal['hasPendingChildRuns']
    assert terminal['model']=='grok-4.7' and {'id':'reasoningEffort','value':'xhigh'} in terminal['options']
    first=read(FOLDER/'prompt-receipt-G.json')['result']['structuredContent']['items'][0]
    assert not first['textTruncated'] and first['text']==(FOLDER/'delivered-prompt-G.txt').read_text()
    assert first['text']=='Act as the review sub-agent for this task.\n\n'+(FOLDER/'prompt-G.txt').read_text()
    write_new('parent-material-verification-G.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        status='identity_verified',objects=len(entries),bytes=total,mismatches=0,
        report_sha256=sha(FOLDER/'grok.md'),audit_sha256=sha(FOLDER/'grok-access-audit.json'),
        reviewer_reading_levels=dict(levels),exact_delivered_prompt=True,terminal_settings_verified=True,
        scope='Parent full byte/SHA pass and complete report; self-reported reading levels, not OS sandbox proof.'))
    rows=read(ROOT/'artifacts/c16-shared-total-inventory-v1/A-draft.json')
    write_new('parent-G-contract-audit.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        actions=['C17-G-R01','C17-G-R02'],status='preaudit_only_not_training_admission',
        probability_contract=probability_factorization(),ten_ms_target_capacity=ten_ms_capacity(rows),
        duration_prior_leaf_mode=prior_duration_readout(rows),
        actual_neural_forwards=0,new_optimizer_updates=0,new_free_charts=0,cloud_calls=0,
        wall_s=time.monotonic()-started))
    print(dict(status='G_collection_and_math_preaudit_complete',objects=len(entries),bytes=total,
               new_neural_forwards=0,new_updates=0,cloud_calls=0))


if __name__ == '__main__':
    main()
