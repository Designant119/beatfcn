"""Audit new council proposals against saved outputs, without new inference."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'reports/council-frontend-health-v1/parent-callback-audit.json'


def load(name, identities):
    path = ROOT / name
    data = path.read_bytes()
    identities[name] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the original callback audit')
    identities = {}
    plan = load('bench/rhythm-v2/local-frontend-comparison-v2.json', identities)
    tape = load(plan['tape'], identities)
    charts = {r['chart_id']: r for r in tape['charts']}
    full = {r['audio_sha256']: r['full_song_context'] for r in plan['feature_context_audit']}
    result = {}
    for arm in ('pupu', 'mert'):
        groups = defaultdict(list)
        excluded = []
        for task in ('new_draft', 'reference_masked'):
            name = f'artifacts/local-frontend-comparison-v2/{arm}/{task}-final.json'
            for row in load(name, identities):
                audio = charts[row['chart_id']]['audio_sha256']
                if row['index'] in plan['health_indices'] + [plan['empty_health_index']] or not full[audio]:
                    excluded.append(dict(task=task, index=row['index'], audio_sha256=audio))
                    continue
                groups[row['component']].append(row)
        components = {c:dict(
            onset10=mean(r['scores']['onset_10ms']['f1'] or 0 for r in rows),
            full_note_f1=mean(r['scores']['full_note_f1'] for r in rows),
            requests=len(rows)) for c, rows in groups.items()}
        aggregates = {k:mean(v[k] for v in components.values()) for k in ('onset10', 'full_note_f1')}
        # Worst leave-one-component-out mean tests removal of the strongest component
        # for each endpoint without selecting a favorable removal.
        loo = {k:min(mean(v[k] for c, v in components.items() if c != omitted)
                     for omitted in components) for k in aggregates}
        empties = load(f'artifacts/local-frontend-comparison-v2/{arm}/empty-final.json', identities)
        empty_counts = [len(r['prediction']['events']) for r in empties]
        result[arm] = dict(components=components, component_count=len(components),
            excluded=excluded, macro=aggregates, minimum_leave_one_out=loo,
            empty_event_counts=empty_counts,
            grok_new_proposal_only=dict(
                component_support=len(components) >= 10,
                onset10=aggregates['onset10'] >= .1848 and loo['onset10'] >= .1848,
                full_note_f1=aggregates['full_note_f1'] >= .1957 and loo['full_note_f1'] >= .1957,
                empty_silent=not any(empty_counts)))
    health = load('artifacts/local-frontend-comparison-v1/health-predictions.json', identities)
    holds = {}
    key = lambda e: (e['onset_us'], e['end_us'], e['type'], e['lane'])
    for arm, rows in health.items():
        row = next(r for r in rows if r['index'] == 1283)
        pred = Counter(key(e) for e in row['prediction']['events'] if e['type'] == 'hold')
        truth = Counter(key(e) for e in row['target'] if e['type'] == 'hold')
        holds[arm] = dict(predicted=sum(pred.values()), target=sum(truth.values()),
            exact_hold_multiset=pred == truth,
            head_only_pair_tail_error=row['scores']['tail_error_given_head_10ms'])
    output = dict(recording_basis='Post-callback audit; not preregistration or a new acceptance gate',
        optimizer_updates=0, new_inference_calls=0, cloud_compute_calls=0,
        grok_proposal_audit=result, health_1283_hold_audit=holds, source_sha256=identities,
        interpretation=[
            'The original investment screen explicitly reports empty false positives as risk, not a conjunctive gate.',
            'New draft and masked middle-completion requests have different target regions; this pooled audit is not the original draft comparison.',
            'Head-only maximum matching need not minimize tail error; a nonzero diagnostic does not imply emitted exact holds are incorrect.',
            'No existing scores, frozen sources, weights, acceptance decisions or budget reservations were changed.'])
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({arm:{k:v for k,v in r.items() if k!='components'}
                      for arm,r in result.items()}, ensure_ascii=False, indent=2))
    print(json.dumps(holds, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
