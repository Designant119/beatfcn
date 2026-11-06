"""Trace-only C14 analysis; never runs a model or chooses a new decoder."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from beatfcn.train.native_diagnostic_priors import KINDS, type_probability


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/c14-type-song-prefix-v1'
DONE = ROOT / 'artifacts/c14-type-diagnostic-completion-v1'
DEST = ROOT / 'reports/local-integration-v1/c14-type-effective-facts.json'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic(arm, prefix, task, index):
    name = f'diagnostic-{arm}-{prefix}-{task}-{index}.json'
    path = DONE / name
    return path if path.exists() else OLD / name


def proper_scores(items, prior):
    """Supplementary TT diagnostics, never a survivor-only SELF score."""
    items = [r for r in items if not r['forced']]
    result = {}
    for kind, name in enumerate(KINDS):
        positive = [r for r in items if r['true_type'] == kind]
        negative = [r for r in items if r['true_type'] != kind]
        scores = {}
        for label in ('model', 'frequency', 'history'):
            values = []
            for rows, target in ((positive, True), (negative, False)):
                losses = []
                for r in rows:
                    prev = KINDS.index(r['previous']['type']) if r['previous'] else -1
                    p = (r['type_probability'] if label == 'model' else
                         type_probability(prior, r['rank'], prev, r['kind_allowed'], history=label == 'history'))[kind]
                    value = p if target else 1 - p
                    losses.append(-math.log(value) if value > 0 else math.inf)
                values.append(sum(losses) / len(rows) if rows else None)
            total = sum((v or 0) * n for v, n in zip(values, (len(positive), len(negative)))) / len(items)
            scores[label] = dict(positive_nll=values[0], negative_nll=values[1], binary_nll=total)
        result[name] = dict(positive=len(positive), negative=len(negative), scores=scores)
    return result


def main():
    if DEST.exists():
        raise FileExistsError('Preserve completed analysis identity')
    report = read(DONE / 'report.json')
    old = read(OLD / 'report.json')
    plan = read(ROOT / 'bench/rhythm-v2/c14-type-diagnostic-completion-v1.json')
    prior = read(ROOT / plan['prior_path'])
    if not report['all_free_complete'] or not report['all_conditionals_complete']:
        raise ValueError('Full denominators required')
    if report['weight_digest_before'] != report['weight_digest_after']:
        raise ValueError('Weight drift')
    identity = {}
    for path, expected in plan['reused_free_files'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Reused free trace drift')
        identity[path] = expected
    paths = [ROOT / 'bench/rhythm-v2/c14-type-song-prefix-v1.json',
             ROOT / 'bench/rhythm-v2/c14-type-diagnostic-completion-v1.json',
             OLD / 'report.json', DONE / 'report.json', ROOT / plan['prior_path']]
    for path in paths:
        identity[str(path.relative_to(ROOT))] = digest(path)
    if digest(OLD / 'report.json') != plan['old_report_sha256']:
        raise ValueError('Original failure was changed')
    output = dict(scope='Frozen trace-only;TT is oracle,SELF fixes true count/current head;not deployment likelihood',
                  model_calls=0, optimizer_updates=0, cloud_calls=0, identity=identity,
                  execution=dict(free=510, base_identities=len(old['base_certificates']), conditionals=384,
                      inherited_conditionals=1, new_conditionals=383, inherited_failed_attempts=1,
                      original_wall_s=old['wall_s'], completion_wall_s=report['wall_s'],
                      cumulative_wall_s=report['cumulative_managed_wall_s'],
                      original_failure=old['failure'], weight_digest=report['weight_digest_after'],
                      original_peak_bytes=old['telemetry']['peak_sampled_physical_bytes'],
                      completion_peak_bytes=report['telemetry']['peak_sampled_physical_bytes']),
                  mechanism=report['mechanism'], conditional_summary=report['conditional_summary'],
                  free_summary={a:{t:{k:v for k,v in s.items() if k!='components'} for t,s in ts.items()}
                                for a,ts in report['free_summary'].items()},
                  supplemental_tt_proper={}, rank_diagnostics={}, self_support_causes={},
                  rank0_tt_self_identity={}, source_coverage={}, free_head_count_identity={})
    by_arm = {}
    for arm in ('B', 'Z', 'T'):
        by_arm[arm] = {}
        output['rank_diagnostics'][arm] = {}
        output['self_support_causes'][arm] = {}
        output['supplemental_tt_proper'][arm] = {}
        output['rank0_tt_self_identity'][arm] = {}
        output['source_coverage'][arm] = {}
        output['free_head_count_identity'][arm] = {}
        for task in report['free_summary'][arm]:
            rows = read(OLD / f'{arm}-{task}.json')
            by_arm[arm][task] = rows
            output['source_coverage'][arm][task] = dict(
                queries=sum(r['type_song_coverage']['total'] for r in rows),
                outside=sum(r['type_song_coverage']['out_of_range'] for r in rows))
            if arm != 'B':
                base = {r['index']:r for r in by_arm['B'][task]}
                output['free_head_count_identity'][arm][task] = dict(
                    requests=len(rows), count_equal=sum(r['prediction']['count_decision']==base[r['index']]['prediction']['count_decision'] for r in rows),
                    head_multiset_equal=sum(Counter(e['onset_us'] for e in r['prediction']['events']) ==
                        Counter(e['onset_us'] for e in base[r['index']]['prediction']['events']) for r in rows))
        for task in ('draft', 'completion', 'reference_masked'):
            data = {}
            for prefix in ('TT', 'SELF'):
                data[prefix] = []
                for rec in report['condition_rows'][arm][prefix][task]:
                    path = diagnostic(arm, prefix, task, rec['index'])
                    identity[str(path.relative_to(ROOT))] = digest(path)
                    value = read(path)
                    if value['summary'] != rec['summary']:
                        raise ValueError('Conditional summary drift')
                    data[prefix].extend(dict(r, request_index=rec['index']) for r in value['items'])
                output['rank_diagnostics'][arm].setdefault(task, {})[prefix] = {}
                for name, is_zero in (('rank0', True), ('rank_ge1', False)):
                    subset = [r for r in data[prefix] if (r['rank'] == 0) == is_zero]
                    nonforced = [r for r in subset if not r['forced']]
                    failed = any(r['head_support_failure'] or r['type_support_failure'] or r['numerical_underflow'] for r in nonforced)
                    output['rank_diagnostics'][arm][task][prefix][name] = dict(
                        events=len(subset), nonforced=len(nonforced),
                        support_failures=sum(r['score_semantics']=='infinite' for r in subset),
                        type_nll=None if failed else sum(r['type_nll'] for r in nonforced)/len(nonforced) if nonforced else None,
                        frequency_prior_nll=None if failed else sum(r['type_global_prior_nll'] for r in nonforced)/len(nonforced) if nonforced else None,
                        history_prior_nll=None if failed else sum(r['type_history_prior_nll'] for r in nonforced)/len(nonforced) if nonforced else None)
            output['supplemental_tt_proper'][arm][task] = proper_scores(data['TT'], prior)
            own = data['SELF']; causes = Counter()
            failures_by_type = Counter()
            examples = []
            for r in own:
                prev = r['previous']
                if r['head_support_failure']:
                    causes['head_true_before_previous'] += bool(prev and r['true_head_us'] < prev['onset_us'])
                    if len(examples) < 4:
                        examples.append({k:r[k] for k in ('request_index','cell','rank','true_head_us','true_type','true_end_us','previous','score_semantics')})
                if r['type_support_failure']:
                    causes['type_at_same_head_below_previous_kind'] += bool(prev and r['true_head_us']==prev['onset_us'] and r['true_type']<KINDS.index(prev['type']))
                if r['endpoint_support_failure']:
                    causes['endpoint_before_previous_same_head_hold'] += bool(prev and prev['type']=='hold' and r['true_head_us']==prev['onset_us'] and r['true_end_us']<prev['end_us'])
                if r['score_semantics']=='infinite':
                    failures_by_type[KINDS[r['true_type']]] += 1
            output['self_support_causes'][arm][task] = dict(causes=causes, failure_events_by_true_type=failures_by_type,
                raw_head_failures=sum(r['head_support_failure'] for r in own), raw_type_failures=sum(r['type_support_failure'] for r in own),
                raw_endpoint_failures=sum(r['endpoint_support_failure'] for r in own), examples=examples,
                note='Summary endpoint failures also include head-invalid hold targets;raw flags here are separated. Same-rank conditioning can exclude truth after a legal generated prefix;this is not an illegal free chart.')
            tt0={(r['request_index'],r['cell']):r for r in data['TT'] if r['rank']==0}
            own0={(r['request_index'],r['cell']):r for r in own if r['rank']==0}
            if tt0.keys()!=own0.keys():
                raise ValueError('Rank0 denominator mismatch')
            difference=max((max(abs(a-b) for a,b in zip(tt0[k]['type_probability'],own0[k]['type_probability'])) for k in tt0),default=0.)
            if difference>1e-4:
                raise ValueError('TT/SELF BOS probability mismatch')
            output['rank0_tt_self_identity'][arm][task]=dict(members=len(tt0),max_type_probability_diff=difference)
    DEST.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',diagnostic_files=384,weight_unchanged=True,
        mechanism=output['mechanism'],rank_draft=output['rank_diagnostics']['B']['draft'],
        self_causes_B=output['self_support_causes']['B']['draft']['causes']),ensure_ascii=False))


if __name__ == '__main__':
    main()
