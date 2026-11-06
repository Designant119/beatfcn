"""Training-method helpers shared by bounded rounds and regression tests."""
from dataclasses import replace
import hashlib
import json

from beatfcn.model import features as F
from beatfcn.model.evaluation import groups_to_events, legality_issues
from beatfcn.model.sample import generate_core, core_context


INITIAL_POLICY = {
    'request_mix': {'from_scratch': .40, 'middle_holes': .30, 'continuation': .10,
                    'sparse_anchors': .10, 'reference': .10},
    'control_drop_all_p': .50,
    'paired_references': True,
    'rollout_fraction': .02,
    'replay_max_uses': 32,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def own_history_core(model, sample, chart_events, song, c0, *, seed, device='cpu', deadline=None,
                     generator=generate_core):
    """Generate one training core, then supervise the unchanged human next core.

    Conflicting futures are rejected in full and counted by the caller. Never repair
    the target by silently deleting notes. This is short-rollout continuation, not
    scheduled sampling inside a teacher-forced sequence or self-labelled targets.
    """
    duration = sample.request['audio_duration_us']
    if c0 < 0 or c0 + 2 * F.CORE_US > duration:
        raise F.SkipExample('own-history needs two complete adjacent cores')
    li = F.line_index_of(chart_events)
    ex = F.build_core(sample, chart_events, c0, duration, line_index=li)
    holds, runs, heads = core_context(sample, ex, li)
    diagnostics = {}
    groups = generator(model, ex, song, holds, runs, heads, seed=seed, device=device,
                       greedy=False, diagnostics=diagnostics, deadline=deadline)
    if diagnostics.get('deadline_reached') or diagnostics.get('group_limit_reached'):
        raise F.SkipExample('own-history rollout did not finish')
    generated = groups_to_events(groups, c0, sample.request['mode'], li, f'rollout:{seed}', model.c.phira_location)
    if legality_issues(sample.request, generated, placement=model.c.phira_location):
        raise F.SkipExample('own-history rollout is illegal')
    following = c0 + F.CORE_US
    previous = [e for e in sample.hidden if e.onset_us < c0]
    future = [e for e in sample.hidden if e.onset_us >= following]
    live = replace(sample, hidden=tuple(previous + generated + future))
    target = F.build_core(live, chart_events, following, duration, line_index=li)
    bad = F.check_targets(target)
    if bad:
        raise F.SkipExample('own-history conflicts with unchanged future: ' + ','.join(bad))
    if not target.groups:
        raise F.SkipExample('own-history future is empty')
    target.provenance = {'kind': 'own_history_continuation', 'c0_us': following, 'rollout_seed': seed,
                         'generated_prefix_notes': len(generated), 'notes': sum(len(g['notes']) for g in target.groups),
                         'holds': sum('end_rel_us' in n for g in target.groups for n in g['notes'])}
    return target


def reserve_stage(ledger, name, resources):
    """Conservative allocations remain charged against the cap even after a failed run."""
    if name in ledger['stages']:
        raise ValueError(f'{name} was already dispatched; inspect it before choosing a new attempt')
    rate = ({None: 0.0, 'L40S': .000542, 'H100': .001097}[resources.get('gpu')]
            + resources['cpu'] * .0000131 + resources['memory_gib'] * .00000222)
    reserved = rate * (resources['timeout'] + 120)
    if sum(s['reserved_usd'] for s in ledger['stages'].values()) + reserved > ledger['approved_usd']:
        raise ValueError('campaign budget exhausted; no compute dispatched')
    stage = {'reserved_usd': reserved, 'rate_usd_s': rate, 'resources': resources, 'status': 'reserved'}
    ledger['stages'][name] = stage
    return stage
