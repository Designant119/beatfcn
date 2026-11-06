"""D45: audit edit-label support from frozen data and saved outputs only."""
import ast
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import EMPTY
from beatfcn.model.group_canvas import new_group_canvas, masked_group_view, decode_group_targets
from scripts.audit_draft_exposure import canvas_payload, saved_payload, signature
from scripts.matched_acoustic_data import make_example
from scripts.r09_local_data import file_sha, verify_files

ROOT = Path(__file__).resolve().parents[1]


def group_relation(source, target, mode):
    """Target comparison is an offline label; None means unresolvable ownership."""
    if source is None:
        return dict(label='unresolved', content_negative=None, binding_swap_negative=None)
    if not source:
        raise ValueError('only visible positive source groups are in scope')
    head = source[0]['onset_us']
    if any(e['onset_us'] != head for e in source):
        raise ValueError('source is not one onset group')
    exact = signature(source) == signature(target)
    same_head = bool(target) and head == target[0]['onset_us']
    same_count = len(source) == len(target)
    content_negative = not exact and same_head and same_count
    marginal = lambda events, keys: Counter(tuple(e[k] for k in keys) for e in events)
    binding_swap = content_negative and (
        marginal(source, ('type', 'end_us')) == marginal(target, ('type', 'end_us'))
        and marginal(source, ('lane',)) == marginal(target, ('lane',)))
    valid = all(e['type'] in TYPES and (
        e['end_us'] > head if e['type'] == 'hold' else e['end_us'] == head)
        for e in source)
    if mode == 'mania_4k':
        valid = valid and all(e['type'] in ('tap', 'hold') and e['lane'] in range(4) for e in source)
        valid = valid and len({e['lane'] for e in source}) == len(source)
    elif mode == 'phira':
        valid = valid and all(e['lane'] == -1 for e in source)
    else:
        raise ValueError(mode)
    return dict(label='keep' if exact else 'replace', target_positive=bool(target),
        same_head=same_head, same_count=same_count, content_negative=content_negative,
        binding_swap_negative=binding_swap, locally_valid_payload=valid,
        source_types=sorted({e['type'] for e in source}),
        target_types=sorted({e['type'] for e in target}),
        source_has_short_hold=any(e['type'] == 'hold' and 0 < e['end_us']-head <= 1_000_000 for e in source))


def transition_record(source, target, following, mode, following_active):
    relation = group_relation(source, target, mode)
    # Eligibility only sees predicted activity, visibility (caller) and resolvability.
    eligible = following_active and source is not None
    after = group_relation(following, target, mode) if following else {
        'label': 'unresolved' if following_active else ('replace' if target else 'keep')}
    known = relation['label'] != 'unresolved' and after['label'] != 'unresolved'
    return dict(**relation, following_active=following_active,
        copy_eligible_in_saved_transition=eligible, following_label=after['label'],
        correct_group_damaged=known and relation['label']=='keep' and after['label']!='keep',
        wrong_group_repaired=known and relation['label']=='replace' and after['label']=='keep',
        source_and_following_resolved=known)


class Summary:
    def __init__(self):
        self.counts = Counter()
        self.source_groups = defaultdict(set)
        self.drafts = defaultdict(set)
        self.examples = defaultdict(list)

    def add(self, row, source, target):
        key = (row['panel_id'], row['block'], row['group'])
        labels = ['visible_positive_groups', 'label_'+row['label']]
        for name in ('content_negative', 'binding_swap_negative', 'locally_valid_payload',
                     'source_has_short_hold', 'correct_group_damaged', 'wrong_group_repaired',
                     'copy_eligible_in_saved_transition', 'source_and_following_resolved'):
            if row.get(name) is True:
                labels.append(name)
        if row.get('content_negative'):
            if row.get('locally_valid_payload'):
                labels.append('locally_valid_content_negative')
            if row.get('source_has_short_hold'):
                labels.append('content_negative_with_short_hold')
        if row.get('copy_eligible_in_saved_transition'):
            labels.append('eligible_label_'+row['label'])
            for name in ('correct_group_damaged', 'wrong_group_repaired'):
                if row.get(name): labels.append('eligible_'+name)
        for kind in row.get('source_types', []):
            labels.append('source_'+kind+'_'+row['label'])
            if row.get('content_negative'): labels.append('source_'+kind+'_content_negative')
        for kind in row.get('target_types', []):
            if row.get('content_negative'): labels.append('target_'+kind+'_content_negative')
        payload_key = tuple(sorted(signature(source).items())) if source is not None else None
        for label in labels:
            self.counts[label] += 1
            self.source_groups[label].add(key)
            self.drafts[label].add((key, payload_key))
            if len(self.examples[label]) < 3:
                self.examples[label].append(dict(**row, source_group=source, target_group=target))

    def result(self):
        return {label:dict(exposures=n, distinct_source_slots=len(self.source_groups[label]),
            distinct_source_payloads=len(self.drafts[label]), examples=self.examples[label])
            for label,n in sorted(self.counts.items())}


def count(summary, label):
    return summary.get(label, {}).get('exposures', 0)


def main():
    started = time.perf_counter()
    plan_path = ROOT/'bench/rhythm-v2/d45-edit-support-v1.json'
    plan = json.loads(plan_path.read_text()); verify_files(plan)
    destination = ROOT/plan['result']
    if destination.exists(): raise FileExistsError('preserve prior audit')
    manifest = json.loads((ROOT/plan['panel_manifest']).read_text()); verify_files(manifest)
    panel = json.loads((ROOT/plan['panel_plan']).read_text())
    torch.set_num_threads(2)
    examples = [make_example(item, panel) for item in manifest['items']]
    reports = {name:json.loads((ROOT/path).read_text()) for name,path in plan['reports'].items()}
    for report in reports.values():
        if not report['complete'] or report['optimizer_updates'] != 100:
            raise ValueError('source is not a completed fixed100 probe')
        for (_,labels,identity),core in zip(examples, report['generation']['original']['cores']):
            if core['identity'] != identity or core['target_events'] != [asdict(e) for e in decode_group_targets(labels)]:
                raise ValueError('source target or identity differs')
    source_tree = ast.parse((ROOT/plan['corruption_source']).read_text())
    functions = [n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name in ('blank_for', 'corruption')]
    if len(functions) != 2: raise ValueError('corruption contract differs')
    scope = dict(np=np, new_group_canvas=new_group_canvas, masked_group_view=masked_group_view, EMPTY=EMPTY)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<frozen-corruption>', 'exec'), scope)
    prior = json.loads((ROOT/plan['previous_data_audit']).read_text())
    train = defaultdict(Summary); train_views = []; digest = hashlib.sha256()
    rng = np.random.default_rng(plan['seed'])
    for step in range(100):
        for inputs,labels,identity in examples:
            view,kind = scope['corruption'](labels, inputs, step, rng)
            truth = canvas_payload(labels); payload = canvas_payload(view)
            prior_view = prior['training_views'][len(train_views)]
            if (prior_view['identity'] != identity or prior_view['kind'] != kind
                or prior_view['stats']['visible_positive_groups'] != len(payload)
                or any(r['history'][step]['view'] != kind for r in reports.values())):
                raise ValueError('data replay differs from frozen D44')
            view_counts = Counter()
            for (block,group),events in payload.items():
                target = truth.get((block,group), [])
                relation = group_relation(events, target, labels.mode)
                row = dict(panel_id=identity['panel_id'], step=step+1, view=kind,
                           block=block, group=group, **relation)
                train[labels.mode].add(row, events, target)
                train[labels.mode+'/'+kind].add(row, events, target)
                view_counts[relation['label']] += 1
                digest.update(json.dumps(dict(row=row, source=events, target=target), sort_keys=True).encode())
            train_views.append(dict(identity=identity, step=step+1, kind=kind, counts=dict(view_counts)))
    if len(train_views) != 800: raise ValueError('training replay is incomplete')
    generated = {}; all_transitions = {}; all_unresolved = {}
    for name,report in reports.items():
        summaries = defaultdict(Summary); transitions = []; unresolved = []; view_count = 0
        for (_,labels,identity),core in zip(examples, report['generation']['original']['cores']):
            truth = canvas_payload(labels); rng = np.random.default_rng(plan['seed'])
            if len(core['rounds']) != 4: raise ValueError('four actual rounds required')
            for index,threshold in enumerate((.5, .25, .125)):
                previous,following = core['rounds'][index:index+2]
                sizes,_,payload,_,issues = saved_payload(previous, labels.layout)
                next_sizes,_,next_payload,_,next_issues = saved_payload(following, labels.layout)
                visible = rng.random(sizes.shape) >= threshold
                for stage,stage_issues in [('source', issues), ('following', next_issues)]:
                    unresolved.extend(dict(panel_id=identity['panel_id'], from_round=index+1,
                        stage=stage, **issue) for issue in stage_issues)
                for block,group in zip(*np.nonzero(visible & (sizes > 0))):
                    pos = (int(block),int(group)); events = payload.get(pos); target = truth.get(pos, [])
                    relation = transition_record(events, target, next_payload.get(pos), labels.mode,
                                                 bool(next_sizes[pos] > 0))
                    row = dict(panel_id=identity['panel_id'], block=pos[0], group=pos[1],
                        from_round=index+1, to_round=index+2, **relation)
                    summaries[labels.mode].add(row, events, target)
                    transitions.append(dict(**row, mode=labels.mode, source_group=events,
                        target_group=target, following_group=next_payload.get(pos)))
                view_count += 1
        if view_count != 24: raise ValueError('generated replay is incomplete')
        generated[name] = dict(views=view_count, modes={k:v.result() for k,v in summaries.items()})
        all_transitions[name] = transitions; all_unresolved[name] = unresolved
    training = {k:v.result() for k,v in train.items()}
    gate = {}
    for mode in ('mania_4k', 'phira'):
        train_negative = count(training[mode], 'content_negative')
        actual_negative = {name:count(value['modes'][mode], 'content_negative') for name,value in generated.items()}
        gate[mode] = dict(training_same_head_count_wrong_content=train_negative,
            saved_generated_same_head_count_wrong_content=actual_negative,
            zero_support_block=train_negative == 0 and max(actual_negative.values()) > 0)
    result = dict(id='D45', complete=True, plan_sha256=file_sha(plan_path), input_sha256=plan['input_sha256'],
        scope='offline edit labels and one-transition copy-risk counts; not a new free generation result',
        training_view_count=len(train_views), training_views=train_views, training_summary=training,
        training_full_row_stream_sha256=digest.hexdigest(), generated_summary=generated,
        generated_transitions=all_transitions, unresolved_ownership=all_unresolved,
        preregistered_support_gate=gate,
        unchanged_corruption_keep_probe_blocked=any(v['zero_support_block'] for v in gate.values()),
        model_forwards=0, optimizer_updates=0, new_paid_compute_usd=0, model_promotion=False,
        wall_s=time.perf_counter()-started, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    compact = {name:{mode:{label:count(values,label) for label in (
        'visible_positive_groups','label_keep','label_replace','label_unresolved','content_negative',
        'eligible_correct_group_damaged','eligible_wrong_group_repaired')}
        for mode,values in report['modes'].items()} for name,report in generated.items()}
    print(json.dumps(dict(gates=gate, generated=compact, wall_s=result['wall_s'],
                         peak_rss_gib=result['peak_rss_gib']), ensure_ascii=False))


if __name__ == '__main__': main()
