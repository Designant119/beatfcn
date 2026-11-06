"""Bounded local combined-method experiment, explicitly authorized by the user."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import FIELDS
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.local_bundle import LocalBundleModel, bundle_loss, generate_bundle
from beatfcn.model.local_song_rollout import canonical, independent_legality, generate_complete_songs
from beatfcn.model.evaluation import score_events
from beatfcn.train.local_bundle import make_row, batch_rows, prefix_row, tensor_audit
from beatfcn.train.local_request_panel import LocalRequestPanel

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/local-bundle-v1.json'
REPORT = ROOT / 'reports/local_bundle_v1.json'
ART = ROOT / 'artifacts/local-bundle-v1'
ARMS = ('baseline', 'bundle')
VIEWS = ('draft', 'single_condition', 'all_conditions', 'random_groups', 'middle', 'continuation')


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def log(message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


def journal(message):
    with (ROOT / 'EXPERIMENT_LOG.md').open('a') as stream:
        stream.write('\n\nlocal-bundle-v1 ' + message + '\n')


def panel_for(plan):
    return LocalRequestPanel(ROOT, plan['tape'], plan['features'])


def prepare():
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError('Frozen experiment already exists')
    started = time.monotonic()
    plan = dict(id='local-bundle-v1', seed=20261027,
        tape='reports/d94_expanded_target_tape_v1.json', features='reports/d94_merged_feature_admission_v1.json',
        authorization='2026-10-27 user: small models/data, local experiments, all methods together; supersedes single-hypothesis restriction for this bundle only',
        dimensions=['Architecture', 'Objective', 'DataRepresentation', 'TrainingDynamics'],
        hypothesis='Combined physically addressed audio readout, direct conditions/style, gesture and tail heads, learned group cardinality, field-balanced supervision and generated-history recovery improve small-data held-out chart generation versus the original model.',
        causal_scope='Whole recipe comparison; no attribution to an individual component',
        model=asdict(FullEventConfig(d=96, heads=4, kv_heads=2, ffn=256,
            encoder_layers=1, decoder_layers=3, dropout=0., mel_bins=128)),
        steps=1000, effective_batch=2, checkpoints=[250, 500, 1000],
        learning_rate=.001, minimum_lr=.0001, warmup_steps=50, clip_norm=1., weight_decay=0.,
        health_updates=100, health_wall_s=900, train_wall_s=3000, evaluation_wall_s=1200,
        replay_steps=[200, 500, 800], replay_token_cap=384, generation_token_cap=1024,
        full_song_token_cap=2048, cloud_compute_limit=0,
        gates=dict(draft_full_f1_gain=.03, draft_onset_f1_gain=.05,
                   draft_complete_fraction=.875, minimum_correct_non_tap_types=2,
                   completion_full_f1_gain=.02, density_direction_songs=3),
        condition_generation='Manually fixed density bands only; no hidden validation attributes',
        notes=['Phira-only small-scale feasibility experiment; not a full product or Mania validation.',
               'Native labels retained. Small training windows limited to <=768 tokens; denser regimes remain outside this run.',
               'Baseline and bundle use identical common initialization, source indices, ordinary view schedule and update count. Bundle additionally uses generated prefix references, different loss, and added heads.',
               'No grid snapping, short-interval deletion, fixed type quota, or hidden-target density at inference.',
               'All methods form one candidate; health weights are discarded. Fresh local run, not an extension of a prior deadline.'])
    panel = panel_for(plan)
    by_component = defaultdict(list)
    for ci, chart in enumerate(panel.charts):
        if chart['mode'] == 'phira':
            by_component[(chart['split'], chart['component'])].append(ci)
    selected = {}
    for split, wanted in (('train', 24), ('val', 8)):
        buckets = defaultdict(list)
        for (sp, component), charts in by_component.items():
            if sp != split:
                continue
            representative = min(charts, key=lambda i: panel.charts[i]['stratum'])
            strata = panel.charts[representative]['stratum']
            digest = hashlib.sha256(f"{plan['seed']}:{component}".encode()).hexdigest()
            buckets[strata].append((len(charts) < 2, digest, component, charts))
        for values in buckets.values():
            values.sort()
        chosen = []
        # Round robin exact source strata, preserving labels instead of merging HD/IN/AT.
        while len(chosen) < wanted and any(buckets.values()):
            for stratum in sorted(buckets):
                if buckets[stratum] and len(chosen) < wanted:
                    chosen.append(buckets[stratum].pop(0))
        if len(chosen) != wanted:
            raise ValueError('Not enough independent songs')
        selected[split] = chosen
    indices = {'train': [], 'val': []}
    chart_ids = {'train': [], 'val': []}
    for split, chosen in selected.items():
        for _, _, component, charts in chosen:
            use = sorted(charts, key=lambda c: panel.charts[c]['chart_id'])[:2 if split == 'train' else 1]
            for ci in use:
                chart_ids[split].append(panel.charts[ci]['chart_id'])
                rows = [i for i, r in enumerate(panel.records) if r['chart_index'] == ci]
                eligible = [i for i in rows if 20 <= panel.records[i]['tokens'] <= 768
                            and panel.records[i]['request']['end_us'] - panel.records[i]['request']['start_us'] == 8_000_000]
                if len(eligible) < 2:
                    raise ValueError('Selected chart lacks two eligible full windows')
                ordered = sorted(eligible, key=lambda i: (panel.records[i]['event_count'], i))
                if split == 'train':
                    picks = []
                    for kind in ('hold', 'flick', 'drag'):
                        best = max(eligible, key=lambda i: (panel.records[i]['types'].get(kind, 0), -i))
                        if panel.records[best]['types'].get(kind, 0):
                            picks.append(best)
                    picks += [ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]]
                    picks = list(dict.fromkeys(picks))[:4]
                    empty = [i for i in rows if panel.records[i]['event_count'] == 0]
                    if empty:
                        picks.append(empty[0])
                else:
                    picks = [ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]]
                    picks = list(dict.fromkeys(picks))
                    if len(picks) == 1:
                        picks.append(next(i for i in ordered if i != picks[0]))
                indices[split].extend(picks)
    train_components = {panel.records[i]['component'] for i in indices['train']}
    val_components = {panel.records[i]['component'] for i in indices['val']}
    if train_components & val_components:
        raise ValueError('Song component leakage')
    train_audio = {panel.records[i]['audio_sha256'] for i in indices['train']}
    val_audio = {panel.records[i]['audio_sha256'] for i in indices['val']}
    if train_audio & val_audio:
        raise ValueError('Audio identity leakage')
    plan['indices'], plan['chart_ids'] = indices, chart_ids
    plan['data_summary'] = {}
    for split, chosen_indices in indices.items():
        counts = Counter(); support = defaultdict(set)
        for i in chosen_indices:
            row = panel.records[i]; counts.update(row['types'])
            for kind, number in row['types'].items():
                if number:
                    support[kind].add(row['component'])
        plan['data_summary'][split] = dict(windows=len(chosen_indices),
            components=len({panel.records[i]['component'] for i in chosen_indices}),
            charts=len({panel.records[i]['chart_id'] for i in chosen_indices}), types=dict(counts),
            type_song_support={k: len(v) for k, v in support.items()},
            empty_windows=sum(panel.records[i]['event_count'] == 0 for i in chosen_indices))
    positive = [i for i in indices['train'] if panel.records[i]['event_count']]
    health = []
    for kind in ('hold', 'flick', 'drag', 'tap'):
        best = max((i for i in positive if panel.records[i]['chart_id'] not in
                    {panel.records[j]['chart_id'] for j in health}),
                   key=lambda i: (panel.records[i]['types'].get(kind, 0), -panel.records[i]['tokens']))
        health.append(best)
    plan['health_indices'] = health
    plan['replay_indices'] = health + positive[::max(1, len(positive) // 4)][:4]
    rng = np.random.default_rng(plan['seed'])
    rare = {kind: [i for i in positive if panel.records[i]['types'].get(kind)]
            for kind in ('hold', 'flick', 'drag')}
    plan['schedule'] = []
    for step in range(plan['steps']):
        a = indices['train'][int(rng.integers(len(indices['train'])))]
        pool = rare[('hold', 'flick', 'drag')[step % 3]] if step % 2 == 0 else indices['train']
        b = pool[int(rng.integers(len(pool)))]
        plan['schedule'].append([dict(index=i, view=VIEWS[(step * 2 + j) % len(VIEWS)], field_index=step)
                                  for j, i in enumerate((a, b))])
    plan['replay_schedule_note'] = 'Every tenth update after step200 uses one buffered generated-history continuation; both arms share the same suffix target, baseline keeps the human prefix.'
    # Freeze small native targets and existing audit identities for large cached audio.
    sources = [plan['tape'], plan['features'], 'beatfcn/model/full_event_model.py',
        'beatfcn/model/full_event_codec.py', 'beatfcn/model/full_event_generate.py',
        'beatfcn/model/local_bundle.py', 'beatfcn/train/local_bundle.py', 'scripts/run_local_bundle.py',
        'beatfcn/train/local_request_panel.py', 'beatfcn/train/reference_completion.py']
    plan['source_sha256'] = {p: sha(ROOT / p) for p in sources}
    plan['target_sha256'] = {}
    for i in indices['train'] + indices['val']:
        chart = panel.charts[panel.records[i]['chart_index']]
        path = chart['target_path']
        if path not in plan['target_sha256']:
            actual = sha(ROOT / path)
            if actual != chart['target_sha256']:
                raise ValueError('Native target changed: ' + path)
            plan['target_sha256'][path] = actual
    plan['admitted_feature_identities'] = {a: panel.features.records[a]['sha256'] for a in train_audio | val_audio}
    plan['frozen_at_unix'] = time.time()
    plan['preparation_wall_s'] = time.monotonic() - started
    write(PLAN, plan)
    write(REPORT, dict(id=plan['id'], status='prepared', complete=False, passed=False,
        cloud_compute_calls=0, updates={a: 0 for a in ARMS}, tokens={a: 0 for a in ARMS},
        curve=[], evaluations={}, failures=[], data_summary=plan['data_summary']))
    journal('Inspect/Hypothesize：用户本轮明确授权小模型/小数据本地实验，并要求所有方法联合，故本轮以完整组合路线比较覆盖一次单因素惯例；不拆消融、不声称单项因果。冻结24训练/8验证Phira歌曲组件、原生标签、共同新初值与1000更新上限；组合包含时间直读音频、类型/持续及条件直连、学习组大小、分字段监督、稀有类型跨歌采样、参考补全与自身历史恢复。健康900s/训练3000s/评价1200s，云0，旧失败权重不续用、自动任务不恢复。维度混合，不擅自改写既有Objective6/Architecture5/DataRepresentation7/Dynamics2单项计数。拥有local_bundle新模块、runner、tests、bench及对应report/artifacts；仅追加本日志。')
    log(dict(prepared=True, data=plan['data_summary'], health=health))


def make_models(plan):
    torch.manual_seed(plan['seed'])
    config = FullEventConfig(**plan['model'])
    base = FullEventModel(config)
    bundle = LocalBundleModel(config)
    state = bundle.state_dict()
    for name, value in base.state_dict().items():
        state[name] = value.clone()
    bundle.load_state_dict(state)
    return {'baseline': base, 'bundle': bundle}


def configure():
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise RuntimeError('Native local MPS is required; no silent CPU fallback')
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.55)


@torch.no_grad()
def teacher(model, panel, indices, device='mps'):
    model.eval(); rows = []; totals = defaultdict(lambda: [0, 0])
    for index in indices:
        row = make_row(panel, index)
        inp, seq = batch_rows([row], device)
        logits = model(inp, seq)
        pred = logits.masked_fill(~seq['allowed'], -1e9).argmax(-1)
        fields = {}
        for field, name in enumerate(FIELDS):
            mask = (seq['fields'] == field) & seq['valid']
            n = int(mask.sum()); correct = int(((pred == seq['targets']) & mask).sum())
            totals[name][0] += correct; totals[name][1] += n
            fields[name] = dict(correct=correct, count=n)
        rows.append(dict(index=index, ce=float(event_loss(logits, seq)), fields=fields))
    return dict(mean_ce=float(np.mean([r['ce'] for r in rows])), rows=rows,
                accuracy={k: dict(correct=v[0], count=v[1], accuracy=v[0] / max(v[1], 1)) for k, v in totals.items()})


def health(plan, report):
    if report.get('health'):
        raise ValueError('Health already attempted; preserve evidence')
    configure(); panel = panel_for(plan)
    rows = [make_row(panel, i, view=v) for i, v in zip(plan['health_indices'],
                ('draft', 'single_condition', 'random_groups', 'middle'))]
    model = make_models(plan)['bundle'].eval()
    inp, seq = batch_rows([rows[0]])
    audit = tensor_audit(inp, seq)
    with torch.no_grad():
        cpu = model(inp, seq)
    model.to('mps')
    with torch.no_grad():
        gpu = model(*batch_rows([rows[0]], 'mps')).cpu()
    error = float((cpu - gpu).abs().max())
    if error > 1e-4:
        raise ValueError('CPU/MPS mismatch')
    batches = [batch_rows([row], 'mps') for row in rows]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=0.)
    started = time.monotonic(); curve = []
    for step in range(1, plan['health_updates'] + 1):
        if time.monotonic() - started > plan['health_wall_s']:
            raise TimeoutError('Health local wall limit')
        model.train(); optimizer.zero_grad(set_to_none=True); total = 0.
        for inputs, sequence in batches:
            loss = bundle_loss(model, model(inputs, sequence), sequence) / len(batches)
            loss.backward(); total += float(loss.detach())
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step()
        if step == 1 or step % 25 == 0:
            curve.append(dict(step=step, loss=total, grad=grad, wall_s=time.monotonic() - started))
            log(dict(stage='health', **curve[-1]))
    model.eval(); final = []
    for row, (inputs, sequence) in zip(rows, batches):
        with torch.no_grad():
            logits = model(inputs, sequence)
            prediction = logits.masked_fill(~sequence['allowed'], -1e9).argmax(-1)
            accuracy = float(((prediction == sequence['targets']) & sequence['valid']).sum() / sequence['valid'].sum())
            ce = float(event_loss(logits, sequence))
        generated = generate_bundle(model, inputs, [row['request']], int(sequence['valid'].sum()) + 64,
                                    deadline=time.monotonic() + 40)[0]
        exact = generated['complete'] and sorted(RhythmEvent(**e) for e in generated['events']) == sorted(row['targets'])
        final.append(dict(index=row['index'], ce=ce, accuracy=accuracy, free_exact=exact,
                          complete=generated['complete'], generated_events=len(generated['events'])))
    passed = error <= 1e-4 and all(r['accuracy'] >= .98 and r['ce'] < .05 for r in final)
    report['health'] = dict(passed=passed, curve=curve, requests=final, real_tensor_audit=audit,
                            cpu_mps_max_error=error, wall_s=time.monotonic() - started,
                            optimizer_updates=plan['health_updates'], quality_weights_reused=False,
                            free_exact=sum(r['free_exact'] for r in final))
    report['status'] = 'health_passed' if passed else 'health_failed'
    write(REPORT, report)
    journal(f"Implement/Probe：真实4请求100更新健康{'通过' if passed else '失败'}，CPU/MPS最大差{error:.3g}，自由精确{report['health']['free_exact']}/4，wall{report['health']['wall_s']:.3f}s；健康权重丢弃，0云。")
    log(dict(health=report['health']['passed'], requests=final))
    if not passed:
        raise RuntimeError('Health failed; quality not launched')


def generation(model, panel, indices, *, view='draft', manual_bands=None, token_cap=1024,
               deadline=None):
    model.eval(); records = []
    generator = generate_bundle if isinstance(model, LocalBundleModel) else generate_batch
    for index in indices:
        if deadline is not None and time.monotonic() >= deadline:
            records.append(dict(index=index, attempted=False, reason='evaluation_wall_limit')); continue
        row = make_row(panel, index, view, manual_bands=manual_bands)
        inputs, _ = batch_rows([row], 'mps')
        local_deadline = min(deadline or float('inf'), time.monotonic() + 45)
        result = generator(model, inputs, [row['request']], token_cap, deadline=local_deadline)[0]
        pred = [RhythmEvent(**e) for e in result['events']]
        scores = score_events(canonical(pred, 'phira'), canonical(row['targets'], 'phira'), 'phira')
        full_tp = scores['note_50ms']['tp'] - scores['hold_head_50ms']['tp'] + scores['hold_joint_50ms']['tp']
        scores['full_note_f1'] = 2 * full_tp / max(len(pred) + len(row['targets']), 1)
        scores['onset_f1'] = scores['onset_50ms']['f1'] or 0.
        scores['per_type'] = {}
        for kind in ('tap', 'hold', 'flick', 'drag'):
            pp = [e for e in pred if e.type == kind]
            tt = [e for e in row['targets'] if e.type == kind]
            detail = score_events(canonical(pp, 'phira'), canonical(tt, 'phira'), 'phira')
            scores['per_type'][kind] = detail['hold_joint_50ms' if kind == 'hold' else 'note_50ms']
        gaps = np.diff(sorted(set(e.onset_us for e in pred)))
        records.append(dict(index=index, attempted=True, chart_id=panel.records[index]['chart_id'],
            view=view, request=asdict(row['request']), prediction=result,
            target=[asdict(e) for e in row['targets']], visible=[asdict(e) for e in row['visible']],
            visible_preservation='Visible objects are retained verbatim by external merge, not regenerated by the model', scores=scores,
            legality=independent_legality(pred, 'phira', row['request'].duration_us),
            types=dict(Counter(e.type for e in pred)), target_types=dict(Counter(e.type for e in row['targets'])),
            positive_gap_median_ms=float(np.median(gaps) / 1000) if len(gaps) else None,
            positive_gaps_under_10ms=int((gaps < 10000).sum()), positive_gaps=len(gaps),
            max_same_time=max(Counter(e.onset_us for e in pred).values(), default=0)))
    return records


def checkpoint(model, optimizer, plan, arm, step):
    path = ART / arm / f'step-{step:04d}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=plan['model'],
                    arm=arm, updates=step, plan_sha256=sha(PLAN), cpu_rng=torch.get_rng_state(),
                    mps_rng=torch.mps.get_rng_state()), path)
    return str(path.relative_to(ROOT))


def train(plan, report):
    if not report.get('health', {}).get('passed') or any(report['updates'].values()):
        raise ValueError('Fresh quality requires passed health and zero prior updates')
    for p, digest in plan['source_sha256'].items():
        if sha(ROOT / p) != digest:
            raise ValueError('Frozen implementation changed: ' + p)
    configure(); panel = panel_for(plan); models = make_models(plan)
    report['parameters'] = {a: sum(p.numel() for p in m.parameters()) for a, m in models.items()}
    report['initial_common_parameters_equal'] = all(torch.equal(v, models['bundle'].state_dict()[k])
        for k, v in models['baseline'].state_dict().items())
    optimizers = {}
    for arm, model in models.items():
        model.to('mps')
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
    replay = {}; report['replay'] = []
    report['status'] = 'training'; write(REPORT, report)
    started = time.monotonic()
    def interrupted(signum, frame):
        raise InterruptedError(f'Local training interrupted by signal {signum}')
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for step, original_items in enumerate(plan['schedule'], 1):
            if time.monotonic() - started > plan['train_wall_s']:
                raise TimeoutError('Training reached frozen local wall bound')
            items = deepcopy(original_items)
            if step > 200 and step % 10 == 0 and replay:
                index = sorted(replay)[(step // 10) % len(replay)]
                items[0] = dict(index=index, view='continuation', field_index=0, use_replay=True)
            prepare_start = time.monotonic()
            rows = [make_row(panel, i['index'], i['view'], field_index=i['field_index']) for i in items]
            baseline_batches = [batch_rows([r]) for r in rows]
            bundle_batches = [batch_rows([make_row(panel, i['index'], 'continuation',
                generated_history=replay[i['index']]['events'])]) if i.get('use_replay') else batch
                for i, batch in zip(items, baseline_batches)]
            for (_, before), (_, after) in zip(baseline_batches, bundle_batches):
                if any(not torch.equal(before[k], after[k]) for k in before):
                    raise ValueError('Generated-history recovery changed the shared suffix targets')
            prepare_wall = time.monotonic() - prepare_start
            ratio = min(step / plan['warmup_steps'], 1.)
            progress = max(0., (step - plan['warmup_steps']) / (plan['steps'] - plan['warmup_steps']))
            lr = ratio * (plan['minimum_lr'] + (plan['learning_rate'] - plan['minimum_lr']) * .5 * (1 + math.cos(math.pi * progress)))
            losses = {}
            for arm, batches in zip(ARMS, (baseline_batches, bundle_batches)):
                arm_start = time.monotonic()
                model, opt = models[arm], optimizers[arm]
                model.train(); opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:
                    group['lr'] = lr
                total = 0.; tokens = 0
                for inp, seq in batches:
                    inputs, sequence = ({k: v.to('mps') for k, v in part.items()} for part in (inp, seq))
                    logits = model(inputs, sequence)
                    objective = (bundle_loss(model, logits, sequence, action_weight=2. if step <= 200 else 1.)
                                 if arm == 'bundle' else event_loss(logits, sequence))
                    loss = objective / len(batches); loss.backward()
                    total += float(loss.detach()); tokens += int(sequence['valid'].sum())
                grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), plan['clip_norm'], error_if_nonfinite=True))
                opt.step(); report['updates'][arm] = step; report['tokens'][arm] += tokens
                losses[arm] = dict(objective=total, grad_norm=grad, wall_s=time.monotonic() - arm_start)
            if step == 1 or step % 25 == 0:
                point = dict(step=step, lr=lr, arms=losses, prepare_wall_s=prepare_wall,
                             wall_s=time.monotonic() - started)
                report['curve'].append(point); report['training_wall_s'] = point['wall_s']; write(REPORT, report)
                log(dict(stage='train', **point))
            if step in plan['replay_steps']:
                model = models['bundle'].eval(); new_records = []
                for index in plan['replay_indices']:
                    row = prefix_row(panel, index); inputs, _ = batch_rows([row], 'mps')
                    result = generate_bundle(model, inputs, [row['request']], plan['replay_token_cap'],
                        deadline=min(started + plan['train_wall_s'], time.monotonic() + 12))[0]
                    replay[index] = result
                    new_records.append(dict(index=index, complete=result['complete'],
                                            event_count=len(result['events']), tokens=len(result['tokens'])))
                write(ART / f'replay-{step:04d}.json', replay)
                report['replay'].append(dict(step=step, records=new_records)); write(REPORT, report)
                log(dict(stage='replay', step=step, records=new_records))
            if step in plan['checkpoints']:
                evaluated = {}
                for arm, model in models.items():
                    path = checkpoint(model, optimizers[arm], plan, arm, step)
                    ev = dict(checkpoint=path, validation=teacher(model, panel, plan['indices']['val']),
                              train=teacher(model, panel, plan['health_indices']))
                    if step < plan['steps']:
                        free = generation(model, panel, plan['indices']['val'][::4], token_cap=768,
                            deadline=min(started + plan['train_wall_s'], time.monotonic() + 120))
                        write(ART / arm / f'windows-{step:04d}.json', free)
                        ev['free_summary'] = summarize(free)
                    evaluated[arm] = ev
                report['evaluations'][str(step)] = evaluated; write(REPORT, report)
                log(dict(stage='checkpoint', step=step, validation={a: v['validation']['mean_ce'] for a, v in evaluated.items()},
                         free={a: v.get('free_summary') for a, v in evaluated.items()}))
        report['training_wall_s'] = time.monotonic() - started
        report['status'] = 'trained'; write(REPORT, report)
    except Exception as exc:
        report['status'] = 'training_stopped'; report['failures'].append(f'{type(exc).__name__}: {exc}')
        report['training_wall_s'] = time.monotonic() - started
        for arm, model in models.items():
            checkpoint(model, optimizers[arm], plan, arm, report['updates'][arm])
        write(REPORT, report)
        raise


def summarize(records):
    attempted = [r for r in records if r.get('attempted')]
    types = Counter(); target = Counter()
    for r in attempted:
        types.update(r['types']); target.update(r['target_types'])
    return dict(attempted=len(attempted), unattempted=len(records) - len(attempted),
        complete=sum(r['prediction']['complete'] and not r['legality'] for r in attempted),
        predicted_types=dict(types), target_types=dict(target),
        nonempty=sum(bool(r['prediction']['events']) for r in attempted),
        mean_full_note_f1=float(np.mean([r['scores']['full_note_f1'] for r in attempted])) if attempted else None,
        mean_onset_f1=float(np.mean([r['scores']['onset_f1'] for r in attempted])) if attempted else None)


def evaluate(plan, report):
    if report['status'] != 'trained':
        raise ValueError('Final evaluation requires completed matched training')
    configure(); panel = panel_for(plan); started = time.monotonic(); deadline = started + plan['evaluation_wall_s']
    final = {}
    for arm in ARMS:
        saved = torch.load(ART / arm / f"step-{plan['steps']:04d}.pt", map_location='cpu', weights_only=False)
        model = (LocalBundleModel if arm == 'bundle' else FullEventModel)(FullEventConfig(**plan['model']))
        model.load_state_dict(saved['model']); model.to('mps').eval(); del saved
        sets = {}
        for name, indices, view, bands in (
            ('draft', plan['indices']['val'], 'draft', None),
            ('completion', plan['indices']['val'][::2], 'middle', None),
            ('density_low', plan['indices']['val'][::4], 'draft', {'density_notes_per_second': [2., 3.]}),
            ('density_high', plan['indices']['val'][::4], 'draft', {'density_notes_per_second': [6., 8.]})):
            records = generation(model, panel, indices, view=view, manual_bands=bands,
                token_cap=plan['generation_token_cap'], deadline=deadline)
            write(ART / arm / f'{name}-final.json', records)
            sets[name] = summarize(records)
            log(dict(stage='evaluation', arm=arm, task=name, summary=sets[name]))
        final[arm] = sets
        del model; torch.mps.empty_cache()
    report['final'] = final; report['evaluation_wall_s'] = time.monotonic() - started
    report['status'] = 'evaluated'; write(REPORT, report)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'health', 'train', 'evaluate'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(); return
    plan = json.loads(PLAN.read_text()); report = json.loads(REPORT.read_text())
    {'health': health, 'train': train, 'evaluate': evaluate}[args.stage](plan, report)


if __name__ == '__main__':
    main()
