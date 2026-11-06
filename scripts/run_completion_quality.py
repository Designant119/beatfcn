"""Matched reference/no-reference training and complete-song masking evaluation.

The only arm difference is the visible-reference input. Health weights are never
loaded. Archived failure runners and production defaults are not modified.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.local_song_rollout import canonical, independent_legality
from beatfcn.model.evaluation import score_events
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_panel import collate
from beatfcn.train.local_shapes import pad_input_shapes
from beatfcn.train.reference_completion import TASKS, completion_row
from scripts.run_reference_completion import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/completion-quality-v1.json'
ARMS = ('no_reference', 'reference')


def prepare():
    if (ROOT / PLAN).exists():
        raise FileExistsError(PLAN)
    started = time.monotonic()
    health = read('reports/reference_completion_probe_v1.json')
    if not health['complete'] or not health['passed'] or not health['parent_unchanged']:
        raise ValueError('Completion health gate not passed')
    base = read('bench/rhythm-v2/reference-completion-v1.json')
    panel = LocalRequestPanel(ROOT, base['tape'], base['features'])
    exposure_path = 'inputs/d94_local_requests/learning-exposure-1000-v1.json'
    exposure = read(exposure_path)
    steps, counts, taskcounts, typecounts = [], Counter(), Counter(), defaultdict(Counter)
    max_reference = 0
    for n, step in enumerate(exposure['steps']):
        items = []
        for j, index in enumerate(step['indices']):
            row = panel.records[index]
            if row['split'] != 'train':
                raise ValueError('Validation exposure in training')
            counts[row['mode']] += 1
            task = TASKS[(n * 4 + j) % 3] if row['mode'] == 'phira' else None
            items.append(dict(index=index, task=task))
            if task is not None:
                value, _, metadata = completion_row(panel, index, task, reference=True, seed=base['seed'])
                max_reference = max(max_reference, len(value['inputs']['reference']))
                taskcounts[task] += 1
                typecounts[task].update(e['type'] for e in metadata['hidden'])
        steps.append(items)
    expected_charts = set(read('reports/d94_data_scale_generation_v1.json')['song_files'])
    songs = []
    for chart_id in expected_charts:
        indices = [i for i, r in enumerate(panel.records) if r['chart_id'] == chart_id and r['split'] == 'val']
        if not indices:
            raise ValueError('Original evaluation song not available: ' + chart_id)
        indices.sort(key=lambda i: panel.records[i]['request']['start_us'])
        chart = panel.charts[panel.records[indices[0]]['chart_index']]
        # Existing tape covers the entire decoded track, including the final tail.
        cursor = 0
        total = 0
        for i in indices:
            row = panel.records[i]
            if row['request']['start_us'] != cursor:
                raise ValueError('Evaluation tape has a missing/overlapping core')
            cursor = row['request']['end_us']
            total += row['target_stop'] - row['target_start']
        if cursor != chart['duration_us']:
            raise ValueError('Evaluation tape does not cover full audio')
        songs.append(dict(chart_id=chart_id, audio_sha256=chart['audio_sha256'],
                          component=chart['component'], stratum=chart['stratum'], indices=indices,
                          human_events=total, duration_us=chart['duration_us']))
    songs.sort(key=lambda s: (s['human_events'] / (s['duration_us'] / 1e6), s['chart_id']))
    # Early sentinels span the original low/middle/high burdens, selected before predictions.
    first = list(dict.fromkeys([0, len(songs) // 3, 2 * len(songs) // 3, len(songs) - 1]))
    songs = [songs[i] for i in first] + [s for i, s in enumerate(songs) if i not in first]
    for n, song in enumerate(songs):
        song['task'] = TASKS[n % 3]
    sources = dict(base['source_sha256'])
    for path in ('scripts/run_completion_quality.py', 'reports/reference_completion_probe_v1.json',
                 'bench/rhythm-v2/reference-completion-v1.json', exposure_path,
                 'reports/d94_data_scale_generation_v1.json'):
        sources[path] = sha(path)
    # Freeze every chart and audio cache that can enter either arm.
    used = {i['index'] for step in steps for i in step} | {i for s in songs for i in s['indices']}
    for i in used:
        row = panel.records[i]
        chart = panel.charts[row['chart_index']]
        sources.update(panel.features.records[row['audio_sha256']]['sha256'])
        sources.update({chart['target_path']: chart['target_sha256'], chart['token_path']: chart['token_sha256']})
    plan = dict(id='completion-quality-v1', seed=base['seed'], parent=base['parent'],
        tape=base['tape'], features=base['features'], source_sha256=sources,
        steps=steps, mode_exposure=dict(counts), task_exposure=dict(taskcounts),
        hidden_type_exposure={k: dict(v) for k, v in typecounts.items()},
        max_reference_rows=max_reference, songs=songs, learning_rate=.001,
        train_wall_s=3000, evaluation_wall_s=1800, deadline_unix=base['deadline_unix'],
        token_cap=4096, checkpoints=[250, 500, 1000],
        validation_indices=[i for s in songs[:6] for i in [s['indices'][len(s['indices']) // 2]]],
        report='reports/completion_quality_v1.json', artifacts='artifacts/completion-quality-v1',
        quality_gates=dict(song_f1_gain=.05, task_degradation_max=.02,
            improving_songs=12, complete_fraction=.95, close_interval_degradation_max=.05),
        early_stop=dict(failed_reference_songs=2, collapsed_sentinel_songs=4, collapse_min_predicted_notes=100),
        evaluation_contract='Twenty whole songs, balanced assigned mask task; every 8s core uses the same frozen masking task. Continuation is local half-core continuation throughout the song, not one single whole-song halfway cut.',
        preparation_wall_s=time.monotonic() - started)
    write(PLAN, plan)
    print(json.dumps(dict(prepared=True, wall_s=plan['preparation_wall_s'], exposure=plan['mode_exposure'],
                         tasks=plan['task_exposure'], max_reference_rows=max_reference, songs=len(songs))), flush=True)


def batches(panel, item, plan):
    index, task = item['index'], item['task']
    if task is None:
        controls = np.asarray(panel.records[index]['controls'], np.float32).copy()
        controls[:, :3] = 0
        value = panel.row(index, controls=controls)
        ref = value
        request = None
    else:
        value, request, _ = completion_row(panel, index, task, reference=False, seed=plan['seed'])
        ref, other, _ = completion_row(panel, index, task, reference=True, seed=plan['seed'])
        if request != other:
            raise ValueError('Arm request mismatch')
        for key, v in value['inputs'].items():
            if not key.startswith('reference') and not np.array_equal(v, ref['inputs'][key]):
                raise ValueError('Arm input mismatch: ' + key)
        for key, v in value['sequence'].items():
            if not np.array_equal(v, ref['sequence'][key]):
                raise ValueError('Arm supervision mismatch: ' + key)
    length = len(ref['inputs']['reference'])
    shape = dict(local=400, mel=1600, song=1024, reference=max(64, ((length + 63) // 64) * 64))
    result = {}
    for arm, v in zip(ARMS, (value, ref)):
        inp, seq = collate([v])
        result[arm] = (pad_input_shapes(inp, shape), seq)
    return result


def train(plan, report):
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise RuntimeError('Native MPS required')
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.6)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    saved = torch.load(ROOT / plan['parent'], map_location='cpu', weights_only=False)
    if saved['updates'] != 2:
        raise ValueError('Wrong training parent')
    models, optimizers = {}, {}
    for arm in ARMS:
        model = FullEventModel(FullEventConfig(**saved['config'])).to('mps')
        model.load_state_dict(saved['model'])
        opt = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
        opt.load_state_dict(deepcopy(saved['optimizer']))
        models[arm], optimizers[arm] = model, opt
        if any(g['lr'] != plan['learning_rate'] or g['weight_decay'] != 0 for g in opt.param_groups):
            raise ValueError('Parent hyperparameter mismatch')
    report['parent_model_exactly_equal'] = all(torch.equal(models[ARMS[0]].state_dict()[k], v) for k, v in models[ARMS[1]].state_dict().items())
    if not report['parent_model_exactly_equal']:
        raise ValueError('Unmatched initial model states')
    torch.set_rng_state(saved['cpu_rng']); torch.mps.set_rng_state(saved['mps_rng'])
    del saved
    started = time.monotonic()
    for n, items in enumerate(plan['steps'], 1):
        if time.monotonic() - started >= plan['train_wall_s'] or time.time() >= plan['deadline_unix']:
            raise TimeoutError('Matched training bound')
        # Prepare each shared request once; both arms see identical ordering and targets.
        cpu = [batches(panel, item, plan) for item in items]
        losses = {}
        for arm in ARMS:
            model, opt = models[arm], optimizers[arm]
            model.train(); opt.zero_grad(set_to_none=True)
            total, tokens = 0., 0
            for pair in cpu:
                inp, seq = ({k: v.to('mps') for k, v in section.items()} for section in pair[arm])
                loss = event_loss(model(inp, seq), seq) / len(cpu)
                loss.backward()
                total += float(loss.detach()); tokens += int(seq['valid'].sum())
                del inp, seq, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            report['updates'][arm] = n
            report['tokens'][arm] += tokens
            losses[arm] = dict(ce=total, preclip_norm=float(norm))
        report['training_wall_s'] = time.monotonic() - started
        report['status'] = 'matched_training'
        if n % 25 == 0:
            report['curve'].append(dict(step=n, training=losses, training_wall_s=report['training_wall_s']))
            write(plan['report'], report)
            print(json.dumps(dict(step=n, ce=losses, wall_s=report['training_wall_s'])), flush=True)
        if n in plan['checkpoints']:
            validation = {}
            with torch.no_grad():
                for arm in ARMS:
                    model = models[arm].eval()
                    values = []
                    for index in plan['validation_indices']:
                        for task in TASKS:
                            cpuval = batches(panel, dict(index=index, task=task), plan)[arm]
                            inp, seq = ({k: v.to('mps') for k, v in section.items()} for section in cpuval)
                            values.append(dict(index=index, task=task, ce=float(event_loss(model(inp, seq), seq))))
                    validation[arm] = values
            report['validation'].append(dict(step=n, rows=validation))
            write(plan['report'], report)
    for arm in ARMS:
        path = ROOT / f"{plan['artifacts']}/{arm}/step-1000.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            torch.save(dict(model={k: v.detach().cpu() for k, v in models[arm].state_dict().items()},
                       config=asdict(models[arm].c), updates=1000, parent_updates=2,
                       parent=plan['parent'], plan_sha256=sha(PLAN), arm=arm,
                       optimizer=optimizers[arm].state_dict(), diagnostic_only=False,
                       auto_resume=False), stream)
        report['checkpoints'][arm] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path.relative_to(ROOT)))
    del models, optimizers
    torch.mps.empty_cache()
    report.update(status='trained', training_complete=True, training_wall_s=time.monotonic() - started)
    write(plan['report'], report)


def metrics(pred, hidden, visible, complete, duration_us):
    p = [RhythmEvent(**e) for e in pred]
    t = [RhythmEvent(**e) for e in hidden]
    scores = score_events(canonical(p, 'phira'), canonical(t, 'phira'), 'phira')
    full = scores['edit_utility']['full_note']
    denom = full['pred'] + full['true']
    times = sorted({e.onset_us for e in p})
    delta = np.diff(times)
    # Exact extra copies of protected objects are reported, never removed for scoring.
    protected = Counter((e['onset_us'], e['end_us'], e['type']) for e in visible)
    duplicates = sum(min(n, protected[k]) for k, n in Counter((e['onset_us'], e['end_us'], e['type']) for e in pred).items())
    return dict(full_note_f1=2 * full['matched'] / denom if denom else 1., scores=scores,
                complete=complete, pred_count=len(p), hidden_count=len(t),
                predicted_types=dict(Counter(e.type for e in p)), protected_duplicates=duplicates,
                short_interval_fraction=float(np.mean(delta < 10000)) if len(delta) else 0.,
                legality=independent_legality(p, 'phira', duration_us))


@torch.inference_mode()
def evaluate(plan, report):
    torch.set_num_threads(2)
    models = {}
    for arm in ARMS:
        checkpoint = report['checkpoints'][arm]
        if sha(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Quality checkpoint changed')
        state = torch.load(ROOT / checkpoint['path'], map_location='cpu', weights_only=False)
        model = FullEventModel(FullEventConfig(**state['config'])).eval()
        model.load_state_dict(state['model']); models[arm] = model
        del state
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    started = time.monotonic()
    deadline = min(started + plan['evaluation_wall_s'], started + plan['deadline_unix'] - time.time())
    for song in plan['songs']:
        predictions, cores = {a: [] for a in ARMS}, {a: [] for a in ARMS}
        failed = {a: None for a in ARMS}
        hidden, visible = [], []
        for index in song['indices']:
            value, request, meta = completion_row(panel, index, song['task'], reference=True, seed=plan['seed'])
            hidden.extend(meta['hidden']); visible.extend(meta['visible'])
            # Hidden truth is used only for scoring, never forwarded to generate_batch.
            pair = batches(panel, dict(index=index, task=song['task']), plan)
            for arm in ARMS:
                if failed[arm] is not None:
                    continue
                prediction = generate_batch(models[arm], pair[arm][0], [request], plan['token_cap'], deadline=deadline)[0]
                predictions[arm].extend(prediction['events'])
                cores[arm].append(dict(index=index, request=asdict(request), prediction=prediction))
                if not prediction['complete']:
                    failed[arm] = 'deadline' if prediction['deadline_reached'] else 'token_cap'
        outputs = {}
        for arm in ARMS:
            score = metrics(predictions[arm], hidden, visible, failed[arm] is None, song['duration_us'])
            path = f"{plan['artifacts']}/{arm}/songs/{song['chart_id'].replace(':', '-')}.json"
            audio = panel.features.records[song['audio_sha256']]
            write(path, dict(song=song, audio_cache_paths=audio['paths'], cores=cores[arm],
                    hidden=hidden, visible=visible, prediction=predictions[arm],
                    merged_unmodified_visible_and_prediction=visible + predictions[arm],
                    metrics=score, failure=failed[arm]))
            csv = ROOT / path.replace('.json', '.csv')
            with csv.open('w') as stream:
                stream.write('source,onset_us,end_us,type\n')
                for source, values in (('visible', visible), ('hidden', hidden), ('predicted', predictions[arm])):
                    for e in values:
                        stream.write(f"{source},{e['onset_us']},{e['end_us']},{e['type']}\n")
            outputs[arm] = dict(artifact=path, **{k: v for k, v in score.items() if k != 'scores'}, failure=failed[arm])
        report['songs'].append(dict(chart_id=song['chart_id'], stratum=song['stratum'], task=song['task'], outputs=outputs))
        report.update(status='whole_song_evaluation', evaluation_wall_s=time.monotonic() - started)
        write(plan['report'], report)
        print(json.dumps(dict(evaluated_songs=len(report['songs']), song=song['chart_id'], task=song['task'],
                              outputs=outputs, wall_s=report['evaluation_wall_s'])), flush=True)
        ref_failures = sum(not r['outputs']['reference']['complete'] for r in report['songs'])
        collapsed = len(report['songs']) >= plan['early_stop']['collapsed_sentinel_songs'] and all(
            r['outputs']['reference']['pred_count'] >= plan['early_stop']['collapse_min_predicted_notes']
            and len(r['outputs']['reference']['predicted_types']) <= 1 for r in report['songs'][:4])
        if ref_failures >= plan['early_stop']['failed_reference_songs'] or collapsed or time.monotonic() >= deadline:
            report['stop_reason'] = 'reference_completion_failures' if ref_failures >= 2 else ('single_type_collapse_sentinels' if collapsed else 'evaluation_deadline')
            break
    complete_panel = len(report['songs']) == len(plan['songs'])
    rows = report['songs']
    gains = [r['outputs']['reference']['full_note_f1'] - r['outputs']['no_reference']['full_note_f1'] for r in rows]
    task_gains = {task: float(np.mean([g for r, g in zip(rows, gains) if r['task'] == task]))
                  for task in TASKS if any(r['task'] == task for r in rows)}
    types = Counter()
    for r in rows:
        types.update(r['outputs']['reference']['predicted_types'])
    complete_fraction = sum(r['outputs']['reference']['complete'] for r in rows) / len(plan['songs'])
    short_degradation = float(np.mean([r['outputs']['reference']['short_interval_fraction'] - r['outputs']['no_reference']['short_interval_fraction'] for r in rows]))
    passed = (complete_panel and np.mean(gains) >= plan['quality_gates']['song_f1_gain']
              and sum(g > 0 for g in gains) >= plan['quality_gates']['improving_songs']
              and len(task_gains) == len(TASKS) and min(task_gains.values()) >= -.02
              and complete_fraction >= .95 and len(types) >= 2 and types['hold'] > 0
              and short_degradation <= .05)
    report.update(evaluation_panel_complete=complete_panel, quality_passed=bool(passed),
        summary=dict(attempted_songs=len(rows), expected_songs=len(plan['songs']),
            paired_song_mean_f1_gain=float(np.mean(gains)), improving_songs=sum(g > 0 for g in gains),
            task_gains=task_gains, reference_types=dict(types),
            complete_fraction_of_expected=complete_fraction, short_interval_degradation=short_degradation,
            interpretation='Partial-panel metrics are diagnostic only; unattempted songs are not silently excluded from acceptance.'),
        evaluation_wall_s=time.monotonic() - started)


def run():
    plan = read(PLAN)
    if (ROOT / plan['report']).exists():
        raise FileExistsError(plan['report'])
    started = time.monotonic()
    report = dict(id=plan['id'], plan_sha256=sha(PLAN), complete=False, training_complete=False,
        quality_passed=False, updates={a: 0 for a in ARMS}, tokens={a: 0 for a in ARMS},
        curve=[], validation=[], checkpoints={}, songs=[], new_cloud_spend_usd=0, status='verifying')
    try:
        for path, expected in plan['source_sha256'].items():
            if sha(path) != expected:
                raise ValueError('Frozen source changed: ' + path)
        train(plan, report)
        evaluate(plan, report)
        report.update(complete=True, status='quality_passed' if report['quality_passed'] else 'quality_failed',
                      parent_unchanged=sha(plan['parent']) == plan['source_sha256'][plan['parent']])
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_s'] = time.monotonic() - started
        write(plan['report'], report)
    print(json.dumps(dict(status=report['status'], updates=report['updates'], summary=report.get('summary'), wall_s=report['wall_s'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
